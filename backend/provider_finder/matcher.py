"""
backend/provider_finder/matcher.py

Phase 7 — Provider Finder: Deterministic matching engine.

find_providers(query, provider_pool) → ProviderSearchResult

Matching sequence (documented, auditable, neutral):
  1. Required specialty match (case-insensitive, exact)
  2. Optional city/location match (case-insensitive, exact)
  3. Optional language match (case-insensitive, any in list)
  4. Optional insurance match (case-insensitive, any in list)

Ordering (neutral, deterministic):
  1. Exact specialty match (always true for results — kept for clarity)
  2. Exact city match first
  3. Explicit language match
  4. Alphabetical provider_id

NEVER:
  - Rank by quality
  - Rank by outcomes
  - Rank by reputation
  - Fabricate distance
  - Fabricate availability
  - Add quality claims
  - Infer patient location
  - Infer patient insurance
  - Modify the care plan
"""
from __future__ import annotations

import copy
import re
from typing import List, Optional, Tuple

from backend.provider_finder.neutrality import audit_provider_neutrality, sanitize_display_text
from backend.provider_finder.provenance import validate_provider_provenance
from backend.provider_finder.types import (
    FreshnessStatus,
    MatchAttribute,
    NeutralityStatus,
    ProviderMatch,
    ProviderSearchQuery,
    ProviderSearchResult,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace for comparison."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _specialty_matches(provider_specialty: str, required: str) -> bool:
    """
    Case-insensitive specialty match.
    Also handles compound specialties like "Cardiology / Dermatology" —
    we require an EXACT component match, not a substring match.
    """
    # Split compound specialties on / or ;
    components = re.split(r"[/;]", provider_specialty)
    norm_required = _normalize(required)
    return any(_normalize(c) == norm_required for c in components)


def _language_matches(provider_languages: List[str], requested: str) -> bool:
    norm_req = _normalize(requested)
    return any(_normalize(lang) == norm_req for lang in provider_languages)


def _insurance_matches(accepted: List[str], requested: str) -> bool:
    norm_req = _normalize(requested)
    return any(_normalize(plan) == norm_req for plan in accepted)


def _city_matches(provider_city: Optional[str], requested: Optional[str]) -> bool:
    if not requested or not provider_city:
        return False
    return _normalize(provider_city) == _normalize(requested)


def _determine_insurance_note(provider: ProviderMatch, query: ProviderSearchQuery) -> str:
    """
    Build an insurance display note. Never guarantee coverage.
    """
    if not provider.accepted_insurance:
        return "Insurance information not available."
    listed = "; ".join(provider.accepted_insurance)
    note = f"Listed as accepting: {listed}."
    if query.insurance:
        if _insurance_matches(provider.accepted_insurance, query.insurance):
            note += f" ({query.insurance} is listed.)"
        else:
            note += f" ({query.insurance} is not listed.)"
    return note


# ─── Core audit & build ───────────────────────────────────────────────────────


def _audit_and_build(
    raw_provider: ProviderMatch,
    query: ProviderSearchQuery,
    matched_attrs: List[MatchAttribute],
    has_city_match: bool,
    has_lang_match: bool,
) -> ProviderMatch:
    """
    Deep-copy the provider, run neutrality + provenance audits,
    set patient_releasable/review_required, and return the audited copy.
    """
    p = copy.deepcopy(raw_provider)
    # Reset review state — audit will set review_required=True only for genuine reasons.
    # Fixture defaults conservatively to True; the audit is the authoritative gate.
    p.review_required = False
    p.review_reasons = []

    # ── Neutrality audit ──────────────────────────────────────────────────────
    neutrality_status, violations = audit_provider_neutrality(
        p.display_name,
        p.facility_name,
        None,  # no free-text description in Phase 7
    )
    p.neutrality_status = neutrality_status
    p.neutrality_violations = violations

    # If neutrality fails: sanitize display fields and block patient release
    if neutrality_status != NeutralityStatus.PASS:
        p.display_name = sanitize_display_text(p.display_name)
        if p.facility_name:
            p.facility_name = sanitize_display_text(p.facility_name)
        p.review_required = True
        p.review_reasons.append(
            "Provider description contained unsupported quality claims. "
            "Record sanitized and flagged for review."
        )

    # ── Conflicting / ambiguous specialty ────────────────────────────────────
    if "/" in raw_provider.specialty or ";" in raw_provider.specialty:
        p.review_required = True
        p.review_reasons.append(
            f"Provider specialty '{raw_provider.specialty}' is ambiguous or multi-value. "
            "Requires clinical review."
        )

    # ── Stale data flag ───────────────────────────────────────────────────────
    if p.freshness_status == FreshnessStatus.STALE:
        p.review_required = True
        p.review_reasons.append(
            "Provider data is marked STALE. Information may be outdated."
        )

    # ── Provenance audit ──────────────────────────────────────────────────────
    prov_ok, prov_notes = validate_provider_provenance(p)
    p.provenance_valid = prov_ok
    p.provenance_notes = prov_notes

    if not prov_ok:
        p.review_required = True
        p.review_reasons.append("Provenance validation failed — see provenance_notes.")

    # ── Matching attributes ───────────────────────────────────────────────────
    p.matching_attributes = list(matched_attrs)

    # ── Insurance note (never guarantee coverage) ─────────────────────────────
    p.insurance_note = _determine_insurance_note(p, query)

    # ── Final patient_releasable ──────────────────────────────────────────────
    p.patient_releasable = (
        prov_ok
        and neutrality_status == NeutralityStatus.PASS
        and not p.review_required
        and len(p.review_reasons) == 0
    )

    # Safety: if stale or ambiguous, never releasable without review
    if p.freshness_status == FreshnessStatus.STALE or "/" in raw_provider.specialty:
        p.patient_releasable = False

    return p


# ─── Public API ───────────────────────────────────────────────────────────────


def find_providers(
    query: ProviderSearchQuery,
    provider_pool: Optional[List[ProviderMatch]] = None,
) -> ProviderSearchResult:
    """
    Find providers matching the search query.

    Args:
        query: Explicit patient-supplied search parameters.
        provider_pool: Override the default synthetic fixture pool
                       (used by tests or future data connectors).

    Returns:
        ProviderSearchResult with only patient_releasable providers in .providers,
        plus .excluded_count of those that failed audit.

    ORDERING (neutral, deterministic):
      1. City match (providers in requested city first)
      2. Language match
      3. Alphabetical provider_id

    NEVER modifies the care plan. NEVER creates appointments or reminders.
    """
    if provider_pool is None:
        from backend.provider_finder.fixtures import ALL_SYNTHETIC_PROVIDERS
        provider_pool = ALL_SYNTHETIC_PROVIDERS

    releasable: List[Tuple[ProviderMatch, bool, bool]] = []  # (audited, city_match, lang_match)
    excluded_count = 0
    seen_ids = set()

    for raw in provider_pool:
        # ── Step 0: Deduplication by provider_id ─────────────────────────────
        if raw.provider_id in seen_ids:
            continue
        seen_ids.add(raw.provider_id)

        # ── Step 1: Specialty filter (hard gate) ─────────────────────────────
        if not _specialty_matches(raw.specialty, query.specialty):
            continue  # wrong specialty — not included at all

        matched_attrs: List[MatchAttribute] = [MatchAttribute.SPECIALTY]

        # ── Step 2: Location filter ───────────────────────────────────────────
        city_match = False
        if query.location_city:
            if not _city_matches(raw.city, query.location_city):
                continue  # city was requested but doesn't match — exclude
            city_match = True
            matched_attrs.append(MatchAttribute.LOCATION)

        # ── Step 3: Language filter ───────────────────────────────────────────
        lang_match = False
        if query.language:
            if not _language_matches(raw.languages, query.language):
                continue  # language requested but not listed — exclude
            lang_match = True
            matched_attrs.append(MatchAttribute.LANGUAGE)

        # ── Step 4: Insurance filter ──────────────────────────────────────────
        if query.insurance:
            if _insurance_matches(raw.accepted_insurance, query.insurance):
                matched_attrs.append(MatchAttribute.INSURANCE)
            # Insurance mismatch does NOT exclude — shown with a note

        # ── Audit ─────────────────────────────────────────────────────────────
        audited = _audit_and_build(raw, query, matched_attrs, city_match, lang_match)

        if audited.patient_releasable:
            releasable.append((audited, city_match, lang_match))
        else:
            excluded_count += 1

    # ── Ordering: deterministic, neutral ─────────────────────────────────────
    # Sort key: (not city_match DESC, not lang_match DESC, provider_id ASC)
    releasable.sort(key=lambda t: (not t[1], not t[2], t[0].provider_id))
    sorted_providers = [t[0] for t in releasable]

    # ── Patient message ───────────────────────────────────────────────────────
    if not sorted_providers:
        if query.location_city:
            msg = (
                f"No {query.specialty} providers found in {query.location_city}. "
                f"Try broadening the search or removing the location filter."
            )
        else:
            msg = f"No providers matching {query.specialty} were found."
    else:
        location_part = f" in {query.location_city}" if query.location_city else ""
        msg = (
            f"Showing {len(sorted_providers)} {query.specialty} provider(s){location_part}. "
            f"CAREBRIDGE helps you find providers that match the care plan. "
            f"It does not rank or select providers."
        )

    # ── Aggregate freshness ────────────────────────────────────────────────────
    freshness_values = {p.freshness_status for p in sorted_providers}
    if FreshnessStatus.STALE in freshness_values:
        agg_freshness = FreshnessStatus.STALE
    elif FreshnessStatus.CURRENT in freshness_values:
        agg_freshness = FreshnessStatus.CURRENT
    else:
        agg_freshness = FreshnessStatus.UNKNOWN

    return ProviderSearchResult(
        query=query,
        providers=sorted_providers,
        result_count=len(sorted_providers),
        excluded_count=excluded_count,
        ordering_explanation=(
            "Ordered by: (1) exact city match, (2) explicit language match, "
            "(3) alphabetical provider_id."
        ),
        patient_message=msg,
        source_status="synthetic",
        freshness_status=agg_freshness,
        patient_releasable=len(sorted_providers) > 0 and excluded_count == 0,
        review_required=excluded_count > 0,
    )

