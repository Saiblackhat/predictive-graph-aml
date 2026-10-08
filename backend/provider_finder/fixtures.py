"""
backend/provider_finder/fixtures.py

Phase 7 — Provider Finder: Synthetic provider directory.

ALL entries are clearly labeled source="synthetic".
No real provider data. No real phone numbers. No real addresses.
No fabricated availability. No quality claims. No ranking.

12 synthetic providers covering:
  Cardiology, Nephrology, Dermatology, Oncology, Primary Care
  in Chennai, Bengaluru, Chengalpattu
  with English, Spanish, Tamil languages
"""
from __future__ import annotations

from typing import List

from backend.provider_finder.types import (
    FreshnessStatus,
    NeutralityStatus,
    ProviderMatch,
    ProviderType,
)

# ─── Source constant ──────────────────────────────────────────────────────────
_SRC = "synthetic"
_RETRIEVED = "2026-10-01T00:00:00"
_FRESH = FreshnessStatus.CURRENT


def _make(
    pid: str,
    name: str,
    ptype: ProviderType,
    specialty: str,
    facility: str,
    address: str,
    city: str,
    state: str,
    postal: str,
    phone: str,
    languages: List[str],
    insurance: List[str],
) -> ProviderMatch:
    """Build a synthetic ProviderMatch with safe defaults."""
    return ProviderMatch(
        provider_id=pid,
        display_name=name,
        provider_type=ptype,
        specialty=specialty,
        facility_name=facility,
        address=address,
        city=city,
        state=state,
        postal_code=postal,
        phone=phone,
        website=None,                      # no fabricated URLs
        languages=languages,
        accepted_insurance=insurance,
        distance_km=None,                  # never fabricated
        source=_SRC,
        source_url=None,
        source_retrieved_at=_RETRIEVED,
        freshness_status=_FRESH,
        freshness_note=None,
        matching_attributes=[],            # populated by matcher
        patient_releasable=False,          # set by matcher after audit
        review_required=True,              # default conservative
        review_reasons=[],
        neutrality_status=NeutralityStatus.PASS,            # set by matcher
        neutrality_violations=[],
        provenance_valid=False,            # set by provenance validator
        provenance_notes=[],
        availability_note="Availability not verified.",
        insurance_note=(
            "Listed as accepting: " + ", ".join(insurance) + "."
            if insurance
            else "Insurance information not available."
        ),
    )


# ─── Synthetic provider directory ─────────────────────────────────────────────

SYNTHETIC_PROVIDERS: List[ProviderMatch] = [
    # P01 — Cardiology, Chennai, English + Spanish
    _make(
        "P01", "Dr. A. Rajan (Synthetic)",
        ProviderType.SPECIALIST, "Cardiology",
        "Synthetic Hospital Alpha", "1 Hospital Road",
        "Chennai", "Tamil Nadu", "600001", "+91-44-0000-0001",
        ["English", "Spanish"],
        ["Synthetic Health Plan A", "Synthetic Health Plan B"],
    ),

    # P02 — Cardiology, Chennai, English only
    _make(
        "P02", "Dr. B. Krishnan (Synthetic)",
        ProviderType.SPECIALIST, "Cardiology",
        "Synthetic Clinic Beta", "2 Clinic Street",
        "Chennai", "Tamil Nadu", "600002", "+91-44-0000-0002",
        ["English"],
        ["Synthetic Health Plan A"],
    ),

    # P03 — Dermatology, Chennai, English
    _make(
        "P03", "Dr. C. Sharma (Synthetic)",
        ProviderType.SPECIALIST, "Dermatology",
        "Synthetic Clinic Gamma", "3 Skin Centre Lane",
        "Chennai", "Tamil Nadu", "600003", "+91-44-0000-0003",
        ["English"],
        ["Synthetic Health Plan C"],
    ),

    # P04 — Cardiology, Bengaluru, English
    _make(
        "P04", "Dr. D. Nair (Synthetic)",
        ProviderType.SPECIALIST, "Cardiology",
        "Synthetic Hospital Delta", "4 Heart Avenue",
        "Bengaluru", "Karnataka", "560001", "+91-80-0000-0004",
        ["English"],
        ["Synthetic Health Plan B"],
    ),

    # P05 — Nephrology, Chennai, English
    _make(
        "P05", "Dr. E. Menon (Synthetic)",
        ProviderType.SPECIALIST, "Nephrology",
        "Synthetic Hospital Epsilon", "5 Kidney Road",
        "Chennai", "Tamil Nadu", "600005", "+91-44-0000-0005",
        ["English"],
        ["Synthetic Health Plan A"],
    ),

    # P06 — Cardiology, Chengalpattu, English
    _make(
        "P06", "Dr. F. Pillai (Synthetic)",
        ProviderType.CLINIC, "Cardiology",
        "Synthetic Clinic Zeta", "6 Cardio Street",
        "Chengalpattu", "Tamil Nadu", "603001", "+91-44-0000-0006",
        ["English"],
        [],
    ),

    # P07 — Cardiology, Chennai, Tamil
    _make(
        "P07", "Dr. G. Murugan (Synthetic)",
        ProviderType.HOSPITAL, "Cardiology",
        "Synthetic Hospital Eta", "7 Rajiv Gandhi Salai",
        "Chennai", "Tamil Nadu", "600007", "+91-44-0000-0007",
        ["Tamil"],
        ["Synthetic Health Plan D"],
    ),

    # P08 — Nephrology, Bengaluru, English
    _make(
        "P08", "Dr. H. Iyer (Synthetic)",
        ProviderType.HOSPITAL, "Nephrology",
        "Synthetic Hospital Theta", "8 Kidney Avenue",
        "Bengaluru", "Karnataka", "560008", "+91-80-0000-0008",
        ["English"],
        ["Synthetic Health Plan B"],
    ),

    # P09 — Oncology, Chennai, English
    _make(
        "P09", "Dr. I. Balan (Synthetic)",
        ProviderType.HOSPITAL, "Oncology",
        "Synthetic Cancer Center Iota", "9 Oncology Road",
        "Chennai", "Tamil Nadu", "600009", "+91-44-0000-0009",
        ["English"],
        ["Synthetic Health Plan A", "Synthetic Health Plan C"],
    ),

    # P10 — Cardiology, Chennai, Spanish
    _make(
        "P10", "Dr. J. Castillo (Synthetic)",
        ProviderType.CLINIC, "Cardiology",
        "Synthetic Clinic Kappa", "10 Cardio Lane",
        "Chennai", "Tamil Nadu", "600010", "+91-44-0000-0010",
        ["Spanish"],
        [],
    ),

    # P11 — Primary Care, Chennai, English
    _make(
        "P11", "Dr. K. Patel (Synthetic)",
        ProviderType.CLINIC, "Primary Care",
        "Synthetic Clinic Lambda", "11 Primary Road",
        "Chennai", "Tamil Nadu", "600011", "+91-44-0000-0011",
        ["English"],
        ["Synthetic Health Plan A", "Synthetic Health Plan D"],
    ),

    # P12 — Cardiology, Chennai, English (second hospital)
    _make(
        "P12", "Dr. L. Venkat (Synthetic)",
        ProviderType.HOSPITAL, "Cardiology",
        "Synthetic Hospital Mu", "12 Heart Road",
        "Chennai", "Tamil Nadu", "600012", "+91-44-0000-0012",
        ["English"],
        ["Synthetic Health Plan B", "Synthetic Health Plan C"],
    ),
]

# ─── Provider with an intentional quality claim (for adversarial tests) ───────
# Source text contains a banned claim; neutrality checker must catch it.

ADVERSARIAL_CLAIMED_PROVIDER = ProviderMatch(
    provider_id="ADV-G",
    display_name="Best Cardiologist in Chennai (Synthetic)",   # BANNED CLAIM
    provider_type=ProviderType.SPECIALIST,
    specialty="Cardiology",
    facility_name="Top-Rated Heart Clinic (Synthetic)",         # BANNED CLAIM
    address="99 Adversarial Road",
    city="Chennai",
    state="Tamil Nadu",
    postal_code="600099",
    phone="+91-44-0000-9999",
    website=None,
    languages=["English"],
    accepted_insurance=["Synthetic Health Plan X"],
    distance_km=None,
    source=_SRC,
    source_url=None,
    source_retrieved_at=_RETRIEVED,
    freshness_status=FreshnessStatus.CURRENT,
    freshness_note=None,
    matching_attributes=[],
    patient_releasable=False,
    review_required=True,
    review_reasons=[],
    neutrality_status=NeutralityStatus.PASS,
    neutrality_violations=[],
    provenance_valid=False,
    provenance_notes=[],
    availability_note="Availability not verified.",
    insurance_note="Listed as accepting: Synthetic Health Plan X.",
)

ADVERSARIAL_STALE_PROVIDER = ProviderMatch(
    provider_id="ADV-K",
    display_name="Dr. Stale Data (Synthetic)",
    provider_type=ProviderType.SPECIALIST,
    specialty="Cardiology",
    facility_name="Synthetic Stale Clinic",
    address="55 Old Data Street",
    city="Chennai",
    state="Tamil Nadu",
    postal_code="600055",
    phone="+91-44-0000-5555",
    website=None,
    languages=["English"],
    accepted_insurance=["Synthetic Health Plan Z"],
    distance_km=None,
    source=_SRC,
    source_url=None,
    source_retrieved_at="2024-01-01T00:00:00",    # old retrieval date
    freshness_status=FreshnessStatus.STALE,
    freshness_note="Provider data not refreshed since 2024. Information may be outdated.",
    matching_attributes=[],
    patient_releasable=False,
    review_required=True,
    review_reasons=[],
    neutrality_status=NeutralityStatus.PASS,
    neutrality_violations=[],
    provenance_valid=False,
    provenance_notes=[],
    availability_note="Availability not verified.",
    insurance_note="Listed as accepting: Synthetic Health Plan Z.",
)

ADVERSARIAL_NO_SOURCE_PROVIDER = ProviderMatch(
    provider_id="ADV-J",
    display_name="Dr. No Source (Synthetic)",
    provider_type=ProviderType.UNKNOWN,
    specialty="Cardiology",
    facility_name=None,
    address=None,
    city="Chennai",
    state=None,
    postal_code=None,
    phone=None,
    website=None,
    languages=[],
    accepted_insurance=[],
    distance_km=None,
    source="unknown",                   # "unknown" is explicitly rejected by provenance.py
    source_url=None,
    source_retrieved_at=None,
    freshness_status=FreshnessStatus.UNKNOWN,
    freshness_note=None,
    matching_attributes=[],
    patient_releasable=False,
    review_required=True,
    review_reasons=[],
    neutrality_status=NeutralityStatus.PASS,
    neutrality_violations=[],
    provenance_valid=False,
    provenance_notes=[],
    availability_note="Availability not verified.",
    insurance_note="Insurance information not available.",
)

ADVERSARIAL_CONFLICTING_SPECIALTY_PROVIDER = ProviderMatch(
    provider_id="ADV-O",
    display_name="Dr. Conflicting Specialty (Synthetic)",
    provider_type=ProviderType.SPECIALIST,
    specialty="Cardiology / Dermatology",   # ambiguous multi-specialty
    facility_name="Synthetic Multi Clinic",
    address="77 Conflict Road",
    city="Chennai",
    state="Tamil Nadu",
    postal_code="600077",
    phone="+91-44-0000-7777",
    website=None,
    languages=["English"],
    accepted_insurance=[],
    distance_km=None,
    source=_SRC,
    source_url=None,
    source_retrieved_at=_RETRIEVED,
    freshness_status=FreshnessStatus.CURRENT,
    freshness_note=None,
    matching_attributes=[],
    patient_releasable=False,
    review_required=True,
    review_reasons=[],
    neutrality_status=NeutralityStatus.PASS,
    neutrality_violations=[],
    provenance_valid=False,
    provenance_notes=[],
    availability_note="Availability not verified.",
    insurance_note="Insurance information not available.",
)

# ─── All-providers list for search (only the clean 12 synthetic ones) ─────────
ALL_SYNTHETIC_PROVIDERS = list(SYNTHETIC_PROVIDERS)
