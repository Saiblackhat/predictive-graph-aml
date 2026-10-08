"""
backend/provider_finder/neutrality.py

Phase 7 — Provider Finder: Neutrality checker.

Scans provider text fields for unsupported quality claims, ranking language,
guaranteed availability, or superiority claims, and blocks them from reaching
patient-facing output.

Design principles:
  - Deterministic (no ML, no scoring).
  - Conservative: if a claim is borderline, flag it.
  - Does NOT alter provider clinical data.
  - Applies to display_name, facility_name, and any notes from the source.
"""
from __future__ import annotations

import re
from typing import Tuple, List

from backend.provider_finder.types import NeutralityStatus

# ─── Banned quality/ranking phrases ─────────────────────────────────────────
# These phrases must never appear in patient-facing provider descriptions.
# If found, the description is sanitized and the violation is logged.

_QUALITY_CLAIM_PATTERNS: List[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bbest\b",
    r"\btop[\s-]rated\b",
    r"\btop\s+\d+\b",           # "top 10"
    r"\btop\b",                 # standalone "top"
    r"\brecommended\b",
    r"\bmost\s+qualified\b",
    r"\bhighest[\s-]rated\b",
    r"\bleading\b",
    r"\btrusted\b",
    r"\bexpert\b",
    r"\bnumber\s+one\b",
    r"\b#\s*1\b",
    r"\bperfect\s+match\b",
    r"\bgold\s+standard\b",
    r"\bsuperior\b",
    r"\bpreferred\s+provider\b",
    r"\bpreferred\b",
    r"\bexcellent\s+(doctor|provider|physician|specialist|cardiologist|care)\b",
    r"\bworld[- ]class\b",
    r"\bpremier\b",
    r"\belite\b",
    r"\brenowned\b",
    r"\baward[-\s]winning\b",
    r"\bhighly\s+rated\b",
    r"\bhighly\s+recommended\b",
]]

_AVAILABILITY_CLAIM_PATTERNS: List[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bguaranteed\s+appointment\b",
    r"\bavailable\s+now\b",
    r"\bavailable\s+tomorrow\b",
    r"\bsame[\s-]day\s+appointment\b",  # unless verified
    r"\bbookable\s+now\b",
    r"\binstant\s+booking\b",
    r"\bappointment\s+guaranteed\b",
]]

_RANKING_CLAIM_PATTERNS: List[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\b\d{1,3}%\s+match\b",
    r"\bbest\s+match\b",
    r"\btop\s+choice\b",
    r"\btop\s+pick\b",
    r"\b#\s*1\s+pick\b",
    r"\bour\s+pick\b",
    r"\bwe\s+recommend\b",
    r"\bcarebridge\s+recommends\b",
]]


def check_text_neutrality(text: str) -> Tuple[NeutralityStatus, List[str]]:
    """
    Scan a free-text string for banned claims.

    Returns:
        (NeutralityStatus, list_of_violations)

    The list of violations contains human-readable descriptions of what was found.
    """
    if not text or not text.strip():
        return NeutralityStatus.PASS, []

    violations: List[str] = []

    for pat in _QUALITY_CLAIM_PATTERNS:
        m = pat.search(text)
        if m:
            violations.append(f"Unsupported quality claim: '{m.group(0)}' in '{text[:80]}'")

    for pat in _AVAILABILITY_CLAIM_PATTERNS:
        m = pat.search(text)
        if m:
            violations.append(f"Unverified availability claim: '{m.group(0)}' in '{text[:80]}'")

    for pat in _RANKING_CLAIM_PATTERNS:
        m = pat.search(text)
        if m:
            violations.append(f"Ranking claim: '{m.group(0)}' in '{text[:80]}'")

    if not violations:
        return NeutralityStatus.PASS, []

    # Determine primary status
    for pat in _AVAILABILITY_CLAIM_PATTERNS:
        if pat.search(text):
            return NeutralityStatus.FAIL_AVAILABILITY_CLAIM, violations
    for pat in _RANKING_CLAIM_PATTERNS:
        if pat.search(text):
            return NeutralityStatus.FAIL_RANKING_CLAIM, violations
    return NeutralityStatus.FAIL_QUALITY_CLAIM, violations


def sanitize_display_text(text: str) -> str:
    """
    Remove or replace banned phrases from a provider display string.

    This is applied when the raw source data contains quality claims so
    they cannot reach patient-facing output even if a provider passes other checks.

    Replacement strategy:
      - Quality/ranking adjectives → removed (e.g. "Best Cardiologist" → "Cardiologist")
      - Does NOT alter specialty name, facility name, or address.
    """
    result = text
    for pat in _QUALITY_CLAIM_PATTERNS + _RANKING_CLAIM_PATTERNS + _AVAILABILITY_CLAIM_PATTERNS:
        result = pat.sub("", result)
    # Collapse multiple spaces
    result = re.sub(r"\s{2,}", " ", result).strip(" ,.-")
    return result


def audit_provider_neutrality(
    display_name: str,
    facility_name: str | None,
    source_description: str | None,
) -> Tuple[NeutralityStatus, List[str]]:
    """
    Full neutrality audit across all patient-visible text fields of a provider record.
    Returns overall (status, violations).
    """
    all_violations: List[str] = []
    overall_status = NeutralityStatus.PASS

    for field_value in [display_name, facility_name, source_description]:
        if field_value:
            status, violations = check_text_neutrality(field_value)
            all_violations.extend(violations)
            if status != NeutralityStatus.PASS:
                overall_status = status  # last non-PASS wins; all failures logged

    return overall_status, all_violations
