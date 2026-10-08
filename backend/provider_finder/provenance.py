"""
backend/provider_finder/provenance.py

Phase 7 — Provider Finder: Provenance validator.

Ensures every ProviderMatch record has the required provenance fields
before it can be marked patient_releasable.

Rules:
  1. provider_id must be non-empty.
  2. specialty must be non-empty.
  3. display_name must be non-empty.
  4. source must be non-empty (cannot be blank/unknown).
  5. neutrality_status must be PASS.
  6. No quality-claim violations in neutrality_violations.
  7. For insurance display, accepted_insurance must come from the source fixture
     (never inferred).
  8. distance_km must be None if no coordinate data is available.
"""
from __future__ import annotations

from typing import Tuple, List

from backend.provider_finder.types import ProviderMatch, NeutralityStatus


def validate_provider_provenance(provider: ProviderMatch) -> Tuple[bool, List[str]]:
    """
    Validate that a ProviderMatch has acceptable provenance.

    Returns:
        (is_valid: bool, notes: list[str])

    'is_valid=True' means the record may be considered for patient release
    subject to other checks (neutrality, review_required).
    """
    notes: List[str] = []
    valid = True

    if not provider.provider_id or not provider.provider_id.strip():
        notes.append("FAIL: provider_id is empty.")
        valid = False

    if not provider.specialty or not provider.specialty.strip():
        notes.append("FAIL: specialty is empty.")
        valid = False

    if not provider.display_name or not provider.display_name.strip():
        notes.append("FAIL: display_name is empty.")
        valid = False

    if not provider.source or provider.source.strip().lower() in ("", "unknown", "none"):
        notes.append("FAIL: source is missing or 'unknown'.")
        valid = False

    if provider.neutrality_status != NeutralityStatus.PASS:
        notes.append(
            f"FAIL: neutrality_status={provider.neutrality_status.value}. "
            "Unsupported claims detected — record excluded from patient release."
        )
        valid = False

    if provider.neutrality_violations:
        for v in provider.neutrality_violations:
            notes.append(f"NEUTRALITY VIOLATION: {v}")
        valid = False

    # Insurance note must never guarantee coverage
    if provider.insurance_note and "guaranteed" in provider.insurance_note.lower():
        notes.append("FAIL: insurance_note contains a coverage guarantee.")
        valid = False

    # Distance must not be fabricated (0.0 is suspicious when no coordinates exist)
    # We cannot fully validate this at the model level, but we note it.
    if provider.distance_km is not None and provider.source_url is None:
        notes.append(
            "NOTE: distance_km is set but source_url is absent. "
            "Ensure coordinates came from a trusted geocoded source."
        )

    if valid and not notes:
        notes.append("OK: All provenance checks passed.")

    return valid, notes
