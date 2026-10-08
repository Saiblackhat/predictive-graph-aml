"""
backend/provider_finder/types.py

Phase 7 — Provider Finder: Data contracts.

These models define what a ProviderMatch and ProviderSearchResult look like.
They are strict, auditable, and carry full provenance.

SAFETY BOUNDARY:
  - No ranking scores
  - No quality claims
  - No fabricated availability
  - No fabricated insurance guarantees
  - No fabricated distance
  - No inferred patient data
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import Field

from backend.schemas.models import StrictSchemaModel


# ─── Enumerations ────────────────────────────────────────────────────────────


class ProviderType(str, Enum):
    PHYSICIAN = "physician"
    SPECIALIST = "specialist"
    HOSPITAL = "hospital"
    CLINIC = "clinic"
    GROUP_PRACTICE = "group_practice"
    TELEHEALTH = "telehealth"
    UNKNOWN = "unknown"


class FreshnessStatus(str, Enum):
    CURRENT = "CURRENT"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class MatchAttribute(str, Enum):
    SPECIALTY = "specialty"
    LOCATION = "location"
    LANGUAGE = "language"
    INSURANCE = "insurance"


class NeutralityStatus(str, Enum):
    """Result of the provider neutrality check."""
    PASS = "PASS"
    FAIL_QUALITY_CLAIM = "FAIL_QUALITY_CLAIM"
    FAIL_AVAILABILITY_CLAIM = "FAIL_AVAILABILITY_CLAIM"
    FAIL_RANKING_CLAIM = "FAIL_RANKING_CLAIM"


# ─── Provider data model ─────────────────────────────────────────────────────


class ProviderMatch(StrictSchemaModel):
    """
    A single provider candidate from the discovery layer.

    DISPLAY RULES for Phase 8 (enforced here as documentation):
      - Do NOT display quality claims (best, top, recommended, etc.)
      - Do NOT display availability claims unless live-verified
      - Do NOT display insurance guarantees
      - Do NOT display fabricated distance
      - DO display matching_attributes to explain WHY the provider appears
    """

    provider_id: str = Field(..., min_length=1, max_length=50)
    display_name: str = Field(..., min_length=1, max_length=200)
    provider_type: ProviderType = Field(default=ProviderType.UNKNOWN)
    specialty: str = Field(..., min_length=1, max_length=200)
    facility_name: Optional[str] = Field(default=None, max_length=200)
    address: Optional[str] = Field(default=None, max_length=500)
    city: Optional[str] = Field(default=None, max_length=100)
    state: Optional[str] = Field(default=None, max_length=100)
    postal_code: Optional[str] = Field(default=None, max_length=20)
    phone: Optional[str] = Field(default=None, max_length=30)
    website: Optional[str] = Field(default=None, max_length=300)

    # Languages explicitly listed in the source. Never inferred.
    languages: List[str] = Field(default_factory=list)

    # Insurance plans explicitly listed in the source. Never guaranteed.
    accepted_insurance: List[str] = Field(default_factory=list)

    # Distance in km from patient location. None if coordinates unavailable.
    distance_km: Optional[float] = Field(default=None, ge=0.0)

    # Provenance
    source: str = Field(..., min_length=1, max_length=200,
                        description="Data source label (e.g. 'synthetic', 'public_directory').")
    source_url: Optional[str] = Field(default=None, max_length=500)
    source_retrieved_at: Optional[str] = Field(default=None, max_length=50,
                                               description="ISO 8601 timestamp or None.")

    # Freshness
    freshness_status: FreshnessStatus = Field(default=FreshnessStatus.UNKNOWN)
    freshness_note: Optional[str] = Field(default=None, max_length=300)

    # Matching explanation — why this provider appears in results
    matching_attributes: List[MatchAttribute] = Field(default_factory=list)

    # Safety / releasability
    patient_releasable: bool = Field(default=False)
    review_required: bool = Field(default=True)
    review_reasons: List[str] = Field(default_factory=list)

    # Neutrality audit (populated by neutrality.py)
    neutrality_status: NeutralityStatus = Field(default=NeutralityStatus.PASS)
    neutrality_violations: List[str] = Field(default_factory=list)

    # Provenance audit
    provenance_valid: bool = Field(default=False)
    provenance_notes: List[str] = Field(default_factory=list)

    # Availability: NEVER fabricated. Always stated as unverified unless live data.
    availability_note: str = Field(
        default="Availability not verified.",
        max_length=300,
    )

    # Insurance note: never guarantee coverage
    insurance_note: str = Field(
        default="Insurance information not available.",
        max_length=300,
    )


# ─── Search query ─────────────────────────────────────────────────────────────


class ProviderSearchQuery(StrictSchemaModel):
    """Explicit, auditable search query. All patient-supplied fields only."""
    specialty: str = Field(..., min_length=1, max_length=200)
    location_city: Optional[str] = Field(default=None, max_length=100)
    location_state: Optional[str] = Field(default=None, max_length=100)
    language: Optional[str] = Field(default=None, max_length=50)
    insurance: Optional[str] = Field(default=None, max_length=200)


# ─── Search result ────────────────────────────────────────────────────────────


class ProviderSearchResult(StrictSchemaModel):
    """
    Output of find_providers().

    INVARIANTS:
      - providers list contains only patient_releasable entries (review_required=False)
        unless the caller explicitly requests all candidates.
      - result_count reflects releasable providers only.
      - No ranking scores. No quality claims.
      - Empty list is valid — never fabricate a provider.
    """
    query: ProviderSearchQuery
    providers: List[ProviderMatch] = Field(default_factory=list)
    result_count: int = Field(default=0, ge=0)
    excluded_count: int = Field(default=0, ge=0,
                                description="Providers excluded due to review/neutrality failures.")

    # Ordering explanation — deterministic, neutral, auditable
    ordering_explanation: str = Field(
        default=(
            "Ordered by: (1) exact specialty match, (2) exact city match, "
            "(3) explicit language match, (4) alphabetical provider_id."
        )
    )

    # Informational message for the patient
    patient_message: str = Field(default="", max_length=500)

    # Source quality of the data layer
    source_status: str = Field(default="synthetic", max_length=100)
    freshness_status: FreshnessStatus = Field(default=FreshnessStatus.UNKNOWN)

    # Aggregate safety
    patient_releasable: bool = Field(default=False)
    review_required: bool = Field(default=True)
