"""Core Pydantic v2 data contracts for CAREBRIDGE.

Enforces strict type validation, forbids undeclared fields, requires explicit
representation of unknown values (None), and preserves verbatim source evidence,
confidence scores, and safety/review status across every extracted item.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MANDATORY_CLINICAL_DISCLAIMER = (
    "CAREBRIDGE organizes hospital discharge instructions and does NOT diagnose, "
    "prescribe, modify medications, recommend treatment, or guarantee provider "
    "suitability or availability. Always verify instructions with your care team."
)

SPAN_NOT_IMPLEMENTED_NOTE = "Source span computation not yet implemented."


class SafetyStatus(str, Enum):
    """UI and pipeline safety classification required across all extracted items."""

    SAFE_ORGANIZED = "SAFE / ORGANIZED"
    NEEDS_REVIEW = "NEEDS REVIEW"
    CONFLICT_DETECTED = "CONFLICT DETECTED"


class ReviewDecisionStatus(str, Enum):
    """Lifecycle state for human-in-the-loop clinical review."""

    AUTO_APPROVED = "AUTO_APPROVED"
    PENDING_HUMAN_REVIEW = "PENDING_HUMAN_REVIEW"
    HUMAN_RESOLVED = "HUMAN_RESOLVED"
    ESCALATED_TO_CLINICIAN = "PENDING_HUMAN_REVIEW"


class MedicationAction(str, Enum):
    """Explicit medication transition action documented at discharge."""

    CONTINUE = "CONTINUE"
    NEW_START = "NEW_START"
    STOP_DISCONTINUE = "STOP_DISCONTINUE"
    DOSE_CHANGE = "DOSE_CHANGE"
    HOLD_TEMPORARY = "HOLD_TEMPORARY"
    UNKNOWN = "UNKNOWN"


class UrgencyLevel(str, Enum):
    """Urgency classification for warning signs and follow-up actions."""

    ROUTINE = "ROUTINE"
    SOON = "SOON"
    URGENT = "URGENT"
    EMERGENT_911 = "EMERGENT_911"
    UNKNOWN = "UNKNOWN"


class HazardCode(str, Enum):
    """Mandatory safety hazard codes detected by the Safety Validation Layer."""

    MISSING_FOLLOWUP_TIMEFRAME = "MISSING_FOLLOWUP_TIMEFRAME"
    MISSING_SPECIALTY = "MISSING_SPECIALTY"
    AMBIGUOUS_MEDICATION_INSTRUCTION = "AMBIGUOUS_MEDICATION_INSTRUCTION"
    MEDICATION_CHANGE_DETECTED = "MEDICATION_CHANGE_DETECTED"
    STOP_START_MEDICATION = "STOP_START_MEDICATION"
    UNCLEAR_DOSAGE = "UNCLEAR_DOSAGE"
    CONFLICTING_INSTRUCTIONS = "CONFLICTING_INSTRUCTIONS"
    UNCLEAR_SYMPTOMS = "UNCLEAR_SYMPTOMS"
    WARNING_SIGN_ESCALATION = "WARNING_SIGN_ESCALATION"
    UNSUPPORTED_PROVIDER_CLAIM = "UNSUPPORTED_PROVIDER_CLAIM"
    MISSING_SOURCE_EVIDENCE = "MISSING_SOURCE_EVIDENCE"
    PROHIBITED_CLINICAL_ACTION = "PROHIBITED_CLINICAL_ACTION"
    LOW_EXTRACTION_CONFIDENCE = "LOW_EXTRACTION_CONFIDENCE"
    UNSUPPORTED_FIELD_VALUE = "UNSUPPORTED_FIELD_VALUE"
    OMITTED_INSTRUCTION = "OMITTED_INSTRUCTION"
    INCOMPLETE_EXTRACTION = "INCOMPLETE_EXTRACTION"
    DUPLICATE_ITEM = "DUPLICATE_ITEM"


class HazardSeverity(str, Enum):
    """Severity level of a detected safety hazard."""

    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class StrictSchemaModel(BaseModel):
    """Base model enforcing strict types and rejecting any undeclared fields."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_strict_enums_from_str(cls, data: object) -> object:
        """Allow valid string representations of Enums when FastAPI validates JSON dicts under strict=True."""
        if not isinstance(data, dict):
            return data
        coerced = dict(data)
        enum_map: dict[str, type[Enum]] = {
            "safety_status": SafetyStatus,
            "overall_safety_status": SafetyStatus,
            "review_status": ReviewDecisionStatus,
            "action": MedicationAction,
            "urgency": UrgencyLevel,
            "code": HazardCode,
            "severity": HazardSeverity,
        }
        for field_name, enum_cls in enum_map.items():
            raw_val = coerced.get(field_name)
            if isinstance(raw_val, str) and not isinstance(raw_val, enum_cls):
                if enum_cls is ReviewDecisionStatus and raw_val == "ESCALATED_TO_CLINICIAN":
                    coerced[field_name] = ReviewDecisionStatus.PENDING_HUMAN_REVIEW
                    continue
                try:
                    coerced[field_name] = enum_cls(raw_val)
                except ValueError:
                    # Leave invalid string untouched so Pydantic raises a ValidationError
                    pass
        if "hazard_codes_detected" in coerced and isinstance(
            coerced["hazard_codes_detected"], list
        ):
            new_codes = []
            for item in coerced["hazard_codes_detected"]:
                if isinstance(item, str) and not isinstance(item, HazardCode):
                    try:
                        new_codes.append(HazardCode(item))
                    except ValueError:
                        new_codes.append(item)
                else:
                    new_codes.append(item)
            coerced["hazard_codes_detected"] = new_codes
        return coerced


class SourceEvidence(StrictSchemaModel):
    """Verbatim source evidence anchoring an extracted instruction to the document."""

    source_text: str = Field(
        ...,
        min_length=1,
        description="Verbatim quote from the discharge document.",
    )
    source_section: Optional[str] = Field(
        default=None,
        description="Document section header (e.g., 'Discharge Medications').",
    )
    source_page: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-indexed page number when available (unverified until document processor maps pages).",
    )
    start_char: Optional[int] = Field(
        default=None,
        ge=0,
        description="0-indexed starting character offset in raw document.",
    )
    end_char: Optional[int] = Field(
        default=None,
        ge=0,
        description="0-indexed ending character offset in raw document.",
    )
    span_status_note: str = Field(
        default=SPAN_NOT_IMPLEMENTED_NOTE,
        description="Explicit notice when character spans are not yet computed.",
    )
    source_page_verified: bool = Field(
        default=False,
        description="False until document processor computes real page boundaries; do not show source_page as verified UI evidence.",
    )

    @model_validator(mode="after")
    def validate_char_span(self) -> SourceEvidence:
        if self.start_char is not None and self.end_char is not None:
            if self.end_char < self.start_char:
                raise ValueError("end_char must be greater than or equal to start_char")
        return self


class SafetyFlag(StrictSchemaModel):
    """Structured safety finding emitted by the Safety Validation Layer."""

    code: HazardCode
    severity: HazardSeverity
    message: str = Field(..., min_length=3)
    item_id: Optional[str] = None
    requires_human_review: bool = True
    evidence_snippet: Optional[str] = None


class ExtractedItemBase(StrictSchemaModel):
    """Base contract for every extracted clinical instruction."""

    id: str = Field(..., min_length=1, description="Unique identifier for this item.")
    source_evidence: Optional[SourceEvidence] = Field(
        default=None,
        description="Source text and location; if missing or unverified, item is escalated.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score in [0.0, 1.0].",
    )
    safety_status: SafetyStatus = Field(
        default=SafetyStatus.SAFE_ORGANIZED,
        description="Current UI/safety status badge.",
    )
    review_status: ReviewDecisionStatus = Field(
        default=ReviewDecisionStatus.AUTO_APPROVED,
        description="Human review workflow status.",
    )
    safety_flags: list[SafetyFlag] = Field(
        default_factory=list,
        description="All safety hazards attached to this item.",
    )
    patient_friendly_summary: Optional[str] = Field(
        default=None,
        description="Plain-language explanation constrained to source evidence.",
    )


class MedicationInstruction(ExtractedItemBase):
    """Structured representation of a medication instruction at discharge."""

    medication_name: str = Field(..., min_length=1)
    action: MedicationAction = Field(default=MedicationAction.UNKNOWN)
    dosage: Optional[str] = Field(
        default=None,
        description="Explicit dosage (e.g., '20 mg'). Must be None if unknown.",
    )
    frequency: Optional[str] = Field(
        default=None,
        description="Explicit schedule/frequency (e.g., 'twice daily'). None if unknown.",
    )
    route: Optional[str] = Field(
        default=None,
        description="Route of administration (e.g., 'PO', 'by mouth'). None if unknown.",
    )
    previous_dosage: Optional[str] = Field(
        default=None,
        description="Pre-admission or prior dosage when a dose change is noted.",
    )
    indication_or_note: Optional[str] = Field(
        default=None,
        description="Additional instructions or conditions noted in source text.",
    )


class FollowUpInstruction(ExtractedItemBase):
    """Structured representation of a clinical follow-up appointment instruction."""

    specialty: Optional[str] = Field(
        default=None,
        description="Target medical specialty (e.g., 'Cardiology'). None if missing.",
    )
    provider_name: Optional[str] = Field(
        default=None,
        description="Specific clinician or clinic name if stated.",
    )
    timeframe: Optional[str] = Field(
        default=None,
        description="Explicit follow-up window (e.g., 'within 7 days'). None if missing.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Documented purpose of follow-up.",
    )
    appointment_scheduled: Optional[bool] = Field(
        default=None,
        description="Whether appointment is already booked, if known.",
    )
    provider_claim_note: Optional[str] = Field(
        default=None,
        description="Any claim made regarding provider availability or suitability.",
    )


class WarningSignInstruction(ExtractedItemBase):
    """Structured representation of a post-discharge return precaution or red flag."""

    symptom_description: str = Field(..., min_length=1)
    urgency: UrgencyLevel = Field(default=UrgencyLevel.UNKNOWN)
    action_to_take: Optional[str] = Field(
        default=None,
        description="Explicit action documented (e.g., 'Call 911', 'Call Cardiology').",
    )


class CareTaskItem(ExtractedItemBase):
    """Trackable patient checklist item derived from validated discharge instructions."""

    category: Literal[
        "MEDICATION",
        "FOLLOW_UP",
        "WARNING_SIGN",
        "ACTIVITY_DIET",
        "LAB_TEST",
    ]
    title: str = Field(..., min_length=1)
    due_window: Optional[str] = None
    linked_item_id: str = Field(..., min_length=1)


class SyntheticProviderMatch(StrictSchemaModel):
    """Synthetic directory entry with mandatory non-endorsement & availability disclaimer."""

    provider_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    specialty: str = Field(..., min_length=1)
    facility: str = Field(..., min_length=1)
    distance_miles: float = Field(..., ge=0.0)
    accepts_new_patients_listed: bool
    verification_disclaimer: str = Field(
        default=(
            "Synthetic directory listing only. CAREBRIDGE does not guarantee "
            "provider suitability, insurance network coverage, or appointment availability."
        ),
        min_length=10,
    )
    claim_text: Optional[str] = Field(
        default=None,
        description="Optional descriptive text audited for unsupported guarantees.",
    )


def _item_clinical_fingerprint(item: ExtractedItemBase) -> dict[str, object]:
    """Return a canonical dictionary of an item's clinical content (excluding status/flags)."""
    dumped = item.model_dump(
        exclude={"safety_status", "review_status", "safety_flags"}
    )
    return dumped


class DischargeExtractionPayload(StrictSchemaModel):
    """Complete structured discharge plan subject to deterministic safety validation."""

    case_id: str = Field(..., min_length=1)
    patient_identifier: str = Field(..., min_length=1)
    raw_discharge_text: str = Field(..., min_length=1)
    medications: list[MedicationInstruction] = Field(default_factory=list)
    follow_ups: list[FollowUpInstruction] = Field(default_factory=list)
    warning_signs: list[WarningSignInstruction] = Field(default_factory=list)
    tasks: list[CareTaskItem] = Field(default_factory=list)
    provider_matches: list[SyntheticProviderMatch] = Field(default_factory=list)
    overall_safety_status: SafetyStatus = Field(default=SafetyStatus.SAFE_ORGANIZED)
    requires_human_review: bool = Field(default=False)
    global_safety_flags: list[SafetyFlag] = Field(default_factory=list)
    clinical_disclaimer: str = Field(default=MANDATORY_CLINICAL_DISCLAIMER)

    @field_validator("clinical_disclaimer")
    @classmethod
    def ensure_non_empty_disclaimer(cls, value: str) -> str:
        if len(value.strip()) < 20:
            raise ValueError("clinical_disclaimer must be a substantive safety notice.")
        return value

    @model_validator(mode="after")
    def reject_conflicting_duplicate_ids(self) -> DischargeExtractionPayload:
        """Enforce unique IDs: same ID with different clinical content is a hard schema error."""
        seen_by_id: dict[str, dict[str, object]] = {}
        all_items: list[ExtractedItemBase] = [
            *self.medications,
            *self.follow_ups,
            *self.warning_signs,
            *self.tasks,
        ]
        for item in all_items:
            fp = _item_clinical_fingerprint(item)
            if item.id in seen_by_id:
                if seen_by_id[item.id] != fp:
                    raise ValueError(
                        f"Duplicate item ID '{item.id}' with different clinical content is prohibited."
                    )
            else:
                seen_by_id[item.id] = fp
        return self


class ValidationReport(StrictSchemaModel):
    """Result returned by the Safety Validation Layer after auditing a payload."""

    case_id: str
    overall_safety_status: SafetyStatus
    requires_human_review: bool
    patient_releasable: bool = Field(
        default=False,
        description="True ONLY when the entire validated payload is safe and all items are approved.",
    )
    total_flags: int = Field(..., ge=0)
    hazard_codes_detected: list[HazardCode] = Field(default_factory=list)
    validated_payload: DischargeExtractionPayload
