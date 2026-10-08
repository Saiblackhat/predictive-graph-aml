"""Core Pydantic v2 data contracts for CAREBRIDGE.

Enforces strict type validation, forbids undeclared fields, requires explicit
representation of unknown values (None), enforces length and collection caps,
and preserves verbatim source evidence, confidence scores, and safety/review
status across every extracted item.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MANDATORY_CLINICAL_DISCLAIMER = (
    "CAREBRIDGE organizes hospital discharge instructions and does NOT diagnose, "
    "prescribe, modify medications, recommend treatment, or guarantee provider "
    "suitability or availability. Always verify instructions with your care team."
)

SPAN_NOT_IMPLEMENTED_NOTE = "Source span computation not yet implemented."

MAX_RAW_DISCHARGE_TEXT_BYTES = 204_800  # 200 KB cap


class RawTextTooLargeError(ValueError):
    """Raised when raw_discharge_text exceeds the 200 KB (204,800 byte) cap."""


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
    SUSPICIOUS_TEXT_ENCODING = "SUSPICIOUS_TEXT_ENCODING"


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
                if (
                    enum_cls is ReviewDecisionStatus
                    and raw_val == "ESCALATED_TO_CLINICIAN"
                ):
                    coerced[field_name] = ReviewDecisionStatus.PENDING_HUMAN_REVIEW
                    continue
                try:
                    coerced[field_name] = enum_cls(raw_val)
                except ValueError:
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
        max_length=1000,
        description="Verbatim quote from the discharge document.",
    )
    source_section: Optional[str] = Field(
        default=None,
        max_length=200,
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
        max_length=200,
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
    message: str = Field(..., min_length=3, max_length=1000)
    item_id: Optional[str] = Field(default=None, max_length=200)
    requires_human_review: bool = True
    evidence_snippet: Optional[str] = Field(default=None, max_length=1000)


class ExtractedItemBase(StrictSchemaModel):
    """Base contract for every extracted clinical instruction."""

    id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique identifier for this item.",
    )
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
    release_blocked: bool = Field(
        default=False,
        description="True when item or global safety hazards block patient release.",
    )
    safety_flags: list[SafetyFlag] = Field(
        default_factory=list,
        max_length=100,
        description="All safety hazards attached to this item.",
    )
    patient_friendly_summary: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Plain-language explanation constrained to source evidence.",
    )
    provider_claim_note: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Any claim made regarding provider availability or suitability.",
    )


class MedicationInstruction(ExtractedItemBase):
    """Structured representation of a medication instruction at discharge."""

    medication_name: str = Field(..., min_length=1, max_length=200)
    action: MedicationAction = Field(default=MedicationAction.UNKNOWN)
    dosage: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Explicit dosage (e.g., '20 mg'). Must be None if unknown.",
    )
    frequency: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Explicit schedule/frequency (e.g., 'twice daily'). None if unknown.",
    )
    route: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Route of administration (e.g., 'PO', 'by mouth'). None if unknown.",
    )
    previous_dosage: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Pre-admission or prior dosage when a dose change is noted.",
    )
    indication_or_note: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Additional instructions or conditions noted in source text.",
    )


class FollowUpInstruction(ExtractedItemBase):
    """Structured representation of a clinical follow-up appointment instruction."""

    specialty: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Target medical specialty (e.g., 'Cardiology'). None if missing.",
    )
    provider_name: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Specific clinician or clinic name if stated.",
    )
    timeframe: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Explicit follow-up window (e.g., 'within 7 days'). None if missing.",
    )
    reason: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Documented purpose of follow-up.",
    )
    appointment_scheduled: Optional[bool] = Field(
        default=None,
        description="Whether appointment is already booked, if known.",
    )


class WarningSignInstruction(ExtractedItemBase):
    """Structured representation of a post-discharge return precaution or red flag."""

    symptom_description: str = Field(..., min_length=1, max_length=1000)
    urgency: UrgencyLevel = Field(default=UrgencyLevel.UNKNOWN)
    action_to_take: Optional[str] = Field(
        default=None,
        max_length=1000,
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
    title: str = Field(..., min_length=1, max_length=1000)
    due_window: Optional[str] = Field(default=None, max_length=200)
    linked_item_id: str = Field(..., min_length=1, max_length=200)


class SyntheticProviderMatch(StrictSchemaModel):
    """Synthetic directory entry with mandatory non-endorsement & availability disclaimer."""

    provider_id: str = Field(..., min_length=1, max_length=200)
    name: str = Field(..., min_length=1, max_length=200)
    specialty: str = Field(..., min_length=1, max_length=200)
    facility: str = Field(..., min_length=1, max_length=200)
    distance_miles: float = Field(..., ge=0.0)
    accepts_new_patients_listed: bool
    verification_disclaimer: str = Field(
        default=(
            "Synthetic directory listing only. CAREBRIDGE does not guarantee "
            "provider suitability, insurance network coverage, or appointment availability."
        ),
        min_length=10,
        max_length=1000,
    )
    claim_text: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional descriptive text audited for unsupported guarantees.",
    )


def _canonical_id_key(raw_id: str) -> str:
    """Case-insensitive, NFKC-normalized ID key for uniqueness checks."""
    norm = unicodedata.normalize("NFKC", raw_id)
    cleaned = "".join(
        ch for ch in norm if unicodedata.category(ch) != "Cf" and ch != "\u00ad"
    )
    return " ".join(cleaned.split()).casefold()


def _item_clinical_fingerprint(item: ExtractedItemBase) -> dict[str, object]:
    """Return a canonical dictionary of an item's clinical content (excluding ID and status/flags)."""
    dumped = item.model_dump(
        exclude={"id", "safety_status", "review_status", "release_blocked", "safety_flags"}
    )
    return dumped


def compute_payload_sha256(payload: DischargeExtractionPayload) -> str:
    """Compute a deterministic SHA-256 digest of a DischargeExtractionPayload."""
    serialized = json.dumps(
        payload.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class DischargeExtractionPayload(StrictSchemaModel):
    """Complete structured discharge plan subject to deterministic safety validation."""

    case_id: str = Field(..., min_length=1, max_length=200)
    patient_identifier: str = Field(..., min_length=1, max_length=200)
    raw_discharge_text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_RAW_DISCHARGE_TEXT_BYTES,
    )
    medications: list[MedicationInstruction] = Field(
        default_factory=list, max_length=100
    )
    follow_ups: list[FollowUpInstruction] = Field(
        default_factory=list, max_length=100
    )
    warning_signs: list[WarningSignInstruction] = Field(
        default_factory=list, max_length=100
    )
    tasks: list[CareTaskItem] = Field(default_factory=list, max_length=200)
    provider_matches: list[SyntheticProviderMatch] = Field(
        default_factory=list, max_length=50
    )
    overall_safety_status: SafetyStatus = Field(default=SafetyStatus.SAFE_ORGANIZED)
    requires_human_review: bool = Field(default=False)
    global_safety_flags: list[SafetyFlag] = Field(
        default_factory=list, max_length=200
    )
    clinical_disclaimer: str = Field(
        default=MANDATORY_CLINICAL_DISCLAIMER, max_length=1000
    )

    @field_validator("raw_discharge_text")
    @classmethod
    def check_raw_text_byte_cap(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_RAW_DISCHARGE_TEXT_BYTES:
            raise RawTextTooLargeError(
                f"raw_discharge_text exceeds maximum size of {MAX_RAW_DISCHARGE_TEXT_BYTES} bytes (200 KB)."
            )
        return value

    @field_validator("clinical_disclaimer")
    @classmethod
    def ensure_non_empty_disclaimer(cls, value: str) -> str:
        if len(value.strip()) < 20:
            raise ValueError("clinical_disclaimer must be a substantive safety notice.")
        return value

    @model_validator(mode="after")
    def reject_conflicting_duplicate_ids(self) -> DischargeExtractionPayload:
        """Enforce case-insensitive unique IDs: same canonical ID with different clinical content is a hard schema error."""
        seen_by_id: dict[str, dict[str, object]] = {}
        all_items: list[ExtractedItemBase] = [
            *self.medications,
            *self.follow_ups,
            *self.warning_signs,
            *self.tasks,
        ]
        for item in all_items:
            cid = _canonical_id_key(item.id)
            fp = _item_clinical_fingerprint(item)
            if cid in seen_by_id:
                if seen_by_id[cid] != fp:
                    raise ValueError(
                        f"Duplicate item ID '{item.id}' (canonical '{cid}') with different clinical content is prohibited."
                    )
            else:
                seen_by_id[cid] = fp
        return self


class ValidationReport(StrictSchemaModel):
    """Result returned by the Safety Validation Layer after auditing a payload."""

    case_id: str = Field(..., min_length=1, max_length=200)
    overall_safety_status: SafetyStatus
    requires_human_review: bool
    patient_releasable: bool = Field(
        default=False,
        description="True ONLY when the entire validated payload is safe and all items are approved.",
    )
    payload_sha256: str = Field(
        default="",
        max_length=200,
        description="Cryptographic SHA-256 digest of validated_payload produced by validate().",
    )
    total_flags: int = Field(..., ge=0)
    hazard_codes_detected: list[HazardCode] = Field(
        default_factory=list, max_length=200
    )
    validated_payload: DischargeExtractionPayload
