"""Data contracts and provenance models for Phase 4 Document Intelligence.

All models enforce strict schema validation and preserve exact character-span
and sentence/section provenance without fabricating page numbers.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from backend.schemas.models import (
    DischargeExtractionPayload,
    MedicationAction,
    SourceEvidence,
    StrictSchemaModel,
    UrgencyLevel,
    ValidationReport,
)


class RawPreservingSchemaModel(StrictSchemaModel):
    """Strict schema model that preserves exact leading/trailing whitespace on raw text fields."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
        str_strip_whitespace=False,
    )


class DocumentFormat(str, Enum):
    """Supported or planned input document formats for ingestion."""

    PLAIN_TEXT = "PLAIN_TEXT"
    SYNTHETIC_SUMMARY = "SYNTHETIC_SUMMARY"
    PDF = "PDF"
    DOCX = "DOCX"


class SectionCategory(str, Enum):
    """Deterministic canonical categories for hospital discharge summary sections."""

    DISCHARGE_MEDICATIONS = "DISCHARGE_MEDICATIONS"
    MEDICATIONS = "MEDICATIONS"
    FOLLOW_UP = "FOLLOW_UP"
    APPOINTMENTS = "APPOINTMENTS"
    TESTS = "TESTS"
    REFERRALS = "REFERRALS"
    WOUND_CARE = "WOUND_CARE"
    ACTIVITY = "ACTIVITY"
    DIET = "DIET"
    WARNING_SIGNS = "WARNING_SIGNS"
    WHEN_TO_SEEK_HELP = "WHEN_TO_SEEK_HELP"
    INSTRUCTIONS = "INSTRUCTIONS"
    HOSPITAL_COURSE = "HOSPITAL_COURSE"
    HEADER_METADATA = "HEADER_METADATA"
    UNKNOWN = "UNKNOWN"


CareInstructionCategory = Literal[
    "WOUND_CARE",
    "DIET",
    "ACTIVITY",
    "REHABILITATION",
    "OTHER_CARE_INSTRUCTION",
]


class SuspiciousCharFinding(StrictSchemaModel):
    """Record of a zero-width, soft-hyphen, or bidirectional control character in raw text."""

    codepoint: str = Field(..., min_length=4, max_length=20)
    character_name: str = Field(..., min_length=1, max_length=100)
    raw_offset: int = Field(..., ge=0)
    context_snippet: str = Field(..., min_length=1, max_length=200)


class NormalizedDocumentText(RawPreservingSchemaModel):
    """Normalized discharge text with bidirectional character-offset mapping to raw_text."""

    raw_text: str = Field(..., min_length=1)
    normalized_text: str = Field(..., min_length=1)
    norm_to_raw_start: list[int] = Field(default_factory=list)
    norm_to_raw_end: list[int] = Field(default_factory=list)
    suspicious_chars: list[SuspiciousCharFinding] = Field(default_factory=list)
    hyphenations_repaired: int = Field(default=0, ge=0)

    def map_norm_span_to_raw(self, norm_start: int, norm_end: int) -> tuple[int, int, str]:
        """Map a [norm_start, norm_end) slice in normalized_text back to raw_text offsets and substring."""
        if not self.norm_to_raw_start or norm_start >= norm_end:
            return (0, 0, "")
        clamped_start = max(0, min(norm_start, len(self.norm_to_raw_start) - 1))
        clamped_end = max(clamped_start + 1, min(norm_end, len(self.norm_to_raw_end)))
        raw_start = self.norm_to_raw_start[clamped_start]
        raw_end = self.norm_to_raw_end[clamped_end - 1]
        return (raw_start, raw_end, self.raw_text[raw_start:raw_end])


class DetectedSection(StrictSchemaModel):
    """A detected section within a discharge document, preserving unknown headers."""

    section_id: str = Field(..., min_length=1, max_length=100)
    canonical_category: SectionCategory
    raw_header: Optional[str] = Field(default=None, max_length=300)
    section_name: str = Field(..., min_length=1, max_length=300)
    raw_start_char: int = Field(..., ge=0)
    raw_end_char: int = Field(..., ge=0)
    norm_start_char: int = Field(..., ge=0)
    norm_end_char: int = Field(..., ge=0)
    content_norm_start_char: int = Field(..., ge=0)
    content_norm_end_char: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _validate_offsets(self) -> DetectedSection:
        if self.raw_end_char < self.raw_start_char:
            raise ValueError("raw_end_char must be >= raw_start_char")
        if self.norm_end_char < self.norm_start_char:
            raise ValueError("norm_end_char must be >= norm_start_char")
        return self


class SegmentedSentence(StrictSchemaModel):
    """A single segmented sentence anchored to its section and exact raw character offsets."""

    sentence_id: str = Field(..., min_length=1, max_length=100)
    section_id: str = Field(..., min_length=1, max_length=100)
    section: str = Field(..., min_length=1, max_length=300)
    canonical_section: SectionCategory
    source_text: str = Field(..., min_length=1, max_length=2000)
    normalized_text: str = Field(..., min_length=1, max_length=2000)
    start_offset: int = Field(..., ge=0)
    end_offset: int = Field(..., ge=0)
    norm_start_offset: int = Field(..., ge=0)
    norm_end_offset: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _validate_sentence_span(self) -> SegmentedSentence:
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must be >= start_offset")
        return self


class EvidenceSpan(StrictSchemaModel):
    """Provenance anchor pointing to a verbatim sentence/character span in raw_text."""

    sentence_id: str = Field(..., min_length=1, max_length=100)
    section: Optional[str] = Field(default=None, max_length=200)
    source_text: str = Field(..., min_length=1, max_length=1000)
    start_offset: int = Field(..., ge=0)
    end_offset: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _validate_span(self) -> EvidenceSpan:
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must be >= start_offset")
        return self

    def to_source_evidence(self) -> SourceEvidence:
        """Convert this EvidenceSpan into a validator-compatible SourceEvidence without fabricating page numbers."""
        return SourceEvidence(
            source_text=self.source_text,
            source_section=self.section,
            source_page=None,
            start_char=self.start_offset,
            end_char=self.end_offset,
            span_status_note=(
                f"Verified character span [{self.start_offset}:{self.end_offset}] "
                f"in sentence {self.sentence_id} (no page metadata)."
            ),
            source_page_verified=False,
        )


class ExtractedMedicationRecord(StrictSchemaModel):
    """Phase 4 extracted medication instruction with full provenance."""

    id: str = Field(..., min_length=1, max_length=200)
    medication_name: str = Field(..., min_length=1, max_length=200)
    raw_medication_display: str = Field(..., min_length=1, max_length=200)
    action: MedicationAction = Field(default=MedicationAction.UNKNOWN)
    dosage: Optional[str] = Field(default=None, max_length=200)
    previous_dosage: Optional[str] = Field(default=None, max_length=200)
    frequency: Optional[str] = Field(default=None, max_length=200)
    route: Optional[str] = Field(default=None, max_length=200)
    timeframe_or_duration: Optional[str] = Field(default=None, max_length=200)
    indication_or_note: Optional[str] = Field(default=None, max_length=1000)
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_ambiguous: bool = Field(default=False)
    ambiguity_note: Optional[str] = Field(default=None, max_length=500)
    evidence: EvidenceSpan


class ExtractedFollowUpRecord(StrictSchemaModel):
    """Phase 4 extracted follow-up appointment instruction with full provenance."""

    id: str = Field(..., min_length=1, max_length=200)
    specialty: Optional[str] = Field(default=None, max_length=200)
    provider_name: Optional[str] = Field(default=None, max_length=200)
    timeframe: Optional[str] = Field(default=None, max_length=200)
    reason: Optional[str] = Field(default=None, max_length=1000)
    appointment_scheduled: Optional[bool] = Field(default=None)
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_ambiguous: bool = Field(default=False)
    ambiguity_note: Optional[str] = Field(default=None, max_length=500)
    evidence: EvidenceSpan


class ExtractedTestRecord(StrictSchemaModel):
    """Phase 4 extracted outpatient laboratory or diagnostic test instruction."""

    id: str = Field(..., min_length=1, max_length=200)
    test_name: str = Field(..., min_length=1, max_length=300)
    timeframe: Optional[str] = Field(default=None, max_length=200)
    reason_or_note: Optional[str] = Field(default=None, max_length=500)
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_ambiguous: bool = Field(default=False)
    evidence: EvidenceSpan


class ExtractedReferralRecord(StrictSchemaModel):
    """Phase 4 extracted specialty/service referral instruction."""

    id: str = Field(..., min_length=1, max_length=200)
    specialty_or_service: Optional[str] = Field(default=None, max_length=200)
    provider_name: Optional[str] = Field(default=None, max_length=200)
    timeframe: Optional[str] = Field(default=None, max_length=200)
    reason: Optional[str] = Field(default=None, max_length=500)
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_ambiguous: bool = Field(default=False)
    evidence: EvidenceSpan


class ExtractedWarningSignRecord(StrictSchemaModel):
    """Phase 4 extracted return precaution or emergency warning sign."""

    id: str = Field(..., min_length=1, max_length=200)
    symptom_description: str = Field(..., min_length=1, max_length=1000)
    action_to_take: Optional[str] = Field(default=None, max_length=1000)
    urgency: UrgencyLevel = Field(default=UrgencyLevel.UNKNOWN)
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_ambiguous: bool = Field(default=False)
    evidence: EvidenceSpan


class ExtractedCareInstructionRecord(StrictSchemaModel):
    """Phase 4 extracted care instruction (wound care, diet, activity, rehabilitation, other)."""

    id: str = Field(..., min_length=1, max_length=200)
    instruction_type: CareInstructionCategory
    title: str = Field(..., min_length=1, max_length=1000)
    detail: str = Field(..., min_length=1, max_length=1000)
    timeframe: Optional[str] = Field(default=None, max_length=200)
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_ambiguous: bool = Field(default=False)
    evidence: EvidenceSpan


class IngestedDocument(RawPreservingSchemaModel):
    """Output of the Document Ingestion + Normalization + Section + Sentence pipeline."""

    document_id: str = Field(..., min_length=1, max_length=200)
    patient_identifier: str = Field(..., min_length=1, max_length=200)
    source_format: DocumentFormat = Field(default=DocumentFormat.SYNTHETIC_SUMMARY)
    raw_text: str = Field(..., min_length=1)
    normalized_text: str = Field(..., min_length=1)
    sections: list[DetectedSection] = Field(default_factory=list)
    sentences: list[SegmentedSentence] = Field(default_factory=list)
    suspicious_chars: list[SuspiciousCharFinding] = Field(default_factory=list)
    hyphenations_repaired: int = Field(default=0, ge=0)


class DocumentIntelligenceResult(StrictSchemaModel):
    """Complete Care-Plan-Ready output combining provenance-rich extraction and safety validation."""

    document: IngestedDocument
    medications: list[ExtractedMedicationRecord] = Field(default_factory=list)
    follow_ups: list[ExtractedFollowUpRecord] = Field(default_factory=list)
    tests: list[ExtractedTestRecord] = Field(default_factory=list)
    referrals: list[ExtractedReferralRecord] = Field(default_factory=list)
    warning_signs: list[ExtractedWarningSignRecord] = Field(default_factory=list)
    care_instructions: list[ExtractedCareInstructionRecord] = Field(default_factory=list)
    extraction_completeness_score: float = Field(..., ge=0.0, le=1.0)
    unresolved_or_ambiguous_items: list[str] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)
    validator_payload: DischargeExtractionPayload
    validation_report: ValidationReport
