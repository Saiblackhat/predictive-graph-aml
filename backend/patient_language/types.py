"""Phase 6 Patient-Friendly Language and Multilingual Types.

Strict Pydantic v2 schemas for patient-friendly items, translated items,
and complete patient care plans.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field

from backend.document_intelligence.types import EvidenceSpan, RawPreservingSchemaModel
from backend.schemas.models import SafetyStatus
from backend.task_orchestration.types import TaskCategory


class LanguageCode(str, Enum):
    """Supported language codes."""

    EN = "en"
    ES = "es"
    HI = "hi"
    TA = "ta"
    TE = "te"
    KN = "kn"
    ML = "ml"
    BN = "bn"
    MR = "mr"
    GU = "gu"
    PA = "pa"
    OR = "or"


class TransformationStatus(str, Enum):
    """Quality and verification status of language transformation."""

    FAITHFUL = "FAITHFUL"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    BLOCKED = "BLOCKED"


class PatientLanguageItem(RawPreservingSchemaModel):
    """A simplified, patient-friendly representation of an orchestrated care task."""

    item_id: str
    source_task_id: str
    source_item_id: str
    category: TaskCategory
    simplified_title: str
    simplified_instruction: str
    language: LanguageCode = LanguageCode.EN
    transformation_status: TransformationStatus
    patient_releasable: bool
    requires_human_review: bool
    review_reasons: list[str] = Field(default_factory=list)
    evidence: EvidenceSpan
    original_source_text: str
    timeframe_display: Optional[str] = None
    preserved_fields: dict[str, Any] = Field(default_factory=dict)


class TranslatedPatientLanguageItem(RawPreservingSchemaModel):
    """A verified, controlled translation of a patient-friendly item."""

    item_id: str
    source_language_item_id: str
    source_task_id: str
    source_language: LanguageCode
    target_language: LanguageCode
    translated_title: str
    translated_instruction: str
    transformation_status: TransformationStatus
    patient_releasable: bool
    requires_human_review: bool
    review_reasons: list[str] = Field(default_factory=list)
    evidence: EvidenceSpan
    original_source_text: str
    preserved_fields: dict[str, Any] = Field(default_factory=dict)


class PatientCarePlan(RawPreservingSchemaModel):
    """Complete patient-facing care plan containing simplified and translated tasks."""

    plan_id: str
    document_id: str
    patient_identifier: Optional[str] = None
    language: LanguageCode = LanguageCode.EN
    items: list[PatientLanguageItem] = Field(default_factory=list)
    translations: list[TranslatedPatientLanguageItem] = Field(default_factory=list)
    total_items: int
    releasable_items: int
    review_required_items: int
    blocked_items: int
    overall_safety_status: SafetyStatus
    patient_releasable: bool
    requires_human_review: bool
    clinical_disclaimer: str
    fidelity_audit_passed: bool
