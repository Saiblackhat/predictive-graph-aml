"""Phase 5 Task & Timeline Orchestration Data Models.

Strict Pydantic v2 schemas for tasks, parsed timeframes, care timeline,
and complete care journey.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field

from backend.document_intelligence.types import EvidenceSpan, RawPreservingSchemaModel
from backend.schemas.models import HazardCode, SafetyStatus


class TaskCategory(str, Enum):
    """Category of clinical care task."""

    MEDICATION = "MEDICATION"
    FOLLOW_UP = "FOLLOW_UP"
    TEST = "TEST"
    REFERRAL = "REFERRAL"
    WARNING_MONITORING = "WARNING_MONITORING"
    CARE_INSTRUCTION = "CARE_INSTRUCTION"


class TaskStatus(str, Enum):
    """Lifecycle / execution status of a task."""

    PENDING = "PENDING"
    SCHEDULED = "SCHEDULED"
    COMPLETED = "COMPLETED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    BLOCKED = "BLOCKED"


class TimeframeType(str, Enum):
    """Semantics of a clinical timeframe."""

    IMMEDIATE = "IMMEDIATE"
    RELATIVE_WINDOW = "RELATIVE_WINDOW"
    ABSOLUTE_DATE = "ABSOLUTE_DATE"
    RECURRING = "RECURRING"
    UNSPECIFIED = "UNSPECIFIED"


class ParsedTimeframe(RawPreservingSchemaModel):
    """Structured representation of an explicitly stated timeframe."""

    raw_text: Optional[str] = None
    timeframe_type: TimeframeType
    relative_window: Optional[str] = None
    relative_operator: Optional[str] = None
    relative_duration_days: Optional[int] = None
    absolute_date: Optional[str] = None
    recurring_pattern: Optional[str] = None
    is_missing: bool = False


class OrchestratedTask(RawPreservingSchemaModel):
    """A deterministic, evidence-backed care task derived from validated discharge data."""

    task_id: str
    task_category: TaskCategory
    title: str
    description: str
    status: TaskStatus
    timeframe: ParsedTimeframe
    due_date: Optional[str] = None
    due_window: Optional[str] = None
    source_item_id: str
    evidence: EvidenceSpan
    patient_releasable: bool
    requires_human_review: bool
    review_reasons: list[str] = Field(default_factory=list)
    dependency_ids: list[str] = Field(default_factory=list)
    priority: int = 10
    clinical_metadata: dict[str, Any] = Field(default_factory=dict)


class CareTimeline(RawPreservingSchemaModel):
    """Chronologically organized care tasks grouped by timeframe semantics."""

    timeline_id: str
    document_id: str
    patient_identifier: Optional[str] = None
    immediate_tasks: list[OrchestratedTask] = Field(default_factory=list)
    relative_window_tasks: list[OrchestratedTask] = Field(default_factory=list)
    absolute_date_tasks: list[OrchestratedTask] = Field(default_factory=list)
    recurring_tasks: list[OrchestratedTask] = Field(default_factory=list)
    unspecified_timeframe_tasks: list[OrchestratedTask] = Field(default_factory=list)
    review_or_blocked_tasks: list[OrchestratedTask] = Field(default_factory=list)
    all_tasks: list[OrchestratedTask] = Field(default_factory=list)
    total_task_count: int
    releasable_task_count: int
    review_required_task_count: int
    blocked_task_count: int


class CareJourney(RawPreservingSchemaModel):
    """Complete orchestrated care journey with safety and provenance audit guarantees."""

    journey_id: str
    document_id: str
    patient_identifier: Optional[str] = None
    timeline: CareTimeline
    safety_status: SafetyStatus
    patient_releasable: bool
    requires_human_review: bool
    hazard_codes: list[HazardCode] = Field(default_factory=list)
    clinical_disclaimer: str
    provenance_audit_passed: bool
