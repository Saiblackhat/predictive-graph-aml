"""Phase 6.5 Care Reminders Data Models.

Strict Pydantic v2 schemas for reminder types, presets, attendance status,
and reminder entities adhering to CAREBRIDGE safety invariants.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field

from backend.document_intelligence.types import EvidenceSpan, RawPreservingSchemaModel


class ReminderType(str, Enum):
    """Categorization of care reminders."""

    APPOINTMENT = "APPOINTMENT"
    TEST = "TEST"
    CARE_TASK = "CARE_TASK"


class ReminderStatus(str, Enum):
    """Lifecycle state of a care reminder."""

    SCHEDULED = "SCHEDULED"
    TRIGGERED = "TRIGGERED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    DISMISSED = "DISMISSED"
    CANCELLED = "CANCELLED"


class ReminderPreset(int, Enum):
    """Deterministic reminder timing presets in minutes."""

    ONE_WEEK = 10080   # 7 * 24 * 60
    THREE_DAYS = 4320  # 3 * 24 * 60
    ONE_DAY = 1440     # 24 * 60
    TWO_HOURS = 120    # 2 * 60


class AttendanceStatus(str, Enum):
    """Patient attendance enquiry options."""

    NO_RESPONSE = "NO_RESPONSE"
    COMPLETED = "COMPLETED"
    NOT_YET = "NOT_YET"
    NEED_HELP = "NEED_HELP"


class AttendanceEnquiry(RawPreservingSchemaModel):
    """Structured attendance follow-up dialog without clinical inference."""

    question: str = "Did you attend your appointment?"
    options: list[str] = Field(default_factory=lambda: ["Completed", "Not yet", "I need help"])
    response: Optional[AttendanceStatus] = None
    response_timestamp: Optional[str] = None
    clinical_advice_provided: bool = False
    followup_note: Optional[str] = None


class EligibilityResult(RawPreservingSchemaModel):
    """Deterministic reminder eligibility assessment for a care task."""

    is_eligible: bool
    task_id: str
    reminder_type: Optional[ReminderType] = None
    event_date: Optional[str] = None
    event_time: Optional[str] = None
    event_datetime_iso: Optional[str] = None
    has_explicit_time: bool = False
    timezone: Optional[str] = None
    timeframe_display: Optional[str] = None
    derived_from_reference_date: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)


class Reminder(RawPreservingSchemaModel):
    """A patient-controlled care reminder grounded in a validated care task."""

    reminder_id: str
    task_id: str
    source_task_id: str
    title: str
    reminder_type: ReminderType
    scheduled_for: str  # ISO string of reminder trigger datetime
    event_time: str     # ISO string of appointment / test datetime
    offset_minutes: int
    status: ReminderStatus = ReminderStatus.SCHEDULED
    patient_releasable: bool = True
    source_evidence: EvidenceSpan
    created_from_validated_task: bool = True
    provenance_valid: bool = True
    timezone: Optional[str] = None
    custom_label: Optional[str] = None
    attendance_enquiry: Optional[AttendanceEnquiry] = None
    language: str = "en"
    ui_display_text: Optional[str] = None
    derived_from_reference_date: bool = False
