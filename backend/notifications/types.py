"""Phase 6.6 Notification & Alert Orchestration Data Models.

Strict Pydantic v2 schemas for notification types, statuses, priorities,
channels, and entities adhering to CAREBRIDGE safety invariants.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field, model_validator

from backend.document_intelligence.types import EvidenceSpan, RawPreservingSchemaModel

# ── Supported Patient Communication Languages ─────────────────────────────────
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
}


class NotificationType(str, Enum):
    """Categorization of notifications originating from validated care events."""

    APPOINTMENT_REMINDER = "APPOINTMENT_REMINDER"
    FOLLOW_UP_REMINDER = "FOLLOW_UP_REMINDER"
    TEST_REMINDER = "TEST_REMINDER"
    CARE_TASK_REMINDER = "CARE_TASK_REMINDER"
    MEDICATION_REMINDER = "MEDICATION_REMINDER"
    WARNING_SIGN_ALERT = "WARNING_SIGN_ALERT"
    TASK_REMINDER = "TASK_REMINDER"
    HUMAN_REVIEW_NOTIFICATION = "HUMAN_REVIEW_NOTIFICATION"
    SAFETY_ESCALATION = "SAFETY_ESCALATION"
    PROVIDER_FOLLOW_UP = "PROVIDER_FOLLOW_UP"
    GENERAL_CARE_INSTRUCTION = "GENERAL_CARE_INSTRUCTION"
    SYSTEM_STATUS = "SYSTEM_STATUS"


class NotificationChannel(str, Enum):
    """Delivery channels for patient communications."""

    IN_APP = "IN_APP"
    PUSH = "PUSH"
    SMS = "SMS"


class TranslationStatus(str, Enum):
    """Deterministic validation status of multilingual patient-facing communications."""

    VERIFIED = "VERIFIED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class NotificationStatus(str, Enum):
    """Lifecycle state of a patient notification.

    NOTE: No MISSED_APPOINTMENT state exists by design.
    Absence of patient response does NOT prove that an appointment was missed.
    """

    SCHEDULED = "SCHEDULED"
    TRIGGERED = "TRIGGERED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    DISMISSED = "DISMISSED"
    CANCELLED = "CANCELLED"


class NotificationPriority(str, Enum):
    """Operational priority levels.

    NOTE: Operational only — never represents clinical severity or emergency urgency.
    """

    NORMAL = "NORMAL"
    IMPORTANT = "IMPORTANT"


class NotificationAction(str, Enum):
    """User operational actions on a notification."""

    VIEW = "VIEW"
    ACKNOWLEDGE = "ACKNOWLEDGE"
    DISMISS = "DISMISS"
    CANCEL = "CANCEL"
    SNOOZE = "SNOOZE"


class Notification(RawPreservingSchemaModel):
    """A patient-facing notification grounded in a validated care reminder."""

    notification_id: str
    id: Optional[str] = None
    patient_id: str = "SYNTH-PT-801"
    reminder_id: str
    task_id: str
    source_task_id: Optional[str] = None
    notification_type: NotificationType
    channel: NotificationChannel = NotificationChannel.IN_APP
    title: str = Field(..., max_length=160)
    body: str = Field(..., max_length=600)
    status: NotificationStatus = NotificationStatus.SCHEDULED
    priority: NotificationPriority = NotificationPriority.NORMAL
    scheduled_for: str  # ISO string of notification trigger datetime
    scheduled_at: Optional[str] = None
    triggered_at: Optional[str] = None
    acknowledged_at: Optional[str] = None
    dismissed_at: Optional[str] = None
    cancelled_at: Optional[str] = None
    unread: bool = True
    patient_releasable: bool = False
    source_evidence: EvidenceSpan
    provenance_valid: bool = False
    created_from_validated_reminder: bool = False
    language: str = "en"
    translation_status: str = "VERIFIED"
    translation_verification_notes: list[str] = Field(default_factory=list)
    timezone: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def sync_aliases(self) -> Notification:
        if not self.id:
            self.id = self.notification_id
        if not self.scheduled_at:
            self.scheduled_at = self.scheduled_for
        if not self.source_task_id:
            self.source_task_id = self.task_id
        return self

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        d = super().model_dump(*args, **kwargs)
        d["id"] = self.id or self.notification_id
        d["scheduled_at"] = self.scheduled_at or self.scheduled_for
        d["source_task_id"] = self.source_task_id or self.task_id
        return d


class NotificationSummary(RawPreservingSchemaModel):
    """Lightweight contract for Phase 8 Cinematic UI consumption."""

    notification_id: str
    id: Optional[str] = None
    patient_id: str = "SYNTH-PT-801"
    title: str
    body: str
    type: NotificationType
    channel: NotificationChannel = NotificationChannel.IN_APP
    status: NotificationStatus
    priority: NotificationPriority
    scheduled_for: str
    scheduled_at: Optional[str] = None
    triggered_at: Optional[str] = None
    unread: bool = True
    patient_releasable: bool = False
    task_id: str
    source_task_id: Optional[str] = None
    reminder_id: str
    language: str = "en"
    translation_status: str = "VERIFIED"
    source_label: str = "Discharge Plan"

    @model_validator(mode="after")
    def sync_aliases(self) -> NotificationSummary:
        if not self.id:
            self.id = self.notification_id
        if not self.scheduled_at:
            self.scheduled_at = self.scheduled_for
        if not self.source_task_id:
            self.source_task_id = self.task_id
        return self

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        d = super().model_dump(*args, **kwargs)
        d["id"] = self.id or self.notification_id
        d["scheduled_at"] = self.scheduled_at or self.scheduled_for
        d["source_task_id"] = self.source_task_id or self.task_id
        return d


class NotificationCenterState(RawPreservingSchemaModel):
    """Aggregated notification center state for patient view."""

    total_count: int = 0
    unread_count: int = 0
    notifications: list[NotificationSummary] = Field(default_factory=list)
