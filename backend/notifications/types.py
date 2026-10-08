"""Phase 6.6 Notification & Alert Orchestration Data Models.

Strict Pydantic v2 schemas for notification types, statuses, priorities,
and entities adhering to CAREBRIDGE safety invariants.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field

from backend.document_intelligence.types import EvidenceSpan, RawPreservingSchemaModel


class NotificationType(str, Enum):
    """Categorization of notifications originating from validated care events."""

    APPOINTMENT_REMINDER = "APPOINTMENT_REMINDER"
    FOLLOW_UP_REMINDER = "FOLLOW_UP_REMINDER"
    TEST_REMINDER = "TEST_REMINDER"
    CARE_TASK_REMINDER = "CARE_TASK_REMINDER"
    SYSTEM_STATUS = "SYSTEM_STATUS"


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
    reminder_id: str
    task_id: str
    notification_type: NotificationType
    title: str = Field(..., max_length=120)
    body: str = Field(..., max_length=500)
    status: NotificationStatus = NotificationStatus.SCHEDULED
    priority: NotificationPriority = NotificationPriority.NORMAL
    scheduled_for: str  # ISO string of notification trigger datetime
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
    timezone: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NotificationSummary(RawPreservingSchemaModel):
    """Lightweight contract for Phase 8 Cinematic UI consumption."""

    notification_id: str
    title: str
    body: str
    type: NotificationType
    status: NotificationStatus
    priority: NotificationPriority
    scheduled_for: str
    triggered_at: Optional[str] = None
    unread: bool = True
    patient_releasable: bool = False
    task_id: str
    reminder_id: str
    language: str = "en"
    source_label: str = "Discharge Plan"


class NotificationCenterState(RawPreservingSchemaModel):
    """Aggregated notification center state for patient view."""

    total_count: int = 0
    unread_count: int = 0
    notifications: list[NotificationSummary] = Field(default_factory=list)
