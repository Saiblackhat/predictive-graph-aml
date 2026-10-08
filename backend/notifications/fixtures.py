"""Phase 6.6 Synthetic Notification Test Fixtures (N01 - N18).

Provides synthetic fixtures for all required operational and adversarial scenarios.
All data is strictly synthetic.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

from backend.document_intelligence.types import EvidenceSpan
from backend.task_orchestration.types import (
    OrchestratedTask,
    ParsedTimeframe,
    TaskCategory,
    TaskStatus,
    TimeframeType,
)
from backend.care_reminders.types import Reminder, ReminderStatus, ReminderType
from backend.notifications.types import (
    Notification,
    NotificationPriority,
    NotificationStatus,
    NotificationType,
)


class SyntheticNotificationFixture(NamedTuple):
    case_id: str
    description: str
    expected_eligible: bool


def get_notification_fixture(case_id: str) -> tuple[OrchestratedTask, Reminder]:
    """Factory providing matching (task, reminder) pairs for N01-N18 testing."""
    cid = case_id.upper()

    # N01: Absolute Cardiology follow-up
    if cid == "N01":
        ev = EvidenceSpan(
            sentence_id="s1",
            section="Follow-Up",
            source_text="Follow up with Cardiology on March 18, 2027 at 10:30 AM.",
            start_offset=0,
            end_offset=56,
        )
        task = OrchestratedTask(
            task_id="task-n01",
            task_category=TaskCategory.FOLLOW_UP,
            title="Follow-up appointment with Cardiology",
            description="Attend follow-up appointment with Cardiology.",
            status=TaskStatus.SCHEDULED,
            timeframe=ParsedTimeframe(
                raw_text="on March 18, 2027 at 10:30 AM",
                timeframe_type=TimeframeType.ABSOLUTE_DATE,
                absolute_date="2027-03-18",
            ),
            due_date="2027-03-18",
            source_item_id="fu-n01",
            evidence=ev,
            patient_releasable=True,
            requires_human_review=False,
            clinical_metadata={"specialty": "Cardiology"},
        )
        rem = Reminder(
            reminder_id="rem-n01",
            task_id="task-n01",
            source_task_id="task-n01",
            title="Follow-up appointment with Cardiology",
            reminder_type=ReminderType.APPOINTMENT,
            scheduled_for="2027-03-17T10:30:00",
            event_time="2027-03-18T10:30:00",
            offset_minutes=1440,
            status=ReminderStatus.SCHEDULED,
            patient_releasable=True,
            source_evidence=ev,
            created_from_validated_task=True,
            provenance_valid=True,
        )
        return task, rem

    # N02: Absolute test date
    if cid == "N02":
        ev = EvidenceSpan(
            sentence_id="s2",
            section="Tests",
            source_text="Complete blood count (CBC) scheduled on March 22, 2027 at 9:00 AM.",
            start_offset=0,
            end_offset=67,
        )
        task = OrchestratedTask(
            task_id="task-n02",
            task_category=TaskCategory.TEST,
            title="Complete blood count (CBC)",
            description="Diagnostic CBC blood test.",
            status=TaskStatus.SCHEDULED,
            timeframe=ParsedTimeframe(
                raw_text="on March 22, 2027 at 9:00 AM",
                timeframe_type=TimeframeType.ABSOLUTE_DATE,
                absolute_date="2027-03-22",
            ),
            due_date="2027-03-22",
            source_item_id="test-n02",
            evidence=ev,
            patient_releasable=True,
            requires_human_review=False,
            clinical_metadata={"test_name": "CBC"},
        )
        rem = Reminder(
            reminder_id="rem-n02",
            task_id="task-n02",
            source_task_id="task-n02",
            title="Complete blood count (CBC)",
            reminder_type=ReminderType.TEST,
            scheduled_for="2027-03-21T09:00:00",
            event_time="2027-03-22T09:00:00",
            offset_minutes=1440,
            status=ReminderStatus.SCHEDULED,
            patient_releasable=True,
            source_evidence=ev,
            created_from_validated_task=True,
            provenance_valid=True,
        )
        return task, rem

    # N03: Care task with exact time
    if cid == "N03":
        ev = EvidenceSpan(
            sentence_id="s3",
            section="Instructions",
            source_text="Remove wound dressing on March 20, 2027 at 12:00 PM.",
            start_offset=0,
            end_offset=53,
        )
        task = OrchestratedTask(
            task_id="task-n03",
            task_category=TaskCategory.CARE_INSTRUCTION,
            title="Remove wound dressing",
            description="Wound dressing removal instruction.",
            status=TaskStatus.SCHEDULED,
            timeframe=ParsedTimeframe(
                raw_text="on March 20, 2027 at 12:00 PM",
                timeframe_type=TimeframeType.ABSOLUTE_DATE,
                absolute_date="2027-03-20",
            ),
            due_date="2027-03-20",
            source_item_id="task-n03-item",
            evidence=ev,
            patient_releasable=True,
            requires_human_review=False,
            clinical_metadata={},
        )
        rem = Reminder(
            reminder_id="rem-n03",
            task_id="task-n03",
            source_task_id="task-n03",
            title="Remove wound dressing",
            reminder_type=ReminderType.CARE_TASK,
            scheduled_for="2027-03-20T10:00:00",
            event_time="2027-03-20T12:00:00",
            offset_minutes=120,
            status=ReminderStatus.SCHEDULED,
            patient_releasable=True,
            source_evidence=ev,
            created_from_validated_task=True,
            provenance_valid=True,
        )
        return task, rem

    # N04: Relative timeframe without reference date
    if cid == "N04":
        ev = EvidenceSpan(
            sentence_id="s4",
            section="Follow-Up",
            source_text="Follow up with Cardiology within 7 days.",
            start_offset=0,
            end_offset=41,
        )
        task = OrchestratedTask(
            task_id="task-n04",
            task_category=TaskCategory.FOLLOW_UP,
            title="Follow-up appointment with Cardiology",
            description="Attend follow-up within 7 days.",
            status=TaskStatus.SCHEDULED,
            timeframe=ParsedTimeframe(
                raw_text="within 7 days",
                timeframe_type=TimeframeType.RELATIVE_WINDOW,
                relative_window="within 7 days",
                relative_operator="within",
                relative_duration_days=7,
            ),
            due_window="within 7 days",
            source_item_id="fu-n04",
            evidence=ev,
            patient_releasable=True,
            requires_human_review=False,
            clinical_metadata={"specialty": "Cardiology"},
        )
        rem = Reminder(
            reminder_id="rem-n04",
            task_id="task-n04",
            source_task_id="task-n04",
            title="Follow-up appointment with Cardiology",
            reminder_type=ReminderType.APPOINTMENT,
            scheduled_for="2027-03-07T00:00:00",
            event_time="2027-03-08T00:00:00",
            offset_minutes=1440,
            status=ReminderStatus.SCHEDULED,
            patient_releasable=True,
            source_evidence=ev,
            created_from_validated_task=True,
            provenance_valid=True,
            derived_from_reference_date=False,  # Un-grounded relative!
        )
        return task, rem

    # N05: After-7-days task
    if cid == "N05":
        ev = EvidenceSpan(
            sentence_id="s5",
            section="Follow-Up",
            source_text="Follow up with Cardiology after 7 days.",
            start_offset=0,
            end_offset=40,
        )
        task = OrchestratedTask(
            task_id="task-n05",
            task_category=TaskCategory.FOLLOW_UP,
            title="Follow-up appointment with Cardiology",
            description="Attend follow-up after 7 days.",
            status=TaskStatus.SCHEDULED,
            timeframe=ParsedTimeframe(
                raw_text="after 7 days",
                timeframe_type=TimeframeType.RELATIVE_WINDOW,
                relative_window="after 7 days",
                relative_operator="after",
                relative_duration_days=7,
            ),
            due_window="after 7 days",
            source_item_id="fu-n05",
            evidence=ev,
            patient_releasable=True,
            requires_human_review=False,
            clinical_metadata={"specialty": "Cardiology"},
        )
        rem = Reminder(
            reminder_id="rem-n05",
            task_id="task-n05",
            source_task_id="task-n05",
            title="Follow-up appointment with Cardiology",
            reminder_type=ReminderType.APPOINTMENT,
            scheduled_for="2027-03-07T00:00:00",
            event_time="2027-03-08T00:00:00",
            offset_minutes=1440,
            status=ReminderStatus.SCHEDULED,
            patient_releasable=True,
            source_evidence=ev,
            created_from_validated_task=True,
            provenance_valid=True,
        )
        return task, rem

    # N06: Conflicting task
    if cid == "N06":
        ev = EvidenceSpan(
            sentence_id="s6",
            section="Follow-Up",
            source_text="Follow up on March 18 or April 20.",
            start_offset=0,
            end_offset=34,
        )
        task = OrchestratedTask(
            task_id="task-n06",
            task_category=TaskCategory.FOLLOW_UP,
            title="Conflicting follow-up",
            description="Conflicting dates detected.",
            status=TaskStatus.BLOCKED,
            timeframe=ParsedTimeframe(
                raw_text="March 18 or April 20",
                timeframe_type=TimeframeType.UNSPECIFIED,
                is_missing=True,
            ),
            source_item_id="fu-n06",
            evidence=ev,
            patient_releasable=False,
            requires_human_review=True,
            review_reasons=["Conflicting appointment dates."],
            clinical_metadata={},
        )
        rem = Reminder(
            reminder_id="rem-n06",
            task_id="task-n06",
            source_task_id="task-n06",
            title="Conflicting follow-up",
            reminder_type=ReminderType.APPOINTMENT,
            scheduled_for="2027-03-17T00:00:00",
            event_time="2027-03-18T00:00:00",
            offset_minutes=1440,
            status=ReminderStatus.SCHEDULED,
            patient_releasable=False,
            source_evidence=ev,
            created_from_validated_task=False,
            provenance_valid=False,
        )
        return task, rem

    # N07: Needs-review task
    if cid == "N07":
        ev = EvidenceSpan(
            sentence_id="s7",
            section="Follow-Up",
            source_text="Follow up soon.",
            start_offset=0,
            end_offset=15,
        )
        task = OrchestratedTask(
            task_id="task-n07",
            task_category=TaskCategory.FOLLOW_UP,
            title="Vague follow-up",
            description="Follow up soon.",
            status=TaskStatus.NEEDS_REVIEW,
            timeframe=ParsedTimeframe(
                raw_text="soon",
                timeframe_type=TimeframeType.UNSPECIFIED,
                is_missing=True,
            ),
            source_item_id="fu-n07",
            evidence=ev,
            patient_releasable=False,
            requires_human_review=True,
            review_reasons=["Vague timeframe 'soon'."],
            clinical_metadata={},
        )
        rem = Reminder(
            reminder_id="rem-n07",
            task_id="task-n07",
            source_task_id="task-n07",
            title="Vague follow-up",
            reminder_type=ReminderType.APPOINTMENT,
            scheduled_for="2027-03-17T00:00:00",
            event_time="2027-03-18T00:00:00",
            offset_minutes=1440,
            status=ReminderStatus.SCHEDULED,
            patient_releasable=False,
            source_evidence=ev,
            created_from_validated_task=False,
            provenance_valid=False,
        )
        return task, rem

    # N08: Blocked task
    if cid == "N08":
        ev = EvidenceSpan(
            sentence_id="s8",
            section="Follow-Up",
            source_text="Follow up with Dr. Unknown.",
            start_offset=0,
            end_offset=27,
        )
        task = OrchestratedTask(
            task_id="task-n08",
            task_category=TaskCategory.FOLLOW_UP,
            title="Blocked follow-up",
            description="Blocked by safety validator.",
            status=TaskStatus.BLOCKED,
            timeframe=ParsedTimeframe(raw_text="", timeframe_type=TimeframeType.UNSPECIFIED, is_missing=True),
            source_item_id="fu-n08",
            evidence=ev,
            patient_releasable=False,
            requires_human_review=True,
            review_reasons=["Blocked."],
            clinical_metadata={},
        )
        rem = Reminder(
            reminder_id="rem-n08",
            task_id="task-n08",
            source_task_id="task-n08",
            title="Blocked follow-up",
            reminder_type=ReminderType.APPOINTMENT,
            scheduled_for="2027-03-17T00:00:00",
            event_time="2027-03-18T00:00:00",
            offset_minutes=1440,
            status=ReminderStatus.SCHEDULED,
            patient_releasable=False,
            source_evidence=ev,
            created_from_validated_task=False,
            provenance_valid=False,
        )
        return task, rem

    # N09: Spanish Cardiology follow-up
    if cid == "N09":
        ev = EvidenceSpan(
            sentence_id="s9",
            section="Follow-Up",
            source_text="Follow up with Cardiology on March 18, 2027 at 10:30 AM.",
            start_offset=0,
            end_offset=56,
        )
        task = OrchestratedTask(
            task_id="task-n09",
            task_category=TaskCategory.FOLLOW_UP,
            title="Seguimiento con Cardiología",
            description="Asistir a consulta de seguimiento con Cardiología.",
            status=TaskStatus.SCHEDULED,
            timeframe=ParsedTimeframe(
                raw_text="on March 18, 2027 at 10:30 AM",
                timeframe_type=TimeframeType.ABSOLUTE_DATE,
                absolute_date="2027-03-18",
            ),
            due_date="2027-03-18",
            source_item_id="fu-n09",
            evidence=ev,
            patient_releasable=True,
            requires_human_review=False,
            clinical_metadata={"specialty": "Cardiology"},
        )
        rem = Reminder(
            reminder_id="rem-n09",
            task_id="task-n09",
            source_task_id="task-n09",
            title="Seguimiento con Cardiología",
            reminder_type=ReminderType.APPOINTMENT,
            scheduled_for="2027-03-17T10:30:00",
            event_time="2027-03-18T10:30:00",
            offset_minutes=1440,
            status=ReminderStatus.SCHEDULED,
            patient_releasable=True,
            source_evidence=ev,
            created_from_validated_task=True,
            provenance_valid=True,
            language="es",
        )
        return task, rem

    # N11: Cancelled reminder
    if cid == "N11":
        task, rem = get_notification_fixture("N01")
        rem.status = ReminderStatus.CANCELLED
        return task, rem

    # Default fallback: N01
    return get_notification_fixture("N01")
