"""Phase 6.6 Notification Eligibility Engine.

Evaluates whether a notification can be safely scheduled from a reminder and task.
Strictly enforces: Notification orchestration can NEVER upgrade safety.
"""

from __future__ import annotations

import datetime
from typing import Optional

from backend.task_orchestration.types import (
    OrchestratedTask,
    TaskCategory,
    TaskStatus,
    TimeframeType,
)
from backend.care_reminders.types import Reminder, ReminderStatus
from backend.notifications.types import NotificationType


class NotificationEligibilityResult:
    """Result of assessing whether a notification can be created and scheduled."""

    def __init__(
        self,
        is_eligible: bool,
        notification_type: Optional[NotificationType] = None,
        scheduled_for: Optional[str] = None,
        rejection_reasons: Optional[list[str]] = None,
    ):
        self.is_eligible = is_eligible
        self.notification_type = notification_type
        self.scheduled_for = scheduled_for
        self.rejection_reasons = rejection_reasons or []


def check_notification_eligibility(
    reminder: Reminder,
    task: OrchestratedTask,
) -> NotificationEligibilityResult:
    """Assess whether a reminder and task are eligible for notification creation.

    Invariants:
    1. If task is not patient-releasable -> Ineligible.
    2. If reminder is not patient-releasable -> Ineligible.
    3. If reminder is cancelled -> Ineligible.
    4. If task is NEEDS_REVIEW or BLOCKED -> Ineligible.
    5. If task timeframe is missing, ambiguous, or relative without reference -> Ineligible.
    6. Relative operator 'after' is rejected (open-ended).
    7. Timing must be a valid, trusted ISO string.
    """
    reasons: list[str] = []

    # 1. Releasability check
    if not task.patient_releasable:
        reasons.append(f"Source task {task.task_id} is not patient-releasable.")

    if not reminder.patient_releasable:
        reasons.append(f"Source reminder {reminder.reminder_id} is not patient-releasable.")

    if reminder.status == ReminderStatus.CANCELLED:
        reasons.append(f"Source reminder {reminder.reminder_id} is CANCELLED.")

    # 2. Task status check
    if task.status in (TaskStatus.NEEDS_REVIEW, TaskStatus.BLOCKED):
        reasons.append(f"Source task {task.task_id} has review/blocked status '{task.status.value}'.")

    if task.requires_human_review:
        reasons.append(f"Source task {task.task_id} requires human review.")

    # 3. Timeframe check
    tf = task.timeframe
    if tf.is_missing or tf.timeframe_type == TimeframeType.UNSPECIFIED:
        reasons.append("Task timeframe is missing or unspecified.")

    if tf.relative_operator and tf.relative_operator.lower() == "after":
        reasons.append("Relative operator 'after' defines an open-ended bound and cannot be scheduled.")

    if tf.timeframe_type == TimeframeType.RELATIVE_WINDOW and not getattr(reminder, "derived_from_reference_date", False):
        # A relative window reminder requires a trusted reference date to ground it
        if "relative" in tf.raw_text.lower() or "within" in tf.raw_text.lower():
            if not getattr(reminder, "derived_from_reference_date", False):
                reasons.append("Relative timeframe reminder lacks trusted reference date grounding.")

    # 4. Scheduled time validation
    if not reminder.scheduled_for:
        reasons.append("Reminder has no scheduled_for timestamp.")
    else:
        try:
            datetime.datetime.fromisoformat(reminder.scheduled_for)
        except (ValueError, TypeError):
            reasons.append(f"Reminder scheduled_for '{reminder.scheduled_for}' is not a valid ISO datetime.")

    # 5. Determine NotificationType
    notif_type = NotificationType.CARE_TASK_REMINDER
    if task.task_category == TaskCategory.FOLLOW_UP:
        notif_type = NotificationType.FOLLOW_UP_REMINDER
    elif task.task_category == TaskCategory.TEST:
        notif_type = NotificationType.TEST_REMINDER
    elif task.task_category == TaskCategory.MEDICATION:
        # Carebridge policy: medications are recurring regimens, not point-in-time notification events
        reasons.append("Medication regimens do not map to point-in-time notification events.")

    if reasons:
        return NotificationEligibilityResult(
            is_eligible=False,
            notification_type=None,
            scheduled_for=None,
            rejection_reasons=reasons,
        )

    return NotificationEligibilityResult(
        is_eligible=True,
        notification_type=notif_type,
        scheduled_for=reminder.scheduled_for,
        rejection_reasons=[],
    )
