"""Phase 6.6 Notification Provenance Validator.

Verifies unbroken provenance chains: Notification -> Reminder -> Task -> Evidence.
Detects tampering, unauthorized overrides, and content fidelity violations.
"""

from __future__ import annotations

from typing import Optional

from backend.task_orchestration.types import OrchestratedTask
from backend.care_reminders.types import Reminder
from backend.notifications.types import Notification
from backend.notifications.formatter import audit_notification_content_fidelity


def validate_notification_provenance(
    notification: Notification,
    reminder: Reminder,
    task: OrchestratedTask,
) -> tuple[bool, list[str]]:
    """Validate unbroken provenance and integrity across the notification stack.

    Returns:
        (is_valid: bool, violations: list[str])
    """
    violations: list[str] = []

    # 1. Identifier linkage
    if notification.reminder_id != reminder.reminder_id:
        violations.append(
            f"Notification reminder_id '{notification.reminder_id}' does not match "
            f"reminder.reminder_id '{reminder.reminder_id}'."
        )

    if notification.task_id != reminder.task_id:
        violations.append(
            f"Notification task_id '{notification.task_id}' does not match "
            f"reminder.task_id '{reminder.task_id}'."
        )

    if notification.task_id != task.task_id:
        violations.append(
            f"Notification task_id '{notification.task_id}' does not match "
            f"task.task_id '{task.task_id}'."
        )

    # 2. Timing consistency
    if notification.scheduled_for != reminder.scheduled_for:
        violations.append(
            f"Notification scheduled_for '{notification.scheduled_for}' differs from "
            f"reminder scheduled_for '{reminder.scheduled_for}'."
        )

    # 3. Evidence preservation
    if notification.source_evidence != reminder.source_evidence:
        violations.append("Notification source_evidence does not match reminder source_evidence.")

    if notification.source_evidence != task.evidence:
        violations.append("Notification source_evidence does not match task evidence.")

    # 4. Releasability invariants (Notification can NEVER upgrade safety)
    if notification.patient_releasable:
        if not reminder.patient_releasable:
            violations.append(
                "Notification marked patient_releasable=True but source reminder is not patient_releasable."
            )
        if not task.patient_releasable:
            violations.append(
                "Notification marked patient_releasable=True but source task is not patient_releasable."
            )

    # 5. Content fidelity and neutrality check
    fidelity_ok, content_issues = audit_notification_content_fidelity(
        notification.title, notification.body
    )
    if not fidelity_ok:
        violations.extend(content_issues)

    return len(violations) == 0, violations
