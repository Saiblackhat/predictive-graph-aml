"""Phase 6.5 Care Reminder Provenance Engine.

Validates that every Reminder strictly maintains unbroken end-to-end evidence
provenance back to the original clinical task, evidence span, and discharge text.
"""

from __future__ import annotations

import datetime

from backend.task_orchestration.types import OrchestratedTask
from backend.care_reminders.types import Reminder


def validate_reminder_provenance(reminder: Reminder, task: OrchestratedTask) -> bool:
    """Deterministically verify unbroken provenance between reminder and care task.

    Returns True if and only if all provenance invariants hold.
    """
    # 1. Identity Consistency
    if reminder.task_id != task.task_id:
        return False
    if reminder.source_task_id != task.task_id:
        return False

    # 2. Evidence Grounding
    if reminder.source_evidence != task.evidence:
        return False
    if not task.evidence.source_text or not task.evidence.source_text.strip():
        return False
    if task.evidence.start_offset < 0 or task.evidence.end_offset <= task.evidence.start_offset:
        return False

    # 3. Patient Safety Consistency
    if reminder.patient_releasable != task.patient_releasable:
        return False

    # 4. Mathematical Grounding
    try:
        event_dt = datetime.datetime.fromisoformat(reminder.event_time)
        sched_dt = datetime.datetime.fromisoformat(reminder.scheduled_for)
        diff_minutes = int((event_dt - sched_dt).total_seconds() / 60)
        if diff_minutes != reminder.offset_minutes:
            return False
        if reminder.offset_minutes <= 0:
            return False
    except (ValueError, TypeError):
        return False

    return True
