"""Phase 6.5 Care Reminders Module.

Enables patient-controlled reminders for validated, date/time-grounded care tasks
while strictly adhering to CAREBRIDGE clinical safety invariants.
"""

from __future__ import annotations

from backend.care_reminders.types import (
    AttendanceEnquiry,
    AttendanceStatus,
    EligibilityResult,
    Reminder,
    ReminderPreset,
    ReminderStatus,
    ReminderType,
)
from backend.care_reminders.eligibility import (
    check_reminder_eligibility,
    is_reminder_eligible,
)
from backend.care_reminders.scheduler import (
    cancel_reminder,
    create_reminder,
    dismiss_reminder,
    respond_to_attendance_enquiry,
    trigger_reminder,
)
from backend.care_reminders.provenance import validate_reminder_provenance

__all__ = [
    "AttendanceEnquiry",
    "AttendanceStatus",
    "EligibilityResult",
    "Reminder",
    "ReminderPreset",
    "ReminderStatus",
    "ReminderType",
    "check_reminder_eligibility",
    "is_reminder_eligible",
    "cancel_reminder",
    "create_reminder",
    "dismiss_reminder",
    "respond_to_attendance_enquiry",
    "trigger_reminder",
    "validate_reminder_provenance",
]
