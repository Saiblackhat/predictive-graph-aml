"""Phase 6.5 Care Reminder Scheduler & Lifecycle Engine.

Provides deterministic reminder creation, preset calculations, deduplication,
and lifecycle management without altering clinical task semantics.
"""

from __future__ import annotations

import datetime
from typing import Optional, Union

from backend.task_orchestration.types import (
    OrchestratedTask,
    TaskCategory,
    TaskStatus,
)
from backend.care_reminders.types import (
    AttendanceEnquiry,
    AttendanceStatus,
    Reminder,
    ReminderPreset,
    ReminderStatus,
)
from backend.care_reminders.eligibility import check_reminder_eligibility
from backend.care_reminders.provenance import validate_reminder_provenance

SPANISH_SPECIALTY_MAP: dict[str, str] = {
    "cardiology": "Cardiología",
    "primary care": "Atención primaria",
    "oncology": "Oncología",
    "neurology": "Neurología",
    "orthopedics": "Ortopedia",
    "endocrinology": "Endocrinología",
    "pulmonology": "Neumología",
    "gastroenterology": "Gastroenterología",
    "dermatology": "Dermatología",
    "nephrology": "Nefrología",
    "surgery": "Cirugía",
}

SPANISH_MONTHS = [
    "",
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
]


def _format_event_display(
    event_dt: datetime.datetime,
    has_explicit_time: bool,
    language: str,
) -> str:
    """Format event date/time for patient-friendly display."""
    if language == "es":
        month_name = SPANISH_MONTHS[event_dt.month]
        date_str = f"{event_dt.day} de {month_name} de {event_dt.year}"
        if has_explicit_time:
            time_str = f"{event_dt.hour:02d}:{event_dt.minute:02d}"
            return f"{date_str} · {time_str}"
        return date_str

    # English default
    month_name = event_dt.strftime("%B")
    date_str = f"{month_name} {event_dt.day}, {event_dt.year}"
    if has_explicit_time:
        time_str = event_dt.strftime("%I:%M %p").lstrip("0")
        return f"{date_str} · {time_str}"
    return date_str


def create_reminder(
    task: OrchestratedTask,
    offset_minutes: int,
    custom_label: Optional[str] = None,
    timezone: Optional[str] = None,
    language: str = "en",
    reference_date: Optional[Union[datetime.date, str]] = None,
    existing_reminders: Optional[list[Reminder]] = None,
) -> Reminder:
    """Deterministically create a patient-facing reminder grounded in an orchestrated care task.

    Enforces:
    1. Task eligibility.
    2. Strictly positive offset (must be scheduled before the event).
    3. Deduplication against existing active reminders.
    4. Unbroken provenance validation.
    """
    elig = check_reminder_eligibility(task, reference_date=reference_date)
    if not elig.is_eligible or elig.event_datetime_iso is None or elig.reminder_type is None:
        raise ValueError(
            f"Task {task.task_id} is not eligible for reminders: {'; '.join(elig.rejection_reasons)}"
        )

    if offset_minutes <= 0:
        raise ValueError("Reminder offset must be positive (scheduled before the event).")

    # 1. Deduplication check
    if existing_reminders:
        for ex in existing_reminders:
            if (
                ex.task_id == task.task_id
                and ex.offset_minutes == offset_minutes
                and ex.status != ReminderStatus.CANCELLED
            ):
                return ex

    # 2. Timing calculation
    event_dt = datetime.datetime.fromisoformat(elig.event_datetime_iso)
    sched_dt = event_dt - datetime.timedelta(minutes=offset_minutes)

    if sched_dt >= event_dt:
        raise ValueError("Reminder scheduled_for cannot be at or after event time.")

    # 3. Timezone resolution
    resolved_tz = timezone or elig.timezone

    # 4. Title and Display Text
    spec_raw = task.clinical_metadata.get("specialty", "")
    if language == "es":
        spec_es = SPANISH_SPECIALTY_MAP.get(spec_raw.lower(), spec_raw) if spec_raw else ""
        if task.task_category == TaskCategory.FOLLOW_UP and spec_es:
            title = f"Seguimiento con {spec_es}"
        elif task.task_category == TaskCategory.TEST:
            title = f"Prueba diagnóstica: {task.title}"
        else:
            title = custom_label or task.title
    else:
        if task.task_category == TaskCategory.FOLLOW_UP and spec_raw:
            title = f"{spec_raw} follow-up"
        elif task.task_category == TaskCategory.TEST:
            title = f"Diagnostic test: {task.title}"
        else:
            title = custom_label or task.title

    display_timing = _format_event_display(event_dt, elig.has_explicit_time, language)
    ui_display = f"{title}\n{display_timing}"

    reminder_id = f"rem-{task.task_id}-{offset_minutes}"

    rem = Reminder(
        reminder_id=reminder_id,
        task_id=task.task_id,
        source_task_id=task.task_id,
        title=title,
        reminder_type=elig.reminder_type,
        scheduled_for=sched_dt.isoformat(),
        event_time=elig.event_datetime_iso,
        offset_minutes=offset_minutes,
        status=ReminderStatus.SCHEDULED,
        patient_releasable=task.patient_releasable,
        source_evidence=task.evidence,
        created_from_validated_task=True,
        provenance_valid=True,
        timezone=resolved_tz,
        custom_label=custom_label,
        language=language,
        ui_display_text=ui_display,
        derived_from_reference_date=elig.derived_from_reference_date,
    )

    rem.provenance_valid = validate_reminder_provenance(rem, task)
    return rem


def cancel_reminder(reminder: Reminder, task: OrchestratedTask) -> tuple[Reminder, OrchestratedTask]:
    """Cancel a reminder without modifying or cancelling the underlying medical task.

    Strict invariant: Cancel reminder != Cancel appointment.
    """
    reminder.status = ReminderStatus.CANCELLED
    # task status remains unchanged
    return reminder, task


def dismiss_reminder(reminder: Reminder) -> Reminder:
    """Dismiss a triggered reminder."""
    reminder.status = ReminderStatus.DISMISSED
    return reminder


def trigger_reminder(reminder: Reminder) -> Reminder:
    """Mark a reminder as triggered and initiate attendance enquiry."""
    reminder.status = ReminderStatus.TRIGGERED
    reminder.attendance_enquiry = AttendanceEnquiry()
    return reminder


def respond_to_attendance_enquiry(
    reminder: Reminder,
    response: AttendanceStatus,
) -> Reminder:
    """Record patient response to appointment attendance enquiry.

    Strict invariant: No response != missed appointment.
    Never generates clinical diagnosis or ungrounded advice.
    """
    if reminder.attendance_enquiry is None:
        reminder.attendance_enquiry = AttendanceEnquiry()

    enq = reminder.attendance_enquiry
    enq.response = response
    enq.response_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    enq.clinical_advice_provided = False

    if response == AttendanceStatus.COMPLETED:
        reminder.status = ReminderStatus.ACKNOWLEDGED
        enq.followup_note = "Appointment marked completed."
    elif response == AttendanceStatus.NOT_YET:
        reminder.status = ReminderStatus.ACKNOWLEDGED
        enq.followup_note = "If you need to reschedule, contact your provider's office directly."
    elif response == AttendanceStatus.NEED_HELP:
        reminder.status = ReminderStatus.ACKNOWLEDGED
        enq.followup_note = "Please call your care coordinator or clinic phone line for assistance."
    elif response == AttendanceStatus.NO_RESPONSE:
        # Never mark as missed!
        reminder.status = ReminderStatus.TRIGGERED
        enq.followup_note = None

    return reminder
