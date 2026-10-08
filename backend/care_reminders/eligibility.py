"""Phase 6.5 Care Reminder Eligibility Engine.

Evaluates whether a Phase 5 OrchestratedTask meets strict safety and grounding
criteria to allow patient-enabled care reminders.

INVARIANTS:
1. Must be patient-releasable and not blocked or requiring human review.
2. Must not invent dates, times, providers, or urgency.
3. Medication instructions and warning monitoring cannot become appointment reminders.
4. Relative windows without a trusted reference date are not eligible for calendar reminders.
"""

from __future__ import annotations

import datetime
import re
from typing import Optional, Union

from backend.task_orchestration.types import (
    OrchestratedTask,
    TaskCategory,
    TaskStatus,
    TimeframeType,
)
from backend.care_reminders.types import EligibilityResult, ReminderType

TIME_RE = re.compile(
    r"\b(?:at\s+)?(\d{1,2}):(\d{2})(?:\s*(am|pm))?\b",
    re.IGNORECASE,
)
TIME_HOUR_AMPM_RE = re.compile(
    r"\b(?:at\s+)?(\d{1,2})\s*(am|pm)\b",
    re.IGNORECASE,
)
TIMEZONE_RE = re.compile(
    r"\b(EST|EDT|CST|CDT|MST|MDT|PST|PDT|UTC)\b"
)


def _extract_time_from_text(text: str) -> tuple[Optional[int], Optional[int]]:
    """Deterministically parse hour (0-23) and minute from clinical text."""
    m = TIME_RE.search(text)
    if m:
        h_str, m_str, am_pm = m.groups()
        hour = int(h_str)
        minute = int(m_str)
        if am_pm:
            am_pm = am_pm.upper()
            if am_pm == "PM" and hour < 12:
                hour += 12
            elif am_pm == "AM" and hour == 12:
                hour = 0
        return hour, minute

    m_hour = TIME_HOUR_AMPM_RE.search(text)
    if m_hour:
        h_str, am_pm = m_hour.groups()
        hour = int(h_str)
        am_pm = am_pm.upper()
        if am_pm == "PM" and hour < 12:
            hour += 12
        elif am_pm == "AM" and hour == 12:
            hour = 0
        return hour, 0

    return None, None


def check_reminder_eligibility(
    task: OrchestratedTask,
    reference_date: Optional[Union[datetime.date, str]] = None,
) -> EligibilityResult:
    """Deterministically check if an orchestrated task is eligible for reminder creation."""
    rejection_reasons: list[str] = []

    # 1. Core Safety and Status Checks
    if not task.patient_releasable:
        rejection_reasons.append("Task is not patient-releasable.")

    if task.status in (TaskStatus.BLOCKED, TaskStatus.NEEDS_REVIEW):
        rejection_reasons.append(f"Task status is {task.status.value}.")

    if task.requires_human_review:
        rejection_reasons.append("Task requires clinician review.")

    for reason in task.review_reasons:
        lower = reason.lower()
        if any(term in lower for term in ("unsupported", "best", "conflict", "ambiguous")):
            rejection_reasons.append(f"Task has safety review flag: {reason}")

    # 2. Category Applicability
    if task.task_category == TaskCategory.MEDICATION:
        rejection_reasons.append("Medication instructions cannot be scheduled as appointment reminders.")
    elif task.task_category == TaskCategory.WARNING_MONITORING:
        rejection_reasons.append("Symptom warning monitoring cannot be scheduled as appointment reminders.")

    reminder_type: Optional[ReminderType] = None
    if task.task_category == TaskCategory.FOLLOW_UP:
        reminder_type = ReminderType.APPOINTMENT
    elif task.task_category == TaskCategory.TEST:
        reminder_type = ReminderType.TEST
    elif task.task_category in (TaskCategory.CARE_INSTRUCTION, TaskCategory.REFERRAL):
        reminder_type = ReminderType.CARE_TASK

    # 3. Timeframe Grounding Checks
    tf = task.timeframe
    event_date: Optional[str] = None
    event_time: Optional[str] = None
    event_datetime_iso: Optional[str] = None
    has_explicit_time = False
    tz: Optional[str] = None
    timeframe_disp: Optional[str] = tf.raw_text or task.due_date or task.due_window
    derived_from_ref = False

    if tf.is_missing or tf.timeframe_type == TimeframeType.UNSPECIFIED:
        rejection_reasons.append("Task timeframe is missing or unspecified.")
    elif tf.timeframe_type == TimeframeType.IMMEDIATE:
        rejection_reasons.append("Immediate care actions are not eligible for scheduled reminders.")
    elif tf.timeframe_type == TimeframeType.RECURRING:
        rejection_reasons.append("Recurring general schedules are not eligible for appointment reminders.")
    elif tf.timeframe_type == TimeframeType.ABSOLUTE_DATE:
        if not tf.absolute_date:
            rejection_reasons.append("Absolute date value is empty.")
        else:
            event_date = tf.absolute_date
            ev_text = task.evidence.source_text or ""
            parsed_hour, parsed_min = _extract_time_from_text(ev_text)
            if parsed_hour is None and tf.raw_text:
                parsed_hour, parsed_min = _extract_time_from_text(tf.raw_text)

            if parsed_hour is not None and parsed_min is not None:
                has_explicit_time = True
                event_time = f"{parsed_hour:02d}:{parsed_min:02d}:00"
                event_datetime_iso = f"{event_date}T{event_time}"
            else:
                has_explicit_time = False
                event_time = "09:00:00"
                event_datetime_iso = f"{event_date}T{event_time}"

            m_tz = TIMEZONE_RE.search(ev_text)
            if m_tz:
                tz = m_tz.group(1)

    elif tf.timeframe_type == TimeframeType.RELATIVE_WINDOW:
        if reference_date is None:
            rejection_reasons.append(
                "Relative timeframe ('within X days') cannot produce calendar reminder without trusted reference date."
            )
        else:
            # Parse trusted reference date
            ref_dt: Optional[datetime.date] = None
            if isinstance(reference_date, str):
                try:
                    ref_dt = datetime.date.fromisoformat(reference_date)
                except ValueError:
                    rejection_reasons.append(f"Invalid reference date format: {reference_date}")
            elif isinstance(reference_date, datetime.date):
                ref_dt = reference_date

            if ref_dt:
                op = (tf.relative_operator or "").lower()
                dur = tf.relative_duration_days
                if op in ("within", "in", "by") and dur is not None and dur > 0:
                    target_dt = ref_dt + datetime.timedelta(days=dur)
                    event_date = target_dt.isoformat()
                    event_time = "09:00:00"
                    event_datetime_iso = f"{event_date}T{event_time}"
                    has_explicit_time = False
                    derived_from_ref = True
                else:
                    rejection_reasons.append(
                        f"Relative operator '{op}' or duration '{dur}' not eligible for calendar derived scheduling."
                    )

    is_eligible = len(rejection_reasons) == 0 and event_datetime_iso is not None

    return EligibilityResult(
        is_eligible=is_eligible,
        task_id=task.task_id,
        reminder_type=reminder_type if is_eligible else None,
        event_date=event_date if is_eligible else None,
        event_time=event_time if is_eligible else None,
        event_datetime_iso=event_datetime_iso if is_eligible else None,
        has_explicit_time=has_explicit_time if is_eligible else False,
        timezone=tz if is_eligible else None,
        timeframe_display=timeframe_disp,
        derived_from_reference_date=derived_from_ref,
        rejection_reasons=rejection_reasons,
    )


def is_reminder_eligible(
    task: OrchestratedTask,
    reference_date: Optional[Union[datetime.date, str]] = None,
) -> bool:
    """Convenience boolean check for reminder eligibility."""
    return check_reminder_eligibility(task, reference_date=reference_date).is_eligible
