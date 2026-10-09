"""Phase 6.6 Notification Formatter.

Formats notification title and body deterministically in English and Spanish,
ensuring strict clinical neutrality and zero unauthorized inference.
"""

from __future__ import annotations

import datetime
import re
from typing import Optional

from backend.task_orchestration.types import OrchestratedTask, TaskCategory
from backend.care_reminders.types import Reminder
from backend.notifications.types import NotificationType


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

# Strict list of prohibited tokens/phrases in patient notifications
BANNED_NOTIFICATION_PHRASES: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bappointment\s+(?:is\s+)?confirmed\b",
    r"\bappointment\s+booked\b",
    r"\bdon'?t\s+miss\b",
    r"\byour\s+doctor\s+is\s+expecting\s+you\b",
    r"\byour\s+heart\s+needs\s+attention\b",
    r"\byou\s+need\s+to\s+see\b",
    r"\bcondition\s+is\s+getting\s+worse\b",
    r"\bbest\s+(?:doctor|cardiologist|provider|specialist)\b",
    r"\btop[\s-]rated\b",
    r"\brecommended\s+provider\b",
    r"\bmissed\s+appointment\b",
    r"\bdebe\b",
    r"\bnecesita\b",
    r"\burgente\b",
    r"\bemergency\b",
    r"\bcritical\b",
    r"\bdanger\b",
]]


def format_event_datetime(dt: datetime.datetime, language: str = "en") -> str:
    """Format appointment or event datetime safely for display."""
    if language == "es":
        month_name = SPANISH_MONTHS[dt.month]
        date_str = f"{dt.day} de {month_name} de {dt.year}"
        time_str = f"{dt.hour:02d}:{dt.minute:02d}"
        return f"{date_str} · {time_str}"

    month_name = dt.strftime("%B")
    date_str = f"{month_name} {dt.day}, {dt.year}"
    time_str = dt.strftime("%I:%M %p").lstrip("0")
    return f"{date_str} · {time_str}"


def format_notification_content(
    reminder: Reminder,
    task: OrchestratedTask,
    language: str = "en",
    notification_type: Optional[NotificationType] = None,
) -> tuple[str, str]:
    """Deterministically format (title, body) for a notification across all 12 supported languages.

    Never creates clinical meaning or infers confirmed bookings.
    """
    lang = (language or "en").lower().strip()
    if lang not in ("en", "es"):
        from backend.notifications.multilingual import format_multilingual_notification
        title, body, _ = format_multilingual_notification(
            task=task, reminder=reminder, language=lang, notification_type=notification_type
        )
        return title, body
    event_dt = datetime.datetime.fromisoformat(reminder.event_time)
    dt_display = format_event_datetime(event_dt, language=language)

    if task.task_category == TaskCategory.FOLLOW_UP:
        specialty_raw = task.clinical_metadata.get("specialty", "Specialty Clinic") or "Specialty Clinic"
        if language == "es":
            specialty = SPANISH_SPECIALTY_MAP.get(specialty_raw.lower(), specialty_raw)
            title = f"Próximo seguimiento: {specialty}" if specialty != "Specialty Clinic" else "Próximo seguimiento"
            body = (
                f"{specialty}\n"
                f"{dt_display}\n"
                f"De su plan de alta.\n"
                f"Ver plan de cuidados →"
            )
        else:
            title = f"Upcoming follow-up: {specialty_raw}" if specialty_raw != "Specialty Clinic" else "Upcoming follow-up"
            body = (
                f"{specialty_raw}\n"
                f"{dt_display}\n"
                f"From your discharge plan.\n"
                f"View care journey →"
            )

    elif task.task_category == TaskCategory.TEST:
        test_name = task.clinical_metadata.get("test_name", task.title) or task.title
        if language == "es":
            title = f"Próxima prueba: {test_name}"
            body = (
                f"{test_name}\n"
                f"{dt_display}\n"
                f"De su plan de cuidados.\n"
                f"Ver plan de cuidados →"
            )
        else:
            title = f"Upcoming test: {test_name}"
            body = (
                f"{test_name}\n"
                f"{dt_display}\n"
                f"From your care plan.\n"
                f"View care journey →"
            )

    else:
        # Generic care task
        if language == "es":
            title = "Recordatorio de tarea de cuidados"
            body = (
                f"{task.title}\n"
                f"{dt_display}\n"
                f"De su plan de alta.\n"
                f"Ver plan de cuidados →"
            )
        else:
            title = "Care task reminder"
            body = (
                f"{task.title}\n"
                f"{dt_display}\n"
                f"From your discharge plan.\n"
                f"View care journey →"
            )

    return title, body


def audit_notification_content_fidelity(title: str, body: str) -> tuple[bool, list[str]]:
    """Scan notification text to ensure zero banned or unsafe phrases exist."""
    violations: list[str] = []
    combined = f"{title} {body}"

    for pat in BANNED_NOTIFICATION_PHRASES:
        m = pat.search(combined)
        if m:
            violations.append(f"Forbidden phrase '{m.group(0)}' found in notification text.")

    if len(title) > 120:
        violations.append(f"Title exceeds 120 characters ({len(title)} chars).")

    if len(body) > 500:
        violations.append(f"Body exceeds 500 characters ({len(body)} chars).")

    return len(violations) == 0, violations
