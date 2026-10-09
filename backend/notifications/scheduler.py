"""Phase 6.6 Notification Scheduler & Lifecycle Engine.

Provides deterministic notification scheduling, duplicate prevention,
idempotent execution, and operational lifecycle state transitions.
"""

from __future__ import annotations

import datetime
from typing import Optional

from backend.task_orchestration.types import OrchestratedTask
from backend.care_reminders.types import Reminder
from backend.notifications.types import (
    Notification,
    NotificationAction,
    NotificationCenterState,
    NotificationChannel,
    NotificationPriority,
    NotificationStatus,
    NotificationSummary,
    NotificationType,
)
from backend.notifications.eligibility import check_notification_eligibility
from backend.notifications.formatter import format_notification_content
from backend.notifications.provenance import validate_notification_provenance


def create_notification(
    reminder: Reminder,
    task: OrchestratedTask,
    language: str = "en",
    timezone: Optional[str] = None,
    existing_notifications: Optional[list[Notification]] = None,
    patient_id: Optional[str] = None,
    channel: NotificationChannel = NotificationChannel.IN_APP,
    notification_type: Optional[NotificationType] = None,
) -> Notification:
    """Deterministically create an operational notification from a validated reminder.

    Enforces:
    1. Eligibility check.
    2. Idempotent duplicate prevention.
    3. Multilingual content formatting and channel rendering.
    4. Translation fidelity verification.
    5. Full provenance validation.
    """
    elig = check_notification_eligibility(reminder, task)
    if not elig.is_eligible or elig.scheduled_for is None or elig.notification_type is None:
        raise ValueError(
            f"Reminder {reminder.reminder_id} is not eligible for notification: "
            f"{'; '.join(elig.rejection_reasons)}"
        )

    target_type = notification_type or elig.notification_type
    lang = (language or "en").lower().strip()

    # 1. Idempotency & Duplicate prevention
    if existing_notifications:
        for ex in existing_notifications:
            if (
                ex.reminder_id == reminder.reminder_id
                and (ex.scheduled_for == elig.scheduled_for or getattr(ex, "scheduled_at", None) == elig.scheduled_for)
                and ex.notification_type == target_type
                and getattr(ex, "channel", NotificationChannel.IN_APP) == channel
                and ex.status != NotificationStatus.CANCELLED
            ):
                return ex

    # 2. Content formatting & Channel rendering
    from backend.notifications.multilingual import (
        build_channel_payloads,
        format_multilingual_notification,
        render_channel_payload,
        verify_translation_fidelity,
    )

    if lang in ("en", "es"):
        raw_title, raw_body = format_notification_content(
            reminder, task, language=lang, notification_type=target_type
        )
        trans_status = "VERIFIED"
    else:
        raw_title, raw_body, trans_status = format_multilingual_notification(
            task=task,
            reminder=reminder,
            language=lang,
            notification_type=target_type,
        )

    title, body = render_channel_payload(
        title=raw_title,
        body=raw_body,
        channel=channel,
        language=lang,
    )

    # 3. Translation fidelity verification
    is_verified, trans_status_ver, trans_notes = verify_translation_fidelity(
        title=title,
        body=body,
        task=task,
        reminder=reminder,
        language=lang,
    )
    final_trans_status = trans_status_ver if not is_verified else trans_status

    # 4. Notification instance construction
    notif_id = f"notif-{reminder.reminder_id}"
    if channel != NotificationChannel.IN_APP:
        notif_id = f"{notif_id}-{channel.value.lower()}"
    resolved_tz = timezone or reminder.timezone
    resolved_patient_id = patient_id or getattr(task, "patient_id", "SYNTH-PT-801") or "SYNTH-PT-801"

    is_releasable = reminder.patient_releasable and task.patient_releasable
    if not is_verified:
        is_releasable = False

    notif = Notification(
        notification_id=notif_id,
        id=notif_id,
        patient_id=resolved_patient_id,
        reminder_id=reminder.reminder_id,
        task_id=task.task_id,
        source_task_id=task.task_id,
        notification_type=target_type,
        channel=channel,
        title=title,
        body=body,
        status=NotificationStatus.SCHEDULED,
        priority=NotificationPriority.NORMAL,
        scheduled_for=elig.scheduled_for,
        scheduled_at=elig.scheduled_for,
        triggered_at=None,
        acknowledged_at=None,
        dismissed_at=None,
        cancelled_at=None,
        unread=True,
        patient_releasable=is_releasable,
        source_evidence=reminder.source_evidence,
        provenance_valid=False,  # verified below
        created_from_validated_reminder=True,
        language=lang,
        translation_status=final_trans_status,
        translation_verification_notes=trans_notes,
        timezone=resolved_tz,
        metadata={
            "offset_minutes": reminder.offset_minutes,
            "event_time": reminder.event_time,
            "specialty": task.clinical_metadata.get("specialty"),
            "channel": channel.value if hasattr(channel, "value") else str(channel),
            "channel_payloads": build_channel_payloads(raw_title, raw_body, language=lang),
        },
    )

    # 5. Provenance validation
    prov_ok, _ = validate_notification_provenance(notif, reminder, task)
    notif.provenance_valid = prov_ok

    if not prov_ok:
        notif.patient_releasable = False

    return notif


def trigger_notification(
    notification: Notification,
    now_iso: Optional[str] = None,
) -> Notification:
    """Transition notification from SCHEDULED to TRIGGERED."""
    if notification.status != NotificationStatus.SCHEDULED:
        return notification

    timestamp = now_iso or datetime.datetime.now(datetime.timezone.utc).isoformat()
    notification.status = NotificationStatus.TRIGGERED
    notification.triggered_at = timestamp
    return notification


def acknowledge_notification(
    notification: Notification,
    now_iso: Optional[str] = None,
) -> Notification:
    """Patient acknowledges notification. Changes notification state only."""
    if notification.status == NotificationStatus.CANCELLED:
        return notification

    timestamp = now_iso or datetime.datetime.now(datetime.timezone.utc).isoformat()
    notification.status = NotificationStatus.ACKNOWLEDGED
    notification.acknowledged_at = timestamp
    notification.unread = False
    return notification


def dismiss_notification(
    notification: Notification,
    now_iso: Optional[str] = None,
) -> Notification:
    """Patient dismisses notification. Changes notification state only."""
    if notification.status == NotificationStatus.CANCELLED:
        return notification

    timestamp = now_iso or datetime.datetime.now(datetime.timezone.utc).isoformat()
    notification.status = NotificationStatus.DISMISSED
    notification.dismissed_at = timestamp
    return notification


def cancel_notification(
    notification: Notification,
    now_iso: Optional[str] = None,
) -> Notification:
    """Cancel notification schedule. Does NOT modify the underlying task."""
    timestamp = now_iso or datetime.datetime.now(datetime.timezone.utc).isoformat()
    notification.status = NotificationStatus.CANCELLED
    notification.cancelled_at = timestamp
    return notification


def snooze_notification(
    notification: Notification,
    snooze_minutes: int,
    base_dt: Optional[datetime.datetime] = None,
) -> Notification:
    """Create a new snoozed operational notification schedule.

    Never modifies the original care task, appointment date, or provider.
    """
    if snooze_minutes <= 0:
        raise ValueError("Snooze duration must be positive.")

    current = base_dt or datetime.datetime.now(datetime.timezone.utc)
    new_sched = (current + datetime.timedelta(minutes=snooze_minutes)).isoformat()

    snoozed = notification.model_copy(deep=True)
    snoozed.notification_id = f"{notification.notification_id}-snooze-{snooze_minutes}"
    snoozed.scheduled_for = new_sched
    snoozed.status = NotificationStatus.SCHEDULED
    snoozed.triggered_at = None
    snoozed.acknowledged_at = None
    snoozed.dismissed_at = None
    snoozed.cancelled_at = None
    snoozed.unread = True
    return snoozed


def get_notifications(
    notifications: list[Notification],
    status_filter: Optional[NotificationStatus] = None,
    unread_only: bool = False,
) -> NotificationCenterState:
    """Aggregate notification summaries for Phase 8 Cinematic UI consumption."""
    filtered: list[Notification] = []

    for n in notifications:
        if status_filter and n.status != status_filter:
            continue
        if unread_only and not n.unread:
            continue
        filtered.append(n)

    summaries = [
        NotificationSummary(
            notification_id=n.notification_id,
            id=n.id or n.notification_id,
            patient_id=n.patient_id,
            title=n.title,
            body=n.body,
            type=n.notification_type,
            channel=n.channel,
            status=n.status,
            priority=n.priority,
            scheduled_for=n.scheduled_for,
            scheduled_at=n.scheduled_at or n.scheduled_for,
            triggered_at=n.triggered_at,
            unread=n.unread,
            patient_releasable=n.patient_releasable,
            task_id=n.task_id,
            source_task_id=n.source_task_id or n.task_id,
            reminder_id=n.reminder_id,
            language=n.language,
            translation_status=n.translation_status,
            source_label="Discharge Plan",
        )
        for n in filtered
    ]

    unread_count = sum(1 for n in notifications if n.unread and n.status != NotificationStatus.CANCELLED)

    return NotificationCenterState(
        total_count=len(summaries),
        unread_count=unread_count,
        notifications=summaries,
    )
