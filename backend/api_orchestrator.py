"""backend/api_orchestrator.py

Phase 8 — Unified API Orchestrator for CareBridge Cinematic UI.
Connects the UI to Phase 4, Phase 5, Phase 6, Phase 6.5, Phase 6.6, and Phase 7 backend domains.
Enforces all frozen safety invariants without duplicating clinical logic.
"""

from __future__ import annotations

import datetime
from typing import Any, Optional

from backend.care_reminders import (
    ReminderPreset,
    check_reminder_eligibility,
    create_reminder,
)
from backend.care_reminders.types import Reminder
from backend.document_intelligence import run_document_intelligence_pipeline
from backend.document_intelligence.fixtures import ALL_PHASE4_SYNTHETIC_DOCUMENTS
from backend.notifications import (
    Notification,
    NotificationAction,
    NotificationCenterState,
    NotificationStatus,
    acknowledge_notification,
    cancel_notification,
    check_notification_eligibility,
    create_notification,
    dismiss_notification,
    get_notifications,
    snooze_notification,
)
from backend.patient_language import generate_patient_care_plan
from backend.provider_finder import find_providers, ProviderSearchQuery
from backend.schemas.models import MANDATORY_CLINICAL_DISCLAIMER
from backend.task_orchestration import orchestrate_care_journey
from backend.task_orchestration.types import TaskCategory, TaskStatus


# In-memory storage for active demo notifications to support interactive state transitions
ACTIVE_NOTIFICATION_STORE: dict[str, Notification] = {}


def get_available_cases() -> list[dict[str, Any]]:
    """Return available synthetic cases with metadata for the UI case selector."""
    cases = []
    # Dedicated absolute follow-up case (benchmark for reminders and notifications)
    cases.append({
        "case_id": "case_absolute_cardiology",
        "title": "Cardiology Absolute Follow-up (March 18, 2027)",
        "scenario": "Grounded Absolute Appointment & Notifications",
        "patient_id": "SYNTH-PT-801",
        "description": "Discharge instructions with explicit appointment: Follow up with Cardiology on March 18, 2027 at 10:30 AM.",
        "raw_text": (
            "Discharge Instructions:\n"
            "Take metoprolol succinate 50 mg by mouth once daily for blood pressure.\n"
            "Take amlodipine 5 mg by mouth once daily in the morning.\n\n"
            "Follow-Up:\n"
            "Follow up with Cardiology on March 18, 2027 at 10:30 AM.\n\n"
            "Precautions:\n"
            "Monitor for mild ankle swelling and call clinic during office hours if noted.\n"
        ),
        "expected_safety": "SAFE / ORGANIZED",
    })

    for key, doc in ALL_PHASE4_SYNTHETIC_DOCUMENTS.items():
        safety_hint = "SAFE / ORGANIZED"
        if "ambiguous" in key:
            safety_hint = "NEEDS REVIEW"
        elif "conflicting" in key:
            safety_hint = "CONFLICT DETECTED"

        cases.append({
            "case_id": doc.scenario_key,
            "title": doc.description,
            "scenario": doc.scenario_key.replace("_", " ").title(),
            "patient_id": doc.patient_identifier,
            "description": doc.description,
            "raw_text": doc.raw_text,
            "expected_safety": safety_hint,
        })

    return cases


def orchestrate_pipeline(
    case_id: Optional[str] = None,
    raw_text: Optional[str] = None,
    language: str = "en",
) -> dict[str, Any]:
    """Execute the end-to-end CAREBRIDGE pipeline and return unified UI state."""
    # Resolve input text
    selected_case = None
    if raw_text and raw_text.strip():
        text_to_process = raw_text.strip()
        case_label = "Custom User Input"
        patient_id = "SYNTH-PT-CUSTOM"
    elif case_id:
        cases = {c["case_id"]: c for c in get_available_cases()}
        selected_case = cases.get(case_id)
        if selected_case:
            text_to_process = selected_case["raw_text"]
            case_label = selected_case["title"]
            patient_id = selected_case["patient_id"]
        else:
            first = cases["case_absolute_cardiology"]
            text_to_process = first["raw_text"]
            case_label = first["title"]
            patient_id = first["patient_id"]
    else:
        cases = {c["case_id"]: c for c in get_available_cases()}
        first = cases["case_absolute_cardiology"]
        text_to_process = first["raw_text"]
        case_label = first["title"]
        patient_id = first["patient_id"]

    # ── Phase 4 & Phase 3C: Document Intelligence & Safety Validation ─────────
    doc_result = run_document_intelligence_pipeline(text_to_process)
    val_report = doc_result.validation_report

    # ── Phase 5: Task & Timeline Orchestration ─────────────────────────────────
    care_journey = orchestrate_care_journey(doc_result)

    # ── Phase 6: Patient-Friendly Language & Spanish Fidelity ─────────────────
    from backend.patient_language.types import LanguageCode
    target_lang = LanguageCode.ES if language.lower() == "es" else None
    care_plan = generate_patient_care_plan(care_journey, target_language=target_lang)

    # ── Phase 6.5 & 6.6: Reminders & Notifications ────────────────────────────
    reminders: list[Reminder] = []
    notifications: list[Notification] = []

    for task in care_journey.timeline.all_tasks:
        elig = check_reminder_eligibility(task)
        if elig.is_eligible and elig.event_datetime_iso:
            try:
                rem = create_reminder(
                    task=task,
                    offset_minutes=ReminderPreset.ONE_DAY.value,
                    language=language,
                )
                reminders.append(rem)

                notif_elig = check_notification_eligibility(rem, task)
                if notif_elig.is_eligible:
                    notif = create_notification(
                        reminder=rem,
                        task=task,
                        language=language,
                    )
                    notifications.append(notif)
                    ACTIVE_NOTIFICATION_STORE[notif.notification_id] = notif
            except Exception:
                pass  # Graceful fallback without crashing

    notif_center = get_notifications(notifications)

    # ── Human Review Items Aggregation ─────────────────────────────────────────
    review_items = []
    # From Validation Report
    for code in val_report.hazard_codes_detected:
        review_items.append({
            "item_id": "document-hazard",
            "hazard_code": code,
            "severity": "CRITICAL" if "CONFLICT" in code or "PROHIBITED" in code else "WARNING",
            "message": f"Clinical Safety Flag detected: {code}",
            "evidence_snippet": "",
            "status": "PENDING_HUMAN_REVIEW",
        })

    if val_report.validated_payload:
        vp = val_report.validated_payload
        all_items = (
            getattr(vp, "medications", [])
            + getattr(vp, "follow_ups", [])
            + getattr(vp, "warning_signs", [])
            + getattr(vp, "tasks", [])
        )
        for item in all_items:
            for flag in getattr(item, "safety_flags", []):
                review_items.append({
                    "item_id": item.id,
                    "hazard_code": flag.code.value if hasattr(flag.code, "value") else str(flag.code),
                    "severity": flag.severity.value if hasattr(flag.severity, "value") else str(flag.severity),
                    "message": flag.message,
                    "evidence_snippet": flag.evidence_snippet or "",
                    "status": "PENDING_HUMAN_REVIEW",
                })

    # From Orchestrated Tasks
    for t in care_journey.timeline.all_tasks:
        if t.status in (TaskStatus.NEEDS_REVIEW, TaskStatus.BLOCKED) or t.requires_human_review:
            review_items.append({
                "item_id": t.task_id,
                "title": t.title,
                "category": t.task_category.value,
                "hazard_code": "TASK_REVIEW_REQUIRED",
                "severity": "WARNING",
                "message": "; ".join(t.review_reasons) if t.review_reasons else "Requires human review.",
                "evidence_snippet": t.evidence.source_text if t.evidence else "",
                "status": t.status.value,
            })

    # Build Document Sections & Spans
    sections = []
    for s in doc_result.document.sections:
        sections.append({
            "name": s.section_name or s.raw_header or "General",
            "start_char": s.raw_start_char,
            "end_char": s.raw_end_char,
            "category": s.canonical_category.value if hasattr(s.canonical_category, "value") else str(s.canonical_category),
        })

    sentences = []
    for sent in doc_result.document.sentences:
        sentences.append({
            "id": sent.sentence_id,
            "text": sent.source_text,
            "start": sent.start_offset,
            "end": sent.end_offset,
            "section": sent.section,
        })

    # Build Medications
    medications = []
    for m in doc_result.medications:
        medications.append({
            "id": m.id,
            "name": m.medication_name,
            "raw_display": m.raw_medication_display,
            "action": m.action.value if hasattr(m.action, "value") else str(m.action),
            "dosage": m.dosage,
            "previous_dosage": m.previous_dosage,
            "route": m.route,
            "frequency": m.frequency,
            "indication": m.indication_or_note,
            "evidence": m.evidence.model_dump(mode="json") if m.evidence else None,
            "confidence": m.confidence,
            "is_ambiguous": m.is_ambiguous,
            "ambiguity_note": m.ambiguity_note,
        })

    # Build Tasks
    tasks = []
    for t in care_journey.timeline.all_tasks:
        tasks.append({
            "task_id": t.task_id,
            "title": t.title,
            "category": t.task_category.value,
            "description": t.description,
            "status": t.status.value,
            "timeframe_display": t.timeframe.raw_text if t.timeframe else None,
            "absolute_date": t.timeframe.absolute_date if t.timeframe else None,
            "relative_window": t.timeframe.relative_window if t.timeframe else None,
            "patient_releasable": t.patient_releasable,
            "evidence": t.evidence.model_dump(mode="json") if t.evidence else None,
            "specialty": t.clinical_metadata.get("specialty"),
        })

    # Construct complete unified state
    return {
        "case_info": {
            "case_id": case_id or "case_absolute_cardiology",
            "title": case_label,
            "patient_identifier": patient_id,
        },
        "raw_text": text_to_process,
        "sections": sections,
        "sentences": sentences,
        "validation_report": val_report.model_dump(mode="json"),
        "overall_safety_status": val_report.overall_safety_status.value,
        "patient_releasable": val_report.patient_releasable,
        "requires_human_review": val_report.requires_human_review,
        "medications": medications,
        "tasks": tasks,
        "care_journey": {
            "total_tasks": care_journey.timeline.total_task_count,
            "releasable_tasks": care_journey.timeline.releasable_task_count,
            "review_tasks": care_journey.timeline.review_required_task_count,
            "blocked_tasks": care_journey.timeline.blocked_task_count,
            "is_fully_releasable": care_journey.patient_releasable,
        },
        "care_plan": care_plan.model_dump(mode="json"),
        "reminders": [r.model_dump(mode="json") for r in reminders],
        "notifications": [n.model_dump(mode="json") for n in notifications],
        "notification_center": notif_center.model_dump(mode="json"),
        "human_review_items": review_items,
        "disclaimer": MANDATORY_CLINICAL_DISCLAIMER,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def search_providers_endpoint(
    specialty: str,
    location_city: Optional[str] = None,
    language: Optional[str] = None,
    insurance: Optional[str] = None,
) -> dict[str, Any]:
    """Execute Phase 7 deterministic provider search."""
    q = ProviderSearchQuery(
        specialty=specialty,
        location_city=location_city,
        language=language,
        insurance=insurance,
    )
    result = find_providers(q)
    return result.model_dump(mode="json")


def update_notification_state(
    notification_id: str,
    action: str,
    snooze_minutes: int = 60,
) -> dict[str, Any]:
    """Apply lifecycle action to a notification in memory and return updated center state."""
    act = action.upper()
    notif = ACTIVE_NOTIFICATION_STORE.get(notification_id)

    if notif:
        if act == "ACKNOWLEDGE":
            acknowledge_notification(notif)
        elif act == "DISMISS":
            dismiss_notification(notif)
        elif act == "CANCEL":
            cancel_notification(notif)
        elif act == "SNOOZE":
            snoozed = snooze_notification(notif, snooze_minutes)
            ACTIVE_NOTIFICATION_STORE[snoozed.notification_id] = snoozed

    center = get_notifications(list(ACTIVE_NOTIFICATION_STORE.values()))
    return center.model_dump(mode="json")
