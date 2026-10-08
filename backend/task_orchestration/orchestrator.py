"""Deterministic Task & Timeline Orchestration Engine for Phase 5.

Transforms validated Phase 4 discharge intelligence into structured,
evidence-grounded care tasks and timelines.

STRICT INVARIANTS:
1. Orchestration MUST NEVER upgrade safety:
   patient_releasable(task) MUST NOT be True if source_item.patient_releasable == False.
2. Every task carries provenance back to verbatim raw text:
   raw_text[task.evidence.start_offset : task.evidence.end_offset] == task.evidence.source_text.
3. No invention of clinical actions, dates, dependencies, or provider relationships.
4. Relative windows ('within 7 days' != 'after 7 days') are strictly preserved.
5. Conflicting instructions are represented and blocked, never silently resolved.
6. Meaningfully distinct formulations ('metoprolol tartrate' vs 'metoprolol succinate')
   are never deduplicated.
"""

from __future__ import annotations

from typing import Any, Optional

from backend.document_intelligence.types import (
    DocumentIntelligenceResult,
    ExtractedCareInstructionRecord,
    ExtractedFollowUpRecord,
    ExtractedMedicationRecord,
    ExtractedReferralRecord,
    ExtractedTestRecord,
    ExtractedWarningSignRecord,
)
from backend.schemas.models import (
    MANDATORY_CLINICAL_DISCLAIMER,
    HazardCode,
    MedicationAction,
    SafetyStatus,
    UrgencyLevel,
)
from backend.task_orchestration.timeframes import parse_clinical_timeframe
from backend.task_orchestration.types import (
    CareJourney,
    CareTimeline,
    OrchestratedTask,
    TaskCategory,
    TaskStatus,
    TimeframeType,
)


def _build_medication_task(
    med: ExtractedMedicationRecord,
    task_idx: int,
    overall_safe: bool,
    conflicts_present: bool,
    raw_text: str,
) -> OrchestratedTask:
    """Deterministically map an ExtractedMedicationRecord into an OrchestratedTask."""
    tf = parse_clinical_timeframe(med.frequency or med.timeframe_or_duration)

    review_reasons: list[str] = []
    status = TaskStatus.PENDING

    # Determine description and title based on clinical action
    action = med.action
    med_name = med.medication_name
    dose = med.dosage or ""
    freq = med.frequency or ""
    route = f"by {med.route}" if med.route else ""

    if action == MedicationAction.STOP_DISCONTINUE:
        title = f"Discontinue {med_name}"
        desc = f"Stop taking {med_name} {dose}. Do not take this medication after discharge."
        status = TaskStatus.COMPLETED
    elif action == MedicationAction.HOLD_TEMPORARY:
        title = f"Hold {med_name}"
        desc = f"Temporarily pause {med_name} {dose}. {med.indication_or_note or ''}".strip()
    elif action == MedicationAction.DOSE_CHANGE:
        title = f"Adjust dose of {med_name} to {dose}"
        desc = f"Change {med_name} dosage {f'from {med.previous_dosage} ' if med.previous_dosage else ''}to {dose} {freq} {route}.".strip()
    elif action == MedicationAction.NEW_START:
        title = f"Start new medication: {med_name} {dose}"
        desc = f"Begin taking {med_name} {dose} {freq} {route}.".strip()
    elif action == MedicationAction.CONTINUE:
        title = f"Continue {med_name} {dose}"
        desc = f"Continue taking {med_name} {dose} {freq} {route}.".strip()
    else:
        title = f"Medication instruction: {med_name}"
        desc = f"Review regimen for {med_name} {dose} {freq} {route} with clinician.".strip()
        status = TaskStatus.NEEDS_REVIEW
        review_reasons.append("Unclear or non-standard medication action verb.")

    # Safety propagation checks
    if med.is_ambiguous or med.dosage is None and action in (MedicationAction.CONTINUE, MedicationAction.NEW_START, MedicationAction.DOSE_CHANGE):
        status = TaskStatus.NEEDS_REVIEW
        review_reasons.append(med.ambiguity_note or f"Missing or ambiguous dosage for '{med_name}'.")

    if conflicts_present:
        status = TaskStatus.BLOCKED
        review_reasons.append(f"Contradictory instructions detected in discharge summary for '{med_name}'.")

    if not overall_safe:
        status = TaskStatus.BLOCKED if conflicts_present else TaskStatus.NEEDS_REVIEW
        if "Discharge document flagged by safety validator." not in review_reasons:
            review_reasons.append("Discharge document flagged by safety validator.")

    # Provenance verification check
    ev = med.evidence
    prov_slice = raw_text[ev.start_offset : ev.end_offset]
    if prov_slice != ev.source_text:
        status = TaskStatus.BLOCKED
        review_reasons.append("Provenance failure: evidence offset does not match raw text slice.")

    releasable = overall_safe and not med.is_ambiguous and len(review_reasons) == 0 and status not in (TaskStatus.NEEDS_REVIEW, TaskStatus.BLOCKED)
    review_required = not releasable

    return OrchestratedTask(
        task_id=f"task-med-{task_idx:03d}",
        task_category=TaskCategory.MEDICATION,
        title=title,
        description=desc,
        status=status,
        timeframe=tf,
        due_date=tf.absolute_date,
        due_window=tf.relative_window,
        source_item_id=med.id,
        evidence=ev,
        patient_releasable=releasable,
        requires_human_review=review_required,
        review_reasons=review_reasons,
        dependency_ids=[],
        priority=1,
        clinical_metadata={
            "medication_name": med_name,
            "raw_medication_display": med.raw_medication_display,
            "action": action.value,
            "dosage": med.dosage,
            "previous_dosage": med.previous_dosage,
            "frequency": med.frequency,
            "route": med.route,
        },
    )


def _build_followup_task(
    fu: ExtractedFollowUpRecord,
    task_idx: int,
    overall_safe: bool,
    raw_text: str,
) -> OrchestratedTask:
    """Deterministically map an ExtractedFollowUpRecord into an OrchestratedTask."""
    tf = parse_clinical_timeframe(fu.timeframe)

    review_reasons: list[str] = []
    status = TaskStatus.SCHEDULED if fu.appointment_scheduled else TaskStatus.PENDING

    spec = fu.specialty or "Specialty Clinic"
    prov = f" ({fu.provider_name})" if fu.provider_name else ""
    tf_desc = f" {fu.timeframe}" if fu.timeframe else ""
    title = f"Follow-up appointment with {spec}{prov}"
    desc = f"Attend follow-up appointment with {spec}{prov}{tf_desc}."
    if fu.reason:
        desc += f" Reason: {fu.reason}."

    if fu.is_ambiguous or fu.timeframe is None:
        status = TaskStatus.NEEDS_REVIEW
        review_reasons.append(fu.ambiguity_note or f"Follow-up with '{spec}' lacks an explicit timeframe.")

    if fu.specialty is None:
        status = TaskStatus.NEEDS_REVIEW
        review_reasons.append("Follow-up appointment missing explicit clinical specialty.")

    if not overall_safe:
        status = TaskStatus.NEEDS_REVIEW
        review_reasons.append("Discharge document flagged by safety validator.")

    # Provenance verification check
    ev = fu.evidence
    prov_slice = raw_text[ev.start_offset : ev.end_offset]
    if prov_slice != ev.source_text:
        status = TaskStatus.BLOCKED
        review_reasons.append("Provenance failure: evidence offset does not match raw text slice.")

    releasable = overall_safe and not fu.is_ambiguous and len(review_reasons) == 0 and status not in (TaskStatus.NEEDS_REVIEW, TaskStatus.BLOCKED)
    review_required = not releasable

    return OrchestratedTask(
        task_id=f"task-fu-{task_idx:03d}",
        task_category=TaskCategory.FOLLOW_UP,
        title=title,
        description=desc,
        status=status,
        timeframe=tf,
        due_date=tf.absolute_date,
        due_window=tf.relative_window,
        source_item_id=fu.id,
        evidence=ev,
        patient_releasable=releasable,
        requires_human_review=review_required,
        review_reasons=review_reasons,
        dependency_ids=[],
        priority=2,
        clinical_metadata={
            "specialty": fu.specialty,
            "provider_name": fu.provider_name,
            "appointment_scheduled": fu.appointment_scheduled,
            "reason": fu.reason,
        },
    )


def _build_test_task(
    test: ExtractedTestRecord,
    task_idx: int,
    overall_safe: bool,
    raw_text: str,
) -> OrchestratedTask:
    """Deterministically map an ExtractedTestRecord into an OrchestratedTask."""
    tf = parse_clinical_timeframe(test.timeframe)

    review_reasons: list[str] = []
    status = TaskStatus.PENDING

    title = f"Diagnostic Test: {test.test_name}"
    desc = f"Complete {test.test_name}{f' {test.timeframe}' if test.timeframe else ''}."
    if test.reason_or_note:
        desc += f" Indication: {test.reason_or_note}."

    if test.is_ambiguous or test.timeframe is None:
        status = TaskStatus.NEEDS_REVIEW
        review_reasons.append(f"Diagnostic test '{test.test_name}' missing explicit timeframe.")

    if not overall_safe:
        status = TaskStatus.NEEDS_REVIEW
        review_reasons.append("Discharge document flagged by safety validator.")

    ev = test.evidence
    prov_slice = raw_text[ev.start_offset : ev.end_offset]
    if prov_slice != ev.source_text:
        status = TaskStatus.BLOCKED
        review_reasons.append("Provenance failure: evidence offset does not match raw text slice.")

    releasable = overall_safe and not test.is_ambiguous and len(review_reasons) == 0 and status not in (TaskStatus.NEEDS_REVIEW, TaskStatus.BLOCKED)
    review_required = not releasable

    return OrchestratedTask(
        task_id=f"task-test-{task_idx:03d}",
        task_category=TaskCategory.TEST,
        title=title,
        description=desc,
        status=status,
        timeframe=tf,
        due_date=tf.absolute_date,
        due_window=tf.relative_window,
        source_item_id=test.id,
        evidence=ev,
        patient_releasable=releasable,
        requires_human_review=review_required,
        review_reasons=review_reasons,
        dependency_ids=[],
        priority=3,
        clinical_metadata={
            "test_name": test.test_name,
            "timeframe": test.timeframe,
            "reason_or_note": test.reason_or_note,
        },
    )


def _build_referral_task(
    ref: ExtractedReferralRecord,
    task_idx: int,
    overall_safe: bool,
    raw_text: str,
) -> OrchestratedTask:
    """Deterministically map an ExtractedReferralRecord into an OrchestratedTask."""
    tf = parse_clinical_timeframe(ref.timeframe)

    review_reasons: list[str] = []
    status = TaskStatus.PENDING

    title = f"Referral: {ref.specialty_or_service}"
    desc = f"Initiate outpatient referral for {ref.specialty_or_service}{f' ({ref.provider_name})' if ref.provider_name else ''}{f' {ref.timeframe}' if ref.timeframe else ''}."
    if ref.reason:
        desc += f" Reason: {ref.reason}."

    if ref.is_ambiguous:
        status = TaskStatus.NEEDS_REVIEW
        review_reasons.append(f"Referral for '{ref.specialty_or_service}' contains ambiguous scheduling details.")

    if not overall_safe:
        status = TaskStatus.NEEDS_REVIEW
        review_reasons.append("Discharge document flagged by safety validator.")

    ev = ref.evidence
    prov_slice = raw_text[ev.start_offset : ev.end_offset]
    if prov_slice != ev.source_text:
        status = TaskStatus.BLOCKED
        review_reasons.append("Provenance failure: evidence offset does not match raw text slice.")

    releasable = overall_safe and not ref.is_ambiguous and len(review_reasons) == 0 and status not in (TaskStatus.NEEDS_REVIEW, TaskStatus.BLOCKED)
    review_required = not releasable

    return OrchestratedTask(
        task_id=f"task-ref-{task_idx:03d}",
        task_category=TaskCategory.REFERRAL,
        title=title,
        description=desc,
        status=status,
        timeframe=tf,
        due_date=tf.absolute_date,
        due_window=tf.relative_window,
        source_item_id=ref.id,
        evidence=ev,
        patient_releasable=releasable,
        requires_human_review=review_required,
        review_reasons=review_reasons,
        dependency_ids=[],
        priority=4,
        clinical_metadata={
            "specialty_or_service": ref.specialty_or_service,
            "provider_name": ref.provider_name,
            "timeframe": ref.timeframe,
            "reason": ref.reason,
        },
    )


def _build_warning_task(
    ws: ExtractedWarningSignRecord,
    task_idx: int,
    overall_safe: bool,
    raw_text: str,
) -> OrchestratedTask:
    """Deterministically map an ExtractedWarningSignRecord into an OrchestratedTask."""
    tf = parse_clinical_timeframe("immediately" if ws.urgency == UrgencyLevel.EMERGENT_911 else "daily")

    review_reasons: list[str] = []
    status = TaskStatus.PENDING

    title = f"Warning Sign Alert: {ws.symptom_description}"
    action_text = ws.action_to_take or "Seek prompt medical attention"
    desc = f"Monitor for: {ws.symptom_description}. Action: {action_text}. Urgency: {ws.urgency.value}."

    if ws.urgency == UrgencyLevel.EMERGENT_911:
        # Emergency red flags always require clinician attention in care coordination
        status = TaskStatus.NEEDS_REVIEW
        review_reasons.append("Emergent 911 red flag instruction requires clinician safety verification.")

    if ws.is_ambiguous:
        status = TaskStatus.NEEDS_REVIEW
        review_reasons.append("Warning sign contains ambiguous symptom phrasing.")

    if not overall_safe:
        status = TaskStatus.NEEDS_REVIEW
        if "Discharge document flagged by safety validator." not in review_reasons:
            review_reasons.append("Discharge document flagged by safety validator.")

    ev = ws.evidence
    prov_slice = raw_text[ev.start_offset : ev.end_offset]
    if prov_slice != ev.source_text:
        status = TaskStatus.BLOCKED
        review_reasons.append("Provenance failure: evidence offset does not match raw text slice.")

    releasable = overall_safe and not ws.is_ambiguous and len(review_reasons) == 0 and status not in (TaskStatus.NEEDS_REVIEW, TaskStatus.BLOCKED)
    review_required = not releasable

    return OrchestratedTask(
        task_id=f"task-warn-{task_idx:03d}",
        task_category=TaskCategory.WARNING_MONITORING,
        title=title,
        description=desc,
        status=status,
        timeframe=tf,
        due_date=None,
        due_window="Continuous monitoring",
        source_item_id=ws.id,
        evidence=ev,
        patient_releasable=releasable,
        requires_human_review=review_required,
        review_reasons=review_reasons,
        dependency_ids=[],
        priority=0,  # highest priority
        clinical_metadata={
            "symptom_description": ws.symptom_description,
            "action_to_take": ws.action_to_take,
            "urgency": ws.urgency.value,
        },
    )


def _build_care_instruction_task(
    ci: ExtractedCareInstructionRecord,
    task_idx: int,
    overall_safe: bool,
    raw_text: str,
) -> OrchestratedTask:
    """Deterministically map an ExtractedCareInstructionRecord into an OrchestratedTask."""
    tf = parse_clinical_timeframe(ci.timeframe)

    review_reasons: list[str] = []
    status = TaskStatus.PENDING

    clean_type = ci.instruction_type.replace("_", " ").title()
    title = f"{clean_type}: {ci.title}"
    desc = ci.detail

    if ci.is_ambiguous:
        status = TaskStatus.NEEDS_REVIEW
        review_reasons.append(ci.ambiguity_note or "Care instruction contains ambiguous phrasing.")

    if not overall_safe:
        status = TaskStatus.NEEDS_REVIEW
        review_reasons.append("Discharge document flagged by safety validator.")

    ev = ci.evidence
    prov_slice = raw_text[ev.start_offset : ev.end_offset]
    if prov_slice != ev.source_text:
        status = TaskStatus.BLOCKED
        review_reasons.append("Provenance failure: evidence offset does not match raw text slice.")

    releasable = overall_safe and not ci.is_ambiguous and len(review_reasons) == 0 and status not in (TaskStatus.NEEDS_REVIEW, TaskStatus.BLOCKED)
    review_required = not releasable

    return OrchestratedTask(
        task_id=f"task-care-{task_idx:03d}",
        task_category=TaskCategory.CARE_INSTRUCTION,
        title=title,
        description=desc,
        status=status,
        timeframe=tf,
        due_date=tf.absolute_date,
        due_window=tf.relative_window,
        source_item_id=ci.id,
        evidence=ev,
        patient_releasable=releasable,
        requires_human_review=review_required,
        review_reasons=review_reasons,
        dependency_ids=[],
        priority=5,
        clinical_metadata={
            "instruction_type": ci.instruction_type,
            "timeframe": ci.timeframe,
        },
    )


def _deduplicate_tasks_without_merging_clinical_differences(
    tasks: list[OrchestratedTask],
) -> list[OrchestratedTask]:
    """Deterministically deduplicate repeated equivalent instructions.

    Never merges clinically distinct formulations ('metoprolol tartrate' vs 'metoprolol succinate')
    or distinct timeframes ('within 7 days' vs 'after 7 days').
    """
    seen_keys: set[tuple[Any, ...]] = set()
    deduped: list[OrchestratedTask] = []

    for t in tasks:
        # Build precise semantic key
        meta = t.clinical_metadata
        if t.task_category == TaskCategory.MEDICATION:
            key = (
                t.task_category,
                str(meta.get("medication_name", "")).lower().strip(),
                str(meta.get("action", "")),
                str(meta.get("dosage", "")).lower().strip(),
                str(meta.get("frequency", "")).lower().strip(),
                str(meta.get("route", "")).lower().strip(),
            )
        elif t.task_category == TaskCategory.FOLLOW_UP:
            key = (
                t.task_category,
                str(meta.get("specialty", "")).lower().strip(),
                str(meta.get("provider_name", "")).lower().strip(),
                str(t.timeframe.raw_text or "").lower().strip(),
                str(t.timeframe.relative_operator or "").lower().strip(),
            )
        elif t.task_category == TaskCategory.TEST:
            key = (
                t.task_category,
                str(meta.get("test_name", "")).lower().strip(),
                str(t.timeframe.raw_text or "").lower().strip(),
            )
        elif t.task_category == TaskCategory.REFERRAL:
            key = (
                t.task_category,
                str(meta.get("specialty_or_service", "")).lower().strip(),
                str(t.timeframe.raw_text or "").lower().strip(),
            )
        elif t.task_category == TaskCategory.WARNING_MONITORING:
            key = (
                t.task_category,
                str(meta.get("symptom_description", "")).lower().strip(),
                str(meta.get("action_to_take", "")).lower().strip(),
            )
        elif t.task_category == TaskCategory.CARE_INSTRUCTION:
            key = (
                t.task_category,
                str(meta.get("instruction_type", "")).lower().strip(),
                t.title.lower().strip(),
            )
        else:
            key = (t.task_category, t.title.lower().strip())

        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(t)

    return deduped


def orchestrate_care_journey(
    extraction_result: DocumentIntelligenceResult,
) -> CareJourney:
    """Orchestrate validated extraction into a structured, evidence-grounded CareJourney.

    Enforces safety inheritance, provenance verification, and timeline assembly.
    """
    doc = extraction_result.document
    raw_text = doc.raw_text
    val_report = extraction_result.validation_report

    overall_safe = val_report.patient_releasable and val_report.overall_safety_status == SafetyStatus.SAFE_ORGANIZED
    conflicts_present = (
        HazardCode.CONFLICTING_INSTRUCTIONS in val_report.hazard_codes_detected
        or val_report.overall_safety_status == SafetyStatus.CONFLICT_DETECTED
    )

    raw_tasks: list[OrchestratedTask] = []

    # 1. Medications
    for idx, med in enumerate(extraction_result.medications, start=1):
        raw_tasks.append(
            _build_medication_task(
                med=med,
                task_idx=idx,
                overall_safe=overall_safe,
                conflicts_present=conflicts_present,
                raw_text=raw_text,
            )
        )

    # 2. Follow-ups
    for idx, fu in enumerate(extraction_result.follow_ups, start=1):
        raw_tasks.append(
            _build_followup_task(
                fu=fu,
                task_idx=idx,
                overall_safe=overall_safe,
                raw_text=raw_text,
            )
        )

    # 3. Tests
    for idx, test in enumerate(extraction_result.tests, start=1):
        raw_tasks.append(
            _build_test_task(
                test=test,
                task_idx=idx,
                overall_safe=overall_safe,
                raw_text=raw_text,
            )
        )

    # 4. Referrals
    for idx, ref in enumerate(extraction_result.referrals, start=1):
        raw_tasks.append(
            _build_referral_task(
                ref=ref,
                task_idx=idx,
                overall_safe=overall_safe,
                raw_text=raw_text,
            )
        )

    # 5. Warning Signs
    for idx, ws in enumerate(extraction_result.warning_signs, start=1):
        raw_tasks.append(
            _build_warning_task(
                ws=ws,
                task_idx=idx,
                overall_safe=overall_safe,
                raw_text=raw_text,
            )
        )

    # 6. Care Instructions
    for idx, ci in enumerate(extraction_result.care_instructions, start=1):
        raw_tasks.append(
            _build_care_instruction_task(
                ci=ci,
                task_idx=idx,
                overall_safe=overall_safe,
                raw_text=raw_text,
            )
        )

    # Deduplicate while strictly preserving meaningful clinical differences
    tasks = _deduplicate_tasks_without_merging_clinical_differences(raw_tasks)

    # Provenance audit invariant verification
    provenance_audit_passed = True
    for t in tasks:
        ev = t.evidence
        if not ev or raw_text[ev.start_offset : ev.end_offset] != ev.source_text:
            provenance_audit_passed = False
            t.status = TaskStatus.BLOCKED
            t.patient_releasable = False
            t.requires_human_review = True
            t.review_reasons.append("Provenance validation failed during final audit.")

    # Categorize tasks into timeline buckets
    immediate_tasks: list[OrchestratedTask] = []
    relative_window_tasks: list[OrchestratedTask] = []
    absolute_date_tasks: list[OrchestratedTask] = []
    recurring_tasks: list[OrchestratedTask] = []
    unspecified_timeframe_tasks: list[OrchestratedTask] = []
    review_or_blocked_tasks: list[OrchestratedTask] = []

    for t in tasks:
        if t.status in (TaskStatus.NEEDS_REVIEW, TaskStatus.BLOCKED):
            review_or_blocked_tasks.append(t)

        tf_type = t.timeframe.timeframe_type
        if tf_type == TimeframeType.IMMEDIATE:
            immediate_tasks.append(t)
        elif tf_type == TimeframeType.RELATIVE_WINDOW:
            relative_window_tasks.append(t)
        elif tf_type == TimeframeType.ABSOLUTE_DATE:
            absolute_date_tasks.append(t)
        elif tf_type == TimeframeType.RECURRING:
            recurring_tasks.append(t)
        else:
            unspecified_timeframe_tasks.append(t)

    # Sort tasks within lists deterministically by priority, then task_id
    tasks.sort(key=lambda x: (x.priority, x.task_id))

    releasable_count = sum(1 for t in tasks if t.patient_releasable)
    review_count = sum(1 for t in tasks if t.requires_human_review)
    blocked_count = sum(1 for t in tasks if t.status == TaskStatus.BLOCKED)

    timeline = CareTimeline(
        timeline_id=f"timeline-{doc.document_id}",
        document_id=doc.document_id,
        patient_identifier=doc.patient_identifier,
        immediate_tasks=immediate_tasks,
        relative_window_tasks=relative_window_tasks,
        absolute_date_tasks=absolute_date_tasks,
        recurring_tasks=recurring_tasks,
        unspecified_timeframe_tasks=unspecified_timeframe_tasks,
        review_or_blocked_tasks=review_or_blocked_tasks,
        all_tasks=tasks,
        total_task_count=len(tasks),
        releasable_task_count=releasable_count,
        review_required_task_count=review_count,
        blocked_task_count=blocked_count,
    )

    journey_releasable = overall_safe and provenance_audit_passed and review_count == 0

    return CareJourney(
        journey_id=f"journey-{doc.document_id}",
        document_id=doc.document_id,
        patient_identifier=doc.patient_identifier,
        timeline=timeline,
        safety_status=val_report.overall_safety_status,
        patient_releasable=journey_releasable,
        requires_human_review=not journey_releasable,
        hazard_codes=val_report.hazard_codes_detected,
        clinical_disclaimer=MANDATORY_CLINICAL_DISCLAIMER,
        provenance_audit_passed=provenance_audit_passed,
    )
