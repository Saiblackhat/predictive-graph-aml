"""Deterministic Patient-Friendly Language Simplifier for Phase 6.

Transforms validated Phase 5 care tasks into plain, readable language
without changing clinical meaning.

CORE INVARIANTS:
1. No safety upgrade:
   patient_releasable(item) MUST NOT be True if task.patient_releasable == False.
2. Clinical fidelity:
   Every material clinical fact (name, salt/formulation, dose, frequency, route,
   action, timeframe, threshold, urgency, negation) is strictly preserved.
3. No hallucination of explanations, dates, or subjective endorsements.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from backend.schemas.models import MedicationAction, UrgencyLevel
from backend.task_orchestration.types import (
    OrchestratedTask,
    TaskCategory,
    TaskStatus,
    TimeframeType,
)
from backend.patient_language.types import (
    LanguageCode,
    PatientLanguageItem,
    TransformationStatus,
)

FREQUENCY_EXPANSIONS: dict[str, str] = {
    "bid": "twice a day",
    "b.i.d.": "twice a day",
    "tid": "three times a day",
    "t.i.d.": "three times a day",
    "qid": "four times a day",
    "q.i.d.": "four times a day",
    "qd": "once a day",
    "q.d.": "once a day",
    "once daily": "once a day",
    "daily": "once a day",
    "twice daily": "twice a day",
    "three times daily": "three times a day",
    "four times daily": "four times a day",
    "every morning": "every morning",
    "every evening": "every evening",
    "every night": "every night",
    "at bedtime": "at bedtime",
    "qhs": "at bedtime",
    "as needed": "as needed",
    "prn": "as needed",
}

ROUTE_TRANSLATIONS: dict[str, str] = {
    "by mouth": "by mouth",
    "orally": "by mouth",
    "po": "by mouth",
    "iv": "through an IV",
    "intravenously": "through an IV",
    "subcutaneously": "as an injection under the skin",
    "subq": "as an injection under the skin",
    "sublingually": "under your tongue",
    "topically": "on your skin",
    "nasally": "in your nose",
}


def expand_frequency(raw_freq: Optional[str]) -> Optional[str]:
    """Deterministically expand medical frequency abbreviation into plain language.

    Never converts 'as needed' into a scheduled frequency.
    """
    if not raw_freq:
        return None
    lower = raw_freq.strip().lower()
    for pattern, expanded in sorted(FREQUENCY_EXPANSIONS.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(rf"\b{re.escape(pattern)}\b", lower):
            # Replace the abbreviation with expanded phrase
            return re.sub(rf"\b{re.escape(pattern)}\b", expanded, lower, count=1).strip()
    return raw_freq.strip()


def expand_route(raw_route: Optional[str]) -> Optional[str]:
    """Deterministically expand medical route abbreviation into plain language."""
    if not raw_route:
        return None
    lower = raw_route.strip().lower()
    for pattern, expanded in ROUTE_TRANSLATIONS.items():
        if re.search(rf"\b{re.escape(pattern)}\b", lower):
            return expanded
    return raw_route.strip()


def simplify_care_task(task: OrchestratedTask, item_index: int) -> PatientLanguageItem:
    """Deterministically transform an OrchestratedTask into a patient-friendly item.

    Strictly preserves clinical fidelity and safety status.
    """
    meta = task.clinical_metadata
    category = task.task_category
    raw_ev = task.evidence.source_text

    review_reasons = list(task.review_reasons)
    status = task.status
    safe_to_release = task.patient_releasable and status not in (TaskStatus.NEEDS_REVIEW, TaskStatus.BLOCKED)

    simplified_title: str
    simplified_instruction: str
    preserved_fields: dict[str, Any] = {}

    if category == TaskCategory.MEDICATION:
        med_name = meta.get("medication_name", "your medicine")
        dose = meta.get("dosage")
        freq = meta.get("frequency")
        route = meta.get("route")
        action_val = meta.get("action", MedicationAction.UNKNOWN.value)

        freq_str = expand_frequency(freq)
        route_str = expand_route(route)

        dose_text = f" {dose}" if dose else ""
        freq_text = f" {freq_str}" if freq_str else ""
        route_text = f" {route_str}" if route_str else ""

        preserved_fields["medication_name"] = med_name
        preserved_fields["dosage"] = dose
        preserved_fields["frequency"] = freq
        preserved_fields["frequency_expanded"] = freq_str
        preserved_fields["route"] = route
        preserved_fields["action"] = action_val

        if action_val == MedicationAction.STOP_DISCONTINUE.value:
            simplified_title = f"Stop taking {med_name}{dose_text}"
            simplified_instruction = f"Do not take {med_name}{dose_text} anymore. Stop taking this medicine after leaving the hospital."
            preserved_fields["negation"] = "Do not take"
        elif action_val == MedicationAction.HOLD_TEMPORARY.value:
            simplified_title = f"Temporarily pause {med_name}{dose_text}"
            simplified_instruction = f"Do not take {med_name}{dose_text} right now. Temporarily pause this medicine as directed."
            preserved_fields["negation"] = "Do not take right now"
        elif action_val == MedicationAction.NEW_START.value:
            simplified_title = f"Start taking {med_name}{dose_text}"
            simplified_instruction = f"Start taking your new medicine {med_name}{dose_text}{freq_text}{route_text}."
        elif action_val == MedicationAction.DOSE_CHANGE.value:
            prev_dose = meta.get("previous_dosage")
            prev_str = f" from {prev_dose}" if prev_dose else ""
            simplified_title = f"Change your {med_name} dose to{dose_text}"
            simplified_instruction = f"Change your {med_name} dose{prev_str} to{dose_text}{freq_text}{route_text}."
            preserved_fields["previous_dosage"] = prev_dose
        elif action_val == MedicationAction.CONTINUE.value:
            simplified_title = f"Keep taking {med_name}{dose_text}"
            simplified_instruction = f"Keep taking {med_name}{dose_text}{freq_text}{route_text}."
        else:
            simplified_title = f"Review prescription for {med_name}"
            simplified_instruction = f"Talk to your doctor before taking {med_name}{dose_text}. Dosing instructions need clinician verification."
            safe_to_release = False
            review_reasons.append("Unclear medication action verb requires clinician verification.")

    elif category == TaskCategory.FOLLOW_UP:
        spec = meta.get("specialty", "Specialist")
        prov = meta.get("provider_name")
        prov_str = f" with {prov}" if prov else ""
        tf_display = task.due_date or task.due_window or ""
        tf_str = f" {tf_display}" if tf_display else ""

        simplified_title = f"See {spec}{prov_str}"
        simplified_instruction = f"Schedule and attend an appointment with {spec}{prov_str}{tf_str}."

        preserved_fields["specialty"] = spec
        preserved_fields["provider_name"] = prov
        preserved_fields["timeframe"] = tf_display
        preserved_fields["due_date"] = task.due_date

        if not tf_display:
            safe_to_release = False
            review_reasons.append("Follow-up appointment lacks a clear timeframe.")

    elif category == TaskCategory.TEST:
        test_name = meta.get("test_name", "Medical test")
        tf_display = task.due_date or task.due_window or ""
        tf_str = f" {tf_display}" if tf_display else ""

        simplified_title = f"Get lab test: {test_name}"
        simplified_instruction = f"Get this lab test completed: {test_name}{tf_str}."

        preserved_fields["test_name"] = test_name
        preserved_fields["timeframe"] = tf_display

    elif category == TaskCategory.REFERRAL:
        spec = meta.get("specialty_or_service", "Specialty service")
        prov = meta.get("provider_name")
        prov_str = f" with {prov}" if prov else ""
        tf_display = task.due_date or task.due_window or ""
        tf_str = f" {tf_display}" if tf_display else ""

        simplified_title = f"Referral: {spec}{prov_str}"
        simplified_instruction = f"Schedule an appointment for {spec}{prov_str}{tf_str}."

        preserved_fields["specialty_or_service"] = spec
        preserved_fields["timeframe"] = tf_display

    elif category == TaskCategory.WARNING_MONITORING:
        symp = meta.get("symptom_description", "warning signs")
        action = meta.get("action_to_take", "Call your doctor or clinic")
        urgency = meta.get("urgency", UrgencyLevel.UNKNOWN.value)

        simplified_title = f"Warning sign alert: {symp}"
        simplified_instruction = f"Call the clinic right away if {symp}. Action to take: {action}."

        preserved_fields["symptom_description"] = symp
        preserved_fields["action_to_take"] = action
        preserved_fields["urgency"] = urgency

        if urgency == UrgencyLevel.EMERGENT_911.value:
            simplified_instruction = f"Call 911 or go to the nearest emergency room immediately if {symp}."
            safe_to_release = False
            review_reasons.append("Urgent emergency warning sign requires clinician safety verification.")

    elif category == TaskCategory.CARE_INSTRUCTION:
        ctype = meta.get("instruction_type", "Care Instruction").replace("_", " ").title()
        simplified_title = f"{ctype}: {task.title}"
        simplified_instruction = task.description
        preserved_fields["instruction_type"] = meta.get("instruction_type")
        preserved_fields["detail"] = task.description

    else:
        simplified_title = task.title
        simplified_instruction = task.description

    # Transformation status determination
    if not safe_to_release:
        if task.status == TaskStatus.BLOCKED:
            t_status = TransformationStatus.BLOCKED
        else:
            t_status = TransformationStatus.NEEDS_REVIEW
        patient_releasable = False
        requires_human_review = True
    else:
        t_status = TransformationStatus.FAITHFUL
        patient_releasable = True
        requires_human_review = False

    return PatientLanguageItem(
        item_id=f"lang-{item_index:03d}",
        source_task_id=task.task_id,
        source_item_id=task.source_item_id,
        category=category,
        simplified_title=simplified_title,
        simplified_instruction=simplified_instruction,
        language=LanguageCode.EN,
        transformation_status=t_status,
        patient_releasable=patient_releasable,
        requires_human_review=requires_human_review,
        review_reasons=review_reasons,
        evidence=task.evidence,
        original_source_text=raw_ev,
        timeframe_display=task.due_date or task.due_window,
        preserved_fields=preserved_fields,
    )
