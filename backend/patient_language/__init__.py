"""CAREBRIDGE Phase 6 — Patient-Friendly Language & Multilingual Layer.

Transforms validated Phase 5 care tasks into plain, readable,
accessible language and controlled translations with strict semantic fidelity.
"""

from __future__ import annotations

from typing import Optional

from backend.schemas.models import MANDATORY_CLINICAL_DISCLAIMER, SafetyStatus
from backend.task_orchestration.types import CareJourney, TaskStatus
from backend.patient_language.fidelity import (
    FidelityAuditReport,
    audit_patient_language_fidelity,
    run_care_plan_fidelity_audit,
)
from backend.patient_language.fixtures import (
    ALL_PHASE6_FIXTURES,
    SyntheticLanguageFixture,
)
from backend.patient_language.simplifier import (
    expand_frequency,
    expand_route,
    simplify_care_task,
)
from backend.patient_language.translator import (
    translate_patient_item_to_spanish,
    translate_timeframe_to_spanish,
)
from backend.patient_language.types import (
    LanguageCode,
    PatientCarePlan,
    PatientLanguageItem,
    TransformationStatus,
    TranslatedPatientLanguageItem,
)


def generate_patient_care_plan(
    journey: CareJourney,
    target_language: Optional[LanguageCode] = LanguageCode.ES,
) -> PatientCarePlan:
    """Generate verified, patient-friendly care plan with optional controlled translation."""
    items: list[PatientLanguageItem] = []
    translations: list[TranslatedPatientLanguageItem] = []

    for idx, task in enumerate(journey.timeline.all_tasks, start=1):
        item = simplify_care_task(task, idx)
        items.append(item)

        if target_language == LanguageCode.ES:
            trans_item = translate_patient_item_to_spanish(item, idx)
            translations.append(trans_item)

    fidelity_report = run_care_plan_fidelity_audit(items, translations)

    releasable_count = sum(1 for it in items if it.patient_releasable)
    review_count = sum(1 for it in items if it.requires_human_review)
    blocked_count = sum(1 for it in items if it.transformation_status == TransformationStatus.BLOCKED)

    overall_releasable = (
        journey.patient_releasable
        and fidelity_report.passed
        and review_count == 0
        and blocked_count == 0
    )

    return PatientCarePlan(
        plan_id=f"plan-{journey.document_id}",
        document_id=journey.document_id,
        patient_identifier=journey.patient_identifier,
        language=LanguageCode.EN,
        items=items,
        translations=translations,
        total_items=len(items),
        releasable_items=releasable_count,
        review_required_items=review_count,
        blocked_items=blocked_count,
        overall_safety_status=journey.safety_status,
        patient_releasable=overall_releasable,
        requires_human_review=not overall_releasable,
        clinical_disclaimer=MANDATORY_CLINICAL_DISCLAIMER,
        fidelity_audit_passed=fidelity_report.passed,
    )


__all__ = [
    "generate_patient_care_plan",
    "simplify_care_task",
    "translate_patient_item_to_spanish",
    "run_care_plan_fidelity_audit",
    "audit_patient_language_fidelity",
    "expand_frequency",
    "expand_route",
    "translate_timeframe_to_spanish",
    "LanguageCode",
    "TransformationStatus",
    "PatientLanguageItem",
    "TranslatedPatientLanguageItem",
    "PatientCarePlan",
    "FidelityAuditReport",
    "ALL_PHASE6_FIXTURES",
    "SyntheticLanguageFixture",
]
