"""Semantic Fidelity Audit Engine for Phase 6.

Validates that patient-friendly simplification and translation preserve every
clinically material fact without drift or loss.

CHECKS:
1. Medication identity & formulation (tartrate vs succinate).
2. Exact numeric dosage and unit.
3. Frequency expansion semantics (BID -> twice a day).
4. Clinical action & negation ('Do not take' for STOP).
5. Timeframe operators ('within' != 'after').
6. Warning threshold numbers, units, and inequality operators.
7. Specialty neutrality (no 'best' or 'recommended').
"""

from __future__ import annotations

import re
from typing import NamedTuple

from backend.schemas.models import MedicationAction
from backend.task_orchestration.types import TaskCategory
from backend.patient_language.types import PatientLanguageItem, TranslatedPatientLanguageItem


class FidelityAuditFinding(NamedTuple):
    item_id: str
    field_checked: str
    expected_value: str
    actual_value: str
    passed: bool
    details: str


class FidelityAuditReport(NamedTuple):
    total_checks: int
    passed_checks: int
    failed_checks: int
    findings: list[FidelityAuditFinding]

    @property
    def passed(self) -> bool:
        return self.failed_checks == 0


def audit_patient_language_fidelity(
    item: PatientLanguageItem,
    translation: TranslatedPatientLanguageItem | None = None,
) -> list[FidelityAuditFinding]:
    """Audit single patient language item and optional translation for semantic fidelity."""
    findings: list[FidelityAuditFinding] = []
    meta = item.preserved_fields
    text_en = f"{item.simplified_title} {item.simplified_instruction}".lower()
    text_es = f"{translation.translated_title} {translation.translated_instruction}".lower() if translation else ""

    # 1. Medication Identity & Formulation Preservation
    if item.category == TaskCategory.MEDICATION:
        med_name = meta.get("medication_name", "").lower()
        if med_name:
            # Check English
            en_has_med = med_name in text_en
            findings.append(
                FidelityAuditFinding(
                    item_id=item.item_id,
                    field_checked="medication_name_en",
                    expected_value=med_name,
                    actual_value=item.simplified_instruction,
                    passed=en_has_med,
                    details=f"Medication name '{med_name}' in English simplification",
                )
            )
            # Check Translation (Medication name must NOT be translated)
            if translation:
                es_has_med = med_name in text_es
                findings.append(
                    FidelityAuditFinding(
                        item_id=translation.item_id,
                        field_checked="medication_name_es",
                        expected_value=med_name,
                        actual_value=translation.translated_instruction,
                        passed=es_has_med,
                        details=f"Medication name '{med_name}' in Spanish translation",
                    )
                )

        # Formulation Salt Distinction
        for salt in ("tartrate", "succinate"):
            if salt in med_name:
                salt_in_en = salt in text_en
                findings.append(
                    FidelityAuditFinding(
                        item_id=item.item_id,
                        field_checked=f"formulation_{salt}_en",
                        expected_value=salt,
                        actual_value=item.simplified_instruction,
                        passed=salt_in_en,
                        details=f"Formulation modifier '{salt}' preserved in English",
                    )
                )
                if translation:
                    salt_in_es = salt in text_es
                    findings.append(
                        FidelityAuditFinding(
                            item_id=translation.item_id,
                            field_checked=f"formulation_{salt}_es",
                            expected_value=salt,
                            actual_value=translation.translated_instruction,
                            passed=salt_in_es,
                            details=f"Formulation modifier '{salt}' preserved in Spanish",
                        )
                    )

        # Dosage Preservation
        dose = meta.get("dosage")
        if dose:
            dose_lower = dose.lower()
            findings.append(
                FidelityAuditFinding(
                    item_id=item.item_id,
                    field_checked="dosage_en",
                    expected_value=dose,
                    actual_value=item.simplified_instruction,
                    passed=dose_lower in text_en,
                    details=f"Dosage '{dose}' in English",
                )
            )
            if translation:
                findings.append(
                    FidelityAuditFinding(
                        item_id=translation.item_id,
                        field_checked="dosage_es",
                        expected_value=dose,
                        actual_value=translation.translated_instruction,
                        passed=dose_lower in text_es,
                        details=f"Dosage '{dose}' in Spanish",
                    )
                )

        # Frequency Preservation in Spanish (scheduled active medications only)
        freq_exp = meta.get("frequency_expanded")
        action_val = meta.get("action")
        is_scheduled = (
            action_val in (
                MedicationAction.CONTINUE.value,
                MedicationAction.NEW_START.value,
                MedicationAction.DOSE_CHANGE.value,
            )
            and item.patient_releasable
        )
        if freq_exp and translation and is_scheduled:
            freq_lower = freq_exp.lower()
            expected_es_freq = None
            if "twice a day" in freq_lower:
                expected_es_freq = "dos veces al día"
            elif "three times a day" in freq_lower:
                expected_es_freq = "tres veces al día"
            elif "four times a day" in freq_lower:
                expected_es_freq = "cuatro veces al día"
            elif "once a day" in freq_lower:
                expected_es_freq = "una vez al día"
            elif "as needed" in freq_lower:
                expected_es_freq = "según sea necesario"

            if expected_es_freq:
                findings.append(
                    FidelityAuditFinding(
                        item_id=translation.item_id,
                        field_checked="frequency_es",
                        expected_value=expected_es_freq,
                        actual_value=translation.translated_instruction,
                        passed=expected_es_freq in text_es,
                        details=f"Frequency '{expected_es_freq}' preserved in Spanish",
                    )
                )

        # Route Preservation in Spanish (scheduled active medications only)
        route = meta.get("route")
        if route and translation and is_scheduled:
            route_lower = route.lower()
            expected_es_route = None
            if route_lower in ("by mouth", "orally", "po"):
                expected_es_route = "por vía oral"
            elif route_lower in ("iv", "intravenously"):
                expected_es_route = "por vía intravenosa"
            elif route_lower in ("sublingually", "sublingual"):
                expected_es_route = "debajo de la lengua"

            if expected_es_route:
                findings.append(
                    FidelityAuditFinding(
                        item_id=translation.item_id,
                        field_checked="route_es",
                        expected_value=expected_es_route,
                        actual_value=translation.translated_instruction,
                        passed=expected_es_route in text_es,
                        details=f"Route '{expected_es_route}' preserved in Spanish",
                    )
                )

        # Action & Negation
        action = meta.get("action")
        if action == "STOP_DISCONTINUE":
            neg_en = "do not take" in text_en or "stop taking" in text_en
            findings.append(
                FidelityAuditFinding(
                    item_id=item.item_id,
                    field_checked="negation_action_en",
                    expected_value="Stop / Do not take",
                    actual_value=item.simplified_instruction,
                    passed=neg_en,
                    details="Negation preserved for STOP action in English",
                )
            )
            if translation:
                neg_es = ("no tome" in text_es or "deje de tomar" in text_es) and "continúe tomando" not in text_es
                findings.append(
                    FidelityAuditFinding(
                        item_id=translation.item_id,
                        field_checked="negation_action_es",
                        expected_value="No tome / Deje de tomar",
                        actual_value=translation.translated_instruction,
                        passed=neg_es,
                        details="Negation preserved for STOP action in Spanish",
                    )
                )
        elif action == "CONTINUE" and translation:
            cont_es = "continúe tomando" in text_es and "no tome" not in text_es
            findings.append(
                FidelityAuditFinding(
                    item_id=translation.item_id,
                    field_checked="continue_action_es",
                    expected_value="Continúe tomando",
                    actual_value=translation.translated_instruction,
                    passed=cont_es,
                    details="Continuation preserved for CONTINUE action in Spanish",
                )
            )

    # 2. Timeframe Preservation & Operator Distinction
    tf_str = meta.get("timeframe", "")
    if tf_str:
        tf_lower = tf_str.lower()
        if "within" in tf_lower:
            findings.append(
                FidelityAuditFinding(
                    item_id=item.item_id,
                    field_checked="timeframe_operator_within_en",
                    expected_value="within",
                    actual_value=text_en,
                    passed="within" in text_en and "after" not in text_en,
                    details="Timeframe 'within' not converted to 'after' in English",
                )
            )
            if translation:
                findings.append(
                    FidelityAuditFinding(
                        item_id=translation.item_id,
                        field_checked="timeframe_operator_within_es",
                        expected_value="dentro de",
                        actual_value=text_es,
                        passed="dentro de" in text_es and "después de" not in text_es,
                        details="Timeframe 'within' translated as 'dentro de' in Spanish",
                    )
                )
        elif "after" in tf_lower:
            findings.append(
                FidelityAuditFinding(
                    item_id=item.item_id,
                    field_checked="timeframe_operator_after_en",
                    expected_value="after",
                    actual_value=text_en,
                    passed="after" in text_en and "within" not in text_en,
                    details="Timeframe 'after' not converted to 'within' in English",
                )
            )
            if translation:
                findings.append(
                    FidelityAuditFinding(
                        item_id=translation.item_id,
                        field_checked="timeframe_operator_after_es",
                        expected_value="después de",
                        actual_value=text_es,
                        passed="después de" in text_es and "dentro de" not in text_es,
                        details="Timeframe 'after' translated as 'después de' in Spanish",
                    )
                )

    # 3. Warning Thresholds & Inequalities
    if item.category == TaskCategory.WARNING_MONITORING:
        symp = meta.get("symptom_description", "")
        # Extract any numeric threshold like "3 pounds", "3.5 pounds", "101.5 F"
        m_thresh = re.search(r"(\d+(?:\.\d+)?)\s*(pounds?|f|degrees?|c)", symp, re.IGNORECASE)
        if m_thresh:
            num = m_thresh.group(1)
            findings.append(
                FidelityAuditFinding(
                    item_id=item.item_id,
                    field_checked="warning_threshold_number_en",
                    expected_value=num,
                    actual_value=text_en,
                    passed=num in text_en,
                    details=f"Warning threshold numeric value '{num}' preserved in English",
                )
            )
            if translation:
                findings.append(
                    FidelityAuditFinding(
                        item_id=translation.item_id,
                        field_checked="warning_threshold_number_es",
                        expected_value=num,
                        actual_value=text_es,
                        passed=num in text_es,
                        details=f"Warning threshold numeric value '{num}' preserved in Spanish",
                    )
                )

        # Inequality semantics in Spanish
        if "more than" in symp.lower() and translation:
            findings.append(
                FidelityAuditFinding(
                    item_id=translation.item_id,
                    field_checked="inequality_more_than_es",
                    expected_value="más de",
                    actual_value=text_es,
                    passed="más de" in text_es and "menos de" not in text_es,
                    details="Inequality 'more than' strictly translated as 'más de' in Spanish",
                )
            )
        elif "greater than" in symp.lower() and translation:
            findings.append(
                FidelityAuditFinding(
                    item_id=translation.item_id,
                    field_checked="inequality_greater_than_es",
                    expected_value="mayor de",
                    actual_value=text_es,
                    passed="mayor de" in text_es,
                    details="Inequality 'greater than' strictly translated as 'mayor de' in Spanish",
                )
            )

    # 4. Specialty Neutrality
    for prohibited in ("best", "recommended", "top-rated", "guaranteed"):
        findings.append(
            FidelityAuditFinding(
                item_id=item.item_id,
                field_checked=f"neutrality_{prohibited}_en",
                expected_value=f"no '{prohibited}'",
                actual_value=text_en,
                passed=prohibited not in text_en,
                details=f"No subjective claim '{prohibited}' in English",
            )
        )
        if translation:
            prohibited_es = "mejor" if prohibited == "best" else ("recomendado" if prohibited == "recommended" else "garantizado")
            findings.append(
                FidelityAuditFinding(
                    item_id=translation.item_id,
                    field_checked=f"neutrality_{prohibited_es}_es",
                    expected_value=f"no '{prohibited_es}'",
                    actual_value=text_es,
                    passed=prohibited_es not in text_es,
                    details=f"No subjective claim '{prohibited_es}' in Spanish",
                )
            )

    return findings


def run_care_plan_fidelity_audit(
    items: list[PatientLanguageItem],
    translations: list[TranslatedPatientLanguageItem],
) -> FidelityAuditReport:
    """Run comprehensive semantic fidelity audit on all items in care plan."""
    trans_map = {t.source_language_item_id: t for t in translations}
    all_findings: list[FidelityAuditFinding] = []

    for item in items:
        trans = trans_map.get(item.item_id)
        findings = audit_patient_language_fidelity(item, trans)
        all_findings.extend(findings)

    passed = sum(1 for f in all_findings if f.passed)
    failed = sum(1 for f in all_findings if not f.passed)

    return FidelityAuditReport(
        total_checks=len(all_findings),
        passed_checks=passed,
        failed_checks=failed,
        findings=all_findings,
    )
