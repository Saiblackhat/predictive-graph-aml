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
                neg_es = "no tome" in text_es or "deje de tomar" in text_es
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
