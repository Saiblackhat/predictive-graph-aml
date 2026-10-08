"""Comprehensive test suite for Phase 6 Patient-Friendly Language & Multilingual.

Tests:
1. Synthetic fixtures L01 - L20 end-to-end.
2. Adversarial break tests A - P.
3. Semantic fidelity audit across all items.
"""

from __future__ import annotations

import unittest

from backend.document_intelligence import run_document_intelligence_pipeline
from backend.task_orchestration import orchestrate_care_journey
from backend.patient_language import (
    ALL_PHASE6_FIXTURES,
    LanguageCode,
    PatientCarePlan,
    TransformationStatus,
    generate_patient_care_plan,
    run_care_plan_fidelity_audit,
    simplify_care_task,
    translate_patient_item_to_spanish,
)
from backend.patient_language.fixtures import (
    L01_SIMPLE_MED,
    L02_FREQUENCY_EXPANSION,
    L03_FORMULATION_PRESERVATION,
    L04_FOLLOWUP_SIMPLIFICATION,
    L05_RELATIVE_TIMEFRAME,
    L06_ABSOLUTE_DATE,
    L07_WARNING_THRESHOLD,
    L08_DECIMAL_THRESHOLD,
    L09_NEGATION_PRESERVATION,
    L10_CONDITIONAL_PRESERVATION,
    L11_AMBIGUOUS_SOURCE,
    L12_CONFLICTING_MEDICATION,
    L13_UNSUPPORTED_MEDICATION,
    L14_PROVIDER_NEUTRALITY,
    L15_TRANSLATION_SIMPLE,
    L16_TRANSLATION_NUMBERS_UNITS,
    L17_TRANSLATION_NEGATION,
    L18_TRANSLATION_WARNING,
    L19_TRANSLATION_TIMEFRAME,
    L20_SOURCE_PROVENANCE,
)


class TestPhase6PatientLanguage(unittest.TestCase):
    """Test suite for Phase 6 Patient-Friendly Language & Multilingual layer."""

    # ------------------------------------------------------------------
    # Synthetic Fixtures L01 - L20
    # ------------------------------------------------------------------
    def test_all_20_synthetic_fixtures(self) -> None:
        self.assertEqual(len(ALL_PHASE6_FIXTURES), 20)

        for fix in ALL_PHASE6_FIXTURES:
            ext = run_document_intelligence_pipeline(fix.raw_text, document_id=fix.case_id)
            journey = orchestrate_care_journey(ext)
            plan = generate_patient_care_plan(journey, target_language=LanguageCode.ES)

            self.assertEqual(
                plan.patient_releasable,
                fix.expected_patient_releasable,
                f"Fixture {fix.case_id} patient_releasable mismatch: expected {fix.expected_patient_releasable}, got {plan.patient_releasable}",
            )
            self.assertTrue(plan.fidelity_audit_passed, f"Fixture {fix.case_id} failed fidelity audit")

    def test_l01_simple_medication_simplification(self) -> None:
        ext = run_document_intelligence_pipeline(L01_SIMPLE_MED.raw_text, document_id=L01_SIMPLE_MED.case_id)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        self.assertTrue(plan.patient_releasable)
        med_item = next(it for it in plan.items if it.preserved_fields.get("medication_name") == "aspirin")
        self.assertIn("Keep taking aspirin 81 mg once a day", med_item.simplified_instruction)

    def test_l02_frequency_expansion_bid(self) -> None:
        ext = run_document_intelligence_pipeline(L02_FREQUENCY_EXPANSION.raw_text, document_id=L02_FREQUENCY_EXPANSION.case_id)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        med_item = next(it for it in plan.items if it.preserved_fields.get("medication_name") == "metformin")
        self.assertIn("twice a day", med_item.simplified_instruction)
        self.assertNotIn("BID", med_item.simplified_instruction)

    def test_l03_formulation_preservation_tartrate(self) -> None:
        ext = run_document_intelligence_pipeline(L03_FORMULATION_PRESERVATION.raw_text, document_id=L03_FORMULATION_PRESERVATION.case_id)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        med_item = next(it for it in plan.items if "metoprolol" in it.simplified_instruction)
        self.assertIn("metoprolol tartrate", med_item.simplified_instruction)

    def test_l07_warning_threshold_preservation(self) -> None:
        ext = run_document_intelligence_pipeline(L07_WARNING_THRESHOLD.raw_text, document_id=L07_WARNING_THRESHOLD.case_id)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        warn_item = next(it for it in plan.items if "weight" in it.simplified_instruction)
        self.assertIn("3 pounds", warn_item.simplified_instruction)
        self.assertIn("24 hours", warn_item.simplified_instruction)

    def test_l08_decimal_threshold_preservation(self) -> None:
        ext = run_document_intelligence_pipeline(L08_DECIMAL_THRESHOLD.raw_text, document_id=L08_DECIMAL_THRESHOLD.case_id)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        warn_item = next(it for it in plan.items if "temperature" in it.simplified_instruction)
        self.assertIn("101.5 F", warn_item.simplified_instruction)

    def test_l09_negation_preservation_stop(self) -> None:
        ext = run_document_intelligence_pipeline(L09_NEGATION_PRESERVATION.raw_text, document_id=L09_NEGATION_PRESERVATION.case_id)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        med_item = next(it for it in plan.items if "lisinopril" in it.simplified_instruction)
        self.assertIn("Do not take lisinopril", med_item.simplified_instruction)
        self.assertFalse(plan.patient_releasable)

    def test_l15_l16_translation_spanish(self) -> None:
        ext = run_document_intelligence_pipeline(L16_TRANSLATION_NUMBERS_UNITS.raw_text, document_id=L16_TRANSLATION_NUMBERS_UNITS.case_id)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext), target_language=LanguageCode.ES)
        self.assertTrue(plan.patient_releasable)
        self.assertGreater(len(plan.translations), 0)
        es_med = next(t for t in plan.translations if "metformin" in t.translated_instruction)
        self.assertIn("metformin", es_med.translated_instruction)
        self.assertIn("500 mg", es_med.translated_instruction)
        self.assertIn("dos veces al día", es_med.translated_instruction)

    # ------------------------------------------------------------------
    # Adversarial Break Tests A - P
    # ------------------------------------------------------------------
    def test_adv_a_never_lose_tartrate(self) -> None:
        raw = "Discharge Medications:\nContinue metoprolol tartrate 25 mg twice daily.\nFollow-Up Appointments:\nFollow up with Cardiology in 7 days."
        ext = run_document_intelligence_pipeline(raw)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        med_en = next(it for it in plan.items if "metoprolol" in it.simplified_instruction)
        self.assertIn("tartrate", med_en.simplified_instruction)

    def test_adv_b_never_change_frequency_twice_to_once(self) -> None:
        raw = "Discharge Medications:\nContinue metoprolol tartrate 25 mg twice daily.\nFollow-Up Appointments:\nFollow up with Cardiology in 7 days."
        ext = run_document_intelligence_pipeline(raw)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        med_en = next(it for it in plan.items if "metoprolol" in it.simplified_instruction)
        self.assertIn("twice a day", med_en.simplified_instruction)
        self.assertNotIn("once a day", med_en.simplified_instruction)

    def test_adv_c_never_turn_continue_into_stop(self) -> None:
        raw = "Discharge Medications:\nContinue aspirin 81 mg daily.\nFollow-Up Appointments:\nFollow up with Primary Care in 14 days."
        ext = run_document_intelligence_pipeline(raw)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        med_en = next(it for it in plan.items if "aspirin" in it.simplified_instruction)
        self.assertIn("Keep taking", med_en.simplified_instruction)
        self.assertNotIn("Stop taking", med_en.simplified_instruction)

    def test_adv_d_preserve_do_not_stop(self) -> None:
        raw = "Discharge Medications:\nContinue aspirin 81 mg daily.\nFollow-Up Appointments:\nFollow up with Primary Care in 14 days."
        ext = run_document_intelligence_pipeline(raw)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        med_en = next(it for it in plan.items if "aspirin" in it.simplified_instruction)
        self.assertNotIn("Stop", med_en.simplified_instruction)

    def test_adv_e_never_drop_temperature_threshold(self) -> None:
        raw = "Discharge Medications:\nContinue aspirin 81 mg daily.\nWarning Signs:\nCall if temperature is greater than 101.5 F.\nFollow-Up Appointments:\nFollow up with Primary Care in 7 days."
        ext = run_document_intelligence_pipeline(raw)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        warn = next(it for it in plan.items if "temperature" in it.simplified_instruction)
        self.assertIn("101.5 F", warn.simplified_instruction)

    def test_adv_f_never_round_decimal_weight_threshold(self) -> None:
        raw = "Discharge Medications:\nContinue furosemide 40 mg daily.\nWarning Signs:\nCall clinic if weight increases by more than 3.5 pounds in 24 hours.\nFollow-Up Appointments:\nFollow up with Cardiology in 14 days."
        ext = run_document_intelligence_pipeline(raw)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        warn = next(it for it in plan.items if "weight" in it.simplified_instruction)
        self.assertIn("3.5 pounds", warn.simplified_instruction)

    def test_adv_g_never_turn_within_into_after(self) -> None:
        raw = "Discharge Medications:\nContinue aspirin 81 mg daily.\nFollow-Up Appointments:\nFollow up with Cardiology within 7 days."
        ext = run_document_intelligence_pipeline(raw)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        fu = next(it for it in plan.items if "Cardiology" in it.simplified_instruction)
        self.assertIn("within 7 days", fu.simplified_instruction)
        self.assertNotIn("after 7 days", fu.simplified_instruction)

    def test_adv_h_never_turn_after_into_within(self) -> None:
        raw = "Discharge Medications:\nContinue aspirin 81 mg daily.\nFollow-Up Appointments:\nFollow up with Cardiology after 7 days."
        ext = run_document_intelligence_pipeline(raw)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        fu = next(it for it in plan.items if "Cardiology" in it.simplified_instruction)
        self.assertIn("after 7 days", fu.simplified_instruction)
        self.assertNotIn("within 7 days", fu.simplified_instruction)

    def test_adv_i_missing_timeframe_never_invents_date(self) -> None:
        raw = "Discharge Medications:\nContinue aspirin 81 mg daily.\nFollow-Up Appointments:\nFollow up with Cardiology for evaluation."
        ext = run_document_intelligence_pipeline(raw)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        self.assertFalse(plan.patient_releasable)
        fu = next(it for it in plan.items if "Cardiology" in it.simplified_instruction)
        self.assertIsNone(fu.timeframe_display)

    def test_adv_j_conflicting_medication_never_releasable(self) -> None:
        raw = "Hospital Course:\nContinue lisinopril 10 mg daily.\nDischarge Medications:\nStop lisinopril 10 mg daily.\nFollow-Up Appointments:\nFollow up in 7 days."
        ext = run_document_intelligence_pipeline(raw)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        self.assertFalse(plan.patient_releasable)

    def test_adv_k_ambiguous_followup_never_releasable(self) -> None:
        raw = "Discharge Medications:\nContinue aspirin 81 mg daily.\nFollow-Up Appointments:\nFollow up when convenient in future."
        ext = run_document_intelligence_pipeline(raw)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        self.assertFalse(plan.patient_releasable)

    def test_adv_l_unsupported_provider_claim_rejected(self) -> None:
        raw = "Discharge Medications:\nContinue aspirin 81 mg daily.\nFollow-Up Appointments:\nFollow up with Cardiology in 7 days."
        ext = run_document_intelligence_pipeline(raw)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext))
        for it in plan.items:
            self.assertNotIn("best", it.simplified_instruction.lower())
            self.assertNotIn("recommended", it.simplified_instruction.lower())
            self.assertNotIn("guaranteed", it.simplified_instruction.lower())

    def test_adv_m_translation_preserves_negation(self) -> None:
        item = simplify_care_task(
            task=orchestrate_care_journey(
                run_document_intelligence_pipeline("Discharge Medications:\nStop lisinopril 10 mg daily.\nFollow-Up Appointments:\nFollow up in 7 days.")
            ).timeline.all_tasks[0],
            item_index=1,
        )
        trans = translate_patient_item_to_spanish(item, 1)
        self.assertIn("No tome lisinopril", trans.translated_instruction)

    def test_adv_n_translation_preserves_numeric_dosage(self) -> None:
        ext = run_document_intelligence_pipeline("Discharge Medications:\nTake metformin 500 mg BID.\nFollow-Up Appointments:\nFollow up in 14 days.")
        plan = generate_patient_care_plan(orchestrate_care_journey(ext), target_language=LanguageCode.ES)
        es_med = next(t for t in plan.translations if "metformin" in t.translated_instruction)
        self.assertIn("500 mg", es_med.translated_instruction)

    def test_adv_o_translation_preserves_warning_threshold(self) -> None:
        ext = run_document_intelligence_pipeline(L07_WARNING_THRESHOLD.raw_text)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext), target_language=LanguageCode.ES)
        es_warn = next(t for t in plan.translations if "peso" in t.translated_instruction)
        self.assertIn("más de 3 libras en 24 horas", es_warn.translated_instruction)

    def test_adv_p_translation_preserves_tartrate_vs_succinate(self) -> None:
        raw = (
            "Discharge Medications:\n"
            "1. Continue metoprolol tartrate 25 mg twice daily.\n"
            "2. Continue metoprolol succinate 50 mg daily.\n"
            "Follow-Up Appointments:\n"
            "Follow up with Cardiology in 7 days.\n"
        )
        ext = run_document_intelligence_pipeline(raw)
        plan = generate_patient_care_plan(orchestrate_care_journey(ext), target_language=LanguageCode.ES)
        es_texts = [t.translated_instruction for t in plan.translations if "metoprolol" in t.translated_instruction]
        self.assertEqual(len(es_texts), 2)
        has_tartrate = any("tartrate" in t for t in es_texts)
        has_succinate = any("succinate" in t for t in es_texts)
        self.assertTrue(has_tartrate)
        self.assertTrue(has_succinate)


if __name__ == "__main__":
    unittest.main()
