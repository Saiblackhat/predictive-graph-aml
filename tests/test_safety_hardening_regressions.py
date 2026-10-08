"""Phase 2 Focused Regression Test Suite for CAREBRIDGE Safety Hardening.

Covers all 22 mandatory general safety invariants (Sections A through L)
without modifying the frozen independent adversarial test suite.
"""

from __future__ import annotations

import unittest
from pydantic import ValidationError

from backend.fixtures.discharge_cases import build_normal_case
from backend.safety.validator import (
    is_patient_releasable,
    normalize_timeframe,
    validate_discharge_payload,
)
from backend.schemas.models import (
    SPAN_NOT_IMPLEMENTED_NOTE,
    CareTaskItem,
    DischargeExtractionPayload,
    FollowUpInstruction,
    HazardCode,
    MedicationAction,
    MedicationInstruction,
    ReviewDecisionStatus,
    SafetyStatus,
    SourceEvidence,
    SyntheticProviderMatch,
    UrgencyLevel,
    WarningSignInstruction,
)


def _make_clean_med_payload() -> DischargeExtractionPayload:
    raw = "Continue Atorvastatin 40 mg by mouth once daily."
    return DischargeExtractionPayload(
        case_id="REG-CLEAN",
        patient_identifier="PT-REG",
        raw_discharge_text=raw,
        medications=[
            MedicationInstruction(
                id="med-1",
                medication_name="Atorvastatin",
                action=MedicationAction.CONTINUE,
                dosage="40 mg",
                frequency="once daily",
                route="by mouth",
                confidence=0.98,
                source_evidence=SourceEvidence(
                    source_text=raw,
                    source_section="Discharge Medications",
                ),
                patient_friendly_summary="Continue Atorvastatin 40 mg by mouth once daily.",
            )
        ],
    )


class TestSafetyHardeningRegressions(unittest.TestCase):
    """Comprehensive regression tests for Phase 1 safety hardening (22 scenarios)."""

    def test_01_unsupported_dosage(self) -> None:
        raw = "Continue aspirin daily."
        payload = DischargeExtractionPayload(
            case_id="REG-01",
            patient_identifier="PT-01",
            raw_discharge_text=raw,
            medications=[
                MedicationInstruction(
                    id="med-01",
                    medication_name="aspirin",
                    action=MedicationAction.CONTINUE,
                    dosage="81 mg",  # Not in source text!
                    frequency="daily",
                    confidence=0.95,
                    source_evidence=SourceEvidence(source_text=raw),
                )
            ],
        )
        report = validate_discharge_payload(payload)
        self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
        self.assertTrue(report.requires_human_review)
        self.assertIn(HazardCode.UNSUPPORTED_FIELD_VALUE, report.hazard_codes_detected)
        self.assertFalse(is_patient_releasable(report))

    def test_02_unsupported_route(self) -> None:
        raw = "Continue Metoprolol 25 mg twice daily."
        payload = DischargeExtractionPayload(
            case_id="REG-02",
            patient_identifier="PT-02",
            raw_discharge_text=raw,
            medications=[
                MedicationInstruction(
                    id="med-02",
                    medication_name="Metoprolol",
                    action=MedicationAction.CONTINUE,
                    dosage="25 mg",
                    frequency="twice daily",
                    route="by mouth",  # Not in source text!
                    confidence=0.95,
                    source_evidence=SourceEvidence(source_text=raw),
                )
            ],
        )
        report = validate_discharge_payload(payload)
        self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
        self.assertIn(HazardCode.UNSUPPORTED_FIELD_VALUE, report.hazard_codes_detected)
        self.assertFalse(is_patient_releasable(report))

    def test_03_unsupported_frequency(self) -> None:
        raw = "Continue Lisinopril 10 mg by mouth once daily."
        payload = DischargeExtractionPayload(
            case_id="REG-03",
            patient_identifier="PT-03",
            raw_discharge_text=raw,
            medications=[
                MedicationInstruction(
                    id="med-03",
                    medication_name="Lisinopril",
                    action=MedicationAction.CONTINUE,
                    dosage="10 mg",
                    frequency="twice daily",  # Source says once daily!
                    route="by mouth",
                    confidence=0.95,
                    source_evidence=SourceEvidence(source_text=raw),
                )
            ],
        )
        report = validate_discharge_payload(payload)
        self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
        self.assertIn(HazardCode.UNSUPPORTED_FIELD_VALUE, report.hazard_codes_detected)
        self.assertFalse(is_patient_releasable(report))

    def test_04_unsupported_medication_name(self) -> None:
        raw = "Continue Atorvastatin 40 mg by mouth once daily."
        payload = DischargeExtractionPayload(
            case_id="REG-04",
            patient_identifier="PT-04",
            raw_discharge_text=raw,
            medications=[
                MedicationInstruction(
                    id="med-04",
                    medication_name="Warfarin",  # Source is Atorvastatin!
                    action=MedicationAction.CONTINUE,
                    dosage="40 mg",
                    frequency="once daily",
                    route="by mouth",
                    confidence=0.95,
                    source_evidence=SourceEvidence(source_text=raw),
                )
            ],
        )
        report = validate_discharge_payload(payload)
        self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
        self.assertIn(HazardCode.UNSUPPORTED_FIELD_VALUE, report.hazard_codes_detected)
        self.assertIn(HazardCode.MISSING_SOURCE_EVIDENCE, report.hazard_codes_detected)
        self.assertFalse(is_patient_releasable(report))

    def test_05_unsupported_patient_friendly_text(self) -> None:
        raw = "Continue Atorvastatin 40 mg once daily."
        # Free text invents 80 mg, by mouth, twice daily, and Lisinopril
        payload = DischargeExtractionPayload(
            case_id="REG-05",
            patient_identifier="PT-05",
            raw_discharge_text=raw,
            medications=[
                MedicationInstruction(
                    id="med-05",
                    medication_name="Atorvastatin",
                    action=MedicationAction.CONTINUE,
                    dosage="40 mg",
                    frequency="once daily",
                    route=None,
                    confidence=0.95,
                    source_evidence=SourceEvidence(source_text=raw),
                    patient_friendly_summary=(
                        "Take Atorvastatin 80 mg and Lisinopril by mouth twice daily."
                    ),
                )
            ],
        )
        report = validate_discharge_payload(payload)
        self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
        self.assertIn(HazardCode.UNSUPPORTED_FIELD_VALUE, report.hazard_codes_detected)
        self.assertFalse(is_patient_releasable(report))

    def test_06_unrelated_source_evidence(self) -> None:
        raw = (
            "Continue Atorvastatin 40 mg by mouth once daily.\n"
            "Follow up with Cardiology within 7 days."
        )
        payload = DischargeExtractionPayload(
            case_id="REG-06",
            patient_identifier="PT-06",
            raw_discharge_text=raw,
            medications=[
                MedicationInstruction(
                    id="med-06",
                    medication_name="Atorvastatin",
                    action=MedicationAction.CONTINUE,
                    dosage="40 mg",
                    frequency="once daily",
                    route="by mouth",
                    confidence=0.96,
                    # Points to the follow-up sentence instead of the medication sentence
                    source_evidence=SourceEvidence(
                        source_text="Follow up with Cardiology within 7 days."
                    ),
                )
            ],
            follow_ups=[
                FollowUpInstruction(
                    id="fu-06",
                    specialty="Cardiology",
                    timeframe="within 7 days",
                    confidence=0.96,
                    # Points to the medication sentence instead of the follow-up sentence
                    source_evidence=SourceEvidence(
                        source_text="Continue Atorvastatin 40 mg by mouth once daily."
                    ),
                )
            ],
        )
        report = validate_discharge_payload(payload)
        self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
        self.assertIn(HazardCode.MISSING_SOURCE_EVIDENCE, report.hazard_codes_detected)
        self.assertIn(HazardCode.UNSUPPORTED_FIELD_VALUE, report.hazard_codes_detected)
        self.assertFalse(is_patient_releasable(report))

    def test_07_omitted_medication_switch(self) -> None:
        raw = (
            "Continue Atorvastatin 40 mg by mouth once daily.\n"
            "Switch from Cozaar to valsartan 80 mg daily."
        )
        payload = _make_clean_med_payload()
        payload.raw_discharge_text = raw
        report = validate_discharge_payload(payload)
        self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
        self.assertIn(HazardCode.OMITTED_INSTRUCTION, report.hazard_codes_detected)
        self.assertIn(HazardCode.INCOMPLETE_EXTRACTION, report.hazard_codes_detected)
        self.assertFalse(is_patient_releasable(report))

    def test_08_omitted_stop_hold_start(self) -> None:
        for omitted_line in (
            "Stop taking ibuprofen immediately.",
            "Hold Metformin for 48 hours after contrast.",
            "Start new Eliquis 5 mg twice daily.",
            "Taper prednisone by 5 mg every 3 days.",
            "Withhold evening insulin dose if NPO.",
            "Avoid all NSAIDs.",
        ):
            payload = _make_clean_med_payload()
            payload.raw_discharge_text = (
                f"{payload.raw_discharge_text}\n{omitted_line}"
            )
            report = validate_discharge_payload(payload)
            self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
            self.assertIn(HazardCode.OMITTED_INSTRUCTION, report.hazard_codes_detected)
            self.assertIn(HazardCode.INCOMPLETE_EXTRACTION, report.hazard_codes_detected)
            self.assertFalse(is_patient_releasable(report))

    def test_09_omitted_follow_up(self) -> None:
        payload = _make_clean_med_payload()
        payload.raw_discharge_text = (
            "Continue Atorvastatin 40 mg by mouth once daily.\n"
            "Follow up with Nephrology clinic within 14 days for renal function check."
        )
        report = validate_discharge_payload(payload)
        self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
        self.assertIn(HazardCode.OMITTED_INSTRUCTION, report.hazard_codes_detected)
        self.assertIn(HazardCode.INCOMPLETE_EXTRACTION, report.hazard_codes_detected)
        self.assertFalse(is_patient_releasable(report))

    def test_10_omitted_test(self) -> None:
        payload = _make_clean_med_payload()
        payload.raw_discharge_text = (
            "Continue Atorvastatin 40 mg by mouth once daily.\n"
            "Repeat BMP and potassium check in 3 days."
        )
        report = validate_discharge_payload(payload)
        self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
        self.assertIn(HazardCode.OMITTED_INSTRUCTION, report.hazard_codes_detected)
        self.assertIn(HazardCode.INCOMPLETE_EXTRACTION, report.hazard_codes_detected)
        self.assertFalse(is_patient_releasable(report))

    def test_11_omitted_warning_sign(self) -> None:
        payload = _make_clean_med_payload()
        payload.raw_discharge_text = (
            "Continue Atorvastatin 40 mg by mouth once daily.\n"
            "Call 911 immediately if you develop chest pain or shortness of breath."
        )
        report = validate_discharge_payload(payload)
        self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
        self.assertIn(HazardCode.OMITTED_INSTRUCTION, report.hazard_codes_detected)
        self.assertIn(HazardCode.INCOMPLETE_EXTRACTION, report.hazard_codes_detected)
        self.assertFalse(is_patient_releasable(report))

    def test_12_brand_generic_conflict(self) -> None:
        raw = "Continue Lasix 20 mg daily. STOP Furosemide 20 mg daily."
        payload = DischargeExtractionPayload(
            case_id="REG-12",
            patient_identifier="PT-12",
            raw_discharge_text=raw,
            medications=[
                MedicationInstruction(
                    id="med-lasix",
                    medication_name="Lasix",
                    action=MedicationAction.CONTINUE,
                    dosage="20 mg",
                    frequency="daily",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Continue Lasix 20 mg daily."
                    ),
                ),
                MedicationInstruction(
                    id="med-furosemide",
                    medication_name="Furosemide",
                    action=MedicationAction.STOP_DISCONTINUE,
                    dosage="20 mg",
                    frequency="daily",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="STOP Furosemide 20 mg daily."
                    ),
                ),
            ],
        )
        report = validate_discharge_payload(payload)
        self.assertEqual(report.overall_safety_status, SafetyStatus.CONFLICT_DETECTED)
        self.assertIn(HazardCode.CONFLICTING_INSTRUCTIONS, report.hazard_codes_detected)
        self.assertFalse(is_patient_releasable(report))

    def test_13_conflict_item_level_gating(self) -> None:
        raw = "Continue Prinivil 10 mg daily. STOP Lisinopril 10 mg daily."
        payload = DischargeExtractionPayload(
            case_id="REG-13",
            patient_identifier="PT-13",
            raw_discharge_text=raw,
            medications=[
                MedicationInstruction(
                    id="med-prinivil",
                    medication_name="Prinivil",
                    action=MedicationAction.CONTINUE,
                    dosage="10 mg",
                    frequency="daily",
                    confidence=0.98,
                    source_evidence=SourceEvidence(
                        source_text="Continue Prinivil 10 mg daily."
                    ),
                ),
                MedicationInstruction(
                    id="med-lisinopril",
                    medication_name="Lisinopril",
                    action=MedicationAction.STOP_DISCONTINUE,
                    dosage="10 mg",
                    frequency="daily",
                    confidence=0.98,
                    source_evidence=SourceEvidence(
                        source_text="STOP Lisinopril 10 mg daily."
                    ),
                ),
            ],
        )
        report = validate_discharge_payload(payload)
        # EVERY item in the conflict group must be CONFLICT_DETECTED and PENDING_HUMAN_REVIEW
        for med in report.validated_payload.medications:
            self.assertEqual(med.safety_status, SafetyStatus.CONFLICT_DETECTED)
            self.assertEqual(
                med.review_status, ReviewDecisionStatus.PENDING_HUMAN_REVIEW
            )
            self.assertNotEqual(med.review_status, ReviewDecisionStatus.AUTO_APPROVED)

    def test_14_timeframe_1_week_equals_7_days(self) -> None:
        self.assertEqual(
            normalize_timeframe("within 1 week"),
            normalize_timeframe("within 7 days"),
        )
        self.assertEqual(
            normalize_timeframe("within 2 weeks"),
            normalize_timeframe("within 14 days"),
        )
        self.assertEqual(
            normalize_timeframe("within 24 hours"),
            normalize_timeframe("within 1 day"),
        )

    def test_15_timeframe_within_7_days_not_equal_after_7_days_and_months_distinct(
        self,
    ) -> None:
        self.assertNotEqual(
            normalize_timeframe("within 7 days"),
            normalize_timeframe("after 7 days"),
        )
        self.assertNotEqual(
            normalize_timeframe("within 1 month"),
            normalize_timeframe("within 30 days"),
        )
        self.assertIsNone(normalize_timeframe("weekly"))
        self.assertIsNone(normalize_timeframe("daily"))
        self.assertIsNone(normalize_timeframe("soon"))

        raw = (
            "Follow up with Cardiology within 7 days.\n"
            "Follow up with Cardiology after 7 days."
        )
        payload = DischargeExtractionPayload(
            case_id="REG-15",
            patient_identifier="PT-15",
            raw_discharge_text=raw,
            follow_ups=[
                FollowUpInstruction(
                    id="fu-within",
                    specialty="Cardiology",
                    timeframe="within 7 days",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Follow up with Cardiology within 7 days."
                    ),
                ),
                FollowUpInstruction(
                    id="fu-after",
                    specialty="Cardiology",
                    timeframe="after 7 days",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Follow up with Cardiology after 7 days."
                    ),
                ),
            ],
        )
        report = validate_discharge_payload(payload)
        self.assertEqual(report.overall_safety_status, SafetyStatus.CONFLICT_DETECTED)
        self.assertIn(HazardCode.CONFLICTING_INSTRUCTIONS, report.hazard_codes_detected)

    def test_16_duplicate_item_flagged_and_preserved(self) -> None:
        payload = _make_clean_med_payload()
        dup = payload.medications[0].model_copy(deep=True)
        dup.id = "med-2-dup-content"
        payload.medications.append(dup)

        report = validate_discharge_payload(payload)
        self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
        self.assertTrue(report.requires_human_review)
        self.assertIn(HazardCode.DUPLICATE_ITEM, report.hazard_codes_detected)
        # Audit trail preserved
        self.assertEqual(len(report.validated_payload.medications), 2)
        self.assertFalse(is_patient_releasable(report))

    def test_17_duplicate_id_with_different_content_is_error(self) -> None:
        raw = "Continue Atorvastatin 40 mg once daily. Continue Amlodipine 5 mg once daily."
        with self.assertRaises(ValidationError):
            DischargeExtractionPayload(
                case_id="REG-17",
                patient_identifier="PT-17",
                raw_discharge_text=raw,
                medications=[
                    MedicationInstruction(
                        id="same-id-1",
                        medication_name="Atorvastatin",
                        action=MedicationAction.CONTINUE,
                        dosage="40 mg",
                        frequency="once daily",
                        confidence=0.95,
                        source_evidence=SourceEvidence(
                            source_text="Continue Atorvastatin 40 mg once daily."
                        ),
                    ),
                    MedicationInstruction(
                        id="same-id-1",
                        medication_name="Amlodipine",
                        action=MedicationAction.CONTINUE,
                        dosage="5 mg",
                        frequency="once daily",
                        confidence=0.95,
                        source_evidence=SourceEvidence(
                            source_text="Continue Amlodipine 5 mg once daily."
                        ),
                    ),
                ],
            )

    def test_18_provider_claim_recommended_and_other_evaluative_terms(self) -> None:
        for forbidden_phrase in (
            "Dr. Smith is our recommended specialist.",
            "We recommend this clinic for you.",
            "Hospital recommends this doctor.",
            "Preferred cardiology provider.",
            "Ideal doctor for your condition.",
            "Suitable specialist for your heart.",
            "Trusted physician in the network.",
            "Leading cardiologist in the city.",
            "Appointments available now.",
            "Currently accepting patients.",
        ):
            payload = _make_clean_med_payload()
            payload.provider_matches = [
                SyntheticProviderMatch(
                    provider_id="prov-reg-18",
                    name="Dr. Smith",
                    specialty="Cardiology",
                    facility="Metro Clinic (Synthetic)",
                    distance_miles=1.5,
                    accepts_new_patients_listed=True,
                    claim_text=forbidden_phrase,
                )
            ]
            report = validate_discharge_payload(payload)
            self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
            self.assertIn(
                HazardCode.UNSUPPORTED_PROVIDER_CLAIM, report.hazard_codes_detected
            )
            self.assertFalse(is_patient_releasable(report))

    def test_19_provider_claim_best_top_and_guaranteed(self) -> None:
        for forbidden_phrase in (
            "Best cardiologist in the region.",
            "Best match for your care plan.",
            "Top specialist in heart failure.",
            "Highly rated clinic.",
            "Most qualified provider.",
            "Guaranteed appointment tomorrow.",
        ):
            payload = _make_clean_med_payload()
            payload.provider_matches = [
                SyntheticProviderMatch(
                    provider_id="prov-reg-19",
                    name="Dr. Smith",
                    specialty="Cardiology",
                    facility="Metro Clinic (Synthetic)",
                    distance_miles=1.5,
                    accepts_new_patients_listed=True,
                    claim_text=forbidden_phrase,
                )
            ]
            report = validate_discharge_payload(payload)
            self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
            self.assertIn(
                HazardCode.UNSUPPORTED_PROVIDER_CLAIM, report.hazard_codes_detected
            )
            self.assertFalse(is_patient_releasable(report))

    def test_20_negation_safe_and_uncertain_negation_escalated(self) -> None:
        for negated_sentence in (
            "No change to medication is required.",
            "Medication remains unchanged.",
            "Do not stop taking the medication.",
            "No new medications.",
        ):
            raw = f"{negated_sentence} Continue Atorvastatin 40 mg by mouth once daily."
            payload = DischargeExtractionPayload(
                case_id="REG-20-SAFE",
                patient_identifier="PT-20",
                raw_discharge_text=raw,
                medications=[
                    MedicationInstruction(
                        id="med-20",
                        medication_name="Atorvastatin",
                        action=MedicationAction.CONTINUE,
                        dosage="40 mg",
                        frequency="once daily",
                        route="by mouth",
                        confidence=0.98,
                        source_evidence=SourceEvidence(source_text=raw),
                    )
                ],
            )
            report = validate_discharge_payload(payload)
            self.assertEqual(report.overall_safety_status, SafetyStatus.SAFE_ORGANIZED)
            self.assertNotIn(
                HazardCode.MEDICATION_CHANGE_DETECTED, report.hazard_codes_detected
            )
            self.assertNotIn(
                HazardCode.STOP_START_MEDICATION, report.hazard_codes_detected
            )
            self.assertTrue(is_patient_releasable(report))

        # Uncertain negation -> escalates to review
        uncertain_raw = "Unclear if no change to medication. Continue Atorvastatin 40 mg by mouth once daily."
        uncertain_payload = DischargeExtractionPayload(
            case_id="REG-20-UNCERTAIN",
            patient_identifier="PT-20U",
            raw_discharge_text=uncertain_raw,
            medications=[
                MedicationInstruction(
                    id="med-20u",
                    medication_name="Atorvastatin",
                    action=MedicationAction.CONTINUE,
                    dosage="40 mg",
                    frequency="once daily",
                    route="by mouth",
                    confidence=0.98,
                    source_evidence=SourceEvidence(source_text=uncertain_raw),
                )
            ],
        )
        uncertain_report = validate_discharge_payload(uncertain_payload)
        self.assertEqual(
            uncertain_report.overall_safety_status, SafetyStatus.NEEDS_REVIEW
        )
        self.assertFalse(is_patient_releasable(uncertain_report))

    def test_21_empty_extraction_flagged_as_incomplete(self) -> None:
        payload = DischargeExtractionPayload(
            case_id="REG-21",
            patient_identifier="PT-21",
            raw_discharge_text="Continue Atorvastatin 40 mg by mouth once daily.",
            medications=[],
            follow_ups=[],
            warning_signs=[],
            tasks=[],
        )
        report = validate_discharge_payload(payload)
        self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
        self.assertTrue(report.requires_human_review)
        self.assertIn(HazardCode.INCOMPLETE_EXTRACTION, report.hazard_codes_detected)
        self.assertIn(HazardCode.OMITTED_INSTRUCTION, report.hazard_codes_detected)
        self.assertFalse(is_patient_releasable(report))

    def test_22_patient_release_gate_and_source_span_verification(self) -> None:
        normal_report = validate_discharge_payload(build_normal_case())
        self.assertTrue(normal_report.patient_releasable)
        self.assertTrue(is_patient_releasable(normal_report))
        self.assertTrue(is_patient_releasable(normal_report.validated_payload))

        # Verify null span reports explicit note and unverified page flag
        med_ev = normal_report.validated_payload.medications[0].source_evidence
        assert med_ev is not None
        self.assertEqual(med_ev.span_status_note, SPAN_NOT_IMPLEMENTED_NOTE)
        self.assertFalse(med_ev.source_page_verified)

        # Verify valid character span passes
        valid_span_payload = _make_clean_med_payload()
        assert valid_span_payload.medications[0].source_evidence is not None
        valid_span_payload.medications[0].source_evidence.start_char = 0
        valid_span_payload.medications[0].source_evidence.end_char = len(
            valid_span_payload.raw_discharge_text
        )
        valid_span_report = validate_discharge_payload(valid_span_payload)
        self.assertTrue(valid_span_report.patient_releasable)

        # Verify mismatched character span fails closed -> NEEDS REVIEW
        bad_span_payload = _make_clean_med_payload()
        assert bad_span_payload.medications[0].source_evidence is not None
        bad_span_payload.medications[0].source_evidence.start_char = 0
        bad_span_payload.medications[0].source_evidence.end_char = 10
        bad_span_report = validate_discharge_payload(bad_span_payload)
        self.assertEqual(
            bad_span_report.overall_safety_status, SafetyStatus.NEEDS_REVIEW
        )
        self.assertIn(
            HazardCode.MISSING_SOURCE_EVIDENCE, bad_span_report.hazard_codes_detected
        )
        self.assertFalse(bad_span_report.patient_releasable)
        self.assertFalse(is_patient_releasable(bad_span_report))

    def test_23_additional_edge_cases_for_warning_signs_tasks_and_actions(self) -> None:
        """Cover remaining validator branches: warning sign & task evidence mismatch, invalid span bounds, and unsupported DOSE_CHANGE action."""
        raw = (
            "Continue Atorvastatin 40 mg by mouth once daily.\n"
            "Monitor for mild ankle swelling and call Primary Care clinic if noted."
        )
        payload = DischargeExtractionPayload(
            case_id="REG-23",
            patient_identifier="PT-23",
            raw_discharge_text=raw,
            medications=[
                MedicationInstruction(
                    id="med-23",
                    medication_name="Atorvastatin",
                    action=MedicationAction.DOSE_CHANGE,  # Unsupported DOSE_CHANGE action!
                    dosage="40 mg",
                    previous_dosage="20 mg",  # Not in source!
                    frequency="once daily",
                    route="by mouth",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Continue Atorvastatin 40 mg by mouth once daily.",
                        start_char=0,
                        end_char=5000,  # Out-of-bounds end_char!
                    ),
                )
            ],
            warning_signs=[
                WarningSignInstruction(
                    id="ws-23",
                    symptom_description="Severe headache and blurred vision",  # Not in source!
                    urgency=UrgencyLevel.ROUTINE,
                    action_to_take="Go to nearest urgent care immediately",  # Not in source!
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Monitor for mild ankle swelling and call Primary Care clinic if noted."
                    ),
                )
            ],
            tasks=[
                CareTaskItem(
                    id="task-23",
                    category="MEDICATION",
                    title="Take Atorvastatin 40 mg",
                    due_window="within 3 days",  # Not in source!
                    linked_item_id="med-23",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Continue Atorvastatin 40 mg by mouth once daily."
                    ),
                )
            ],
            provider_matches=[
                SyntheticProviderMatch(
                    provider_id="prov-no-disclaimer",
                    name="Dr. Plain",
                    specialty="Primary Care",
                    facility="Plain Clinic",
                    distance_miles=1.0,
                    accepts_new_patients_listed=True,
                    verification_disclaimer="Synthetic directory listing without mandatory phrase.",
                )
            ],
        )
        report = validate_discharge_payload(payload)
        self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
        self.assertFalse(is_patient_releasable(report))
        self.assertIn(HazardCode.PROHIBITED_CLINICAL_ACTION, report.hazard_codes_detected)
        self.assertIn(HazardCode.UNSUPPORTED_FIELD_VALUE, report.hazard_codes_detected)
        self.assertIn(HazardCode.MISSING_SOURCE_EVIDENCE, report.hazard_codes_detected)
        self.assertIn(HazardCode.UNSUPPORTED_PROVIDER_CLAIM, report.hazard_codes_detected)


if __name__ == "__main__":
    unittest.main()
