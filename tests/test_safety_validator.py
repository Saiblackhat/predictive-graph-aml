"""Comprehensive safety validation and API tests across all synthetic fixtures and edge cases."""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from backend.fixtures.discharge_cases import (
    build_ambiguous_case,
    build_conflicting_case,
    build_incomplete_case,
    build_normal_case,
    build_unsafe_case,
)
from backend.main import app
from backend.safety.validator import validate_discharge_payload
from backend.schemas.models import (
    DischargeExtractionPayload,
    HazardCode,
    MedicationAction,
    MedicationInstruction,
    ReviewDecisionStatus,
    SafetyStatus,
    SourceEvidence,
)


class TestSafetyValidatorAndFixtures(unittest.TestCase):
    """Test the deterministic Safety Validation Layer against all 5 synthetic cases and red-team inputs."""

    def test_normal_case_passes_as_safe_organized(self) -> None:
        payload = build_normal_case()
        report = validate_discharge_payload(payload)

        self.assertEqual(report.overall_safety_status, SafetyStatus.SAFE_ORGANIZED)
        self.assertFalse(report.requires_human_review)
        self.assertEqual(report.total_flags, 0)
        self.assertEqual(report.hazard_codes_detected, [])

        for med in report.validated_payload.medications:
            self.assertEqual(med.safety_status, SafetyStatus.SAFE_ORGANIZED)
            self.assertEqual(med.review_status, ReviewDecisionStatus.AUTO_APPROVED)

    def test_incomplete_case_detects_missing_timeframe_specialty_dosage_and_evidence(
        self,
    ) -> None:
        payload = build_incomplete_case()
        report = validate_discharge_payload(payload)

        self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
        self.assertTrue(report.requires_human_review)

        codes = set(report.hazard_codes_detected)
        self.assertIn(HazardCode.MISSING_FOLLOWUP_TIMEFRAME, codes)
        self.assertIn(HazardCode.MISSING_SPECIALTY, codes)
        self.assertIn(HazardCode.UNCLEAR_DOSAGE, codes)
        self.assertIn(HazardCode.MISSING_SOURCE_EVIDENCE, codes)

        # Ensure missing values were NOT fabricated during validation
        levothyroxine = report.validated_payload.medications[0]
        self.assertIsNone(levothyroxine.dosage)
        self.assertEqual(levothyroxine.safety_status, SafetyStatus.NEEDS_REVIEW)

    def test_ambiguous_case_detects_changes_stop_start_vague_meds_and_unclear_symptoms(
        self,
    ) -> None:
        payload = build_ambiguous_case()
        report = validate_discharge_payload(payload)

        self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
        self.assertTrue(report.requires_human_review)

        codes = set(report.hazard_codes_detected)
        self.assertIn(HazardCode.MEDICATION_CHANGE_DETECTED, codes)
        self.assertIn(HazardCode.STOP_START_MEDICATION, codes)
        self.assertIn(HazardCode.AMBIGUOUS_MEDICATION_INSTRUCTION, codes)
        self.assertIn(HazardCode.UNCLEAR_DOSAGE, codes)
        self.assertIn(HazardCode.UNCLEAR_SYMPTOMS, codes)
        self.assertIn(HazardCode.LOW_EXTRACTION_CONFIDENCE, codes)

    def test_conflicting_case_escalates_to_conflict_detected(self) -> None:
        payload = build_conflicting_case()
        report = validate_discharge_payload(payload)

        self.assertEqual(report.overall_safety_status, SafetyStatus.CONFLICT_DETECTED)
        self.assertTrue(report.requires_human_review)
        self.assertIn(HazardCode.CONFLICTING_INSTRUCTIONS, report.hazard_codes_detected)

        for med in report.validated_payload.medications:
            self.assertEqual(med.safety_status, SafetyStatus.CONFLICT_DETECTED)
            self.assertEqual(
                med.review_status, ReviewDecisionStatus.ESCALATED_TO_CLINICIAN
            )

        for fu in report.validated_payload.follow_ups:
            self.assertEqual(fu.safety_status, SafetyStatus.CONFLICT_DETECTED)
            self.assertEqual(
                fu.review_status, ReviewDecisionStatus.ESCALATED_TO_CLINICIAN
            )

    def test_unsafe_case_detects_red_flags_provider_claims_prohibited_advice_and_hallucinated_quote(
        self,
    ) -> None:
        payload = build_unsafe_case()
        report = validate_discharge_payload(payload)

        self.assertEqual(report.overall_safety_status, SafetyStatus.NEEDS_REVIEW)
        self.assertTrue(report.requires_human_review)

        codes = set(report.hazard_codes_detected)
        self.assertIn(HazardCode.WARNING_SIGN_ESCALATION, codes)
        self.assertIn(HazardCode.UNSUPPORTED_PROVIDER_CLAIM, codes)
        self.assertIn(HazardCode.PROHIBITED_CLINICAL_ACTION, codes)
        self.assertIn(HazardCode.MISSING_SOURCE_EVIDENCE, codes)

    def test_intra_document_silent_medication_stop_conflict(self) -> None:
        """Adversarial test: AI extracts only CONTINUE Lisinopril, but raw text also says Discontinue Lisinopril."""
        raw_text = (
            "Hospital Course: Continue Lisinopril 10 mg daily.\n"
            "Addendum: Discontinue Lisinopril immediately due to hyperkalemia."
        )
        payload = DischargeExtractionPayload(
            case_id="CASE-ADV-CONFLICT",
            patient_identifier="PT-ADV-1",
            raw_discharge_text=raw_text,
            medications=[
                MedicationInstruction(
                    id="med-adv-1",
                    medication_name="Lisinopril",
                    action=MedicationAction.CONTINUE,
                    dosage="10 mg",
                    frequency="daily",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Continue Lisinopril 10 mg daily.",
                        source_section="Hospital Course",
                    ),
                )
            ],
        )
        report = validate_discharge_payload(payload)
        self.assertEqual(report.overall_safety_status, SafetyStatus.CONFLICT_DETECTED)
        self.assertIn(HazardCode.CONFLICTING_INSTRUCTIONS, report.hazard_codes_detected)

    def test_fastapi_endpoints_return_validated_reports(self) -> None:
        client = TestClient(app)

        health_res = client.get("/health")
        self.assertEqual(health_res.status_code, 200)
        self.assertEqual(health_res.json()["status"], "ok")

        for fixture_name, expected_status in [
            ("normal", SafetyStatus.SAFE_ORGANIZED.value),
            ("incomplete", SafetyStatus.NEEDS_REVIEW.value),
            ("ambiguous", SafetyStatus.NEEDS_REVIEW.value),
            ("conflicting", SafetyStatus.CONFLICT_DETECTED.value),
            ("unsafe", SafetyStatus.NEEDS_REVIEW.value),
        ]:
            res = client.get(f"/api/v1/fixtures/{fixture_name}")
            self.assertEqual(res.status_code, 200)
            body = res.json()
            self.assertEqual(body["overall_safety_status"], expected_status)

        not_found_res = client.get("/api/v1/fixtures/nonexistent")
        self.assertEqual(not_found_res.status_code, 404)

        post_res = client.post(
            "/api/v1/validate", json=build_normal_case().model_dump(mode="json")
        )
        self.assertEqual(post_res.status_code, 200)
        self.assertEqual(
            post_res.json()["overall_safety_status"], SafetyStatus.SAFE_ORGANIZED.value
        )


if __name__ == "__main__":
    unittest.main()
