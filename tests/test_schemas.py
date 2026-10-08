"""Unit and adversarial boundary tests for CAREBRIDGE strict Pydantic schemas."""

from __future__ import annotations

import unittest
from pydantic import ValidationError

from backend.schemas.models import (
    DischargeExtractionPayload,
    FollowUpInstruction,
    MedicationAction,
    MedicationInstruction,
    ReviewDecisionStatus,
    SafetyStatus,
    SourceEvidence,
    SyntheticProviderMatch,
    UrgencyLevel,
    WarningSignInstruction,
)


class TestStrictSchemas(unittest.TestCase):
    """Verify strict schema validation and intentional adversarial breakage."""

    def test_valid_medication_instruction_preserves_metadata(self) -> None:
        evidence = SourceEvidence(
            source_text="Continue Metformin 500 mg by mouth twice daily.",
            source_section="Discharge Medications",
            source_page=2,
            start_char=10,
            end_char=57,
        )
        med = MedicationInstruction(
            id="med-1",
            medication_name="Metformin",
            action=MedicationAction.CONTINUE,
            dosage="500 mg",
            frequency="twice daily",
            route="by mouth",
            confidence=0.95,
            source_evidence=evidence,
        )
        self.assertEqual(med.safety_status, SafetyStatus.SAFE_ORGANIZED)
        self.assertEqual(med.review_status, ReviewDecisionStatus.AUTO_APPROVED)
        self.assertIsNotNone(med.source_evidence)
        assert med.source_evidence is not None
        self.assertEqual(med.source_evidence.source_page, 2)

    def test_missing_fields_preserved_as_none_without_fabrication(self) -> None:
        follow_up = FollowUpInstruction(
            id="fu-1",
            specialty=None,
            timeframe=None,
            confidence=0.82,
        )
        self.assertIsNone(follow_up.specialty)
        self.assertIsNone(follow_up.timeframe)
        self.assertIsNone(follow_up.provider_name)
        self.assertIsNone(follow_up.source_evidence)

    def test_rejects_extra_undeclared_llm_fields(self) -> None:
        with self.assertRaises(ValidationError):
            MedicationInstruction(
                id="med-extra",
                medication_name="Aspirin",
                action=MedicationAction.CONTINUE,
                confidence=0.9,
                hallucinated_llm_field="should_fail",  # type: ignore[call-arg]
            )

    def test_rejects_confidence_out_of_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            MedicationInstruction(
                id="med-high-conf",
                medication_name="Aspirin",
                confidence=1.05,
            )
        with self.assertRaises(ValidationError):
            MedicationInstruction(
                id="med-low-conf",
                medication_name="Aspirin",
                confidence=-0.01,
            )

    def test_rejects_non_strict_type_coercion(self) -> None:
        # Passing string "0.95" instead of float 0.95 must fail under strict=True
        with self.assertRaises(ValidationError):
            MedicationInstruction.model_validate(
                {
                    "id": "med-strict",
                    "medication_name": "Aspirin",
                    "confidence": "0.95",
                }
            )

    def test_rejects_invalid_source_evidence_spans_and_empty_text(self) -> None:
        with self.assertRaises(ValidationError):
            SourceEvidence(source_text="   ")

        with self.assertRaises(ValidationError):
            SourceEvidence(
                source_text="Valid quote",
                source_page=0,  # Must be >= 1
            )

        with self.assertRaises(ValidationError):
            SourceEvidence(
                source_text="Valid quote",
                start_char=50,
                end_char=10,  # end_char < start_char
            )

    def test_rejects_empty_clinical_disclaimer(self) -> None:
        with self.assertRaises(ValidationError):
            DischargeExtractionPayload(
                case_id="CASE-ERR",
                patient_identifier="PT-1",
                raw_discharge_text="Sample discharge text.",
                clinical_disclaimer="Too short",
            )

    def test_rejects_negative_provider_distance(self) -> None:
        with self.assertRaises(ValidationError):
            SyntheticProviderMatch(
                provider_id="prov-1",
                name="Dr. Test",
                specialty="Cardiology",
                facility="Test Clinic",
                distance_miles=-1.5,
                accepts_new_patients_listed=True,
            )

    def test_ui_status_enum_exact_labels(self) -> None:
        self.assertEqual(SafetyStatus.SAFE_ORGANIZED.value, "SAFE / ORGANIZED")
        self.assertEqual(SafetyStatus.NEEDS_REVIEW.value, "NEEDS REVIEW")
        self.assertEqual(SafetyStatus.CONFLICT_DETECTED.value, "CONFLICT DETECTED")

    def test_warning_sign_requires_non_empty_description(self) -> None:
        with self.assertRaises(ValidationError):
            WarningSignInstruction(
                id="ws-empty",
                symptom_description="   ",
                urgency=UrgencyLevel.ROUTINE,
                confidence=0.9,
            )

    def test_rejects_invalid_enum_string(self) -> None:
        with self.assertRaises(ValidationError):
            MedicationInstruction.model_validate(
                {
                    "id": "med-bad-enum",
                    "medication_name": "Aspirin",
                    "action": "INVENTED_ACTION",
                    "confidence": 0.9,
                }
            )


if __name__ == "__main__":
    unittest.main()
