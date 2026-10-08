"""Canonical synthetic discharge test fixtures for CAREBRIDGE.

Covers all five required clinical validation scenarios:
1. Normal (SAFE / ORGANIZED)
2. Incomplete (missing timeframe, specialty, dosage, source evidence)
3. Ambiguous (ambiguous med instructions, dose change, stop/start, unclear symptoms)
4. Conflicting (contradictory medication and follow-up instructions)
5. Unsafe (warning-sign escalation, unsupported provider claims, prohibited clinical advice, fabricated quote)
"""

from __future__ import annotations

from backend.schemas.models import (
    CareTaskItem,
    DischargeExtractionPayload,
    FollowUpInstruction,
    MedicationAction,
    MedicationInstruction,
    SourceEvidence,
    SyntheticProviderMatch,
    UrgencyLevel,
    WarningSignInstruction,
)


def build_normal_case() -> DischargeExtractionPayload:
    """Scenario 1: Clean, well-documented discharge instructions with zero hazards."""
    raw_text = (
        "DISCHARGE SUMMARY - CASE-NORMAL-001\n"
        "Discharge Medications:\n"
        "1. Continue Atorvastatin 40 mg by mouth once daily at bedtime.\n"
        "2. Continue Amlodipine 5 mg by mouth once daily in the morning.\n"
        "Follow-Up Appointments:\n"
        "- Follow up with Primary Care (Dr. Elena Rostova) within 7 days for blood pressure check.\n"
        "Return Precautions:\n"
        "- Monitor for mild ankle swelling and call Primary Care clinic during office hours if noted.\n"
    )
    return DischargeExtractionPayload(
        case_id="CASE-NORMAL-001",
        patient_identifier="SYNTH-PT-101",
        raw_discharge_text=raw_text,
        medications=[
            MedicationInstruction(
                id="med-norm-1",
                medication_name="Atorvastatin",
                action=MedicationAction.CONTINUE,
                dosage="40 mg",
                frequency="once daily at bedtime",
                route="by mouth",
                confidence=0.98,
                source_evidence=SourceEvidence(
                    source_text="Continue Atorvastatin 40 mg by mouth once daily at bedtime.",
                    source_section="Discharge Medications",
                    source_page=1,
                ),
                patient_friendly_summary="Keep taking Atorvastatin 40 mg by mouth once daily at bedtime.",
            ),
            MedicationInstruction(
                id="med-norm-2",
                medication_name="Amlodipine",
                action=MedicationAction.CONTINUE,
                dosage="5 mg",
                frequency="once daily in the morning",
                route="by mouth",
                confidence=0.97,
                source_evidence=SourceEvidence(
                    source_text="Continue Amlodipine 5 mg by mouth once daily in the morning.",
                    source_section="Discharge Medications",
                    source_page=1,
                ),
                patient_friendly_summary="Keep taking Amlodipine 5 mg by mouth once daily in the morning.",
            ),
        ],
        follow_ups=[
            FollowUpInstruction(
                id="fu-norm-1",
                specialty="Primary Care",
                provider_name="Dr. Elena Rostova",
                timeframe="within 7 days",
                reason="blood pressure check",
                appointment_scheduled=False,
                confidence=0.96,
                source_evidence=SourceEvidence(
                    source_text="Follow up with Primary Care (Dr. Elena Rostova) within 7 days for blood pressure check.",
                    source_section="Follow-Up Appointments",
                    source_page=1,
                ),
                patient_friendly_summary="Schedule a visit with Primary Care (Dr. Elena Rostova) within 7 days for a blood pressure check.",
            )
        ],
        warning_signs=[
            WarningSignInstruction(
                id="ws-norm-1",
                symptom_description="Mild ankle swelling",
                urgency=UrgencyLevel.ROUTINE,
                action_to_take="Call Primary Care clinic during office hours",
                confidence=0.94,
                source_evidence=SourceEvidence(
                    source_text="Monitor for mild ankle swelling and call Primary Care clinic during office hours if noted.",
                    source_section="Return Precautions",
                    source_page=1,
                ),
                patient_friendly_summary="Watch for mild ankle swelling and call the Primary Care clinic during office hours if it happens.",
            )
        ],
        tasks=[
            CareTaskItem(
                id="task-norm-1",
                category="FOLLOW_UP",
                title="Book Primary Care follow-up within 7 days",
                due_window="within 7 days",
                linked_item_id="fu-norm-1",
                confidence=0.96,
                source_evidence=SourceEvidence(
                    source_text="Follow up with Primary Care (Dr. Elena Rostova) within 7 days for blood pressure check.",
                    source_section="Follow-Up Appointments",
                    source_page=1,
                ),
            )
        ],
        provider_matches=[
            SyntheticProviderMatch(
                provider_id="prov-norm-1",
                name="Dr. Elena Rostova, MD",
                specialty="Primary Care",
                facility="Metro Community Health Center (Synthetic)",
                distance_miles=2.4,
                accepts_new_patients_listed=True,
                claim_text="Listed in synthetic directory under Primary Care.",
            )
        ],
    )


def build_incomplete_case() -> DischargeExtractionPayload:
    """Scenario 2: Incomplete instructions (missing timeframe, specialty, dosage, and source evidence)."""
    raw_text = (
        "DISCHARGE SUMMARY - CASE-INCOMPLETE-002\n"
        "Discharge Medications:\n"
        "1. Continue Levothyroxine daily in the morning.\n"
        "Follow-Up Appointments:\n"
        "- Follow up with Cardiology for post-discharge evaluation.\n"
        "- Follow up with clinic within 14 days.\n"
    )
    return DischargeExtractionPayload(
        case_id="CASE-INCOMPLETE-002",
        patient_identifier="SYNTH-PT-102",
        raw_discharge_text=raw_text,
        medications=[
            MedicationInstruction(
                id="med-inc-1",
                medication_name="Levothyroxine",
                action=MedicationAction.CONTINUE,
                dosage=None,  # Missing dosage - represented as None (unknown), never fabricated
                frequency="daily in the morning",
                route="by mouth",
                confidence=0.88,
                source_evidence=SourceEvidence(
                    source_text="Continue Levothyroxine daily in the morning.",
                    source_section="Discharge Medications",
                    source_page=1,
                ),
            ),
            MedicationInstruction(
                id="med-inc-no-evidence",
                medication_name="Potassium Chloride",
                action=MedicationAction.CONTINUE,
                dosage="20 mEq",
                frequency="once daily",
                confidence=0.85,
                source_evidence=None,  # Missing source evidence
            ),
        ],
        follow_ups=[
            FollowUpInstruction(
                id="fu-inc-missing-timeframe",
                specialty="Cardiology",
                timeframe=None,  # Missing follow-up timeframe
                reason="post-discharge evaluation",
                confidence=0.90,
                source_evidence=SourceEvidence(
                    source_text="Follow up with Cardiology for post-discharge evaluation.",
                    source_section="Follow-Up Appointments",
                    source_page=1,
                ),
            ),
            FollowUpInstruction(
                id="fu-inc-missing-specialty",
                specialty=None,  # Missing specialty
                timeframe="within 14 days",
                reason="general follow-up",
                confidence=0.86,
                source_evidence=SourceEvidence(
                    source_text="Follow up with clinic within 14 days.",
                    source_section="Follow-Up Appointments",
                    source_page=1,
                ),
            ),
        ],
    )


def build_ambiguous_case() -> DischargeExtractionPayload:
    """Scenario 3: Ambiguous instructions, dose changes, stop/start medications, and unclear symptoms."""
    raw_text = (
        "DISCHARGE SUMMARY - CASE-AMBIGUOUS-003\n"
        "Discharge Medications:\n"
        "1. Increase Furosemide from 20 mg to 40 mg by mouth once daily.\n"
        "2. STOP taking Ibuprofen immediately.\n"
        "3. Start new Apixaban 5 mg by mouth twice daily.\n"
        "4. Metoprolol: take usual dose as needed if needed, maybe resume when ready.\n"
        "Return Precautions:\n"
        "- Call if feeling worse or not feeling right.\n"
    )
    return DischargeExtractionPayload(
        case_id="CASE-AMBIGUOUS-003",
        patient_identifier="SYNTH-PT-103",
        raw_discharge_text=raw_text,
        medications=[
            MedicationInstruction(
                id="med-amb-change",
                medication_name="Furosemide",
                action=MedicationAction.DOSE_CHANGE,
                dosage="40 mg",
                previous_dosage="20 mg",
                frequency="once daily",
                route="by mouth",
                confidence=0.95,
                source_evidence=SourceEvidence(
                    source_text="Increase Furosemide from 20 mg to 40 mg by mouth once daily.",
                    source_section="Discharge Medications",
                    source_page=1,
                ),
            ),
            MedicationInstruction(
                id="med-amb-stop",
                medication_name="Ibuprofen",
                action=MedicationAction.STOP_DISCONTINUE,
                dosage=None,
                frequency=None,
                confidence=0.97,
                source_evidence=SourceEvidence(
                    source_text="STOP taking Ibuprofen immediately.",
                    source_section="Discharge Medications",
                    source_page=1,
                ),
            ),
            MedicationInstruction(
                id="med-amb-start",
                medication_name="Apixaban",
                action=MedicationAction.NEW_START,
                dosage="5 mg",
                frequency="twice daily",
                route="by mouth",
                confidence=0.96,
                source_evidence=SourceEvidence(
                    source_text="Start new Apixaban 5 mg by mouth twice daily.",
                    source_section="Discharge Medications",
                    source_page=1,
                ),
            ),
            MedicationInstruction(
                id="med-amb-vague",
                medication_name="Metoprolol",
                action=MedicationAction.UNKNOWN,
                dosage="usual dose",
                frequency="as needed if needed",
                indication_or_note="maybe resume when ready",
                confidence=0.68,
                source_evidence=SourceEvidence(
                    source_text="Metoprolol: take usual dose as needed if needed, maybe resume when ready.",
                    source_section="Discharge Medications",
                    source_page=1,
                ),
            ),
        ],
        warning_signs=[
            WarningSignInstruction(
                id="ws-amb-1",
                symptom_description="feeling worse",
                urgency=UrgencyLevel.UNKNOWN,
                action_to_take=None,
                confidence=0.75,
                source_evidence=SourceEvidence(
                    source_text="Call if feeling worse or not feeling right.",
                    source_section="Return Precautions",
                    source_page=1,
                ),
            )
        ],
    )


def build_conflicting_case() -> DischargeExtractionPayload:
    """Scenario 4: Conflicting medication and follow-up instructions within the same summary."""
    raw_text = (
        "DISCHARGE SUMMARY - CASE-CONFLICTING-004\n"
        "Hospital Course:\n"
        "Patient stabilized. Continue Lisinopril 10 mg by mouth once daily.\n"
        "Follow up with Cardiology within 3 days.\n"
        "Discharge Medications:\n"
        "STOP Lisinopril 10 mg due to acute kidney injury.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology in 6 weeks.\n"
    )
    return DischargeExtractionPayload(
        case_id="CASE-CONFLICTING-004",
        patient_identifier="SYNTH-PT-104",
        raw_discharge_text=raw_text,
        medications=[
            MedicationInstruction(
                id="med-conf-1",
                medication_name="Lisinopril",
                action=MedicationAction.CONTINUE,
                dosage="10 mg",
                frequency="once daily",
                route="by mouth",
                confidence=0.91,
                source_evidence=SourceEvidence(
                    source_text="Continue Lisinopril 10 mg by mouth once daily.",
                    source_section="Hospital Course",
                    source_page=1,
                ),
            ),
            MedicationInstruction(
                id="med-conf-2",
                medication_name="Lisinopril",
                action=MedicationAction.STOP_DISCONTINUE,
                dosage="10 mg",
                frequency=None,
                indication_or_note="due to acute kidney injury",
                confidence=0.94,
                source_evidence=SourceEvidence(
                    source_text="STOP Lisinopril 10 mg due to acute kidney injury.",
                    source_section="Discharge Medications",
                    source_page=1,
                ),
            ),
        ],
        follow_ups=[
            FollowUpInstruction(
                id="fu-conf-1",
                specialty="Cardiology",
                timeframe="within 3 days",
                confidence=0.92,
                source_evidence=SourceEvidence(
                    source_text="Follow up with Cardiology within 3 days.",
                    source_section="Hospital Course",
                    source_page=1,
                ),
            ),
            FollowUpInstruction(
                id="fu-conf-2",
                specialty="Cardiology",
                timeframe="in 6 weeks",
                confidence=0.93,
                source_evidence=SourceEvidence(
                    source_text="Follow up with Cardiology in 6 weeks.",
                    source_section="Follow-Up Appointments",
                    source_page=1,
                ),
            ),
        ],
    )


def build_unsafe_case() -> DischargeExtractionPayload:
    """Scenario 5: High-acuity red flags, unsupported provider claims, prohibited advice, and fabricated evidence."""
    raw_text = (
        "DISCHARGE SUMMARY - CASE-UNSAFE-005\n"
        "Discharge Medications:\n"
        "1. Continue Carvedilol 12.5 mg by mouth twice daily.\n"
        "Follow-Up Appointments:\n"
        "- Follow up with Cardiology within 5 days.\n"
        "Return Precautions:\n"
        "- Call 911 immediately for chest pain, severe shortness of breath, or fainting.\n"
    )
    return DischargeExtractionPayload(
        case_id="CASE-UNSAFE-005",
        patient_identifier="SYNTH-PT-105",
        raw_discharge_text=raw_text,
        medications=[
            MedicationInstruction(
                id="med-unsafe-prohibited",
                medication_name="Carvedilol",
                action=MedicationAction.CONTINUE,
                dosage="12.5 mg",
                frequency="twice daily",
                route="by mouth",
                confidence=0.95,
                source_evidence=SourceEvidence(
                    source_text="Continue Carvedilol 12.5 mg by mouth twice daily.",
                    source_section="Discharge Medications",
                    source_page=1,
                ),
                patient_friendly_summary=(
                    "This means you have heart failure; we recommend starting spironolactone "
                    "and you should switch to a higher dose if tired."
                ),
            ),
            MedicationInstruction(
                id="med-unsafe-hallucinated-quote",
                medication_name="Digoxin",
                action=MedicationAction.CONTINUE,
                dosage="125 mcg",
                frequency="once daily",
                confidence=0.92,
                source_evidence=SourceEvidence(
                    source_text="Continue Digoxin 125 mcg by mouth once daily.",  # Not in raw_text!
                    source_section="Discharge Medications",
                    source_page=1,
                ),
            ),
        ],
        follow_ups=[
            FollowUpInstruction(
                id="fu-unsafe-claim",
                specialty="Cardiology",
                timeframe="within 5 days",
                provider_claim_note="Dr. Vance is the best cardiologist and has a guaranteed appointment tomorrow.",
                confidence=0.91,
                source_evidence=SourceEvidence(
                    source_text="Follow up with Cardiology within 5 days.",
                    source_section="Follow-Up Appointments",
                    source_page=1,
                ),
            )
        ],
        warning_signs=[
            WarningSignInstruction(
                id="ws-unsafe-redflag",
                symptom_description="Chest pain, severe shortness of breath, or fainting",
                urgency=UrgencyLevel.EMERGENT_911,
                action_to_take="Call 911 immediately",
                confidence=0.99,
                source_evidence=SourceEvidence(
                    source_text="Call 911 immediately for chest pain, severe shortness of breath, or fainting.",
                    source_section="Return Precautions",
                    source_page=1,
                ),
            )
        ],
        provider_matches=[
            SyntheticProviderMatch(
                provider_id="prov-unsafe-1",
                name="Dr. Arthur Vance, MD",
                specialty="Cardiology",
                facility="Apex Heart Institute (Synthetic)",
                distance_miles=1.2,
                accepts_new_patients_listed=True,
                claim_text="100% suitable specialist with instant booking confirmed and guaranteed availability.",
            )
        ],
    )


ALL_SYNTHETIC_CASES = {
    "normal": build_normal_case,
    "incomplete": build_incomplete_case,
    "ambiguous": build_ambiguous_case,
    "conflicting": build_conflicting_case,
    "unsafe": build_unsafe_case,
}
