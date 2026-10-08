"""Synthetic discharge summary documents for Phase 4 Document Intelligence testing.

Covers all 8 required realistic synthetic scenarios:
1. Normal uncomplicated discharge
2. Multiple medications + follow-ups
3. Missing/ambiguous follow-up
4. Conflicting medication instructions
5. Warning signs
6. Tests + referrals
7. Rich multi-section discharge
8. Messy formatting / Unicode / line breaks
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticDischargeDocumentFixture:
    """Raw synthetic discharge document fixture with metadata."""

    document_id: str
    patient_identifier: str
    scenario_key: str
    description: str
    raw_text: str


DOC_SYNTH_001_NORMAL = SyntheticDischargeDocumentFixture(
    document_id="DOC-SYNTH-001",
    patient_identifier="SYNTH-PT-401",
    scenario_key="normal_uncomplicated",
    description="1. Normal uncomplicated discharge",
    raw_text=(
        "DISCHARGE SUMMARY - DOC-SYNTH-001\n"
        "Patient Identifier: SYNTH-PT-401\n\n"
        "Discharge Medications:\n"
        "1. Continue Atorvastatin 40 mg by mouth once daily at bedtime.\n"
        "2. Continue Amlodipine 5 mg by mouth once daily in the morning.\n\n"
        "Follow-Up Appointments:\n"
        "- Follow up with Primary Care (Dr. Elena Rostova) within 7 days for blood pressure check.\n\n"
        "Return Precautions:\n"
        "- Monitor for mild ankle swelling and call Primary Care clinic during office hours if noted.\n"
    ),
)

DOC_SYNTH_002_MULTI = SyntheticDischargeDocumentFixture(
    document_id="DOC-SYNTH-002",
    patient_identifier="SYNTH-PT-402",
    scenario_key="multiple_meds_and_followups",
    description="2. Multiple medications + follow-ups",
    raw_text=(
        "DISCHARGE SUMMARY - DOC-SYNTH-002\n"
        "Patient Identifier: SYNTH-PT-402\n\n"
        "Discharge Medications:\n"
        "1. Continue sacubitril-valsartan 49/51 mg by mouth twice daily.\n"
        "2. Continue dapagliflozin 10 mg by mouth once daily.\n"
        "3. Continue carvedilol 12.5 mg by mouth twice daily.\n\n"
        "Follow-Up Appointments:\n"
        "- Follow up with Cardiology (Dr. Marcus Chen) within 7 days for heart failure evaluation.\n"
        "- Follow up with Nephrology within 14 days for renal function review.\n"
    ),
)

DOC_SYNTH_003_MISSING_AMBIGUOUS = SyntheticDischargeDocumentFixture(
    document_id="DOC-SYNTH-003",
    patient_identifier="SYNTH-PT-403",
    scenario_key="missing_ambiguous_followup",
    description="3. Missing/ambiguous follow-up and missing medication dosage",
    raw_text=(
        "DISCHARGE SUMMARY - DOC-SYNTH-003\n"
        "Patient Identifier: SYNTH-PT-403\n\n"
        "Discharge Medications:\n"
        "1. Continue Levothyroxine once daily in the morning.\n\n"
        "Follow-up:\n"
        "- Follow up with Cardiology for post-discharge evaluation.\n"
        "- Follow up with clinic within 14 days.\n"
    ),
)

DOC_SYNTH_004_CONFLICTING_MEDS = SyntheticDischargeDocumentFixture(
    document_id="DOC-SYNTH-004",
    patient_identifier="SYNTH-PT-404",
    scenario_key="conflicting_medications",
    description="4. Conflicting medication instructions across sections",
    raw_text=(
        "DISCHARGE SUMMARY - DOC-SYNTH-004\n"
        "Patient Identifier: SYNTH-PT-404\n\n"
        "Hospital Course:\n"
        "Patient stabilized. Continue Lisinopril 10 mg by mouth once daily.\n\n"
        "Discharge Medications:\n"
        "1. Stop Lisinopril 10 mg due to acute kidney injury.\n\n"
        "Follow-Up Appointments:\n"
        "- Follow up with Nephrology within 7 days.\n"
    ),
)

DOC_SYNTH_005_WARNING_SIGNS = SyntheticDischargeDocumentFixture(
    document_id="DOC-SYNTH-005",
    patient_identifier="SYNTH-PT-405",
    scenario_key="warning_signs",
    description="5. High-acuity emergency warning signs and numeric return precautions",
    raw_text=(
        "DISCHARGE SUMMARY - DOC-SYNTH-005\n"
        "Patient Identifier: SYNTH-PT-405\n\n"
        "Discharge Medications:\n"
        "1. Continue furosemide 20 mg by mouth once daily.\n\n"
        "Follow-Up Appointments:\n"
        "- Follow up with Cardiology within 7 days.\n\n"
        "Warning Signs:\n"
        "- Call 911 immediately for chest pain, severe shortness of breath, or fainting.\n"
        "- Call Cardiology if your weight rises more than 3 lb in one day.\n"
    ),
)

DOC_SYNTH_006_TESTS_AND_REFERRALS = SyntheticDischargeDocumentFixture(
    document_id="DOC-SYNTH-006",
    patient_identifier="SYNTH-PT-406",
    scenario_key="tests_and_referrals",
    description="6. Outpatient laboratory tests and specialty referrals",
    raw_text=(
        "DISCHARGE SUMMARY - DOC-SYNTH-006\n"
        "Patient Identifier: SYNTH-PT-406\n\n"
        "Discharge Medications:\n"
        "1. Continue apixaban 5 mg by mouth twice daily.\n\n"
        "Appointments:\n"
        "- Follow up with Hematology within 14 days.\n\n"
        "Tests:\n"
        "- Obtain a basic metabolic panel within 5 days.\n"
        "- Repeat complete blood count within 7 days.\n\n"
        "Referrals:\n"
        "- Referral to Physical Therapy within 7 days for gait training.\n"
    ),
)

DOC_SYNTH_007_RICH_MULTISECTION = SyntheticDischargeDocumentFixture(
    document_id="DOC-SYNTH-007",
    patient_identifier="SYNTH-PT-407",
    scenario_key="rich_multisection_discharge",
    description="7. Rich multi-section discharge covering medications, follow-ups, tests, referrals, wound care, diet, activity, warning signs, and unknown section",
    raw_text=(
        "DISCHARGE SUMMARY - DOC-SYNTH-007\n"
        "Patient Identifier: SYNTH-PT-407\n\n"
        "Discharge Medications:\n"
        "1. Continue aspirin 81 mg by mouth once daily.\n"
        "2. Continue atorvastatin 40 mg by mouth once daily at bedtime.\n\n"
        "Follow-Up Appointments:\n"
        "- Follow up with Cardiology (Dr. Elena Rostova) within 7 days.\n\n"
        "Tests:\n"
        "- Obtain a basic metabolic panel within 5 days.\n\n"
        "Referrals:\n"
        "- Referral to Cardiac Rehabilitation within 14 days.\n\n"
        "Wound Care:\n"
        "- Keep sternal incision clean and dry for 7 days.\n\n"
        "Diet:\n"
        "- Follow a low-sodium heart-healthy Mediterranean diet as directed by nutrition.\n\n"
        "Activity:\n"
        "- Walk for 20 minutes each afternoon as tolerated.\n"
        "- Perform gentle physical therapy exercises each morning.\n\n"
        "When to Seek Help:\n"
        "- Monitor for mild ankle swelling and call Primary Care clinic during office hours if noted.\n\n"
        "Equipment and Supplies:\n"
        "- Use home blood pressure cuff each morning.\n"
    ),
)

DOC_SYNTH_008_MESSY_UNICODE = SyntheticDischargeDocumentFixture(
    document_id="DOC-SYNTH-008",
    patient_identifier="SYNTH-PT-408",
    scenario_key="messy_unicode_formatting",
    description="8. Messy formatting with curly quotes, NBSP, line-break hyphenation, and zero-width space",
    raw_text=(
        "DISCHARGE SUMMARY - DOC-SYNTH-008\r\n"
        "Patient Identifier: SYNTH-PT-408\r\n\r\n"
        "Discharge Medications:\r\n"
        "1.   Continue   \u2018aspirin\u2019 81\u00a0mg\tby mouth once daily.\r\n"
        "2.   Disconti-\r\n"
        "   nue lisin\u200bopril 10 mg once daily.\r\n\r\n"
        "Follow-up:\r\n"
        "-  Follow-\r\n"
        "   up with Cardiol-\r\n"
        "   ogy within 7 days.\r\n"
    ),
)

ALL_PHASE4_SYNTHETIC_DOCUMENTS: dict[str, SyntheticDischargeDocumentFixture] = {
    f.scenario_key: f
    for f in (
        DOC_SYNTH_001_NORMAL,
        DOC_SYNTH_002_MULTI,
        DOC_SYNTH_003_MISSING_AMBIGUOUS,
        DOC_SYNTH_004_CONFLICTING_MEDS,
        DOC_SYNTH_005_WARNING_SIGNS,
        DOC_SYNTH_006_TESTS_AND_REFERRALS,
        DOC_SYNTH_007_RICH_MULTISECTION,
        DOC_SYNTH_008_MESSY_UNICODE,
    )
}
