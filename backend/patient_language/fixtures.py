"""Phase 6 Synthetic Test Fixtures (L01 - L20).

Covers:
L01: Simple medication simplification
L02: BID/TID/QD expansion
L03: Medication formulation preservation (tartrate vs succinate)
L04: Follow-up simplification
L05: Relative timeframe preservation ('within 7 days')
L06: Absolute date preservation ('March 18, 2027')
L07: Warning threshold preservation (3 pounds in 24 hours)
L08: Decimal threshold preservation (101.5 F)
L09: Negation preservation ('Stop lisinopril')
L10: Conditional statement preservation
L11: Ambiguous source -> review remains
L12: Conflicting medication -> blocked remains
L13: Unsupported medication -> review/blocked remains
L14: Provider neutrality (no 'best' or 'recommended')
L15: Translation of a safe simple instruction
L16: Translation preserving numbers and units
L17: Translation preserving negation
L18: Translation preserving warning threshold
L19: Translation preserving timeframe operators
L20: Source-to-output provenance
"""

from __future__ import annotations

from typing import NamedTuple


class SyntheticLanguageFixture(NamedTuple):
    case_id: str
    description: str
    raw_text: str
    expected_patient_releasable: bool


L01_SIMPLE_MED = SyntheticLanguageFixture(
    case_id="L01",
    description="Simple medication simplification",
    raw_text=(
        "DISCHARGE SUMMARY - L01\n"
        "Discharge Medications:\n"
        "Continue aspirin 81 mg once daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Primary Care in 14 days.\n"
    ),
    expected_patient_releasable=True,
)

L02_FREQUENCY_EXPANSION = SyntheticLanguageFixture(
    case_id="L02",
    description="BID / TID / QD frequency expansion",
    raw_text=(
        "DISCHARGE SUMMARY - L02\n"
        "Discharge Medications:\n"
        "Take metformin 500 mg BID.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Primary Care in 14 days.\n"
    ),
    expected_patient_releasable=True,
)

L03_FORMULATION_PRESERVATION = SyntheticLanguageFixture(
    case_id="L03",
    description="Medication formulation preservation (tartrate)",
    raw_text=(
        "DISCHARGE SUMMARY - L03\n"
        "Discharge Medications:\n"
        "Continue metoprolol tartrate 25 mg twice daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology in 7 days.\n"
    ),
    expected_patient_releasable=True,
)

L04_FOLLOWUP_SIMPLIFICATION = SyntheticLanguageFixture(
    case_id="L04",
    description="Follow-up simplification",
    raw_text=(
        "DISCHARGE SUMMARY - L04\n"
        "Discharge Medications:\n"
        "Continue apixaban 5 mg twice daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Hematology in 14 days.\n"
    ),
    expected_patient_releasable=True,
)

L05_RELATIVE_TIMEFRAME = SyntheticLanguageFixture(
    case_id="L05",
    description="Relative timeframe preservation ('within 7 days')",
    raw_text=(
        "DISCHARGE SUMMARY - L05\n"
        "Discharge Medications:\n"
        "Continue atorvastatin 40 mg daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology within 7 days.\n"
    ),
    expected_patient_releasable=True,
)

L06_ABSOLUTE_DATE = SyntheticLanguageFixture(
    case_id="L06",
    description="Absolute calendar date preservation",
    raw_text=(
        "DISCHARGE SUMMARY - L06\n"
        "Discharge Medications:\n"
        "Continue amlodipine 5 mg daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology on March 18, 2027.\n"
    ),
    expected_patient_releasable=True,
)

L07_WARNING_THRESHOLD = SyntheticLanguageFixture(
    case_id="L07",
    description="Warning threshold preservation (3 pounds)",
    raw_text=(
        "DISCHARGE SUMMARY - L07\n"
        "Discharge Medications:\n"
        "Continue furosemide 40 mg daily.\n"
        "Warning Signs:\n"
        "Call clinic if weight increases by more than 3 pounds in 24 hours.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology in 14 days.\n"
    ),
    expected_patient_releasable=True,
)

L08_DECIMAL_THRESHOLD = SyntheticLanguageFixture(
    case_id="L08",
    description="Decimal threshold preservation (101.5 F)",
    raw_text=(
        "DISCHARGE SUMMARY - L08\n"
        "Discharge Medications:\n"
        "Continue aspirin 81 mg daily.\n"
        "Warning Signs:\n"
        "Call clinic if temperature is greater than 101.5 F.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Primary Care in 7 days.\n"
    ),
    expected_patient_releasable=True,
)

L09_NEGATION_PRESERVATION = SyntheticLanguageFixture(
    case_id="L09",
    description="Negation preservation ('Stop lisinopril')",
    raw_text=(
        "DISCHARGE SUMMARY - L09\n"
        "Discharge Medications:\n"
        "Stop lisinopril 10 mg daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Primary Care in 7 days.\n"
    ),
    expected_patient_releasable=False,  # Stop medication regimen change requires review
)

L10_CONDITIONAL_PRESERVATION = SyntheticLanguageFixture(
    case_id="L10",
    description="Conditional statement preservation (chest pain cue flagged by validator for precaution verification)",
    raw_text=(
        "DISCHARGE SUMMARY - L10\n"
        "Discharge Medications:\n"
        "Take nitroglycerin 0.4 mg sublingual as needed for chest pain.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology in 7 days.\n"
    ),
    expected_patient_releasable=False,
)

L11_AMBIGUOUS_SOURCE = SyntheticLanguageFixture(
    case_id="L11",
    description="Ambiguous source with missing dosage",
    raw_text=(
        "DISCHARGE SUMMARY - L11\n"
        "Discharge Medications:\n"
        "Take levothyroxine daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Endocrinology in 30 days.\n"
    ),
    expected_patient_releasable=False,
)

L12_CONFLICTING_MEDICATION = SyntheticLanguageFixture(
    case_id="L12",
    description="Conflicting medication instructions",
    raw_text=(
        "DISCHARGE SUMMARY - L12\n"
        "Hospital Course:\n"
        "Continue lisinopril 10 mg daily.\n"
        "Discharge Medications:\n"
        "Stop lisinopril 10 mg daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Primary Care in 7 days.\n"
    ),
    expected_patient_releasable=False,
)

L13_UNSUPPORTED_MEDICATION = SyntheticLanguageFixture(
    case_id="L13",
    description="Unsupported vague instruction clause",
    raw_text=(
        "DISCHARGE SUMMARY - L13\n"
        "Discharge Medications:\n"
        "Hold clopidogrel 75 mg daily; maybe resume when ready.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology in 7 days.\n"
    ),
    expected_patient_releasable=False,
)

L14_PROVIDER_NEUTRALITY = SyntheticLanguageFixture(
    case_id="L14",
    description="Provider neutrality without subjective claims",
    raw_text=(
        "DISCHARGE SUMMARY - L14\n"
        "Discharge Medications:\n"
        "Continue carvedilol 12.5 mg twice daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology (Dr. Elena Rostova) within 7 days.\n"
    ),
    expected_patient_releasable=True,
)

L15_TRANSLATION_SIMPLE = SyntheticLanguageFixture(
    case_id="L15",
    description="Translation of safe simple instruction to Spanish",
    raw_text=(
        "DISCHARGE SUMMARY - L15\n"
        "Discharge Medications:\n"
        "Continue aspirin 81 mg once daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Primary Care within 7 days.\n"
    ),
    expected_patient_releasable=True,
)

L16_TRANSLATION_NUMBERS_UNITS = SyntheticLanguageFixture(
    case_id="L16",
    description="Translation preserving numbers and units (500 mg, 14 days)",
    raw_text=(
        "DISCHARGE SUMMARY - L16\n"
        "Discharge Medications:\n"
        "Take metformin 500 mg twice daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Primary Care in 14 days.\n"
    ),
    expected_patient_releasable=True,
)

L17_TRANSLATION_NEGATION = SyntheticLanguageFixture(
    case_id="L17",
    description="Translation preserving negation ('Stop lisinopril' -> 'No tome lisinopril')",
    raw_text=(
        "DISCHARGE SUMMARY - L17\n"
        "Discharge Medications:\n"
        "Stop lisinopril 10 mg daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Primary Care in 7 days.\n"
    ),
    expected_patient_releasable=False,
)

L18_TRANSLATION_WARNING = SyntheticLanguageFixture(
    case_id="L18",
    description="Translation preserving warning threshold in Spanish",
    raw_text=(
        "DISCHARGE SUMMARY - L18\n"
        "Discharge Medications:\n"
        "Continue furosemide 40 mg daily.\n"
        "Warning Signs:\n"
        "Call clinic if weight increases by more than 3 pounds in 24 hours.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology in 14 days.\n"
    ),
    expected_patient_releasable=True,
)

L19_TRANSLATION_TIMEFRAME = SyntheticLanguageFixture(
    case_id="L19",
    description="Translation preserving timeframe operator ('within 7 days' -> 'dentro de 7 días')",
    raw_text=(
        "DISCHARGE SUMMARY - L19\n"
        "Discharge Medications:\n"
        "Continue apixaban 5 mg twice daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Hematology within 7 days.\n"
    ),
    expected_patient_releasable=True,
)

L20_SOURCE_PROVENANCE = SyntheticLanguageFixture(
    case_id="L20",
    description="Complete provenance trace from patient item to raw discharge text",
    raw_text=(
        "DISCHARGE SUMMARY - L20\n"
        "Discharge Medications:\n"
        "Continue metoprolol tartrate 25 mg twice daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology (Dr. Elena Rostova) within 7 days.\n"
    ),
    expected_patient_releasable=True,
)

ALL_PHASE6_FIXTURES: list[SyntheticLanguageFixture] = [
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
]
