"""Phase 5 Synthetic Test Fixtures (P01 - P15).

Defined to validate task generation, timeframe semantics, conflict handling,
deduplication invariants, and provenance audit across clinical domains.
"""

from __future__ import annotations

from typing import NamedTuple


class SyntheticOrchestrationFixture(NamedTuple):
    case_id: str
    description: str
    raw_text: str
    expected_patient_releasable: bool


P01_NORMAL_MEDS = SyntheticOrchestrationFixture(
    case_id="P01",
    description="Normal single medication regimen",
    raw_text=(
        "DISCHARGE SUMMARY - P01\n"
        "Discharge Medications:\n"
        "Continue aspirin 81 mg once daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Primary Care within 14 days.\n"
    ),
    expected_patient_releasable=True,
)

P02_MULTI_MEDS_FOLLOWUPS = SyntheticOrchestrationFixture(
    case_id="P02",
    description="Multiple medications with multiple follow-ups",
    raw_text=(
        "DISCHARGE SUMMARY - P02\n"
        "Discharge Medications:\n"
        "Continue atorvastatin 40 mg daily.\n"
        "Continue amlodipine 5 mg daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology in 7 days.\n"
        "Follow up with Primary Care in 14 days.\n"
    ),
    expected_patient_releasable=True,
)

P03_RELATIVE_TIMEFRAME = SyntheticOrchestrationFixture(
    case_id="P03",
    description="Explicit relative timeframe 'within 7 days'",
    raw_text=(
        "DISCHARGE SUMMARY - P03\n"
        "Discharge Medications:\n"
        "Continue apixaban 5 mg twice daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Hematology within 7 days.\n"
    ),
    expected_patient_releasable=True,
)

P04_ABSOLUTE_DATE = SyntheticOrchestrationFixture(
    case_id="P04",
    description="Explicit absolute calendar date",
    raw_text=(
        "DISCHARGE SUMMARY - P04\n"
        "Discharge Medications:\n"
        "Continue metoprolol succinate 50 mg daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology on March 18, 2027.\n"
    ),
    expected_patient_releasable=True,
)

P05_MISSING_TIMEFRAME = SyntheticOrchestrationFixture(
    case_id="P05",
    description="Follow-up with missing timeframe",
    raw_text=(
        "DISCHARGE SUMMARY - P05\n"
        "Discharge Medications:\n"
        "Continue lisinopril 10 mg daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology for routine re-evaluation.\n"
    ),
    expected_patient_releasable=False,
)

P06_CONFLICTING_MEDS = SyntheticOrchestrationFixture(
    case_id="P06",
    description="Conflicting medication instructions",
    raw_text=(
        "DISCHARGE SUMMARY - P06\n"
        "Hospital Course:\n"
        "Continue lisinopril 10 mg daily.\n"
        "Discharge Medications:\n"
        "Stop lisinopril 10 mg daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Primary Care in 7 days.\n"
    ),
    expected_patient_releasable=False,
)

P07_DUPLICATE_TASKS = SyntheticOrchestrationFixture(
    case_id="P07",
    description="Repeated duplicate medication instruction (flagged by validator as DUPLICATE_ITEM)",
    raw_text=(
        "DISCHARGE SUMMARY - P07\n"
        "Discharge Medications:\n"
        "1. Continue aspirin 81 mg once daily.\n"
        "2. Continue aspirin 81 mg once daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Primary Care in 14 days.\n"
    ),
    expected_patient_releasable=False,
)

P08_DIFFERENT_TIMEFRAMES = SyntheticOrchestrationFixture(
    case_id="P08",
    description="Clinically distinct timeframes ('within 7 days' vs 'after 7 days') flagged as CONFLICTING_INSTRUCTIONS",
    raw_text=(
        "DISCHARGE SUMMARY - P08\n"
        "Discharge Medications:\n"
        "Continue carvedilol 12.5 mg twice daily.\n"
        "Follow-Up Appointments:\n"
        "- Follow up with Cardiology within 7 days.\n"
        "- Follow up with Cardiology after 7 days.\n"
    ),
    expected_patient_releasable=False,
)

P09_TEST_REFERRAL_FOLLOWUP = SyntheticOrchestrationFixture(
    case_id="P09",
    description="Diagnostic test, referral, and follow-up",
    raw_text=(
        "DISCHARGE SUMMARY - P09\n"
        "Discharge Medications:\n"
        "Continue apixaban 5 mg twice daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology in 14 days.\n"
        "Tests:\n"
        "Obtain basic metabolic panel in 5 days.\n"
        "Referrals:\n"
        "Referral to Physical Therapy in 7 days.\n"
    ),
    expected_patient_releasable=True,
)

P10_WARNING_MONITORING = SyntheticOrchestrationFixture(
    case_id="P10",
    description="Warning-sign monitoring task",
    raw_text=(
        "DISCHARGE SUMMARY - P10\n"
        "Discharge Medications:\n"
        "Continue furosemide 40 mg daily.\n"
        "Warning Signs:\n"
        "Call clinic if weight increases by more than 3 pounds in 24 hours.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology in 14 days.\n"
    ),
    expected_patient_releasable=True,
)

P11_AMBIGUOUS_ITEM = SyntheticOrchestrationFixture(
    case_id="P11",
    description="Ambiguous medication lacking explicit dosage",
    raw_text=(
        "DISCHARGE SUMMARY - P11\n"
        "Discharge Medications:\n"
        "Take levothyroxine daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Endocrinology in 30 days.\n"
    ),
    expected_patient_releasable=False,
)

P12_UNSUPPORTED_ITEM = SyntheticOrchestrationFixture(
    case_id="P12",
    description="Ambiguous or ungrounded instruction clause",
    raw_text=(
        "DISCHARGE SUMMARY - P12\n"
        "Discharge Medications:\n"
        "Continue aspirin 81 mg daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Dr. Specialist when convenient in future.\n"
    ),
    expected_patient_releasable=False,
)

P13_EXACT_EVIDENCE_SPAN = SyntheticOrchestrationFixture(
    case_id="P13",
    description="Task verification of exact verbatim character offset",
    raw_text=(
        "DISCHARGE SUMMARY - P13\n"
        "Discharge Medications:\n"
        "Continue metoprolol tartrate 25 mg twice daily.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology (Dr. Elena Rostova) within 7 days.\n"
    ),
    expected_patient_releasable=True,
)

P14_BROKEN_AMBIGUOUS_SOURCE = SyntheticOrchestrationFixture(
    case_id="P14",
    description="Vague conditional phrasing ('maybe resume when ready')",
    raw_text=(
        "DISCHARGE SUMMARY - P14\n"
        "Discharge Medications:\n"
        "Hold clopidogrel 75 mg daily; maybe resume when ready.\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology in 7 days.\n"
    ),
    expected_patient_releasable=False,
)

P15_INDEPENDENT_PROVENANCE = SyntheticOrchestrationFixture(
    case_id="P15",
    description="Multiple tasks with completely independent character spans",
    raw_text=(
        "DISCHARGE SUMMARY - P15\n"
        "Discharge Medications:\n"
        "1. Continue Eliquis 5 mg twice daily.\n"
        "2. Continue atorvastatin 40 mg daily.\n"
        "Follow-Up Appointments:\n"
        "- Follow up with Cardiology in 7 days.\n"
        "- Follow up with Primary Care in 14 days.\n"
        "Tests:\n"
        "- Obtain complete blood count in 7 days.\n"
        "Referrals:\n"
        "- Referral to Cardiac Rehabilitation in 14 days.\n"
    ),
    expected_patient_releasable=True,
)

ALL_PHASE5_FIXTURES: list[SyntheticOrchestrationFixture] = [
    P01_NORMAL_MEDS,
    P02_MULTI_MEDS_FOLLOWUPS,
    P03_RELATIVE_TIMEFRAME,
    P04_ABSOLUTE_DATE,
    P05_MISSING_TIMEFRAME,
    P06_CONFLICTING_MEDS,
    P07_DUPLICATE_TASKS,
    P08_DIFFERENT_TIMEFRAMES,
    P09_TEST_REFERRAL_FOLLOWUP,
    P10_WARNING_MONITORING,
    P11_AMBIGUOUS_ITEM,
    P12_UNSUPPORTED_ITEM,
    P13_EXACT_EVIDENCE_SPAN,
    P14_BROKEN_AMBIGUOUS_SOURCE,
    P15_INDEPENDENT_PROVENANCE,
]
