"""Phase 6.5 Synthetic Test Fixtures (R01 - R18).

Covers:
R01: Explicit appointment date/time -> eligible
R02: Explicit date without time -> eligible date reminder
R03: Within 7 days without reference date -> not eligible for exact date
R04: Within 7 days with trusted reference date -> preserve supported semantics
R05: After 7 days -> distinct semantics
R06: Ambiguous appointment -> not eligible
R07: Conflicting appointment dates -> not eligible
R08: Unsupported provider claim -> not eligible
R09: Patient-releasable false -> not eligible
R10: One-day reminder calculation
R11: Two-hour reminder calculation
R12: Custom valid offset
R13: Custom invalid/negative offset -> rejected
R14: Duplicate reminder -> deduplicated
R15: Cancel reminder does not cancel care task
R16: No response to reminder != missed appointment
R17: Spanish reminder preserves date/time and meaning
R18: Medication instruction cannot accidentally become appointment reminder
"""

from __future__ import annotations

from typing import NamedTuple, Optional

from backend.document_intelligence.types import EvidenceSpan
from backend.task_orchestration.types import (
    OrchestratedTask,
    ParsedTimeframe,
    TaskCategory,
    TaskStatus,
    TimeframeType,
)


class SyntheticReminderFixture(NamedTuple):
    case_id: str
    description: str
    raw_text: str
    expected_eligible: bool
    reference_date: Optional[str] = None
    offset_minutes: int = 1440


R01_EXPLICIT_DATETIME = SyntheticReminderFixture(
    case_id="R01",
    description="Explicit appointment date and time",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology on March 18, 2027 at 10:30 AM.\n"
    ),
    expected_eligible=True,
)

R02_EXPLICIT_DATE_NO_TIME = SyntheticReminderFixture(
    case_id="R02",
    description="Explicit appointment date without time",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with Primary Care on March 18, 2027.\n"
    ),
    expected_eligible=True,
)

R03_RELATIVE_NO_REFERENCE = SyntheticReminderFixture(
    case_id="R03",
    description="Relative window without reference date -> not eligible for calendar date",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology within 7 days.\n"
    ),
    expected_eligible=False,
    reference_date=None,
)

R04_RELATIVE_WITH_REFERENCE = SyntheticReminderFixture(
    case_id="R04",
    description="Relative window with trusted discharge reference date",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology within 7 days.\n"
    ),
    expected_eligible=True,
    reference_date="2027-03-01",
)

R05_AFTER_WINDOW = SyntheticReminderFixture(
    case_id="R05",
    description="After 7 days window (distinct semantics, not within)",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology after 7 days.\n"
    ),
    expected_eligible=False,  # 'after' is open-ended, not exact deadline
    reference_date="2027-03-01",
)

R06_AMBIGUOUS_APPOINTMENT = SyntheticReminderFixture(
    case_id="R06",
    description="Ambiguous timeframe appointment -> not eligible",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology soon.\n"
    ),
    expected_eligible=False,
)

R07_CONFLICTING_APPOINTMENTS = SyntheticReminderFixture(
    case_id="R07",
    description="Conflicting appointment instructions -> blocked / not eligible",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology on March 18, 2027.\n"
        "Do not follow up with Cardiology on March 18, 2027.\n"
    ),
    expected_eligible=False,
)

R08_UNSUPPORTED_PROVIDER_CLAIM = SyntheticReminderFixture(
    case_id="R08",
    description="Unsupported provider claim -> not eligible",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with the best cardiologist in the city on March 18, 2027 at 10:30 AM.\n"
    ),
    expected_eligible=False,
)

R09_NOT_RELEASABLE = SyntheticReminderFixture(
    case_id="R09",
    description="Task marked patient_releasable False -> not eligible",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology on March 18, 2027 at 10:30 AM.\n"
    ),
    expected_eligible=False,
)

R10_ONE_DAY_OFFSET = SyntheticReminderFixture(
    case_id="R10",
    description="One day (1440 min) reminder calculation",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology on March 18, 2027 at 10:30 AM.\n"
    ),
    expected_eligible=True,
    offset_minutes=1440,
)

R11_TWO_HOUR_OFFSET = SyntheticReminderFixture(
    case_id="R11",
    description="Two hours (120 min) reminder calculation",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology on March 18, 2027 at 10:30 AM.\n"
    ),
    expected_eligible=True,
    offset_minutes=120,
)

R12_CUSTOM_VALID_OFFSET = SyntheticReminderFixture(
    case_id="R12",
    description="Custom valid offset (360 min = 6 hours)",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology on March 18, 2027 at 10:30 AM.\n"
    ),
    expected_eligible=True,
    offset_minutes=360,
)

R13_CUSTOM_INVALID_OFFSET = SyntheticReminderFixture(
    case_id="R13",
    description="Custom invalid negative offset -> rejected",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology on March 18, 2027 at 10:30 AM.\n"
    ),
    expected_eligible=False,
    offset_minutes=-60,
)

R14_DUPLICATE_DEDUPLICATION = SyntheticReminderFixture(
    case_id="R14",
    description="Duplicate reminder request -> deduplicated to single reminder",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology on March 18, 2027 at 10:30 AM.\n"
    ),
    expected_eligible=True,
    offset_minutes=1440,
)

R15_CANCEL_DOES_NOT_CANCEL_TASK = SyntheticReminderFixture(
    case_id="R15",
    description="Cancelling reminder does not cancel underlying clinical task",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology on March 18, 2027 at 10:30 AM.\n"
    ),
    expected_eligible=True,
    offset_minutes=1440,
)

R16_NO_RESPONSE_NOT_MISSED = SyntheticReminderFixture(
    case_id="R16",
    description="No response to reminder enquiry is never classified as missed appointment",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology on March 18, 2027 at 10:30 AM.\n"
    ),
    expected_eligible=True,
    offset_minutes=1440,
)

R17_SPANISH_REMINDER = SyntheticReminderFixture(
    case_id="R17",
    description="Spanish reminder preserves exact date, time, and clinical meaning",
    raw_text=(
        "Discharge Summary:\n"
        "Follow-Up Appointments:\n"
        "Follow up with Cardiology on March 18, 2027 at 10:30 AM.\n"
    ),
    expected_eligible=True,
    offset_minutes=1440,
)

R18_MEDICATION_CANNOT_BE_APPOINTMENT = SyntheticReminderFixture(
    case_id="R18",
    description="Medication instruction cannot accidentally become an appointment reminder",
    raw_text=(
        "Discharge Summary:\n"
        "Discharge Medications:\n"
        "Continue aspirin 81 mg daily.\n"
    ),
    expected_eligible=False,
)

ALL_CARE_REMINDER_FIXTURES: list[SyntheticReminderFixture] = [
    R01_EXPLICIT_DATETIME,
    R02_EXPLICIT_DATE_NO_TIME,
    R03_RELATIVE_NO_REFERENCE,
    R04_RELATIVE_WITH_REFERENCE,
    R05_AFTER_WINDOW,
    R06_AMBIGUOUS_APPOINTMENT,
    R07_CONFLICTING_APPOINTMENTS,
    R08_UNSUPPORTED_PROVIDER_CLAIM,
    R09_NOT_RELEASABLE,
    R10_ONE_DAY_OFFSET,
    R11_TWO_HOUR_OFFSET,
    R12_CUSTOM_VALID_OFFSET,
    R13_CUSTOM_INVALID_OFFSET,
    R14_DUPLICATE_DEDUPLICATION,
    R15_CANCEL_DOES_NOT_CANCEL_TASK,
    R16_NO_RESPONSE_NOT_MISSED,
    R17_SPANISH_REMINDER,
    R18_MEDICATION_CANNOT_BE_APPOINTMENT,
]


def get_fixture_task(case_id: str) -> OrchestratedTask:
    """Return an authoritative OrchestratedTask representing the given fixture scenario."""
    cid = case_id.upper()
    if cid in ("R01", "R10", "R11", "R12", "R13", "R14", "R15", "R16", "R17"):
        ev_text = "Follow up with Cardiology on March 18, 2027 at 10:30 AM."
        return OrchestratedTask(
            task_id="task-r01",
            task_category=TaskCategory.FOLLOW_UP,
            title="Follow-up appointment with Cardiology",
            description="Schedule and attend follow-up appointment with Cardiology on March 18, 2027 at 10:30 AM.",
            status=TaskStatus.SCHEDULED,
            timeframe=ParsedTimeframe(
                raw_text="on March 18, 2027 at 10:30 AM",
                timeframe_type=TimeframeType.ABSOLUTE_DATE,
                absolute_date="2027-03-18",
            ),
            due_date="2027-03-18",
            source_item_id="fu-r01",
            evidence=EvidenceSpan(
                sentence_id="sent-001",
                section="Follow-Up Appointments",
                source_text=ev_text,
                start_offset=0,
                end_offset=len(ev_text),
            ),
            patient_releasable=True,
            requires_human_review=False,
            clinical_metadata={"specialty": "Cardiology", "provider_name": None},
        )

    if cid == "R02":
        ev_text = "Follow up with Primary Care on March 18, 2027."
        return OrchestratedTask(
            task_id="task-r02",
            task_category=TaskCategory.FOLLOW_UP,
            title="Follow-up appointment with Primary Care",
            description="Schedule and attend follow-up appointment with Primary Care on March 18, 2027.",
            status=TaskStatus.SCHEDULED,
            timeframe=ParsedTimeframe(
                raw_text="on March 18, 2027",
                timeframe_type=TimeframeType.ABSOLUTE_DATE,
                absolute_date="2027-03-18",
            ),
            due_date="2027-03-18",
            source_item_id="fu-r02",
            evidence=EvidenceSpan(
                sentence_id="sent-001",
                section="Follow-Up Appointments",
                source_text=ev_text,
                start_offset=0,
                end_offset=len(ev_text),
            ),
            patient_releasable=True,
            requires_human_review=False,
            clinical_metadata={"specialty": "Primary Care", "provider_name": None},
        )

    if cid in ("R03", "R04"):
        ev_text = "Follow up with Cardiology within 7 days."
        return OrchestratedTask(
            task_id="task-r03",
            task_category=TaskCategory.FOLLOW_UP,
            title="Follow-up appointment with Cardiology",
            description="Schedule and attend follow-up appointment with Cardiology within 7 days.",
            status=TaskStatus.SCHEDULED,
            timeframe=ParsedTimeframe(
                raw_text="within 7 days",
                timeframe_type=TimeframeType.RELATIVE_WINDOW,
                relative_window="within 7 days",
                relative_operator="within",
                relative_duration_days=7,
            ),
            due_window="within 7 days",
            source_item_id="fu-r03",
            evidence=EvidenceSpan(
                sentence_id="sent-001",
                section="Follow-Up Appointments",
                source_text=ev_text,
                start_offset=0,
                end_offset=len(ev_text),
            ),
            patient_releasable=True,
            requires_human_review=False,
            clinical_metadata={"specialty": "Cardiology", "provider_name": None},
        )

    if cid == "R05":
        ev_text = "Follow up with Cardiology after 7 days."
        return OrchestratedTask(
            task_id="task-r05",
            task_category=TaskCategory.FOLLOW_UP,
            title="Follow-up appointment with Cardiology",
            description="Schedule and attend follow-up appointment with Cardiology after 7 days.",
            status=TaskStatus.SCHEDULED,
            timeframe=ParsedTimeframe(
                raw_text="after 7 days",
                timeframe_type=TimeframeType.RELATIVE_WINDOW,
                relative_window="after 7 days",
                relative_operator="after",
                relative_duration_days=7,
            ),
            due_window="after 7 days",
            source_item_id="fu-r05",
            evidence=EvidenceSpan(
                sentence_id="sent-001",
                section="Follow-Up Appointments",
                source_text=ev_text,
                start_offset=0,
                end_offset=len(ev_text),
            ),
            patient_releasable=True,
            requires_human_review=False,
            clinical_metadata={"specialty": "Cardiology", "provider_name": None},
        )

    if cid == "R06":
        ev_text = "Follow up with Cardiology soon."
        return OrchestratedTask(
            task_id="task-r06",
            task_category=TaskCategory.FOLLOW_UP,
            title="Follow-up appointment with Cardiology",
            description="Schedule and attend follow-up appointment with Cardiology soon.",
            status=TaskStatus.SCHEDULED,
            timeframe=ParsedTimeframe(
                raw_text="soon",
                timeframe_type=TimeframeType.UNSPECIFIED,
                is_missing=True,
            ),
            source_item_id="fu-r06",
            evidence=EvidenceSpan(
                sentence_id="sent-001",
                section="Follow-Up Appointments",
                source_text=ev_text,
                start_offset=0,
                end_offset=len(ev_text),
            ),
            patient_releasable=True,
            requires_human_review=False,
            clinical_metadata={"specialty": "Cardiology", "provider_name": None},
        )

    if cid == "R07":
        ev_text = "Follow up with Cardiology on March 18, 2027. Do not follow up on March 18, 2027."
        return OrchestratedTask(
            task_id="task-r07",
            task_category=TaskCategory.FOLLOW_UP,
            title="Follow-up appointment with Cardiology",
            description="Conflicting follow-up orders.",
            status=TaskStatus.BLOCKED,
            timeframe=ParsedTimeframe(
                raw_text="on March 18, 2027",
                timeframe_type=TimeframeType.ABSOLUTE_DATE,
                absolute_date="2027-03-18",
            ),
            due_date="2027-03-18",
            source_item_id="fu-r07",
            evidence=EvidenceSpan(
                sentence_id="sent-001",
                section="Follow-Up Appointments",
                source_text=ev_text,
                start_offset=0,
                end_offset=len(ev_text),
            ),
            patient_releasable=False,
            requires_human_review=True,
            review_reasons=["Conflicting appointment instructions."],
            clinical_metadata={"specialty": "Cardiology", "provider_name": None},
        )

    if cid == "R08":
        ev_text = "Follow up with the best cardiologist in the city on March 18, 2027 at 10:30 AM."
        return OrchestratedTask(
            task_id="task-r08",
            task_category=TaskCategory.FOLLOW_UP,
            title="Follow-up appointment with Cardiology",
            description="Follow up with the best cardiologist.",
            status=TaskStatus.NEEDS_REVIEW,
            timeframe=ParsedTimeframe(
                raw_text="on March 18, 2027 at 10:30 AM",
                timeframe_type=TimeframeType.ABSOLUTE_DATE,
                absolute_date="2027-03-18",
            ),
            due_date="2027-03-18",
            source_item_id="fu-r08",
            evidence=EvidenceSpan(
                sentence_id="sent-001",
                section="Follow-Up Appointments",
                source_text=ev_text,
                start_offset=0,
                end_offset=len(ev_text),
            ),
            patient_releasable=False,
            requires_human_review=True,
            review_reasons=["Unsupported provider claim: 'best cardiologist'"],
            clinical_metadata={"specialty": "Cardiology", "provider_name": "best cardiologist"},
        )

    if cid == "R09":
        ev_text = "Follow up with Cardiology on March 18, 2027 at 10:30 AM."
        return OrchestratedTask(
            task_id="task-r09",
            task_category=TaskCategory.FOLLOW_UP,
            title="Follow-up appointment with Cardiology",
            description="Follow-up appointment with Cardiology",
            status=TaskStatus.NEEDS_REVIEW,
            timeframe=ParsedTimeframe(
                raw_text="on March 18, 2027 at 10:30 AM",
                timeframe_type=TimeframeType.ABSOLUTE_DATE,
                absolute_date="2027-03-18",
            ),
            due_date="2027-03-18",
            source_item_id="fu-r09",
            evidence=EvidenceSpan(
                sentence_id="sent-001",
                section="Follow-Up Appointments",
                source_text=ev_text,
                start_offset=0,
                end_offset=len(ev_text),
            ),
            patient_releasable=False,
            requires_human_review=True,
            review_reasons=["Clinician verification required."],
            clinical_metadata={"specialty": "Cardiology", "provider_name": None},
        )

    if cid == "R18":
        ev_text = "Continue aspirin 81 mg daily."
        return OrchestratedTask(
            task_id="task-r18",
            task_category=TaskCategory.MEDICATION,
            title="Continue aspirin 81 mg",
            description="Keep taking aspirin 81 mg daily.",
            status=TaskStatus.SCHEDULED,
            timeframe=ParsedTimeframe(
                raw_text="daily",
                timeframe_type=TimeframeType.RECURRING,
                recurring_pattern="daily",
            ),
            source_item_id="med-r18",
            evidence=EvidenceSpan(
                sentence_id="sent-001",
                section="Discharge Medications",
                source_text=ev_text,
                start_offset=0,
                end_offset=len(ev_text),
            ),
            patient_releasable=True,
            requires_human_review=False,
            clinical_metadata={
                "medication_name": "aspirin",
                "dosage": "81 mg",
                "frequency": "daily",
                "action": "CONTINUE",
            },
        )

    raise ValueError(f"Unknown fixture case ID: {case_id}")
