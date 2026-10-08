"""
tests/test_phase41_absolute_followup.py

Phase 4.1 — Absolute Follow-up Date/Time Extraction Tests.

Tests F41-01 through F41-10 and the full end-to-end reminder creation path.

DO NOT modify backend/safety/validator.py or backend/schemas/models.py.
"""
from __future__ import annotations

import re
import unittest
from datetime import datetime, timezone
from typing import Optional

from backend.document_intelligence import run_document_intelligence_pipeline
from backend.document_intelligence.types import ExtractedFollowUpRecord
from backend.task_orchestration import orchestrate_care_journey
from backend.task_orchestration.types import TimeframeType, TaskCategory


# ---------------------------------------------------------------------------
# Helper: run Phase 4 on a synthetic one-sentence discharge text
# ---------------------------------------------------------------------------

def _extract_followups(sentence: str) -> list[ExtractedFollowUpRecord]:
    result = run_document_intelligence_pipeline(sentence)
    return result.follow_ups


def _with_med_context(sentence: str) -> str:
    """Wrap sentence in enough medication context to satisfy the safety validator."""
    return (
        "Discharge Instructions:\n"
        "Take amlodipine 5 mg by mouth once daily for blood pressure.\n"
        "Follow-Up:\n"
        + sentence
    )


def _run_full_pipeline(text: str):
    """Returns (DocumentIntelligenceResult, CareJourney)."""
    ext = run_document_intelligence_pipeline(text)
    journey = orchestrate_care_journey(ext)
    return ext, journey


# ---------------------------------------------------------------------------
# F41-01 — Absolute date WITH time
# ---------------------------------------------------------------------------

class TestF4101AbsoluteDateWithTime(unittest.TestCase):

    def setUp(self):
        self.fus = _extract_followups(
            "Follow up with Cardiology on March 18, 2027 at 10:30 AM."
        )

    def test_one_followup_extracted(self):
        self.assertEqual(len(self.fus), 1, "Expected exactly one follow-up")

    def test_specialty_is_cardiology(self):
        self.assertEqual(self.fus[0].specialty, "Cardiology")

    def test_absolute_date_in_timeframe(self):
        tf = self.fus[0].timeframe
        self.assertIsNotNone(tf)
        self.assertIn("March 18, 2027", tf)
        self.assertRegex(tf, r"(?i)\bon\b")  # starts with "on"

    def test_time_is_captured_in_timeframe_string(self):
        # After the fix, timeframe must include "at 10:30 AM" so both sides get "IN" qualifier
        tf = self.fus[0].timeframe
        self.assertIn("10:30", tf)

    def test_appointment_time_extracted(self):
        t = self.fus[0].appointment_time
        self.assertIsNotNone(t, "appointment_time must be set for an absolute date with time")
        self.assertEqual(t, "10:30")

    def test_not_relative_timeframe(self):
        tf = self.fus[0].timeframe or ""
        self.assertNotRegex(tf, r"(?i)\bin\s+\d+", "Should not look like a relative offset")


# ---------------------------------------------------------------------------
# F41-02 — Absolute date WITHOUT time
# ---------------------------------------------------------------------------

class TestF4102AbsoluteDateNoTime(unittest.TestCase):

    def setUp(self):
        self.fus = _extract_followups(
            "Follow up with Cardiology on March 18, 2027."
        )

    def test_date_in_timeframe(self):
        tf = self.fus[0].timeframe
        self.assertIsNotNone(tf)
        self.assertIn("March 18, 2027", tf)

    def test_appointment_time_absent(self):
        self.assertIsNone(
            self.fus[0].appointment_time,
            "appointment_time should be None when no time is stated",
        )


# ---------------------------------------------------------------------------
# F41-03 — Relative: "in 7 days"
# ---------------------------------------------------------------------------

class TestF4103RelativeIn(unittest.TestCase):

    def setUp(self):
        self.fus = _extract_followups(
            "Follow up with Cardiology in 7 days."
        )

    def test_timeframe_preserved(self):
        tf = self.fus[0].timeframe
        self.assertIsNotNone(tf)
        self.assertRegex(tf, r"(?i)\bin\s+7\s+days?\b")

    def test_appointment_time_absent(self):
        self.assertIsNone(self.fus[0].appointment_time)


# ---------------------------------------------------------------------------
# F41-04 — Relative: "within 7 days" — WITHIN preserved
# ---------------------------------------------------------------------------

class TestF4104RelativeWithin(unittest.TestCase):

    def setUp(self):
        self.fus = _extract_followups(
            "Follow up with Cardiology within 7 days."
        )

    def test_within_preserved(self):
        tf = self.fus[0].timeframe
        self.assertIsNotNone(tf)
        self.assertRegex(tf, r"(?i)\bwithin\s+7\s+days?\b")

    def test_not_collapsed_to_in(self):
        tf = (self.fus[0].timeframe or "").lower()
        self.assertTrue(tf.startswith("within"), f"Should start with 'within', got: {tf}")


# ---------------------------------------------------------------------------
# F41-05 — Relative: "after 7 days" — AFTER preserved
# ---------------------------------------------------------------------------

class TestF4105RelativeAfter(unittest.TestCase):

    def setUp(self):
        self.fus = _extract_followups(
            "Follow up with Cardiology after 7 days."
        )

    def test_after_preserved(self):
        tf = self.fus[0].timeframe
        self.assertIsNotNone(tf)
        self.assertRegex(tf, r"(?i)\bafter\s+7\s+days?\b")

    def test_not_collapsed_to_within(self):
        tf = (self.fus[0].timeframe or "").lower()
        self.assertNotIn("within", tf)


# ---------------------------------------------------------------------------
# F41-06 — Vague: "soon" — ambiguous / non-releasable
# ---------------------------------------------------------------------------

class TestF4106VagueSoon(unittest.TestCase):

    def test_ambiguous_or_no_timeframe(self):
        fus = _extract_followups("Follow up with Cardiology soon.")
        if not fus:
            return  # no extraction → trivially not releasable
        fu = fus[0]
        # "soon" must either not match TIMEFRAME_RE or mark is_ambiguous
        self.assertTrue(
            fu.timeframe is None or fu.is_ambiguous,
            "A 'soon' timeframe should be missing or flagged as ambiguous",
        )

    def test_full_pipeline_non_releasable(self):
        ext, journey = _run_full_pipeline(_with_med_context("Follow up with Cardiology soon."))
        fu_tasks = [t for t in journey.timeline.all_tasks if t.task_category == TaskCategory.FOLLOW_UP]
        self.assertTrue(len(fu_tasks) > 0)
        self.assertFalse(
            fu_tasks[0].patient_releasable,
            "Vague 'soon' follow-up must not be patient-releasable",
        )


# ---------------------------------------------------------------------------
# F41-07 — Evidence-span verification
# ---------------------------------------------------------------------------

class TestF4107EvidenceSpan(unittest.TestCase):

    RAW = "Follow up with Cardiology on March 18, 2027 at 10:30 AM."

    def setUp(self):
        self.ext = run_document_intelligence_pipeline(self.RAW)
        # Evidence offsets are relative to document.normalized_text
        self.raw = self.ext.document.normalized_text

    def test_evidence_span_matches_raw_slice(self):
        fu = self.ext.follow_ups[0]
        ev = fu.evidence
        sliced = self.raw[ev.start_offset: ev.end_offset]
        self.assertEqual(
            sliced,
            ev.source_text,
            f"Evidence slice mismatch:\n  slice={sliced!r}\n  stored={ev.source_text!r}",
        )

    def test_evidence_contains_cardiology(self):
        ev = self.ext.follow_ups[0].evidence
        self.assertIn("Cardiology", ev.source_text)

    def test_evidence_contains_march_18(self):
        ev = self.ext.follow_ups[0].evidence
        self.assertIn("March 18", ev.source_text)


# ---------------------------------------------------------------------------
# F41-08 — Two distinct follow-ups with correct dates/times
# ---------------------------------------------------------------------------

class TestF4108TwoFollowUps(unittest.TestCase):

    RAW = (
        "Follow up with Cardiology on March 18, 2027 at 10:30 AM.\n"
        "Follow up with Nephrology on April 2, 2027 at 2 PM."
    )

    def setUp(self):
        self.fus = _extract_followups(self.RAW)

    def test_two_followups_extracted(self):
        self.assertEqual(len(self.fus), 2, "Expected exactly two follow-ups")

    def test_specialties_distinct(self):
        specs = {fu.specialty for fu in self.fus}
        self.assertIn("Cardiology", specs)
        self.assertIn("Nephrology", specs)

    def test_dates_distinct(self):
        tfs = [fu.timeframe or "" for fu in self.fus]
        self.assertTrue(any("March 18" in t for t in tfs), "March 18 not found")
        self.assertTrue(any("April 2" in t for t in tfs), "April 2 not found")

    def test_times_extracted(self):
        times = {fu.appointment_time for fu in self.fus if fu.appointment_time}
        self.assertIn("10:30", times, "10:30 AM should be captured")
        self.assertIn("14:00", times, "2 PM should normalise to 14:00")


# ---------------------------------------------------------------------------
# F41-09 — Complete pipeline: Phase 4 → 5 → 6 → 6.5
# ---------------------------------------------------------------------------

class TestF4109CompletePipeline(unittest.TestCase):

    RAW = _with_med_context("Follow up with Cardiology on March 18, 2027 at 10:30 AM.")

    def setUp(self):
        from backend.patient_language import generate_patient_care_plan
        from backend.care_reminders.eligibility import check_reminder_eligibility

        self.ext, self.journey = _run_full_pipeline(self.RAW)
        self.plan = generate_patient_care_plan(self.journey)
        self.eligibility_results = {}
        for task in self.journey.timeline.all_tasks:
            if task.task_category == TaskCategory.FOLLOW_UP:
                self.eligibility_results[task.task_id] = check_reminder_eligibility(task)

    # Phase 4
    def test_phase4_extracts_cardiology(self):
        specs = [fu.specialty for fu in self.ext.follow_ups]
        self.assertIn("Cardiology", specs)

    def test_phase4_absolute_date(self):
        fu = next(fu for fu in self.ext.follow_ups if fu.specialty == "Cardiology")
        self.assertIn("2027", fu.timeframe or "")

    def test_phase4_appointment_time_10_30(self):
        fu = next(fu for fu in self.ext.follow_ups if fu.specialty == "Cardiology")
        self.assertEqual(fu.appointment_time, "10:30")

    # Phase 5
    def test_phase5_absolute_date_task(self):
        fu_tasks = [t for t in self.journey.timeline.all_tasks
                    if t.task_category == TaskCategory.FOLLOW_UP]
        self.assertTrue(any(
            t.timeframe.timeframe_type == TimeframeType.ABSOLUTE_DATE for t in fu_tasks
        ))

    def test_phase5_task_not_needs_review_due_to_qualifier_mismatch(self):
        """The fix must ensure ON-vs-IN mismatch no longer blocks the task."""
        fu_tasks = [t for t in self.journey.timeline.all_tasks
                    if t.task_category == TaskCategory.FOLLOW_UP]
        self.assertTrue(len(fu_tasks) > 0)
        fu_task = fu_tasks[0]
        qualifier_msgs = [r for r in fu_task.review_reasons
                          if "qualifier" in r.lower() or "unsupported_field_value" in r.lower()]
        self.assertEqual(
            qualifier_msgs, [],
            f"Task should not have qualifier-mismatch review reasons, got: {fu_task.review_reasons}",
        )

    def test_phase5_task_patient_releasable(self):
        fu_tasks = [t for t in self.journey.timeline.all_tasks
                    if t.task_category == TaskCategory.FOLLOW_UP]
        self.assertTrue(fu_tasks[0].patient_releasable,
                        f"Follow-up task should be patient-releasable, reasons: {fu_tasks[0].review_reasons}")

    def test_phase5_absolute_date_is_2027_03_18(self):
        fu_tasks = [t for t in self.journey.timeline.all_tasks
                    if t.task_category == TaskCategory.FOLLOW_UP]
        self.assertEqual(fu_tasks[0].timeframe.absolute_date, "2027-03-18")

    def test_phase5_appointment_time_in_clinical_metadata(self):
        fu_tasks = [t for t in self.journey.timeline.all_tasks
                    if t.task_category == TaskCategory.FOLLOW_UP]
        meta = fu_tasks[0].clinical_metadata
        self.assertEqual(meta.get("appointment_time"), "10:30")

    # Phase 6
    def test_phase6_plan_not_empty(self):
        self.assertTrue(len(self.plan.items) > 0)

    def test_phase6_followup_item_exists(self):
        from backend.task_orchestration.types import TaskCategory as TC
        fu_items = [i for i in self.plan.items if i.category == TC.FOLLOW_UP]
        self.assertTrue(len(fu_items) > 0, "Plan should contain a follow-up item")

    # Phase 6.5
    def test_phase65_followup_task_eligible_for_reminder(self):
        fu_tasks = [t for t in self.journey.timeline.all_tasks
                    if t.task_category == TaskCategory.FOLLOW_UP]
        self.assertTrue(len(fu_tasks) > 0)
        result = self.eligibility_results[fu_tasks[0].task_id]
        self.assertTrue(result.is_eligible,
                        f"Task should be eligible for reminder, reason: {result.rejection_reasons}")


# ---------------------------------------------------------------------------
# F41-10 — Ambiguous numeric date (03/04/2027) — must NOT silently resolve
# ---------------------------------------------------------------------------

class TestF4110AmbiguousNumericDate(unittest.TestCase):

    def test_numeric_date_not_silently_resolved(self):
        """
        '03/04/2027' is ambiguous (March 4 vs April 3).
        The extractor must not emit an absolute_date for it via Phase 5.
        Either the timeframe is None/ambiguous, or it's preserved as a string without
        converting to ABSOLUTE_DATE with a fabricated month choice.
        """
        src = _with_med_context("Follow up with Cardiology on 03/04/2027.")
        ext, journey = _run_full_pipeline(src)
        fu_tasks = [t for t in journey.timeline.all_tasks
                    if t.task_category == TaskCategory.FOLLOW_UP]
        if not fu_tasks:
            return  # no follow-up extracted → not silently resolved ✓
        for task in fu_tasks:
            # Must NOT be ABSOLUTE_DATE with a silently chosen month
            if task.timeframe.timeframe_type == TimeframeType.ABSOLUTE_DATE:
                # If it IS parsed as absolute, the date must come from an unambiguous string
                # The only acceptable absolute would be if the system had a locale rule.
                # In the absence of such a rule, fail.
                self.fail(
                    f"Ambiguous numeric date '03/04/2027' was silently resolved to "
                    f"absolute_date={task.timeframe.absolute_date!r}. "
                    "This is forbidden — the date must be preserved as-is or flagged as ambiguous."
                )


# ---------------------------------------------------------------------------
# Regression: Reminder creation path (Spec §12)
# ---------------------------------------------------------------------------

class TestPhase41ReminderCreation(unittest.TestCase):
    """
    Full reminder creation path for F41-09 primary test case.
    Expected:
      event_time  = 2027-03-18 10:30
      1-day offset reminder = 2027-03-17 10:30
      provenance valid
      patient_releasable = True
    """

    def setUp(self):
        from backend.patient_language import generate_patient_care_plan
        from backend.care_reminders.eligibility import check_reminder_eligibility, is_reminder_eligible
        from backend.care_reminders.scheduler import create_reminder
        from backend.care_reminders.provenance import validate_reminder_provenance
        from backend.care_reminders.types import ReminderPreset

        raw = _with_med_context("Follow up with Cardiology on March 18, 2027 at 10:30 AM.")
        ext, journey = _run_full_pipeline(raw)

        fu_tasks = [t for t in journey.timeline.all_tasks
                    if t.task_category == TaskCategory.FOLLOW_UP]
        self.assertTrue(len(fu_tasks) > 0, "Need at least one follow-up task")
        self.task = fu_tasks[0]

        self.eligibility = check_reminder_eligibility(self.task)
        if self.eligibility.is_eligible:
            self.reminder = create_reminder(self.task, ReminderPreset.ONE_DAY)
            self.provenance_ok = validate_reminder_provenance(self.reminder, self.task)
        else:
            self.reminder = None
            self.provenance_ok = None

    def test_task_eligible(self):
        self.assertTrue(
            self.eligibility.is_eligible,
            f"Task must be eligible for reminder. Reason: {self.eligibility.rejection_reasons}",
        )

    def test_reminder_created(self):
        self.assertIsNotNone(self.reminder, "Reminder must be created for eligible task")

    def test_reminder_patient_releasable(self):
        self.assertTrue(self.reminder.patient_releasable)

    def test_reminder_event_time_is_2027_03_18_10_30(self):
        ev = self.reminder.event_time  # ISO string e.g. '2027-03-18T10:30:00'
        self.assertIn('2027-03-18', ev, f'event_time={ev!r} must include 2027-03-18')
        self.assertIn('10:30', ev, f'event_time={ev!r} must include 10:30')

    def test_one_day_offset_scheduled_for_2027_03_17_10_30(self):
        sched = self.reminder.scheduled_for  # ISO string e.g. '2027-03-17T10:30:00'
        self.assertIn('2027-03-17', sched, f'scheduled_for={sched!r} must include 2027-03-17')
        self.assertIn('10:30', sched, f'scheduled_for={sched!r} must include 10:30')

    def test_provenance_valid(self):
        self.assertTrue(
            self.provenance_ok,
            "Reminder provenance must be valid (evidence, task_id, math must all match)",
        )

    def test_no_clinical_info_changed(self):
        # The reminder title should reference Cardiology and the date, not invent new content
        title = self.reminder.title or ""
        self.assertNotIn("invented", title.lower())
        # Must not reference a date that doesn't exist in the source
        self.assertNotIn("March 19", title)
        self.assertNotIn("March 17", title)  # Reminder title = event date, not offset date


if __name__ == "__main__":
    unittest.main()
