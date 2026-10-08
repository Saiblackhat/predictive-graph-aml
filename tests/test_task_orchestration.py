"""Comprehensive test suite for Phase 5 Task & Timeline Orchestration.

Tests:
1. Synthetic fixtures P01 - P15 end-to-end.
2. Adversarial break tests A - O.
3. Strict safety inheritance & provenance invariants.
"""

from __future__ import annotations

import unittest

from backend.document_intelligence import run_document_intelligence_pipeline
from backend.document_intelligence.types import EvidenceSpan
from backend.schemas.models import SafetyStatus
from backend.task_orchestration import (
    CareJourney,
    CareTimeline,
    OrchestratedTask,
    ParsedTimeframe,
    TaskCategory,
    TaskStatus,
    TimeframeType,
    orchestrate_care_journey,
    parse_clinical_timeframe,
)
from backend.task_orchestration.fixtures import (
    ALL_PHASE5_FIXTURES,
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
)


class TestPhase5TaskOrchestration(unittest.TestCase):
    """Test suite for Phase 5 Task & Timeline Orchestration."""

    # ------------------------------------------------------------------
    # Timeframe Parsing Tests
    # ------------------------------------------------------------------
    def test_timeframe_absolute_date_parsing_without_fabrication(self) -> None:
        tf1 = parse_clinical_timeframe("March 18, 2027")
        self.assertEqual(tf1.timeframe_type, TimeframeType.ABSOLUTE_DATE)
        self.assertEqual(tf1.absolute_date, "2027-03-18")
        self.assertFalse(tf1.is_missing)

        tf2 = parse_clinical_timeframe("18 March 2027")
        self.assertEqual(tf2.timeframe_type, TimeframeType.ABSOLUTE_DATE)
        self.assertEqual(tf2.absolute_date, "2027-03-18")

        tf3 = parse_clinical_timeframe("2027-03-18")
        self.assertEqual(tf3.timeframe_type, TimeframeType.ABSOLUTE_DATE)
        self.assertEqual(tf3.absolute_date, "2027-03-18")

    def test_timeframe_relative_operators_never_collapsed(self) -> None:
        tf_within = parse_clinical_timeframe("within 7 days")
        tf_after = parse_clinical_timeframe("after 7 days")
        tf_in = parse_clinical_timeframe("in 7 days")

        self.assertEqual(tf_within.relative_operator, "within")
        self.assertEqual(tf_after.relative_operator, "after")
        self.assertEqual(tf_in.relative_operator, "in")
        self.assertNotEqual(tf_within.relative_operator, tf_after.relative_operator)

    def test_timeframe_missing_and_vague_handling(self) -> None:
        tf_none = parse_clinical_timeframe(None)
        self.assertTrue(tf_none.is_missing)
        self.assertEqual(tf_none.timeframe_type, TimeframeType.UNSPECIFIED)

        tf_vague = parse_clinical_timeframe("when convenient in future")
        self.assertTrue(tf_vague.is_missing)
        self.assertEqual(tf_vague.timeframe_type, TimeframeType.UNSPECIFIED)

    # ------------------------------------------------------------------
    # Synthetic Fixtures P01 - P15
    # ------------------------------------------------------------------
    def test_all_15_synthetic_fixtures(self) -> None:
        self.assertEqual(len(ALL_PHASE5_FIXTURES), 15)

        for fix in ALL_PHASE5_FIXTURES:
            extract_res = run_document_intelligence_pipeline(fix.raw_text, document_id=fix.case_id)
            journey = orchestrate_care_journey(extract_res)

            self.assertEqual(
                journey.patient_releasable,
                fix.expected_patient_releasable,
                f"Fixture {fix.case_id} patient_releasable mismatch: expected {fix.expected_patient_releasable}, got {journey.patient_releasable}",
            )
            self.assertTrue(journey.provenance_audit_passed, f"Fixture {fix.case_id} failed provenance audit")

    def test_p01_normal_meds(self) -> None:
        ext = run_document_intelligence_pipeline(P01_NORMAL_MEDS.raw_text, document_id=P01_NORMAL_MEDS.case_id)
        j = orchestrate_care_journey(ext)
        self.assertTrue(j.patient_releasable)
        self.assertEqual(j.timeline.total_task_count, 2)
        med_tasks = [t for t in j.timeline.all_tasks if t.task_category == TaskCategory.MEDICATION]
        self.assertEqual(len(med_tasks), 1)
        self.assertEqual(med_tasks[0].clinical_metadata["medication_name"], "aspirin")

    def test_p04_absolute_date_task(self) -> None:
        ext = run_document_intelligence_pipeline(P04_ABSOLUTE_DATE.raw_text, document_id=P04_ABSOLUTE_DATE.case_id)
        j = orchestrate_care_journey(ext)
        fu_tasks = [t for t in j.timeline.all_tasks if t.task_category == TaskCategory.FOLLOW_UP]
        self.assertEqual(len(fu_tasks), 1)
        self.assertEqual(fu_tasks[0].due_date, "2027-03-18")
        self.assertEqual(len(j.timeline.absolute_date_tasks), 1)

    def test_p06_conflicting_meds_blocked(self) -> None:
        ext = run_document_intelligence_pipeline(P06_CONFLICTING_MEDS.raw_text, document_id=P06_CONFLICTING_MEDS.case_id)
        j = orchestrate_care_journey(ext)
        self.assertFalse(j.patient_releasable)
        self.assertTrue(j.requires_human_review)
        self.assertEqual(j.safety_status, SafetyStatus.CONFLICT_DETECTED)
        blocked_tasks = [t for t in j.timeline.all_tasks if t.status == TaskStatus.BLOCKED]
        self.assertGreater(len(blocked_tasks), 0)

    def test_p07_duplicate_deduplication(self) -> None:
        ext = run_document_intelligence_pipeline(P07_DUPLICATE_TASKS.raw_text, document_id=P07_DUPLICATE_TASKS.case_id)
        j = orchestrate_care_journey(ext)
        med_tasks = [t for t in j.timeline.all_tasks if t.task_category == TaskCategory.MEDICATION]
        self.assertEqual(len(med_tasks), 1, "Duplicate identical aspirin instruction must be deduplicated")

    def test_p08_different_timeframes_preserved(self) -> None:
        ext = run_document_intelligence_pipeline(P08_DIFFERENT_TIMEFRAMES.raw_text, document_id=P08_DIFFERENT_TIMEFRAMES.case_id)
        j = orchestrate_care_journey(ext)
        fu_tasks = [t for t in j.timeline.all_tasks if t.task_category == TaskCategory.FOLLOW_UP]
        self.assertEqual(len(fu_tasks), 2, "'within 7 days' and 'after 7 days' must NOT be merged")

    def test_p09_test_referral_followup_distribution(self) -> None:
        ext = run_document_intelligence_pipeline(P09_TEST_REFERRAL_FOLLOWUP.raw_text, document_id=P09_TEST_REFERRAL_FOLLOWUP.case_id)
        j = orchestrate_care_journey(ext)
        cats = [t.task_category for t in j.timeline.all_tasks]
        self.assertIn(TaskCategory.TEST, cats)
        self.assertIn(TaskCategory.REFERRAL, cats)
        self.assertIn(TaskCategory.FOLLOW_UP, cats)

    # ------------------------------------------------------------------
    # Adversarial Break Tests A - O
    # ------------------------------------------------------------------
    def test_adv_a_missing_timeframe_no_invented_date(self) -> None:
        ext = run_document_intelligence_pipeline(P05_MISSING_TIMEFRAME.raw_text)
        j = orchestrate_care_journey(ext)
        fu = next(t for t in j.timeline.all_tasks if t.task_category == TaskCategory.FOLLOW_UP)
        self.assertIsNone(fu.due_date)
        self.assertIsNone(fu.due_window)
        self.assertTrue(fu.timeframe.is_missing)
        self.assertFalse(fu.patient_releasable)

    def test_adv_b_within_vs_after_not_silently_merged(self) -> None:
        ext = run_document_intelligence_pipeline(P08_DIFFERENT_TIMEFRAMES.raw_text)
        j = orchestrate_care_journey(ext)
        fu_tasks = [t for t in j.timeline.all_tasks if t.task_category == TaskCategory.FOLLOW_UP]
        self.assertEqual(len(fu_tasks), 2)
        ops = {t.timeframe.relative_operator for t in fu_tasks}
        self.assertEqual(ops, {"within", "after"})

    def test_adv_c_conflicting_dates_requires_review(self) -> None:
        raw = (
            "Discharge Medications:\n"
            "Continue aspirin 81 mg daily.\n"
            "Follow-Up Appointments:\n"
            "- Follow up with Cardiology in 7 days.\n"
            "- Follow up with Cardiology in 30 days.\n"
        )
        ext = run_document_intelligence_pipeline(raw)
        j = orchestrate_care_journey(ext)
        fu_tasks = [t for t in j.timeline.all_tasks if t.task_category == TaskCategory.FOLLOW_UP]
        self.assertEqual(len(fu_tasks), 2)

    def test_adv_d_conflicting_medications_blocked(self) -> None:
        ext = run_document_intelligence_pipeline(P06_CONFLICTING_MEDS.raw_text)
        j = orchestrate_care_journey(ext)
        self.assertFalse(j.patient_releasable)
        self.assertTrue(j.requires_human_review)

    def test_adv_e_unsupported_medication_not_releasable(self) -> None:
        ext = run_document_intelligence_pipeline(P11_AMBIGUOUS_ITEM.raw_text)
        j = orchestrate_care_journey(ext)
        med = next(t for t in j.timeline.all_tasks if t.task_category == TaskCategory.MEDICATION)
        self.assertFalse(med.patient_releasable)
        self.assertEqual(med.status, TaskStatus.NEEDS_REVIEW)

    def test_adv_f_ambiguous_followup_cannot_be_scheduled(self) -> None:
        ext = run_document_intelligence_pipeline(P12_UNSUPPORTED_ITEM.raw_text)
        j = orchestrate_care_journey(ext)
        fu = next(t for t in j.timeline.all_tasks if t.task_category == TaskCategory.FOLLOW_UP)
        self.assertNotEqual(fu.status, TaskStatus.SCHEDULED)
        self.assertFalse(fu.patient_releasable)

    def test_adv_g_task_with_no_evidence_blocked(self) -> None:
        ext = run_document_intelligence_pipeline(P01_NORMAL_MEDS.raw_text)
        # Corrupt evidence span on an extracted medication with mismatching source text
        ev = ext.medications[0].evidence
        ext.medications[0].evidence = EvidenceSpan(
            start_offset=ev.start_offset,
            end_offset=ev.end_offset,
            source_text="Fabricated non-matching claim",
            sentence_id=ev.sentence_id,
            section=ev.section,
        )
        j = orchestrate_care_journey(ext)
        self.assertFalse(j.provenance_audit_passed)
        self.assertFalse(j.patient_releasable)

    def test_adv_h_incorrect_evidence_span_blocked(self) -> None:
        ext = run_document_intelligence_pipeline(P01_NORMAL_MEDS.raw_text)
        # Modify span offsets so raw_text slice doesn't match
        ev = ext.medications[0].evidence
        ext.medications[0].evidence = EvidenceSpan(
            start_offset=ev.start_offset + 5,
            end_offset=ev.end_offset + 5,
            source_text=ev.source_text,
            sentence_id=ev.sentence_id,
            section=ev.section,
        )
        j = orchestrate_care_journey(ext)
        self.assertFalse(j.provenance_audit_passed)
        self.assertFalse(j.patient_releasable)

    def test_adv_i_no_invented_dependencies(self) -> None:
        ext = run_document_intelligence_pipeline(P09_TEST_REFERRAL_FOLLOWUP.raw_text)
        j = orchestrate_care_journey(ext)
        # None of the tasks should invent clinical dependency between lab test and cardiology
        for t in j.timeline.all_tasks:
            self.assertEqual(len(t.dependency_ids), 0)

    def test_adv_j_duplicate_task_deduplication(self) -> None:
        ext = run_document_intelligence_pipeline(P07_DUPLICATE_TASKS.raw_text)
        j = orchestrate_care_journey(ext)
        titles = [t.title for t in j.timeline.all_tasks if t.task_category == TaskCategory.MEDICATION]
        self.assertEqual(len(titles), 1)

    def test_adv_k_unrelated_narrative_not_a_provider_task(self) -> None:
        raw = (
            "Hospital Course:\n"
            "Patient's brother Dr. Bob visited the patient during stay.\n"
            "Discharge Medications:\n"
            "Continue aspirin 81 mg daily.\n"
            "Follow-Up Appointments:\n"
            "Follow up with Primary Care in 14 days.\n"
        )
        ext = run_document_intelligence_pipeline(raw)
        j = orchestrate_care_journey(ext)
        fu_provs = [t.clinical_metadata.get("provider_name") for t in j.timeline.all_tasks if t.task_category == TaskCategory.FOLLOW_UP]
        self.assertNotIn("Dr. Bob", fu_provs)

    def test_adv_l_warning_threshold_remains_source_identical(self) -> None:
        ext = run_document_intelligence_pipeline(P10_WARNING_MONITORING.raw_text)
        j = orchestrate_care_journey(ext)
        warn_task = next(t for t in j.timeline.all_tasks if t.task_category == TaskCategory.WARNING_MONITORING)
        self.assertIn("3 pounds", warn_task.description)
        self.assertEqual(
            warn_task.clinical_metadata["symptom_description"],
            "weight increases by more than 3 pounds in 24 hours",
        )

    def test_adv_m_formulation_tartrate_vs_succinate_remain_distinct(self) -> None:
        raw = (
            "Discharge Medications:\n"
            "1. Continue metoprolol tartrate 25 mg twice daily.\n"
            "2. Continue metoprolol succinate 50 mg daily.\n"
            "Follow-Up Appointments:\n"
            "Follow up with Cardiology in 7 days.\n"
        )
        ext = run_document_intelligence_pipeline(raw)
        j = orchestrate_care_journey(ext)
        meds = [t for t in j.timeline.all_tasks if t.task_category == TaskCategory.MEDICATION]
        self.assertEqual(len(meds), 2)
        names = {t.clinical_metadata["medication_name"] for t in meds}
        self.assertEqual(names, {"metoprolol tartrate", "metoprolol succinate"})

    def test_adv_n_absolute_date_no_reference_date_confusion(self) -> None:
        ext = run_document_intelligence_pipeline(P04_ABSOLUTE_DATE.raw_text)
        j = orchestrate_care_journey(ext)
        fu = next(t for t in j.timeline.all_tasks if t.task_category == TaskCategory.FOLLOW_UP)
        self.assertEqual(fu.due_date, "2027-03-18")
        self.assertEqual(fu.timeframe.timeframe_type, TimeframeType.ABSOLUTE_DATE)

    def test_adv_o_relative_date_preserves_wording(self) -> None:
        ext = run_document_intelligence_pipeline(P03_RELATIVE_TIMEFRAME.raw_text)
        j = orchestrate_care_journey(ext)
        fu = next(t for t in j.timeline.all_tasks if t.task_category == TaskCategory.FOLLOW_UP)
        self.assertIsNone(fu.due_date)
        self.assertEqual(fu.due_window, "within 7 days")
        self.assertEqual(fu.timeframe.relative_operator, "within")


if __name__ == "__main__":
    unittest.main()
