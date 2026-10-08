"""Comprehensive test suite for Phase 6.5 Care Reminders.

Covers:
1. All 18 synthetic fixtures (R01 - R18).
2. All 17 adversarial checks (A - Q).
3. End-to-end integration with Phase 4, Phase 5, and Phase 6.
4. Provenance preservation and tamper detection.
5. Lifecycle states, attendance enquiry, and cancellation independence.
"""

from __future__ import annotations

import datetime
import unittest

from backend.document_intelligence import run_document_intelligence_pipeline
from backend.task_orchestration import orchestrate_care_journey
from backend.task_orchestration.types import (
    OrchestratedTask,
    TaskCategory,
    TaskStatus,
    TimeframeType,
)
from backend.care_reminders import (
    AttendanceEnquiry,
    AttendanceStatus,
    EligibilityResult,
    Reminder,
    ReminderPreset,
    ReminderStatus,
    ReminderType,
    cancel_reminder,
    check_reminder_eligibility,
    create_reminder,
    dismiss_reminder,
    is_reminder_eligible,
    respond_to_attendance_enquiry,
    trigger_reminder,
    validate_reminder_provenance,
)
from backend.care_reminders.fixtures import (
    ALL_CARE_REMINDER_FIXTURES,
    get_fixture_task,
)


class TestPhase65CareReminders(unittest.TestCase):
    """Rigorous tests for Care Reminders eligibility, safety, and lifecycle."""

    # ------------------------------------------------------------------
    # Synthetic Fixtures R01 - R18
    # ------------------------------------------------------------------
    def test_r01_explicit_datetime(self) -> None:
        task = get_fixture_task("R01")
        elig = check_reminder_eligibility(task)
        self.assertTrue(elig.is_eligible)
        self.assertEqual(elig.event_date, "2027-03-18")
        self.assertEqual(elig.event_time, "10:30:00")
        self.assertTrue(elig.has_explicit_time)
        self.assertEqual(elig.reminder_type, ReminderType.APPOINTMENT)

    def test_r02_explicit_date_no_time(self) -> None:
        task = get_fixture_task("R02")
        elig = check_reminder_eligibility(task)
        self.assertTrue(elig.is_eligible)
        self.assertEqual(elig.event_date, "2027-03-18")
        self.assertEqual(elig.event_time, "09:00:00")
        self.assertFalse(elig.has_explicit_time)

    def test_r03_relative_no_reference(self) -> None:
        task = get_fixture_task("R03")
        elig = check_reminder_eligibility(task, reference_date=None)
        self.assertFalse(elig.is_eligible)
        self.assertTrue(any("trusted reference date" in r for r in elig.rejection_reasons))

    def test_r04_relative_with_reference(self) -> None:
        task = get_fixture_task("R04")
        elig = check_reminder_eligibility(task, reference_date="2027-03-01")
        self.assertTrue(elig.is_eligible)
        self.assertEqual(elig.event_date, "2027-03-08")
        self.assertTrue(elig.derived_from_reference_date)

    def test_r05_after_window(self) -> None:
        task = get_fixture_task("R05")
        elig = check_reminder_eligibility(task, reference_date="2027-03-01")
        self.assertFalse(elig.is_eligible)
        self.assertTrue(any("Relative operator 'after'" in r for r in elig.rejection_reasons))

    def test_r06_ambiguous_appointment(self) -> None:
        task = get_fixture_task("R06")
        elig = check_reminder_eligibility(task)
        self.assertFalse(elig.is_eligible)

    def test_r07_conflicting_appointments(self) -> None:
        task = get_fixture_task("R07")
        elig = check_reminder_eligibility(task)
        self.assertFalse(elig.is_eligible)

    def test_r08_unsupported_provider_claim(self) -> None:
        task = get_fixture_task("R08")
        elig = check_reminder_eligibility(task)
        self.assertFalse(elig.is_eligible)

    def test_r09_not_releasable(self) -> None:
        task = get_fixture_task("R09")
        elig = check_reminder_eligibility(task)
        self.assertFalse(elig.is_eligible)
        self.assertTrue(any("not patient-releasable" in r for r in elig.rejection_reasons))

    def test_r10_one_day_offset(self) -> None:
        task = get_fixture_task("R10")
        rem = create_reminder(task, offset_minutes=ReminderPreset.ONE_DAY.value)
        self.assertEqual(rem.offset_minutes, 1440)
        self.assertEqual(rem.event_time, "2027-03-18T10:30:00")
        self.assertEqual(rem.scheduled_for, "2027-03-17T10:30:00")
        self.assertTrue(rem.provenance_valid)

    def test_r11_two_hour_offset(self) -> None:
        task = get_fixture_task("R11")
        rem = create_reminder(task, offset_minutes=ReminderPreset.TWO_HOURS.value)
        self.assertEqual(rem.offset_minutes, 120)
        self.assertEqual(rem.scheduled_for, "2027-03-18T08:30:00")
        self.assertTrue(rem.provenance_valid)

    def test_r12_custom_valid_offset(self) -> None:
        task = get_fixture_task("R12")
        rem = create_reminder(task, offset_minutes=360)  # 6 hours
        self.assertEqual(rem.offset_minutes, 360)
        self.assertEqual(rem.scheduled_for, "2027-03-18T04:30:00")
        self.assertTrue(rem.provenance_valid)

    def test_r13_custom_invalid_offset(self) -> None:
        task = get_fixture_task("R13")
        with self.assertRaises(ValueError):
            create_reminder(task, offset_minutes=-60)
        with self.assertRaises(ValueError):
            create_reminder(task, offset_minutes=0)

    def test_r14_duplicate_deduplication(self) -> None:
        task = get_fixture_task("R14")
        rem1 = create_reminder(task, offset_minutes=1440)
        rem2 = create_reminder(task, offset_minutes=1440, existing_reminders=[rem1])
        self.assertEqual(rem1.reminder_id, rem2.reminder_id)
        # Distinct offset produces distinct reminder
        rem3 = create_reminder(task, offset_minutes=120, existing_reminders=[rem1])
        self.assertNotEqual(rem1.reminder_id, rem3.reminder_id)

    def test_r15_cancel_does_not_cancel_task(self) -> None:
        task = get_fixture_task("R15")
        rem = create_reminder(task, offset_minutes=1440)
        rem_cancelled, task_unmodified = cancel_reminder(rem, task)
        self.assertEqual(rem_cancelled.status, ReminderStatus.CANCELLED)
        self.assertEqual(task_unmodified.status, task.status)
        self.assertNotEqual(task_unmodified.status, TaskStatus.BLOCKED)

    def test_r16_no_response_not_missed(self) -> None:
        task = get_fixture_task("R16")
        rem = create_reminder(task, offset_minutes=1440)
        rem = trigger_reminder(rem)
        self.assertEqual(rem.status, ReminderStatus.TRIGGERED)
        self.assertIsNotNone(rem.attendance_enquiry)

        rem = respond_to_attendance_enquiry(rem, AttendanceStatus.NO_RESPONSE)
        self.assertEqual(rem.status, ReminderStatus.TRIGGERED)
        self.assertIsNone(rem.attendance_enquiry.followup_note)
        self.assertFalse(rem.attendance_enquiry.clinical_advice_provided)

    def test_r17_spanish_reminder(self) -> None:
        task = get_fixture_task("R17")
        rem = create_reminder(task, offset_minutes=1440, language="es")
        self.assertIn("Seguimiento con Cardiología", rem.title)
        self.assertIn("18 de marzo de 2027 · 10:30", rem.ui_display_text)
        self.assertTrue(rem.provenance_valid)

    def test_r18_medication_cannot_be_appointment(self) -> None:
        task = get_fixture_task("R18")
        elig = check_reminder_eligibility(task)
        self.assertFalse(elig.is_eligible)
        self.assertTrue(any("Medication instructions cannot" in r for r in elig.rejection_reasons))

    # ------------------------------------------------------------------
    # Adversarial Suite (A - Q)
    # ------------------------------------------------------------------
    def test_adv_a_explicit_datetime_eligible(self) -> None:
        task = get_fixture_task("R01")
        self.assertTrue(is_reminder_eligible(task))

    def test_adv_b_within_window_no_ref_not_eligible(self) -> None:
        task = get_fixture_task("R03")
        self.assertFalse(is_reminder_eligible(task, reference_date=None))

    def test_adv_c_after_window_distinct(self) -> None:
        task = get_fixture_task("R05")
        self.assertFalse(is_reminder_eligible(task, reference_date="2027-03-01"))

    def test_adv_d_vague_timeframe_not_eligible(self) -> None:
        task = get_fixture_task("R06")
        self.assertFalse(is_reminder_eligible(task))

    def test_adv_e_conflicting_dates_not_eligible(self) -> None:
        task = get_fixture_task("R07")
        self.assertFalse(is_reminder_eligible(task))

    def test_adv_f_ambiguous_provider_not_eligible(self) -> None:
        task = get_fixture_task("R09")
        self.assertFalse(is_reminder_eligible(task))

    def test_adv_g_unsupported_provider_claim_not_eligible(self) -> None:
        task = get_fixture_task("R08")
        self.assertFalse(is_reminder_eligible(task))

    def test_adv_h_task_needs_review_not_eligible(self) -> None:
        task = get_fixture_task("R01")
        task_review = task.model_copy(update={"status": TaskStatus.NEEDS_REVIEW, "requires_human_review": True})
        self.assertFalse(is_reminder_eligible(task_review))

    def test_adv_i_task_blocked_not_eligible(self) -> None:
        task = get_fixture_task("R01")
        task_blocked = task.model_copy(update={"status": TaskStatus.BLOCKED, "patient_releasable": False})
        self.assertFalse(is_reminder_eligible(task_blocked))

    def test_adv_j_offset_after_appointment_rejected(self) -> None:
        task = get_fixture_task("R01")
        with self.assertRaises(ValueError):
            create_reminder(task, offset_minutes=-1440)

    def test_adv_k_duplicate_identical_deduplicated(self) -> None:
        task = get_fixture_task("R01")
        r1 = create_reminder(task, offset_minutes=1440)
        r2 = create_reminder(task, offset_minutes=1440, existing_reminders=[r1])
        self.assertIs(r1, r2)

    def test_adv_l_cancel_reminder_task_remains_active(self) -> None:
        task = get_fixture_task("R01")
        r = create_reminder(task, offset_minutes=1440)
        r_canc, t_active = cancel_reminder(r, task)
        self.assertEqual(r_canc.status, ReminderStatus.CANCELLED)
        self.assertIn(t_active.status, (TaskStatus.SCHEDULED, TaskStatus.PENDING))

    def test_adv_m_no_response_never_missed(self) -> None:
        task = get_fixture_task("R01")
        r = create_reminder(task, offset_minutes=1440)
        r = trigger_reminder(r)
        r = respond_to_attendance_enquiry(r, AttendanceStatus.NO_RESPONSE)
        self.assertNotIn("missed", str(r.attendance_enquiry.model_dump()).lower())

    def test_adv_n_continue_aspirin_no_appointment_reminder(self) -> None:
        task = get_fixture_task("R18")
        self.assertFalse(is_reminder_eligible(task))

    def test_adv_o_do_not_stop_aspirin_no_appointment_reminder(self) -> None:
        task = get_fixture_task("R18").model_copy(update={"title": "Do not stop taking aspirin 81 mg"})
        self.assertFalse(is_reminder_eligible(task))

    def test_adv_p_cardiology_retains_exact_datetime(self) -> None:
        task = get_fixture_task("R01")
        r = create_reminder(task, offset_minutes=120)
        self.assertEqual(r.event_time, "2027-03-18T10:30:00")
        self.assertEqual(r.scheduled_for, "2027-03-18T08:30:00")

    def test_adv_q_spanish_exact_datetime_preserved(self) -> None:
        task = get_fixture_task("R01")
        r = create_reminder(task, offset_minutes=1440, language="es")
        self.assertEqual(r.event_time, "2027-03-18T10:30:00")
        self.assertIn("18 de marzo de 2027", r.ui_display_text)
        self.assertIn("10:30", r.ui_display_text)

    # ------------------------------------------------------------------
    # Provenance Validation & Tamper Resistance
    # ------------------------------------------------------------------
    def test_provenance_tamper_detection(self) -> None:
        task = get_fixture_task("R01")
        rem = create_reminder(task, offset_minutes=1440)
        self.assertTrue(validate_reminder_provenance(rem, task))

        # Tampered task_id
        tampered_rem = rem.model_copy(update={"task_id": "other-id"})
        self.assertFalse(validate_reminder_provenance(tampered_rem, task))

        # Tampered patient_releasable
        tampered_rem2 = rem.model_copy(update={"patient_releasable": not rem.patient_releasable})
        self.assertFalse(validate_reminder_provenance(tampered_rem2, task))

        # Tampered timing calculation
        tampered_rem3 = rem.model_copy(update={"scheduled_for": "2027-03-18T10:29:00"})
        self.assertFalse(validate_reminder_provenance(tampered_rem3, task))

    # ------------------------------------------------------------------
    # Attendance Enquiry Flows
    # ------------------------------------------------------------------
    def test_attendance_enquiry_options(self) -> None:
        task = get_fixture_task("R01")
        rem = create_reminder(task, offset_minutes=1440)
        rem = trigger_reminder(rem)

        # 1. Completed
        rem_comp = respond_to_attendance_enquiry(rem.model_copy(deep=True), AttendanceStatus.COMPLETED)
        self.assertEqual(rem_comp.status, ReminderStatus.ACKNOWLEDGED)
        self.assertIn("completed", rem_comp.attendance_enquiry.followup_note.lower())
        self.assertFalse(rem_comp.attendance_enquiry.clinical_advice_provided)

        # 2. Not yet
        rem_ny = respond_to_attendance_enquiry(rem.model_copy(deep=True), AttendanceStatus.NOT_YET)
        self.assertEqual(rem_ny.status, ReminderStatus.ACKNOWLEDGED)
        self.assertIn("reschedule", rem_ny.attendance_enquiry.followup_note.lower())
        self.assertFalse(rem_ny.attendance_enquiry.clinical_advice_provided)

        # 3. Need help
        rem_help = respond_to_attendance_enquiry(rem.model_copy(deep=True), AttendanceStatus.NEED_HELP)
        self.assertEqual(rem_help.status, ReminderStatus.ACKNOWLEDGED)
        self.assertIn("coordinator", rem_help.attendance_enquiry.followup_note.lower())
        self.assertFalse(rem_help.attendance_enquiry.clinical_advice_provided)

    # ------------------------------------------------------------------
    # End-to-End Pipeline Integration with Upstream Phases
    # ------------------------------------------------------------------
    def test_end_to_end_pipeline_integration(self) -> None:
        doc = (
            "DISCHARGE SUMMARY - E2E\n"
            "Discharge Medications:\n"
            "Continue metoprolol succinate 50 mg daily.\n"
            "Follow-Up Appointments:\n"
            "Follow up with Cardiology on March 18, 2027.\n"
        )
        ext = run_document_intelligence_pipeline(doc)
        journey = orchestrate_care_journey(ext)
        self.assertTrue(journey.patient_releasable)

        fu_task = next(t for t in journey.timeline.all_tasks if t.task_category == TaskCategory.FOLLOW_UP)
        self.assertTrue(is_reminder_eligible(fu_task))

        rem = create_reminder(fu_task, offset_minutes=ReminderPreset.ONE_DAY.value)
        self.assertEqual(rem.offset_minutes, 1440)
        self.assertEqual(rem.scheduled_for, "2027-03-17T09:00:00")
        self.assertTrue(rem.provenance_valid)
        self.assertEqual(rem.title, "Cardiology follow-up")


if __name__ == "__main__":
    unittest.main()
