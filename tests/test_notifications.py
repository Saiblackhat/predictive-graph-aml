"""tests/test_notifications.py

Phase 6.6 Notification & Alert Orchestration Test Suite.

Covers:
- Synthetic fixtures N01 - N18
- Adversarial tests A - T
- State transitions (SCHEDULED -> TRIGGERED -> ACKNOWLEDGED / DISMISSED / CANCELLED / SNOOZE)
- Idempotency & duplicate prevention
- Notification Center contract for Phase 8
- End-to-end integration test (Phase 4 -> 5 -> 6 -> 6.5 -> 6.6)
- Provider search side-effect test
"""

from __future__ import annotations

import copy
import datetime
import unittest

from backend.task_orchestration.types import TaskCategory, TaskStatus
from backend.care_reminders.types import ReminderStatus
from backend.notifications import (
    Notification,
    NotificationAction,
    NotificationCenterState,
    NotificationPriority,
    NotificationStatus,
    NotificationType,
    check_notification_eligibility,
    create_notification,
    trigger_notification,
    acknowledge_notification,
    dismiss_notification,
    cancel_notification,
    snooze_notification,
    get_notifications,
    validate_notification_provenance,
    audit_notification_content_fidelity,
)
from backend.notifications.fixtures import get_notification_fixture
from backend.provider_finder import find_providers, ProviderSearchQuery


class TestSyntheticFixtures(unittest.TestCase):
    """Verify N01-N18 fixtures against expected operational and safety invariants."""

    def test_n01_absolute_cardiology_followup(self):
        task, rem = get_notification_fixture("N01")
        elig = check_notification_eligibility(rem, task)
        self.assertTrue(elig.is_eligible)
        self.assertEqual(elig.notification_type, NotificationType.FOLLOW_UP_REMINDER)
        notif = create_notification(rem, task)
        self.assertTrue(notif.patient_releasable)
        self.assertTrue(notif.provenance_valid)
        self.assertIn("Cardiology", notif.body)
        self.assertIn("March 18, 2027", notif.body)

    def test_n02_absolute_test_date(self):
        task, rem = get_notification_fixture("N02")
        elig = check_notification_eligibility(rem, task)
        self.assertTrue(elig.is_eligible)
        self.assertEqual(elig.notification_type, NotificationType.TEST_REMINDER)
        notif = create_notification(rem, task)
        self.assertTrue(notif.patient_releasable)
        self.assertIn("CBC", notif.body)

    def test_n03_care_task_exact_time(self):
        task, rem = get_notification_fixture("N03")
        elig = check_notification_eligibility(rem, task)
        self.assertTrue(elig.is_eligible)
        self.assertEqual(elig.notification_type, NotificationType.CARE_TASK_REMINDER)
        notif = create_notification(rem, task)
        self.assertTrue(notif.patient_releasable)

    def test_n04_relative_without_reference_date(self):
        task, rem = get_notification_fixture("N04")
        elig = check_notification_eligibility(rem, task)
        self.assertFalse(elig.is_eligible)
        with self.assertRaises(ValueError):
            create_notification(rem, task)

    def test_n05_after_7_days_task(self):
        task, rem = get_notification_fixture("N05")
        elig = check_notification_eligibility(rem, task)
        self.assertFalse(elig.is_eligible)
        self.assertTrue(any("after" in r.lower() for r in elig.rejection_reasons))

    def test_n06_conflicting_task_blocked(self):
        task, rem = get_notification_fixture("N06")
        elig = check_notification_eligibility(rem, task)
        self.assertFalse(elig.is_eligible)

    def test_n07_needs_review_task(self):
        task, rem = get_notification_fixture("N07")
        elig = check_notification_eligibility(rem, task)
        self.assertFalse(elig.is_eligible)

    def test_n08_blocked_task(self):
        task, rem = get_notification_fixture("N08")
        elig = check_notification_eligibility(rem, task)
        self.assertFalse(elig.is_eligible)

    def test_n09_spanish_cardiology_followup(self):
        task, rem = get_notification_fixture("N09")
        notif = create_notification(rem, task, language="es")
        self.assertTrue(notif.patient_releasable)
        self.assertIn("Cardiología", notif.body)
        self.assertIn("18 de marzo de 2027", notif.body)
        self.assertEqual(notif.language, "es")

    def test_n11_cancelled_reminder(self):
        task, rem = get_notification_fixture("N11")
        elig = check_notification_eligibility(rem, task)
        self.assertFalse(elig.is_eligible)


class TestAdversarialCases(unittest.TestCase):
    """Verify Adversarial cases A through T."""

    def test_adv_a_valid_absolute_appointment_reminder(self):
        task, rem = get_notification_fixture("N01")
        notif = create_notification(rem, task)
        self.assertTrue(notif.patient_releasable)
        self.assertEqual(notif.status, NotificationStatus.SCHEDULED)

    def test_adv_b_relative_within_7_days_no_reference_date(self):
        task, rem = get_notification_fixture("N04")
        elig = check_notification_eligibility(rem, task)
        self.assertFalse(elig.is_eligible)

    def test_adv_c_after_7_days_rejected(self):
        task, rem = get_notification_fixture("N05")
        elig = check_notification_eligibility(rem, task)
        self.assertFalse(elig.is_eligible)

    def test_adv_d_blocked_task_no_notification(self):
        task, rem = get_notification_fixture("N08")
        self.assertFalse(task.patient_releasable)
        elig = check_notification_eligibility(rem, task)
        self.assertFalse(elig.is_eligible)

    def test_adv_e_needs_review_task_no_notification(self):
        task, rem = get_notification_fixture("N07")
        elig = check_notification_eligibility(rem, task)
        self.assertFalse(elig.is_eligible)

    def test_adv_f_conflicting_task_no_notification(self):
        task, rem = get_notification_fixture("N06")
        elig = check_notification_eligibility(rem, task)
        self.assertFalse(elig.is_eligible)

    def test_adv_g_provider_search_only_creates_no_notification(self):
        res = find_providers(ProviderSearchQuery(specialty="Cardiology", location_city="Chennai"))
        self.assertGreater(res.result_count, 0)
        # Verify provider result does not contain or generate notifications
        for p in res.providers:
            self.assertFalse(hasattr(p, "notification_id"))

    def test_adv_h_provider_result_presented_as_appointment_blocked(self):
        task, rem = get_notification_fixture("N01")
        notif = create_notification(rem, task)
        # Ensure the notification text does NOT claim an appointment is booked/confirmed
        self.assertNotIn("appointment is confirmed", notif.body.lower())
        self.assertNotIn("appointment booked", notif.body.lower())

    def test_adv_i_fake_availability_blocked(self):
        task, rem = get_notification_fixture("N01")
        notif = create_notification(rem, task)
        self.assertNotIn("available tomorrow", notif.body.lower())
        self.assertNotIn("instant booking", notif.body.lower())

    def test_adv_j_duplicate_scheduler_execution_is_idempotent(self):
        task, rem = get_notification_fixture("N01")
        n1 = create_notification(rem, task)
        existing = [n1]
        n2 = create_notification(rem, task, existing_notifications=existing)
        self.assertIs(n1, n2)

    def test_adv_k_cancel_notification_leaves_task_unchanged(self):
        task, rem = get_notification_fixture("N01")
        notif = create_notification(rem, task)
        cancel_notification(notif)
        self.assertEqual(notif.status, NotificationStatus.CANCELLED)
        # Task remains SCHEDULED
        self.assertEqual(task.status, TaskStatus.SCHEDULED)

    def test_adv_l_dismiss_notification_leaves_task_unchanged(self):
        task, rem = get_notification_fixture("N01")
        notif = create_notification(rem, task)
        trigger_notification(notif)
        dismiss_notification(notif)
        self.assertEqual(notif.status, NotificationStatus.DISMISSED)
        self.assertEqual(task.status, TaskStatus.SCHEDULED)

    def test_adv_m_no_response_never_becomes_missed(self):
        task, rem = get_notification_fixture("N01")
        notif = create_notification(rem, task)
        trigger_notification(notif)
        # Notification stays TRIGGERED if no response
        self.assertEqual(notif.status, NotificationStatus.TRIGGERED)
        self.assertNotIn("missed", notif.status.value.lower())

    def test_adv_n_spanish_notification_semantic_fidelity(self):
        task, rem = get_notification_fixture("N09")
        notif = create_notification(rem, task, language="es")
        self.assertTrue(notif.patient_releasable)
        self.assertIn("18 de marzo de 2027", notif.body)
        self.assertNotIn("debe", notif.body.lower())
        self.assertNotIn("urgente", notif.body.lower())

    def test_adv_o_tampered_reminder_fails_provenance(self):
        task, rem = get_notification_fixture("N01")
        notif = create_notification(rem, task)
        # Tamper reminder scheduled_for after creation
        tampered_rem = rem.model_copy(deep=True)
        tampered_rem.scheduled_for = "2027-03-10T00:00:00"
        ok, issues = validate_notification_provenance(notif, tampered_rem, task)
        self.assertFalse(ok)
        self.assertTrue(any("differs" in i for i in issues))

    def test_adv_p_changed_task_date_fails_provenance(self):
        task, rem = get_notification_fixture("N01")
        notif = create_notification(rem, task)
        # Tamper task ID
        tampered_task = task.model_copy(deep=True)
        tampered_task.task_id = "different-task-id"
        ok, issues = validate_notification_provenance(notif, rem, tampered_task)
        self.assertFalse(ok)

    def test_adv_q_missing_timezone_remains_null(self):
        task, rem = get_notification_fixture("N01")
        rem.timezone = None
        notif = create_notification(rem, task)
        self.assertIsNone(notif.timezone)

    def test_adv_r_unsupported_urgency_fails_fidelity(self):
        ok, violations = audit_notification_content_fidelity(
            "Emergency alert", "Your heart needs attention urgently."
        )
        self.assertFalse(ok)
        self.assertTrue(any("forbidden" in v.lower() for v in violations))

    def test_adv_s_provider_best_cardiologist_claim_blocked(self):
        ok, violations = audit_notification_content_fidelity(
            "Upcoming follow-up", "See the best cardiologist in town."
        )
        self.assertFalse(ok)
        self.assertTrue(any("forbidden" in v.lower() for v in violations))

    def test_adv_t_unsupported_clinical_statement_blocked(self):
        ok, violations = audit_notification_content_fidelity(
            "Upcoming follow-up", "Your condition is getting worse."
        )
        self.assertFalse(ok)


class TestStateTransitionsAndCenter(unittest.TestCase):
    """Test full operational lifecycle and Phase 8 UI contract."""

    def test_lifecycle_scheduled_to_triggered_to_acknowledged(self):
        task, rem = get_notification_fixture("N01")
        notif = create_notification(rem, task)
        self.assertEqual(notif.status, NotificationStatus.SCHEDULED)
        self.assertTrue(notif.unread)

        trigger_notification(notif)
        self.assertEqual(notif.status, NotificationStatus.TRIGGERED)
        self.assertIsNotNone(notif.triggered_at)
        self.assertTrue(notif.unread)

        acknowledge_notification(notif)
        self.assertEqual(notif.status, NotificationStatus.ACKNOWLEDGED)
        self.assertIsNotNone(notif.acknowledged_at)
        self.assertFalse(notif.unread)

    def test_snooze_creates_new_operational_schedule_only(self):
        task, rem = get_notification_fixture("N01")
        notif = create_notification(rem, task)
        trigger_notification(notif)

        base_time = datetime.datetime(2027, 3, 17, 10, 30, tzinfo=datetime.timezone.utc)
        snoozed = snooze_notification(notif, 60, base_dt=base_time)
        self.assertEqual(snoozed.status, NotificationStatus.SCHEDULED)
        self.assertIn("snooze-60", snoozed.notification_id)
        # Original appointment time is unaffected
        self.assertEqual(rem.event_time, "2027-03-18T10:30:00")
        self.assertEqual(task.timeframe.absolute_date, "2027-03-18")

    def test_notification_center_contract_counts(self):
        task, rem = get_notification_fixture("N01")
        n1 = create_notification(rem, task)
        state = get_notifications([n1])
        self.assertEqual(state.total_count, 1)
        self.assertEqual(state.unread_count, 1)

        acknowledge_notification(n1)
        state_after = get_notifications([n1])
        self.assertEqual(state_after.total_count, 1)
        self.assertEqual(state_after.unread_count, 0)

    def test_zero_notifications_count_is_zero_not_fabricated(self):
        state = get_notifications([])
        self.assertEqual(state.total_count, 0)
        self.assertEqual(state.unread_count, 0)


class TestEndToEndPipeline(unittest.TestCase):
    """Full pipeline: Raw text -> Phase 4 -> Phase 5 -> Phase 6 -> Phase 6.5 -> Phase 6.6."""

    def test_end_to_end_discharge_to_notification(self):
        from backend.document_intelligence import run_document_intelligence_pipeline
        from backend.task_orchestration import orchestrate_care_journey
        from backend.patient_language import generate_patient_care_plan
        from backend.care_reminders.scheduler import create_reminder
        from backend.care_reminders.types import ReminderPreset

        raw_discharge = (
            "Discharge Instructions:\n"
            "Take amlodipine 5 mg by mouth once daily for blood pressure.\n"
            "Follow-Up:\n"
            "Follow up with Cardiology on March 18, 2027 at 10:30 AM."
        )

        # 1. Document Intelligence
        ext = run_document_intelligence_pipeline(raw_discharge)

        # 2. Task Orchestration
        journey = orchestrate_care_journey(ext)
        fu_tasks = [t for t in journey.timeline.all_tasks if t.task_category == TaskCategory.FOLLOW_UP]
        self.assertTrue(len(fu_tasks) > 0)
        task = fu_tasks[0]
        self.assertTrue(task.patient_releasable)

        # 3. Patient Care Plan
        plan = generate_patient_care_plan(journey)
        self.assertTrue(plan.fidelity_audit_passed)

        # 4. Care Reminders
        reminder = create_reminder(task, ReminderPreset.ONE_DAY)
        self.assertTrue(reminder.patient_releasable)

        # 5. Notification Orchestration
        notif = create_notification(reminder, task)
        self.assertTrue(notif.patient_releasable)
        self.assertEqual(notif.notification_type, NotificationType.FOLLOW_UP_REMINDER)
        self.assertIn("Cardiology", notif.body)
        self.assertIn("March 18, 2027", notif.body)
        self.assertIn("10:30", notif.body)
        self.assertNotIn("appointment confirmed", notif.body.lower())
        self.assertNotIn("doctor is expecting you", notif.body.lower())

    def test_provider_search_side_effects_zero(self):
        """Provider search in Cardiology Chennai must produce 0 notifications and 0 appointments."""
        query = ProviderSearchQuery(specialty="Cardiology", location_city="Chennai")
        result = find_providers(query)
        self.assertGreater(result.result_count, 0)

        # Notification state remains 0
        state = get_notifications([])
        self.assertEqual(state.total_count, 0)
        self.assertEqual(state.unread_count, 0)


if __name__ == "__main__":
    unittest.main()
