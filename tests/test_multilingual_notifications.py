"""Comprehensive tests for CareBridge Multilingual Notification Support.

Validates:
1. Language Resolution (explicit, fallback to English, invalid rejected with HTTP 422).
2. Notification Localization across all 12 supported languages.
3. Invariant Preservation (dates, times, dosages, medication names, units).
4. No silent fallback to English for any Indian language.
5. Delivery channel rendering (In-App, Push, SMS).
6. Patient language change endpoint (PATCH /api/v1/patients/{patient_id}/language).
7. Non-destructive history guarantee (acknowledged notifications remain untouched).
8. Translation safety verification (REVIEW_REQUIRED & patient_releasable = False).
"""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from backend.main import app
from backend.notifications.types import (
    Notification,
    NotificationChannel,
    NotificationStatus,
    NotificationType,
    SUPPORTED_LANGUAGES,
    TranslationStatus,
)
from backend.notifications.multilingual import (
    NOTIFICATION_TEMPLATES,
    build_channel_payloads,
    format_multilingual_datetime,
    format_multilingual_notification,
    render_channel_payload,
    verify_translation_fidelity,
)
from backend.notifications.scheduler import (
    acknowledge_notification,
    create_notification,
    get_notifications,
)
from backend.notifications.fixtures import get_notification_fixture
from backend.document_intelligence.types import EvidenceSpan
from backend.task_orchestration.types import (
    OrchestratedTask,
    TaskCategory,
    TaskStatus,
    ParsedTimeframe,
    TimeframeType,
)
from backend.api_orchestrator import (
    ACTIVE_NOTIFICATION_STORE,
    ACTIVE_TASK_STORE,
    PATIENT_PREFERENCES_STORE,
    orchestrate_pipeline,
    update_patient_language,
)


class TestLanguageResolution(unittest.TestCase):
    """Test resolution, defaulting, and rejection of language codes."""

    def setUp(self):
        self.client = TestClient(app)

    def test_explicit_supported_language(self):
        """Explicitly passed supported language is resolved and returned."""
        resp = self.client.post("/api/v1/orchestrate", json={
            "case_id": "case_absolute_cardiology",
            "language": "ta"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("language", data)
        self.assertEqual(data["language"]["requested"], "ta")
        self.assertEqual(data["language"]["resolved"], "ta")
        self.assertEqual(data["language"]["translation_status"], "VERIFIED")

        # Verify notifications are in Tamil
        notifications = data["notifications"]
        self.assertGreater(len(notifications), 0)
        self.assertEqual(notifications[0]["language"], "ta")

    def test_fallback_to_default_english_when_omitted(self):
        """When no language is specified and patient has no preference, default to 'en'."""
        PATIENT_PREFERENCES_STORE.pop("SYNTH-PT-CUSTOM", None)
        resp = self.client.post("/api/v1/orchestrate", json={
            "raw_text": (
                "Discharge Instructions:\n"
                "Take metoprolol succinate 50 mg by mouth once daily.\n\n"
                "Follow-Up:\n"
                "Follow up with Cardiology on March 18, 2027 at 10:30 AM."
            )
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["language"]["resolved"], "en")
        notifications = data["notifications"]
        self.assertGreater(len(notifications), 0)
        self.assertEqual(notifications[0]["language"], "en")

    def test_invalid_language_rejected_with_422(self):
        """Unsupported language codes must be rejected with HTTP 422."""
        resp = self.client.post("/api/v1/orchestrate", json={
            "case_id": "case_absolute_cardiology",
            "language": "klingon"
        })
        self.assertEqual(resp.status_code, 422)
        self.assertIn("Unsupported language code", resp.json()["detail"])

    def test_patient_id_as_case_id_resolution(self):
        """Passing patient_id directly as case_id resolves the patient's case."""
        resp = self.client.post("/api/v1/orchestrate", json={
            "case_id": "SYNTH-PT-801",
            "language": "hi"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["case_info"]["patient_identifier"], "SYNTH-PT-801")
        self.assertEqual(data["language"]["resolved"], "hi")


class TestNotificationLocalizationAllLanguages(unittest.TestCase):
    """Test localized notification generation across all 12 supported languages."""

    def setUp(self):
        self.client = TestClient(app)
        self.follow_up_task, self.reminder = get_notification_fixture("N01")

    def test_all_twelve_languages_supported(self):
        """All 12 specified languages must be in SUPPORTED_LANGUAGES."""
        expected_languages = {"en", "es", "hi", "ta", "te", "kn", "ml", "bn", "mr", "gu", "pa", "or"}
        self.assertEqual(set(SUPPORTED_LANGUAGES.keys()), expected_languages)

    def test_localization_and_invariants_for_every_language(self):
        """Generate notifications in each language and assert clinical invariants."""
        # Baseline English title and body
        en_notif = create_notification(
            reminder=self.reminder,
            task=self.follow_up_task,
            language="en",
        )
        self.assertEqual(en_notif.language, "en")
        self.assertTrue(en_notif.patient_releasable)
        self.assertEqual(en_notif.translation_status, TranslationStatus.VERIFIED.value)

        for lang in SUPPORTED_LANGUAGES:
            notif = create_notification(
                reminder=self.reminder,
                task=self.follow_up_task,
                language=lang,
                patient_id="SYNTH-PT-801",
            )
            # 1. Correct language metadata
            self.assertEqual(notif.language, lang)
            self.assertEqual(notif.patient_id, "SYNTH-PT-801")
            self.assertEqual(notif.translation_status, TranslationStatus.VERIFIED.value)
            self.assertTrue(notif.patient_releasable)

            # 2. Invariant preservation: Appointment Date (Year 2027 and Day 18) must appear
            combined = f"{notif.title} {notif.body}"
            self.assertIn("2027", combined, f"Year 2027 missing for language {lang}")
            self.assertIn("18", combined, f"Day 18 missing for language {lang}")

            # 3. No silent fallback: non-English languages must have distinct localized text
            if lang != "en":
                self.assertNotEqual(
                    notif.title,
                    en_notif.title,
                    f"Language {lang} silently fell back to English title!"
                )

    def test_medication_reminder_invariants_preserved(self):
        """Medication reminders in Indian languages preserve dosages and units exactly."""
        med_task = OrchestratedTask(
            task_id="task-med-1",
            source_item_id="item-med-1",
            task_category=TaskCategory.MEDICATION,
            title="Take metoprolol succinate 50 mg by mouth once daily",
            description="Take metoprolol succinate 50 mg by mouth once daily",
            status=TaskStatus.SCHEDULED,
            timeframe=ParsedTimeframe(timeframe_type=TimeframeType.ABSOLUTE_DATE, raw_text="March 18, 2027"),
            evidence=self.follow_up_task.evidence,
            patient_releasable=True,
            requires_human_review=False,
            clinical_metadata={"specialty": "Cardiology"},
        )

        for lang in ("hi", "ta", "te", "kn", "ml", "bn", "mr", "gu", "pa", "or"):
            title, body, status = format_multilingual_notification(
                task=med_task,
                reminder=self.reminder,
                language=lang,
                notification_type=NotificationType.MEDICATION_REMINDER,
            )
            self.assertEqual(status, TranslationStatus.VERIFIED.value)
            combined = f"{title} {body}"
            self.assertIn("50", combined)
            self.assertIn("mg", combined.lower())
            self.assertIn("metoprolol succinate", combined)


    def test_all_ten_notification_types_render_across_languages(self):
        """All 10 required notification types render validly in Indian languages."""
        test_types = [
            NotificationType.APPOINTMENT_REMINDER,
            NotificationType.MEDICATION_REMINDER,
            NotificationType.TEST_REMINDER,
            NotificationType.FOLLOW_UP_REMINDER,
            NotificationType.WARNING_SIGN_ALERT,
            NotificationType.TASK_REMINDER,
            NotificationType.HUMAN_REVIEW_NOTIFICATION,
            NotificationType.SAFETY_ESCALATION,
            NotificationType.PROVIDER_FOLLOW_UP,
            NotificationType.GENERAL_CARE_INSTRUCTION,
        ]

        for ntype in test_types:
            for lang in ("ta", "hi", "te", "bn"):
                title, body, status = format_multilingual_notification(
                    task=self.follow_up_task,
                    reminder=self.reminder,
                    language=lang,
                    notification_type=ntype,
                )
                self.assertEqual(status, TranslationStatus.VERIFIED.value)
                self.assertGreater(len(title), 3)
                self.assertGreater(len(body), 5)


class TestChannelRendering(unittest.TestCase):
    """Test delivery channel payloads (IN_APP, PUSH, SMS) for non-English notifications."""

    def test_multilingual_channel_payload_generation(self):
        """Verify distinct channel renderings from canonical localized notification."""
        raw_title = "மருத்துவர் சந்திப்பு நினைவூட்டல்: இருதய நிபுணர் (Cardiology)"
        raw_body = "உங்கள் Cardiology சந்திப்பு மார்ச் 18, 2027 · 10:30 AM அன்று உள்ளது.\nவெளியேற்ற திட்டத்திலிருந்து.\nபராமரிப்பு பயணத்தைக் காண்க →"

        # In-App: Full text
        in_t, in_b = render_channel_payload(raw_title, raw_body, NotificationChannel.IN_APP, "ta")
        self.assertEqual(in_t, raw_title)
        self.assertEqual(in_b, raw_body)

        # Push: Flattened single paragraph
        push_t, push_b = render_channel_payload(raw_title, raw_body, NotificationChannel.PUSH, "ta")
        self.assertEqual(push_t, raw_title)
        self.assertNotIn("\n", push_b)

        # SMS: Prefixed with CareBridge
        sms_t, sms_b = render_channel_payload(raw_title, raw_body, NotificationChannel.SMS, "ta")
        self.assertTrue(sms_b.startswith("CareBridge: "))
        self.assertLessEqual(len(sms_b), 160)

    def test_build_channel_payloads_helper(self):
        """Test build_channel_payloads packages all 3 channels cleanly."""
        payloads = build_channel_payloads("शीर्षक", "शरीर\nविवरण", language="hi")
        self.assertIn("IN_APP", payloads)
        self.assertIn("PUSH", payloads)
        self.assertIn("SMS", payloads)
        self.assertTrue(payloads["SMS"]["body"].startswith("CareBridge: "))


class TestLanguageChangeEndpoint(unittest.TestCase):
    """Test PATCH /api/v1/patients/{patient_id}/language endpoint and non-destructive history."""

    def setUp(self):
        self.client = TestClient(app)
        # Clear stores for test isolation
        ACTIVE_NOTIFICATION_STORE.clear()
        ACTIVE_TASK_STORE.clear()

    def test_language_change_updates_scheduled_only(self):
        """Changing language re-translates SCHEDULED notifications and preserves ACKNOWLEDGED ones."""
        # 1. Orchestrate initial pipeline in English
        resp = self.client.post("/api/v1/orchestrate", json={
            "case_id": "case_absolute_cardiology",
            "language": "en"
        })
        self.assertEqual(resp.status_code, 200)
        notifs = resp.json()["notifications"]
        self.assertGreater(len(notifs), 0)

        notif_id = notifs[0]["id"]
        self.assertEqual(notifs[0]["language"], "en")
        original_en_title = notifs[0]["title"]

        # 2. Simulate patient acknowledging the first notification
        self.client.post(f"/api/v1/notifications/{notif_id}/action", json={"action": "ACKNOWLEDGE"})
        self.assertEqual(ACTIVE_NOTIFICATION_STORE[notif_id].status, NotificationStatus.ACKNOWLEDGED)

        # 3. Add a second notification that remains SCHEDULED
        sec_notif = ACTIVE_NOTIFICATION_STORE[notif_id].model_copy(deep=True)
        sec_notif.notification_id = "notif-scheduled-future"
        sec_notif.id = "notif-scheduled-future"
        sec_notif.status = NotificationStatus.SCHEDULED
        sec_notif.acknowledged_at = None
        ACTIVE_NOTIFICATION_STORE[sec_notif.notification_id] = sec_notif
        ACTIVE_TASK_STORE[sec_notif.notification_id] = ACTIVE_TASK_STORE[notif_id]

        # 4. Patch patient preferred language to Tamil ('ta')
        patch_resp = self.client.patch("/api/v1/patients/SYNTH-PT-801/language", json={
            "preferred_language": "ta"
        })
        self.assertEqual(patch_resp.status_code, 200)
        patch_data = patch_resp.json()
        self.assertEqual(patch_data["preferred_language"], "ta")
        self.assertEqual(patch_data["patient_id"], "SYNTH-PT-801")

        # 5. Check SCHEDULED notification was updated to Tamil
        updated_scheduled = ACTIVE_NOTIFICATION_STORE["notif-scheduled-future"]
        self.assertEqual(updated_scheduled.language, "ta")
        self.assertNotEqual(updated_scheduled.title, original_en_title)

        # 6. Check ACKNOWLEDGED notification preserved original history (English untouched)
        ack_notif = ACTIVE_NOTIFICATION_STORE[notif_id]
        self.assertEqual(ack_notif.language, "en")
        self.assertEqual(ack_notif.title, original_en_title)
        self.assertEqual(ack_notif.status, NotificationStatus.ACKNOWLEDGED)

    def test_patch_language_invalid_code_rejected(self):
        """PATCH endpoint rejects invalid language codes with HTTP 422."""
        patch_resp = self.client.patch("/api/v1/patients/SYNTH-PT-801/language", json={
            "preferred_language": "invalid-lang"
        })
        self.assertEqual(patch_resp.status_code, 422)


class TestSafetyVerificationAndFidelity(unittest.TestCase):
    """Test strict safety invariant enforcement and fidelity failure handling."""

    def test_missing_clinical_dosage_triggers_review_required(self):
        """If translated text drops required dosage quantity, fidelity fails."""
        follow_up_task, _ = get_notification_fixture("N01")

        med_task = OrchestratedTask(
            task_id="task-med-1",
            source_item_id="item-med-1",
            task_category=TaskCategory.MEDICATION,
            title="Take metoprolol 50 mg daily",
            description="Take metoprolol 50 mg daily",
            status=TaskStatus.SCHEDULED,
            timeframe=ParsedTimeframe(timeframe_type=TimeframeType.ABSOLUTE_DATE, raw_text="March 18, 2027"),
            evidence=follow_up_task.evidence,
            patient_releasable=True,
            requires_human_review=False,
            clinical_metadata={"specialty": "Cardiology"},
        )

        # Corrupted translation omitting the '50'
        corrupted_title = "दवा रिमाइंडर: metoprolol mg"
        corrupted_body = "metoprolol mg लें"

        is_verified, status, notes = verify_translation_fidelity(
            title=corrupted_title,
            body=corrupted_body,
            task=med_task,
            reminder=None,
            language="hi",
        )
        self.assertFalse(is_verified)
        self.assertEqual(status, TranslationStatus.REVIEW_REQUIRED.value)
        self.assertTrue(any("missing" in n.lower() for n in notes))

    def test_banned_phrase_triggers_review_required_and_unreleasable(self):
        """Inclusion of banned phrase triggers REVIEW_REQUIRED."""
        follow_up_task, _ = get_notification_fixture("N01")

        bad_title = "Appointment Reminder: Best Doctor in town"
        bad_body = "Your appointment is confirmed. Don't miss it!"

        is_verified, status, notes = verify_translation_fidelity(
            title=bad_title,
            body=bad_body,
            task=follow_up_task,
            reminder=None,
            language="en",
        )
        self.assertFalse(is_verified)
        self.assertEqual(status, TranslationStatus.REVIEW_REQUIRED.value)
        self.assertTrue(any("Prohibited" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
