"""tests/test_phase8_ui.py

Comprehensive tests for Phase 8 — CareBridge Cinematic UI and Unified API.
Verifies:
1. Serving of static HTML command center at / and /app.
2. Real-time orchestration API /api/v1/orchestrate.
3. Case listing endpoint /api/v1/cases.
4. Provider search API /api/v1/providers/search.
5. Notification action transitions /api/v1/notifications/{notif_id}/action.
6. Strict elimination of fake AI metrics (no fake "94.8%").
7. Presence of mandatory clinical disclaimers.
8. Preservation of frozen safety SHA256 hashes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from backend.main import app

EXPECTED_VALIDATOR_SHA256 = (
    "30c7f7d4e7ba6467d4a5830a4c771714599074130e8b5e9e0f5fda067198073f"
)
EXPECTED_MODELS_SHA256 = (
    "fd38a149a8b1bef8755567328caa981b7b57b7d702b8ea1f867936da2023e4d7"
)


class TestPhase8UIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_frozen_hashes_untouched(self):
        """Verify safety validator and schemas models remain strictly unchanged."""
        base_dir = Path(__file__).parent.parent
        val_path = base_dir / "backend" / "safety" / "validator.py"
        models_path = base_dir / "backend" / "schemas" / "models.py"

        val_hash = hashlib.sha256(val_path.read_bytes()).hexdigest()
        models_hash = hashlib.sha256(models_path.read_bytes()).hexdigest()

        self.assertEqual(val_hash.lower(), EXPECTED_VALIDATOR_SHA256.lower())
        self.assertEqual(models_hash.lower(), EXPECTED_MODELS_SHA256.lower())

    def test_serve_html_app_endpoints(self):
        """Verify / and /app return the CareBridge Cinematic Command Center HTML."""
        for path in ["/", "/app"]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/html", response.headers.get("content-type", ""))
            body = response.text

            # Verify brand, fonts, and template elements
            self.assertIn("CAREBRIDGE", body)
            self.assertIn("Syne", body)
            self.assertIn("IBM Plex Mono", body)
            self.assertIn("DM Sans", body)
            self.assertIn("CLINICAL WORKSPACE", body)
            self.assertIn("workspace-grid", body)

            # CRITICAL SAFETY: Verify NO fake confidence score exists in HTML
            self.assertNotIn("94.8%", body)

            # Verify mandatory clinical disclaimer
            self.assertIn("MANDATORY CLINICAL SAFETY BOUNDARY STATEMENT", body)

    def test_get_available_cases_endpoint(self):
        """Verify /api/v1/cases lists all test fixtures with safety hints."""
        response = self.client.get("/api/v1/cases")
        self.assertEqual(response.status_code, 200)
        cases = response.json()
        self.assertIsInstance(cases, list)
        self.assertGreaterEqual(len(cases), 5)

        case_ids = [c["case_id"] for c in cases]
        self.assertIn("case_absolute_cardiology", case_ids)
        self.assertIn("normal_uncomplicated", case_ids)
        self.assertIn("conflicting_medications", case_ids)
        self.assertIn("missing_ambiguous_followup", case_ids)

        for c in cases:
            self.assertIn("title", c)
            self.assertIn("raw_text", c)
            self.assertIn("expected_safety", c)

    def test_orchestrate_absolute_cardiology_case(self):
        """Verify full pipeline execution for the absolute cardiology follow-up benchmark."""
        response = self.client.post(
            "/api/v1/orchestrate",
            json={"case_id": "case_absolute_cardiology", "language": "en"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Overall safety must be SAFE / ORGANIZED
        self.assertEqual(data["overall_safety_status"], "SAFE / ORGANIZED")
        self.assertTrue(data["patient_releasable"])
        self.assertFalse(data["requires_human_review"])

        # Tasks must include follow-up with Cardiology on March 18, 2027
        fu_tasks = [t for t in data["tasks"] if t["category"] == "FOLLOW_UP"]
        self.assertGreaterEqual(len(fu_tasks), 1)
        cardio_task = fu_tasks[0]
        self.assertIn("Cardiology", cardio_task["title"])
        self.assertIn("March 18, 2027", cardio_task["timeframe_display"])

        # Reminders must include 1 scheduled reminder
        self.assertGreaterEqual(len(data["reminders"]), 1)
        rem = data["reminders"][0]
        self.assertEqual(rem["task_id"], cardio_task["task_id"])

        # Notifications must include 1 follow-up alert
        self.assertGreaterEqual(len(data["notifications"]), 1)
        notif = data["notifications"][0]
        self.assertIn("Cardiology", notif["title"])

        # Provenance sentences must be mapped
        self.assertGreaterEqual(len(data["sentences"]), 3)

        # Medication formulations must be preserved
        med_names = [m["name"].lower() for m in data["medications"]]
        self.assertTrue(any("succinate" in m for m in med_names))

    def test_orchestrate_ambiguous_case_escalates(self):
        """Verify ambiguous case correctly triggers human review escalations in UI state."""
        response = self.client.post(
            "/api/v1/orchestrate",
            json={"case_id": "conflicting_medications", "language": "en"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn(data["overall_safety_status"], ["NEEDS REVIEW", "CONFLICT DETECTED"])
        self.assertGreater(len(data["human_review_items"]), 0)

    def test_orchestrate_spanish_language(self):
        """Verify language parameter correctly invokes Phase 6 Spanish generation."""
        response = self.client.post(
            "/api/v1/orchestrate",
            json={"case_id": "case_absolute_cardiology", "language": "es"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        plan = data["care_plan"]
        translations = plan.get("translations", [])
        self.assertGreater(len(translations), 0)
        self.assertEqual(translations[0].get("target_language"), "es")

    def test_orchestrate_custom_raw_text(self):
        """Verify pipeline handles custom raw text input safely."""
        custom_text = (
            "Discharge instructions: Take lisinopril 10 mg daily. "
            "Follow up with Dr. Smith in 2 weeks."
        )
        response = self.client.post(
            "/api/v1/orchestrate",
            json={"raw_text": custom_text, "language": "en"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["case_info"]["patient_identifier"], "SYNTH-PT-CUSTOM")
        self.assertIn("lisinopril", [m["name"].lower() for m in data["medications"]])

    def test_provider_search_api_no_ranking(self):
        """Verify provider search API adheres to Phase 7 safety guarantees."""
        response = self.client.post(
            "/api/v1/providers/search",
            json={"specialty": "Cardiology", "language": "English"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("providers", data)
        self.assertIn("patient_message", data)
        self.assertIn("does not rank or select providers", data["patient_message"])

        for p in data["providers"]:
            # No rating, no score, no quality ranking
            self.assertNotIn("rating", p)
            self.assertNotIn("stars", p)
            self.assertNotIn("score", p)

    def test_notification_action_transitions(self):
        """Verify /api/v1/notifications/{notif_id}/action transitions state cleanly."""
        # First orchestrate to populate notifications
        orch_res = self.client.post(
            "/api/v1/orchestrate",
            json={"case_id": "case_absolute_cardiology", "language": "en"},
        )
        data = orch_res.json()
        if not data["notifications"]:
            self.skipTest("No notifications generated for test case.")

        notif_id = data["notifications"][0]["notification_id"]

        # Acknowledge
        ack_res = self.client.post(
            f"/api/v1/notifications/{notif_id}/action",
            json={"action": "ACKNOWLEDGE"},
        )
        self.assertEqual(ack_res.status_code, 200)

        # Snooze
        snooze_res = self.client.post(
            f"/api/v1/notifications/{notif_id}/action",
            json={"action": "SNOOZE", "snooze_minutes": 30},
        )
        self.assertEqual(snooze_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
