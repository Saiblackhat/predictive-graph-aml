import unittest
import hashlib
import json
import re
from fastapi.testclient import TestClient
from backend.main import app

EXPECTED_VALIDATOR_HASH = "30C7F7D4E7BA6467D4A5830A4C771714599074130E8B5E9E0F5FDA067198073F"
EXPECTED_MODELS_HASH = "FD38A149A8B1BEF8755567328CAA981B7B57B7D702B8EA1F867936DA2023E4D7"
SUPPORTED_LANGUAGES = ["en", "es", "hi", "ta", "te", "kn", "ml", "bn", "mr", "gu", "pa", "or"]

class TestUiLanguageSwitching(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        with open("backend/static/index.html", "r", encoding="utf-8") as f:
            self.html = f.read()

    def test_01_frozen_files_hash_preservation(self):
        """Confirm frozen safety files are untouched."""
        val_hash = hashlib.sha256(open("backend/safety/validator.py", "rb").read()).hexdigest().upper()
        models_hash = hashlib.sha256(open("backend/schemas/models.py", "rb").read()).hexdigest().upper()
        self.assertEqual(val_hash, EXPECTED_VALIDATOR_HASH, "backend/safety/validator.py modified!")
        self.assertEqual(models_hash, EXPECTED_MODELS_HASH, "backend/schemas/models.py modified!")

    def test_02_all_12_languages_defined_in_frontend_dictionary(self):
        """Verify that UI_STRINGS in index.html defines all 12 supported languages."""
        match = re.search(r"const UI_STRINGS = ({.*?});\s*const browserLang", self.html, re.DOTALL)
        self.assertIsNotNone(match, "UI_STRINGS dictionary not found in index.html")
        ui_strings = json.loads(match.group(1))

        self.assertEqual(set(ui_strings.keys()), set(SUPPORTED_LANGUAGES))
        for lang in SUPPORTED_LANGUAGES:
            self.assertIn(lang, ui_strings)
            lang_dict = ui_strings[lang]
            # Key checks
            required_keys = [
                "workspaceTag", "navIntelligence", "navIntake", "navCarePlan", "navMeds",
                "navJourney", "navSafetyGate", "navAlerts", "navProviders", "navAudit",
                "btnDocIntake", "btnRunPipeline", "metricPatient", "metricGrounding",
                "sidebarStart", "sidebarCaseOverview", "sidebarDocIntake", "sidebarClinicalOutputs",
                "sidebarCarePlan", "sidebarMedPrecision", "sidebarRecoveryTimeline",
                "sidebarSafety", "sidebarSafetyReview", "sidebarAlerts", "sidebarCoordinate",
                "sidebarProviderFinder", "sidebarClinicalAudit", "sidebarUploadBoxTitle",
                "overviewKicker", "overviewH1", "overviewLead", "overviewBtnCare", "overviewBtnSafety",
                "proofEvidence", "proofSafety", "proofReview", "cardValidationGate",
                "intakeKicker", "intakeH1", "dropTitle", "btnChooseFile", "btnRunPipelineAction",
                "carePlanKicker", "carePlanH1", "tasksLabel",
                "medsKicker", "medsH1", "thMedication", "thFormulation", "thDose", "thRoute", "thFrequency", "thSource", "thSafety",
                "timelineKicker", "timelineH1",
                "safetyKicker", "safetyH1", "safetyMandatoryStatement", "btnAssignReviewer", "btnKeepPaused",
                "alertsKicker", "alertsH1", "btnAcknowledge", "btnSnooze", "btnDismiss",
                "providerKicker", "providerH1", "labelSpecialty", "labelCity", "btnSearchProviders", "providerDisclosure",
                "auditKicker", "auditH1",
                "tabProvenance", "tabReminders", "tabProvider", "tabAudit",
                "inspectorProvenanceKicker", "inspectorCoords", "inspectorCharOffset", "inspectorGrounding", "inspectorSafety"
            ]
            for rk in required_keys:
                self.assertIn(rk, lang_dict, f"Missing key {rk} in language {lang}")
                self.assertTrue(len(lang_dict[rk].strip()) > 0, f"Empty string for {rk} in language {lang}")

    def test_03_declarative_i18n_attributes_in_html(self):
        """Verify that index.html contains data-i18n attributes on primary UI elements."""
        self.assertIn('data-i18n="workspaceTag"', self.html)
        self.assertIn('data-i18n="navIntelligence"', self.html)
        self.assertIn('data-i18n="navCarePlan"', self.html)
        self.assertIn('data-i18n="sidebarCaseOverview"', self.html)
        self.assertIn('data-i18n="sidebarDocIntake"', self.html)
        self.assertIn('data-i18n="overviewKicker"', self.html)
        self.assertIn('data-i18n="dropTitle"', self.html)
        self.assertIn('data-i18n="thMedication"', self.html)
        self.assertIn('data-i18n="safetyMandatoryStatement"', self.html)
        self.assertIn('data-i18n="btnSearchProviders"', self.html)
        self.assertIn('data-i18n="labelSpecialty"', self.html)

    def test_04_served_endpoints_contain_i18n(self):
        """Verify GET / and GET /app both serve the localized cockpit."""
        r_root = self.client.get("/")
        self.assertEqual(r_root.status_code, 200)
        self.assertIn("const UI_STRINGS", r_root.text)
        self.assertIn("function applyI18n", r_root.text)

        r_app = self.client.get("/app")
        self.assertEqual(r_app.status_code, 200)
        self.assertIn("const UI_STRINGS", r_app.text)
        self.assertIn("function applyI18n", r_app.text)

    def test_05_orchestration_all_12_languages(self):
        """Verify backend orchestration returns proper localized results for all 12 languages."""
        for lang in SUPPORTED_LANGUAGES:
            r = self.client.post("/api/v1/orchestrate", json={
                "case_id": "case_absolute_cardiology",
                "language": lang
            })
            self.assertEqual(r.status_code, 200, f"Failed orchestrate for {lang}")
            data = r.json()
            self.assertIn("language", data)
            self.assertEqual(data["language"]["resolved"], lang)
            self.assertIn("tasks", data)
            self.assertIn("medications", data)
            self.assertIn("notifications", data)

    def test_06_patient_language_patch_and_propagation(self):
        """Verify PATCH /api/v1/patients/{patient_id}/language updates preferred language."""
        r = self.client.patch("/api/v1/patients/SYNTH-PT-801/language", json={"language": "kn"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["preferred_language"], "kn")

        # Now orchestrate without language param and verify it resolves to kn
        r_orch = self.client.post("/api/v1/orchestrate", json={"case_id": "case_absolute_cardiology"})
        self.assertEqual(r_orch.status_code, 200)
        self.assertEqual(r_orch.json()["language"]["resolved"], "kn")

        # Reset back to ta (original preferred language for case_absolute_cardiology)
        self.client.patch("/api/v1/patients/SYNTH-PT-801/language", json={"language": "ta"})

    def test_07_invalid_language_handling(self):
        """Verify backend and frontend reject/fallback unsupported languages."""
        r = self.client.post("/api/v1/orchestrate", json={
            "case_id": "case_absolute_cardiology",
            "language": "xx_invalid"
        })
        # Backend rejects with 422
        self.assertIn(r.status_code, [400, 422])

    def test_08_race_condition_and_persistence_mechanisms(self):
        """Verify index.html contains sequence guard and language preservation logic."""
        self.assertIn("state.requestSeq", self.html)
        self.assertIn("seq !== state.requestSeq", self.html)
        self.assertIn('localStorage.setItem("carebridge-language", lang)', self.html)
        self.assertIn('document.documentElement.lang = lang', self.html)

    def test_09_all_12_options_present_in_language_selector(self):
        """Verify index.html selector has options for all 12 languages."""
        for lang in SUPPORTED_LANGUAGES:
            self.assertIn(f'<option value="{lang}">', self.html)

if __name__ == "__main__":
    unittest.main()
