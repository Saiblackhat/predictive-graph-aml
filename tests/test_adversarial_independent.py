"""Independent Adversarial Verification Gate for CAREBRIDGE.

Exploratory probes against the current validation boundary (POST /api/v1/validate).
Does NOT modify production code or assert that the implementation passes;
records actual HTTP status, validation status, flags, human review state,
and compares actual vs. expected clinical safety behavior.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from typing import Any, Callable

from fastapi.testclient import TestClient

from backend.main import app


@dataclass
class ProbeResult:
    case_id: str
    expected_summary: str
    http_status: int
    actual_status: str
    flags: list[str]
    human_review: str
    result_label: str
    raw_response: Any


class TestAdversarialIndependentGate(unittest.TestCase):
    """Runs all 18 adversarial probes and outputs a complete verification report."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app, raise_server_exceptions=False)
        cls.results: list[ProbeResult] = []

    def _base_clean_dict(self) -> dict[str, Any]:
        return {
            "case_id": "ADV-CASE",
            "patient_identifier": "PT-ADV",
            "raw_discharge_text": (
                "Continue Atorvastatin 40 mg by mouth once daily. "
                "Follow up with Primary Care within 7 days."
            ),
            "medications": [
                {
                    "id": "med-1",
                    "medication_name": "Atorvastatin",
                    "action": "CONTINUE",
                    "dosage": "40 mg",
                    "frequency": "once daily",
                    "route": "by mouth",
                    "previous_dosage": None,
                    "indication_or_note": None,
                    "confidence": 0.98,
                    "safety_status": "SAFE / ORGANIZED",
                    "review_status": "AUTO_APPROVED",
                    "safety_flags": [],
                    "patient_friendly_summary": "Continue Atorvastatin 40 mg once daily.",
                    "source_evidence": {
                        "source_text": "Continue Atorvastatin 40 mg by mouth once daily.",
                        "source_section": "Medications",
                        "source_page": 1,
                        "start_char": None,
                        "end_char": None,
                    },
                }
            ],
            "follow_ups": [],
            "warning_signs": [],
            "tasks": [],
            "provider_matches": [],
            "overall_safety_status": "SAFE / ORGANIZED",
            "requires_human_review": False,
            "global_safety_flags": [],
        }

    def _send_json(self, payload: Any) -> tuple[int, Any]:
        res = self.client.post("/api/v1/validate", json=payload)
        try:
            body = res.json()
        except Exception:
            body = res.text
        return res.status_code, body

    def _send_raw_content(self, raw_bytes: bytes) -> tuple[int, Any]:
        res = self.client.post(
            "/api/v1/validate",
            content=raw_bytes,
            headers={"Content-Type": "application/json"},
        )
        try:
            body = res.json()
        except Exception:
            body = res.text
        return res.status_code, body

    def _extract_summary(
        self, http_status: int, body: Any
    ) -> tuple[str, list[str], str]:
        if http_status == 200 and isinstance(body, dict):
            actual_status = str(body.get("overall_safety_status", "UNKNOWN"))
            flags = [str(code) for code in body.get("hazard_codes_detected", [])]
            human_review = str(body.get("requires_human_review", False))
            return actual_status, flags, human_review
        return f"HTTP_{http_status}_REJECTED", [], "N/A (REJECTED)"

    def _record_probe(
        self,
        case_id: str,
        expected_summary: str,
        http_status: int,
        body: Any,
        predicate: Callable[[int, str, list[str], str, Any], bool],
    ) -> None:
        actual_status, flags, human_review = self._extract_summary(http_status, body)
        matches = predicate(http_status, actual_status, flags, human_review, body)
        label = "MATCHES-EXPECTED" if matches else "DIFFERS"
        self.results.append(
            ProbeResult(
                case_id=case_id,
                expected_summary=expected_summary,
                http_status=http_status,
                actual_status=actual_status,
                flags=flags,
                human_review=human_review,
                result_label=label,
                raw_response=body,
            )
        )

    def test_run_all_18_adversarial_probes(self) -> None:
        # 1. UNRELATED REAL SOURCE QUOTE
        p1 = self._base_clean_dict()
        p1["medications"][0]["source_evidence"]["source_text"] = (
            "Follow up with Primary Care within 7 days."
        )
        code1, body1 = self._send_json(p1)
        self._record_probe(
            case_id="1. UNRELATED_REAL_SOURCE_QUOTE",
            expected_summary="NEEDS REVIEW / Evidence-Claim Mismatch",
            http_status=code1,
            body=body1,
            predicate=lambda code, status, _flags, hr, _b: (
                code == 200 and hr == "True" and status != "SAFE / ORGANIZED"
            ),
        )

        # 2. STRING CONFIDENCE ("0.9")
        p2 = self._base_clean_dict()
        p2["medications"][0]["confidence"] = "0.9"
        code2, body2 = self._send_json(p2)
        self._record_probe(
            case_id="2. STRING_CONFIDENCE",
            expected_summary="HTTP 422 Schema Rejection (strict float)",
            http_status=code2,
            body=body2,
            predicate=lambda code, _status, _flags, _hr, _b: code == 422,
        )

        # 3. STRING BOOLEAN ("false")
        p3 = self._base_clean_dict()
        p3["requires_human_review"] = "false"
        code3, body3 = self._send_json(p3)
        self._record_probe(
            case_id="3. STRING_BOOLEAN",
            expected_summary="HTTP 422 Schema Rejection (strict bool)",
            http_status=code3,
            body=body3,
            predicate=lambda code, _status, _flags, _hr, _b: code == 422,
        )

        # 4. NEGATED MEDICATION CHANGE
        p4 = self._base_clean_dict()
        p4["raw_discharge_text"] = (
            "No change to medication is required. Continue Atorvastatin 40 mg by mouth once daily."
        )
        p4["medications"][0]["source_evidence"]["source_text"] = (
            "No change to medication is required. Continue Atorvastatin 40 mg by mouth once daily."
        )
        code4, body4 = self._send_json(p4)
        self._record_probe(
            case_id="4. NEGATED_MEDICATION_CHANGE",
            expected_summary="SAFE / ORGANIZED (No false-positive change)",
            http_status=code4,
            body=body4,
            predicate=lambda code, status, flags, hr, _b: (
                code == 200
                and status == "SAFE / ORGANIZED"
                and "MEDICATION_CHANGE_DETECTED" not in flags
                and hr == "False"
            ),
        )

        # 5. SILENT MEDICATION CONFLICT
        p5 = self._base_clean_dict()
        p5["raw_discharge_text"] = (
            "Continue lisinopril 10 mg daily. Hold lisinopril until review."
        )
        p5["medications"][0]["medication_name"] = "lisinopril"
        p5["medications"][0]["action"] = "CONTINUE"
        p5["medications"][0]["dosage"] = "10 mg"
        p5["medications"][0]["frequency"] = "daily"
        p5["medications"][0]["patient_friendly_summary"] = "Continue lisinopril."
        p5["medications"][0]["source_evidence"]["source_text"] = (
            "Continue lisinopril 10 mg daily."
        )
        code5, body5 = self._send_json(p5)
        self._record_probe(
            case_id="5. SILENT_MEDICATION_CONFLICT",
            expected_summary="CONFLICT DETECTED / HUMAN REVIEW",
            http_status=code5,
            body=body5,
            predicate=lambda code, status, flags, hr, _b: (
                code == 200
                and status == "CONFLICT DETECTED"
                and "CONFLICTING_INSTRUCTIONS" in flags
                and hr == "True"
            ),
        )

        # 6. OMITTED MEDICATION CHANGE
        p6 = self._base_clean_dict()
        p6["raw_discharge_text"] = (
            "Continue Atorvastatin 40 mg by mouth once daily. Switch from Prinivil to losartan."
        )
        code6, body6 = self._send_json(p6)
        self._record_probe(
            case_id="6. OMITTED_MEDICATION_CHANGE",
            expected_summary="NEEDS REVIEW (Omitted med switch in doc)",
            http_status=code6,
            body=body6,
            predicate=lambda code, status, _flags, hr, _b: (
                code == 200 and status != "SAFE / ORGANIZED" and hr == "True"
            ),
        )

        # 7. BRAND/GENERIC CONFLICT (Prinivil -> CONTINUE, Lisinopril -> STOP)
        p7 = self._base_clean_dict()
        p7["raw_discharge_text"] = (
            "Prinivil (lisinopril): Continue Prinivil 10 mg daily. STOP Lisinopril 10 mg daily."
        )
        p7["medications"] = [
            {
                "id": "med-bg-1",
                "medication_name": "Prinivil",
                "action": "CONTINUE",
                "dosage": "10 mg",
                "frequency": "daily",
                "route": "by mouth",
                "previous_dosage": None,
                "indication_or_note": None,
                "confidence": 0.95,
                "safety_status": "SAFE / ORGANIZED",
                "review_status": "AUTO_APPROVED",
                "safety_flags": [],
                "patient_friendly_summary": None,
                "source_evidence": {
                    "source_text": "Continue Prinivil 10 mg daily.",
                    "source_section": "Medications",
                    "source_page": 1,
                    "start_char": None,
                    "end_char": None,
                },
            },
            {
                "id": "med-bg-2",
                "medication_name": "Lisinopril",
                "action": "STOP_DISCONTINUE",
                "dosage": "10 mg",
                "frequency": "daily",
                "route": None,
                "previous_dosage": None,
                "indication_or_note": None,
                "confidence": 0.95,
                "safety_status": "SAFE / ORGANIZED",
                "review_status": "AUTO_APPROVED",
                "safety_flags": [],
                "patient_friendly_summary": None,
                "source_evidence": {
                    "source_text": "STOP Lisinopril 10 mg daily.",
                    "source_section": "Medications",
                    "source_page": 1,
                    "start_char": None,
                    "end_char": None,
                },
            },
        ]
        code7, body7 = self._send_json(p7)
        self._record_probe(
            case_id="7. BRAND_GENERIC_CONFLICT",
            expected_summary="CONFLICT DETECTED (Prinivil vs Lisinopril)",
            http_status=code7,
            body=body7,
            predicate=lambda code, status, flags, _hr, _b: (
                code == 200
                and status == "CONFLICT DETECTED"
                and "CONFLICTING_INSTRUCTIONS" in flags
            ),
        )

        # 8. EQUIVALENT TIMEFRAMES ("within 1 week" vs "within 7 days")
        p8 = self._base_clean_dict()
        p8["raw_discharge_text"] = (
            "Follow up with Cardiology within 1 week. Follow up with Cardiology within 7 days."
        )
        p8["medications"] = []
        p8["follow_ups"] = [
            {
                "id": "fu-eq-1",
                "specialty": "Cardiology",
                "provider_name": None,
                "timeframe": "within 1 week",
                "reason": None,
                "appointment_scheduled": None,
                "provider_claim_note": None,
                "confidence": 0.95,
                "safety_status": "SAFE / ORGANIZED",
                "review_status": "AUTO_APPROVED",
                "safety_flags": [],
                "patient_friendly_summary": None,
                "source_evidence": {
                    "source_text": "Follow up with Cardiology within 1 week.",
                    "source_section": "Follow-Up",
                    "source_page": 1,
                    "start_char": None,
                    "end_char": None,
                },
            },
            {
                "id": "fu-eq-2",
                "specialty": "Cardiology",
                "provider_name": None,
                "timeframe": "within 7 days",
                "reason": None,
                "appointment_scheduled": None,
                "provider_claim_note": None,
                "confidence": 0.95,
                "safety_status": "SAFE / ORGANIZED",
                "review_status": "AUTO_APPROVED",
                "safety_flags": [],
                "patient_friendly_summary": None,
                "source_evidence": {
                    "source_text": "Follow up with Cardiology within 7 days.",
                    "source_section": "Follow-Up",
                    "source_page": 1,
                    "start_char": None,
                    "end_char": None,
                },
            },
        ]
        code8, body8 = self._send_json(p8)
        self._record_probe(
            case_id="8. EQUIVALENT_TIMEFRAMES",
            expected_summary="SAFE / ORGANIZED (1 week == 7 days)",
            http_status=code8,
            body=body8,
            predicate=lambda code, status, flags, _hr, _b: (
                code == 200
                and status != "CONFLICT DETECTED"
                and "CONFLICTING_INSTRUCTIONS" not in flags
            ),
        )

        # 9. UNSUPPORTED DOSAGE (Extracted: "aspirin 75 mg", Raw doc: "aspirin daily")
        p9 = self._base_clean_dict()
        p9["raw_discharge_text"] = "Continue aspirin daily."
        p9["medications"][0]["medication_name"] = "aspirin"
        p9["medications"][0]["dosage"] = "75 mg"
        p9["medications"][0]["frequency"] = "daily"
        p9["medications"][0]["patient_friendly_summary"] = "aspirin 75 mg"
        p9["medications"][0]["source_evidence"]["source_text"] = (
            "Continue aspirin daily."
        )
        code9, body9 = self._send_json(p9)
        self._record_probe(
            case_id="9. UNSUPPORTED_DOSAGE",
            expected_summary="NEEDS REVIEW (Unverified '75 mg' dosage)",
            http_status=code9,
            body=body9,
            predicate=lambda code, status, _flags, hr, _b: (
                code == 200 and status != "SAFE / ORGANIZED" and hr == "True"
            ),
        )

        # 10. MISSING SOURCE EVIDENCE
        p10 = self._base_clean_dict()
        p10["medications"][0]["source_evidence"] = None
        code10, body10 = self._send_json(p10)
        self._record_probe(
            case_id="10. MISSING_SOURCE_EVIDENCE",
            expected_summary="NEEDS REVIEW + MISSING_SOURCE_EVIDENCE",
            http_status=code10,
            body=body10,
            predicate=lambda code, status, flags, hr, _b: (
                code == 422
                or (
                    code == 200
                    and status == "NEEDS REVIEW"
                    and "MISSING_SOURCE_EVIDENCE" in flags
                    and hr == "True"
                )
            ),
        )

        # 11. EMPTY ITEMS
        p11 = self._base_clean_dict()
        p11["medications"] = []
        code11, body11 = self._send_json(p11)
        self._record_probe(
            case_id="11. EMPTY_ITEMS",
            expected_summary="NEEDS REVIEW or 422 (Not SAFE / ORGANIZED)",
            http_status=code11,
            body=body11,
            predicate=lambda code, status, _flags, hr, _b: (
                code == 422
                or (code == 200 and status != "SAFE / ORGANIZED" and hr == "True")
            ),
        )

        # 12A. EMPTY RAW DOCUMENT ("")
        p12a = self._base_clean_dict()
        p12a["raw_discharge_text"] = ""
        code12a, body12a = self._send_json(p12a)
        self._record_probe(
            case_id="12A. EMPTY_RAW_DOCUMENT_EMPTY_STR",
            expected_summary="HTTP 422 Schema Rejection",
            http_status=code12a,
            body=body12a,
            predicate=lambda code, _status, _flags, _hr, _b: code == 422,
        )

        # 12B. WHITESPACE-ONLY RAW DOCUMENT ("   \n\t  ")
        p12b = self._base_clean_dict()
        p12b["raw_discharge_text"] = "   \n\t  "
        code12b, body12b = self._send_json(p12b)
        self._record_probe(
            case_id="12B. EMPTY_RAW_DOCUMENT_WHITESPACE",
            expected_summary="HTTP 422 Schema Rejection",
            http_status=code12b,
            body=body12b,
            predicate=lambda code, _status, _flags, _hr, _b: code == 422,
        )

        # 13. DUPLICATE ITEMS (exact same extracted item twice)
        p13 = self._base_clean_dict()
        p13["medications"].append(dict(p13["medications"][0]))
        code13, body13 = self._send_json(p13)
        self._record_probe(
            case_id="13. DUPLICATE_ITEMS",
            expected_summary="Deduplicated (len=1) or Flagged for Review",
            http_status=code13,
            body=body13,
            predicate=lambda code, status, _flags, hr, b: (
                code == 200
                and (
                    hr == "True"
                    or status != "SAFE / ORGANIZED"
                    or len(b.get("validated_payload", {}).get("medications", [])) == 1
                )
            ),
        )

        # 14. LARGE DOCUMENT (~200 KB of raw discharge text)
        p14 = self._base_clean_dict()
        padding_line = "Patient remained hemodynamically stable during observation.\n"
        p14["raw_discharge_text"] = p14["raw_discharge_text"] + (padding_line * 3400)
        code14, body14 = self._send_json(p14)
        self._record_probe(
            case_id="14. LARGE_DOCUMENT_200KB",
            expected_summary="HTTP 200 Graceful Processing (~200KB)",
            http_status=code14,
            body=body14,
            predicate=lambda code, status, _flags, _hr, _b: (
                code == 200 and status == "SAFE / ORGANIZED"
            ),
        )

        # 15A. UNSUPPORTED PROVIDER CLAIMS ("best")
        p15a = self._base_clean_dict()
        p15a["provider_matches"] = [
            {
                "provider_id": "prov-1",
                "name": "Dr. Rostova",
                "specialty": "Primary Care",
                "facility": "Metro Clinic",
                "distance_miles": 2.0,
                "accepts_new_patients_listed": True,
                "verification_disclaimer": "CAREBRIDGE does not guarantee provider suitability or availability.",
                "claim_text": "Dr. Rostova is the best doctor for you.",
            }
        ]
        code15a, body15a = self._send_json(p15a)
        self._record_probe(
            case_id="15A. PROVIDER_CLAIM_BEST",
            expected_summary="NEEDS REVIEW + UNSUPPORTED_PROVIDER_CLAIM",
            http_status=code15a,
            body=body15a,
            predicate=lambda code, _status, flags, hr, _b: (
                code == 200 and "UNSUPPORTED_PROVIDER_CLAIM" in flags and hr == "True"
            ),
        )

        # 15B. UNSUPPORTED PROVIDER CLAIMS ("recommended")
        p15b = self._base_clean_dict()
        p15b["provider_matches"] = [
            {
                "provider_id": "prov-1",
                "name": "Dr. Rostova",
                "specialty": "Primary Care",
                "facility": "Metro Clinic",
                "distance_miles": 2.0,
                "accepts_new_patients_listed": True,
                "verification_disclaimer": "CAREBRIDGE does not guarantee provider suitability or availability.",
                "claim_text": "Dr. Rostova is our recommended specialist.",
            }
        ]
        code15b, body15b = self._send_json(p15b)
        self._record_probe(
            case_id="15B. PROVIDER_CLAIM_RECOMMENDED",
            expected_summary="NEEDS REVIEW + UNSUPPORTED_PROVIDER_CLAIM",
            http_status=code15b,
            body=body15b,
            predicate=lambda code, _status, flags, hr, _b: (
                code == 200 and "UNSUPPORTED_PROVIDER_CLAIM" in flags and hr == "True"
            ),
        )

        # 15C. UNSUPPORTED PROVIDER CLAIMS ("guaranteed")
        p15c = self._base_clean_dict()
        p15c["provider_matches"] = [
            {
                "provider_id": "prov-1",
                "name": "Dr. Rostova",
                "specialty": "Primary Care",
                "facility": "Metro Clinic",
                "distance_miles": 2.0,
                "accepts_new_patients_listed": True,
                "verification_disclaimer": "CAREBRIDGE does not guarantee provider suitability or availability.",
                "claim_text": "Appointment is guaranteed tomorrow.",
            }
        ]
        code15c, body15c = self._send_json(p15c)
        self._record_probe(
            case_id="15C. PROVIDER_CLAIM_GUARANTEED",
            expected_summary="NEEDS REVIEW + UNSUPPORTED_PROVIDER_CLAIM",
            http_status=code15c,
            body=body15c,
            predicate=lambda code, _status, flags, hr, _b: (
                code == 200 and "UNSUPPORTED_PROVIDER_CLAIM" in flags and hr == "True"
            ),
        )

        # 16. PROHIBITED MEDICATION ACTION ("increase dose to 20 mg", requires_human_review=false)
        p16 = self._base_clean_dict()
        p16["requires_human_review"] = False
        p16["medications"][0]["indication_or_note"] = "increase dose to 20 mg"
        p16["medications"][0]["patient_friendly_summary"] = "increase dose to 20 mg"
        code16, body16 = self._send_json(p16)
        self._record_probe(
            case_id="16. PROHIBITED_MEDICATION_ACTION",
            expected_summary="HUMAN REVIEW FORCED + PROHIBITED_CLINICAL_ACTION",
            http_status=code16,
            body=body16,
            predicate=lambda code, status, flags, hr, _b: (
                code == 200
                and hr == "True"
                and status == "NEEDS REVIEW"
                and "PROHIBITED_CLINICAL_ACTION" in flags
            ),
        )

        # 17. EXTRA UNKNOWN FIELD
        p17 = self._base_clean_dict()
        p17["medications"][0]["unexpected_llm_key"] = "injected_value"
        code17, body17 = self._send_json(p17)
        self._record_probe(
            case_id="17. EXTRA_UNKNOWN_FIELD",
            expected_summary="HTTP 422 Schema Rejection (extra=forbid)",
            http_status=code17,
            body=body17,
            predicate=lambda code, _status, _flags, _hr, _b: code == 422,
        )

        # 18A-E. MALFORMED REQUESTS
        code18a, body18a = self._send_raw_content(b"{malformed_json: true")
        self._record_probe(
            case_id="18A. MALFORMED_JSON_SYNTAX",
            expected_summary="HTTP 4xx Validation Error",
            http_status=code18a,
            body=body18a,
            predicate=lambda code, _status, _flags, _hr, _b: 400 <= code < 500,
        )

        code18b, body18b = self._send_json([])
        self._record_probe(
            case_id="18B. JSON_ARRAY_ROOT",
            expected_summary="HTTP 422 Validation Error",
            http_status=code18b,
            body=body18b,
            predicate=lambda code, _status, _flags, _hr, _b: code == 422,
        )

        code18c, body18c = self._send_raw_content(b"null")
        self._record_probe(
            case_id="18C. JSON_NULL_ROOT",
            expected_summary="HTTP 422 Validation Error",
            http_status=code18c,
            body=body18c,
            predicate=lambda code, _status, _flags, _hr, _b: code == 422,
        )

        code18d, body18d = self._send_json({})
        self._record_probe(
            case_id="18D. EMPTY_JSON_OBJECT",
            expected_summary="HTTP 422 Validation Error",
            http_status=code18d,
            body=body18d,
            predicate=lambda code, _status, _flags, _hr, _b: code == 422,
        )

        code18e, body18e = self._send_json(
            {"case_id": 12345, "medications": "not_a_list"}
        )
        self._record_probe(
            case_id="18E. STRUCTURALLY_INVALID_OBJECT",
            expected_summary="HTTP 422 Validation Error",
            http_status=code18e,
            body=body18e,
            predicate=lambda code, _status, _flags, _hr, _b: code == 422,
        )

        self._print_verification_report()

    def _print_verification_report(self) -> None:
        header = (
            f"{'CASE':<34} | {'EXPECTED':<44} | {'HTTP_STATUS':<11} | "
            f"{'ACTUAL_STATUS':<18} | {'FLAGS':<35} | {'HUMAN_REVIEW':<14} | {'RESULT'}"
        )
        separator = "-" * len(header)
        print("\n" + "=" * len(header))
        print("INDEPENDENT ADVERSARIAL VERIFICATION GATE REPORT")
        print("=" * len(header))
        print(header)
        print(separator)

        matches_count = 0
        differs_count = 0
        differs_list: list[ProbeResult] = []

        for r in self.results:
            if r.result_label == "MATCHES-EXPECTED":
                matches_count += 1
            else:
                differs_count += 1
                differs_list.append(r)

            flags_str = ",".join(r.flags) if r.flags else "NONE"
            print(
                f"{r.case_id:<34} | {r.expected_summary[:44]:<44} | "
                f"{r.http_status:<11} | {r.actual_status:<18} | "
                f"{flags_str:<35} | {r.human_review:<14} | {r.result_label}"
            )

        print(separator)
        print(f"TOTAL CASES: {len(self.results)}")
        print(f"MATCHES: {matches_count}")
        print(f"DIFFERS: {differs_count}")
        print("=" * len(header))

        if differs_list:
            print("\nFULL RESPONSES FOR ALL 'DIFFERS' CASES:")
            print("=" * 80)
            for d in differs_list:
                print(f"\n--- [DIFFERS] {d.case_id} ---")
                print(f"Expected: {d.expected_summary}")
                print(f"HTTP Status: {d.http_status}")
                print("Full Response:")
                print(json.dumps(d.raw_response))
            print("=" * 80)


if __name__ == "__main__":
    unittest.main()
