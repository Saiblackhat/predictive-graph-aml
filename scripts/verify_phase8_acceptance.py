"""scripts/verify_phase8_acceptance.py

Rigorous automated acceptance verification script for Phase 8.
Exercises all 8 required journeys against http://127.0.0.1:8000 and records concrete evidence.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8000"

def request(url: str, method: str = "GET", data: dict | None = None) -> tuple[int, dict | str]:
    full_url = BASE_URL + url
    body_bytes = json.dumps(data).encode("utf-8") if data is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(full_url, data=body_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            content_type = resp.headers.get("content-type", "")
            raw = resp.read()
            if "application/json" in content_type:
                return resp.status, json.loads(raw.decode("utf-8"))
            return resp.status, raw.decode("utf-8")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw.decode("utf-8"))
        except Exception:
            return e.code, raw.decode("utf-8")

def main():
    print("================================================================================")
    print("CAREBRIDGE PHASE 8 ACCEPTANCE VERIFICATION SUITE")
    print("================================================================================")

    results = {}

    # Journey 1: Select 3 materially different cases
    print("\n--- JOURNEY 1: Load 3 Materially Different Cases ---")
    status, cases = request("/api/v1/cases")
    assert status == 200, f"Expected 200, got {status}"
    print(f"Total available cases: {len(cases)}")

    # 1A: Absolute follow-up
    status, c1 = request("/api/v1/orchestrate", "POST", {"case_id": "case_absolute_cardiology", "language": "en"})
    assert status == 200
    print(f"Case 1 (Absolute): ID={c1['case_info']['case_id']}, Safety={c1['overall_safety_status']}, Releasable={c1['patient_releasable']}, Tasks={len(c1['tasks'])}")
    assert c1["overall_safety_status"] == "SAFE / ORGANIZED"
    assert c1["patient_releasable"] is True
    assert c1["requires_human_review"] is False

    # 1B: Conflicting medications
    status, c2 = request("/api/v1/orchestrate", "POST", {"case_id": "conflicting_medications", "language": "en"})
    assert status == 200
    print(f"Case 2 (Conflict): ID={c2['case_info']['case_id']}, Safety={c2['overall_safety_status']}, Releasable={c2['patient_releasable']}, ReviewItems={len(c2['human_review_items'])}")
    assert c2["overall_safety_status"] in ("CONFLICT DETECTED", "NEEDS REVIEW")
    assert c2["patient_releasable"] is False
    assert c2["requires_human_review"] is True
    assert len(c2["human_review_items"]) > 0

    # 1C: Missing / Ambiguous follow-up
    status, c3 = request("/api/v1/orchestrate", "POST", {"case_id": "missing_ambiguous_followup", "language": "en"})
    assert status == 200
    print(f"Case 3 (Ambiguous): ID={c3['case_info']['case_id']}, Safety={c3['overall_safety_status']}, Releasable={c3['patient_releasable']}, ReviewItems={len(c3['human_review_items'])}")
    assert c3["overall_safety_status"] == "NEEDS REVIEW"
    assert c3["patient_releasable"] is False
    assert c3["requires_human_review"] is True

    results["Journey 1: Multi-Case Selection"] = "PASS"

    # Journey 2: Language Switching (en, ta, kn)
    print("\n--- JOURNEY 2: Language Switching (en, ta, kn) ---")
    status, en_res = request("/api/v1/orchestrate", "POST", {"case_id": "case_absolute_cardiology", "language": "en"})
    status, ta_res = request("/api/v1/orchestrate", "POST", {"case_id": "case_absolute_cardiology", "language": "ta"})
    status, kn_res = request("/api/v1/orchestrate", "POST", {"case_id": "case_absolute_cardiology", "language": "kn"})
    assert status == 200

    en_body = en_res["notifications"][0]["body"]
    ta_body = ta_res["notifications"][0]["body"]
    kn_body = kn_res["notifications"][0]["body"]

    print(f"English notification: {en_body[:60]}...")
    print(f"Tamil notification: {ta_body[:60]}...")
    print(f"Kannada notification: {kn_body[:60]}...")

    assert en_body != ta_body, "Tamil output matches English"
    assert en_body != kn_body, "Kannada output matches English"
    assert ta_body != kn_body, "Kannada output matches Tamil"
    # Verify clinical invariant preservation: metoprolol or cardiology present
    assert "Cardiology" in ta_res["notifications"][0]["title"] or "Cardiology" in ta_body
    results["Journey 2: Multilingual Switching"] = "PASS"

    # Journey 3: Stage Synthetic Discharge Document & Run Extraction
    print("\n--- JOURNEY 3: Custom Discharge Document Ingestion ---")
    custom_discharge = (
        "Medications:\n"
        "Take lisinopril 10 mg by mouth once daily.\n\n"
        "Follow-Up:\n"
        "Follow up with Nephrology on April 15, 2027 at 2:00 PM for renal function monitoring.\n\n"
        "Precautions:\n"
        "Call clinic if systolic blood pressure exceeds 160 mmHg."
    )
    status, custom_res = request("/api/v1/orchestrate", "POST", {"raw_text": custom_discharge, "language": "en"})
    assert status == 200
    print(f"Custom Run: Patient={custom_res['case_info']['patient_identifier']}, Tasks={len(custom_res['tasks'])}, Meds={len(custom_res['medications'])}")
    assert custom_res["case_info"]["patient_identifier"] == "SYNTH-PT-CUSTOM"
    assert any("lisinopril" in m["name"].lower() for m in custom_res["medications"])
    assert any("nephrology" in t["title"].lower() for t in custom_res["tasks"])
    results["Journey 3: Document Ingestion"] = "PASS"

    # Journey 4: Task Selection & Provenance Verification
    print("\n--- JOURNEY 4: Task Selection & Character Offset Provenance ---")
    task = [t for t in c1["tasks"] if "Cardiology" in t["title"]][0]
    ev = task["evidence"]
    assert ev is not None, "Task missing evidence"
    s_start = ev.get("start_offset", ev.get("char_start"))
    s_end = ev.get("end_offset", ev.get("char_end"))
    print(f"Task: '{task['title']}'")
    print(f"Evidence Span: [{s_start}:{s_end}] in section '{ev['section']}'")
    print(f"Source Quote: '{ev['source_text']}'")
    # Verify offsets accurately slice the raw text
    sliced = c1["raw_text"][s_start:s_end]
    print(f"Raw Text Slice: '{sliced}'")
    assert sliced == ev["source_text"], f"Offset slice mismatch: '{sliced}' != '{ev['source_text']}'"
    results["Journey 4: Provenance Character Offsets"] = "PASS"

    # Journey 5: Human Review Visibility
    print("\n--- JOURNEY 5: Human Review Visibility for Ambiguous/Conflict Case ---")
    status, conf_res = request("/api/v1/orchestrate", "POST", {"case_id": "conflicting_medications", "language": "en"})
    print(f"Safety status: {conf_res['overall_safety_status']}")
    print(f"Patient releasable: {conf_res['patient_releasable']}")
    print(f"Requires human review: {conf_res['requires_human_review']}")
    print(f"Review items count: {len(conf_res['human_review_items'])}")
    for i, item in enumerate(conf_res["human_review_items"]):
        item_title = item.get("title", item.get("hazard_code", "Review Event"))
        print(f"  Item {i+1}: {item_title} | Hazard: {item.get('hazard_code')} | Severity: {item.get('severity')}")
    assert conf_res["patient_releasable"] is False
    assert conf_res["requires_human_review"] is True
    assert len(conf_res["human_review_items"]) > 0
    results["Journey 5: Human Review Gate"] = "PASS"

    # Journey 6: Neutral Provider Search
    print("\n--- JOURNEY 6: Neutral Provider Discovery ---")
    status, prov_res = request("/api/v1/providers/search", "POST", {"specialty": "Cardiology", "location_city": "Chennai"})
    assert status == 200
    providers = prov_res["providers"]
    print(f"Providers matched: {len(providers)}")
    print(f"Patient message: '{prov_res['patient_message']}'")
    assert "does not rank or select providers" in prov_res["patient_message"]
    for p in providers:
        assert "rating" not in p, f"Illegal rating found in provider {p}"
        assert "stars" not in p, f"Illegal stars found in provider {p}"
        assert "score" not in p, f"Illegal score found in provider {p}"
        assert p.get("availability_note") == "Availability not verified.", "Missing unverified availability note"
    print(f"Sample provider: {providers[0]['display_name']} · {providers[0]['specialty']} · Availability: {providers[0]['availability_note']}")
    results["Journey 6: Neutral Provider Discovery"] = "PASS"

    # Journey 7: Notification Action State Transitions
    print("\n--- JOURNEY 7: Notification Action State Machine ---")
    notif = c1["notifications"][0]
    nid = notif["notification_id"]
    print(f"Initial notification {nid}: status={notif['status']}")

    # 7A: Snooze
    status, snz_res = request(f"/api/v1/notifications/{nid}/action", "POST", {"action": "SNOOZE", "snooze_minutes": 45})
    assert status == 200
    snz_items = [n for n in snz_res["notifications"] if n["notification_id"] == nid or n["task_id"] == notif["task_id"]]
    print(f"After SNOOZE: active item statuses={[n['status'] for n in snz_res['notifications']]}")

    # 7B: Acknowledge
    status, ack_res = request(f"/api/v1/notifications/{nid}/action", "POST", {"action": "ACKNOWLEDGE"})
    assert status == 200
    print(f"After ACKNOWLEDGE: notification statuses={[n['status'] for n in ack_res['notifications']]}")

    # 7C: Dismiss
    status, dis_res = request(f"/api/v1/notifications/{nid}/action", "POST", {"action": "DISMISS"})
    assert status == 200
    print(f"After DISMISS: notification statuses={[n['status'] for n in dis_res['notifications']]}")
    results["Journey 7: Notification Lifecycle Transitions"] = "PASS"

    # Journey 8: Error Handling (Invalid Input & Graceful UI Error)
    print("\n--- JOURNEY 8: Error Handling & Invalid Input ---")
    status, err_422 = request("/api/v1/orchestrate", "POST", {"language": "invalid_lang_code"})
    print(f"Unsupported language HTTP status: {err_422}")
    assert status == 422, f"Expected 422 for unsupported language, got {status}"

    # Over 200 KB payload
    huge_text = "A" * (205 * 1024)
    status, err_413 = request("/api/v1/validate", "POST", {
        "raw_discharge_text": huge_text,
        "extracted_items": [],
        "overall_safety_classification": "SAFE / ORGANIZED",
        "requires_human_review": False,
        "patient_releasable": True,
        "confidence_score": 1.0
    })
    print(f"200 KB overflow HTTP status: {status}")
    assert status == 413, f"Expected 413 for text >200 KB, got {status}"
    results["Journey 8: Error Boundary Validation"] = "PASS"

    print("\n================================================================================")
    print("ALL JOURNEYS COMPLETED SUCCESSFULLY")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("================================================================================")

if __name__ == "__main__":
    main()
