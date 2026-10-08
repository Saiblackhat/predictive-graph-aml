"""Runner for the immutable external semantic safety suite V01-V14."""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from backend.main import app


def main() -> None:
    cases_path = Path(__file__).resolve().parent / "v01_v14_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    client = TestClient(app, raise_server_exceptions=False)

    header = (
        "case | HTTP | overall_status | item_safety_status | item_review_status "
        "| hazard_codes | patient_releasable | expected_match | latency_ms"
    )
    lines = [header]
    full_records = []

    for entry in cases:
        case_id = entry["case_id"]
        payload = entry["payload"]
        t0 = time.perf_counter()
        resp = client.post("/api/v1/validate", json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        http_status = resp.status_code
        body = resp.json()
        if http_status == 200:
            vp = body.get("validated_payload", {})
            overall_status = body.get("overall_safety_status", "UNKNOWN")
            patient_releasable = body.get("patient_releasable", False)
            hazard_codes_list = body.get("hazard_codes_detected", [])
            hazard_codes = ",".join(hazard_codes_list) if hazard_codes_list else "NONE"
            items = (
                vp.get("medications", [])
                + vp.get("follow_ups", [])
                + vp.get("warning_signs", [])
                + vp.get("tasks", [])
            )
            item_safety = (
                "; ".join(f"{it['id']}:{it['safety_status']}" for it in items)
                if items
                else "NONE"
            )
            item_review = (
                "; ".join(f"{it['id']}:{it['review_status']}" for it in items)
                if items
                else "NONE"
            )
        else:
            overall_status = f"HTTP_{http_status}_REJECTED"
            patient_releasable = False
            hazard_codes = "NONE"
            item_safety = "N/A"
            item_review = "N/A"

        expected_statuses = entry.get("expected_overall_status", [])
        expected_rel = entry.get("expected_patient_releasable")
        matches = (
            http_status == 200
            and overall_status in expected_statuses
            and patient_releasable == expected_rel
        )
        match_label = "PASS" if matches else "FAIL"

        line = (
            f"{case_id} | {http_status} | {overall_status} | {item_safety} | "
            f"{item_review} | {hazard_codes} | {patient_releasable} | "
            f"{match_label} | {elapsed_ms:.2f}"
        )
        lines.append(line)
        full_records.append(
            {
                "case_id": case_id,
                "description": entry.get("description", ""),
                "http_status": http_status,
                "expected_overall_status": expected_statuses,
                "actual_overall_status": overall_status,
                "expected_patient_releasable": expected_rel,
                "actual_patient_releasable": patient_releasable,
                "matches_expected": matches,
                "latency_ms": round(elapsed_ms, 2),
                "input_payload": payload,
                "validator_output": body,
            }
        )

    output_text = "\n".join(lines) + "\n"
    print(output_text, end="")

    out_json_env = os.environ.get("V01_V14_JSON_OUT")
    if out_json_env:
        Path(out_json_env).write_text(
            json.dumps(full_records, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
