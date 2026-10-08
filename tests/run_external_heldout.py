"""External held-out evaluation runner for Claude-style blind/semi-blind JSON cases.

Usage:
    python tests/run_external_heldout.py [path/to/cases.json]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient

from backend.main import app


def run_external_cases(json_path: Path) -> list[str]:
    cases = json.loads(json_path.read_text(encoding="utf-8"))
    client = TestClient(app, raise_server_exceptions=False)
    header = "case | HTTP | overall_status | item_safety_status | item_review_status | hazard_codes | patient_releasable | latency_ms"
    lines = [header]
    print(header)

    for entry in cases:
        cid = entry["case_id"]
        payload = entry["payload"]
        t0 = time.perf_counter()
        res = client.post("/api/v1/validate", json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        http_code = res.status_code
        if http_code == 200:
            data = res.json()
            vp = data["validated_payload"]
            items = (
                vp.get("medications", [])
                + vp.get("follow_ups", [])
                + vp.get("warning_signs", [])
                + vp.get("tasks", [])
            )
            item_safety = (
                "; ".join(f"{i['id']}:{i['safety_status']}" for i in items)
                if items
                else "NONE"
            )
            item_review = (
                "; ".join(f"{i['id']}:{i['review_status']}" for i in items)
                if items
                else "NONE"
            )
            hazards = (
                ",".join(data["hazard_codes_detected"])
                if data["hazard_codes_detected"]
                else "NONE"
            )
            row = (
                f"{cid} | {http_code} | {data['overall_safety_status']} | "
                f"{item_safety} | {item_review} | {hazards} | "
                f"{data['patient_releasable']} | {elapsed_ms:.2f}"
            )
        else:
            row = (
                f"{cid} | {http_code} | HTTP_{http_code}_REJECTED | "
                f"N/A | N/A | NONE | False | {elapsed_ms:.2f}"
            )
        lines.append(row)
        print(row)

    return lines


def main() -> None:
    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent / "external_claude_style_cases.json"
    )
    lines = run_external_cases(target)
    out_path = Path(__file__).resolve().parent / "external_claude_style_output.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
