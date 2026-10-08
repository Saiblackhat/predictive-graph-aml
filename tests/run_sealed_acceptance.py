"""Sealed one-shot acceptance runner for Phase 3C (V01-V14 and V15-V20).

Strictly enforces:
- Presence check for both reviewer-sealed files before execution:
  - tests/v01_v14_cases.json
  - tests/v15_v20_cases.json
  If either file is missing, prints ONLY:
  WAITING FOR REVIEWER-SEALED V01–V14 AND V15–V20 TEST FILES.
- Clean git working tree verification before and after execution.
- Pre-execution and post-execution SHA256 verification of both case files,
  backend/safety/validator.py, backend/schemas/models.py,
  tests/test_adversarial_independent.py, and this runner.
- Single-execution lock so any attempt to rerun the suite fails loudly.
- Zero modification of supplied JSON files.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

V01_V14_PATH = ROOT_DIR / "tests" / "v01_v14_cases.json"
V15_V20_PATH = ROOT_DIR / "tests" / "v15_v20_cases.json"
VALIDATOR_PATH = ROOT_DIR / "backend" / "safety" / "validator.py"
MODELS_PATH = ROOT_DIR / "backend" / "schemas" / "models.py"
ADVERSARIAL_PATH = ROOT_DIR / "tests" / "test_adversarial_independent.py"
RUNNER_PATH = Path(__file__).resolve()

LOCK_DIR = Path(
    os.environ.get(
        "SEALED_ACCEPTANCE_OUT_DIR",
        str(Path.home() / ".gemini" / "antigravity" / "sealed_acceptance_state"),
    )
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def get_git_sha() -> str:
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


def get_git_status_short() -> str:
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


def _extract_cases_list(raw_json: Any) -> list[dict[str, Any]]:
    if isinstance(raw_json, list):
        return raw_json
    if isinstance(raw_json, dict) and isinstance(raw_json.get("cases"), list):
        return raw_json["cases"]
    raise ValueError("Unsupported top-level JSON structure in sealed cases file.")


def _run_suite_once(
    suite_name: str,
    file_path: Path,
    expected_sha256: str,
    client: Any,
) -> dict[str, Any]:
    # Verify file hash before reading
    actual_before = sha256_file(file_path)
    if actual_before != expected_sha256:
        raise RuntimeError(
            f"INTEGRITY ERROR: {file_path} SHA256 changed before execution "
            f"(expected {expected_sha256}, got {actual_before})."
        )

    raw_bytes = file_path.read_bytes()
    parsed_json = json.loads(raw_bytes.decode("utf-8"))
    cases = _extract_cases_list(parsed_json)

    records: list[dict[str, Any]] = []
    table_rows: list[str] = [
        "case_id | HTTP | overall_safety_status | patient_releasable | requires_human_review | item_statuses | hazard_codes | latency_ms"
    ]

    for idx, entry in enumerate(cases, start=1):
        case_id = str(
            entry.get("case_id")
            or entry.get("id")
            or (entry.get("payload", {}) if isinstance(entry.get("payload"), dict) else {}).get("case_id")
            or f"{suite_name}-{idx:02d}"
        )
        request_payload = entry["payload"] if isinstance(entry, dict) and "payload" in entry else entry

        t0 = time.perf_counter()
        response = client.post("/api/v1/validate", json=request_payload)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        http_status = response.status_code
        try:
            resp_body = response.json()
        except Exception:
            resp_body = {"raw_text": response.text}

        if http_status == 200 and isinstance(resp_body, dict):
            overall_status = resp_body.get("overall_safety_status", "UNKNOWN")
            patient_releasable = resp_body.get("patient_releasable", False)
            requires_human_review = resp_body.get("requires_human_review", True)
            hazards_list = resp_body.get("hazard_codes_detected", [])
            hazards_str = ",".join(hazards_list) if hazards_list else "NONE"
            vp = resp_body.get("validated_payload", {})
            items = (
                vp.get("medications", [])
                + vp.get("follow_ups", [])
                + vp.get("warning_signs", [])
                + vp.get("tasks", [])
            )
            item_statuses = (
                "; ".join(
                    f"{it.get('id')}:{it.get('safety_status')}/{it.get('review_status')}/blocked={it.get('release_blocked')}"
                    for it in items
                )
                if items
                else "NONE"
            )
        else:
            overall_status = f"HTTP_{http_status}_REJECTED"
            patient_releasable = False
            requires_human_review = True
            hazards_str = "NONE"
            item_statuses = "N/A"

        row = (
            f"{case_id} | {http_status} | {overall_status} | {patient_releasable} | "
            f"{requires_human_review} | {item_statuses} | {hazards_str} | {latency_ms:.2f}"
        )
        table_rows.append(row)

        records.append(
            {
                "case_id": case_id,
                "http_status": http_status,
                "overall_safety_status": overall_status,
                "patient_releasable": patient_releasable,
                "requires_human_review": requires_human_review,
                "item_statuses": item_statuses,
                "hazard_codes": hazards_list if http_status == 200 and isinstance(resp_body, dict) else [],
                "latency_ms": latency_ms,
                "expected_metadata_in_case_file": {
                    k: v for k, v in entry.items() if k != "payload"
                }
                if isinstance(entry, dict) and "payload" in entry
                else {},
                "raw_input": entry,
                "raw_output": resp_body,
            }
        )

    # Verify file hash after execution to prove zero modification
    actual_after = sha256_file(file_path)
    if actual_after != expected_sha256:
        raise RuntimeError(
            f"INTEGRITY ERROR: {file_path} was modified during execution "
            f"(expected {expected_sha256}, got {actual_after})."
        )

    return {
        "suite_name": suite_name,
        "file_path": str(file_path),
        "file_sha256": expected_sha256,
        "run_count": 1,
        "table": "\n".join(table_rows),
        "cases": records,
    }


def main() -> None:
    if not V01_V14_PATH.is_file() or not V15_V20_PATH.is_file():
        print("WAITING FOR REVIEWER-SEALED V01–V14 AND V15–V20 TEST FILES.")
        return

    import fastapi
    import pydantic
    from fastapi.testclient import TestClient
    from backend.main import app

    git_status_before = get_git_status_short()
    if git_status_before:
        raise RuntimeError(
            f"ABORT: Git working tree is not clean before execution:\n{git_status_before}"
        )

    v01_v14_hash = sha256_file(V01_V14_PATH)
    v15_v20_hash = sha256_file(V15_V20_PATH)
    validator_hash = sha256_file(VALIDATOR_PATH)
    models_hash = sha256_file(MODELS_PATH)
    adversarial_hash = sha256_file(ADVERSARIAL_PATH)
    runner_hash = sha256_file(RUNNER_PATH)
    git_sha = get_git_sha()

    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = LOCK_DIR / f"execution_lock_{git_sha}_{v01_v14_hash[:12]}_{v15_v20_hash[:12]}.json"
    if lock_file.exists():
        raise RuntimeError(
            f"REPRODUCIBILITY VIOLATION: Sealed acceptance suites have ALREADY been executed once "
            f"(lock file exists at {lock_file}). Rerunning is strictly prohibited."
        )

    timestamp_iso = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
    # Write lock file immediately before running so no second run can ever occur
    lock_file.write_text(
        json.dumps(
            {
                "locked_at": timestamp_iso,
                "git_sha": git_sha,
                "v01_v14_sha256": v01_v14_hash,
                "v15_v20_sha256": v15_v20_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    client = TestClient(app, raise_server_exceptions=False)

    v01_v14_result = _run_suite_once("V01_V14", V01_V14_PATH, v01_v14_hash, client)
    v15_v20_result = _run_suite_once("V15_V20", V15_V20_PATH, v15_v20_hash, client)

    git_status_after = get_git_status_short()
    if git_status_after:
        raise RuntimeError(
            f"INTEGRITY ERROR: Git working tree became dirty after execution:\n{git_status_after}"
        )

    manifest = {
        "timestamp": timestamp_iso,
        "git_sha": git_sha,
        "git_working_tree_clean": git_status_after == "",
        "python_version": platform.python_version(),
        "fastapi_version": fastapi.__version__,
        "pydantic_version": pydantic.__version__,
        "sha256": {
            "v01_v14_cases.json": v01_v14_hash,
            "v15_v20_cases.json": v15_v20_hash,
            "validator.py": validator_hash,
            "models.py": models_hash,
            "test_adversarial_independent.py": adversarial_hash,
            "run_sealed_acceptance.py": runner_hash,
        },
        "v01_v14_run_count": 1,
        "v15_v20_run_count": 1,
        "v01_v14": v01_v14_result,
        "v15_v20": v15_v20_result,
    }

    output_file = LOCK_DIR / "sealed_acceptance_full_output.json"
    output_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== REPRODUCIBILITY MANIFEST ===")
    print(f"TIMESTAMP: {timestamp_iso}")
    print(f"GIT_SHA: {git_sha}")
    print(f"GIT_CLEAN: {git_status_after == ''}")
    print(f"PYTHON_VERSION: {platform.python_version()}")
    print(f"FASTAPI_VERSION: {fastapi.__version__}")
    print(f"PYDANTIC_VERSION: {pydantic.__version__}")
    for k, v in manifest["sha256"].items():
        print(f"SHA256({k}): {v}")
    print("\n=== V01-V14 RESULTS ===")
    print(v01_v14_result["table"])
    print("\n=== V15-V20 RESULTS ===")
    print(v15_v20_result["table"])
    print(f"\nCOMPLETE_RAW_IO_SAVED_TO: {output_file}")


if __name__ == "__main__":
    main()
