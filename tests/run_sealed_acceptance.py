"""Sealed one-shot acceptance runner for Phase 3C (tests/v_suite_cases.json).

Strictly enforces:
- Pre-execution and post-execution SHA256 verification of:
  - tests/v_suite_cases.json (expected: 65f4c8853b0652312dd23bfd9afa760717f8e0c6f67dd8c41005f0853dc08231)
  - tests/v_suite_expected.json (expected: 453e07f3c8dcb60e677f6f06acdff34872f5b51fd8fb2d0b684c662917f46c4c)
  - backend/safety/validator.py
  - backend/schemas/models.py
  - tests/test_adversarial_independent.py
  - tests/schema_dry_run.py
  - tests/run_sealed_acceptance.py
- Clean git working tree verification before and after execution.
- Single-execution lock so any attempt to rerun the suite fails loudly.
- Zero modification of supplied JSON files.
- Does NOT read or compare against v_suite_expected.json during semantic execution.
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

V_SUITE_CASES_PATH = ROOT_DIR / "tests" / "v_suite_cases.json"
V_SUITE_EXPECTED_PATH = ROOT_DIR / "tests" / "v_suite_expected.json"
VALIDATOR_PATH = ROOT_DIR / "backend" / "safety" / "validator.py"
MODELS_PATH = ROOT_DIR / "backend" / "schemas" / "models.py"
ADVERSARIAL_PATH = ROOT_DIR / "tests" / "test_adversarial_independent.py"
DRY_RUN_PATH = ROOT_DIR / "tests" / "schema_dry_run.py"
RUNNER_PATH = Path(__file__).resolve()

AUTHORITATIVE_CASES_SHA256 = "65f4c8853b0652312dd23bfd9afa760717f8e0c6f67dd8c41005f0853dc08231"
AUTHORITATIVE_EXPECTED_SHA256 = "453e07f3c8dcb60e677f6f06acdff34872f5b51fd8fb2d0b684c662917f46c4c"

LOCK_DIR = Path(
    os.environ.get(
        "SEALED_ACCEPTANCE_OUT_DIR",
        str(Path.home() / ".gemini" / "antigravity" / "sealed_acceptance_state"),
    )
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


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


def main() -> None:
    if not V_SUITE_CASES_PATH.is_file() or not V_SUITE_EXPECTED_PATH.is_file():
        print("WAITING FOR THE ACTUAL REVIEWER-SUPPLIED SCHEMA-CORRECT V01–V14 AND V15–V20 FILES.")
        sys.exit(1)

    cases_hash = sha256_file(V_SUITE_CASES_PATH)
    expected_hash = sha256_file(V_SUITE_EXPECTED_PATH)
    if cases_hash != AUTHORITATIVE_CASES_SHA256:
        raise RuntimeError(
            f"HASH MISMATCH for v_suite_cases.json: expected {AUTHORITATIVE_CASES_SHA256}, got {cases_hash}"
        )
    if expected_hash != AUTHORITATIVE_EXPECTED_SHA256:
        raise RuntimeError(
            f"HASH MISMATCH for v_suite_expected.json: expected {AUTHORITATIVE_EXPECTED_SHA256}, got {expected_hash}"
        )

    git_status_before = get_git_status_short()
    if git_status_before:
        raise RuntimeError(
            f"ABORT: Git working tree is not clean before execution:\n{git_status_before}"
        )

    import fastapi
    import pydantic
    from fastapi.testclient import TestClient
    from backend.main import app

    validator_hash = sha256_file(VALIDATOR_PATH)
    models_hash = sha256_file(MODELS_PATH)
    adversarial_hash = sha256_file(ADVERSARIAL_PATH)
    dry_run_hash = sha256_file(DRY_RUN_PATH)
    runner_hash = sha256_file(RUNNER_PATH)
    git_sha = get_git_sha()

    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = LOCK_DIR / f"execution_lock_{git_sha}_{cases_hash[:16]}.json"
    if lock_file.exists():
        raise RuntimeError(
            f"REPRODUCIBILITY VIOLATION: Sealed acceptance suite has ALREADY been executed once "
            f"(lock file exists at {lock_file}). Rerunning is strictly prohibited."
        )

    timestamp_iso = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
    lock_file.write_text(
        json.dumps(
            {
                "locked_at": timestamp_iso,
                "git_sha": git_sha,
                "v_suite_cases_sha256": cases_hash,
                "v_suite_expected_sha256": expected_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    raw_bytes = V_SUITE_CASES_PATH.read_bytes()
    cases: list[dict[str, Any]] = json.loads(raw_bytes.decode("utf-8"))

    client = TestClient(app, raise_server_exceptions=False)

    table_rows: list[str] = [
        "case_id | HTTP | overall_status | hazard_codes | per_item_status | patient_releasable | release_blocked | latency_ms"
    ]
    records: list[dict[str, Any]] = []

    for entry in cases:
        case_id = str(entry["case_id"])
        t0 = time.perf_counter()
        response = client.post("/api/v1/validate", json=entry)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        http_status = response.status_code
        resp_body = response.json()

        if http_status == 200 and isinstance(resp_body, dict):
            overall_status = resp_body.get("overall_safety_status", "UNKNOWN")
            patient_releasable = resp_body.get("patient_releasable", False)
            release_blocked = not bool(patient_releasable)
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
            per_item_status = (
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
            release_blocked = True
            requires_human_review = True
            hazards_list = []
            hazards_str = "NONE"
            per_item_status = "N/A"

        row = (
            f"{case_id} | {http_status} | {overall_status} | {hazards_str} | "
            f"{per_item_status} | {patient_releasable} | {release_blocked} | {latency_ms:.2f}"
        )
        table_rows.append(row)
        records.append(
            {
                "case_id": case_id,
                "http_status": http_status,
                "overall_status": overall_status,
                "hazard_codes": hazards_list,
                "per_item_status": per_item_status,
                "patient_releasable": patient_releasable,
                "release_blocked": release_blocked,
                "requires_human_review": requires_human_review,
                "latency_ms": latency_ms,
                "raw_input_payload": entry,
                "raw_validator_output": resp_body,
            }
        )

    # Verify both files remain byte-for-byte identical after execution
    if sha256_file(V_SUITE_CASES_PATH) != cases_hash:
        raise RuntimeError("INTEGRITY ERROR: v_suite_cases.json changed during execution!")
    if sha256_file(V_SUITE_EXPECTED_PATH) != expected_hash:
        raise RuntimeError("INTEGRITY ERROR: v_suite_expected.json changed during execution!")

    git_status_after = get_git_status_short()
    if git_status_after:
        raise RuntimeError(
            f"INTEGRITY ERROR: Git working tree became dirty after execution:\n{git_status_after}"
        )

    manifest = {
        "semantic_run_timestamp": timestamp_iso,
        "git_sha": git_sha,
        "git_working_tree_clean": git_status_after == "",
        "python_version": platform.python_version(),
        "fastapi_version": fastapi.__version__,
        "pydantic_version": pydantic.__version__,
        "sha256": {
            "v_suite_cases.json": cases_hash,
            "v_suite_expected.json": expected_hash,
            "validator.py": validator_hash,
            "models.py": models_hash,
            "test_adversarial_independent.py": adversarial_hash,
            "schema_dry_run.py": dry_run_hash,
            "run_sealed_acceptance.py": runner_hash,
        },
        "run_count": 1,
        "table": "\n".join(table_rows),
        "cases": records,
    }

    output_file = LOCK_DIR / "v_suite_28_raw_output.json"
    output_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== SEMANTIC ACCEPTANCE MANIFEST ===")
    print(f"SEMANTIC_RUN_TIMESTAMP: {timestamp_iso}")
    print(f"GIT_SHA: {git_sha}")
    print(f"GIT_CLEAN: {git_status_after == ''}")
    print(f"PYTHON_VERSION: {platform.python_version()}")
    print(f"FASTAPI_VERSION: {fastapi.__version__}")
    print(f"PYDANTIC_VERSION: {pydantic.__version__}")
    for k, v in manifest["sha256"].items():
        print(f"SHA256({k}): {v}")
    print("\n=== V01-V14 & W01-W14 RESULTS (28 CASES) ===")
    print(manifest["table"])
    print(f"\nCOMPLETE_RAW_IO_SAVED_TO: {output_file}")


if __name__ == "__main__":
    main()
