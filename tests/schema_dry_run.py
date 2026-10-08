"""Schema-only dry-run validator for tests/v_suite_cases.json.

Validates every payload in tests/v_suite_cases.json against the exact production
Pydantic schema (DischargeExtractionPayload) WITHOUT calling
backend/safety/validator.py and WITHOUT consuming or altering the one-shot
acceptance state.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.schemas.models import DischargeExtractionPayload, RawTextTooLargeError

DEFAULT_SUITE_PATH = ROOT_DIR / "tests" / "v_suite_cases.json"
EXPECTED_PATH = ROOT_DIR / "tests" / "v_suite_expected.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def _extract_cases(raw_json: Any) -> list[dict[str, Any]]:
    if isinstance(raw_json, list):
        return raw_json
    if isinstance(raw_json, dict) and isinstance(raw_json.get("cases"), list):
        return raw_json["cases"]
    raise ValueError("Unsupported top-level JSON structure.")


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SUITE_PATH
    if not target.is_file():
        print(f"MISSING FILE: {target}")
        sys.exit(1)

    ts = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
    cases_hash = sha256_file(target)
    expected_hash = sha256_file(EXPECTED_PATH) if EXPECTED_PATH.is_file() else "MISSING"

    data = json.loads(target.read_text(encoding="utf-8"))
    cases = _extract_cases(data)
    all_valid = True

    print(f"SCHEMA_DRY_RUN_TIMESTAMP: {ts}")
    print(f"SHA256(v_suite_cases.json): {cases_hash}")
    print(f"SHA256(v_suite_expected.json): {expected_hash}")
    print(f"TOTAL_CASES: {len(cases)}")
    print("case_id | schema_valid | http_equivalent | pydantic_errors")

    for idx, entry in enumerate(cases, start=1):
        payload = entry["payload"] if isinstance(entry, dict) and "payload" in entry else entry
        case_id = str(
            entry.get("case_id")
            or entry.get("id")
            or (payload.get("case_id") if isinstance(payload, dict) else None)
            or f"CASE-{idx:02d}"
        )
        try:
            DischargeExtractionPayload.model_validate(payload)
            print(f"{case_id} | true | 200 | NONE")
        except RawTextTooLargeError as exc:
            all_valid = False
            print(f"{case_id} | false | 413 | {exc}")
        except ValidationError as exc:
            all_valid = False
            errs = json.dumps(exc.errors(include_url=False, include_input=False))
            print(f"{case_id} | false | 422 | {errs}")

    # Verify target file was not modified
    if sha256_file(target) != cases_hash:
        print("INTEGRITY ERROR: v_suite_cases.json changed during schema dry run.")
        sys.exit(3)

    if not all_valid:
        sys.exit(2)


if __name__ == "__main__":
    main()
