"""Schema-only dry-run validator for reviewer-supplied V01-V14 and V15-V20 files.

Validates each payload in tests/v01_v14_cases.json and tests/v15_v20_cases.json
against the exact production Pydantic schema (DischargeExtractionPayload)
WITHOUT calling backend/safety/validator.py and WITHOUT consuming the one-shot
sealed acceptance lock.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.schemas.models import DischargeExtractionPayload, RawTextTooLargeError

V01_V14_PATH = ROOT_DIR / "tests" / "v01_v14_cases.json"
V15_V20_PATH = ROOT_DIR / "tests" / "v15_v20_cases.json"


def _extract_cases(raw_json: Any) -> list[dict[str, Any]]:
    if isinstance(raw_json, list):
        return raw_json
    if isinstance(raw_json, dict) and isinstance(raw_json.get("cases"), list):
        return raw_json["cases"]
    raise ValueError("Unsupported top-level JSON structure.")


def check_file(label: str, path: Path) -> bool:
    if not path.is_file():
        print(f"{label}: MISSING ({path})")
        return False

    data = json.loads(path.read_text(encoding="utf-8"))
    cases = _extract_cases(data)
    all_valid = True
    print(f"=== {label} SCHEMA-ONLY DRY RUN ({len(cases)} cases) ===")
    print("case_id | schema_valid | http_equivalent | pydantic_errors")
    for idx, entry in enumerate(cases, start=1):
        payload = entry["payload"] if isinstance(entry, dict) and "payload" in entry else entry
        case_id = str(
            entry.get("case_id")
            or entry.get("id")
            or (payload.get("case_id") if isinstance(payload, dict) else None)
            or f"{label}-{idx:02d}"
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
    return all_valid


def main() -> None:
    if not V01_V14_PATH.is_file() or not V15_V20_PATH.is_file():
        print("WAITING FOR THE ACTUAL REVIEWER-SUPPLIED SCHEMA-CORRECT V01–V14 AND V15–V20 FILES.")
        sys.exit(1)

    ok_1 = check_file("V01-V14", V01_V14_PATH)
    ok_2 = check_file("V15-V20", V15_V20_PATH)
    if not (ok_1 and ok_2):
        sys.exit(2)


if __name__ == "__main__":
    main()
