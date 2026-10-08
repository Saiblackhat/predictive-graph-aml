"""Phase 3 True Acceptance Gate Runner for CAREBRIDGE.

Executes the frozen independent adversarial test suite (`tests.test_adversarial_independent`),
verifies its cryptographic SHA256 integrity, parses `TOTAL CASES`, `MATCHES`, and `DIFFERS`,
computes statement coverage on `backend/safety/validator.py` across the full test suite,
and exits with code 0 ONLY when `DIFFERS == 0` (and `MATCHES == TOTAL CASES == 25`).
"""

from __future__ import annotations

import hashlib
import io
import re
import subprocess
import sys
import tempfile
import trace
import unittest
from pathlib import Path

EXPECTED_FROZEN_SHA256 = (
    "F96761ADAFB0AF54888BC35FB2BB160A44847590F0E535158D1FFE8D6877B891"
)


def compute_validator_coverage(repo_root: Path) -> tuple[int, int, float]:
    """Run full unittest suite under Python's trace module and compute validator.py coverage."""
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    tracer = trace.Trace(count=1, trace=0)

    def _run_suite() -> None:
        loader = unittest.TestLoader()
        suite = loader.discover(str(repo_root / "tests"))
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=0)
        runner.run(suite)

    tracer.runfunc(_run_suite)
    results = tracer.results()

    with tempfile.TemporaryDirectory() as tmpdir:
        results.write_results(show_missing=True, coverdir=tmpdir)
        cover_file = Path(tmpdir) / "backend.safety.validator.cover"
        if not cover_file.exists():
            return 0, 0, 0.0
        lines = cover_file.read_text(encoding="utf-8").splitlines()
        hit = sum(
            1
            for line in lines
            if ":" in line and line.split(":", 1)[0].strip().isdigit()
        )
        missed = sum(1 for line in lines if line.lstrip().startswith(">>>>>>"))
        total = hit + missed
        pct = (hit / total * 100.0) if total > 0 else 0.0
        return hit, total, pct


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    frozen_file = repo_root / "tests" / "test_adversarial_independent.py"

    if not frozen_file.exists():
        print("FAIL: Frozen adversarial test file is missing.", file=sys.stderr)
        return 1

    actual_sha256 = hashlib.sha256(frozen_file.read_bytes()).hexdigest().upper()
    if actual_sha256 != EXPECTED_FROZEN_SHA256:
        print(
            f"FAIL: Frozen adversarial test SHA256 mismatch!\n"
            f"  Expected: {EXPECTED_FROZEN_SHA256}\n"
            f"  Actual:   {actual_sha256}",
            file=sys.stderr,
        )
        return 1

    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_adversarial_independent", "-v"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    combined_output = f"{proc.stdout}\n{proc.stderr}"
    print(combined_output)

    if proc.returncode != 0:
        print(
            f"FAIL: unittest process exited with non-zero code {proc.returncode}.",
            file=sys.stderr,
        )
        return 1

    total_match = re.search(r"^TOTAL CASES:\s*(\d+)\s*$", combined_output, re.MULTILINE)
    matches_match = re.search(r"^MATCHES:\s*(\d+)\s*$", combined_output, re.MULTILINE)
    differs_match = re.search(r"^DIFFERS:\s*(\d+)\s*$", combined_output, re.MULTILINE)

    if not (total_match and matches_match and differs_match):
        print(
            "FAIL: Could not parse TOTAL CASES / MATCHES / DIFFERS from report.",
            file=sys.stderr,
        )
        return 1

    total_cases = int(total_match.group(1))
    matches = int(matches_match.group(1))
    differs = int(differs_match.group(1))

    hit, total_stmts, cov_pct = compute_validator_coverage(repo_root)
    print(
        f"VALIDATOR.PY COVERAGE: {hit}/{total_stmts} executable lines ({cov_pct:.1f}%)"
    )

    if differs > 0 or matches != total_cases or total_cases != 25:
        print(
            f"TRUE ACCEPTANCE GATE: FAIL (TOTAL={total_cases}, MATCHES={matches}, DIFFERS={differs})",
            file=sys.stderr,
        )
        return 1

    print(
        f"TRUE ACCEPTANCE GATE: PASS (SHA256={actual_sha256}, TOTAL={total_cases}, MATCHES={matches}, DIFFERS={differs})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
