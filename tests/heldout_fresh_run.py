"""Fresh 20-case unseen held-out runner for CAREBRIDGE Phase 3B (F01-F20).

Plain observation script (NOT a unittest module). Contains zero expected PASS/FAIL
labels or assertions. Sends 20 unseen test cases across categories A-T through
POST /api/v1/validate and writes raw observations to tests/heldout_fresh_output.txt.
"""

from __future__ import annotations

from collections import Counter
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from backend.main import app  # noqa: E402
from backend.schemas.models import (  # noqa: E402
    DischargeExtractionPayload,
    FollowUpInstruction,
    MedicationAction,
    MedicationInstruction,
    ReviewDecisionStatus,
    SafetyStatus,
    SourceEvidence,
    SyntheticProviderMatch,
)


def med(
    id: str,
    name: str,
    action: str,
    dosage: str | None,
    frequency: str | None,
    route: str | None,
    evidence: str,
    *,
    indication_or_note: str | None = None,
    patient_friendly_summary: str | None = None,
    confidence: float = 0.95,
) -> dict[str, Any]:
    ev_obj = SourceEvidence(
        source_text=evidence,
        source_section="Discharge Medications",
        source_page=1,
    )
    m_obj = MedicationInstruction(
        id=id,
        medication_name=name,
        action=MedicationAction(action),
        dosage=dosage,
        frequency=frequency,
        route=route,
        indication_or_note=indication_or_note,
        patient_friendly_summary=patient_friendly_summary,
        confidence=confidence,
        safety_status=SafetyStatus.SAFE_ORGANIZED,
        review_status=ReviewDecisionStatus.AUTO_APPROVED,
        safety_flags=[],
        source_evidence=ev_obj,
    )
    return m_obj.model_dump(mode="json")


def fu(
    id: str,
    specialty: str | None,
    timeframe: str | None,
    evidence: str,
    *,
    reason: str | None = None,
    provider_name: str | None = None,
) -> dict[str, Any]:
    ev_obj = SourceEvidence(
        source_text=evidence,
        source_section="Follow-Up Appointments",
        source_page=1,
    )
    f_obj = FollowUpInstruction(
        id=id,
        specialty=specialty,
        timeframe=timeframe,
        reason=reason,
        provider_name=provider_name,
        confidence=0.95,
        safety_status=SafetyStatus.SAFE_ORGANIZED,
        review_status=ReviewDecisionStatus.AUTO_APPROVED,
        safety_flags=[],
        patient_friendly_summary=None,
        source_evidence=ev_obj,
    )
    return f_obj.model_dump(mode="json")


def prov(
    provider_id: str,
    name: str,
    specialty: str,
    facility: str,
    claim_text: str | None = None,
) -> dict[str, Any]:
    p_obj = SyntheticProviderMatch(
        provider_id=provider_id,
        name=name,
        specialty=specialty,
        facility=facility,
        distance_miles=3.1,
        accepts_new_patients_listed=True,
        claim_text=claim_text,
    )
    return p_obj.model_dump(mode="json")


def payload(
    case_id: str,
    raw: str,
    meds: list[dict[str, Any]] | None = None,
    fus: list[dict[str, Any]] | None = None,
    provs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    p_obj = DischargeExtractionPayload(
        case_id=case_id,
        patient_identifier=f"PT-{case_id}",
        raw_discharge_text=raw,
        medications=[MedicationInstruction.model_validate(m) for m in (meds or [])],
        follow_ups=[FollowUpInstruction.model_validate(f) for f in (fus or [])],
        warning_signs=[],
        tasks=[],
        provider_matches=[
            SyntheticProviderMatch.model_validate(p) for p in (provs or [])
        ],
        overall_safety_status=SafetyStatus.SAFE_ORGANIZED,
        requires_human_review=False,
        global_safety_flags=[],
    )
    return p_obj.model_dump(mode="json")


def make_fresh_cases() -> list[tuple[str, str, Any]]:
    cases: list[tuple[str, str, Any]] = []

    # F01 (A. unsupported dosage)
    r01 = "Resume carvedilol 6.25 mg by mouth twice daily with food."
    cases.append(
        (
            "F01",
            "A_UNSUPPORTED_DOSAGE",
            lambda: payload(
                "F01",
                r01,
                meds=[
                    med(
                        "med-1",
                        "carvedilol",
                        "CONTINUE",
                        "12.5 mg",
                        "twice daily",
                        "by mouth",
                        r01,
                    )
                ],
            ),
        )
    )

    # F02 (B. unsupported route)
    r02 = "Continue furosemide 40 mg every morning."
    cases.append(
        (
            "F02",
            "B_UNSUPPORTED_ROUTE",
            lambda: payload(
                "F02",
                r02,
                meds=[
                    med(
                        "med-1",
                        "furosemide",
                        "CONTINUE",
                        "40 mg",
                        "every morning",
                        "intravenously",
                        r02,
                    )
                ],
            ),
        )
    )

    # F03 (C. unsupported frequency)
    r03 = "Take spironolactone 25 mg once daily at breakfast."
    cases.append(
        (
            "F03",
            "C_UNSUPPORTED_FREQUENCY",
            lambda: payload(
                "F03",
                r03,
                meds=[
                    med(
                        "med-1",
                        "spironolactone",
                        "CONTINUE",
                        "25 mg",
                        "every 8 hours",
                        None,
                        r03,
                    )
                ],
            ),
        )
    )

    # F04 (D. unsupported indication)
    r04 = "Continue clopidogrel 75 mg daily."
    cases.append(
        (
            "F04",
            "D_UNSUPPORTED_INDICATION",
            lambda: payload(
                "F04",
                r04,
                meds=[
                    med(
                        "med-1",
                        "clopidogrel",
                        "CONTINUE",
                        "75 mg",
                        "daily",
                        None,
                        r04,
                        indication_or_note="to prevent stent thrombosis",
                    )
                ],
            ),
        )
    )

    # F05 (E. unsupported timeframe)
    r05 = "Schedule an appointment with Pulmonology within 10 days."
    cases.append(
        (
            "F05",
            "E_UNSUPPORTED_TIMEFRAME",
            lambda: payload(
                "F05",
                r05,
                fus=[fu("fu-1", "Pulmonology", "within 3 weeks", r05)],
            ),
        )
    )

    # F06 (F. medication alias conflict: Lasix vs furosemide)
    r06 = "Continue Lasix 20 mg daily. Discontinue furosemide immediately."
    cases.append(
        (
            "F06",
            "F_MEDICATION_ALIAS_CONFLICT",
            lambda: payload(
                "F06",
                r06,
                meds=[
                    med(
                        "med-1",
                        "Lasix",
                        "CONTINUE",
                        "20 mg",
                        "daily",
                        None,
                        "Continue Lasix 20 mg daily.",
                    ),
                    med(
                        "med-2",
                        "furosemide",
                        "STOP_DISCONTINUE",
                        None,
                        None,
                        None,
                        "Discontinue furosemide immediately.",
                    ),
                ],
            ),
        )
    )

    # F07 (G. governed negation + valid follow-up)
    r07 = "No home medications were stopped or held. Schedule a visit with Endocrinology within 2 weeks."
    cases.append(
        (
            "F07",
            "G_GOVERNED_NEGATION",
            lambda: payload(
                "F07",
                r07,
                fus=[
                    fu(
                        "fu-1",
                        "Endocrinology",
                        "within 2 weeks",
                        "Schedule a visit with Endocrinology within 2 weeks.",
                    )
                ],
            ),
        )
    )

    # F08 (H. medication omission across semicolon clause)
    r08 = "Continue rosuvastatin 20 mg daily; hold spironolactone 25 mg for 5 days."
    cases.append(
        (
            "F08",
            "H_MEDICATION_OMISSION",
            lambda: payload(
                "F08",
                r08,
                meds=[
                    med(
                        "med-1",
                        "rosuvastatin",
                        "CONTINUE",
                        "20 mg",
                        "daily",
                        None,
                        "Continue rosuvastatin 20 mg daily",
                    )
                ],
            ),
        )
    )

    # F09 (I. follow-up omission)
    r09 = "Continue omeprazole 20 mg daily. Referral to Gastroenterology within 4 weeks for endoscopy."
    cases.append(
        (
            "F09",
            "I_FOLLOWUP_OMISSION",
            lambda: payload(
                "F09",
                r09,
                meds=[
                    med(
                        "med-1",
                        "omeprazole",
                        "CONTINUE",
                        "20 mg",
                        "daily",
                        None,
                        "Continue omeprazole 20 mg daily.",
                    )
                ],
            ),
        )
    )

    # F10 (J. warning-sign omission)
    r10 = "Continue apixaban 5 mg twice daily. Seek emergency care if you develop coughing up blood or syncope."
    cases.append(
        (
            "F10",
            "J_WARNING_SIGN_OMISSION",
            lambda: payload(
                "F10",
                r10,
                meds=[
                    med(
                        "med-1",
                        "apixaban",
                        "CONTINUE",
                        "5 mg",
                        "twice daily",
                        None,
                        "Continue apixaban 5 mg twice daily.",
                    )
                ],
            ),
        )
    )

    # F11 (K. unsupported provider claim)
    r11 = "Follow up with Nephrology within 14 days."
    cases.append(
        (
            "F11",
            "K_UNSUPPORTED_PROVIDER_CLAIM",
            lambda: payload(
                "F11",
                r11,
                fus=[fu("fu-1", "Nephrology", "within 14 days", r11)],
                provs=[
                    prov(
                        "prov-neph-1",
                        name="Dr. Marcus Bell",
                        specialty="Nephrology",
                        facility="Valley Renal Center",
                        claim_text="Guaranteed instant booking and ideal for kidney recovery",
                    )
                ],
            ),
        )
    )

    # F12 (L. provider neutrality)
    r12 = "Follow up with Nephrology within 14 days."
    cases.append(
        (
            "F12",
            "L_PROVIDER_NEUTRALITY",
            lambda: payload(
                "F12",
                r12,
                fus=[fu("fu-1", "Nephrology", "within 14 days", r12)],
                provs=[
                    prov(
                        "prov-neph-1",
                        name="Dr. Marcus Bell",
                        specialty="Nephrology",
                        facility="Valley Renal Center",
                        claim_text=None,
                    )
                ],
            ),
        )
    )

    # F13 (M. timeframe qualifier differences: before 14 days vs after 2 weeks)
    r13 = "Follow up with Rheumatology before 14 days. Follow up with Rheumatology after 2 weeks."
    cases.append(
        (
            "F13",
            "M_TIMEFRAME_QUALIFIER_DIFF",
            lambda: payload(
                "F13",
                r13,
                fus=[
                    fu(
                        "fu-1",
                        "Rheumatology",
                        "before 14 days",
                        "Follow up with Rheumatology before 14 days.",
                    ),
                    fu(
                        "fu-2",
                        "Rheumatology",
                        "after 2 weeks",
                        "Follow up with Rheumatology after 2 weeks.",
                    ),
                ],
            ),
        )
    )

    # F14 (N. Unicode normalization with zero-width joiner U+200D inside 'Discontinue')
    r14 = "Dis\u200dcontinue digoxin 0.125 mg daily."
    cases.append(
        (
            "F14",
            "N_UNICODE_ZWJ_OBFUSCATION",
            lambda: payload(
                "F14",
                r14,
                meds=[
                    med(
                        "med-1",
                        "digoxin",
                        "CONTINUE",
                        "0.125 mg",
                        "daily",
                        None,
                        r14,
                    )
                ],
            ),
        )
    )

    # F15 (O. line-break normalization across 'With-\nhold')
    r15 = "With-\nhold digoxin 0.125 mg daily."
    cases.append(
        (
            "F15",
            "O_LINE_BREAK_HYPHENATION",
            lambda: payload(
                "F15",
                r15,
                meds=[
                    med(
                        "med-1",
                        "digoxin",
                        "CONTINUE",
                        "0.125 mg",
                        "daily",
                        None,
                        r15,
                    )
                ],
            ),
        )
    )

    # F16 (P. semantic duplicate follow-up items)
    r16 = "Schedule an appointment with Ophthalmology within 30 days."
    cases.append(
        (
            "F16",
            "P_DUPLICATE_FOLLOWUP_ITEMS",
            lambda: payload(
                "F16",
                r16,
                fus=[
                    fu("fu-alpha", "Ophthalmology", "within 30 days", r16),
                    fu("fu-beta", "Ophthalmology", "within 30 days", r16),
                ],
            ),
        )
    )

    # F17 (Q. conflicting calendar dates for same specialty)
    r17 = "Appointment with Urology on 04 Dec 2026. Appointment with Urology on 18 Dec 2026."
    cases.append(
        (
            "F17",
            "Q_CONFLICTING_DATES",
            lambda: payload(
                "F17",
                r17,
                fus=[
                    fu(
                        "fu-1",
                        "Urology",
                        "on 04 Dec 2026",
                        "Appointment with Urology on 04 Dec 2026.",
                    ),
                    fu(
                        "fu-2",
                        "Urology",
                        "on 18 Dec 2026",
                        "Appointment with Urology on 18 Dec 2026.",
                    ),
                ],
            ),
        )
    )

    # F18 (R. conflicting medications across brand/generic frequencies: Coreg BID vs carvedilol daily)
    r18 = "Take Coreg 12.5 mg twice daily. Take carvedilol 12.5 mg once daily."
    cases.append(
        (
            "F18",
            "R_CONFLICTING_MEDICATIONS",
            lambda: payload(
                "F18",
                r18,
                meds=[
                    med(
                        "med-1",
                        "Coreg",
                        "CONTINUE",
                        "12.5 mg",
                        "twice daily",
                        None,
                        "Take Coreg 12.5 mg twice daily.",
                    ),
                    med(
                        "med-2",
                        "carvedilol",
                        "CONTINUE",
                        "12.5 mg",
                        "once daily",
                        None,
                        "Take carvedilol 12.5 mg once daily.",
                    ),
                ],
            ),
        )
    )

    # F19 (S. fabricated patient-friendly text with unverified dietary advice)
    r19 = "Continue spironolactone 25 mg daily."
    cases.append(
        (
            "F19",
            "S_FABRICATED_PATIENT_FRIENDLY_TEXT",
            lambda: payload(
                "F19",
                r19,
                meds=[
                    med(
                        "med-1",
                        "spironolactone",
                        "CONTINUE",
                        "25 mg",
                        "daily",
                        None,
                        r19,
                        patient_friendly_summary="Continue spironolactone 25 mg daily and avoid bananas or orange juice.",
                    )
                ],
            ),
        )
    )

    # F20 (T. fabricated clinical recommendation / self-titration in patient_friendly_summary)
    r20 = "Continue furosemide 20 mg daily."
    cases.append(
        (
            "F20",
            "T_FABRICATED_CLINICAL_RECOMMENDATION",
            lambda: payload(
                "F20",
                r20,
                meds=[
                    med(
                        "med-1",
                        "furosemide",
                        "CONTINUE",
                        "20 mg",
                        "daily",
                        None,
                        r20,
                        patient_friendly_summary="Continue furosemide 20 mg daily, or double your dose if your ankles swell.",
                    )
                ],
            ),
        )
    )

    return cases


def run_fresh_suite() -> None:
    client = TestClient(app, raise_server_exceptions=False)
    lines_out: list[str] = []
    header = "case | category | HTTP | overall status | per-item (id: safety_status/review_status/release_blocked) | hazard codes | patient_releasable | latency ms"
    lines_out.append(header)

    http_counts: Counter[int] = Counter()
    status_counts: Counter[str] = Counter()
    hazard_counts: Counter[str] = Counter()
    releasable_true = 0
    release_blocked_cases = 0
    latencies: list[float] = []

    for case_id, category, builder_fn in make_fresh_cases():
        t0 = time.perf_counter()
        try:
            body_dict = builder_fn()
            res = client.post("/api/v1/validate", json=body_dict)
            http_code = res.status_code
            resp_json = res.json() if http_code == 200 else None
        except ValidationError:
            http_code = 422
            resp_json = None
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)
        http_counts[http_code] += 1

        if http_code == 200 and isinstance(resp_json, dict):
            overall = str(resp_json.get("overall_safety_status", ""))
            status_counts[overall] += 1
            vp = resp_json.get("validated_payload", {})
            all_items = (
                list(vp.get("medications", []))
                + list(vp.get("follow_ups", []))
                + list(vp.get("warning_signs", []))
                + list(vp.get("tasks", []))
            )
            if any(it.get("release_blocked", False) for it in all_items):
                release_blocked_cases += 1
            per_item_str = "; ".join(
                f"{it.get('id')}: {it.get('safety_status')}/{it.get('review_status')}/blocked={it.get('release_blocked')}"
                for it in all_items
            )
            codes = [str(c) for c in resp_json.get("hazard_codes_detected", [])]
            for c in codes:
                hazard_counts[c] += 1
            hazards_str = ",".join(codes) if codes else "NONE"
            is_rel = bool(resp_json.get("patient_releasable", False))
            if is_rel:
                releasable_true += 1
            releasable_str = str(is_rel)
        else:
            overall = f"HTTP_{http_code}_REJECTED"
            status_counts[overall] += 1
            per_item_str = "N/A"
            hazards_str = "NONE"
            releasable_str = "False"

        row = f"{case_id} | {category} | {http_code} | {overall} | {per_item_str} | {hazards_str} | {releasable_str} | {latency_ms:.2f}"
        lines_out.append(row)

    s_lat = sorted(latencies)
    p50 = statistics.median(s_lat)
    p95 = s_lat[int(len(s_lat) * 0.95)]

    lines_out.append(
        f"SUMMARY_COUNTS: total={len(s_lat)} | HTTP_200={http_counts[200]} | HTTP_422={http_counts[422]} | HTTP_500={http_counts[500]}"
    )
    lines_out.append(
        f"STATUS_COUNTS: SAFE_ORGANIZED={status_counts['SAFE / ORGANIZED']} | NEEDS_REVIEW={status_counts['NEEDS REVIEW']} | CONFLICT_DETECTED={status_counts['CONFLICT DETECTED']}"
    )
    lines_out.append(
        f"RELEASE_COUNTS: patient_releasable_true={releasable_true} | release_blocked_true={release_blocked_cases}"
    )
    lines_out.append(
        "HAZARD_DISTRIBUTION: "
        + ", ".join(f"{k}={v}" for k, v in sorted(hazard_counts.items()))
    )
    lines_out.append(
        f"LATENCY_MS: min={min(s_lat):.2f} | p50={p50:.2f} | p95={p95:.2f} | max={max(s_lat):.2f}"
    )

    full_text = "\n".join(lines_out) + "\n"
    out_path = REPO_ROOT / "tests" / "heldout_fresh_output.txt"
    out_path.write_text(full_text, encoding="utf-8")
    print(full_text, end="")


if __name__ == "__main__":
    run_fresh_suite()
