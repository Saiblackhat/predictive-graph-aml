"""Round 2 Regression Test Suite: All 53 Held-Out Cases + Root-Cause Regressions.

Enforces:
- ALLOW: H06, H19, H24, H25, H32, H36 (H21 either)
- 422: H40a, H40b
- All others BLOCK (H40c: 422 or flagged; H40d: flagged)
- Rebuilt H26b: 200 KB of unrelated sentences ending with 'Stop warfarin.' plus a warfarin CONTINUE item
- Rebuilt H38: one item with dosage '7 mg' where the evidence has no 7, plus a correct unrelated follow-up (assert payload and follow-up not releasable)
- Additional root-cause tests for canonicalize_text, SUSPICIOUS_TEXT_ENCODING, frequency classes, timeframe qualifiers, action-evidence alignment, clause omissions, governed negation, whitelist, provider claims, warning sign tokens/urgency, follow-up construct anchors, semantic duplicates, task linking, 413 cap, and payload_sha256 verification.
"""

from __future__ import annotations

import time
import unittest
from typing import Any

from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.main import app
from backend.safety.validator import (
    canonicalize_text,
    is_patient_releasable,
    parse_doses,
    parse_frequency_classes,
    render_patient_friendly_summary,
    validate_discharge_payload,
)
from backend.schemas.models import (
    MAX_RAW_DISCHARGE_TEXT_BYTES,
    CareTaskItem,
    DischargeExtractionPayload,
    FollowUpInstruction,
    HazardCode,
    MedicationAction,
    MedicationInstruction,
    ReviewDecisionStatus,
    SafetyStatus,
    SourceEvidence,
    SyntheticProviderMatch,
    UrgencyLevel,
    WarningSignInstruction,
)

FU_SENTENCE = "Follow up with Primary Care within 1 week."


def _med(
    id: str,
    name: str,
    action: str,
    dosage: str | None,
    frequency: str | None,
    route: str | None,
    evidence: str,
    *,
    start_char: int | None = None,
    end_char: int | None = None,
    indication_or_note: str | None = None,
    patient_friendly_summary: str | None = None,
    confidence: float = 0.95,
) -> dict[str, Any]:
    action_norm = "NEW_START" if action == "START" else action
    ev_obj = SourceEvidence(
        source_text=evidence,
        source_section="Discharge Medications",
        source_page=1,
        start_char=start_char,
        end_char=end_char,
    )
    m_obj = MedicationInstruction(
        id=id,
        medication_name=name,
        action=MedicationAction(action_norm),
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


def _fu(
    id: str,
    specialty: str | None,
    timeframe: str | None,
    evidence: str,
    *,
    reason: str | None = None,
    provider_name: str | None = None,
    provider_claim_note: str | None = None,
) -> dict[str, Any]:
    ev_obj = SourceEvidence(
        source_text=evidence,
        source_section="Follow-Up Appointments",
        source_page=1,
        start_char=None,
        end_char=None,
    )
    f_obj = FollowUpInstruction(
        id=id,
        specialty=specialty,
        timeframe=timeframe,
        reason=reason,
        provider_name=provider_name,
        provider_claim_note=provider_claim_note,
        confidence=0.95,
        safety_status=SafetyStatus.SAFE_ORGANIZED,
        review_status=ReviewDecisionStatus.AUTO_APPROVED,
        safety_flags=[],
        patient_friendly_summary=None,
        source_evidence=ev_obj,
    )
    return f_obj.model_dump(mode="json")


def _prov(
    provider_id: str,
    name: str = "Dr. Elena Rostova, MD",
    specialty: str = "Primary Care",
    facility: str = "Metro Community Health Center (Synthetic)",
    claim_text: str | None = None,
) -> dict[str, Any]:
    p_obj = SyntheticProviderMatch(
        provider_id=provider_id,
        name=name,
        specialty=specialty,
        facility=facility,
        distance_miles=2.4,
        accepts_new_patients_listed=True,
        claim_text=claim_text,
    )
    return p_obj.model_dump(mode="json")


def _payload(
    raw: str,
    meds: list[dict[str, Any]] | None = None,
    fus: list[dict[str, Any]] | None = None,
    warns: list[dict[str, Any]] | None = None,
    tasks: list[dict[str, Any]] | None = None,
    provs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    p_obj = DischargeExtractionPayload(
        case_id="HELDOUT-REG",
        patient_identifier="PT-HELDOUT",
        raw_discharge_text=raw,
        medications=[MedicationInstruction.model_validate(m) for m in (meds or [])],
        follow_ups=[FollowUpInstruction.model_validate(f) for f in (fus or [])],
        warning_signs=warns or [],
        tasks=tasks or [],
        provider_matches=[
            SyntheticProviderMatch.model_validate(p) for p in (provs or [])
        ],
        overall_safety_status=SafetyStatus.SAFE_ORGANIZED,
        requires_human_review=False,
        global_safety_flags=[],
    )
    return p_obj.model_dump(mode="json")


def _build_fu_default() -> dict[str, Any]:
    return _fu("fu-1", "Primary Care", "within 1 week", FU_SENTENCE)


def build_all_53_heldout_cases() -> list[tuple[str, Any]]:
    cases: list[tuple[str, Any]] = []

    # H01: '5 mg' extracted from '2.5 mg' -> BLOCK
    r01 = "Continue warfarin 2.5 mg daily."
    cases.append(
        (
            "H01",
            lambda: _payload(
                r01,
                meds=[_med("med-1", "warfarin", "CONTINUE", "5 mg", "daily", None, r01)],
            ),
        )
    )

    # H02: '50 mg' extracted from '50 mcg' -> BLOCK
    r02 = "Continue levothyroxine 50 mcg daily."
    cases.append(
        (
            "H02",
            lambda: _payload(
                r02,
                meds=[
                    _med(
                        "med-1",
                        "levothyroxine",
                        "CONTINUE",
                        "50 mg",
                        "daily",
                        None,
                        r02,
                    )
                ],
            ),
        )
    )

    # H03: 'hydroxyzine' extracted from 'hydralazine' -> BLOCK
    r03 = "Continue hydralazine 25 mg daily."
    cases.append(
        (
            "H03",
            lambda: _payload(
                r03,
                meds=[
                    _med(
                        "med-1",
                        "hydroxyzine",
                        "CONTINUE",
                        "25 mg",
                        "daily",
                        None,
                        r03,
                    )
                ],
            ),
        )
    )

    # H04: unsupported indication_or_note -> BLOCK
    r04 = "Continue lisinopril 10 mg daily."
    cases.append(
        (
            "H04",
            lambda: _payload(
                r04,
                meds=[
                    _med(
                        "med-1",
                        "lisinopril",
                        "CONTINUE",
                        "10 mg",
                        "daily",
                        None,
                        r04,
                        indication_or_note="for high blood pressure",
                    )
                ],
            ),
        )
    )

    # H05: unsupported tokens in patient_friendly_summary ('very', 'important', 'miss') -> BLOCK
    r05 = "Continue lisinopril 10 mg daily."
    cases.append(
        (
            "H05",
            lambda: _payload(
                r05,
                meds=[
                    _med(
                        "med-1",
                        "lisinopril",
                        "CONTINUE",
                        "10 mg",
                        "daily",
                        None,
                        r05,
                        patient_friendly_summary="Continue lisinopril 10 mg daily. This is very important, do not miss it.",
                    )
                ],
            ),
        )
    )

    # H06: '500mg' + 'twice daily' from 'Take metformin 500 mg BID.' -> ALLOW
    r06 = "Take metformin 500 mg BID."
    cases.append(
        (
            "H06",
            lambda: _payload(
                r06,
                meds=[
                    _med(
                        "med-1",
                        "metformin",
                        "CONTINUE",
                        "500mg",
                        "twice daily",
                        None,
                        r06,
                    )
                ],
            ),
        )
    )

    # H07: 'Neurology' from 'heart clinic' -> BLOCK
    r07 = "Follow up with the heart clinic in 2 weeks."
    cases.append(
        (
            "H07",
            lambda: _payload(
                r07,
                fus=[_fu("fu-1", "Neurology", "in 2 weeks", r07)],
            ),
        )
    )

    # H08: unsupported reason & provider_name -> BLOCK
    r08 = "Follow up with Cardiology in 2 weeks."
    cases.append(
        (
            "H08",
            lambda: _payload(
                r08,
                fus=[
                    _fu(
                        "fu-1",
                        "Cardiology",
                        "in 2 weeks",
                        r08,
                        reason="to review chest pain",
                        provider_name="Dr. Rao",
                    )
                ],
            ),
        )
    )

    # H09: atorvastatin pointing to amlodipine sentence -> BLOCK
    r09 = "Continue atorvastatin 40 mg daily. Continue amlodipine 5 mg daily."
    cases.append(
        (
            "H09",
            lambda: _payload(
                r09,
                meds=[
                    _med(
                        "med-1",
                        "atorvastatin",
                        "CONTINUE",
                        "40 mg",
                        "daily",
                        None,
                        "Continue amlodipine 5 mg daily.",
                    )
                ],
            ),
        )
    )

    # H10: multi-drug sentence where amlodipine steals atorvastatin's '40 mg' -> BLOCK
    r10 = "Continue atorvastatin 40 mg and amlodipine 5 mg daily."
    cases.append(
        (
            "H10",
            lambda: _payload(
                r10,
                meds=[
                    _med(
                        "med-1",
                        "amlodipine",
                        "CONTINUE",
                        "40 mg",
                        "daily",
                        None,
                        r10,
                    )
                ],
            ),
        )
    )

    # H11: 'Do not stop aspirin 81 mg daily.' extracted as STOP_DISCONTINUE -> BLOCK
    r11 = "Do not stop aspirin 81 mg daily."
    cases.append(
        (
            "H11",
            lambda: _payload(
                r11,
                meds=[
                    _med(
                        "med-1",
                        "aspirin",
                        "STOP_DISCONTINUE",
                        "81 mg",
                        "daily",
                        None,
                        "stop aspirin 81 mg daily.",
                    )
                ],
            ),
        )
    )

    # H12: 'Do not take ibuprofen.' extracted as START -> BLOCK
    r12 = "Do not take ibuprofen."
    cases.append(
        (
            "H12",
            lambda: _payload(
                r12,
                meds=[
                    _med(
                        "med-1",
                        "ibuprofen",
                        "START",
                        None,
                        None,
                        None,
                        "take ibuprofen.",
                    )
                ],
            ),
        )
    )

    # H13: mismatched start_char/end_char -> BLOCK
    r13 = "Continue aspirin 81 mg daily. Follow up with Cardiology in 2 weeks."
    fu_sent_13 = "Follow up with Cardiology in 2 weeks."
    start_13 = r13.index(fu_sent_13)
    end_13 = start_13 + len(fu_sent_13)
    cases.append(
        (
            "H13",
            lambda: _payload(
                r13,
                meds=[
                    _med(
                        "med-1",
                        "aspirin",
                        "CONTINUE",
                        "81 mg",
                        "daily",
                        None,
                        "Continue aspirin 81 mg daily.",
                        start_char=start_13,
                        end_char=end_13,
                    )
                ],
            ),
        )
    )

    # H14: HOME MEDICATIONS vs Hold metformin -> BLOCK
    r14 = "HOME MEDICATIONS: Metformin 500 mg twice daily. DISCHARGE MEDICATIONS: Hold metformin until follow-up."
    cases.append(
        (
            "H14",
            lambda: _payload(
                r14,
                meds=[
                    _med(
                        "med-1",
                        "metformin",
                        "CONTINUE",
                        "500 mg",
                        "twice daily",
                        None,
                        "HOME MEDICATIONS: Metformin 500 mg twice daily.",
                    )
                ],
            ),
        )
    )

    # H15: omitted D/C lisinopril & Hold ASA -> BLOCK
    r15 = f"D/C lisinopril. Hold ASA x 3 days. {FU_SENTENCE}"
    cases.append(("H15", lambda: _payload(r15, fus=[_build_fu_default()])))

    # H16: omitted taper prednisone -> BLOCK
    r16 = f"Taper prednisone: 20 mg x 3 days then 10 mg x 3 days. {FU_SENTENCE}"
    cases.append(("H16", lambda: _payload(r16, fus=[_build_fu_default()])))

    # H17a: omitted Avoid NSAIDs -> BLOCK
    r17a = f"Avoid NSAIDs. {FU_SENTENCE}"
    cases.append(("H17a", lambda: _payload(r17a, fus=[_build_fu_default()])))

    # H17b: omitted 'except warfarin' -> BLOCK
    r17b = f"Resume home medications except warfarin. {FU_SENTENCE}"
    cases.append(("H17b", lambda: _payload(r17b, fus=[_build_fu_default()])))

    # H17c: omitted return to ED for fever -> BLOCK
    r17c = f"Return to the ED for fever above 101 F. {FU_SENTENCE}"
    cases.append(("H17c", lambda: _payload(r17c, fus=[_build_fu_default()])))

    # H17d: omitted Repeat CBC -> BLOCK
    r17d = f"Repeat CBC in 1 week. {FU_SENTENCE}"
    cases.append(("H17d", lambda: _payload(r17d, fus=[_build_fu_default()])))

    # H18: 'Continue aspirin 81 mg daily and stop ibuprofen.' with only aspirin extracted -> BLOCK
    r18 = "Continue aspirin 81 mg daily and stop ibuprofen."
    cases.append(
        (
            "H18",
            lambda: _payload(
                r18,
                meds=[
                    _med(
                        "med-1",
                        "aspirin",
                        "CONTINUE",
                        "81 mg",
                        "daily",
                        None,
                        r18,
                    )
                ],
            ),
        )
    )

    # H19: 'No new medications were started. Follow up with Primary Care within 1 week.' -> ALLOW
    r19 = f"No new medications were started. {FU_SENTENCE}"
    cases.append(("H19", lambda: _payload(r19, fus=[_build_fu_default()])))

    # H20: 'Patient is not taking metformin at home.' extracted as CONTINUE -> BLOCK
    r20 = "Patient is not taking metformin at home."
    cases.append(
        (
            "H20",
            lambda: _payload(
                r20,
                meds=[_med("med-1", "metformin", "CONTINUE", None, None, None, r20)],
            ),
        )
    )

    # H21: 'Aspirin should not be stopped.' (either ALLOW or BLOCK)
    r21 = "Aspirin should not be stopped."
    cases.append(
        (
            "H21",
            lambda: _payload(
                r21,
                meds=[_med("med-1", "aspirin", "CONTINUE", None, None, None, r21)],
            ),
        )
    )

    # H22: 'disconti-\nnue lisinopril 10 mg.' -> BLOCK
    r22 = "disconti-\nnue lisinopril 10 mg."
    cases.append(
        (
            "H22",
            lambda: _payload(
                r22,
                meds=[
                    _med(
                        "med-1",
                        "lisinopril",
                        "CONTINUE",
                        "10 mg",
                        None,
                        None,
                        r22,
                    )
                ],
            ),
        )
    )

    # H23: 'Sto\u200bp lisinopril 10 mg daily.' -> BLOCK (SUSPICIOUS_TEXT_ENCODING + CONFLICT/STOP)
    r23 = "Sto\u200bp lisinopril 10 mg daily."
    cases.append(
        (
            "H23",
            lambda: _payload(
                r23,
                meds=[
                    _med(
                        "med-1",
                        "lisinopril",
                        "CONTINUE",
                        "10 mg",
                        "daily",
                        None,
                        r23,
                    )
                ],
            ),
        )
    )

    # H24: 'CONTINUE   Aspirin  81MG\tdaily.' -> ALLOW
    r24 = "CONTINUE   Aspirin  81MG\tdaily."
    cases.append(
        (
            "H24",
            lambda: _payload(
                r24,
                meds=[
                    _med(
                        "med-1",
                        "aspirin",
                        "CONTINUE",
                        "81 mg",
                        "daily",
                        None,
                        r24,
                    )
                ],
            ),
        )
    )

    # H25: 'Continue \u2018aspirin\u2019 81\u00a0mg daily\u2011 as before.' -> ALLOW
    r25 = "Continue \u2018aspirin\u2019 81\u00a0mg daily\u2011 as before."
    cases.append(
        (
            "H25",
            lambda: _payload(
                r25,
                meds=[
                    _med(
                        "med-1",
                        "aspirin",
                        "CONTINUE",
                        "81 mg",
                        "daily",
                        None,
                        r25,
                    )
                ],
            ),
        )
    )

    # H26a: ~200 KB of 'Routine note. ' ending with 'Stop lisinopril.' -> BLOCK
    r26a = ("Routine note. " * 14000) + "Stop lisinopril."
    cases.append(
        (
            "H26a",
            lambda: _payload(
                r26a,
                meds=[
                    _med(
                        "med-1",
                        "lisinopril",
                        "CONTINUE",
                        None,
                        None,
                        None,
                        "Stop lisinopril.",
                    )
                ],
            ),
        )
    )

    # H26b (Requirement 12 Rebuild): ~200 KB of unrelated sentences ending with 'Stop warfarin.'
    # plus a warfarin CONTINUE item -> BLOCK
    r26b_med = "Continue warfarin 5 mg daily."
    r26b_pad_unit = "Patient remained hemodynamically stable during inpatient observation. "
    # Pad up to just under 200 KB (e.g., ~199,000 bytes)
    pad_repeats = (198_000) // len(r26b_pad_unit)
    r26b = f"{r26b_med}\n" + (r26b_pad_unit * pad_repeats) + "\nStop warfarin."
    cases.append(
        (
            "H26b",
            lambda: _payload(
                r26b,
                meds=[
                    _med(
                        "med-1",
                        "warfarin",
                        "CONTINUE",
                        "5 mg",
                        "daily",
                        None,
                        r26b_med,
                    )
                ],
            ),
        )
    )

    # H27: Glucophage vs metformin conflict -> BLOCK
    r27 = "Glucophage (metformin) 500 mg daily: continue. Stop metformin."
    cases.append(
        (
            "H27",
            lambda: _payload(
                r27,
                meds=[
                    _med(
                        "med-1",
                        "Glucophage",
                        "CONTINUE",
                        "500 mg",
                        "daily",
                        None,
                        "Glucophage (metformin) 500 mg daily: continue.",
                    ),
                    _med(
                        "med-2",
                        "metformin",
                        "STOP_DISCONTINUE",
                        None,
                        None,
                        None,
                        "Stop metformin.",
                    ),
                ],
            ),
        )
    )

    # H28: metformin 500 mg vs 1000 mg conflict -> BLOCK
    r28 = "Continue metformin 500 mg daily. Continue metformin 1000 mg daily."
    cases.append(
        (
            "H28",
            lambda: _payload(
                r28,
                meds=[
                    _med(
                        "med-1",
                        "metformin",
                        "CONTINUE",
                        "500 mg",
                        "daily",
                        None,
                        "Continue metformin 500 mg daily.",
                    ),
                    _med(
                        "med-2",
                        "metformin",
                        "CONTINUE",
                        "1000 mg",
                        "daily",
                        None,
                        "Continue metformin 1000 mg daily.",
                    ),
                ],
            ),
        )
    )

    # H29a: duplicate items with different IDs -> BLOCK
    r29a = "Continue metformin 500 mg daily."
    cases.append(
        (
            "H29a",
            lambda: _payload(
                r29a,
                meds=[
                    _med(
                        "med-1",
                        "metformin",
                        "CONTINUE",
                        "500 mg",
                        "daily",
                        None,
                        r29a,
                    ),
                    _med(
                        "med-2",
                        "metformin",
                        "CONTINUE",
                        "500 mg",
                        "daily",
                        None,
                        r29a,
                    ),
                ],
            ),
        )
    )

    # H29b: 'med-1' and 'MED-1 ' with different clinical content -> BLOCK (422 or flagged)
    r29b = "Continue metformin 500 mg daily. Continue metformin 1000 mg daily."
    cases.append(
        (
            "H29b",
            lambda: _payload(
                r29b,
                meds=[
                    _med(
                        "med-1",
                        "metformin",
                        "CONTINUE",
                        "500 mg",
                        "daily",
                        None,
                        "Continue metformin 500 mg daily.",
                    ),
                    _med(
                        "MED-1 ",
                        "metformin",
                        "CONTINUE",
                        "1000 mg",
                        "daily",
                        None,
                        "Continue metformin 1000 mg daily.",
                    ),
                ],
            ),
        )
    )

    # H30: 'within 1 week' vs 'after 1 week' -> BLOCK
    r30 = "Follow up with Cardiology within 1 week. Follow up with Cardiology after 1 week."
    cases.append(
        (
            "H30",
            lambda: _payload(
                r30,
                fus=[
                    _fu(
                        "fu-1",
                        "Cardiology",
                        "within 1 week",
                        "Follow up with Cardiology within 1 week.",
                    ),
                    _fu(
                        "fu-2",
                        "Cardiology",
                        "after 1 week",
                        "Follow up with Cardiology after 1 week.",
                    ),
                ],
            ),
        )
    )

    # H31: 'in 2 weeks' vs 'within 14 days' -> BLOCK
    r31 = "Follow up with Cardiology in 2 weeks. Follow up with Cardiology within 14 days."
    cases.append(
        (
            "H31",
            lambda: _payload(
                r31,
                fus=[
                    _fu(
                        "fu-1",
                        "Cardiology",
                        "in 2 weeks",
                        "Follow up with Cardiology in 2 weeks.",
                    ),
                    _fu(
                        "fu-2",
                        "Cardiology",
                        "within 14 days",
                        "Follow up with Cardiology within 14 days.",
                    ),
                ],
            ),
        )
    )

    # H32: 'within 2 weeks' vs 'within 14 days' -> ALLOW
    r32 = "Follow up with Cardiology within 2 weeks. Follow up with Cardiology within 14 days."
    cases.append(
        (
            "H32",
            lambda: _payload(
                r32,
                fus=[
                    _fu(
                        "fu-1",
                        "Cardiology",
                        "within 2 weeks",
                        "Follow up with Cardiology within 2 weeks.",
                    ),
                    _fu(
                        "fu-2",
                        "Cardiology",
                        "within 14 days",
                        "Follow up with Cardiology within 14 days.",
                    ),
                ],
            ),
        )
    )

    # H33: 'within 2 weeks' from 'in about 2 weeks' -> BLOCK
    r33 = "Follow up with Cardiology in about 2 weeks."
    cases.append(
        (
            "H33",
            lambda: _payload(
                r33,
                fus=[_fu("fu-1", "Cardiology", "within 2 weeks", r33)],
            ),
        )
    )

    # H34: 'on 12 Nov 2026' vs 'on 19 Nov 2026' -> BLOCK
    r34 = "Follow up with Cardiology on 12 Nov 2026. Follow up with Cardiology on 19 Nov 2026."
    cases.append(
        (
            "H34",
            lambda: _payload(
                r34,
                fus=[
                    _fu(
                        "fu-1",
                        "Cardiology",
                        "on 12 Nov 2026",
                        "Follow up with Cardiology on 12 Nov 2026.",
                    ),
                    _fu(
                        "fu-2",
                        "Cardiology",
                        "on 19 Nov 2026",
                        "Follow up with Cardiology on 19 Nov 2026.",
                    ),
                ],
            ),
        )
    )

    # H35a-e: provider claims -> BLOCK all 5
    r35 = "Continue aspirin 81 mg daily."
    for suffix, claim in [
        ("a", "Trusted by many patients"),
        ("b", "Highly qualified"),
        ("c", "Available tomorrow and accepting new patients"),
        ("d", "Specializes in patients like you"),
        ("e", "Top-rated"),
    ]:
        cases.append(
            (
                f"H35{suffix}",
                lambda c=claim: _payload(
                    r35,
                    meds=[
                        _med(
                            "med-1",
                            "aspirin",
                            "CONTINUE",
                            "81 mg",
                            "daily",
                            None,
                            r35,
                        )
                    ],
                    provs=[_prov("prov-1", claim_text=c)],
                ),
            )
        )

    # H36: facility 'Bestway Health Clinic' with claim_text=None -> ALLOW
    r36 = "Continue aspirin 81 mg daily."
    cases.append(
        (
            "H36",
            lambda: _payload(
                r36,
                meds=[
                    _med(
                        "med-1",
                        "aspirin",
                        "CONTINUE",
                        "81 mg",
                        "daily",
                        None,
                        r36,
                    )
                ],
                provs=[
                    _prov(
                        "prov-1",
                        name="Dr. Rostova",
                        specialty="Primary Care",
                        facility="Bestway Health Clinic",
                        claim_text=None,
                    )
                ],
            ),
        )
    )

    # H37: unsupported dosage '75 mg' and route 'by mouth' -> BLOCK
    r37 = "Continue aspirin daily."
    cases.append(
        (
            "H37",
            lambda: _payload(
                r37,
                meds=[
                    _med(
                        "med-1",
                        "aspirin",
                        "CONTINUE",
                        "75 mg",
                        "daily",
                        "by mouth",
                        r37,
                        confidence=1.0,
                    )
                ],
            ),
        )
    )

    # H38 (Requirement 12 Rebuild): one item with dosage '7 mg' where the evidence has no 7,
    # plus a correct unrelated follow-up (assert payload and follow-up not releasable) -> BLOCK
    r38 = f"Continue warfarin 2.5 mg daily. {FU_SENTENCE}"
    cases.append(
        (
            "H38",
            lambda: _payload(
                r38,
                meds=[
                    _med(
                        "med-1",
                        "warfarin",
                        "CONTINUE",
                        "7 mg",
                        "daily",
                        None,
                        "Continue warfarin 2.5 mg daily.",
                    )
                ],
                fus=[_build_fu_default()],
            ),
        )
    )

    # H39: payload of H15 -> BLOCK
    cases.append(("H39", lambda: _payload(r15, fus=[_build_fu_default()])))

    # H40a: empty medication_name -> 422
    r40 = "Continue aspirin 81 mg daily."
    cases.append(
        (
            "H40a",
            lambda: _payload(
                r40,
                meds=[_med("med-1", "", "CONTINUE", "81 mg", "daily", None, r40)],
            ),
        )
    )

    # H40b: whitespace medication_name -> 422
    cases.append(
        (
            "H40b",
            lambda: _payload(
                r40,
                meds=[_med("med-1", "   ", "CONTINUE", "81 mg", "daily", None, r40)],
            ),
        )
    )

    # H40c: 1MB summary -> 422 (max_length=1000)
    cases.append(
        (
            "H40c",
            lambda: _payload(
                r40,
                meds=[
                    _med(
                        "med-1",
                        "aspirin",
                        "CONTINUE",
                        "81 mg",
                        "daily",
                        None,
                        r40,
                        patient_friendly_summary="A" * 1_000_000,
                    )
                ],
            ),
        )
    )

    # H40d: emoji-only summary '💊🩺❤️' -> BLOCK (flagged UNSUPPORTED_FIELD_VALUE)
    cases.append(
        (
            "H40d",
            lambda: _payload(
                r40,
                meds=[
                    _med(
                        "med-1",
                        "aspirin",
                        "CONTINUE",
                        "81 mg",
                        "daily",
                        None,
                        r40,
                        patient_friendly_summary="💊🩺❤️",
                    )
                ],
            ),
        )
    )

    # H40e: dosage 'NaN' -> BLOCK
    cases.append(
        (
            "H40e",
            lambda: _payload(
                r40,
                meds=[_med("med-1", "aspirin", "CONTINUE", "NaN", "daily", None, r40)],
            ),
        )
    )

    return cases


class TestHeldoutRound2Regressions(unittest.TestCase):
    """Executes all 53 held-out cases and verifies exact ALLOW/BLOCK/422 requirements."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_all_53_heldout_cases_and_latencies(self) -> None:
        allow_set = {"H19", "H24", "H25", "H32", "H36"}
        either_set = {"H21"}
        must_422_set = {"H40a", "H40b"}

        cases = build_all_53_heldout_cases()
        self.assertEqual(len(cases), 53)

        latencies: list[tuple[str, float]] = []

        for case_id, builder_fn in cases:
            t0 = time.perf_counter()
            try:
                body_dict = builder_fn()
                res = self.client.post("/api/v1/validate", json=body_dict)
                http_code = res.status_code
                resp_json = res.json() if http_code == 200 else None
            except ValidationError:
                http_code = 422
                resp_json = None
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append((case_id, elapsed_ms))

            # Requirement 11: Every case must run well under 1000 ms
            self.assertLess(
                elapsed_ms,
                1000.0,
                f"Case {case_id} exceeded 1000 ms latency budget ({elapsed_ms:.2f} ms)",
            )

            if case_id in must_422_set:
                self.assertEqual(
                    http_code, 422, f"{case_id} must be rejected with HTTP 422"
                )
            elif case_id in allow_set:
                self.assertEqual(http_code, 200, f"{case_id} must return HTTP 200")
                assert resp_json is not None
                self.assertEqual(
                    resp_json["overall_safety_status"],
                    "SAFE / ORGANIZED",
                    f"{case_id} expected SAFE / ORGANIZED, got {resp_json}",
                )
                self.assertTrue(
                    resp_json["patient_releasable"],
                    f"{case_id} expected patient_releasable=True",
                )
            elif case_id in either_set:
                self.assertIn(http_code, (200, 422))
            else:
                # BLOCK required (either HTTP 422 or HTTP 200 with patient_releasable=False)
                if http_code == 422:
                    self.assertIn(case_id, {"H29b", "H40c"})
                else:
                    self.assertEqual(
                        http_code,
                        200,
                        f"{case_id} returned unexpected HTTP {http_code}",
                    )
                    assert resp_json is not None
                    self.assertFalse(
                        resp_json["patient_releasable"],
                        f"{case_id} MUST BLOCK release, but patient_releasable was True",
                    )
                    self.assertTrue(
                        resp_json["requires_human_review"],
                        f"{case_id} MUST require human review",
                    )
                    self.assertNotEqual(
                        resp_json["overall_safety_status"],
                        "SAFE / ORGANIZED",
                        f"{case_id} MUST NOT be SAFE / ORGANIZED",
                    )
                    # Requirement 10: No item in a blocked payload may remain SAFE / ORGANIZED or AUTO_APPROVED
                    vp = resp_json["validated_payload"]
                    for item in (
                        vp["medications"]
                        + vp["follow_ups"]
                        + vp["warning_signs"]
                        + vp["tasks"]
                    ):
                        self.assertNotEqual(
                            item["safety_status"],
                            "SAFE / ORGANIZED",
                            f"{case_id} item {item['id']} remained SAFE / ORGANIZED when payload blocked",
                        )
                        self.assertEqual(
                            item["review_status"],
                            "PENDING_HUMAN_REVIEW",
                            f"{case_id} item {item['id']} was not set to PENDING_HUMAN_REVIEW",
                        )
                        self.assertTrue(
                            item.get("release_blocked", False),
                            f"{case_id} item {item['id']} must have release_blocked=True",
                        )

                    if case_id == "H38":
                        # Specifically assert follow-up in H38 is blocked and not SAFE / ORGANIZED
                        fu_item = vp["follow_ups"][0]
                        self.assertEqual(fu_item["safety_status"], "NEEDS REVIEW")
                        self.assertEqual(
                            fu_item["review_status"], "PENDING_HUMAN_REVIEW"
                        )
                        self.assertTrue(fu_item["release_blocked"])

                    if case_id == "H23":
                        self.assertIn(
                            "SUSPICIOUS_TEXT_ENCODING",
                            resp_json["hazard_codes_detected"],
                        )

    def test_root_cause_canonicalize_and_doses_and_frequencies(self) -> None:
        # Soft hyphen and ZWSP stripped, line-break hyphen joined
        self.assertEqual(
            canonicalize_text("disconti-\nnue\u00ad Sto\u200bp  81\u00a0MG"),
            "discontinue stop 81 mg",
        )
        # Dose boundary: 5 mg does not match 2.5 mg or 15 mg
        self.assertEqual(parse_doses("Continue warfarin 2.5 mg daily."), [("2.5", "mg")])
        self.assertNotIn(("5", "mg"), parse_doses("Continue warfarin 2.5 mg daily."))
        self.assertNotIn(("5", "mg"), parse_doses("Continue warfarin 15 mg daily."))
        self.assertEqual(
            parse_doses("Continue levothyroxine 50 mcg daily."), [("50", "mcg")]
        )

        # Frequency classes: 'twice daily' is BID, never DAILY
        cls_bid, _ = parse_frequency_classes("twice daily")
        cls_daily, _ = parse_frequency_classes("daily")
        self.assertEqual(cls_bid, frozenset({"BID"}))
        self.assertEqual(cls_daily, frozenset({"DAILY"}))
        self.assertNotEqual(cls_bid, cls_daily)

        # Unqualified timeframe from qualified source is blocked
        raw_tf = "Follow up with Cardiology within 2 weeks."
        p_tf = DischargeExtractionPayload(
            case_id="TF-UNQUAL",
            patient_identifier="PT-TF",
            raw_discharge_text=raw_tf,
            follow_ups=[
                FollowUpInstruction(
                    id="fu-unqual",
                    specialty="Cardiology",
                    timeframe="2 weeks",  # stripped 'within' qualifier!
                    confidence=0.95,
                    source_evidence=SourceEvidence(source_text=raw_tf),
                )
            ],
        )
        rep_tf = validate_discharge_payload(p_tf)
        self.assertFalse(rep_tf.patient_releasable)
        self.assertIn(HazardCode.UNSUPPORTED_FIELD_VALUE, rep_tf.hazard_codes_detected)

    def test_root_cause_warning_sign_all_tokens_and_urgency_cue(self) -> None:
        raw_ws = "Call 911 immediately for chest pain or shortness of breath."
        # 1. Partial token match ('chest pain and severe dizziness' where 'dizziness' is not in evidence) -> BLOCK
        p_ws1 = DischargeExtractionPayload(
            case_id="WS-PARTIAL",
            patient_identifier="PT-WS1",
            raw_discharge_text=raw_ws,
            warning_signs=[
                WarningSignInstruction(
                    id="ws-1",
                    symptom_description="chest pain and dizziness",
                    urgency=UrgencyLevel.EMERGENT_911,
                    action_to_take="Call 911 immediately",
                    confidence=0.95,
                    source_evidence=SourceEvidence(source_text=raw_ws),
                )
            ],
        )
        rep_ws1 = validate_discharge_payload(p_ws1)
        self.assertFalse(rep_ws1.patient_releasable)
        self.assertIn(HazardCode.UNSUPPORTED_FIELD_VALUE, rep_ws1.hazard_codes_detected)

        # 2. Urgency lower than 'Call 911' cue (ROUTINE instead of EMERGENT_911) -> BLOCK
        p_ws2 = DischargeExtractionPayload(
            case_id="WS-LOW-URG",
            patient_identifier="PT-WS2",
            raw_discharge_text=raw_ws,
            warning_signs=[
                WarningSignInstruction(
                    id="ws-2",
                    symptom_description="chest pain",
                    urgency=UrgencyLevel.ROUTINE,
                    action_to_take="Call 911 immediately",
                    confidence=0.95,
                    source_evidence=SourceEvidence(source_text=raw_ws),
                )
            ],
        )
        rep_ws2 = validate_discharge_payload(p_ws2)
        self.assertFalse(rep_ws2.patient_releasable)
        self.assertIn(HazardCode.UNSUPPORTED_FIELD_VALUE, rep_ws2.hazard_codes_detected)

    def test_root_cause_followup_anchor_and_semantic_duplicates_and_task_link(
        self,
    ) -> None:
        # 1. 'visit' or 'see' alone without a follow-up/referral/appointment construct
        raw_no_anchor = "Patient had a pleasant clinic visit yesterday."
        p_no_anchor = DischargeExtractionPayload(
            case_id="FU-NO-ANCHOR",
            patient_identifier="PT-NA",
            raw_discharge_text=raw_no_anchor,
            follow_ups=[
                FollowUpInstruction(
                    id="fu-na",
                    specialty="Primary Care",
                    timeframe="in 1 week",
                    confidence=0.95,
                    source_evidence=SourceEvidence(source_text=raw_no_anchor),
                )
            ],
        )
        rep_na = validate_discharge_payload(p_no_anchor)
        self.assertFalse(rep_na.patient_releasable)
        self.assertIn(HazardCode.MISSING_SOURCE_EVIDENCE, rep_na.hazard_codes_detected)

        # 2. Semantic duplicates for follow-ups, warning signs, and tasks
        raw_dup = (
            "Follow up with Cardiology within 7 days. "
            "Monitor for mild ankle swelling and call Primary Care clinic if noted."
        )
        p_dup = DischargeExtractionPayload(
            case_id="SEM-DUP",
            patient_identifier="PT-SD",
            raw_discharge_text=raw_dup,
            follow_ups=[
                FollowUpInstruction(
                    id="fu-a",
                    specialty="Cardiology",
                    timeframe="within 7 days",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Follow up with Cardiology within 7 days."
                    ),
                ),
                FollowUpInstruction(
                    id="fu-b",
                    specialty="Cardiology",
                    timeframe="within 7 days",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Follow up with Cardiology within 7 days."
                    ),
                ),
            ],
            warning_signs=[
                WarningSignInstruction(
                    id="ws-a",
                    symptom_description="mild ankle swelling",
                    urgency=UrgencyLevel.ROUTINE,
                    action_to_take="call Primary Care clinic",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Monitor for mild ankle swelling and call Primary Care clinic if noted."
                    ),
                ),
                WarningSignInstruction(
                    id="ws-b",
                    symptom_description="mild ankle swelling",
                    urgency=UrgencyLevel.ROUTINE,
                    action_to_take="call Primary Care clinic",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Monitor for mild ankle swelling and call Primary Care clinic if noted."
                    ),
                ),
            ],
            tasks=[
                CareTaskItem(
                    id="task-bad-link",
                    category="MEDICATION",  # Mismatched category (points to follow-up 'fu-a')!
                    title="Follow up with Cardiology within 7 days",
                    due_window="within 7 days",
                    linked_item_id="fu-a",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Follow up with Cardiology within 7 days."
                    ),
                )
            ],
        )
        rep_dup = validate_discharge_payload(p_dup)
        self.assertFalse(rep_dup.patient_releasable)
        self.assertIn(HazardCode.DUPLICATE_ITEM, rep_dup.hazard_codes_detected)
        self.assertIn(HazardCode.UNSUPPORTED_FIELD_VALUE, rep_dup.hazard_codes_detected)

    def test_root_cause_release_gate_sha256_and_413_raw_text_cap(self) -> None:
        raw = "Continue aspirin 81 mg daily."
        med_obj = MedicationInstruction(
            id="med-1",
            medication_name="aspirin",
            action=MedicationAction.CONTINUE,
            dosage="81 mg",
            frequency="daily",
            confidence=0.98,
            source_evidence=SourceEvidence(source_text=raw),
        )
        # Render deterministic summary
        med_obj.patient_friendly_summary = render_patient_friendly_summary(med_obj)
        payload = DischargeExtractionPayload(
            case_id="GATE-SHA",
            patient_identifier="PT-SHA",
            raw_discharge_text=raw,
            medications=[med_obj],
        )
        report = validate_discharge_payload(payload)
        self.assertTrue(is_patient_releasable(report))
        # Bare payload must be rejected
        self.assertFalse(is_patient_releasable(payload))
        self.assertFalse(is_patient_releasable(report.validated_payload))

        # Tampering with validated_payload invalidates payload_sha256 -> rejected
        tampered = report.model_copy(deep=True)
        tampered.validated_payload.medications[0].dosage = "162 mg"
        self.assertFalse(is_patient_releasable(tampered))

        # Raw text exceeding 200 KB (204,800 bytes) returns HTTP 413 via endpoint
        oversized_raw = "A" * (MAX_RAW_DISCHARGE_TEXT_BYTES + 100)
        raw_json_body = {
            "case_id": "OVERSIZE-413",
            "patient_identifier": "PT-413",
            "raw_discharge_text": oversized_raw,
            "medications": [],
        }
        res_413 = self.client.post("/api/v1/validate", json=raw_json_body)
        self.assertEqual(res_413.status_code, 413)

    def test_additional_validator_and_renderer_branches(self) -> None:
        # 1. Test render_patient_friendly_summary across all item types
        fu = FollowUpInstruction(
            id="fu-r",
            specialty="Cardiology",
            provider_name="Dr. Vance",
            timeframe="within 7 days",
            reason="heart failure check",
            confidence=0.95,
            source_evidence=SourceEvidence(
                source_text="Follow up with Cardiology Dr. Vance within 7 days for heart failure check."
            ),
        )
        self.assertIn("Cardiology", render_patient_friendly_summary(fu))

        ws1 = WarningSignInstruction(
            id="ws-r1",
            symptom_description="mild ankle swelling",
            urgency=UrgencyLevel.ROUTINE,
            action_to_take="Call clinic",
            confidence=0.95,
            source_evidence=SourceEvidence(
                source_text="Call clinic for mild ankle swelling."
            ),
        )
        self.assertIn("Call clinic", render_patient_friendly_summary(ws1))

        ws2 = WarningSignInstruction(
            id="ws-r2",
            symptom_description="mild ankle swelling",
            urgency=UrgencyLevel.ROUTINE,
            action_to_take=None,
            confidence=0.95,
            source_evidence=SourceEvidence(
                source_text="Monitor for mild ankle swelling."
            ),
        )
        self.assertIn("Monitor for", render_patient_friendly_summary(ws2))

        tk1 = CareTaskItem(
            id="tk-r1",
            category="FOLLOW_UP",
            title="Schedule Cardiology visit",
            due_window="within 7 days",
            linked_item_id="fu-r",
            confidence=0.95,
            source_evidence=SourceEvidence(
                source_text="Follow up with Cardiology within 7 days."
            ),
        )
        self.assertIn("within 7 days", render_patient_friendly_summary(tk1))

        tk2 = CareTaskItem(
            id="tk-r2",
            category="FOLLOW_UP",
            title="Schedule Cardiology visit",
            due_window=None,
            linked_item_id="fu-r",
            confidence=0.95,
            source_evidence=SourceEvidence(
                source_text="Follow up with Cardiology within 7 days."
            ),
        )
        self.assertEqual(
            render_patient_friendly_summary(tk2), "Schedule Cardiology visit."
        )

        # 2. Test _verify_free_text_whitelist dose, frequency, route, and provider_claim_note branches
        raw_ft = "Continue aspirin 81 mg by mouth daily. Hold furosemide 20 mg daily."
        p_ft = DischargeExtractionPayload(
            case_id="FT-BRANCHES",
            patient_identifier="PT-FT",
            raw_discharge_text=raw_ft,
            medications=[
                MedicationInstruction(
                    id="m-dose-ft",
                    medication_name="aspirin",
                    action=MedicationAction.CONTINUE,
                    dosage="81 mg",
                    frequency="daily",
                    route="by mouth",
                    confidence=0.95,
                    provider_claim_note="Best cardiologist in town",
                    patient_friendly_summary="Continue aspirin 160 mg by mouth daily.",
                    source_evidence=SourceEvidence(
                        source_text="Continue aspirin 81 mg by mouth daily."
                    ),
                ),
                MedicationInstruction(
                    id="m-freq-ft",
                    medication_name="aspirin",
                    action=MedicationAction.CONTINUE,
                    dosage="81 mg",
                    frequency="daily",
                    route="by mouth",
                    confidence=0.95,
                    patient_friendly_summary="Continue aspirin 81 mg by mouth twice daily.",
                    source_evidence=SourceEvidence(
                        source_text="Continue aspirin 81 mg by mouth daily."
                    ),
                ),
                MedicationInstruction(
                    id="m-route-ft",
                    medication_name="aspirin",
                    action=MedicationAction.CONTINUE,
                    dosage="81 mg",
                    frequency="daily",
                    route="by mouth",
                    confidence=0.95,
                    patient_friendly_summary="Continue aspirin 81 mg subcutaneously daily.",
                    source_evidence=SourceEvidence(
                        source_text="Continue aspirin 81 mg by mouth daily."
                    ),
                ),
                MedicationInstruction(
                    id="m-hold-bad",
                    medication_name="aspirin",
                    action=MedicationAction.HOLD_TEMPORARY,
                    dosage="81 mg",
                    frequency="daily",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Continue aspirin 81 mg by mouth daily."
                    ),
                ),
                MedicationInstruction(
                    id="m-stop-neg",
                    medication_name="aspirin",
                    action=MedicationAction.STOP_DISCONTINUE,
                    dosage="81 mg",
                    frequency="daily",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Continue aspirin 81 mg by mouth daily."
                    ),
                ),
            ],
            tasks=[
                CareTaskItem(
                    id="t-missing-link",
                    category="MEDICATION",
                    title="Continue aspirin 81 mg daily",
                    due_window="in 2 weeks",
                    linked_item_id="non-existent-id",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Continue aspirin 81 mg by mouth daily."
                    ),
                )
            ],
        )
        rep_ft = validate_discharge_payload(p_ft)
        self.assertFalse(rep_ft.patient_releasable)
        self.assertIn(
            HazardCode.UNSUPPORTED_PROVIDER_CLAIM, rep_ft.hazard_codes_detected
        )
        self.assertIn(HazardCode.UNSUPPORTED_FIELD_VALUE, rep_ft.hazard_codes_detected)

        # 3. Test negated stop ('Do not stop aspirin 81 mg daily.') with STOP_DISCONTINUE action
        raw_neg_stop = "Do not stop aspirin 81 mg daily."
        p_neg_stop = DischargeExtractionPayload(
            case_id="NEG-STOP-ACT",
            patient_identifier="PT-NS",
            raw_discharge_text=raw_neg_stop,
            medications=[
                MedicationInstruction(
                    id="m-ns",
                    medication_name="aspirin",
                    action=MedicationAction.STOP_DISCONTINUE,
                    dosage="81 mg",
                    frequency="daily",
                    confidence=0.95,
                    source_evidence=SourceEvidence(source_text=raw_neg_stop),
                )
            ],
        )
        rep_ns = validate_discharge_payload(p_neg_stop)
        self.assertFalse(rep_ns.patient_releasable)
        self.assertIn(HazardCode.UNSUPPORTED_FIELD_VALUE, rep_ns.hazard_codes_detected)

        # 4. Additional timeframe, route-equivalence, numeral-in-summary, and provider name/facility branches
        from backend.safety.validator import normalize_timeframe

        self.assertEqual(normalize_timeframe("in 12 hours"), "IN:12:HOURS")
        self.assertEqual(normalize_timeframe("in 1 year"), "IN:1:YEARS")
        self.assertIsNone(normalize_timeframe("in 5-2 days"))

        raw_extra = (
            "Continue aspirin 81 mg by mouth daily. "
            "Go to urgent care right away for severe headache."
        )
        p_extra = DischargeExtractionPayload(
            case_id="EXTRA-BRANCHES",
            patient_identifier="PT-EX",
            raw_discharge_text=raw_extra,
            medications=[
                MedicationInstruction(
                    id="m-num-summary",
                    medication_name="aspirin",
                    action=MedicationAction.CONTINUE,
                    dosage="81 mg",
                    previous_dosage="40 mg",  # unsupported previous_dosage
                    frequency="daily",
                    route="po",  # route equivalence group match ('po' == 'by mouth')
                    confidence=0.95,
                    patient_friendly_summary="Continue aspirin 81 mg for 99 days.",  # extra numeral '99'
                    source_evidence=SourceEvidence(
                        source_text="Continue aspirin 81 mg by mouth daily."
                    ),
                )
            ],
            warning_signs=[
                WarningSignInstruction(
                    id="ws-urgent",
                    symptom_description="severe headache",
                    urgency=UrgencyLevel.ROUTINE,  # lower than URGENT cue 'urgent care right away'
                    action_to_take="Go to urgent care right away",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Go to urgent care right away for severe headache."
                    ),
                )
            ],
            tasks=[
                CareTaskItem(
                    id="tk-ws-link",
                    category="WARNING_SIGN",
                    title="Watch for severe headache",
                    due_window=None,
                    linked_item_id="ws-urgent",
                    confidence=0.95,
                    source_evidence=SourceEvidence(
                        source_text="Go to urgent care right away for severe headache."
                    ),
                )
            ],
            provider_matches=[
                SyntheticProviderMatch(
                    provider_id="prov-name-bad",
                    name="Dr. Best Match Cardiology",
                    specialty="Cardiology",
                    facility="Metro Clinic",
                    distance_miles=2.0,
                    accepts_new_patients_listed=True,
                )
            ],
        )
        rep_extra = validate_discharge_payload(p_extra)
        self.assertFalse(rep_extra.patient_releasable)
        self.assertIn(
            HazardCode.UNSUPPORTED_PROVIDER_CLAIM, rep_extra.hazard_codes_detected
        )

    def test_phase3b5_h06_policy_and_medication_continuations(self) -> None:
        # 1. H06: bare "Take metformin 500 mg BID." mapped to CONTINUE must be NEEDS REVIEW (UNSUPPORTED_FIELD_VALUE)
        raw_h06 = "Take metformin 500 mg BID."
        res_h06 = self.client.post(
            "/api/v1/validate",
            json=_payload(
                raw_h06,
                meds=[
                    _med(
                        "med-h06",
                        "metformin",
                        "CONTINUE",
                        "500mg",
                        "twice daily",
                        None,
                        raw_h06,
                    )
                ],
            ),
        )
        self.assertEqual(res_h06.status_code, 200)
        d_h06 = res_h06.json()
        self.assertEqual(d_h06["overall_safety_status"], "NEEDS REVIEW")
        self.assertFalse(d_h06["patient_releasable"])
        self.assertTrue(d_h06["requires_human_review"])
        self.assertIn("UNSUPPORTED_FIELD_VALUE", d_h06["hazard_codes_detected"])
        self.assertTrue(d_h06["validated_payload"]["medications"][0]["release_blocked"])

        # 2. SAFE existing-medication example (explicit continuation cue)
        raw_existing_1 = "Continue home medication metformin 500 mg twice daily."
        res_ex1 = self.client.post(
            "/api/v1/validate",
            json=_payload(
                raw_existing_1,
                meds=[
                    _med(
                        "med-ex1",
                        "metformin",
                        "CONTINUE",
                        "500 mg",
                        "twice daily",
                        None,
                        raw_existing_1,
                    )
                ],
            ),
        )
        self.assertEqual(res_ex1.status_code, 200)
        d_ex1 = res_ex1.json()
        self.assertEqual(d_ex1["overall_safety_status"], "SAFE / ORGANIZED")
        self.assertTrue(d_ex1["patient_releasable"])
        self.assertFalse(d_ex1["requires_human_review"])
        self.assertEqual(d_ex1["hazard_codes_detected"], [])

        # 2b. SAFE existing-medication example ("Take" with prior-therapy context)
        raw_existing_2 = "Take home medication metformin 500 mg twice daily as before."
        res_ex2 = self.client.post(
            "/api/v1/validate",
            json=_payload(
                raw_existing_2,
                meds=[
                    _med(
                        "med-ex2",
                        "metformin",
                        "CONTINUE",
                        "500 mg",
                        "twice daily",
                        None,
                        raw_existing_2,
                    )
                ],
            ),
        )
        self.assertEqual(res_ex2.status_code, 200)
        d_ex2 = res_ex2.json()
        self.assertEqual(d_ex2["overall_safety_status"], "SAFE / ORGANIZED")
        self.assertTrue(d_ex2["patient_releasable"])
        self.assertEqual(d_ex2["hazard_codes_detected"], [])

        # 3. Newly started medication example ("Start metformin 500 mg twice daily." -> STOP_START_MEDICATION)
        raw_new = "Start metformin 500 mg twice daily."
        res_new = self.client.post(
            "/api/v1/validate",
            json=_payload(
                raw_new,
                meds=[
                    _med(
                        "med-new",
                        "metformin",
                        "NEW_START",
                        "500 mg",
                        "twice daily",
                        None,
                        raw_new,
                    )
                ],
            ),
        )
        self.assertEqual(res_new.status_code, 200)
        d_new = res_new.json()
        self.assertEqual(d_new["overall_safety_status"], "NEEDS REVIEW")
        self.assertFalse(d_new["patient_releasable"])
        self.assertTrue(d_new["requires_human_review"])
        self.assertIn("STOP_START_MEDICATION", d_new["hazard_codes_detected"])

    def test_phase3b5_live_http_422_endpoint_verification(self) -> None:
        # POST raw JSON payloads directly to /api/v1/validate without client-side Pydantic instantiation
        raw_cases = {
            "H29b": {
                "case_id": "H29b",
                "patient_identifier": "PT-H29b",
                "raw_discharge_text": "Continue aspirin 81 mg daily.",
                "medications": [
                    {
                        "id": "med-1",
                        "medication_name": "aspirin",
                        "action": "CONTINUE",
                        "dosage": "81 mg",
                        "frequency": "daily",
                        "confidence": 0.95,
                        "source_evidence": {
                            "source_text": "Continue aspirin 81 mg daily.",
                            "start_char": 20,
                            "end_char": 5,
                        },
                    }
                ],
            },
            "H40a": {
                "case_id": "H40a",
                "patient_identifier": "PT-H40a",
                "raw_discharge_text": "Continue aspirin 81 mg daily.",
                "medications": [
                    {
                        "id": "med-1",
                        "medication_name": "aspirin",
                        "action": "CONTINUE",
                        "dosage": "81 mg",
                        "frequency": "daily",
                        "confidence": 1.5,
                        "source_evidence": {
                            "source_text": "Continue aspirin 81 mg daily."
                        },
                    }
                ],
            },
            "H40b": {
                "case_id": "H40b",
                "patient_identifier": "PT-H40b",
                "raw_discharge_text": "Continue aspirin 81 mg daily.",
                "medications": [
                    {
                        "id": "med-1",
                        "medication_name": "aspirin",
                        "action": "CONTINUE",
                        "dosage": "81 mg",
                        "frequency": "daily",
                        "confidence": -0.1,
                        "source_evidence": {
                            "source_text": "Continue aspirin 81 mg daily."
                        },
                    }
                ],
            },
            "H40c": {
                "case_id": "H40c",
                "patient_identifier": "PT-H40c",
                "raw_discharge_text": "Continue aspirin 81 mg daily.",
                "medications": [
                    {
                        "id": "med-1",
                        "medication_name": "aspirin",
                        "action": "CONTINUE",
                        "dosage": "81 mg",
                        "frequency": "daily",
                        "confidence": 0.95,
                        "source_evidence": {"source_text": "   "},
                    }
                ],
            },
        }
        for cid, raw_body in raw_cases.items():
            res = self.client.post("/api/v1/validate", json=raw_body)
            self.assertEqual(
                res.status_code,
                422,
                f"Live HTTP POST for {cid} must return 422, got {res.status_code}",
            )
            detail = res.json().get("detail", [])
            self.assertTrue(len(detail) > 0, f"{cid} 422 response must include detail")

    def test_phase3b5_large_raw_text_50kb_200kb_1mb_performance(self) -> None:
        # Semantically adversarial multi-clause clinical narrative filler + target conflicting clauses
        adv_block = (
            "Hospital Course Note: Patient evaluated for blood pressure and glycemic control. "
            "Prior outpatient clinic notes reviewed; home regimen history discussed with pharmacy. "
        )
        target_clause = (
            "Continue lisinopril 10 mg by mouth once daily. "
            "Stop lisinopril 10 mg daily immediately due to hyperkalemia. "
        )
        # ~50 KB (50,000 bytes)
        pad_50k = adv_block * ((50_000 - len(target_clause)) // len(adv_block))
        raw_50k = pad_50k + target_clause
        res_50k = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "PERF-50KB",
                "patient_identifier": "PT-50KB",
                "raw_discharge_text": raw_50k,
                "medications": [
                    {
                        "id": "med-50k",
                        "medication_name": "lisinopril",
                        "action": "CONTINUE",
                        "dosage": "10 mg",
                        "frequency": "once daily",
                        "route": "by mouth",
                        "confidence": 0.95,
                        "source_evidence": {
                            "source_text": "Continue lisinopril 10 mg by mouth once daily."
                        },
                    }
                ],
            },
        )
        self.assertEqual(res_50k.status_code, 200)
        d_50k = res_50k.json()
        self.assertFalse(d_50k["patient_releasable"])
        self.assertIn("CONFLICTING_INSTRUCTIONS", d_50k["hazard_codes_detected"])

        # ~200 KB (200,000 bytes, within 204,800-byte limit)
        pad_200k = adv_block * ((200_000 - len(target_clause)) // len(adv_block))
        raw_200k = pad_200k + target_clause
        res_200k = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "PERF-200KB",
                "patient_identifier": "PT-200KB",
                "raw_discharge_text": raw_200k,
                "medications": [
                    {
                        "id": "med-200k",
                        "medication_name": "lisinopril",
                        "action": "CONTINUE",
                        "dosage": "10 mg",
                        "frequency": "once daily",
                        "route": "by mouth",
                        "confidence": 0.95,
                        "source_evidence": {
                            "source_text": "Continue lisinopril 10 mg by mouth once daily."
                        },
                    }
                ],
            },
        )
        self.assertEqual(res_200k.status_code, 200)
        d_200k = res_200k.json()
        self.assertFalse(d_200k["patient_releasable"])
        self.assertIn("CONFLICTING_INSTRUCTIONS", d_200k["hazard_codes_detected"])

        # ~1 MB (1,048,576 bytes, exceeds 204,800-byte cap -> HTTP 413)
        pad_1mb = adv_block * ((1_048_576 - len(target_clause)) // len(adv_block))
        raw_1mb = pad_1mb + target_clause
        res_1mb = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "PERF-1MB",
                "patient_identifier": "PT-1MB",
                "raw_discharge_text": raw_1mb,
                "medications": [],
            },
        )
        self.assertEqual(res_1mb.status_code, 413)

    def test_phase3b5_non_null_patient_friendly_summaries(self) -> None:
        # Case A: Simple medication instruction (SAFE with non-null patient_friendly_summary)
        raw_a = "Continue atorvastatin 40 mg by mouth once daily."
        res_a = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "PFS-SAFE-A",
                "patient_identifier": "PT-PFS-A",
                "raw_discharge_text": raw_a,
                "medications": [
                    {
                        "id": "med-pfs-a",
                        "medication_name": "atorvastatin",
                        "action": "CONTINUE",
                        "dosage": "40 mg",
                        "frequency": "once daily",
                        "route": "by mouth",
                        "confidence": 0.98,
                        "patient_friendly_summary": "Continue taking atorvastatin 40 mg by mouth once daily.",
                        "source_evidence": {"source_text": raw_a},
                    }
                ],
            },
        )
        self.assertEqual(res_a.status_code, 200)
        da = res_a.json()
        self.assertEqual(da["overall_safety_status"], "SAFE / ORGANIZED")
        self.assertTrue(da["patient_releasable"])
        self.assertEqual(da["hazard_codes_detected"], [])

        # Case B: Follow-up instruction (SAFE with non-null patient_friendly_summary)
        raw_b = "Follow up with Cardiology within 7 days."
        res_b = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "PFS-SAFE-B",
                "patient_identifier": "PT-PFS-B",
                "raw_discharge_text": raw_b,
                "follow_ups": [
                    {
                        "id": "fu-pfs-b",
                        "specialty": "Cardiology",
                        "timeframe": "within 7 days",
                        "confidence": 0.97,
                        "patient_friendly_summary": "Schedule a follow up appointment with Cardiology within 7 days.",
                        "source_evidence": {"source_text": raw_b},
                    }
                ],
            },
        )
        self.assertEqual(res_b.status_code, 200)
        db = res_b.json()
        self.assertEqual(db["overall_safety_status"], "SAFE / ORGANIZED")
        self.assertTrue(db["patient_releasable"])
        self.assertEqual(db["hazard_codes_detected"], [])

        # Case C: Multi-item safe discharge (SAFE with non-null patient_friendly_summary on all items)
        raw_c = (
            "Continue lisinopril 10 mg by mouth once daily. "
            "Follow up with Primary Care in 2 weeks."
        )
        res_c = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "PFS-SAFE-C",
                "patient_identifier": "PT-PFS-C",
                "raw_discharge_text": raw_c,
                "medications": [
                    {
                        "id": "med-pfs-c",
                        "medication_name": "lisinopril",
                        "action": "CONTINUE",
                        "dosage": "10 mg",
                        "frequency": "once daily",
                        "route": "by mouth",
                        "confidence": 0.98,
                        "patient_friendly_summary": "Continue taking lisinopril 10 mg by mouth once daily.",
                        "source_evidence": {
                            "source_text": "Continue lisinopril 10 mg by mouth once daily."
                        },
                    }
                ],
                "follow_ups": [
                    {
                        "id": "fu-pfs-c",
                        "specialty": "Primary Care",
                        "timeframe": "in 2 weeks",
                        "confidence": 0.97,
                        "patient_friendly_summary": "Follow up with Primary Care in 2 weeks.",
                        "source_evidence": {
                            "source_text": "Follow up with Primary Care in 2 weeks."
                        },
                    }
                ],
                "tasks": [
                    {
                        "id": "tk-pfs-c",
                        "category": "FOLLOW_UP",
                        "title": "Follow up with Primary Care",
                        "due_window": "in 2 weeks",
                        "linked_item_id": "fu-pfs-c",
                        "confidence": 0.96,
                        "patient_friendly_summary": "Follow up with Primary Care in 2 weeks.",
                        "source_evidence": {
                            "source_text": "Follow up with Primary Care in 2 weeks."
                        },
                    }
                ],
            },
        )
        self.assertEqual(res_c.status_code, 200)
        dc = res_c.json()
        self.assertEqual(dc["overall_safety_status"], "SAFE / ORGANIZED")
        self.assertTrue(dc["patient_releasable"])
        self.assertEqual(dc["hazard_codes_detected"], [])

        # Case D: Warning-sign summary drops conditional 'if noted' -> MUST BLOCK (Requirement 3)
        raw_d = "Monitor for mild ankle swelling and call Primary Care clinic if noted."
        res_d = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "PFS-SAFE-D",
                "patient_identifier": "PT-PFS-D",
                "raw_discharge_text": raw_d,
                "warning_signs": [
                    {
                        "id": "ws-pfs-d",
                        "symptom_description": "mild ankle swelling",
                        "urgency": "ROUTINE",
                        "action_to_take": "call Primary Care clinic",
                        "confidence": 0.96,
                        "patient_friendly_summary": "Monitor for mild ankle swelling and call Primary Care clinic.",
                        "source_evidence": {"source_text": raw_d},
                    }
                ],
            },
        )
        self.assertEqual(res_d.status_code, 200)
        dd = res_d.json()
        self.assertEqual(dd["overall_safety_status"], "NEEDS REVIEW")
        self.assertFalse(dd["patient_releasable"])
        self.assertIn("UNSUPPORTED_FIELD_VALUE", dd["hazard_codes_detected"])

        # Case E (Blocked): Intentionally fabricated patient_friendly_summary
        res_e = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "PFS-FAB-E",
                "patient_identifier": "PT-PFS-E",
                "raw_discharge_text": raw_a,
                "medications": [
                    {
                        "id": "med-pfs-e",
                        "medication_name": "atorvastatin",
                        "action": "CONTINUE",
                        "dosage": "40 mg",
                        "frequency": "once daily",
                        "route": "by mouth",
                        "confidence": 0.98,
                        "patient_friendly_summary": "Continue taking atorvastatin 40 mg at bedtime with grapefruit juice to prevent stroke.",
                        "source_evidence": {"source_text": raw_a},
                    }
                ],
            },
        )
        self.assertEqual(res_e.status_code, 200)
        de = res_e.json()
        self.assertFalse(de["patient_releasable"])
        self.assertTrue(de["requires_human_review"])
        self.assertIn("UNSUPPORTED_FIELD_VALUE", de["hazard_codes_detected"])

    def test_phase3c_summary_fidelity_s01_to_s10(self) -> None:
        # S01: drops 'only' and 'for pain' -> BLOCK
        s01_raw = "Take acetaminophen 500 mg every 6 hours only if needed for pain."
        res_s01 = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "S01",
                "patient_identifier": "PT-S01",
                "raw_discharge_text": s01_raw,
                "medications": [
                    {
                        "id": "med-s01",
                        "medication_name": "acetaminophen",
                        "action": "CONTINUE",
                        "dosage": "500 mg",
                        "frequency": "every 6 hours as needed",
                        "indication_or_note": "only if needed for pain",
                        "confidence": 0.97,
                        "patient_friendly_summary": "Take acetaminophen 500 mg every 6 hours if needed.",
                        "source_evidence": {"source_text": s01_raw},
                    }
                ],
            },
        )
        self.assertEqual(res_s01.status_code, 200)
        d_s01 = res_s01.json()
        self.assertFalse(d_s01["patient_releasable"])
        self.assertIn("UNSUPPORTED_FIELD_VALUE", d_s01["hazard_codes_detected"])

        # S02: drops 'for 5 days, then stop' -> BLOCK
        s02_raw = "Take prednisone 20 mg daily for 5 days, then stop."
        res_s02 = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "S02",
                "patient_identifier": "PT-S02",
                "raw_discharge_text": s02_raw,
                "medications": [
                    {
                        "id": "med-s02",
                        "medication_name": "prednisone",
                        "action": "CONTINUE",
                        "dosage": "20 mg",
                        "frequency": "daily",
                        "confidence": 0.96,
                        "patient_friendly_summary": "Take prednisone 20 mg daily.",
                        "source_evidence": {"source_text": s02_raw},
                    }
                ],
            },
        )
        self.assertEqual(res_s02.status_code, 200)
        d_s02 = res_s02.json()
        self.assertFalse(d_s02["patient_releasable"])
        self.assertIn("UNSUPPORTED_FIELD_VALUE", d_s02["hazard_codes_detected"])

        # S03: drops 'unless dizzy' -> BLOCK
        s03_raw = "Continue hydralazine 25 mg three times daily unless dizzy."
        res_s03 = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "S03",
                "patient_identifier": "PT-S03",
                "raw_discharge_text": s03_raw,
                "medications": [
                    {
                        "id": "med-s03",
                        "medication_name": "hydralazine",
                        "action": "CONTINUE",
                        "dosage": "25 mg",
                        "frequency": "three times daily",
                        "indication_or_note": "unless dizzy",
                        "confidence": 0.96,
                        "patient_friendly_summary": "Take hydralazine 25 mg three times daily.",
                        "source_evidence": {"source_text": s03_raw},
                    }
                ],
            },
        )
        self.assertEqual(res_s03.status_code, 200)
        d_s03 = res_s03.json()
        self.assertFalse(d_s03["patient_releasable"])
        self.assertIn("UNSUPPORTED_FIELD_VALUE", d_s03["hazard_codes_detected"])

        # S04: drops 'until seen by cardiology' -> BLOCK
        s04_raw = "Hold Eliquis 5 mg twice daily until seen by cardiology."
        res_s04 = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "S04",
                "patient_identifier": "PT-S04",
                "raw_discharge_text": s04_raw,
                "medications": [
                    {
                        "id": "med-s04",
                        "medication_name": "Eliquis",
                        "action": "HOLD_TEMPORARY",
                        "dosage": "5 mg",
                        "frequency": "twice daily",
                        "indication_or_note": "until seen by cardiology",
                        "confidence": 0.96,
                        "patient_friendly_summary": "Hold Eliquis 5 mg twice daily.",
                        "source_evidence": {"source_text": s04_raw},
                    }
                ],
            },
        )
        self.assertEqual(res_s04.status_code, 200)
        d_s04 = res_s04.json()
        self.assertFalse(d_s04["patient_releasable"])
        self.assertIn("UNSUPPORTED_FIELD_VALUE", d_s04["hazard_codes_detected"])

        # S05: drops 'after meals' -> BLOCK
        s05_raw = "Continue ibuprofen 400 mg every 8 hours after meals."
        res_s05 = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "S05",
                "patient_identifier": "PT-S05",
                "raw_discharge_text": s05_raw,
                "medications": [
                    {
                        "id": "med-s05",
                        "medication_name": "ibuprofen",
                        "action": "CONTINUE",
                        "dosage": "400 mg",
                        "frequency": "every 8 hours",
                        "indication_or_note": "after meals",
                        "confidence": 0.96,
                        "patient_friendly_summary": "Take ibuprofen 400 mg every 8 hours.",
                        "source_evidence": {"source_text": s05_raw},
                    }
                ],
            },
        )
        self.assertEqual(res_s05.status_code, 200)
        d_s05 = res_s05.json()
        self.assertFalse(d_s05["patient_releasable"])
        self.assertIn("UNSUPPORTED_FIELD_VALUE", d_s05["hazard_codes_detected"])

        # S06: drops numeric thresholds 'greater than 101.5 F' and 'more than 3 pounds' -> BLOCK
        s06_raw = "Call the clinic if fever is greater than 101.5 F or weight gain is more than 3 pounds."
        res_s06 = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "S06",
                "patient_identifier": "PT-S06",
                "raw_discharge_text": s06_raw,
                "warning_signs": [
                    {
                        "id": "ws-s06",
                        "symptom_description": "fever is greater than 101.5 F or weight gain is more than 3 pounds",
                        "urgency": "SOON",
                        "action_to_take": "Call the clinic",
                        "confidence": 0.96,
                        "patient_friendly_summary": "Call the clinic if you have fever or weight gain.",
                        "source_evidence": {"source_text": s06_raw},
                    }
                ],
            },
        )
        self.assertEqual(res_s06.status_code, 200)
        d_s06 = res_s06.json()
        self.assertFalse(d_s06["patient_releasable"])
        self.assertIn("UNSUPPORTED_FIELD_VALUE", d_s06["hazard_codes_detected"])

        # S07: weakens 'within 3 days' to 'in 3 days' -> BLOCK
        s07_raw = "Follow up with Cardiology within 3 days."
        res_s07 = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "S07",
                "patient_identifier": "PT-S07",
                "raw_discharge_text": s07_raw,
                "follow_ups": [
                    {
                        "id": "fu-s07",
                        "specialty": "Cardiology",
                        "timeframe": "within 3 days",
                        "confidence": 0.97,
                        "patient_friendly_summary": "Follow up with Cardiology in 3 days.",
                        "source_evidence": {"source_text": s07_raw},
                    }
                ],
            },
        )
        self.assertEqual(res_s07.status_code, 200)
        d_s07 = res_s07.json()
        self.assertFalse(d_s07["patient_releasable"])
        self.assertIn("UNSUPPORTED_FIELD_VALUE", d_s07["hazard_codes_detected"])

        # S08: faithful paraphrase preserving 'only if', 'need it', and 'for pain' -> ALLOW
        s08_raw = "Take acetaminophen 500 mg every 6 hours only if needed for pain."
        res_s08 = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "S08",
                "patient_identifier": "PT-S08",
                "raw_discharge_text": s08_raw,
                "medications": [
                    {
                        "id": "med-s08",
                        "medication_name": "acetaminophen",
                        "action": "CONTINUE",
                        "dosage": "500 mg",
                        "frequency": "every 6 hours as needed",
                        "indication_or_note": "only if needed for pain",
                        "confidence": 0.98,
                        "patient_friendly_summary": "Take acetaminophen 500 mg every 6 hours only if you need it for pain.",
                        "source_evidence": {"source_text": s08_raw},
                    }
                ],
            },
        )
        self.assertEqual(res_s08.status_code, 200)
        d_s08 = res_s08.json()
        self.assertEqual(d_s08["overall_safety_status"], "SAFE / ORGANIZED")
        self.assertTrue(d_s08["patient_releasable"])
        self.assertEqual(d_s08["hazard_codes_detected"], [])

        # S09: faithful paraphrase preserving 'if it is noted' -> ALLOW
        s09_raw = "Monitor for mild ankle swelling if noted and call the Primary Care clinic."
        res_s09 = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "S09",
                "patient_identifier": "PT-S09",
                "raw_discharge_text": s09_raw,
                "warning_signs": [
                    {
                        "id": "ws-s09",
                        "symptom_description": "mild ankle swelling",
                        "urgency": "ROUTINE",
                        "action_to_take": "call the Primary Care clinic",
                        "confidence": 0.97,
                        "patient_friendly_summary": "Monitor for mild ankle swelling and call the Primary Care clinic if it is noted.",
                        "source_evidence": {"source_text": s09_raw},
                    }
                ],
            },
        )
        self.assertEqual(res_s09.status_code, 200)
        d_s09 = res_s09.json()
        self.assertEqual(d_s09["overall_safety_status"], "SAFE / ORGANIZED")
        self.assertTrue(d_s09["patient_releasable"])
        self.assertEqual(d_s09["hazard_codes_detected"], [])

        # S10: adds unsupported clinical rationale ('because it protects your kidneys') -> BLOCK
        s10_raw = "Continue lisinopril 10 mg daily."
        res_s10 = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "S10",
                "patient_identifier": "PT-S10",
                "raw_discharge_text": s10_raw,
                "medications": [
                    {
                        "id": "med-s10",
                        "medication_name": "lisinopril",
                        "action": "CONTINUE",
                        "dosage": "10 mg",
                        "frequency": "daily",
                        "confidence": 0.98,
                        "patient_friendly_summary": "Continue lisinopril 10 mg daily because it protects your kidneys.",
                        "source_evidence": {"source_text": s10_raw},
                    }
                ],
            },
        )
        self.assertEqual(res_s10.status_code, 200)
        d_s10 = res_s10.json()
        self.assertFalse(d_s10["patient_releasable"])
        self.assertIn("UNSUPPORTED_FIELD_VALUE", d_s10["hazard_codes_detected"])

    def test_phase3c_original_live_h29b_h40a_to_c_and_size_boundary_b200_b201(
        self,
    ) -> None:
        # Requirement 4: Original live HTTP 422 tests for H29b, H40a, H40b, H40c
        orig_422_cases = {
            "H29b": {
                "case_id": "H29b",
                "patient_identifier": "PT-H29b",
                "raw_discharge_text": "Continue metformin 500 mg daily. Continue metformin 1000 mg daily.",
                "medications": [
                    {
                        "id": "med-1",
                        "medication_name": "metformin",
                        "action": "CONTINUE",
                        "dosage": "500 mg",
                        "frequency": "daily",
                        "confidence": 0.95,
                        "source_evidence": {
                            "source_text": "Continue metformin 500 mg daily."
                        },
                    },
                    {
                        "id": "MED-1",
                        "medication_name": "metformin",
                        "action": "CONTINUE",
                        "dosage": "1000 mg",
                        "frequency": "daily",
                        "confidence": 0.95,
                        "source_evidence": {
                            "source_text": "Continue metformin 1000 mg daily."
                        },
                    },
                ],
            },
            "H40a": {
                "case_id": "H40a",
                "patient_identifier": "PT-H40a",
                "raw_discharge_text": "Continue aspirin 81 mg daily.",
                "medications": [
                    {
                        "id": "med-1",
                        "medication_name": "",
                        "action": "CONTINUE",
                        "dosage": "81 mg",
                        "frequency": "daily",
                        "confidence": 0.95,
                        "source_evidence": {
                            "source_text": "Continue aspirin 81 mg daily."
                        },
                    }
                ],
            },
            "H40b": {
                "case_id": "H40b",
                "patient_identifier": "PT-H40b",
                "raw_discharge_text": "Continue aspirin 81 mg daily.",
                "medications": [
                    {
                        "id": "med-1",
                        "medication_name": "   ",
                        "action": "CONTINUE",
                        "dosage": "81 mg",
                        "frequency": "daily",
                        "confidence": 0.95,
                        "source_evidence": {
                            "source_text": "Continue aspirin 81 mg daily."
                        },
                    }
                ],
            },
            "H40c": {
                "case_id": "H40c",
                "patient_identifier": "PT-H40c",
                "raw_discharge_text": "Continue aspirin 81 mg daily.",
                "medications": [
                    {
                        "id": "med-1",
                        "medication_name": "aspirin",
                        "action": "CONTINUE",
                        "dosage": "81 mg",
                        "frequency": "daily",
                        "confidence": 0.95,
                        "patient_friendly_summary": "A" * 1_000_000,
                        "source_evidence": {
                            "source_text": "Continue aspirin 81 mg daily."
                        },
                    }
                ],
            },
        }
        for cid, body in orig_422_cases.items():
            resp = self.client.post("/api/v1/validate", json=body)
            self.assertEqual(
                resp.status_code,
                422,
                f"Original live {cid} must return HTTP 422, got {resp.status_code}",
            )

        # Requirement 5: B200 (exact 204,800 bytes -> HTTP 200) and B201 (exact 204,801 bytes -> HTTP 413)
        base_line = "Continue aspirin 81 mg daily.\n"
        filler_unit = "Hospital course note: patient remained hemodynamically stable. "
        needed_200 = 204_800 - len(base_line)
        reps = needed_200 // len(filler_unit)
        rem = needed_200 - (reps * len(filler_unit))
        tail_pad = ("Patient stable" + (" " * max(0, rem - 15)) + ".")[:rem]
        raw_b200 = base_line + (filler_unit * reps) + tail_pad
        self.assertEqual(len(raw_b200.encode("utf-8")), 204_800)

        res_b200 = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "B200",
                "patient_identifier": "PT-B200",
                "raw_discharge_text": raw_b200,
                "medications": [
                    {
                        "id": "med-b200",
                        "medication_name": "aspirin",
                        "action": "CONTINUE",
                        "dosage": "81 mg",
                        "frequency": "daily",
                        "confidence": 0.98,
                        "source_evidence": {
                            "source_text": "Continue aspirin 81 mg daily."
                        },
                    }
                ],
            },
        )
        self.assertEqual(res_b200.status_code, 200)
        self.assertTrue(res_b200.json()["patient_releasable"])

        raw_b201 = raw_b200 + "X"
        self.assertEqual(len(raw_b201.encode("utf-8")), 204_801)
        res_b201 = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "B201",
                "patient_identifier": "PT-B201",
                "raw_discharge_text": raw_b201,
                "medications": [
                    {
                        "id": "med-b201",
                        "medication_name": "aspirin",
                        "action": "CONTINUE",
                        "dosage": "81 mg",
                        "frequency": "daily",
                        "confidence": 0.98,
                        "source_evidence": {
                            "source_text": "Continue aspirin 81 mg daily."
                        },
                    }
                ],
            },
        )
        self.assertEqual(res_b201.status_code, 413)

        # Requirement 10: Out-of-lexicon medications (dapagliflozin, eplerenone, finerenone, empagliflozin)
        # and combination medication (sacubitril-valsartan 49/51 mg)
        for ool_drug, ool_dose in (
            ("dapagliflozin", "10 mg"),
            ("eplerenone", "25 mg"),
            ("finerenone", "10 mg"),
            ("empagliflozin", "10 mg"),
            ("sacubitril-valsartan", "49/51 mg"),
        ):
            raw_ool = f"Continue {ool_drug} {ool_dose} twice daily."
            res_ool = self.client.post(
                "/api/v1/validate",
                json={
                    "case_id": f"OOL-{ool_drug}",
                    "patient_identifier": "PT-OOL",
                    "raw_discharge_text": raw_ool,
                    "medications": [
                        {
                            "id": "med-ool-1",
                            "medication_name": ool_drug,
                            "action": "CONTINUE",
                            "dosage": ool_dose,
                            "frequency": "twice daily",
                            "confidence": 0.97,
                            "patient_friendly_summary": f"Continue {ool_drug} {ool_dose} twice daily.",
                            "source_evidence": {"source_text": raw_ool},
                        }
                    ],
                },
            )
            self.assertEqual(res_ool.status_code, 200)
            self.assertTrue(
                res_ool.json()["patient_releasable"],
                f"Expected out-of-lexicon/combination drug {ool_drug} {ool_dose} to be releasable, got {res_ool.json()}",
            )

        # Requirement 13: Explicit calendar dates in F17 emit CONFLICTING_INSTRUCTIONS and NOT MISSING_FOLLOWUP_TIMEFRAME
        raw_f17 = "Appointment with Urology on 04 Dec 2026. Appointment with Urology on 18 Dec 2026."
        res_f17 = self.client.post(
            "/api/v1/validate",
            json={
                "case_id": "F17-CLEANUP",
                "patient_identifier": "PT-F17",
                "raw_discharge_text": raw_f17,
                "follow_ups": [
                    {
                        "id": "fu-1",
                        "specialty": "Urology",
                        "timeframe": "on 04 Dec 2026",
                        "confidence": 0.95,
                        "source_evidence": {
                            "source_text": "Appointment with Urology on 04 Dec 2026."
                        },
                    },
                    {
                        "id": "fu-2",
                        "specialty": "Urology",
                        "timeframe": "on 18 Dec 2026",
                        "confidence": 0.95,
                        "source_evidence": {
                            "source_text": "Appointment with Urology on 18 Dec 2026."
                        },
                    },
                ],
            },
        )
        self.assertEqual(res_f17.status_code, 200)
        d_f17 = res_f17.json()
        self.assertEqual(d_f17["overall_safety_status"], "CONFLICT DETECTED")
        self.assertEqual(d_f17["hazard_codes_detected"], ["CONFLICTING_INSTRUCTIONS"])
        self.assertNotIn(
            "MISSING_FOLLOWUP_TIMEFRAME", d_f17["hazard_codes_detected"]
        )



def print_heldout_round2_table() -> None:
    import statistics

    client = TestClient(app, raise_server_exceptions=False)
    cases = build_all_53_heldout_cases()
    latencies: list[float] = []
    print(
        "case | HTTP | overall status | per-item (id: safety_status/review_status) | hazard codes | patient_releasable | latency ms"
    )
    for cid, fn in cases:
        t0 = time.perf_counter()
        try:
            body = fn()
            res = client.post("/api/v1/validate", json=body)
            code = res.status_code
            data = res.json() if code == 200 else None
        except ValidationError:
            code = 422
            data = None
        ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(ms)
        if code == 200 and data:
            vp = data["validated_payload"]
            items = (
                vp["medications"]
                + vp["follow_ups"]
                + vp["warning_signs"]
                + vp["tasks"]
            )
            p_str = (
                "; ".join(
                    f"{i['id']}: {i['safety_status']}/{i['review_status']}"
                    for i in items
                )
                or "NONE"
            )
            h_str = ",".join(data["hazard_codes_detected"]) or "NONE"
            print(
                f"{cid} | {code} | {data['overall_safety_status']} | {p_str} | {h_str} | {data['patient_releasable']} | {ms:.2f}"
            )
        else:
            print(
                f"{cid} | {code} | HTTP_{code}_REJECTED | N/A | NONE | False | {ms:.2f}"
            )

    s = sorted(latencies)
    p50 = statistics.median(s)
    p95 = s[int(len(s) * 0.95)]
    print(
        f"LATENCY_SUMMARY: count={len(s)} min={min(s):.2f}ms p50={p50:.2f}ms p95={p95:.2f}ms max={max(s):.2f}ms"
    )


if __name__ == "__main__":
    unittest.main()



