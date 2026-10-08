"""Blind 40-case held-out execution script for CAREBRIDGE.

Plain script (NOT a unittest module). Runs once against POST /api/v1/validate
and writes raw unedited results to stdout and tests/heldout_output.txt.
"""

from __future__ import annotations

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

FU_SENTENCE = "Follow up with Primary Care within 1 week."


def med(
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


def fu(
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


def prov(
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


def payload(
    raw: str,
    meds: list[dict[str, Any]] | None = None,
    fus: list[dict[str, Any]] | None = None,
    warns: list[dict[str, Any]] | None = None,
    tasks: list[dict[str, Any]] | None = None,
    provs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    p_obj = DischargeExtractionPayload(
        case_id="HELDOUT",
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


def build_fu_default() -> dict[str, Any]:
    return fu("fu-1", "Primary Care", "within 1 week", FU_SENTENCE)


def run_heldout_suite() -> None:
    client = TestClient(app, raise_server_exceptions=False)

    def make_cases() -> list[tuple[str, Any]]:
        cases: list[tuple[str, Any]] = []

        # H01
        r01 = "Continue warfarin 2.5 mg daily."
        cases.append(
            (
                "H01",
                lambda: payload(
                    r01,
                    meds=[med("med-1", "warfarin", "CONTINUE", "5 mg", "daily", None, r01)],
                ),
            )
        )

        # H02
        r02 = "Continue levothyroxine 50 mcg daily."
        cases.append(
            (
                "H02",
                lambda: payload(
                    r02,
                    meds=[
                        med(
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

        # H03
        r03 = "Continue hydralazine 25 mg daily."
        cases.append(
            (
                "H03",
                lambda: payload(
                    r03,
                    meds=[
                        med(
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

        # H04
        r04 = "Continue lisinopril 10 mg daily."
        cases.append(
            (
                "H04",
                lambda: payload(
                    r04,
                    meds=[
                        med(
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

        # H05
        r05 = "Continue lisinopril 10 mg daily."
        cases.append(
            (
                "H05",
                lambda: payload(
                    r05,
                    meds=[
                        med(
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

        # H06
        r06 = "Take metformin 500 mg BID."
        cases.append(
            (
                "H06",
                lambda: payload(
                    r06,
                    meds=[
                        med(
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

        # H07
        r07 = "Follow up with the heart clinic in 2 weeks."
        cases.append(
            (
                "H07",
                lambda: payload(
                    r07,
                    fus=[fu("fu-1", "Neurology", "in 2 weeks", r07)],
                ),
            )
        )

        # H08
        r08 = "Follow up with Cardiology in 2 weeks."
        cases.append(
            (
                "H08",
                lambda: payload(
                    r08,
                    fus=[
                        fu(
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

        # H09
        r09 = "Continue atorvastatin 40 mg daily. Continue amlodipine 5 mg daily."
        cases.append(
            (
                "H09",
                lambda: payload(
                    r09,
                    meds=[
                        med(
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

        # H10
        r10 = "Continue atorvastatin 40 mg and amlodipine 5 mg daily."
        cases.append(
            (
                "H10",
                lambda: payload(
                    r10,
                    meds=[
                        med(
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

        # H11
        r11 = "Do not stop aspirin 81 mg daily."
        cases.append(
            (
                "H11",
                lambda: payload(
                    r11,
                    meds=[
                        med(
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

        # H12
        r12 = "Do not take ibuprofen."
        cases.append(
            (
                "H12",
                lambda: payload(
                    r12,
                    meds=[
                        med(
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

        # H13
        r13 = "Continue aspirin 81 mg daily. Follow up with Cardiology in 2 weeks."
        fu_sent_13 = "Follow up with Cardiology in 2 weeks."
        start_13 = r13.index(fu_sent_13)
        end_13 = start_13 + len(fu_sent_13)
        cases.append(
            (
                "H13",
                lambda: payload(
                    r13,
                    meds=[
                        med(
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

        # H14
        r14 = "HOME MEDICATIONS: Metformin 500 mg twice daily. DISCHARGE MEDICATIONS: Hold metformin until follow-up."
        cases.append(
            (
                "H14",
                lambda: payload(
                    r14,
                    meds=[
                        med(
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

        # H15
        r15 = f"D/C lisinopril. Hold ASA x 3 days. {FU_SENTENCE}"
        cases.append(
            (
                "H15",
                lambda: payload(r15, fus=[build_fu_default()]),
            )
        )

        # H16
        r16 = f"Taper prednisone: 20 mg x 3 days then 10 mg x 3 days. {FU_SENTENCE}"
        cases.append(
            (
                "H16",
                lambda: payload(r16, fus=[build_fu_default()]),
            )
        )

        # H17a
        r17a = f"Avoid NSAIDs. {FU_SENTENCE}"
        cases.append(
            (
                "H17a",
                lambda: payload(r17a, fus=[build_fu_default()]),
            )
        )

        # H17b
        r17b = f"Resume home medications except warfarin. {FU_SENTENCE}"
        cases.append(
            (
                "H17b",
                lambda: payload(r17b, fus=[build_fu_default()]),
            )
        )

        # H17c
        r17c = f"Return to the ED for fever above 101 F. {FU_SENTENCE}"
        cases.append(
            (
                "H17c",
                lambda: payload(r17c, fus=[build_fu_default()]),
            )
        )

        # H17d
        r17d = f"Repeat CBC in 1 week. {FU_SENTENCE}"
        cases.append(
            (
                "H17d",
                lambda: payload(r17d, fus=[build_fu_default()]),
            )
        )

        # H18
        r18 = "Continue aspirin 81 mg daily and stop ibuprofen."
        cases.append(
            (
                "H18",
                lambda: payload(
                    r18,
                    meds=[
                        med(
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

        # H19
        r19 = f"No new medications were started. {FU_SENTENCE}"
        cases.append(
            (
                "H19",
                lambda: payload(r19, fus=[build_fu_default()]),
            )
        )

        # H20
        r20 = "Patient is not taking metformin at home."
        cases.append(
            (
                "H20",
                lambda: payload(
                    r20,
                    meds=[
                        med("med-1", "metformin", "CONTINUE", None, None, None, r20)
                    ],
                ),
            )
        )

        # H21
        r21 = "Aspirin should not be stopped."
        cases.append(
            (
                "H21",
                lambda: payload(
                    r21,
                    meds=[med("med-1", "aspirin", "CONTINUE", None, None, None, r21)],
                ),
            )
        )

        # H22
        r22 = "disconti-\nnue lisinopril 10 mg."
        cases.append(
            (
                "H22",
                lambda: payload(
                    r22,
                    meds=[
                        med(
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

        # H23
        r23 = "Sto\u200bp lisinopril 10 mg daily."
        cases.append(
            (
                "H23",
                lambda: payload(
                    r23,
                    meds=[
                        med(
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

        # H24
        r24 = "CONTINUE   Aspirin  81MG\tdaily."
        cases.append(
            (
                "H24",
                lambda: payload(
                    r24,
                    meds=[
                        med(
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

        # H25
        r25 = "Continue \u2018aspirin\u2019 81\u00a0mg daily\u2011 as before."
        cases.append(
            (
                "H25",
                lambda: payload(
                    r25,
                    meds=[
                        med(
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

        # H26a
        r26a = ("Routine note. " * 14000) + "Stop lisinopril."
        cases.append(
            (
                "H26a",
                lambda: payload(
                    r26a,
                    meds=[
                        med(
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

        # H26b
        r26b_med = "Continue aspirin 81 mg daily."
        r26b = ("stop " * 40000) + r26b_med
        cases.append(
            (
                "H26b",
                lambda: payload(
                    r26b,
                    meds=[
                        med(
                            "med-1",
                            "aspirin",
                            "CONTINUE",
                            "81 mg",
                            "daily",
                            None,
                            r26b_med,
                        )
                    ],
                ),
            )
        )

        # H27
        r27 = "Glucophage (metformin) 500 mg daily: continue. Stop metformin."
        cases.append(
            (
                "H27",
                lambda: payload(
                    r27,
                    meds=[
                        med(
                            "med-1",
                            "Glucophage",
                            "CONTINUE",
                            "500 mg",
                            "daily",
                            None,
                            "Glucophage (metformin) 500 mg daily: continue.",
                        ),
                        med(
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

        # H28
        r28 = "Continue metformin 500 mg daily. Continue metformin 1000 mg daily."
        cases.append(
            (
                "H28",
                lambda: payload(
                    r28,
                    meds=[
                        med(
                            "med-1",
                            "metformin",
                            "CONTINUE",
                            "500 mg",
                            "daily",
                            None,
                            "Continue metformin 500 mg daily.",
                        ),
                        med(
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

        # H29a
        r29a = "Continue metformin 500 mg daily."
        cases.append(
            (
                "H29a",
                lambda: payload(
                    r29a,
                    meds=[
                        med(
                            "med-1",
                            "metformin",
                            "CONTINUE",
                            "500 mg",
                            "daily",
                            None,
                            r29a,
                        ),
                        med(
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

        # H29b
        r29b = "Continue metformin 500 mg daily. Continue metformin 1000 mg daily."
        cases.append(
            (
                "H29b",
                lambda: payload(
                    r29b,
                    meds=[
                        med(
                            "med-1",
                            "metformin",
                            "CONTINUE",
                            "500 mg",
                            "daily",
                            None,
                            "Continue metformin 500 mg daily.",
                        ),
                        med(
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

        # H30
        r30 = "Follow up with Cardiology within 1 week. Follow up with Cardiology after 1 week."
        cases.append(
            (
                "H30",
                lambda: payload(
                    r30,
                    fus=[
                        fu(
                            "fu-1",
                            "Cardiology",
                            "within 1 week",
                            "Follow up with Cardiology within 1 week.",
                        ),
                        fu(
                            "fu-2",
                            "Cardiology",
                            "after 1 week",
                            "Follow up with Cardiology after 1 week.",
                        ),
                    ],
                ),
            )
        )

        # H31
        r31 = "Follow up with Cardiology in 2 weeks. Follow up with Cardiology within 14 days."
        cases.append(
            (
                "H31",
                lambda: payload(
                    r31,
                    fus=[
                        fu(
                            "fu-1",
                            "Cardiology",
                            "in 2 weeks",
                            "Follow up with Cardiology in 2 weeks.",
                        ),
                        fu(
                            "fu-2",
                            "Cardiology",
                            "within 14 days",
                            "Follow up with Cardiology within 14 days.",
                        ),
                    ],
                ),
            )
        )

        # H32
        r32 = "Follow up with Cardiology within 2 weeks. Follow up with Cardiology within 14 days."
        cases.append(
            (
                "H32",
                lambda: payload(
                    r32,
                    fus=[
                        fu(
                            "fu-1",
                            "Cardiology",
                            "within 2 weeks",
                            "Follow up with Cardiology within 2 weeks.",
                        ),
                        fu(
                            "fu-2",
                            "Cardiology",
                            "within 14 days",
                            "Follow up with Cardiology within 14 days.",
                        ),
                    ],
                ),
            )
        )

        # H33
        r33 = "Follow up with Cardiology in about 2 weeks."
        cases.append(
            (
                "H33",
                lambda: payload(
                    r33,
                    fus=[fu("fu-1", "Cardiology", "within 2 weeks", r33)],
                ),
            )
        )

        # H34
        r34 = "Follow up with Cardiology on 12 Nov 2026. Follow up with Cardiology on 19 Nov 2026."
        cases.append(
            (
                "H34",
                lambda: payload(
                    r34,
                    fus=[
                        fu(
                            "fu-1",
                            "Cardiology",
                            "on 12 Nov 2026",
                            "Follow up with Cardiology on 12 Nov 2026.",
                        ),
                        fu(
                            "fu-2",
                            "Cardiology",
                            "on 19 Nov 2026",
                            "Follow up with Cardiology on 19 Nov 2026.",
                        ),
                    ],
                ),
            )
        )

        # H35a-e
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
                    lambda c=claim: payload(
                        r35,
                        meds=[
                            med(
                                "med-1",
                                "aspirin",
                                "CONTINUE",
                                "81 mg",
                                "daily",
                                None,
                                r35,
                            )
                        ],
                        provs=[prov("prov-1", claim_text=c)],
                    ),
                )
            )

        # H36
        r36 = "Continue aspirin 81 mg daily."
        cases.append(
            (
                "H36",
                lambda: payload(
                    r36,
                    meds=[
                        med(
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
                        prov(
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

        # H37
        r37 = "Continue aspirin daily."
        cases.append(
            (
                "H37",
                lambda: payload(
                    r37,
                    meds=[
                        med(
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

        # H38
        r38 = f"Continue warfarin 2.5 mg daily. {FU_SENTENCE}"
        cases.append(
            (
                "H38",
                lambda: payload(
                    r38,
                    meds=[
                        med(
                            "med-1",
                            "warfarin",
                            "CONTINUE",
                            "5 mg",
                            "daily",
                            None,
                            "Continue warfarin 2.5 mg daily.",
                        )
                    ],
                    fus=[build_fu_default()],
                ),
            )
        )

        # H39 (payload of H15)
        cases.append(
            (
                "H39",
                lambda: payload(r15, fus=[build_fu_default()]),
            )
        )

        # H40a
        r40 = "Continue aspirin 81 mg daily."
        cases.append(
            (
                "H40a",
                lambda: payload(
                    r40,
                    meds=[
                        med("med-1", "", "CONTINUE", "81 mg", "daily", None, r40)
                    ],
                ),
            )
        )

        # H40b
        cases.append(
            (
                "H40b",
                lambda: payload(
                    r40,
                    meds=[
                        med("med-1", "   ", "CONTINUE", "81 mg", "daily", None, r40)
                    ],
                ),
            )
        )

        # H40c
        cases.append(
            (
                "H40c",
                lambda: payload(
                    r40,
                    meds=[
                        med(
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

        # H40d
        cases.append(
            (
                "H40d",
                lambda: payload(
                    r40,
                    meds=[
                        med(
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

        # H40e
        cases.append(
            (
                "H40e",
                lambda: payload(
                    r40,
                    meds=[
                        med("med-1", "aspirin", "CONTINUE", "NaN", "daily", None, r40)
                    ],
                ),
            )
        )

        return cases

    lines_out: list[str] = []
    header = "case | HTTP | overall status | per-item (id: safety_status/review_status) | hazard codes | patient_releasable | latency ms"
    lines_out.append(header)

    count_200 = 0
    count_422 = 0
    count_500 = 0

    for case_id, builder_fn in make_cases():
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

        if http_code == 200 and isinstance(resp_json, dict):
            count_200 += 1
            overall = str(resp_json.get("overall_safety_status", ""))
            vp = resp_json.get("validated_payload", {})
            all_items = (
                list(vp.get("medications", []))
                + list(vp.get("follow_ups", []))
                + list(vp.get("warning_signs", []))
                + list(vp.get("tasks", []))
            )
            per_item_str = "; ".join(
                f"{it.get('id')}: {it.get('safety_status')}/{it.get('review_status')}"
                for it in all_items
            )
            hazards_str = ",".join(
                str(c) for c in resp_json.get("hazard_codes_detected", [])
            )
            if not hazards_str:
                hazards_str = "NONE"
            releasable_str = str(resp_json.get("patient_releasable", False))
        elif http_code == 422:
            count_422 += 1
            overall = "HTTP_422_REJECTED"
            per_item_str = "N/A"
            hazards_str = "NONE"
            releasable_str = "False"
        else:
            if http_code >= 500:
                count_500 += 1
            overall = f"HTTP_{http_code}"
            per_item_str = "N/A"
            hazards_str = "NONE"
            releasable_str = "False"

        row = f"{case_id} | {http_code} | {overall} | {per_item_str} | {hazards_str} | {releasable_str} | {latency_ms:.2f}"
        lines_out.append(row)

    lines_out.append(
        f"TOTALS: HTTP 200={count_200} | HTTP 422={count_422} | HTTP 500={count_500}"
    )

    full_text = "\n".join(lines_out) + "\n"
    out_path = REPO_ROOT / "tests" / "heldout_output.txt"
    out_path.write_text(full_text, encoding="utf-8")
    print(full_text, end="")


if __name__ == "__main__":
    run_heldout_suite()
