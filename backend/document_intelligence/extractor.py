"""Deterministic, provenance-preserving clinical extractor for Phase 4 Document Intelligence.

Extracts structured Medications, Follow-Ups, Tests, Referrals, Warning Signs, and
Care Instructions from an IngestedDocument without inventing missing clinical facts,
then validates the resulting DischargeExtractionPayload through the frozen Phase 3
Safety Validation Layer.
"""

from __future__ import annotations

import re
from typing import Optional

from backend.document_intelligence.ingestion import ingest_discharge_document
from backend.document_intelligence.types import (
    CareInstructionCategory,
    DocumentFormat,
    DocumentIntelligenceResult,
    EvidenceSpan,
    ExtractedCareInstructionRecord,
    ExtractedFollowUpRecord,
    ExtractedMedicationRecord,
    ExtractedReferralRecord,
    ExtractedTestRecord,
    ExtractedWarningSignRecord,
    IngestedDocument,
    SectionCategory,
    SegmentedSentence,
)
from backend.safety.validator import (
    CRITICAL_RED_FLAG_PATTERNS,
    EMERGENT_CUE_REGEX,
    SYNTHETIC_DRUG_ALIASES,
    UNCLEAR_SYMPTOM_PATTERNS,
    URGENT_CUE_REGEX,
    validate_discharge_payload,
)
from backend.schemas.models import (
    CareTaskItem,
    DischargeExtractionPayload,
    FollowUpInstruction,
    MedicationAction,
    MedicationInstruction,
    UrgencyLevel,
    WarningSignInstruction,
)

# Common pharmaceutical salt suffixes that follow a base generic drug name
PHARMACEUTICAL_SALT_SUFFIXES = {
    "succinate",
    "tartrate",
    "potassium",
    "sodium",
    "besylate",
    "hydrochloride",
    "hcl",
    "fumarate",
    "maleate",
    "calcium",
    "acetonide",
    "mononitrate",
    "dinitrate",
    "carboplatin",
    "propionate",
    "valerate",
}

MEDICATION_VERB_START_RE = re.compile(
    r"^(?:please\s+)?(?:continue|start|begin|initiate|stop|discontinue|hold|pause|take|resume|"
    r"increase|decrease|change|switch|reduce|do\s+not\s+stop|do\s+not\s+take)\b",
    re.IGNORECASE,
)

GOVERNED_NEGATION_RE = re.compile(
    r"^no\s+(?:new\s+|home\s+)?medications?\s+were\s+(?:started|stopped|changed|modified|discontinued|held)",
    re.IGNORECASE,
)

GOVERNED_NEGATION_EXCEPTION_RE = re.compile(
    r"\b(?:except(?:\s+for)?|other\s+than|unless|excluding)\s+([A-Za-z][A-Za-z0-9/-]+)",
    re.IGNORECASE,
)

DOSE_CHANGE_FROM_TO_RE = re.compile(
    r"\bfrom\s+(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\s*(?:mg|mcg|g|meq|units?|iu|ml|puffs?|%))\s+to\s+"
    r"(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\s*(?:mg|mcg|g|meq|units?|iu|ml|puffs?|%))",
    re.IGNORECASE,
)

SINGLE_DOSE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\s*(?:mg|mcg|g|meq|units?|iu|ml|puffs?|%|tablets?|capsules?))\b",
    re.IGNORECASE,
)

ROUTE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(by\s+mouth|orally|po)\b", re.IGNORECASE),
    re.compile(r"\b(subcutaneously|subq|sq)\b", re.IGNORECASE),
    re.compile(r"\b(intravenously|iv)\b", re.IGNORECASE),
    re.compile(r"\b(intramuscularly|im)\b", re.IGNORECASE),
    re.compile(r"\b(topically|inhaled|sublingually|nasally)\b", re.IGNORECASE),
]

FREQUENCY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(every\s+\d+\s+hours?\s+only\s+if\s+needed(?:\s+for\s+[a-z\s]+?)?)(?=[.,;]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(every\s+\d+\s+hours?\s+as\s+needed(?:\s+for\s+[a-z\s]+?)?)(?=[.,;]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:once|twice|three\s+times|four\s+times)\s+daily\s+(?:at\s+bedtime|in\s+the\s+morning|in\s+the\s+evening|with\s+(?:the\s+)?(?:evening\s+|morning\s+)?meals?))",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(daily\s+(?:at\s+bedtime|in\s+the\s+morning|in\s+the\s+evening|with\s+meals?))",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(every\s+\d+(?:\s*(?:to|-)\s*\d+)?\s+hours?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(once\s+daily|twice\s+daily|three\s+times\s+daily|four\s+times\s+daily|every\s+morning|every\s+evening|every\s+night|at\s+bedtime|daily|bid|tid|qid|qhs|qd|as\s+needed\s+if\s+needed|only\s+if\s+needed|as\s+needed|prn)\b",
        re.IGNORECASE,
    ),
]

DURATION_RE = re.compile(
    r"\b(for\s+\d+\s+(?:days?|weeks?|months?))\b",
    re.IGNORECASE,
)

CONDITION_NOTE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(until\s+[^.,;]+)", re.IGNORECASE),
    re.compile(r"\b(unless\s+[^.,;]+)", re.IGNORECASE),
    re.compile(r"\b(only\s+if\s+needed(?:\s+for\s+[^.,;]+)?)", re.IGNORECASE),
    re.compile(r"\b(due\s+to\s+[^.,;]+)", re.IGNORECASE),
    re.compile(r"\b(after\s+meals?|with\s+(?:the\s+)?(?:evening\s+|morning\s+)?meals?)", re.IGNORECASE),
    re.compile(r"\b(then\s+stop)\b", re.IGNORECASE),
]

KNOWN_SPECIALTIES = [
    "Primary Care",
    "Cardiology",
    "Nephrology",
    "Neurology",
    "Pulmonology",
    "Endocrinology",
    "Gastroenterology",
    "Hematology",
    "Oncology",
    "Urology",
    "Orthopedics",
    "Rheumatology",
    "Infectious Disease",
    "Vascular Surgery",
    "General Surgery",
    "Cardiac Surgery",
    "Neurosurgery",
    "Surgery",
    "Ophthalmology",
    "Dermatology",
    "Psychiatry",
    "Physical Therapy",
    "Occupational Therapy",
    "Cardiac Rehabilitation",
    "Pulmonary Rehabilitation",
    "Wound Care Clinic",
    "Anticoagulation Clinic",
    "Heart Failure Clinic",
]

SPECIALTY_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(KNOWN_SPECIALTIES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

PROVIDER_NAME_RE = re.compile(
    r"\b(Dr\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)

# Phase 4.1 fix: capture optional time-of-day suffix ("at HH:MM [AM/PM]") on
# absolute-date patterns so the extracted timeframe contains 'at', matching
# the 'IN' qualifier that _extract_timeframe_qualifier also sees in the source
# evidence sentence (which contains "at 10:30 AM").  This eliminates the
# spurious ON-vs-IN qualifier mismatch without modifying the frozen validator.
TIMEFRAME_RE = re.compile(
    r"\b((?:within|in|after)\s+(?:about\s+|approximately\s+)?\d+(?:\s*(?:to|-)\s*\d+)?\s*(?:days?|weeks?|months?|hours?)|"
    r"on\s+(?:\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|[A-Za-z]{3,9}\s+\d{1,2}(?:,\s*\d{4})?|\d{4}-\d{2}-\d{2})"
    r"(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:[AP]M)?)?)",
    re.IGNORECASE,
)

# Phase 4.1: extracts time-of-day component from absolute appointment sentences.
# Supports: '10:30 AM', '10 AM', '14:30', '2 PM'. Returns normalised 24-h string.
APPT_TIME_RE = re.compile(
    r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*([AP]M)?",
    re.IGNORECASE,
)

FOLLOWUP_REASON_RE = re.compile(
    r"\bfor\s+([a-zA-Z0-9\s/-]{3,60}?)(?=\s+(?:within|in|on|at)\b|[.,;]|$)",
    re.IGNORECASE,
)

FOLLOWUP_CUE_RE = re.compile(
    r"\b(?:follow[\s-]?up\s+with|schedule\s+(?:a\s+|an\s+)?(?:visit|appointment|follow[\s-]?up)\s+with|"
    r"appointment\s+with|return\s+to\s+(?:the\s+)?clinic|see\s+Dr\.)\b",
    re.IGNORECASE,
)

REFERRAL_CUE_RE = re.compile(
    r"\b(?:refer(?:red|ral)?\s+to|outpatient\s+referral\s+to|consult(?:ation)?\s+with|"
    r"home\s+health\s+(?:nursing|physical\s+therapy)|cardiac\s+rehabilitation\s+referral)\b",
    re.IGNORECASE,
)

TEST_CUE_RE = re.compile(
    r"\b(?:basic\s+metabolic\s+panel|bmp|comprehensive\s+metabolic\s+panel|cmp|complete\s+blood\s+count|cbc|"
    r"pt\s*/\s*inr|inr\s+check|inr\s+level|lipid\s+panel|renal\s+function\s+panel|liver\s+function\s+tests?|"
    r"hemoglobin\s+a1c|hba1c|echocardiogram|transthoracic\s+echocardiogram|chest\s+x-ray|cxr|ekg|ecg|"
    r"urinalysis|blood\s+cultures?|serum\s+creatinine|potassium\s+level|outpatient\s+lab(?:s|oratory)?)\b",
    re.IGNORECASE,
)

WARNING_CUE_RE = re.compile(
    r"\b(?:call\s+911|go\s+to\s+the\s+emergency|return\s+to\s+the\s+emergency|seek\s+immediate\s+medical|"
    r"call\s+(?:your\s+|the\s+)?(?:doctor|physician|clinic|primary\s+care|cardiology|nephrology|office)|"
    r"call\s+if\b|monitor\s+for\b|watch\s+for\b|if\s+you\s+develop\b|if\s+you\s+experience\b)\b",
    re.IGNORECASE,
)


def _make_evidence(sent: SegmentedSentence) -> EvidenceSpan:
    """Create a verbatim EvidenceSpan from a SegmentedSentence."""
    return EvidenceSpan(
        sentence_id=sent.sentence_id,
        section=sent.section,
        source_text=sent.source_text,
        start_offset=sent.start_offset,
        end_offset=sent.end_offset,
    )


def _extract_medication_from_sentence(
    sent: SegmentedSentence, item_index: int
) -> Optional[ExtractedMedicationRecord]:
    """Deterministically extract a medication instruction from a sentence if present."""
    norm = sent.normalized_text.strip()

    # Check governed negation first (e.g., "No new medications were started.")
    if GOVERNED_NEGATION_RE.search(norm):
        m_exc = GOVERNED_NEGATION_EXCEPTION_RE.search(norm)
        if not m_exc:
            return None
        exc_drug = m_exc.group(1).strip(" .,;:")
        action = (
            MedicationAction.STOP_DISCONTINUE
            if "stopped" in norm.lower() or "discontinued" in norm.lower()
            else MedicationAction.UNKNOWN
        )
        return ExtractedMedicationRecord(
            id=f"med-{item_index}",
            medication_name=exc_drug,
            raw_medication_display=exc_drug,
            action=action,
            dosage=None,
            previous_dosage=None,
            frequency=None,
            route=None,
            timeframe_or_duration=None,
            indication_or_note=f"except {exc_drug}",
            confidence=0.75,
            is_ambiguous=True,
            ambiguity_note="Extracted from governed negation exception clause; requires human verification.",
            evidence=_make_evidence(sent),
        )

    # Determine action
    lower = norm.lower()
    is_ambiguous = False
    ambiguity_note: Optional[str] = None

    if any(cue in lower for cue in ("maybe", "when ready", "as needed if needed", "usual dose")):
        action = MedicationAction.UNKNOWN
        is_ambiguous = True
        ambiguity_note = "Medication instruction contains ambiguous dosing or conditional phrasing."
    elif re.search(r"\b(?:increase|decrease|change|switch|titrate)\b", lower) or DOSE_CHANGE_FROM_TO_RE.search(lower):
        action = MedicationAction.DOSE_CHANGE
    elif re.search(r"\b(?:stop|discontinue|cease|do\s+not\s+take)\b", lower) and not re.search(
        r"\b(?:do\s+not\s+stop|should\s+not\s+be\s+stopped)\b", lower
    ):
        action = MedicationAction.STOP_DISCONTINUE
    elif re.search(r"\b(?:hold|pause|temporarily\s+stop)\b", lower):
        action = MedicationAction.HOLD_TEMPORARY
    elif re.search(r"\b(?:start\s+new|start|begin|initiate)\b", lower):
        action = MedicationAction.NEW_START
    elif re.search(r"\b(?:continue|keep\s+taking|resume|take)\b", lower):
        action = MedicationAction.CONTINUE
    else:
        action = MedicationAction.UNKNOWN
        is_ambiguous = True
        ambiguity_note = "No explicit medication action verb (continue/start/stop/hold/change) found."

    # Extract drug name
    raw_drug_name: Optional[str] = None
    # Pattern 1: "<DrugName>: take ..."
    m_colon = re.match(r"^([A-Z][a-zA-Z0-9/-]+(?:\s+[a-zA-Z]+)?)\s*:\s*", norm)
    if m_colon and m_colon.group(1).lower() not in ("medication", "medications", "note", "discharge"):
        raw_drug_name = m_colon.group(1).strip()

    # Pattern 2: After action verb
    if not raw_drug_name:
        m_verb = re.search(
            r"\b(?:continue\s+taking|stop\s+taking|keep\s+taking|start\s+new|switch\s+from|"
            r"continue|start|begin|initiate|stop|discontinue|hold|pause|resume|increase|decrease|change|take)\s+"
            r"(?:home\s+medication\s+)?([A-Za-z][a-zA-Z0-9/-]+(?:\s+(?:succinate|tartrate|potassium|sodium|besylate|hydrochloride|hcl|fumarate|maleate|calcium|chloride))?)",
            norm,
            re.IGNORECASE,
        )
        if m_verb:
            raw_drug_name = m_verb.group(1).strip()

    # Pattern 3: Match known drug lexicon in sentence
    if not raw_drug_name:
        for alias in sorted(SYNTHETIC_DRUG_ALIASES.keys(), key=len, reverse=True):
            m_lex = re.search(rf"\b({re.escape(alias)})\b", norm, re.IGNORECASE)
            if m_lex:
                raw_drug_name = m_lex.group(1)
                break

    if not raw_drug_name:
        return None

    # Preserve raw medication name without stripping salt suffixes (e.g. metoprolol tartrate)
    canonical_med_name = raw_drug_name

    # Extract dosage & previous_dosage
    previous_dosage: Optional[str] = None
    dosage: Optional[str] = None
    m_from_to = DOSE_CHANGE_FROM_TO_RE.search(norm)
    if m_from_to:
        previous_dosage = m_from_to.group(1).strip()
        dosage = m_from_to.group(2).strip()
    else:
        m_dose = SINGLE_DOSE_RE.search(norm)
        if m_dose:
            dosage = m_dose.group(1).strip()
        elif "usual dose" in lower:
            dosage = "usual dose"

    if dosage is None and action in (MedicationAction.CONTINUE, MedicationAction.NEW_START, MedicationAction.DOSE_CHANGE):
        is_ambiguous = True
        ambiguity_note = ambiguity_note or f"Missing explicit dosage for medication '{raw_drug_name}'."

    # Extract route
    route: Optional[str] = None
    for r_pat in ROUTE_PATTERNS:
        m_r = r_pat.search(norm)
        if m_r:
            route = m_r.group(1).strip()
            break

    # Extract frequency
    frequency: Optional[str] = None
    for f_pat in FREQUENCY_PATTERNS:
        m_f = f_pat.search(norm)
        if m_f:
            frequency = m_f.group(1).strip()
            break

    # Extract duration / timeframe
    timeframe_or_duration: Optional[str] = None
    m_dur = DURATION_RE.search(norm)
    if m_dur:
        timeframe_or_duration = m_dur.group(1).strip()

    # Extract conditional / exception / clinical notes
    notes: list[str] = []
    if timeframe_or_duration:
        notes.append(timeframe_or_duration)
    for c_pat in CONDITION_NOTE_PATTERNS:
        m_c = c_pat.search(norm)
        if m_c:
            val = m_c.group(1).strip()
            if val.lower() not in ( (frequency or "").lower(), (timeframe_or_duration or "").lower() ):
                if val not in notes:
                    notes.append(val)
    if "maybe resume when ready" in lower and "maybe resume when ready" not in notes:
        notes.append("maybe resume when ready")

    indication_or_note = "; ".join(notes) if notes else None

    confidence = 0.70 if is_ambiguous else 0.96
    return ExtractedMedicationRecord(
        id=f"med-{item_index}",
        medication_name=canonical_med_name,
        raw_medication_display=raw_drug_name,
        action=action,
        dosage=dosage,
        previous_dosage=previous_dosage,
        frequency=frequency,
        route=route,
        timeframe_or_duration=timeframe_or_duration,
        indication_or_note=indication_or_note,
        confidence=confidence,
        is_ambiguous=is_ambiguous,
        ambiguity_note=ambiguity_note,
        evidence=_make_evidence(sent),
    )


def _extract_followup_from_sentence(
    sent: SegmentedSentence, item_index: int
) -> Optional[ExtractedFollowUpRecord]:
    """Deterministically extract a follow-up appointment instruction from a sentence."""
    norm = sent.normalized_text.strip()

    m_spec = SPECIALTY_REGEX.search(norm)
    specialty = m_spec.group(1).strip() if m_spec else None

    m_prov = PROVIDER_NAME_RE.search(norm)
    provider_name = m_prov.group(1).strip() if m_prov else None

    m_tf = TIMEFRAME_RE.search(norm)
    timeframe = m_tf.group(1).strip() if m_tf else None

    reason: Optional[str] = None
    m_reason = FOLLOWUP_REASON_RE.search(norm)
    if m_reason:
        candidate_reason = m_reason.group(1).strip(" .,;:")
        # Avoid capturing timeframe or generic words as reason
        if candidate_reason and not TIMEFRAME_RE.search(candidate_reason):
            reason = candidate_reason

    appointment_scheduled: Optional[bool] = None
    if re.search(r"\b(?:scheduled\s+for|appointment\s+is\s+scheduled|booked\s+for)\b", norm, re.IGNORECASE):
        appointment_scheduled = True
    elif re.search(r"\b(?:schedule\s+a|please\s+schedule|call\s+to\s+schedule)\b", norm, re.IGNORECASE):
        appointment_scheduled = False

    is_ambiguous = False
    ambiguity_notes: list[str] = []
    if specialty is None:
        is_ambiguous = True
        ambiguity_notes.append("Follow-up specialty is missing or generic (e.g., 'clinic' or 'doctor').")
    if timeframe is None or re.search(r"\b(?:about|approximately|around|soon)\b", timeframe, re.IGNORECASE):
        is_ambiguous = True
        ambiguity_notes.append("Follow-up timeframe is missing or approximate.")

    confidence = 0.78 if is_ambiguous else 0.96
    # Phase 4.1: extract time-of-day only when an absolute date is present.
    appointment_time: Optional[str] = None
    if timeframe and re.match(r"^on\s+", timeframe, re.IGNORECASE):
        m_time = APPT_TIME_RE.search(norm)
        if m_time:
            hour = int(m_time.group(1))
            minute = int(m_time.group(2)) if m_time.group(2) else 0
            meridiem = (m_time.group(3) or "").upper()
            if meridiem == "PM" and hour != 12:
                hour += 12
            elif meridiem == "AM" and hour == 12:
                hour = 0
            appointment_time = f"{hour:02d}:{minute:02d}"

    return ExtractedFollowUpRecord(
        id=f"fu-{item_index}",
        specialty=specialty,
        provider_name=provider_name,
        timeframe=timeframe,
        reason=reason,
        appointment_scheduled=appointment_scheduled,
        appointment_time=appointment_time,
        confidence=confidence,
        is_ambiguous=is_ambiguous,
        ambiguity_note="; ".join(ambiguity_notes) if ambiguity_notes else None,
        evidence=_make_evidence(sent),
    )


def _extract_referral_from_sentence(
    sent: SegmentedSentence, item_index: int
) -> Optional[ExtractedReferralRecord]:
    """Deterministically extract a specialty/service referral instruction from a sentence."""
    norm = sent.normalized_text.strip()
    m_spec = SPECIALTY_REGEX.search(norm)
    specialty = m_spec.group(1).strip() if m_spec else None
    if not specialty:
        m_to = re.search(
            r"\b(?:refer(?:red|ral)?\s+to|consult(?:ation)?\s+with)\s+([A-Z][A-Za-z\s]{2,35}?)(?=\s+(?:within|in|for|on)\b|[.,;]|$)",
            norm,
        )
        if m_to:
            specialty = m_to.group(1).strip()

    m_prov = PROVIDER_NAME_RE.search(norm)
    provider_name = m_prov.group(1).strip() if m_prov else None

    m_tf = TIMEFRAME_RE.search(norm)
    timeframe = m_tf.group(1).strip() if m_tf else None

    reason: Optional[str] = None
    m_reason = FOLLOWUP_REASON_RE.search(norm)
    if m_reason:
        reason = m_reason.group(1).strip(" .,;:")

    is_ambiguous = specialty is None or timeframe is None
    return ExtractedReferralRecord(
        id=f"ref-{item_index}",
        specialty_or_service=specialty,
        provider_name=provider_name,
        timeframe=timeframe,
        reason=reason,
        confidence=0.82 if is_ambiguous else 0.95,
        is_ambiguous=is_ambiguous,
        evidence=_make_evidence(sent),
    )


def _extract_test_from_sentence(
    sent: SegmentedSentence, item_index: int
) -> Optional[ExtractedTestRecord]:
    """Deterministically extract an outpatient diagnostic/laboratory test instruction."""
    norm = sent.normalized_text.strip()
    m_test = TEST_CUE_RE.search(norm)
    if not m_test:
        return None
    test_name = m_test.group(0).strip()
    m_tf = TIMEFRAME_RE.search(norm)
    timeframe = m_tf.group(1).strip() if m_tf else None

    reason: Optional[str] = None
    m_for = FOLLOWUP_REASON_RE.search(norm)
    if m_for:
        reason = m_for.group(1).strip(" .,;:")

    is_ambiguous = timeframe is None
    return ExtractedTestRecord(
        id=f"test-{item_index}",
        test_name=test_name,
        timeframe=timeframe,
        reason_or_note=reason,
        confidence=0.84 if is_ambiguous else 0.95,
        is_ambiguous=is_ambiguous,
        evidence=_make_evidence(sent),
    )


def _extract_warning_sign_from_sentence(
    sent: SegmentedSentence, item_index: int
) -> Optional[ExtractedWarningSignRecord]:
    """Deterministically extract a return precaution or warning sign from a sentence."""
    norm = sent.normalized_text.strip().rstrip(".")
    lower = norm.lower()

    action_to_take: Optional[str] = None
    symptom_description: Optional[str] = None

    # Pattern A: "Monitor for <symptom> and <action> if noted"
    m_mon = re.match(
        r"^(?:monitor|watch)\s+for\s+(.+?)\s+and\s+(call\s+[^.,;]+?)(?:\s+if\s+noted|\s+if\s+it\s+happens)?$",
        norm,
        re.IGNORECASE,
    )
    if m_mon:
        symptom_description = m_mon.group(1).strip()
        action_to_take = m_mon.group(2).strip()

    # Pattern B: "<Action> if you have/develop/experience/your <symptom>" or "<Action> for <symptom>"
    if not symptom_description:
        m_act_if = re.match(
            r"^(call\s+911(?:\s+immediately)?|go\s+to\s+the\s+emergency\s+(?:department|room)(?:\s+immediately)?|"
            r"return\s+to\s+the\s+emergency\s+(?:department|room)|seek\s+immediate\s+medical\s+(?:attention|care)|"
            r"call\s+(?:your\s+|the\s+)?(?:doctor|physician|primary\s+care(?:\s+clinic)?|cardiology(?:\s+clinic)?|clinic)(?:\s+during\s+office\s+hours)?)\s+"
            r"(?:immediately\s+)?(?:if\s+(?:you\s+have\s+|you\s+develop\s+|you\s+experience\s+|your\s+)?|for\s+)(.+)$",
            norm,
            re.IGNORECASE,
        )
        if m_act_if:
            action_to_take = m_act_if.group(1).strip()
            symptom_description = m_act_if.group(2).strip()

    # Pattern C: "Call if <vague symptom>" (missing specific target)
    if not symptom_description:
        m_vague = re.match(r"^call\s+if\s+(.+)$", norm, re.IGNORECASE)
        if m_vague:
            symptom_description = m_vague.group(1).strip()
            action_to_take = None

    # Fallback inside Warning Signs section: use verbatim clause
    if not symptom_description:
        symptom_description = norm
        m_act = re.search(
            r"\b(call\s+911|go\s+to\s+the\s+emergency\s+(?:department|room)|call\s+(?:your\s+|the\s+)?(?:doctor|clinic|primary\s+care|cardiology))\b",
            norm,
            re.IGNORECASE,
        )
        action_to_take = m_act.group(1).strip() if m_act else None

    # Determine urgency deterministically from source cues
    is_vague = action_to_take is None or any(
        p.search(symptom_description) for p in UNCLEAR_SYMPTOM_PATTERNS
    )
    if EMERGENT_CUE_REGEX.search(lower) or any(
        p.search(lower) for p in CRITICAL_RED_FLAG_PATTERNS
    ):
        urgency = UrgencyLevel.EMERGENT_911
    elif URGENT_CUE_REGEX.search(lower):
        urgency = UrgencyLevel.URGENT
    elif is_vague:
        urgency = UrgencyLevel.UNKNOWN
    elif re.search(r"\b(?:greater\s+than|more\s+than|exceeds|above|\d+(?:\.\d+)?\s*(?:lb|pounds?|f\b))", lower):
        urgency = UrgencyLevel.SOON
    else:
        urgency = UrgencyLevel.ROUTINE

    return ExtractedWarningSignRecord(
        id=f"ws-{item_index}",
        symptom_description=symptom_description,
        action_to_take=action_to_take,
        urgency=urgency,
        confidence=0.72 if is_vague else 0.96,
        is_ambiguous=is_vague,
        evidence=_make_evidence(sent),
    )


def _classify_care_instruction_type(
    sent: SegmentedSentence,
) -> CareInstructionCategory:
    """Classify a care instruction into WOUND_CARE, DIET, ACTIVITY, REHABILITATION, or OTHER_CARE_INSTRUCTION."""
    if sent.canonical_section == SectionCategory.WOUND_CARE:
        return "WOUND_CARE"
    if sent.canonical_section == SectionCategory.DIET:
        return "DIET"
    lower = sent.normalized_text.lower()
    if re.search(r"\b(?:wound|incision|dressing|sutures?|staples?|bandage|surgical\s+site)\b", lower):
        return "WOUND_CARE"
    if re.search(r"\b(?:diet|sodium|salt|fluid\s+restriction|nutrition|carbohydrate|low-fat)\b", lower):
        return "DIET"
    if re.search(r"\b(?:rehab(?:ilitation)?|physical\s+therapy\s+exercises?|range[\s-]of[\s-]motion)\b", lower):
        return "REHABILITATION"
    if sent.canonical_section == SectionCategory.ACTIVITY or re.search(
        r"\b(?:lift(?:ing)?|ambulat(?:e|ion)|walk(?:ing)?|driv(?:e|ing)|stairs?|strenuous\s+activity|exercise)\b",
        lower,
    ):
        return "ACTIVITY"
    return "OTHER_CARE_INSTRUCTION"


def extract_discharge_intelligence(doc: IngestedDocument) -> DocumentIntelligenceResult:
    """Run evidence-first structured extraction across all segmented sentences and validate via Phase 3 validator."""
    medications: list[ExtractedMedicationRecord] = []
    follow_ups: list[ExtractedFollowUpRecord] = []
    tests: list[ExtractedTestRecord] = []
    referrals: list[ExtractedReferralRecord] = []
    warning_signs: list[ExtractedWarningSignRecord] = []
    care_instructions: list[ExtractedCareInstructionRecord] = []
    unresolved_items: list[str] = []
    warnings: list[str] = []

    if doc.suspicious_chars:
        warnings.append(
            f"Detected {len(doc.suspicious_chars)} suspicious invisible/control character(s) in raw document."
        )

    for sent in doc.sentences:
        cat = sent.canonical_section
        norm = sent.normalized_text.strip()
        lower = norm.lower()

        # Skip routine narrative sentences in Hospital Course that contain no clinical directives
        if cat == SectionCategory.HOSPITAL_COURSE and "stabilized" in lower and not (
            MEDICATION_VERB_START_RE.search(norm)
            or FOLLOWUP_CUE_RE.search(norm)
            or WARNING_CUE_RE.search(norm)
            or TEST_CUE_RE.search(norm)
        ):
            continue

        # 1. Medications section or explicit medication sentence
        if cat in (SectionCategory.DISCHARGE_MEDICATIONS, SectionCategory.MEDICATIONS) or (
            cat in (SectionCategory.HOSPITAL_COURSE, SectionCategory.INSTRUCTIONS, SectionCategory.UNKNOWN)
            and (MEDICATION_VERB_START_RE.search(norm) or GOVERNED_NEGATION_RE.search(norm))
            and not FOLLOWUP_CUE_RE.search(norm)
            and not WARNING_CUE_RE.search(norm)
            and not TEST_CUE_RE.search(norm)
            and not re.search(r"\b(?:incision|wound|dressing|lift|diet|sodium)\b", lower)
        ):
            med = _extract_medication_from_sentence(sent, len(medications) + 1)
            if med is not None:
                medications.append(med)
                if med.is_ambiguous:
                    unresolved_items.append(
                        f"{med.id}: {med.ambiguity_note or 'Ambiguous medication instruction'}"
                    )
                continue

        # 2. Warning signs / Return precautions
        if cat in (SectionCategory.WARNING_SIGNS, SectionCategory.WHEN_TO_SEEK_HELP) or (
            WARNING_CUE_RE.search(norm) and not FOLLOWUP_CUE_RE.search(norm)
        ):
            ws = _extract_warning_sign_from_sentence(sent, len(warning_signs) + 1)
            if ws is not None:
                warning_signs.append(ws)
                if ws.is_ambiguous:
                    unresolved_items.append(
                        f"{ws.id}: Vague symptom phrasing or missing action in warning sign."
                    )
                continue

        # 3. Diagnostic / Laboratory Tests
        if cat == SectionCategory.TESTS or (
            TEST_CUE_RE.search(norm)
            and not MEDICATION_VERB_START_RE.search(norm)
            and not FOLLOWUP_CUE_RE.search(norm)
        ):
            t_rec = _extract_test_from_sentence(sent, len(tests) + 1)
            if t_rec is not None:
                tests.append(t_rec)
                if t_rec.is_ambiguous:
                    unresolved_items.append(
                        f"{t_rec.id}: Test '{t_rec.test_name}' is missing an explicit timeframe."
                    )
                continue

        # 4. Referrals
        if cat == SectionCategory.REFERRALS or REFERRAL_CUE_RE.search(norm):
            ref = _extract_referral_from_sentence(sent, len(referrals) + 1)
            if ref is not None:
                referrals.append(ref)
                if ref.is_ambiguous:
                    unresolved_items.append(
                        f"{ref.id}: Referral is missing an explicit specialty or timeframe."
                    )
                continue

        # 5. Follow-up appointments
        if cat in (SectionCategory.FOLLOW_UP, SectionCategory.APPOINTMENTS) or FOLLOWUP_CUE_RE.search(norm):
            fu = _extract_followup_from_sentence(sent, len(follow_ups) + 1)
            if fu is not None:
                follow_ups.append(fu)
                if fu.is_ambiguous:
                    unresolved_items.append(
                        f"{fu.id}: {fu.ambiguity_note or 'Ambiguous follow-up instruction'}"
                    )
                continue

        # 6. Care instructions (Wound Care, Diet, Activity, Rehabilitation, General Instructions)
        if cat in (
            SectionCategory.WOUND_CARE,
            SectionCategory.DIET,
            SectionCategory.ACTIVITY,
            SectionCategory.INSTRUCTIONS,
            SectionCategory.UNKNOWN,
        ):
            ci_type = _classify_care_instruction_type(sent)
            m_tf = TIMEFRAME_RE.search(norm) or DURATION_RE.search(norm)
            tf_str = m_tf.group(1).strip() if m_tf else None
            care_instructions.append(
                ExtractedCareInstructionRecord(
                    id=f"care-{len(care_instructions) + 1}",
                    instruction_type=ci_type,
                    title=norm.rstrip("."),
                    detail=sent.source_text,
                    timeframe=tf_str,
                    confidence=0.94,
                    is_ambiguous=False,
                    evidence=_make_evidence(sent),
                )
            )

    # Assemble validator-compatible DischargeExtractionPayload
    val_meds: list[MedicationInstruction] = [
        MedicationInstruction(
            id=m.id,
            medication_name=m.medication_name,
            action=m.action,
            dosage=m.dosage,
            previous_dosage=m.previous_dosage,
            frequency=m.frequency,
            route=m.route,
            indication_or_note=m.indication_or_note,
            confidence=m.confidence,
            source_evidence=m.evidence.to_source_evidence(),
        )
        for m in medications
    ]

    val_followups: list[FollowUpInstruction] = [
        FollowUpInstruction(
            id=f.id,
            specialty=f.specialty,
            provider_name=f.provider_name,
            timeframe=f.timeframe,
            reason=f.reason,
            appointment_scheduled=f.appointment_scheduled,
            confidence=f.confidence,
            source_evidence=f.evidence.to_source_evidence(),
        )
        for f in follow_ups
    ]

    # Map referrals into FollowUpInstruction for unified safety validation & omission coverage
    for r in referrals:
        val_followups.append(
            FollowUpInstruction(
                id=r.id,
                specialty=r.specialty_or_service,
                provider_name=r.provider_name,
                timeframe=r.timeframe,
                reason=r.reason,
                confidence=r.confidence,
                source_evidence=r.evidence.to_source_evidence(),
            )
        )

    val_warnings: list[WarningSignInstruction] = [
        WarningSignInstruction(
            id=w.id,
            symptom_description=w.symptom_description,
            urgency=w.urgency,
            action_to_take=w.action_to_take,
            confidence=w.confidence,
            source_evidence=w.evidence.to_source_evidence(),
        )
        for w in warning_signs
    ]

    val_tasks: list[CareTaskItem] = []
    # Map outpatient tests into CareTaskItem (linked to first follow-up if available, else create a lab follow-up)
    for idx_t, t_item in enumerate(tests, start=1):
        if val_followups:
            linked_fu_id = val_followups[0].id
        else:
            lab_fu_id = f"fu-lab-{idx_t}"
            val_followups.append(
                FollowUpInstruction(
                    id=lab_fu_id,
                    specialty=t_item.test_name,
                    timeframe=t_item.timeframe,
                    reason=t_item.reason_or_note,
                    confidence=t_item.confidence,
                    source_evidence=t_item.evidence.to_source_evidence(),
                )
            )
            linked_fu_id = lab_fu_id
        val_tasks.append(
            CareTaskItem(
                id=t_item.id,
                category="LAB_TEST",
                title=t_item.evidence.source_text.rstrip("."),
                due_window=t_item.timeframe,
                linked_item_id=linked_fu_id,
                confidence=t_item.confidence,
                source_evidence=t_item.evidence.to_source_evidence(),
            )
        )

    total_extracted = (
        len(medications)
        + len(follow_ups)
        + len(tests)
        + len(referrals)
        + len(warning_signs)
        + len(care_instructions)
    )
    if total_extracted == 0:
        completeness_score = 0.0
    else:
        unambiguous_count = sum(
            1
            for item in (*medications, *follow_ups, *tests, *referrals, *warning_signs, *care_instructions)
            if not item.is_ambiguous
        )
        completeness_score = round(unambiguous_count / total_extracted, 4)

    validator_payload = DischargeExtractionPayload(
        case_id=doc.document_id,
        patient_identifier=doc.patient_identifier,
        raw_discharge_text=doc.raw_text,
        medications=val_meds,
        follow_ups=val_followups,
        warning_signs=val_warnings,
        tasks=val_tasks,
        provider_matches=[],
    )

    validation_report = validate_discharge_payload(validator_payload)

    return DocumentIntelligenceResult(
        document=doc,
        medications=medications,
        follow_ups=follow_ups,
        tests=tests,
        referrals=referrals,
        warning_signs=warning_signs,
        care_instructions=care_instructions,
        extraction_completeness_score=completeness_score,
        unresolved_or_ambiguous_items=unresolved_items,
        extraction_warnings=warnings,
        validator_payload=validation_report.validated_payload,
        validation_report=validation_report,
    )


def run_document_intelligence_pipeline(
    raw_text: str,
    document_id: Optional[str] = None,
    patient_identifier: Optional[str] = None,
    source_format: DocumentFormat = DocumentFormat.SYNTHETIC_SUMMARY,
) -> DocumentIntelligenceResult:
    """End-to-end Phase 4 pipeline: Ingest -> Normalize -> Sections -> Sentences -> Extract -> Validate."""
    ingested = ingest_discharge_document(
        raw_text=raw_text,
        document_id=document_id,
        patient_identifier=patient_identifier,
        source_format=source_format,
    )
    return extract_discharge_intelligence(ingested)
