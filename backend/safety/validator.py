"""Deterministic Safety Validation Engine for CAREBRIDGE.

Audits structured AI extraction payloads against the raw hospital discharge text
before any patient-facing rendering occurs. Never trusts raw LLM output.

Enforces:
- Field-level evidence verification (UNSUPPORTED_FIELD_VALUE)
- Claim-evidence relevance verification (MISSING_SOURCE_EVIDENCE)
- Omitted instruction & incomplete extraction detection (OMITTED_INSTRUCTION, INCOMPLETE_EXTRACTION)
- Negation-aware medication change & stop/start detection
- Brand/generic alias conflict detection & item-level conflict gating
- Qualifier-preserving timeframe normalization (1 week == 7 days; within != after)
- Duplicate item & duplicate ID detection (DUPLICATE_ITEM)
- Comprehensive unsupported provider claim detection (UNSUPPORTED_PROVIDER_CLAIM)
- Source character span verification (start_char:end_char)
- Patient release gating (is_patient_releasable)
"""

from __future__ import annotations

import re
from collections import defaultdict

from backend.schemas.models import (
    SPAN_NOT_IMPLEMENTED_NOTE,
    CareTaskItem,
    DischargeExtractionPayload,
    ExtractedItemBase,
    FollowUpInstruction,
    HazardCode,
    HazardSeverity,
    MedicationAction,
    MedicationInstruction,
    ReviewDecisionStatus,
    SafetyFlag,
    SafetyStatus,
    UrgencyLevel,
    ValidationReport,
    WarningSignInstruction,
)

CONFIDENCE_THRESHOLD: float = 0.80

UNKNOWN_TOKENS: frozenset[str] = frozenset(
    {
        "",
        "unknown",
        "unspecified",
        "tbd",
        "n/a",
        "na",
        "none",
        "not specified",
        "not stated",
        "not provided",
        "unclear",
        "?",
    }
)

VAGUE_SPECIALTY_TOKENS: frozenset[str] = frozenset(
    {
        "doctor",
        "clinic",
        "specialist",
        "physician",
        "provider",
        "hospital",
    }
)

# Small deterministic synthetic brand/generic alias map for demo conflict & evidence resolution.
# Not a comprehensive drug database.
SYNTHETIC_DRUG_ALIASES: dict[str, str] = {
    "prinivil": "lisinopril",
    "zestril": "lisinopril",
    "lisinopril": "lisinopril",
    "cozaar": "losartan",
    "losartan": "losartan",
    "diovan": "valsartan",
    "valsartan": "valsartan",
    "entresto": "sacubitril/valsartan",
    "sacubitril/valsartan": "sacubitril/valsartan",
    "lopressor": "metoprolol",
    "toprol": "metoprolol",
    "toprol-xl": "metoprolol",
    "metoprolol": "metoprolol",
    "coreg": "carvedilol",
    "carvedilol": "carvedilol",
    "lasix": "furosemide",
    "furosemide": "furosemide",
    "aldactone": "spironolactone",
    "spironolactone": "spironolactone",
    "eliquis": "apixaban",
    "apixaban": "apixaban",
    "xarelto": "rivaroxaban",
    "rivaroxaban": "rivaroxaban",
    "coumadin": "warfarin",
    "jantoven": "warfarin",
    "warfarin": "warfarin",
    "plavix": "clopidogrel",
    "clopidogrel": "clopidogrel",
    "ecotrin": "aspirin",
    "aspirin": "aspirin",
    "lipitor": "atorvastatin",
    "atorvastatin": "atorvastatin",
    "crestor": "rosuvastatin",
    "rosuvastatin": "rosuvastatin",
    "norvasc": "amlodipine",
    "amlodipine": "amlodipine",
    "synthroid": "levothyroxine",
    "levothyroxine": "levothyroxine",
    "glucophage": "metformin",
    "metformin": "metformin",
    "lanoxin": "digoxin",
    "digoxin": "digoxin",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "ibuprofen": "ibuprofen",
    "tylenol": "acetaminophen",
    "acetaminophen": "acetaminophen",
    "prilosec": "omeprazole",
    "omeprazole": "omeprazole",
    "potassium chloride": "potassium chloride",
    "klor-con": "potassium chloride",
}

ROUTE_EQUIVALENCE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"by mouth", "po", "p.o.", "oral", "orally"}),
    frozenset({"iv", "i.v.", "intravenous", "intravenously"}),
    frozenset({"sc", "subq", "subcutaneous", "subcutaneously", "sq"}),
    frozenset({"im", "i.m.", "intramuscular", "intramuscularly"}),
    frozenset({"topical", "topically"}),
    frozenset({"inhaled", "inhalation", "inhale"}),
    frozenset({"sl", "sublingual", "sublingually"}),
)

FREQUENCY_ABBREVIATION_MAP: dict[str, tuple[str, ...]] = {
    "once daily": ("once daily", "once a day", "daily", "every day", "qd", "q.d."),
    "daily": ("daily", "once daily", "once a day", "every day", "qd", "q.d."),
    "twice daily": ("twice daily", "twice a day", "two times daily", "2 times daily", "bid", "b.i.d."),
    "bid": ("twice daily", "twice a day", "two times daily", "bid", "b.i.d."),
    "three times daily": ("three times daily", "3 times daily", "tid", "t.i.d."),
    "tid": ("three times daily", "3 times daily", "tid", "t.i.d."),
    "four times daily": ("four times daily", "4 times daily", "qid", "q.i.d."),
    "qid": ("four times daily", "4 times daily", "qid", "q.i.d."),
    "at bedtime": ("at bedtime", "bedtime", "qhs", "hs"),
    "in the morning": ("in the morning", "morning", "qam", "every morning"),
}

AMBIGUOUS_MED_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bas\s+needed\s+if\s+needed\b",
        r"\bmaybe\b",
        r"\bpossibly\b",
        r"\bwhen\s+ready\b",
        r"\bif\s+you\s+feel\s+like\s+it\b",
        r"\bcheck\s+with\s+doctor\s+whether\b",
        r"\badjust\s+as\s+tolerated\b",
        r"\btake\s+some\b",
        r"\bas\s+directed\b",
        r"\buse\s+judgment\b",
        r"\bunclear\b",
        r"\bprn\s+or\s+scheduled\b",
    )
)

UNCERTAIN_NEGATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:unclear|unsure|not\s+sure|maybe|possibly|unknown)\s+(?:if|whether)\b",
        r"\bnot\s+clear\s+(?:if|whether)\b",
    )
)

NEGATED_MED_CHANGE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bno\s+changes?\s+to\s+(?:your\s+|the\s+|home\s+)?medications?(?:\s+is\s+required|\s+are\s+required|\s+needed)?\b",
        r"\bmedications?\s+remains?\s+unchanged\b",
        r"\bno\s+medication\s+changes?\b",
        r"\bwithout\s+(?:any\s+)?medication\s+changes?\b",
        r"\bdo\s+not\s+stop(?:\s+taking)?\b",
        r"\bdon'?t\s+stop(?:\s+taking)?\b",
        r"\bno\s+new\s+medications?\b",
    )
)

UNCLEAR_DOSAGE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\busual\s+dose\b",
        r"\bsame\s+dose\b",
        r"\bprevious\s+dose\b",
        r"\bhome\s+dose\b",
        r"\bsome\s+pills?\b",
        r"\ba\s+few\b",
        r"\b1\s*-\s*3\s+pills\b",
        r"\bvariable\b",
        r"\bunknown\b",
        r"\btbd\b",
    )
)

DOSE_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|meq|units?|iu|tablets?|tabs?|capsules?|caps?|puffs?|drops?|patches?|%)\b",
    re.IGNORECASE,
)

MED_CHANGE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bincreas(?:e|ed|ing)\b",
        r"\bdecreas(?:e|ed|ing)\b",
        r"\bchang(?:e|ed|ing)\s+(?:from|to|dose)\b",
        r"\breduc(?:e|ed|ing)\s+(?:to|dose|from)\b",
        r"\bswitch(?:ed|ing)?\s+(?:from|to)\b",
        r"\btaper(?:ed|ing)?\b",
        r"\bformerly\b",
        r"\bwas\s+taking\b",
        r"\btitrat(?:e|ed|ing)\b",
        r"\bdouble\s+the\s+dose\b",
        r"\bhalve\s+the\s+dose\b",
        r"\binstead\s+of\b",
    )
)

STOP_START_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bstop\s+taking\b",
        r"\bstop\b",
        r"\bdiscontinu(?:e|ed|ing)\b",
        r"\bdo\s+not\s+take\b",
        r"\bno\s+longer\s+tak(?:e|ing)\b",
        r"\bhold\s+(?:for|until)\b",
        r"\bhold\b",
        r"\bwithhold(?:ing)?\b",
        r"\bskip(?:ping)?\b",
        r"\bavoid(?:ing)?\b",
        r"\bstart\s+new\b",
        r"\bnewly\s+prescribed\b",
        r"\bbegin\s+taking\b",
    )
)

SENSITIVE_MED_OMISSION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bswitch(?:ed|ing)?\b",
        r"\bchang(?:e|ed|ing)\b",
        r"\bstop(?:ped|ping)?\b",
        r"\bdiscontinu(?:e|ed|ing)\b",
        r"\bhold(?:ing)?\b",
        r"\bwithhold(?:ing)?\b",
        r"\bstart(?:ed|ing)?\b",
        r"\bincreas(?:e|ed|ing)\b",
        r"\bdecreas(?:e|ed|ing)\b",
        r"\btaper(?:ed|ing)?\b",
        r"\bresum(?:e|ed|ing)\b",
        r"\breduc(?:e|ed|ing)\b",
        r"\bskip(?:ped|ping)?\b",
        r"\bavoid(?:ing)?\b",
        r"\bdo\s+not\s+take\b",
        r"\bno\s+longer\b",
        r"\binstead\s+of\b",
    )
)

UNCLEAR_SYMPTOM_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*feeling\s+worse\s*$",
        r"\bnot\s+feeling\s+right\b",
        r"\bfeeli?ng\s+unwell\b",
        r"\bbad\s+symptoms\b",
        r"\banything\s+unusual\b",
        r"\bif\s+concerned\b",
        r"\bgeneral\s+discomfort\b",
        r"\bfeeling\s+bad\b",
        r"\bget\s+sick\b",
    )
)

CRITICAL_RED_FLAG_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bchest\s+pain\b",
        r"\bshortness\s+of\s+breath\b",
        r"\bdifficulty\s+breathing\b",
        r"\bsyncope\b",
        r"\bfainting\b",
        r"\bpassed?\s+out\b",
        r"\bslurred\s+speech\b",
        r"\bconfusion\b",
        r"\buncontrolled\s+bleeding\b",
        r"\bcoughing\s+up\s+blood\b",
        r"\bhemoptysis\b",
        r"\bsuicid(?:al|e)\b",
        r"\bsevere\s+allergic\b",
        r"\banaphylaxis\b",
        r"\bswelling\s+of\s+(?:the\s+)?(?:face|lips|throat|tongue)\b",
        r"\bfever\s*(?:>|above|over)\s*101\b",
    )
)

UNSUPPORTED_PROVIDER_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bbest\s+match\b",
        r"\bbest\b",
        r"\brecommend(?:ed|s)?\b",
        r"\bpreferred\b",
        r"\btop\b",
        r"\bguarantee(?:d|s)?\b",
        r"\bsuitable\b",
        r"\bideal\b",
        r"\bhighly\s+rated\b",
        r"\bmost\s+qualified\b",
        r"\btrusted\b",
        r"\bleading\b",
        r"\bavailable\s+now\b",
        r"\baccepting\s+(?:new\s+)?patients\b",
        r"\bdefinitely\s+accepts\b",
        r"\binstant\s+booking\b",
        r"\bwill\s+cure\b",
        r"\balways\s+available\b",
    )
)

PROHIBITED_CLINICAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bi\s+diagnose\b",
        r"\byou\s+are\s+diagnosed\s+with\b",
        r"\byou\s+likely\s+have\b",
        r"\bthis\s+means\s+you\s+have\b",
        r"\bwe\s+recommend\s+starting\b",
        r"\bwe\s+prescribe\b",
        r"\bprescribing\s+you\b",
        r"\byou\s+should\s+switch\s+to\b",
        r"\binstead\s+of\s+your\s+prescribed\b",
        r"\bincrease\s+(?:your\s+)?dose\b",
        r"\bdecrease\s+(?:your\s+)?dose\b",
        r"\bchange\s+(?:your\s+)?dose\b",
        r"\bdouble\s+(?:your\s+|the\s+)?dose\b",
        r"\bstop\s+your\s+medication\s+if\s+you\s+feel\s+fine\b",
        r"\btake\s+ibuprofen\s+instead\b",
    )
)

WORD_NUMBER_MAP: dict[str, int] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
    "fourteen": 14,
    "twenty-four": 24,
    "twenty four": 24,
    "forty-eight": 48,
    "forty eight": 48,
    "seventy-two": 72,
    "seventy two": 72,
}

RECURRENCE_TIMEFRAME_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bdaily\b",
        r"\bweekly\b",
        r"\bmonthly\b",
        r"\bevery\s+(?:day|week|month|morning|evening|night|\d+\s*hours?)\b",
        r"\bonce\s+a\s+(?:day|week|month)\b",
        r"\btwice\s+a\s+(?:day|week|month)\b",
        r"\bas\s+needed\b",
        r"\bprn\b",
        r"^\s*soon\s*$",
        r"^\s*later\s*$",
        r"\bwhen\s+convenient\b",
    )
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_missing_or_unknown(value: str | None) -> bool:
    if value is None:
        return True
    return _normalize_whitespace(value) in UNKNOWN_TOKENS


def _strip_negated_clauses(text: str) -> str:
    """Remove explicitly negated medication-change/stop clauses so they do not false-trigger."""
    cleaned = text
    for pattern in NEGATED_MED_CHANGE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return _normalize_whitespace(cleaned)


def resolve_canonical_medication_name(name: str) -> str:
    """Resolve a medication name to its canonical generic identity using the synthetic alias map."""
    norm = _normalize_whitespace(name)
    # Strip parenthetical alias if present e.g. "prinivil (lisinopril)" -> "prinivil"
    base = re.sub(r"\s*\([^)]*\)\s*", " ", norm).strip()
    return SYNTHETIC_DRUG_ALIASES.get(base, SYNTHETIC_DRUG_ALIASES.get(norm, base))


def get_medication_equivalents(name: str) -> set[str]:
    """Return all known lowercase brand/generic aliases for a medication name."""
    norm = _normalize_whitespace(name)
    canonical = resolve_canonical_medication_name(name)
    equivalents = {norm, canonical}
    for alias, target in SYNTHETIC_DRUG_ALIASES.items():
        if target == canonical:
            equivalents.add(alias)
    return equivalents


def normalize_timeframe(raw_timeframe: str | None) -> str | None:
    """Normalize a follow-up timeframe into 'QUALIFIER:AMOUNT:UNIT' while preserving qualifiers.

    Rules:
    - 1 week == 7 days, 2 weeks == 14 days
    - 24 hours == 1 day, 48 hours == 2 days
    - Preserves qualifier: 'within 7 days' ('WITHIN:7:DAYS') != 'after 7 days' ('AFTER:7:DAYS')
    - Does NOT equate months with days ('1 month' -> '1:MONTHS' != '30:DAYS')
    - Rejects recurrence strings ('daily', 'weekly') or vague terms ('soon') -> returns None
    """
    if _is_missing_or_unknown(raw_timeframe):
        return None
    assert raw_timeframe is not None
    norm = _normalize_whitespace(raw_timeframe)

    for rec_pat in RECURRENCE_TIMEFRAME_PATTERNS:
        if rec_pat.search(norm):
            return None

    # Determine qualifier
    qualifier = "IN"
    if re.match(r"^(?:within|no\s+later\s+than|by|in\s+under|up\s+to)\b", norm):
        qualifier = "WITHIN"
    elif re.match(r"^(?:after|at\s+least|post|following|more\s+than)\b", norm):
        qualifier = "AFTER"
    elif re.match(r"^(?:in|at|around|for)\b", norm):
        qualifier = "IN"

    # Replace multi-word numbers first, then single-word numbers
    working = norm
    for word_num in ("twenty-four", "twenty four", "forty-eight", "forty eight", "seventy-two", "seventy two"):
        working = re.sub(rf"\b{word_num}\b", str(WORD_NUMBER_MAP[word_num]), working)
    for word_num, digit in WORD_NUMBER_MAP.items():
        if " " not in word_num and "-" not in word_num:
            working = re.sub(
                rf"\b{word_num}\b(?=\s*(?:hours?|hrs?|days?|weeks?|wks?|months?|mos?|years?|yrs?)\b)",
                str(digit),
                working,
            )

    # Match range e.g. "1-2 weeks" or "7 to 10 days" or single number "7 days"
    m = re.search(
        r"\b(\d+)(?:\s*(?:-|to)\s*(\d+))?\s*(hours?|hrs?|hr|days?|d|weeks?|wks?|wk|months?|mos?|mo|years?|yrs?|yr)\b",
        working,
    )
    if not m:
        return None

    start_val = int(m.group(1))
    end_val = int(m.group(2)) if m.group(2) is not None else None
    raw_unit = m.group(3).lower()

    if start_val <= 0 or (end_val is not None and end_val < start_val):
        return None

    if raw_unit.startswith(("hour", "hr")):
        if start_val % 24 == 0 and (end_val is None or end_val % 24 == 0):
            start_norm = start_val // 24
            end_norm = (end_val // 24) if end_val is not None else None
            unit_norm = "DAYS"
        else:
            start_norm = start_val
            end_norm = end_val
            unit_norm = "HOURS"
    elif raw_unit.startswith(("week", "wk")):
        start_norm = start_val * 7
        end_norm = (end_val * 7) if end_val is not None else None
        unit_norm = "DAYS"
    elif raw_unit.startswith(("day", "d")):
        start_norm = start_val
        end_norm = end_val
        unit_norm = "DAYS"
    elif raw_unit.startswith(("month", "mo")):
        # Never blindly equate months with days
        start_norm = start_val
        end_norm = end_val
        unit_norm = "MONTHS"
    else:
        start_norm = start_val
        end_norm = end_val
        unit_norm = "YEARS"

    amount_str = f"{start_norm}-{end_norm}" if end_norm is not None else str(start_norm)
    return f"{qualifier}:{amount_str}:{unit_norm}"


def is_patient_releasable(
    target: ValidationReport | DischargeExtractionPayload,
) -> bool:
    """Return True ONLY when the entire validated payload is safe and all items are approved."""
    if isinstance(target, ValidationReport):
        payload = target.validated_payload
        if target.overall_safety_status != SafetyStatus.SAFE_ORGANIZED:
            return False
        if target.requires_human_review or target.total_flags > 0:
            return False
    else:
        payload = target
        if payload.overall_safety_status != SafetyStatus.SAFE_ORGANIZED:
            return False
        if payload.requires_human_review or len(payload.global_safety_flags) > 0:
            return False

    all_items: list[ExtractedItemBase] = [
        *payload.medications,
        *payload.follow_ups,
        *payload.warning_signs,
        *payload.tasks,
    ]
    if not all_items:
        return False

    for item in all_items:
        if item.safety_status != SafetyStatus.SAFE_ORGANIZED:
            return False
        if item.review_status not in (
            ReviewDecisionStatus.AUTO_APPROVED,
            ReviewDecisionStatus.HUMAN_RESOLVED,
        ):
            return False
        if len(item.safety_flags) > 0:
            return False
        if item.source_evidence is None or _is_missing_or_unknown(
            item.source_evidence.source_text
        ):
            return False

    return True


class SafetyValidator:
    """Deterministic clinical safety validator for CAREBRIDGE extraction payloads."""

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> None:
        self.confidence_threshold = confidence_threshold

    def validate(self, payload: DischargeExtractionPayload) -> ValidationReport:
        """Audit a DischargeExtractionPayload and return a full ValidationReport."""
        audited = payload.model_copy(deep=True)
        audited.global_safety_flags = []
        raw_doc = audited.raw_discharge_text
        normalized_doc = _normalize_whitespace(raw_doc)

        # 1. Validate individual items and field-level evidence support
        for med in audited.medications:
            med.safety_flags = []
            ev_verified = self._validate_base_item(med, raw_doc, normalized_doc)
            self._validate_medication(med, normalized_doc, ev_verified)
            self._finalize_item_status(med)

        for follow_up in audited.follow_ups:
            follow_up.safety_flags = []
            ev_verified = self._validate_base_item(follow_up, raw_doc, normalized_doc)
            self._validate_follow_up(follow_up, ev_verified)
            self._finalize_item_status(follow_up)

        for warning in audited.warning_signs:
            warning.safety_flags = []
            ev_verified = self._validate_base_item(warning, raw_doc, normalized_doc)
            self._validate_warning_sign(warning, ev_verified)
            self._finalize_item_status(warning)

        for task in audited.tasks:
            task.safety_flags = []
            ev_verified = self._validate_base_item(task, raw_doc, normalized_doc)
            self._validate_task_item(task, ev_verified)
            self._finalize_item_status(task)

        # 2. Detect duplicate items across all extracted lists
        self._detect_duplicate_items(audited)

        # 3. Cross-item conflict detection (medications & follow-ups) with item-level gating
        self._detect_cross_item_conflicts(audited)

        # 4. Detect omitted sensitive instructions and empty extractions
        self._detect_omitted_instructions(audited, raw_doc)

        # 5. Validate synthetic provider matches for unsupported claims
        self._validate_provider_matches(audited)

        # 6. Roll up overall status, global gating, and patient releasability
        all_flags: list[SafetyFlag] = list(audited.global_safety_flags)
        all_items: list[ExtractedItemBase] = [
            *audited.medications,
            *audited.follow_ups,
            *audited.warning_signs,
            *audited.tasks,
        ]
        for item in all_items:
            all_flags.extend(item.safety_flags)

        has_conflict = any(
            f.code == HazardCode.CONFLICTING_INSTRUCTIONS for f in all_flags
        ) or any(
            item.safety_status == SafetyStatus.CONFLICT_DETECTED for item in all_items
        )

        needs_review = has_conflict or any(f.requires_human_review for f in all_flags)

        # Section D: Global safety failures or unsupported clinical content block auto-approval for the entire payload
        has_global_or_unsupported_blocker = any(
            f.code
            in (
                HazardCode.INCOMPLETE_EXTRACTION,
                HazardCode.OMITTED_INSTRUCTION,
                HazardCode.UNSUPPORTED_FIELD_VALUE,
                HazardCode.PROHIBITED_CLINICAL_ACTION,
                HazardCode.UNSUPPORTED_PROVIDER_CLAIM,
                HazardCode.DUPLICATE_ITEM,
            )
            for f in all_flags
        )
        if has_global_or_unsupported_blocker:
            for item in all_items:
                if item.review_status == ReviewDecisionStatus.AUTO_APPROVED:
                    item.review_status = ReviewDecisionStatus.PENDING_HUMAN_REVIEW

        if has_conflict:
            audited.overall_safety_status = SafetyStatus.CONFLICT_DETECTED
        elif needs_review:
            audited.overall_safety_status = SafetyStatus.NEEDS_REVIEW
        else:
            audited.overall_safety_status = SafetyStatus.SAFE_ORGANIZED

        audited.requires_human_review = needs_review

        unique_codes: list[HazardCode] = []
        for flag in all_flags:
            if flag.code not in unique_codes:
                unique_codes.append(flag.code)

        report = ValidationReport(
            case_id=audited.case_id,
            overall_safety_status=audited.overall_safety_status,
            requires_human_review=audited.requires_human_review,
            patient_releasable=False,
            total_flags=len(all_flags),
            hazard_codes_detected=unique_codes,
            validated_payload=audited,
        )
        report.patient_releasable = is_patient_releasable(report)
        return report

    def _validate_base_item(
        self, item: ExtractedItemBase, raw_doc: str, normalized_doc: str
    ) -> bool:
        """Verify source evidence presence, verbatim substring match, character spans, confidence, and free-text safety."""
        evidence_verified = False

        if item.source_evidence is None or _is_missing_or_unknown(
            item.source_evidence.source_text
        ):
            item.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.MISSING_SOURCE_EVIDENCE,
                    severity=HazardSeverity.CRITICAL,
                    message=f"Item '{item.id}' is missing verbatim source evidence from the discharge summary.",
                    item_id=item.id,
                    requires_human_review=True,
                )
            )
        else:
            ev = item.source_evidence
            # Do not show source_page as verified UI evidence until document processor computes real page mapping
            ev.source_page_verified = False

            normalized_quote = _normalize_whitespace(ev.source_text)
            if normalized_quote not in normalized_doc:
                item.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.MISSING_SOURCE_EVIDENCE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Source quote for item '{item.id}' could not be verified as a "
                            "verbatim substring of the discharge document."
                        ),
                        item_id=item.id,
                        requires_human_review=True,
                        evidence_snippet=ev.source_text,
                    )
                )
            else:
                evidence_verified = True

            # Section J: Verify start_char / end_char span if present
            if ev.start_char is not None or ev.end_char is not None:
                if (
                    ev.start_char is None
                    or ev.end_char is None
                    or ev.start_char < 0
                    or ev.end_char > len(raw_doc)
                    or ev.end_char <= ev.start_char
                ):
                    evidence_verified = False
                    ev.span_status_note = "Invalid character span bounds."
                    item.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.MISSING_SOURCE_EVIDENCE,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Source character span [{ev.start_char}:{ev.end_char}] for '{item.id}' "
                                "is incomplete or out of bounds."
                            ),
                            item_id=item.id,
                            requires_human_review=True,
                            evidence_snippet=ev.source_text,
                        )
                    )
                else:
                    sliced_norm = _normalize_whitespace(
                        raw_doc[ev.start_char : ev.end_char]
                    )
                    if sliced_norm != normalized_quote:
                        evidence_verified = False
                        ev.span_status_note = "Character span mismatch against source_text."
                        item.safety_flags.append(
                            SafetyFlag(
                                code=HazardCode.MISSING_SOURCE_EVIDENCE,
                                severity=HazardSeverity.CRITICAL,
                                message=(
                                    f"raw_discharge_text[{ev.start_char}:{ev.end_char}] does not "
                                    f"match source_text for item '{item.id}'."
                                ),
                                item_id=item.id,
                                requires_human_review=True,
                                evidence_snippet=ev.source_text,
                            )
                        )
                    else:
                        ev.span_status_note = "Verified character span."
            else:
                ev.span_status_note = SPAN_NOT_IMPLEMENTED_NOTE

        # Check extraction confidence
        if item.confidence < self.confidence_threshold:
            item.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.LOW_EXTRACTION_CONFIDENCE,
                    severity=HazardSeverity.HIGH,
                    message=(
                        f"Extraction confidence ({item.confidence:.2f}) is below safety "
                        f"threshold ({self.confidence_threshold:.2f})."
                    ),
                    item_id=item.id,
                    requires_human_review=True,
                )
            )

        # Check patient_friendly_summary for prohibited clinical advice and unsupported clinical details
        if item.patient_friendly_summary:
            summary_norm = _normalize_whitespace(item.patient_friendly_summary)
            ev_norm = (
                _normalize_whitespace(item.source_evidence.source_text)
                if item.source_evidence is not None and evidence_verified
                else ""
            )

            # Check prohibited clinical patterns (unless verbatim in verified source evidence)
            for pattern in PROHIBITED_CLINICAL_PATTERNS:
                if pattern.search(summary_norm) and not (
                    ev_norm and pattern.search(ev_norm)
                ):
                    item.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.PROHIBITED_CLINICAL_ACTION,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Patient-friendly summary for '{item.id}' introduces a prohibited "
                                "clinical action, diagnosis, or dose modification not in source evidence."
                            ),
                            item_id=item.id,
                            requires_human_review=True,
                            evidence_snippet=item.patient_friendly_summary,
                        )
                    )
                    break

            # Verify all dose-like tokens in patient_friendly_summary are supported by item's source_evidence
            if ev_norm:
                self._verify_free_text_against_evidence(
                    item=item,
                    free_text=item.patient_friendly_summary,
                    field_label="patient_friendly_summary",
                    ev_norm=ev_norm,
                )

        return evidence_verified

    def _verify_free_text_against_evidence(
        self,
        item: ExtractedItemBase,
        free_text: str,
        field_label: str,
        ev_norm: str,
    ) -> None:
        """Ensure generated or extracted free text does not introduce unsupported doses, drugs, routes, or frequencies."""
        text_norm = _normalize_whitespace(free_text)
        ev_compact = re.sub(r"\s+", "", ev_norm)

        # 1. Check dose-like tokens (e.g., '75 mg', '20 mg', '5 ml')
        for match in DOSE_TOKEN_PATTERN.finditer(text_norm):
            dose_str = _normalize_whitespace(match.group(0))
            dose_compact = re.sub(r"\s+", "", dose_str)
            if dose_str not in ev_norm and dose_compact not in ev_compact:
                item.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Field '{field_label}' on '{item.id}' contains dosage '{match.group(0)}' "
                            "not supported by the item's source evidence."
                        ),
                        item_id=item.id,
                        requires_human_review=True,
                        evidence_snippet=free_text,
                    )
                )

        # 2. Check known drug names in free text against item's source evidence
        for drug_alias in SYNTHETIC_DRUG_ALIASES:
            if re.search(rf"\b{re.escape(drug_alias)}\b", text_norm):
                equivalents = get_medication_equivalents(drug_alias)
                if not any(
                    re.search(rf"\b{re.escape(eq)}\b", ev_norm) for eq in equivalents
                ):
                    item.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Field '{field_label}' on '{item.id}' references medication "
                                f"'{drug_alias}' which is not supported by the item's source evidence."
                            ),
                            item_id=item.id,
                            requires_human_review=True,
                            evidence_snippet=free_text,
                        )
                    )

        # 3. Check routes in free text against item's source evidence
        for group in ROUTE_EQUIVALENCE_GROUPS:
            if any(re.search(rf"\b{re.escape(r)}\b", text_norm) for r in group):
                if not any(re.search(rf"\b{re.escape(r)}\b", ev_norm) for r in group):
                    item.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.HIGH,
                            message=(
                                f"Field '{field_label}' on '{item.id}' introduces a route of "
                                "administration not supported by the item's source evidence."
                            ),
                            item_id=item.id,
                            requires_human_review=True,
                            evidence_snippet=free_text,
                        )
                    )

        # 4. Check explicit frequency phrases in free text against item's source evidence
        for freq_key, eq_tuple in FREQUENCY_ABBREVIATION_MAP.items():
            if re.search(rf"\b{re.escape(freq_key)}\b", text_norm):
                if not any(re.search(rf"\b{re.escape(eq)}\b", ev_norm) for eq in eq_tuple):
                    item.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.HIGH,
                            message=(
                                f"Field '{field_label}' on '{item.id}' introduces frequency "
                                f"'{freq_key}' not supported by the item's source evidence."
                            ),
                            item_id=item.id,
                            requires_human_review=True,
                            evidence_snippet=free_text,
                        )
                    )

    def _validate_medication(
        self,
        med: MedicationInstruction,
        normalized_doc: str,
        evidence_verified: bool,
    ) -> None:
        """Audit a MedicationInstruction for evidence support, changes, stop/start, unclear dose, and ambiguity."""
        source_quote = (
            med.source_evidence.source_text if med.source_evidence is not None else ""
        )
        ev_norm = _normalize_whitespace(source_quote) if evidence_verified else ""
        ev_stripped_negation = _strip_negated_clauses(ev_norm) if ev_norm else ""

        # Check for uncertain negation in source quote
        if ev_norm and any(p.search(ev_norm) for p in UNCERTAIN_NEGATION_PATTERNS):
            med.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.AMBIGUOUS_MEDICATION_INSTRUCTION,
                    severity=HazardSeverity.HIGH,
                    message=(
                        f"Medication '{med.medication_name}' contains uncertain or ambiguous "
                        "negation requiring human clinical review."
                    ),
                    item_id=med.id,
                    requires_human_review=True,
                    evidence_snippet=source_quote or None,
                )
            )

        # Section A & B: Verify that source_evidence supports the medication claim and every non-null field
        med_supported_by_evidence = False
        if evidence_verified and ev_norm:
            med_equivalents = get_medication_equivalents(med.medication_name)
            med_supported_by_evidence = any(
                re.search(rf"\b{re.escape(eq)}\b", ev_norm) for eq in med_equivalents
            )
            if not med_supported_by_evidence:
                med.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.MISSING_SOURCE_EVIDENCE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Source evidence for '{med.id}' does not mention or support "
                            f"medication '{med.medication_name}'."
                        ),
                        item_id=med.id,
                        requires_human_review=True,
                        evidence_snippet=source_quote,
                    )
                )
                med.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Extracted medication_name '{med.medication_name}' is not supported "
                            "by the item's source_evidence."
                        ),
                        item_id=med.id,
                        requires_human_review=True,
                        evidence_snippet=source_quote,
                    )
                )

            # Verify dosage support in source_evidence
            if not _is_missing_or_unknown(med.dosage):
                assert med.dosage is not None
                dose_norm = _normalize_whitespace(med.dosage)
                dose_compact = re.sub(r"\s+", "", dose_norm)
                ev_compact = re.sub(r"\s+", "", ev_norm)
                if dose_norm not in ev_norm and dose_compact not in ev_compact:
                    med.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Extracted dosage '{med.dosage}' for '{med.medication_name}' "
                                "is not supported by the item's source evidence."
                            ),
                            item_id=med.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )

            # Verify previous_dosage support in source_evidence
            if not _is_missing_or_unknown(med.previous_dosage):
                assert med.previous_dosage is not None
                prev_norm = _normalize_whitespace(med.previous_dosage)
                prev_compact = re.sub(r"\s+", "", prev_norm)
                ev_compact = re.sub(r"\s+", "", ev_norm)
                if prev_norm not in ev_norm and prev_compact not in ev_compact:
                    med.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Extracted previous_dosage '{med.previous_dosage}' for "
                                f"'{med.medication_name}' is not supported by source evidence."
                            ),
                            item_id=med.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )

            # Verify route support in source_evidence
            if not _is_missing_or_unknown(med.route):
                assert med.route is not None
                route_norm = _normalize_whitespace(med.route)
                route_supported = route_norm in ev_norm
                if not route_supported:
                    for group in ROUTE_EQUIVALENCE_GROUPS:
                        if route_norm in group and any(
                            re.search(rf"\b{re.escape(candidate)}\b", ev_norm)
                            for candidate in group
                        ):
                            route_supported = True
                            break
                if not route_supported:
                    med.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.HIGH,
                            message=(
                                f"Extracted route '{med.route}' for '{med.medication_name}' "
                                "is not supported by the item's source evidence."
                            ),
                            item_id=med.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )

            # Verify frequency support in source_evidence
            if not _is_missing_or_unknown(med.frequency):
                assert med.frequency is not None
                freq_norm = _normalize_whitespace(med.frequency)
                freq_supported = freq_norm in ev_norm
                if not freq_supported and freq_norm in FREQUENCY_ABBREVIATION_MAP:
                    freq_supported = any(
                        re.search(rf"\b{re.escape(eq)}\b", ev_norm)
                        for eq in FREQUENCY_ABBREVIATION_MAP[freq_norm]
                    )
                if not freq_supported:
                    med.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.HIGH,
                            message=(
                                f"Extracted frequency '{med.frequency}' for '{med.medication_name}' "
                                "is not supported by the item's source evidence."
                            ),
                            item_id=med.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )

            # Verify indication_or_note support in source_evidence
            if not _is_missing_or_unknown(med.indication_or_note):
                assert med.indication_or_note is not None
                note_norm = _normalize_whitespace(med.indication_or_note)
                # Check if note introduces a prohibited or unsupported clinical modification
                has_unsupported_mod = (
                    any(p.search(note_norm) for p in PROHIBITED_CLINICAL_PATTERNS)
                    or any(p.search(note_norm) for p in MED_CHANGE_PATTERNS)
                    or any(p.search(note_norm) for p in STOP_START_PATTERNS)
                ) and note_norm not in ev_norm
                if has_unsupported_mod:
                    med.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.PROHIBITED_CLINICAL_ACTION,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Field 'indication_or_note' on '{med.id}' introduces an "
                                f"unsupported clinical action ('{med.indication_or_note}') not in source evidence."
                            ),
                            item_id=med.id,
                            requires_human_review=True,
                            evidence_snippet=med.indication_or_note,
                        )
                    )
                if note_norm not in ev_norm:
                    med.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.HIGH,
                            message=(
                                f"Field 'indication_or_note' ('{med.indication_or_note}') for "
                                f"'{med.medication_name}' is not supported by source evidence."
                            ),
                            item_id=med.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )

        # 1. Stop / Start / Hold medication detection (supported by source evidence or action)
        source_has_stop_start = bool(
            ev_stripped_negation
            and any(p.search(ev_stripped_negation) for p in STOP_START_PATTERNS)
        )
        is_stop_start = (
            med.action
            in (
                MedicationAction.STOP_DISCONTINUE,
                MedicationAction.NEW_START,
                MedicationAction.HOLD_TEMPORARY,
            )
            or source_has_stop_start
        )
        if is_stop_start:
            med.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.STOP_START_MEDICATION,
                    severity=HazardSeverity.HIGH,
                    message=(
                        f"Medication '{med.medication_name}' involves starting, stopping, "
                        f"or holding therapy ({med.action.value}); requires verification."
                    ),
                    item_id=med.id,
                    requires_human_review=True,
                    evidence_snippet=source_quote or None,
                )
            )

        # 2. Medication change detection (Section I: ONLY when supported by source evidence)
        source_has_dose_change = bool(
            ev_stripped_negation
            and any(p.search(ev_stripped_negation) for p in MED_CHANGE_PATTERNS)
        )
        prev_dose_supported = bool(
            not _is_missing_or_unknown(med.previous_dosage)
            and ev_norm
            and _normalize_whitespace(med.previous_dosage or "") in ev_norm
        )
        if source_has_dose_change or (
            med.action == MedicationAction.DOSE_CHANGE and prev_dose_supported
        ):
            med.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.MEDICATION_CHANGE_DETECTED,
                    severity=HazardSeverity.HIGH,
                    message=(
                        f"Dose or regimen modification detected for '{med.medication_name}'. "
                        "Verify transition against pre-admission medication list."
                    ),
                    item_id=med.id,
                    requires_human_review=True,
                    evidence_snippet=source_quote or None,
                )
            )
        elif med.action == MedicationAction.DOSE_CHANGE and not source_has_dose_change:
            # Action claims DOSE_CHANGE without source evidence support -> prohibited/unsupported
            med.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.PROHIBITED_CLINICAL_ACTION,
                    severity=HazardSeverity.CRITICAL,
                    message=(
                        f"Medication '{med.medication_name}' specifies action DOSE_CHANGE "
                        "without supporting dose-change evidence in the discharge text."
                    ),
                    item_id=med.id,
                    requires_human_review=True,
                    evidence_snippet=source_quote or None,
                )
            )
            med.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                    severity=HazardSeverity.CRITICAL,
                    message=f"Unsupported action 'DOSE_CHANGE' for '{med.medication_name}'.",
                    item_id=med.id,
                    requires_human_review=True,
                    evidence_snippet=source_quote or None,
                )
            )

        # 3. Unclear dosage detection (required for active medications)
        if med.action != MedicationAction.STOP_DISCONTINUE:
            has_unclear_dose = (
                _is_missing_or_unknown(med.dosage)
                or any(
                    p.search(med.dosage or "") or p.search(source_quote)
                    for p in UNCLEAR_DOSAGE_PATTERNS
                )
                or (
                    med.dosage is not None
                    and not DOSE_TOKEN_PATTERN.search(med.dosage)
                )
            )
            if has_unclear_dose:
                med.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.UNCLEAR_DOSAGE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Dosage for '{med.medication_name}' is missing, unquantified, "
                            f"or unclear ({med.dosage!r}). Do not guess dosage."
                        ),
                        item_id=med.id,
                        requires_human_review=True,
                        evidence_snippet=source_quote or None,
                    )
                )

        # 4. Ambiguous medication instruction detection
        combined_text = " ".join(
            filter(
                None,
                [
                    source_quote,
                    med.dosage,
                    med.frequency,
                    med.indication_or_note,
                ],
            )
        )
        has_ambiguous_instruction = (
            med.action == MedicationAction.UNKNOWN
            or (
                med.action != MedicationAction.STOP_DISCONTINUE
                and _is_missing_or_unknown(med.frequency)
            )
            or any(p.search(combined_text) for p in AMBIGUOUS_MED_PATTERNS)
        )
        if has_ambiguous_instruction:
            med.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.AMBIGUOUS_MEDICATION_INSTRUCTION,
                    severity=HazardSeverity.HIGH,
                    message=(
                        f"Medication instruction for '{med.medication_name}' has an "
                        "ambiguous action, schedule, or administration condition."
                    ),
                    item_id=med.id,
                    requires_human_review=True,
                    evidence_snippet=source_quote or None,
                )
            )

        # 5. Intra-document silent conflict check (including brand/generic aliases and negation stripping)
        doc_without_negations = _strip_negated_clauses(normalized_doc)
        if med.action == MedicationAction.CONTINUE:
            for alias in get_medication_equivalents(med.medication_name):
                alias_esc = re.escape(alias)
                contradictory_stop = re.search(
                    rf"\b(?:stop(?:\s+taking)?|discontinu(?:e|ed)|do\s+not\s+take|hold|withhold)\s+{alias_esc}\b",
                    doc_without_negations,
                    re.IGNORECASE,
                )
                if contradictory_stop:
                    med.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.CONFLICTING_INSTRUCTIONS,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Conflicting instructions detected for '{med.medication_name}' "
                                f"(alias '{alias}'): item marked CONTINUE but discharge text also "
                                "contains a STOP/DISCONTINUE/HOLD directive."
                            ),
                            item_id=med.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote or None,
                        )
                    )
                    break

    def _validate_follow_up(
        self, follow_up: FollowUpInstruction, evidence_verified: bool
    ) -> None:
        """Audit a FollowUpInstruction for missing/ambiguous timeframe, specialty, provider claims, and field evidence."""
        source_quote = (
            follow_up.source_evidence.source_text
            if follow_up.source_evidence is not None
            else None
        )
        ev_norm = (
            _normalize_whitespace(source_quote)
            if evidence_verified and source_quote
            else ""
        )

        # Section A & B: Verify source_evidence actually supports a follow-up claim and all non-null fields
        if evidence_verified and ev_norm:
            has_followup_anchor = bool(
                re.search(
                    r"\b(?:follow[\s-]?up|see|visit|appointment|clinic|refer(?:ral|red)?|consult|dr\.|primary\s+care|cardiology|nephrology|neurology|pulmonology|oncology|endocrinology|gastroenterology|orthopedics|urology|surgery)\b",
                    ev_norm,
                )
            )
            if not has_followup_anchor:
                follow_up.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.MISSING_SOURCE_EVIDENCE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Source evidence for follow-up '{follow_up.id}' does not contain "
                            "any follow-up, referral, or appointment instruction."
                        ),
                        item_id=follow_up.id,
                        requires_human_review=True,
                        evidence_snippet=source_quote,
                    )
                )

            if not _is_missing_or_unknown(follow_up.specialty):
                assert follow_up.specialty is not None
                spec_norm = _normalize_whitespace(follow_up.specialty)
                if spec_norm not in ev_norm:
                    follow_up.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.HIGH,
                            message=(
                                f"Extracted specialty '{follow_up.specialty}' on '{follow_up.id}' "
                                "is not supported by the item's source evidence."
                            ),
                            item_id=follow_up.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )

            if not _is_missing_or_unknown(follow_up.timeframe):
                assert follow_up.timeframe is not None
                tf_norm = _normalize_whitespace(follow_up.timeframe)
                if tf_norm not in ev_norm:
                    follow_up.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.HIGH,
                            message=(
                                f"Extracted timeframe '{follow_up.timeframe}' on '{follow_up.id}' "
                                "is not supported by the item's source evidence."
                            ),
                            item_id=follow_up.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )

            if not _is_missing_or_unknown(follow_up.provider_name):
                assert follow_up.provider_name is not None
                prov_norm = _normalize_whitespace(follow_up.provider_name)
                if prov_norm not in ev_norm:
                    follow_up.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.HIGH,
                            message=(
                                f"Extracted provider_name '{follow_up.provider_name}' on '{follow_up.id}' "
                                "is not supported by the item's source evidence."
                            ),
                            item_id=follow_up.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )

            if not _is_missing_or_unknown(follow_up.reason):
                assert follow_up.reason is not None
                reason_norm = _normalize_whitespace(follow_up.reason)
                if reason_norm not in ev_norm:
                    follow_up.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.HIGH,
                            message=(
                                f"Extracted reason '{follow_up.reason}' on '{follow_up.id}' "
                                "is not supported by the item's source evidence."
                            ),
                            item_id=follow_up.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )

        # 1. Missing or unparseable/ambiguous follow-up timeframe
        canonical_tf = normalize_timeframe(follow_up.timeframe)
        if _is_missing_or_unknown(follow_up.timeframe) or canonical_tf is None:
            follow_up.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.MISSING_FOLLOWUP_TIMEFRAME,
                    severity=HazardSeverity.HIGH,
                    message=(
                        f"Follow-up instruction '{follow_up.id}' has a missing or ambiguous "
                        f"timeframe ({follow_up.timeframe!r}). Timeframe requires human review."
                    ),
                    item_id=follow_up.id,
                    requires_human_review=True,
                    evidence_snippet=source_quote,
                )
            )

        # 2. Missing or overly generic specialty
        if _is_missing_or_unknown(follow_up.specialty) or (
            follow_up.specialty is not None
            and _normalize_whitespace(follow_up.specialty) in VAGUE_SPECIALTY_TOKENS
        ):
            follow_up.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.MISSING_SPECIALTY,
                    severity=HazardSeverity.HIGH,
                    message=(
                        f"Follow-up instruction '{follow_up.id}' is missing a specific "
                        "clinical specialty."
                    ),
                    item_id=follow_up.id,
                    requires_human_review=True,
                    evidence_snippet=source_quote,
                )
            )

        # 3. Unsupported provider claims on follow-up item
        claim_candidates = " ".join(
            filter(
                None,
                [
                    follow_up.provider_claim_note,
                    follow_up.patient_friendly_summary,
                ],
            )
        )
        if claim_candidates and any(
            p.search(claim_candidates) for p in UNSUPPORTED_PROVIDER_CLAIM_PATTERNS
        ):
            follow_up.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.UNSUPPORTED_PROVIDER_CLAIM,
                    severity=HazardSeverity.CRITICAL,
                    message=(
                        f"Follow-up instruction '{follow_up.id}' contains an unsupported "
                        "claim guaranteeing provider suitability, quality, or availability."
                    ),
                    item_id=follow_up.id,
                    requires_human_review=True,
                    evidence_snippet=claim_candidates,
                )
            )

    def _validate_warning_sign(
        self, warning: WarningSignInstruction, evidence_verified: bool
    ) -> None:
        """Audit a WarningSignInstruction for evidence support, unclear symptoms, and red-flag escalation."""
        source_quote = (
            warning.source_evidence.source_text
            if warning.source_evidence is not None
            else None
        )
        ev_norm = (
            _normalize_whitespace(source_quote)
            if evidence_verified and source_quote
            else ""
        )
        symptom_text = warning.symptom_description or ""
        combined_text = f"{symptom_text} {source_quote or ''}"

        # Section A & B: Verify symptom_description and action_to_take against source_evidence
        if evidence_verified and ev_norm:
            symp_words = [
                w
                for w in re.findall(r"[a-z0-9]+", symptom_text.lower())
                if len(w) > 3 and w not in {"with", "from", "that", "this", "your", "have"}
            ]
            if symp_words and not any(w in ev_norm for w in symp_words):
                warning.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.MISSING_SOURCE_EVIDENCE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Source evidence for warning sign '{warning.id}' does not support "
                            f"symptom '{warning.symptom_description}'."
                        ),
                        item_id=warning.id,
                        requires_human_review=True,
                        evidence_snippet=source_quote,
                    )
                )
                warning.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Symptom description '{warning.symptom_description}' is not supported "
                            "by the item's source evidence."
                        ),
                        item_id=warning.id,
                        requires_human_review=True,
                        evidence_snippet=source_quote,
                    )
                )

            if not _is_missing_or_unknown(warning.action_to_take):
                assert warning.action_to_take is not None
                action_norm = _normalize_whitespace(warning.action_to_take)
                if action_norm not in ev_norm:
                    warning.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.HIGH,
                            message=(
                                f"Extracted action_to_take '{warning.action_to_take}' on '{warning.id}' "
                                "is not supported by the item's source evidence."
                            ),
                            item_id=warning.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )

        # 1. Unclear symptoms or missing action
        is_unclear = (
            _is_missing_or_unknown(symptom_text)
            or _is_missing_or_unknown(warning.action_to_take)
            or any(p.search(symptom_text) for p in UNCLEAR_SYMPTOM_PATTERNS)
        )
        if is_unclear:
            warning.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.UNCLEAR_SYMPTOMS,
                    severity=HazardSeverity.HIGH,
                    message=(
                        f"Warning sign '{warning.id}' contains vague symptom phrasing or "
                        "lacks a clear patient action."
                    ),
                    item_id=warning.id,
                    requires_human_review=True,
                    evidence_snippet=source_quote,
                )
            )

        # 2. Warning-sign escalation (acute clinical red flags)
        is_red_flag = warning.urgency in (
            UrgencyLevel.URGENT,
            UrgencyLevel.EMERGENT_911,
        ) or any(p.search(combined_text) for p in CRITICAL_RED_FLAG_PATTERNS)

        if is_red_flag:
            warning.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.WARNING_SIGN_ESCALATION,
                    severity=HazardSeverity.CRITICAL,
                    message=(
                        f"High-acuity red-flag warning sign detected ('{warning.symptom_description}'). "
                        "Ensure prominent emergency escalation visibility and clinician review."
                    ),
                    item_id=warning.id,
                    requires_human_review=True,
                    evidence_snippet=source_quote,
                )
            )

    def _validate_task_item(
        self, task: CareTaskItem, evidence_verified: bool
    ) -> None:
        """Verify that CareTaskItem fields (title, due_window) are supported by source evidence."""
        if not evidence_verified or task.source_evidence is None:
            return
        ev_norm = _normalize_whitespace(task.source_evidence.source_text)

        if not _is_missing_or_unknown(task.due_window):
            assert task.due_window is not None
            due_norm = _normalize_whitespace(task.due_window)
            if due_norm not in ev_norm:
                task.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                        severity=HazardSeverity.HIGH,
                        message=(
                            f"Task due_window '{task.due_window}' on '{task.id}' is not "
                            "supported by the task's source evidence."
                        ),
                        item_id=task.id,
                        requires_human_review=True,
                        evidence_snippet=task.source_evidence.source_text,
                    )
                )

        self._verify_free_text_against_evidence(
            item=task,
            free_text=task.title,
            field_label="title",
            ev_norm=ev_norm,
        )

    def _detect_duplicate_items(
        self, payload: DischargeExtractionPayload
    ) -> None:
        """Section G: Detect duplicate IDs or duplicate clinical items and escalate to NEEDS REVIEW."""
        all_items: list[ExtractedItemBase] = [
            *payload.medications,
            *payload.follow_ups,
            *payload.warning_signs,
            *payload.tasks,
        ]

        # 1. Group by item ID
        by_id: dict[str, list[ExtractedItemBase]] = defaultdict(list)
        for item in all_items:
            by_id[item.id].append(item)

        for item_id, group in by_id.items():
            if len(group) > 1:
                for dup in group:
                    dup.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.DUPLICATE_ITEM,
                            severity=HazardSeverity.HIGH,
                            message=(
                                f"Duplicate item ID '{item_id}' detected ({len(group)} occurrences). "
                                "Human review required to prevent duplicate patient tasks."
                            ),
                            item_id=dup.id,
                            requires_human_review=True,
                        )
                    )
                    self._finalize_item_status(dup)

        # 2. Group medications by identical clinical content even if IDs differ
        med_by_signature: dict[tuple[str, str, str, str], list[MedicationInstruction]] = (
            defaultdict(list)
        )
        for med in payload.medications:
            sig = (
                resolve_canonical_medication_name(med.medication_name),
                med.action.value,
                _normalize_whitespace(med.dosage or ""),
                _normalize_whitespace(med.frequency or ""),
            )
            med_by_signature[sig].append(med)

        for sig, med_group in med_by_signature.items():
            if len(med_group) > 1:
                for dup_med in med_group:
                    if not any(
                        f.code == HazardCode.DUPLICATE_ITEM
                        for f in dup_med.safety_flags
                    ):
                        dup_med.safety_flags.append(
                            SafetyFlag(
                                code=HazardCode.DUPLICATE_ITEM,
                                severity=HazardSeverity.HIGH,
                                message=(
                                    f"Duplicate medication instruction for '{dup_med.medication_name}' "
                                    "detected. Preserving audit trail and escalating for review."
                                ),
                                item_id=dup_med.id,
                                requires_human_review=True,
                            )
                        )
                        self._finalize_item_status(dup_med)

    def _detect_cross_item_conflicts(
        self, payload: DischargeExtractionPayload
    ) -> None:
        """Section D, E, F: Detect conflicts across brand/generic medications and normalized follow-up timeframes."""
        meds_by_canonical: dict[str, list[MedicationInstruction]] = defaultdict(list)
        for med in payload.medications:
            key = resolve_canonical_medication_name(med.medication_name)
            meds_by_canonical[key].append(med)

        for canonical_name, group in meds_by_canonical.items():
            if len(group) < 2:
                continue
            actions = {m.action for m in group}
            dosages = {
                _normalize_whitespace(m.dosage)
                for m in group
                if not _is_missing_or_unknown(m.dosage)
            }
            frequencies = {
                _normalize_whitespace(m.frequency)
                for m in group
                if not _is_missing_or_unknown(m.frequency)
            }
            if len(actions) > 1 or len(dosages) > 1 or len(frequencies) > 1:
                for m in group:
                    m.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.CONFLICTING_INSTRUCTIONS,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Conflicting instructions for medication '{m.medication_name}' "
                                f"(canonical identity '{canonical_name}'): multiple entries specify "
                                f"incompatible actions ({[x.action.value for x in group]}), "
                                f"dosages ({[x.dosage for x in group]}), or frequencies."
                            ),
                            item_id=m.id,
                            requires_human_review=True,
                            evidence_snippet=(
                                m.source_evidence.source_text
                                if m.source_evidence
                                else None
                            ),
                        )
                    )
                    # Section D: EVERY item in a conflict group must be CONFLICT_DETECTED + PENDING_HUMAN_REVIEW
                    m.safety_status = SafetyStatus.CONFLICT_DETECTED
                    m.review_status = ReviewDecisionStatus.PENDING_HUMAN_REVIEW

        followups_by_specialty: dict[str, list[FollowUpInstruction]] = defaultdict(list)
        for fu in payload.follow_ups:
            if not _is_missing_or_unknown(fu.specialty):
                key = _normalize_whitespace(fu.specialty or "")
                followups_by_specialty[key].append(fu)

        for spec_name, group in followups_by_specialty.items():
            if len(group) < 2:
                continue
            canonical_timeframes = {
                normalize_timeframe(f.timeframe) or _normalize_whitespace(f.timeframe or "")
                for f in group
                if not _is_missing_or_unknown(f.timeframe)
            }
            if len(canonical_timeframes) > 1:
                for f in group:
                    f.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.CONFLICTING_INSTRUCTIONS,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Conflicting follow-up timeframes for specialty '{f.specialty}': "
                                f"{[x.timeframe for x in group]} (normalized: {sorted(canonical_timeframes)})."
                            ),
                            item_id=f.id,
                            requires_human_review=True,
                            evidence_snippet=(
                                f.source_evidence.source_text
                                if f.source_evidence
                                else None
                            ),
                        )
                    )
                    f.safety_status = SafetyStatus.CONFLICT_DETECTED
                    f.review_status = ReviewDecisionStatus.PENDING_HUMAN_REVIEW

    def _detect_omitted_instructions(
        self, payload: DischargeExtractionPayload, raw_doc: str
    ) -> None:
        """Section C & K: Detect omitted medications, follow-ups, tests, warning signs, and empty extractions."""
        all_items: list[ExtractedItemBase] = [
            *payload.medications,
            *payload.follow_ups,
            *payload.warning_signs,
            *payload.tasks,
        ]

        # Section K: Empty extraction check
        if len(all_items) == 0:
            payload.global_safety_flags.append(
                SafetyFlag(
                    code=HazardCode.INCOMPLETE_EXTRACTION,
                    severity=HazardSeverity.CRITICAL,
                    message=(
                        "Discharge document contains text, but all extracted instruction "
                        "categories are empty. Blocking patient release."
                    ),
                    requires_human_review=True,
                )
            )
            payload.global_safety_flags.append(
                SafetyFlag(
                    code=HazardCode.OMITTED_INSTRUCTION,
                    severity=HazardSeverity.CRITICAL,
                    message="All discharge instructions were omitted from extraction.",
                    requires_human_review=True,
                )
            )
            return

        # Gather verified evidence snippets per category
        med_evidence: list[str] = []
        for med in payload.medications:
            if med.source_evidence and not any(
                f.code == HazardCode.MISSING_SOURCE_EVIDENCE for f in med.safety_flags
            ):
                med_evidence.append(_normalize_whitespace(med.source_evidence.source_text))

        fu_evidence: list[str] = []
        for fu in payload.follow_ups:
            if fu.source_evidence and not any(
                f.code == HazardCode.MISSING_SOURCE_EVIDENCE for f in fu.safety_flags
            ):
                fu_evidence.append(_normalize_whitespace(fu.source_evidence.source_text))
        for task in payload.tasks:
            if task.source_evidence and not any(
                f.code == HazardCode.MISSING_SOURCE_EVIDENCE for f in task.safety_flags
            ):
                fu_evidence.append(_normalize_whitespace(task.source_evidence.source_text))

        ws_evidence: list[str] = []
        for ws in payload.warning_signs:
            if ws.source_evidence and not any(
                f.code == HazardCode.MISSING_SOURCE_EVIDENCE for f in ws.safety_flags
            ):
                ws_evidence.append(_normalize_whitespace(ws.source_evidence.source_text))

        all_verified_evidence = med_evidence + fu_evidence + ws_evidence

        # Segment raw document into lines and sentences (avoiding splits on 'Dr.' or numbered list prefixes)
        raw_lines = [line.strip() for line in raw_doc.splitlines() if line.strip()]
        sentences: list[tuple[str, bool]] = []
        for line_idx, line in enumerate(raw_lines):
            parts = [
                p.strip()
                for p in re.split(
                    r"(?<!\bdr)(?<!\bmr)(?<!\bms)(?<!\bst)(?<!\b\d)(?<=[.!?])\s+",
                    line,
                    flags=re.IGNORECASE,
                )
                if p.strip()
            ]
            for part_idx, part in enumerate(parts):
                # Track whether this sentence starts a line (or is the very first sentence of the doc)
                is_line_start = part_idx == 0
                sentences.append((part, is_line_start))

        for sent, is_line_start in sentences:
            sent_norm = re.sub(
                r"^(?:[-*•]+|\d+[.)])\s*", "", _normalize_whitespace(sent)
            ).strip()
            # Ignore pure section headers or routine inpatient observation notes
            if re.match(
                r"^(?:discharge\s+summary.*|discharge\s+medications:?|medications:?|follow-?up\s+appointments:?|follow-?up:?|return\s+precautions:?|hospital\s+course:?|addendum:?|patient\s+stabilized\.?)$",
                sent_norm,
            ):
                continue
            if "hemodynamically stable" in sent_norm:
                continue

            # Check uncertain negation in raw text
            if any(p.search(sent_norm) for p in UNCERTAIN_NEGATION_PATTERNS):
                if not any(sent_norm in ev or ev in sent_norm for ev in all_verified_evidence):
                    self._append_omission_flags(
                        payload,
                        f"Uncertain negation in discharge text was not reviewed: '{sent}'",
                        sent,
                    )
                continue

            sent_stripped_neg = _strip_negated_clauses(sent_norm)
            if not sent_stripped_neg:
                continue

            # 1. Check omitted medication instructions
            has_sensitive_med_verb = any(
                p.search(sent_stripped_neg) for p in SENSITIVE_MED_OMISSION_PATTERNS
            ) or bool(
                re.search(r"\bcontinue\s+[a-z]{3,}", sent_stripped_neg)
            )
            if has_sensitive_med_verb:
                covered_by_med = any(
                    sent_norm in ev or ev in sent_norm or sent_stripped_neg in ev
                    for ev in med_evidence
                )
                if not covered_by_med:
                    self._append_omission_flags(
                        payload,
                        f"Clinically sensitive medication instruction omitted from extraction: '{sent}'",
                        sent,
                    )
                    continue

            # 2. Check omitted tests / labs / imaging
            has_test_order = bool(
                re.search(
                    r"\b(?:repeat|check|obtain|order|schedule|get|draw)\s+(?:a\s+|an\s+|your\s+|repeat\s+)?(?:bmp|cbc|cmp|inr|labs?|blood\s*work|potassium|creatinine|electrolytes|troponin|bnp|lipid\s+panel|hba1c|a1c|urinalysis|urine\s+culture|blood\s+culture|x-ray|chest\s+x-ray|cxr|ct\s+scan|ct|mri|echocardiogram|echo|ekg|ecg|ultrasound|biopsy)\b",
                    sent_stripped_neg,
                )
            )
            if has_test_order:
                covered_by_any = any(
                    sent_norm in ev or ev in sent_norm for ev in all_verified_evidence
                )
                if not covered_by_any:
                    self._append_omission_flags(
                        payload,
                        f"Diagnostic test or lab instruction omitted from extraction: '{sent}'",
                        sent,
                    )
                    continue

            # 3. Check omitted warning signs / return precautions
            has_warning_precaution = bool(
                re.search(
                    r"\b(?:call\s+911|seek\s+emergency|go\s+to\s+(?:the\s+)?(?:er|ed|emergency\s+room)|return\s+to\s+(?:the\s+)?(?:er|ed|emergency)|call\s+(?:your\s+|the\s+)?(?:doctor|clinic|physician|provider|office)\s+if|monitor\s+for|watch\s+for|if\s+you\s+(?:develop|experience|notice))\b",
                    sent_stripped_neg,
                )
            ) or any(p.search(sent_stripped_neg) for p in CRITICAL_RED_FLAG_PATTERNS)
            if has_warning_precaution:
                covered_by_ws = any(
                    sent_norm in ev or ev in sent_norm for ev in ws_evidence
                )
                if not covered_by_ws:
                    self._append_omission_flags(
                        payload,
                        f"Warning sign or return precaution omitted from extraction: '{sent}'",
                        sent,
                    )
                    continue

            # 4. Check omitted follow-ups / referrals
            has_explicit_referral_or_specialist_fu = bool(
                re.search(
                    r"\b(?:refer(?:red|ral)?\s+to|consult(?:ation)?\s+with|appointment\s+with|schedule\s+(?:a\s+)?(?:visit|appointment|follow-?up))\b",
                    sent_stripped_neg,
                )
                or (
                    re.search(r"\bfollow[\s-]?up\b", sent_stripped_neg)
                    and (
                        is_line_start
                        or re.search(
                            r"\b(?:cardiology|nephrology|neurology|pulmonology|oncology|endocrinology|gastroenterology|orthopedics|urology|surgery|rheumatology|psychiatry|dermatology|ophthalmology|ent|hematology|infectious\s+disease|dr\.|clinic|hospital|for\s+[a-z]+)\b",
                            sent_stripped_neg,
                        )
                    )
                )
            )
            if has_explicit_referral_or_specialist_fu:
                covered_by_fu = any(
                    sent_norm in ev or ev in sent_norm for ev in fu_evidence
                )
                if not covered_by_fu:
                    self._append_omission_flags(
                        payload,
                        f"Follow-up or referral instruction omitted from extraction: '{sent}'",
                        sent,
                    )

    def _append_omission_flags(
        self,
        payload: DischargeExtractionPayload,
        message: str,
        snippet: str,
    ) -> None:
        payload.global_safety_flags.append(
            SafetyFlag(
                code=HazardCode.OMITTED_INSTRUCTION,
                severity=HazardSeverity.CRITICAL,
                message=message,
                requires_human_review=True,
                evidence_snippet=snippet,
            )
        )
        if not any(
            f.code == HazardCode.INCOMPLETE_EXTRACTION
            for f in payload.global_safety_flags
        ):
            payload.global_safety_flags.append(
                SafetyFlag(
                    code=HazardCode.INCOMPLETE_EXTRACTION,
                    severity=HazardSeverity.CRITICAL,
                    message="Extraction is incomplete; one or more clinical instructions in the raw discharge summary were omitted.",
                    requires_human_review=True,
                    evidence_snippet=snippet,
                )
            )

    def _validate_provider_matches(
        self, payload: DischargeExtractionPayload
    ) -> None:
        """Section H: Audit all synthetic provider directory fields for unsupported evaluation/availability claims."""
        for provider in payload.provider_matches:
            for field_name, field_val in (
                ("claim_text", provider.claim_text),
                ("name", provider.name),
                ("facility", provider.facility),
            ):
                if field_val and any(
                    p.search(field_val) for p in UNSUPPORTED_PROVIDER_CLAIM_PATTERNS
                ):
                    payload.global_safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_PROVIDER_CLAIM,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Synthetic provider listing '{provider.provider_id}' field '{field_name}' "
                                "contains an unsupported suitability, endorsement, or availability claim."
                            ),
                            item_id=provider.provider_id,
                            requires_human_review=True,
                            evidence_snippet=field_val,
                        )
                    )

            # Verify mandatory non-guarantee disclaimer is intact and does not itself make a positive claim
            disclaimer_lower = provider.verification_disclaimer.lower()
            if "does not guarantee" not in disclaimer_lower:
                payload.global_safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.UNSUPPORTED_PROVIDER_CLAIM,
                        severity=HazardSeverity.HIGH,
                        message=(
                            f"Synthetic provider '{provider.provider_id}' is missing the mandatory "
                            "non-guarantee disclaimer."
                        ),
                        item_id=provider.provider_id,
                        requires_human_review=True,
                    )
                )

    def _finalize_item_status(self, item: ExtractedItemBase) -> None:
        """Set safety_status and review_status on an item based on its accumulated flags."""
        if any(
            f.code == HazardCode.CONFLICTING_INSTRUCTIONS for f in item.safety_flags
        ):
            item.safety_status = SafetyStatus.CONFLICT_DETECTED
            item.review_status = ReviewDecisionStatus.PENDING_HUMAN_REVIEW
        elif any(f.requires_human_review for f in item.safety_flags):
            item.safety_status = SafetyStatus.NEEDS_REVIEW
            item.review_status = ReviewDecisionStatus.PENDING_HUMAN_REVIEW
        else:
            item.safety_status = SafetyStatus.SAFE_ORGANIZED
            item.review_status = ReviewDecisionStatus.AUTO_APPROVED


def validate_discharge_payload(
    payload: DischargeExtractionPayload,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> ValidationReport:
    """Convenience entrypoint to run the deterministic SafetyValidator."""
    validator = SafetyValidator(confidence_threshold=confidence_threshold)
    return validator.validate(payload)
