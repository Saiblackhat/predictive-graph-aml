"""Deterministic Safety Validation Engine for CAREBRIDGE (Round 2 Hardened).

Audits structured AI extraction payloads against the raw hospital discharge text
before any patient-facing rendering occurs. Never trusts raw LLM output.

Enforces:
1. canonicalize_text(): NFKC, strip Unicode category Cf & soft hyphens, normalize
   dash/hyphen variants to '-', join '-\\n' splits, collapse whitespace, casefold.
   Flags SUSPICIOUS_TEXT_ENCODING when raw text contains Cf/soft-hyphen/ZWSP.
2. Token-exact and clause-bound field support:
   - Exact (number, unit) dose parsing with strict numeric & word boundaries.
   - Canonical frequency class comparison ('daily' is NOT supported by 'twice daily').
   - Qualifier-preserving timeframe comparison (unqualified extraction of qualified
     source is unsupported).
   - Word-boundary matching for specialty, provider, reason, and medication names.
   - Multi-drug clause binding: each medication's fields must be supported inside
     the clause mentioning that medication's identity.
3. Action-to-evidence alignment (CONTINUE requires continue/take/resume/keep taking;
   never start/stop/hold/d/c/dc'd/etc.).
4. Clause-level omission coverage (splits on ';', ':', 'and', 'then', 'but', 'except';
   partial 'ev in sent' is rejected).
5. Governed verb-phrase negation stripping and unresolved negation escalation.
6. Closed template vocabulary whitelist for all generated/free-text fields
   (patient_friendly_summary, task.title, indication_or_note, reason) plus
   deterministic template renderer.
7. Closed template verification for provider match text and universal scanning of
   provider_claim_note across all item types.
8. Warning sign all-token evidence verification and minimum urgency cue enforcement.
9. Strict follow-up construct anchors, semantic duplicate detection across all
   item types, case-insensitive ID uniqueness, and task.linked_item_id + category validation.
10. Release gate: global blockers downgrade item safety_status to NEEDS REVIEW and
    set release_blocked=True; is_patient_releasable accepts ONLY a ValidationReport
    verified via cryptographic payload_sha256.
"""

from __future__ import annotations

import re
import unicodedata
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
    _canonical_id_key,
    compute_payload_sha256,
)

CONFIDENCE_THRESHOLD: float = 0.80

DASH_TRANSLATION_TABLE: dict[int, str] = {
    ord("\u2010"): "-",  # hyphen
    ord("\u2011"): "-",  # non-breaking hyphen
    ord("\u2012"): "-",  # figure dash
    ord("\u2013"): "-",  # en dash
    ord("\u2014"): "-",  # em dash
    ord("\u2015"): "-",  # horizontal bar
    ord("\u2212"): "-",  # minus sign
    ord("\ufe58"): "-",  # small em dash
    ord("\ufe63"): "-",  # small hyphen-minus
    ord("\uff0d"): "-",  # fullwidth hyphen-minus
}

RE_HYPHEN_NEWLINE: re.Pattern[str] = re.compile(r"-\s*[\r\n]+\s*")
RE_WHITESPACE: re.Pattern[str] = re.compile(r"\s+")
RE_WORD_TOKEN: re.Pattern[str] = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")
RE_NUMBER_TOKEN: re.Pattern[str] = re.compile(r"(?<![a-z0-9.])\d+(?:\.\d+)?(?![a-z0-9.])")

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

# Synthetic brand/generic alias map for deterministic conflict & evidence resolution.
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
    "sacubitril-valsartan": "sacubitril/valsartan",
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
    "asa": "aspirin",
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
    "prednisone": "prednisone",
    "hydralazine": "hydralazine",
    "hydroxyzine": "hydroxyzine",
    "nsaids": "nsaids",
    "nsaid": "nsaids",
    "insulin": "insulin",
}

CANONICAL_TO_ALIASES: dict[str, frozenset[str]] = {}
_temp_aliases: dict[str, set[str]] = defaultdict(set)
for _alias, _canon in SYNTHETIC_DRUG_ALIASES.items():
    _temp_aliases[_canon].add(_alias)
    _temp_aliases[_canon].add(_canon)
for _canon, _aset in _temp_aliases.items():
    CANONICAL_TO_ALIASES[_canon] = frozenset(_aset)

ALL_KNOWN_DRUG_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (
        alias,
        re.compile(
            rf"(?<![a-z0-9/-]){re.escape(alias)}(?![a-z0-9/-])", re.IGNORECASE
        ),
    )
    for alias in sorted(SYNTHETIC_DRUG_ALIASES.keys(), key=len, reverse=True)
)

ROUTE_EQUIVALENCE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"by mouth", "po", "p.o.", "oral", "orally"}),
    frozenset({"iv", "i.v.", "intravenous", "intravenously"}),
    frozenset({"sc", "subq", "subcutaneous", "subcutaneously", "sq"}),
    frozenset({"im", "i.m.", "intramuscular", "intramuscularly"}),
    frozenset({"topical", "topically"}),
    frozenset({"inhaled", "inhalation", "inhale"}),
    frozenset({"sl", "sublingual", "sublingually"}),
)

# Ordered from most specific to general so 'twice daily' matches before 'daily'
FREQUENCY_CLASS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "QID",
        re.compile(
            r"(?<![a-z0-9])(?:four\s+times\s+(?:a\s+day|daily)|4\s+times\s+(?:a\s+day|daily)|q\.?i\.?d\.?|every\s+6\s*hours?|q6h)(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "TID",
        re.compile(
            r"(?<![a-z0-9])(?:three\s+times\s+(?:a\s+day|daily)|3\s+times\s+(?:a\s+day|daily)|t\.?i\.?d\.?|every\s+8\s*hours?|q8h)(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "BID",
        re.compile(
            r"(?<![a-z0-9])(?:twice\s+(?:a\s+day|daily)|two\s+times\s+(?:a\s+day|daily)|2\s+times\s+(?:a\s+day|daily)|b\.?i\.?d\.?|every\s+12\s*hours?|q12h)(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "WEEKLY",
        re.compile(
            r"(?<![a-z0-9])(?:once\s+(?:a\s+week|weekly)|every\s+week|weekly|qwk)(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "MONTHLY",
        re.compile(
            r"(?<![a-z0-9])(?:once\s+(?:a\s+month|monthly)|every\s+month|monthly)(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "PRN",
        re.compile(
            r"(?<![a-z0-9])(?:as\s+needed|p\.?r\.?n\.?|if\s+(?:you\s+)?need(?:ed)?|when\s+(?:you\s+)?need(?:ed)?)(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "DAILY",
        re.compile(
            r"(?<![a-z0-9])(?:once\s+(?:a\s+day|daily)|every\s+day|every\s+morning|every\s+evening|every\s+night|q\.?d\.?|q\.?a\.?m\.?|q\.?h\.?s\.?|daily)(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
)

FREQUENCY_MODIFIER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "BEDTIME",
        re.compile(
            r"(?<![a-z0-9])(?:at\s+bedtime|bedtime|q\.?h\.?s\.?|h\.?s\.?)(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "MORNING",
        re.compile(
            r"(?<![a-z0-9])(?:in\s+the\s+morning|every\s+morning|morning|q\.?a\.?m\.?)(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "EVENING",
        re.compile(
            r"(?<![a-z0-9])(?:in\s+the\s+evening|every\s+evening|evening|q\.?p\.?m\.?)(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
)

DOSE_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![a-z0-9./])(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)*)\s*(mg|mcg|ug|g|ml|meq|units?|iu|tablets?|tabs?|capsules?|caps?|puffs?|drops?|patches?|%)(?![a-z0-9/])",
    re.IGNORECASE,
)

UNIT_CANONICAL_MAP: dict[str, str] = {
    "mg": "mg",
    "mcg": "mcg",
    "ug": "mcg",
    "g": "g",
    "ml": "ml",
    "meq": "meq",
    "unit": "units",
    "units": "units",
    "iu": "iu",
    "tablet": "tablets",
    "tablets": "tablets",
    "tab": "tablets",
    "tabs": "tablets",
    "capsule": "capsules",
    "capsules": "capsules",
    "cap": "capsules",
    "caps": "capsules",
    "puff": "puffs",
    "puffs": "puffs",
    "drop": "drops",
    "drops": "drops",
    "patch": "patches",
    "patches": "patches",
    "%": "%",
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

# Governed verb-phrase negation patterns that mean "no medication change/start/stop/hold occurred"
# Guarded with negative lookahead so exception clauses ("... except warfarin") are never stripped.
SAFE_GOVERNED_NEGATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern + r"(?!\s+(?:except|other\s+than|unless|excluding)\b)", re.IGNORECASE)
    for pattern in (
        r"\bno\s+changes?\s+to\s+(?:your\s+|the\s+|home\s+)?medications?(?:\s+is\s+required|\s+are\s+required|\s+needed|\s+were\s+made)?\b",
        r"\bmedications?\s+remains?\s+unchanged\b",
        r"\bno\s+medication\s+changes?(?:\s+were\s+made|\s+are\s+required|\s+needed)?\b",
        r"\bwithout\s+(?:any\s+)?medication\s+changes?\b",
        r"\bno\s+new\s+medications?(?:\s+were\s+(?:started|added|prescribed|initiated)|\s+started|\s+added|\s+prescribed)?\b",
        r"\bno\s+(?:home\s+)?medications?\s+were\s+(?:stopped|discontinued|changed|held|withheld|modified|started|added)\b",
        r"\bdo\s+not\s+stop\s+taking\s+(?:your\s+|the\s+)?medications?\b",
        r"\bdon'?t\s+stop\s+taking\s+(?:your\s+|the\s+)?medications?\b",
    )
)

# Specific drug negation or non-taking statements that require human review or forbid CONTINUE/START/STOP
DRUG_SPECIFIC_NEGATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bdo\s+not\s+(?:stop|take|start|resume|continue|use)\b",
        r"\bdon'?t\s+(?:stop|take|start|resume|continue|use)\b",
        r"\bshould\s+not\s+be\s+(?:stopped|discontinued|taken|started|held|resumed)\b",
        r"\bmust\s+not\s+be\s+(?:stopped|discontinued|taken|started|held|resumed)\b",
        r"\bnot\s+taking\b",
        r"\bis\s+not\s+on\b",
        r"\bnever\s+(?:took|taken|started)\b",
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

STOP_DISCONTINUE_CUE_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![a-z0-9])(?:stop\s+taking|stop(?:ped|ping)?|discontinu(?:e|ed|ing)|d/c(?:'?d)?|dc(?:'d)?|do\s+not\s+take|no\s+longer\s+tak(?:e|ing)|avoid(?:ing)?)(?![a-z0-9])",
    re.IGNORECASE,
)

HOLD_CUE_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![a-z0-9])(?:hold(?:ing)?|withhold(?:ing)?|skip(?:ped|ping)?|pause(?:d)?)(?![a-z0-9])",
    re.IGNORECASE,
)

NEW_START_CUE_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![a-z0-9])(?:start\s+new|start(?:ed|ing)?|begin\s+taking|begin(?:ning)?|initiat(?:e|ed|ing)|add(?:ed|ing)?|newly\s+prescribed|new\s+medication)(?![a-z0-9])",
    re.IGNORECASE,
)

CONTINUE_CUE_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![a-z0-9])(?:continu(?:e|ed|ing)|keep\s+taking|resum(?:e|ed|ing)|remain\s+on|stay\s+on|take\s+(?:your\s+|the\s+)?(?:home|prior|usual)\s+medication|(?:home\s+medications?:?.*\btake\b|\btake\b.*\b(?:as\s+before|home\s+medication|prior\s+to\s+admission|previously|only\s+if\s+needed|as\s+needed\s+for)))(?![a-z0-9])",
    re.IGNORECASE,
)

BARE_TAKE_CUE_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![a-z0-9])take(?![a-z0-9])",
    re.IGNORECASE,
)

STOP_START_PATTERNS: tuple[re.Pattern[str], ...] = (
    STOP_DISCONTINUE_CUE_PATTERN,
    HOLD_CUE_PATTERN,
    NEW_START_CUE_PATTERN,
)

SENSITIVE_MED_OMISSION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?<![a-z0-9])switch(?:ed|ing)?(?![a-z0-9])",
        r"(?<![a-z0-9])chang(?:e|ed|ing)(?![a-z0-9])",
        r"(?<![a-z0-9])stop(?:ped|ping)?(?![a-z0-9])",
        r"(?<![a-z0-9])discontinu(?:e|ed|ing)(?![a-z0-9])",
        r"(?<![a-z0-9])d/c(?:'?d)?(?![a-z0-9])",
        r"(?<![a-z0-9])dc(?:'d)?(?![a-z0-9])",
        r"(?<![a-z0-9])hold(?:ing)?(?![a-z0-9])",
        r"(?<![a-z0-9])withhold(?:ing)?(?![a-z0-9])",
        r"(?<![a-z0-9])start(?:ed|ing)?(?![a-z0-9])",
        r"(?<![a-z0-9])initiat(?:e|ed|ing)(?![a-z0-9])",
        r"(?<![a-z0-9])increas(?:e|ed|ing)(?![a-z0-9])",
        r"(?<![a-z0-9])decreas(?:e|ed|ing)(?![a-z0-9])",
        r"(?<![a-z0-9])taper(?:ed|ing)?(?![a-z0-9])",
        r"(?<![a-z0-9])resum(?:e|ed|ing)(?![a-z0-9])",
        r"(?<![a-z0-9])reduc(?:e|ed|ing)(?![a-z0-9])",
        r"(?<![a-z0-9])skip(?:ped|ping)?(?![a-z0-9])",
        r"(?<![a-z0-9])avoid(?:ing)?(?![a-z0-9])",
        r"(?<![a-z0-9])do\s+not\s+take(?![a-z0-9])",
        r"(?<![a-z0-9])no\s+longer(?![a-z0-9])",
        r"(?<![a-z0-9])instead\s+of(?![a-z0-9])",
        r"(?<![a-z0-9])except(?![a-z0-9])",
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

EMERGENT_CUE_REGEX: re.Pattern[str] = re.compile(
    r"\b(?:call\s+911|911|emergency\s+room|emergency\s+department|go\s+to\s+(?:the\s+)?(?:ed|er)|return\s+to\s+(?:the\s+)?(?:ed|er)|seek\s+emergency|immediately)\b",
    re.IGNORECASE,
)

URGENT_CUE_REGEX: re.Pattern[str] = re.compile(
    r"\b(?:urgent\s+care|urgently|right\s+away|same\s+day|fever\s*(?:>|above|over)\s*101)\b",
    re.IGNORECASE,
)

URGENCY_RANK: dict[UrgencyLevel, int] = {
    UrgencyLevel.UNKNOWN: 0,
    UrgencyLevel.ROUTINE: 1,
    UrgencyLevel.SOON: 2,
    UrgencyLevel.URGENT: 3,
    UrgencyLevel.EMERGENT_911: 4,
}

# Closed template vocabulary for generated patient-friendly text and task titles.
# Every word token in generated free text must either appear in the item's verified
# source evidence OR belong to this strictly bounded non-clinical connector vocabulary.
CLOSED_TEMPLATE_VOCABULARY: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "appointment",
        "as",
        "at",
        "book",
        "by",
        "call",
        "care",
        "check",
        "clinic",
        "continue",
        "daily",
        "day",
        "days",
        "discontinue",
        "do",
        "doctor",
        "dose",
        "during",
        "every",
        "follow",
        "follow-up",
        "followup",
        "for",
        "from",
        "happens",
        "hold",
        "hours",
        "if",
        "in",
        "instruction",
        "is",
        "it",
        "keep",
        "medication",
        "Monitor",
        "monitor",
        "morning",
        "mouth",
        "need",
        "needed",
        "not",
        "noted",
        "of",
        "office",
        "on",
        "once",
        "only",
        "or",
        "oral",
        "orally",
        "plan",
        "please",
        "prescribed",
        "provider",
        "recorded",
        "resume",
        "schedule",
        "scheduled",
        "see",
        "start",
        "stop",
        "summary",
        "synthetic",
        "take",
        "taking",
        "the",
        "this",
        "three",
        "time",
        "times",
        "to",
        "twice",
        "up",
        "visit",
        "watch",
        "week",
        "weeks",
        "when",
        "with",
        "within",
        "you",
        "your",
    }
)

PROVIDER_TEMPLATE_VOCABULARY: frozenset[str] = frozenset(
    {
        "listed",
        "in",
        "synthetic",
        "directory",
        "under",
        "at",
        "facility",
        "specialty",
        "distance",
        "miles",
        "mile",
        "only",
        "entry",
        "for",
        "of",
        "the",
        "and",
    }
)

UNSUPPORTED_PROVIDER_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bbest\s+match\b",
        r"\bbest\b",
        r"\brecommend(?:ed|s)?\b",
        r"\bpreferred\b",
        r"\btop(?:[-\s]?rated)?\b",
        r"\bguarantee(?:d|s)?\b",
        r"\bsuitable\b",
        r"\bideal\b",
        r"\bhighly\s+(?:rated|qualified)\b",
        r"\bmost\s+qualified\b",
        r"\bqualified\b",
        r"\btrusted\b",
        r"\bleading\b",
        r"\bavailable\s+(?:now|tomorrow|today|immediately)\b",
        r"\baccepting\s+(?:new\s+)?patients\b",
        r"\bdefinitely\s+accepts\b",
        r"\binstant\s+booking\b",
        r"\bwill\s+cure\b",
        r"\balways\s+available\b",
        r"\bpatients\s+like\s+you\b",
        r"\bspecializes\s+in\s+patients\b",
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

MONTH_NAME_TO_NUMBER: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

NON_DRUG_PRE_DOSE_TOKENS: frozenset[str] = frozenset(
    {
        "take",
        "taking",
        "continue",
        "continuing",
        "start",
        "starting",
        "begin",
        "beginning",
        "initiate",
        "initiating",
        "add",
        "added",
        "resume",
        "resuming",
        "hold",
        "holding",
        "withhold",
        "withholding",
        "stop",
        "stopping",
        "discontinue",
        "discontinuing",
        "increase",
        "increased",
        "increasing",
        "decrease",
        "decreased",
        "decreasing",
        "reduce",
        "reduced",
        "reducing",
        "change",
        "changed",
        "changing",
        "to",
        "from",
        "at",
        "of",
        "by",
        "for",
        "with",
        "about",
        "around",
        "up",
        "down",
        "than",
        "more",
        "less",
        "greater",
        "fewer",
        "least",
        "most",
        "oral",
        "orally",
        "po",
        "iv",
        "sc",
        "sq",
        "im",
        "sl",
        "daily",
        "twice",
        "once",
        "every",
        "then",
        "give",
        "dose",
        "doses",
    }
)

NON_DRUG_POST_VERB_TOKENS: frozenset[str] = frozenset(
    {
        "taking",
        "medication",
        "medications",
        "home",
        "your",
        "the",
        "a",
        "an",
        "therapy",
        "treatment",
        "dose",
        "doses",
        "regimen",
        "immediately",
        "now",
        "after",
        "before",
        "when",
        "if",
        "until",
        "unless",
        "as",
        "at",
        "in",
        "on",
        "for",
        "all",
        "new",
        "prior",
        "usual",
    }
)

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

# Follow-up construct anchor: requires an explicit follow-up, referral, appointment, or specialty consultation construct
FOLLOWUP_CONSTRUCT_REGEX: re.Pattern[str] = re.compile(
    r"\b(?:follow[\s-]?up|refer(?:ral|red)?(?:\s+to)?|appointment\s+with|schedule\s+(?:a\s+)?(?:visit|appointment|follow[\s-]?up)|consult(?:ation)?\s+with|(?:see|visit)\s+(?:dr\.|primary\s+care|cardiology|nephrology|neurology|pulmonology|oncology|endocrinology|gastroenterology|orthopedics|urology|surgery|rheumatology|psychiatry|dermatology|ophthalmology|ent|hematology|infectious\s+disease|[a-z]+\s+clinic))\b",
    re.IGNORECASE,
)

CLAUSE_SPLIT_REGEX: re.Pattern[str] = re.compile(
    r"\s*(?:;|:(?!\d)|\b(?:and|then|but|except)\b)\s*",
    re.IGNORECASE,
)

SENTENCE_SPLIT_REGEX: re.Pattern[str] = re.compile(
    r"(?<!\bdr)(?<!\bmr)(?<!\bms)(?<!\bst)(?<!\b\d)(?<=[.!?])\s+",
    re.IGNORECASE,
)

TEST_ORDER_REGEX: re.Pattern[str] = re.compile(
    r"\b(?:repeat|check|obtain|order|schedule|get|draw)\s+(?:a\s+|an\s+|your\s+|repeat\s+)?(?:bmp|cbc|cmp|inr|labs?|blood\s*work|potassium|creatinine|electrolytes|troponin|bnp|lipid\s+panel|hba1c|a1c|urinalysis|urine\s+culture|blood\s+culture|x-ray|chest\s+x-ray|cxr|ct\s+scan|ct|mri|echocardiogram|echo|ekg|ecg|ultrasound|biopsy)\b",
    re.IGNORECASE,
)

WARNING_PRECAUTION_REGEX: re.Pattern[str] = re.compile(
    r"\b(?:call\s+911|seek\s+emergency|go\s+to\s+(?:the\s+)?(?:er|ed|emergency\s+room)|return\s+to\s+(?:the\s+)?(?:er|ed|emergency)|call\s+(?:your\s+|the\s+)?(?:doctor|clinic|physician|provider|office)\s+if|monitor\s+for|watch\s+for|if\s+you\s+(?:develop|experience|notice))\b",
    re.IGNORECASE,
)

ROUTINE_HEADER_REGEX: re.Pattern[str] = re.compile(
    r"^(?:discharge\s+summary.*|discharge\s+medications:?|home\s+medications:?|medications:?|follow-?up\s+appointments:?|follow-?up:?|return\s+precautions:?|hospital\s+course:?|addendum:?|patient\s+stabilized\.?|routine\s+note\.?)$",
    re.IGNORECASE,
)


def has_suspicious_encoding(text: str) -> bool:
    """Return True if text contains Unicode format controls (Cf), soft hyphens, or zero-width characters."""
    for ch in text:
        if ch == "\u00ad" or unicodedata.category(ch) == "Cf":
            return True
    return False


def canonicalize_text(text: str) -> str:
    """Canonicalize text before any comparison (Requirement 1).

    Steps:
    1. Unicode NFKC normalization
    2. Delete Unicode category Cf characters and soft hyphens (U+00AD)
    3. Map all Unicode dash/hyphen variants to ASCII '-'
    4. Join hyphenated line breaks ('-\\n') into a single uninterrupted token
    5. Collapse all whitespace runs to a single ASCII space and strip
    6. Casefold
    """
    if not text:
        return ""
    nfkc = unicodedata.normalize("NFKC", text)
    no_cf = "".join(
        ch for ch in nfkc if unicodedata.category(ch) != "Cf" and ch != "\u00ad"
    )
    dash_norm = no_cf.translate(DASH_TRANSLATION_TABLE)
    joined_lines = RE_HYPHEN_NEWLINE.sub("", dash_norm)
    collapsed = RE_WHITESPACE.sub(" ", joined_lines).strip()
    return collapsed.casefold()


def _normalize_whitespace(text: str) -> str:
    return canonicalize_text(text)


def _is_missing_or_unknown(value: str | None) -> bool:
    if value is None:
        return True
    return canonicalize_text(value) in UNKNOWN_TOKENS


def _strip_negated_clauses(text: str) -> str:
    """Remove explicitly governed safe negation phrases (e.g. 'no new medications were started')."""
    cleaned = canonicalize_text(text)
    for pattern in SAFE_GOVERNED_NEGATION_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return RE_WHITESPACE.sub(" ", cleaned).strip()


def resolve_canonical_medication_name(name: str) -> str:
    """Resolve a medication name to its canonical generic identity using the synthetic alias map."""
    norm = canonicalize_text(name)
    base = re.sub(r"\s*\([^)]*\)\s*", " ", norm).strip()
    if base in SYNTHETIC_DRUG_ALIASES:
        return SYNTHETIC_DRUG_ALIASES[base]
    if norm in SYNTHETIC_DRUG_ALIASES:
        return SYNTHETIC_DRUG_ALIASES[norm]
    slash_base = base.replace("-", "/")
    if slash_base in SYNTHETIC_DRUG_ALIASES:
        return SYNTHETIC_DRUG_ALIASES[slash_base]
    return base


def get_medication_equivalents(name: str) -> frozenset[str]:
    """Return all known casefolded brand/generic aliases (and hyphen/slash combination variants) for a medication name."""
    norm = canonicalize_text(name)
    canonical = resolve_canonical_medication_name(name)
    base_set = set(CANONICAL_TO_ALIASES.get(canonical, frozenset())) | {norm, canonical}
    expanded: set[str] = set()
    for item in base_set:
        expanded.add(item)
        if "-" in item:
            expanded.add(item.replace("-", "/"))
        if "/" in item:
            expanded.add(item.replace("/", "-"))
    return frozenset(expanded)


def _normalize_single_num_str(num_part: str) -> str:
    if "." in num_part:
        return num_part.rstrip("0").rstrip(".")
    return num_part.lstrip("0") or "0"


def parse_doses(text: str) -> list[tuple[str, str]]:
    """Parse all (number, canonical_unit) dose tuples with strict boundaries.

    '5 mg' -> ('5', 'mg'); '2.5 mg' -> ('2.5', 'mg'); '50 mcg' -> ('50', 'mcg');
    '49/51 mg' -> ('49/51', 'mg').
    '5 mg' never matches inside '2.5 mg', '15 mg', or '49/51 mg', and 'mg' != 'mcg'.
    """
    canon = canonicalize_text(text)
    results: list[tuple[str, str]] = []
    for m in DOSE_TOKEN_PATTERN.finditer(canon):
        num_str = m.group(1)
        unit_raw = m.group(2).casefold()
        unit_canon = UNIT_CANONICAL_MAP.get(unit_raw, unit_raw)
        if "/" in num_str:
            num_norm = "/".join(_normalize_single_num_str(p) for p in num_str.split("/"))
        else:
            num_norm = _normalize_single_num_str(num_str)
        results.append((num_norm, unit_canon))
    return results


def parse_frequency_classes(text: str) -> tuple[frozenset[str], frozenset[str]]:
    """Extract canonical frequency classes and time-of-day modifiers from text.

    Because multi-word patterns ('twice daily') are matched and masked before
    single-word 'daily', 'twice daily' yields {'BID'} and NEVER {'DAILY'}.
    """
    working = canonicalize_text(text)
    classes: set[str] = set()
    modifiers: set[str] = set()

    for mod_name, mod_pat in FREQUENCY_MODIFIER_PATTERNS:
        if mod_pat.search(working):
            modifiers.add(mod_name)

    for cls_name, cls_pat in FREQUENCY_CLASS_PATTERNS:
        if cls_pat.search(working):
            classes.add(cls_name)
            working = cls_pat.sub(" ", working)

    return frozenset(classes), frozenset(modifiers)


def _extract_timeframe_qualifier(norm_tf: str) -> str:
    """Return explicit qualifier token ('WITHIN', 'AFTER', 'BEFORE', 'IN', 'APPROX', 'ON', or 'UNQUALIFIED')."""
    if re.search(
        r"\b(?:in\s+about|about|approximately|around|roughly|~)\b", norm_tf
    ):
        return "APPROX"
    if re.search(
        r"\b(?:within|no\s+later\s+than|in\s+under|up\s+to)\b", norm_tf
    ):
        return "WITHIN"
    if re.search(r"\b(?:before|prior\s+to|by)\b", norm_tf):
        return "BEFORE"
    if re.search(
        r"\b(?:after|at\s+least|post|following|more\s+than)\b", norm_tf
    ):
        return "AFTER"
    if re.search(r"\b(?:in|at|for)\b", norm_tf):
        return "IN"
    if re.search(r"\bon\b", norm_tf):
        return "ON"
    return "UNQUALIFIED"


def normalize_timeframe(raw_timeframe: str | None) -> str | None:
    """Normalize a follow-up timeframe into 'QUALIFIER:AMOUNT:UNIT' or 'QUALIFIER:DATE:YYYY-MM-DD'."""
    if _is_missing_or_unknown(raw_timeframe):
        return None
    assert raw_timeframe is not None
    norm = canonicalize_text(raw_timeframe)

    for rec_pat in RECURRENCE_TIMEFRAME_PATTERNS:
        if rec_pat.search(norm):
            return None

    qualifier = _extract_timeframe_qualifier(norm)

    # Check explicit calendar date patterns (e.g., 'on 04 Dec 2026', 'Dec 18, 2026', '2026-12-04')
    m_dmy = re.search(
        r"\b(\d{1,2})\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[.,]?\s+(\d{4})\b",
        norm,
    )
    if m_dmy:
        day = int(m_dmy.group(1))
        month = MONTH_NAME_TO_NUMBER.get(m_dmy.group(2), 0)
        year = int(m_dmy.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{qualifier}:DATE:{year:04d}-{month:02d}-{day:02d}"

    m_mdy = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[.,]?\s+(\d{1,2})(?:,|\s)+(\d{4})\b",
        norm,
    )
    if m_mdy:
        month = MONTH_NAME_TO_NUMBER.get(m_mdy.group(1), 0)
        day = int(m_mdy.group(2))
        year = int(m_mdy.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{qualifier}:DATE:{year:04d}-{month:02d}-{day:02d}"

    m_iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", norm)
    if m_iso:
        year = int(m_iso.group(1))
        month = int(m_iso.group(2))
        day = int(m_iso.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{qualifier}:DATE:{year:04d}-{month:02d}-{day:02d}"

    working = norm
    for word_num in (
        "twenty-four",
        "twenty four",
        "forty-eight",
        "forty eight",
        "seventy-two",
        "seventy two",
    ):
        working = re.sub(rf"\b{word_num}\b", str(WORD_NUMBER_MAP[word_num]), working)
    for word_num, digit in WORD_NUMBER_MAP.items():
        if " " not in word_num and "-" not in word_num:
            working = re.sub(
                rf"\b{word_num}\b(?=\s*(?:hours?|hrs?|days?|weeks?|wks?|months?|mos?|years?|yrs?)\b)",
                str(digit),
                working,
            )

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
        start_norm = start_val
        end_norm = end_val
        unit_norm = "MONTHS"
    else:
        start_norm = start_val
        end_norm = end_val
        unit_norm = "YEARS"

    amount_str = (
        f"{start_norm}-{end_norm}" if end_norm is not None else str(start_norm)
    )
    return f"{qualifier}:{amount_str}:{unit_norm}"


def _has_word_boundary_match(needle: str, haystack: str) -> bool:
    """Check whether canonicalized needle appears in canonicalized haystack with word boundaries."""
    n = canonicalize_text(needle)
    h = canonicalize_text(haystack)
    if not n or not h:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])"
    return bool(re.search(pattern, h))


def _split_into_clauses(sentence_canon: str) -> list[str]:
    """Split a canonicalized sentence into clauses on ';', ':', 'and', 'then', 'but', 'except'."""
    parts = [p.strip(" .,;:-") for p in CLAUSE_SPLIT_REGEX.split(sentence_canon)]
    return [p for p in parts if p]


def _clause_mentions_other_drug(
    cl: str, target_equivalents: frozenset[str]
) -> bool:
    """Return True if clause `cl` names a medication distinct from `target_equivalents`.

    Elliptical clauses for the same drug (e.g. 'take 80 mg on days when weight is up more than 3 lb'
    or 'stop' in 'Take prednisone 20 mg daily for 5 days, then stop') do NOT name another drug
    and return False so they stay bound to the target drug's clause.
    """
    if any(
        re.search(rf"(?<![a-z0-9/-]){re.escape(eq)}(?![a-z0-9/-])", cl)
        for eq in target_equivalents
    ):
        return False

    # 1. Check known drug lexicon
    for alias, pat in ALL_KNOWN_DRUG_PATTERNS:
        if alias not in target_equivalents and pat.search(cl):
            return True

    # 2. Check out-of-lexicon drug token immediately preceding a dose
    for m_pre in re.finditer(
        r"\b([a-z][a-z0-9/-]{2,})\s+\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)*\s*(?:mg|mcg|ug|g|ml|meq|units?|iu|tablets?|tabs?|capsules?|caps?|puffs?|drops?|patches?|%)\b",
        cl,
    ):
        cand = m_pre.group(1)
        if (
            cand not in NON_DRUG_PRE_DOSE_TOKENS
            and cand not in target_equivalents
        ):
            return True

    # 3. Check out-of-lexicon drug token immediately following a medication action verb
    for m_post in re.finditer(
        r"\b(?:stop|discontinue|d/c|dc|hold|withhold|start|initiate|continue|resume)\s+([a-z][a-z0-9/-]{2,})\b",
        cl,
    ):
        cand = m_post.group(1)
        if (
            cand not in NON_DRUG_POST_VERB_TOKENS
            and cand not in target_equivalents
        ):
            return True

    return False


def _extract_drug_clauses(
    ev_canon: str, med_equivalents: frozenset[str]
) -> tuple[str, bool]:
    """Extract the clause(s) within ev_canon that apply to the target medication.

    In a multi-drug sentence like 'Continue atorvastatin 40 mg and amlodipine 5 mg daily':
    - Clause 1: 'continue atorvastatin 40 mg' (mentions atorvastatin)
    - Clause 2: 'amlodipine 5 mg daily' (mentions amlodipine -> other drug)
    If a clause mentions ANOTHER drug and NOT the target drug, fields in that other
    clause cannot be used to support the target drug's dosage/action.
    Conversely, elliptical clauses that do not name another drug (e.g. 'take 80 mg on days when...'
    or 'stop' after 'then') remain bound to the target medication's clause.
    """
    clauses = _split_into_clauses(ev_canon)
    if len(clauses) <= 1:
        return ev_canon, False

    clause_mentions_target: list[bool] = []
    clause_has_other_drug: list[bool] = []
    for cl in clauses:
        mentions_target = any(
            re.search(rf"(?<![a-z0-9/-]){re.escape(eq)}(?![a-z0-9/-])", cl)
            for eq in med_equivalents
        )
        clause_mentions_target.append(mentions_target)
        if mentions_target:
            clause_has_other_drug.append(False)
        else:
            clause_has_other_drug.append(
                _clause_mentions_other_drug(cl, med_equivalents)
            )

    is_multi_drug = any(clause_has_other_drug) and any(clause_mentions_target)
    if not is_multi_drug:
        return ev_canon, False

    matching_clauses: list[str] = []
    for idx, cl in enumerate(clauses):
        if clause_mentions_target[idx] or not clause_has_other_drug[idx]:
            matching_clauses.append(cl)

    bound_text = " ; ".join(matching_clauses) if matching_clauses else ""
    return bound_text, True


def render_patient_friendly_summary(item: ExtractedItemBase) -> str:
    """Deterministic template renderer over verified item fields (Requirement 6)."""
    if isinstance(item, MedicationInstruction):
        verb_map = {
            MedicationAction.CONTINUE: "Continue",
            MedicationAction.NEW_START: "Start",
            MedicationAction.STOP_DISCONTINUE: "Stop",
            MedicationAction.DOSE_CHANGE: "Take",
            MedicationAction.HOLD_TEMPORARY: "Hold",
            MedicationAction.UNKNOWN: "Take",
        }
        parts = [verb_map.get(item.action, "Take"), item.medication_name]
        if item.dosage:
            parts.append(item.dosage)
        if item.route:
            parts.append(item.route)
        if item.frequency:
            parts.append(item.frequency)
        return " ".join(parts).strip() + "."
    if isinstance(item, FollowUpInstruction):
        parts = ["Follow up"]
        if item.specialty:
            parts.append(f"with {item.specialty}")
        if item.provider_name:
            parts.append(f"({item.provider_name})")
        if item.timeframe:
            parts.append(item.timeframe)
        if item.reason:
            parts.append(f"for {item.reason}")
        return " ".join(parts).strip() + "."
    if isinstance(item, WarningSignInstruction):
        if item.action_to_take:
            return f"{item.action_to_take} for {item.symptom_description}."
        return f"Monitor for {item.symptom_description}."
    if isinstance(item, CareTaskItem):
        if item.due_window:
            return f"{item.title} ({item.due_window})."
        return f"{item.title}."
    return ""


def is_patient_releasable(target: object) -> bool:
    """Return True ONLY when passed a valid ValidationReport from validate() verified by payload_sha256.

    Requirement 10: Rejects bare DischargeExtractionPayload or forged/tampered ValidationReports.
    """
    if not isinstance(target, ValidationReport):
        return False

    if not target.payload_sha256:
        return False

    expected_sha = compute_payload_sha256(target.validated_payload)
    if target.payload_sha256 != expected_sha:
        return False

    if target.overall_safety_status != SafetyStatus.SAFE_ORGANIZED:
        return False
    if target.requires_human_review or target.total_flags > 0:
        return False

    payload = target.validated_payload
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
        if item.release_blocked:
            return False
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
        """Audit a DischargeExtractionPayload and return a cryptographically signed ValidationReport."""
        audited = payload.model_copy(deep=True)
        audited.global_safety_flags = []
        raw_doc = audited.raw_discharge_text
        normalized_doc = canonicalize_text(raw_doc)

        # Requirement 1: Check raw_discharge_text and source_text for suspicious encoding (Cf, soft hyphen, ZWSP)
        if has_suspicious_encoding(raw_doc):
            audited.global_safety_flags.append(
                SafetyFlag(
                    code=HazardCode.SUSPICIOUS_TEXT_ENCODING,
                    severity=HazardSeverity.HIGH,
                    message=(
                        "Raw discharge text contains hidden Unicode format control characters, "
                        "zero-width spaces, or soft hyphens. Human review required."
                    ),
                    requires_human_review=True,
                )
            )

        # 1. Validate individual items and token-exact, clause-bound field evidence support
        for med in audited.medications:
            med.safety_flags = []
            med.release_blocked = False
            ev_verified = self._validate_base_item(med, raw_doc, normalized_doc)
            self._validate_medication(med, normalized_doc, ev_verified)
            self._finalize_item_status(med)

        for follow_up in audited.follow_ups:
            follow_up.safety_flags = []
            follow_up.release_blocked = False
            ev_verified = self._validate_base_item(follow_up, raw_doc, normalized_doc)
            self._validate_follow_up(follow_up, ev_verified)
            self._finalize_item_status(follow_up)

        for warning in audited.warning_signs:
            warning.safety_flags = []
            warning.release_blocked = False
            ev_verified = self._validate_base_item(warning, raw_doc, normalized_doc)
            self._validate_warning_sign(warning, ev_verified)
            self._finalize_item_status(warning)

        for task in audited.tasks:
            task.safety_flags = []
            task.release_blocked = False
            ev_verified = self._validate_base_item(task, raw_doc, normalized_doc)
            self._validate_task_item(task, audited, ev_verified)
            self._finalize_item_status(task)

        # 2. Detect case-insensitive ID duplicates and semantic duplicates across all item types
        self._detect_duplicate_items(audited)

        # 3. Cross-item conflict detection (medications & follow-ups) with item-level gating
        self._detect_cross_item_conflicts(audited)

        # 4. Detect omitted sensitive instructions (clause-bound) and empty extractions
        self._detect_omitted_instructions(audited, raw_doc)

        # 5. Validate synthetic provider matches for unsupported claims (closed template facts)
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

        # Requirement 10: When ANY payload-level blocker or review requirement exists,
        # no item in the payload may remain SAFE / ORGANIZED + AUTO_APPROVED.
        if needs_review:
            for item in all_items:
                item.release_blocked = True
                if item.review_status == ReviewDecisionStatus.AUTO_APPROVED:
                    item.review_status = ReviewDecisionStatus.PENDING_HUMAN_REVIEW
                if item.safety_status == SafetyStatus.SAFE_ORGANIZED:
                    item.safety_status = SafetyStatus.NEEDS_REVIEW

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

        digest = compute_payload_sha256(audited)
        report = ValidationReport(
            case_id=audited.case_id,
            overall_safety_status=audited.overall_safety_status,
            requires_human_review=audited.requires_human_review,
            patient_releasable=False,
            payload_sha256=digest,
            total_flags=len(all_flags),
            hazard_codes_detected=unique_codes,
            validated_payload=audited,
        )
        report.patient_releasable = is_patient_releasable(report)
        return report

    def _validate_base_item(
        self, item: ExtractedItemBase, raw_doc: str, normalized_doc: str
    ) -> bool:
        """Verify source evidence presence, verbatim substring match, character spans, confidence, provider_claim_note, and free-text whitelist."""
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
            ev.source_page_verified = False

            if has_suspicious_encoding(ev.source_text):
                item.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.SUSPICIOUS_TEXT_ENCODING,
                        severity=HazardSeverity.HIGH,
                        message=f"Source evidence for '{item.id}' contains hidden format characters or soft hyphens.",
                        item_id=item.id,
                        requires_human_review=True,
                        evidence_snippet=ev.source_text,
                    )
                )

            normalized_quote = canonicalize_text(ev.source_text)
            if not normalized_quote or normalized_quote not in normalized_doc:
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

            # Verify start_char / end_char span if present
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
                    sliced_norm = canonicalize_text(raw_doc[ev.start_char : ev.end_char])
                    if sliced_norm != normalized_quote:
                        evidence_verified = False
                        ev.span_status_note = (
                            "Character span mismatch against source_text."
                        )
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

        # Requirement 7: Scan provider_claim_note on EVERY item type
        if not _is_missing_or_unknown(item.provider_claim_note):
            assert item.provider_claim_note is not None
            item.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.UNSUPPORTED_PROVIDER_CLAIM,
                    severity=HazardSeverity.CRITICAL,
                    message=(
                        f"Item '{item.id}' contains a provider claim note ('{item.provider_claim_note}') "
                        "which is not permitted without human review."
                    ),
                    item_id=item.id,
                    requires_human_review=True,
                    evidence_snippet=item.provider_claim_note,
                )
            )

        # Check patient_friendly_summary for prohibited clinical advice and closed whitelist compliance
        if item.patient_friendly_summary is not None:
            summary_norm = canonicalize_text(item.patient_friendly_summary)
            ev_norm = (
                canonicalize_text(item.source_evidence.source_text)
                if item.source_evidence is not None and evidence_verified
                else ""
            )

            if any(
                p.search(item.patient_friendly_summary)
                for p in UNSUPPORTED_PROVIDER_CLAIM_PATTERNS
            ):
                item.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.UNSUPPORTED_PROVIDER_CLAIM,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Patient-friendly summary on '{item.id}' contains an unsupported "
                            "provider suitability or availability claim."
                        ),
                        item_id=item.id,
                        requires_human_review=True,
                        evidence_snippet=item.patient_friendly_summary,
                    )
                )

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

            self._verify_free_text_whitelist(
                item=item,
                free_text=item.patient_friendly_summary,
                field_label="patient_friendly_summary",
                ev_norm=ev_norm,
                allow_template_vocab=True,
            )
            self._verify_summary_semantic_preservation(
                item=item,
                summary_text=item.patient_friendly_summary,
                ev_norm=ev_norm,
            )

        return evidence_verified

    @staticmethod
    def _stem_token(tok: str) -> str:
        """Normalize simple English inflections for morphological token matching."""
        if len(tok) > 5 and tok.endswith("ing"):
            base = tok[:-3]
            if len(base) >= 3 and base[-1] == base[-2]:
                base = base[:-1]
            return base
        if len(tok) > 4 and tok.endswith("ed"):
            return tok[:-2]
        if len(tok) > 4 and tok.endswith("es"):
            return tok[:-2]
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            return tok[:-1]
        return tok

    def _verify_summary_semantic_preservation(
        self,
        item: ExtractedItemBase,
        summary_text: str,
        ev_norm: str,
    ) -> None:
        """Requirement 2 & 3: Enforce semantic qualifier, threshold, duration, and condition preservation in patient_friendly_summary."""
        if not ev_norm:
            return
        if any(f.code == HazardCode.UNSUPPORTED_FIELD_VALUE for f in item.safety_flags):
            return

        ev_effective = _strip_negated_clauses(ev_norm)
        if isinstance(item, MedicationInstruction):
            ev_effective, _ = _extract_drug_clauses(
                ev_effective, get_medication_equivalents(item.medication_name)
            )
        if not ev_effective:
            return

        summary_norm = canonicalize_text(summary_text)

        def _flag_loss(detail: str) -> None:
            item.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                    severity=HazardSeverity.CRITICAL,
                    message=(
                        f"Patient-friendly summary on '{item.id}' loses clinically significant "
                        f"qualifier or specificity from source evidence ({detail})."
                    ),
                    item_id=item.id,
                    requires_human_review=True,
                    evidence_snippet=summary_text[:200],
                )
            )

        # 1. Every dose in ev_effective must be preserved in summary_norm
        ev_doses = parse_doses(ev_effective)
        sum_doses = set(parse_doses(summary_norm))
        for dose_tuple in ev_doses:
            if dose_tuple not in sum_doses:
                _flag_loss(f"omitted dose '{dose_tuple[0]} {dose_tuple[1]}'")
                return

        # 2. Semantic qualifier categories
        qualifier_rules: tuple[tuple[str, re.Pattern[str], re.Pattern[str]], ...] = (
            (
                "exclusive condition ('only if' / 'only when')",
                re.compile(r"\bonly\s+(?:if|when)\b"),
                re.compile(r"\bonly\s+(?:if|when)\b"),
            ),
            (
                "exclusivity qualifier ('only')",
                re.compile(r"\bonly\b"),
                re.compile(r"\bonly\b"),
            ),
            (
                "conditional qualifier ('if' / 'when' / 'whenever')",
                re.compile(r"\b(?:if|when|whenever|in\s+case)\b"),
                re.compile(r"\b(?:if|when|whenever|in\s+case|should)\b"),
            ),
            (
                "exception qualifier ('unless' / 'except')",
                re.compile(r"\b(?:unless|except(?:\s+for)?|other\s+than|excluding)\b"),
                re.compile(r"\b(?:unless|except(?:\s+for)?|other\s+than|excluding|do\s+not)\b"),
            ),
            (
                "duration bound ('until')",
                re.compile(r"\buntil\b"),
                re.compile(r"\buntil\b"),
            ),
            (
                "temporal qualifier ('before' / 'prior to')",
                re.compile(r"(?<!\bas\s)\b(?:before|prior\s+to)\b"),
                re.compile(r"\b(?:before|prior\s+to)\b"),
            ),
            (
                "temporal qualifier ('after')",
                re.compile(r"\b(?:after|following)\b"),
                re.compile(r"\b(?:after|following)\b"),
            ),
            (
                "concurrent qualifier ('while')",
                re.compile(r"\bwhile\b"),
                re.compile(r"\b(?:while|during|when)\b"),
            ),
            (
                "PRN / as-needed qualifier",
                re.compile(r"\b(?:as\s+needed|p\.?r\.?n\.?|if\s+needed|when\s+needed)\b"),
                re.compile(
                    r"\b(?:as\s+needed|p\.?r\.?n\.?|if\s+(?:you\s+)?need(?:ed)?|when\s+(?:you\s+)?need(?:ed)?|need(?:ed)?\s+it)\b"
                ),
            ),
            (
                "upper threshold ('more than' / 'greater than' / 'exceeds' / 'above' / '>')",
                re.compile(r"(?:\b(?:more\s+than|greater\s+than|over|above|exceeds?|exceeding)\b|>)"),
                re.compile(r"(?:\b(?:more\s+than|greater\s+than|over|above|exceeds?|exceeding)\b|>)"),
            ),
            (
                "lower threshold ('less than' / 'below' / 'under' / '<')",
                re.compile(r"(?:\b(?:less\s+than|under|below|fewer\s+than)\b|<)"),
                re.compile(r"(?:\b(?:less\s+than|under|below|fewer\s+than)\b|<)"),
            ),
            (
                "minimum threshold ('at least')",
                re.compile(r"\b(?:at\s+least|minimum\s+of)\b"),
                re.compile(r"\b(?:at\s+least|minimum\s+of)\b"),
            ),
            (
                "maximum threshold ('no more than' / 'at most')",
                re.compile(r"\b(?:no\s+more\s+than|at\s+most|maximum\s+of)\b"),
                re.compile(r"\b(?:no\s+more\s+than|at\s+most|maximum\s+of)\b"),
            ),
            (
                "bounded timeframe qualifier ('within')",
                re.compile(r"\b(?:within|no\s+later\s+than|in\s+under)\b"),
                re.compile(r"\b(?:within|no\s+later\s+than|in\s+under)\b"),
            ),
            (
                "finite course duration ('for X days/weeks/months')",
                re.compile(
                    r"\bfor\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fourteen)\s*(?:hours?|hrs?|days?|weeks?|wks?|months?|mos?)\b"
                ),
                re.compile(
                    r"\bfor\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fourteen)\s*(?:hours?|hrs?|days?|weeks?|wks?|months?|mos?)\b"
                ),
            ),
            (
                "sequential directive ('then')",
                re.compile(r"\b(?:then|afterward|thereafter)\b"),
                re.compile(r"\b(?:then|afterward|thereafter)\b"),
            ),
            (
                "alternative directive ('otherwise')",
                re.compile(r"\botherwise\b"),
                re.compile(r"\botherwise\b"),
            ),
            (
                "stop/discontinue directive",
                re.compile(r"\b(?:stop|discontinue|d/c)\b"),
                re.compile(r"\b(?:stop|discontinue|d/c)\b"),
            ),
            (
                "hold/withhold directive",
                re.compile(r"\b(?:hold|withhold|pause|skip)\b"),
                re.compile(r"\b(?:hold|withhold|pause|skip)\b"),
            ),
            (
                "resume directive",
                re.compile(r"\bresum(?:e|ing)\b"),
                re.compile(r"\bresum(?:e|ing)\b"),
            ),
            (
                "negation directive ('do not' / 'never' / 'not')",
                re.compile(r"\b(?:do\s+not|don'?t|never|not|no\s+longer|avoid)\b"),
                re.compile(
                    r"\b(?:do\s+not|don'?t|never|not|no\s+longer|avoid|stop|unless|except)\b"
                ),
            ),
        )
        for rule_label, src_pat, sum_pat in qualifier_rules:
            if src_pat.search(ev_effective) and not sum_pat.search(summary_norm):
                _flag_loss(f"dropped {rule_label}")
                return

        # 3. Numeric thresholds, durations, and quantities in ev_effective must be preserved
        ev_nums = set(RE_NUMBER_TOKEN.findall(ev_effective))
        sum_nums = set(RE_NUMBER_TOKEN.findall(summary_norm))
        for w_num, digit in WORD_NUMBER_MAP.items():
            if w_num not in ("a", "an") and _has_word_boundary_match(w_num, summary_norm):
                sum_nums.add(str(digit))
        tf_equivalent = (
            normalize_timeframe(ev_effective) is not None
            and normalize_timeframe(ev_effective) == normalize_timeframe(summary_norm)
        )
        for num_tok in ev_nums:
            if num_tok not in sum_nums and not tf_equivalent:
                _flag_loss(f"omitted numeric value '{num_tok}'")
                return

        # 4. Follow-up / task timeframe qualifier and normalized value preservation
        if isinstance(item, (FollowUpInstruction, CareTaskItem)):
            ev_tf = normalize_timeframe(ev_effective)
            sum_tf = normalize_timeframe(summary_norm)
            if ev_tf is not None and sum_tf != ev_tf:
                _flag_loss(f"timeframe changed from '{ev_tf}' to '{sum_tf}'")
                return

        # 5. Warning-sign urgency cues and critical red-flag symptoms preservation
        if isinstance(item, WarningSignInstruction):
            if EMERGENT_CUE_REGEX.search(ev_effective) and not EMERGENT_CUE_REGEX.search(
                summary_norm
            ):
                _flag_loss("dropped emergency action cue")
                return
            if URGENT_CUE_REGEX.search(ev_effective) and not (
                URGENT_CUE_REGEX.search(summary_norm)
                or EMERGENT_CUE_REGEX.search(summary_norm)
            ):
                _flag_loss("dropped urgent action cue")
                return
            for rf_pat in CRITICAL_RED_FLAG_PATTERNS:
                if rf_pat.search(ev_effective) and not rf_pat.search(summary_norm):
                    _flag_loss("omitted critical red-flag symptom from summary")
                    return

        # 6. Content tokens inside qualifying clauses (conditions, indications, exceptions, timing)
        qualifier_stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "to",
            "of",
            "in",
            "on",
            "at",
            "by",
            "with",
            "it",
            "you",
            "your",
            "or",
            "and",
            "as",
            "if",
            "when",
            "whenever",
            "only",
            "unless",
            "except",
            "until",
            "before",
            "prior",
            "after",
            "while",
            "for",
            "needed",
            "need",
            "noted",
            "note",
            "happens",
            "happen",
            "occurs",
            "occur",
            "present",
            "day",
            "days",
            "week",
            "weeks",
            "month",
            "months",
            "hour",
            "hours",
            "time",
            "times",
            "daily",
            "every",
            "once",
            "twice",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "twelve",
            "fourteen",
            "more",
            "less",
            "greater",
            "fewer",
            "than",
            "above",
            "below",
            "over",
            "under",
            "up",
            "exceeds",
            "exceeding",
        }
        sum_word_tokens = set(RE_WORD_TOKEN.findall(summary_norm))
        sum_stems = {self._stem_token(t) for t in sum_word_tokens}
        synonym_groups = (
            frozenset({"lb", "lbs", "pound", "pounds"}),
            frozenset({"dizzy", "dizziness"}),
            frozenset({"seen", "see", "visit", "evaluated"}),
        )

        for m_qual in re.finditer(
            r"\b(?:only\s+if|if|when|whenever|unless|except(?:\s+for)?|until|(?<!as\s)before|prior\s+to|after|while|for)\s+([^.;,]+)",
            ev_effective,
        ):
            phrase = m_qual.group(1)
            for w in RE_WORD_TOKEN.findall(phrase):
                if w in qualifier_stop_words or RE_NUMBER_TOKEN.fullmatch(w):
                    continue
                w_stem = self._stem_token(w)
                matched = w in sum_word_tokens or w_stem in sum_stems
                if not matched:
                    for s_grp in synonym_groups:
                        if w in s_grp and any(alt in sum_word_tokens for alt in s_grp):
                            matched = True
                            break
                if not matched:
                    _flag_loss(f"omitted qualifying detail '{w}'")
                    return

    def _verify_free_text_whitelist(
        self,
        item: ExtractedItemBase,
        free_text: str,
        field_label: str,
        ev_norm: str,
        *,
        allow_template_vocab: bool,
    ) -> None:
        """Requirement 6: Closed-vocabulary whitelist check for free-text fields.

        1. Non-empty free_text with no verified alphanumeric tokens (e.g. emoji-only '💊🩺❤️') -> UNSUPPORTED_FIELD_VALUE.
        2. Every dose (number, unit) in free_text must be in ev_norm's parsed doses.
        3. Every number in free_text must be a subset of numbers in ev_norm.
        4. Every known drug name in free_text must have an equivalent in ev_norm.
        5. Every word token in free_text must appear in ev_norm (or in CLOSED_TEMPLATE_VOCABULARY when allow_template_vocab=True).
        """
        text_norm = canonicalize_text(free_text)
        word_tokens = RE_WORD_TOKEN.findall(text_norm)

        if not word_tokens or not ev_norm:
            item.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                    severity=HazardSeverity.CRITICAL,
                    message=(
                        f"Field '{field_label}' on '{item.id}' contains non-textual or "
                        "unverifiable content not supported by source evidence."
                    ),
                    item_id=item.id,
                    requires_human_review=True,
                    evidence_snippet=free_text[:200],
                )
            )
            return

        # Check exact doses
        ev_doses = set(parse_doses(ev_norm))
        for dose_tuple in parse_doses(text_norm):
            if dose_tuple not in ev_doses:
                item.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Field '{field_label}' on '{item.id}' contains dosage "
                            f"'{dose_tuple[0]} {dose_tuple[1]}' not supported by source evidence."
                        ),
                        item_id=item.id,
                        requires_human_review=True,
                        evidence_snippet=free_text[:200],
                    )
                )
                return

        # Check numerals subset (accounting for word numbers in evidence like 'one' -> '1')
        ev_numbers = set(RE_NUMBER_TOKEN.findall(ev_norm))
        for w_num, digit in WORD_NUMBER_MAP.items():
            if _has_word_boundary_match(w_num, ev_norm):
                ev_numbers.add(str(digit))
        for num_tok in RE_NUMBER_TOKEN.findall(text_norm):
            if num_tok not in ev_numbers:
                item.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Field '{field_label}' on '{item.id}' introduces numeral '{num_tok}' "
                            "not present in source evidence."
                        ),
                        item_id=item.id,
                        requires_human_review=True,
                        evidence_snippet=free_text[:200],
                    )
                )
                return

        # Check drug names subset
        for alias, pat in ALL_KNOWN_DRUG_PATTERNS:
            if pat.search(text_norm):
                equivalents = get_medication_equivalents(alias)
                if not any(
                    _has_word_boundary_match(eq, ev_norm) for eq in equivalents
                ):
                    item.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Field '{field_label}' on '{item.id}' references medication "
                                f"'{alias}' not supported by the item's source evidence."
                            ),
                            item_id=item.id,
                            requires_human_review=True,
                            evidence_snippet=free_text[:200],
                        )
                    )
                    return

        # Check frequency classes if any frequency phrase appears in free_text
        ft_freq_classes, _ = parse_frequency_classes(text_norm)
        ev_freq_classes, _ = parse_frequency_classes(ev_norm)
        if ft_freq_classes and not ft_freq_classes.issubset(ev_freq_classes):
            item.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                    severity=HazardSeverity.HIGH,
                    message=(
                        f"Field '{field_label}' on '{item.id}' introduces frequency class "
                        f"{sorted(ft_freq_classes)} not supported by source evidence ({sorted(ev_freq_classes)})."
                    ),
                    item_id=item.id,
                    requires_human_review=True,
                    evidence_snippet=free_text[:200],
                )
            )
            return

        # Check routes if any route phrase appears in free_text
        for group in ROUTE_EQUIVALENCE_GROUPS:
            if any(_has_word_boundary_match(r, text_norm) for r in group):
                if not any(_has_word_boundary_match(r, ev_norm) for r in group):
                    item.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.HIGH,
                            message=(
                                f"Field '{field_label}' on '{item.id}' introduces a route of "
                                "administration not supported by source evidence."
                            ),
                            item_id=item.id,
                            requires_human_review=True,
                            evidence_snippet=free_text[:200],
                        )
                    )
                    return

        # Whitelist token check: every word token in free_text must be in ev_norm or CLOSED_TEMPLATE_VOCABULARY
        ev_tokens = set(RE_WORD_TOKEN.findall(ev_norm))
        ev_stems = {self._stem_token(t) for t in ev_tokens}
        allowed_tokens = set(ev_tokens)
        if isinstance(item, MedicationInstruction):
            allowed_tokens |= get_medication_equivalents(item.medication_name)
        if allow_template_vocab:
            allowed_tokens |= CLOSED_TEMPLATE_VOCABULARY

        for tok in word_tokens:
            if tok in allowed_tokens or self._stem_token(tok) in ev_stems:
                continue
            if RE_NUMBER_TOKEN.fullmatch(tok) and tok in ev_numbers:
                continue
            item.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                    severity=HazardSeverity.CRITICAL,
                    message=(
                        f"Field '{field_label}' on '{item.id}' contains unsupported token "
                        f"'{tok}' outside the source evidence and closed template vocabulary."
                    ),
                    item_id=item.id,
                    requires_human_review=True,
                    evidence_snippet=free_text[:200],
                )
            )
            return

    def _validate_medication(
        self,
        med: MedicationInstruction,
        normalized_doc: str,
        evidence_verified: bool,
    ) -> None:
        """Audit a MedicationInstruction with clause-bound, token-exact field & action verification."""
        source_quote = (
            med.source_evidence.source_text if med.source_evidence is not None else ""
        )
        ev_norm = canonicalize_text(source_quote) if evidence_verified else ""
        ev_stripped_negation = _strip_negated_clauses(ev_norm) if ev_norm else ""

        # Check for uncertain negation or drug-specific negation in source quote
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

        med_equivalents = get_medication_equivalents(med.medication_name)
        med_supported_by_evidence = False
        clause_norm = ev_norm
        is_multi_drug_ev = False

        if evidence_verified and ev_norm:
            med_supported_by_evidence = any(
                _has_word_boundary_match(eq, ev_norm) for eq in med_equivalents
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
            else:
                clause_norm, is_multi_drug_ev = _extract_drug_clauses(
                    ev_norm, med_equivalents
                )

            # Check if source quote is a truncated substring of a negated sentence in raw_doc
            # e.g. raw_doc has "Do not stop aspirin 81 mg daily." and source_quote is "stop aspirin 81 mg daily."
            if normalized_doc and ev_norm in normalized_doc:
                for neg_prefix in (
                    "do not ",
                    "don't ",
                    "should not ",
                    "must not ",
                    "not ",
                ):
                    if f"{neg_prefix}{ev_norm}" in normalized_doc:
                        med.safety_flags.append(
                            SafetyFlag(
                                code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                                severity=HazardSeverity.CRITICAL,
                                message=(
                                    f"Source quote for '{med.medication_name}' strips a governing "
                                    f"negation prefix ('{neg_prefix.strip()}') present in the discharge text."
                                ),
                                item_id=med.id,
                                requires_human_review=True,
                                evidence_snippet=source_quote,
                            )
                        )
                        break

            # Requirement 3 & 5: Verify MedicationAction is supported by the medication's clause
            self._verify_medication_action_support(
                med=med,
                clause_norm=_strip_negated_clauses(clause_norm),
                raw_ev_norm=ev_norm,
                source_quote=source_quote,
            )

            # Requirement 2a: Token-exact and clause-bound dosage verification
            if not _is_missing_or_unknown(med.dosage):
                assert med.dosage is not None
                extracted_doses = parse_doses(med.dosage)
                clause_doses = parse_doses(clause_norm)
                if not extracted_doses or not all(
                    d in clause_doses for d in extracted_doses
                ):
                    med.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Extracted dosage '{med.dosage}' for '{med.medication_name}' "
                                "is not supported by the medication's clause in source evidence."
                            ),
                            item_id=med.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )
                else:
                    # Check if the medication's clause contains an additional/conditional dose
                    # (e.g., 'Take furosemide 40 mg daily, and take 80 mg on days when weight is up more than 3 lb')
                    # that was omitted from extracted dosage/previous_dosage/indication_or_note
                    prev_doses_set = set(parse_doses(med.previous_dosage or ""))
                    note_doses_set = set(parse_doses(med.indication_or_note or ""))
                    has_change_verb = any(
                        p.search(clause_norm) for p in MED_CHANGE_PATTERNS
                    )
                    unaccounted_doses = [
                        d
                        for d in clause_doses
                        if d not in extracted_doses
                        and d not in prev_doses_set
                        and d not in note_doses_set
                    ]
                    if unaccounted_doses and not has_change_verb:
                        med.safety_flags.append(
                            SafetyFlag(
                                code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                                severity=HazardSeverity.CRITICAL,
                                message=(
                                    f"Extracted dosage '{med.dosage}' for '{med.medication_name}' omits "
                                    f"secondary or conditional dose(s) {unaccounted_doses} present in the medication's clause."
                                ),
                                item_id=med.id,
                                requires_human_review=True,
                                evidence_snippet=source_quote,
                            )
                        )
                        med.safety_flags.append(
                            SafetyFlag(
                                code=HazardCode.OMITTED_INSTRUCTION,
                                severity=HazardSeverity.CRITICAL,
                                message=(
                                    f"Secondary or conditional dosing instruction {unaccounted_doses} for "
                                    f"'{med.medication_name}' was omitted from extraction."
                                ),
                                item_id=med.id,
                                requires_human_review=True,
                                evidence_snippet=source_quote,
                            )
                        )
                        med.safety_flags.append(
                            SafetyFlag(
                                code=HazardCode.INCOMPLETE_EXTRACTION,
                                severity=HazardSeverity.CRITICAL,
                                message=(
                                    f"Incomplete extraction of multi-part dosing regimen for '{med.medication_name}'."
                                ),
                                item_id=med.id,
                                requires_human_review=True,
                                evidence_snippet=source_quote,
                            )
                        )

            # Check if a finite duration ('for 5 days') or conditional hold/exception ('unless ...', 'until ...', 'if heart rate ...')
            # in the medication's clause was omitted from structured fields (frequency / indication_or_note)
            if med_supported_by_evidence and clause_norm:
                structured_qual_text = canonicalize_text(
                    f"{med.frequency or ''} {med.indication_or_note or ''}"
                )
                finite_dur_match = re.search(
                    r"\bfor\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fourteen)\s*(?:days?|weeks?|months?)\b",
                    clause_norm,
                )
                cond_exception_match = re.search(
                    r"\b(?:unless|except(?:\s+for)?|until|on\s+days\s+when|if\s+(?:heart\s+rate|blood\s+pressure|bp|hr|pulse|weight|serum|potassium|creatinine|glucose|blood\s+sugar|dizzy|lightheaded|systolic|diastolic))\b",
                    clause_norm,
                )
                if finite_dur_match and not _has_word_boundary_match(
                    finite_dur_match.group(0), structured_qual_text
                ):
                    med.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.OMITTED_INSTRUCTION,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Finite course duration ('{finite_dur_match.group(0)}') for "
                                f"'{med.medication_name}' was omitted from structured medication fields."
                            ),
                            item_id=med.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )
                    med.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.INCOMPLETE_EXTRACTION,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Incomplete extraction: finite duration '{finite_dur_match.group(0)}' "
                                f"omitted for '{med.medication_name}'."
                            ),
                            item_id=med.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )
                if cond_exception_match and not _has_word_boundary_match(
                    cond_exception_match.group(0), structured_qual_text
                ):
                    med.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.OMITTED_INSTRUCTION,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Conditional administration/hold/exception qualifier ('{cond_exception_match.group(0)}...') "
                                f"for '{med.medication_name}' was omitted from structured medication fields."
                            ),
                            item_id=med.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )
                    med.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.INCOMPLETE_EXTRACTION,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Incomplete extraction: clinical condition/exception omitted for '{med.medication_name}'."
                            ),
                            item_id=med.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )

            # Verify previous_dosage support (token-exact in clause)
            if not _is_missing_or_unknown(med.previous_dosage):
                assert med.previous_dosage is not None
                prev_doses = parse_doses(med.previous_dosage)
                clause_doses = parse_doses(clause_norm)
                if not prev_doses or not all(d in clause_doses for d in prev_doses):
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

            # Verify route support in clause (or shared sentence if single route)
            if not _is_missing_or_unknown(med.route):
                assert med.route is not None
                route_norm = canonicalize_text(med.route)
                target_ev_for_route = clause_norm if is_multi_drug_ev else ev_norm
                route_supported = _has_word_boundary_match(
                    route_norm, target_ev_for_route
                )
                if not route_supported:
                    for group in ROUTE_EQUIVALENCE_GROUPS:
                        if route_norm in group and any(
                            _has_word_boundary_match(candidate, target_ev_for_route)
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

            # Requirement 2b: Canonical frequency class comparison ('daily' is NOT supported by 'twice daily')
            if not _is_missing_or_unknown(med.frequency):
                assert med.frequency is not None
                freq_norm = canonicalize_text(med.frequency)
                ext_classes, ext_mods = parse_frequency_classes(freq_norm)
                # In multi-drug sentences, check the drug's clause first; if the drug's clause has no
                # frequency at all, allow a shared sentence-level frequency only if no other frequency conflicts
                cl_classes, cl_mods = parse_frequency_classes(clause_norm)
                ev_classes, ev_mods = parse_frequency_classes(ev_norm)
                ref_classes = cl_classes if cl_classes else ev_classes
                ref_mods = cl_mods if cl_mods else ev_mods

                freq_supported = False
                if ext_classes:
                    # Canonical classes must match exactly, and any extracted modifiers must be present
                    if ext_classes == ref_classes and ext_mods.issubset(ref_mods):
                        freq_supported = True
                else:
                    # Non-standard frequency phrase (e.g. 'as needed if needed'): require exact word-boundary match
                    freq_supported = _has_word_boundary_match(freq_norm, clause_norm)

                if not freq_supported:
                    med.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.HIGH,
                            message=(
                                f"Extracted frequency '{med.frequency}' (classes={sorted(ext_classes)}) "
                                f"for '{med.medication_name}' is not supported by source evidence "
                                f"(classes={sorted(ref_classes)})."
                            ),
                            item_id=med.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )

            # Verify indication_or_note support in source_evidence
            if not _is_missing_or_unknown(med.indication_or_note):
                assert med.indication_or_note is not None
                note_norm = canonicalize_text(med.indication_or_note)
                has_unsupported_mod = (
                    any(p.search(note_norm) for p in PROHIBITED_CLINICAL_PATTERNS)
                    or any(p.search(note_norm) for p in MED_CHANGE_PATTERNS)
                    or any(p.search(note_norm) for p in STOP_START_PATTERNS)
                ) and not _has_word_boundary_match(note_norm, clause_norm)
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
                if not _has_word_boundary_match(note_norm, clause_norm):
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

        # 1. Stop / Start / Hold medication detection
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

        # 2. Medication change detection (ONLY when supported by source evidence)
        source_has_dose_change = bool(
            ev_stripped_negation
            and any(p.search(ev_stripped_negation) for p in MED_CHANGE_PATTERNS)
        )
        prev_dose_supported = bool(
            not _is_missing_or_unknown(med.previous_dosage)
            and clause_norm
            and all(
                d in parse_doses(clause_norm)
                for d in parse_doses(med.previous_dosage or "")
            )
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
                or (med.dosage is not None and len(parse_doses(med.dosage)) == 0)
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

        # 5. Intra-document silent conflict check (including brand/generic aliases, d/c, hold, stop, or conflicting dose in medication list)
        doc_without_negations = _strip_negated_clauses(normalized_doc)
        if med.action == MedicationAction.CONTINUE:
            for alias in med_equivalents:
                alias_esc = re.escape(alias)
                contradictory_stop = re.search(
                    rf"(?<![a-z0-9/-])(?:stop(?:\s+taking)?|discontinu(?:e|ed|ing)|d/c(?:'?d)?|dc(?:'d)?|do\s+not\s+take|hold|withhold)\s+{alias_esc}(?![a-z0-9/-])",
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

        # Also check if another sentence/clause in the document specifies a conflicting dose for the same medication
        if not _is_missing_or_unknown(med.dosage) and not any(
            f.code == HazardCode.CONFLICTING_INSTRUCTIONS for f in med.safety_flags
        ):
            ext_doses_set = set(parse_doses(med.dosage or ""))
            prev_doses_set = set(parse_doses(med.previous_dosage or ""))
            if ext_doses_set:
                for raw_line in normalized_doc.splitlines():
                    for sent_part in SENTENCE_SPLIT_REGEX.split(raw_line):
                        sent_clean = _strip_negated_clauses(sent_part)
                        if not sent_clean:
                            continue
                        if any(
                            _has_word_boundary_match(alias, sent_clean)
                            for alias in med_equivalents
                        ):
                            bound_cl, _ = _extract_drug_clauses(
                                sent_clean, med_equivalents
                            )
                            cl_doses = set(parse_doses(bound_cl))
                            has_change_in_sent = any(
                                p.search(bound_cl) for p in MED_CHANGE_PATTERNS
                            )
                            if (
                                cl_doses
                                and not cl_doses.issubset(ext_doses_set | prev_doses_set)
                                and not has_change_in_sent
                            ):
                                med.safety_flags.append(
                                    SafetyFlag(
                                        code=HazardCode.CONFLICTING_INSTRUCTIONS,
                                        severity=HazardSeverity.CRITICAL,
                                        message=(
                                            f"Conflicting dosage instructions detected in discharge text for "
                                            f"'{med.medication_name}': extracted {sorted(ext_doses_set)} vs "
                                            f"document clause with {sorted(cl_doses)}."
                                        ),
                                        item_id=med.id,
                                        requires_human_review=True,
                                        evidence_snippet=source_quote or None,
                                    )
                                )
                                break

    def _verify_medication_action_support(
        self,
        med: MedicationInstruction,
        clause_norm: str,
        raw_ev_norm: str,
        source_quote: str,
    ) -> None:
        """Requirement 3 & 5: Verify MedicationAction is explicitly supported by the evidence clause."""
        if not clause_norm:
            return

        # If raw_ev_norm has drug-specific negation ('not taking metformin', 'do not take ibuprofen', 'should not be stopped')
        has_drug_negation = any(
            p.search(raw_ev_norm) for p in DRUG_SPECIFIC_NEGATION_PATTERNS
        )
        if has_drug_negation:
            # 'do not stop aspirin' or 'aspirin should not be stopped' contradicts STOP_DISCONTINUE
            if med.action == MedicationAction.STOP_DISCONTINUE and re.search(
                r"\b(?:do\s+not\s+stop|don'?t\s+stop|should\s+not\s+be\s+stopped|must\s+not\s+be\s+stopped)\b",
                raw_ev_norm,
            ):
                med.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Action '{med.action.value}' on '{med.medication_name}' contradicts "
                            "negated stop instruction in source evidence."
                        ),
                        item_id=med.id,
                        requires_human_review=True,
                        evidence_snippet=source_quote,
                    )
                )
                return
            # 'do not take ibuprofen' or 'patient is not taking metformin' contradicts CONTINUE or NEW_START
            if med.action in (
                MedicationAction.CONTINUE,
                MedicationAction.NEW_START,
            ) and re.search(
                r"\b(?:do\s+not\s+take|don'?t\s+take|not\s+taking|is\s+not\s+on|never\s+took)\b",
                raw_ev_norm,
            ):
                med.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Action '{med.action.value}' on '{med.medication_name}' is not supported "
                            "by negated medication statement in source evidence."
                        ),
                        item_id=med.id,
                        requires_human_review=True,
                        evidence_snippet=source_quote,
                    )
                )
                return

        has_continue_cue = bool(CONTINUE_CUE_PATTERN.search(clause_norm))
        has_start_cue = bool(NEW_START_CUE_PATTERN.search(clause_norm))
        has_stop_cue = bool(STOP_DISCONTINUE_CUE_PATTERN.search(clause_norm))
        has_hold_cue = bool(HOLD_CUE_PATTERN.search(clause_norm))
        has_change_cue = any(p.search(clause_norm) for p in MED_CHANGE_PATTERNS)

        if med.action == MedicationAction.CONTINUE:
            # CONTINUE only with continue/take/resume; NEVER with start/begin/initiate/add/new/stop/hold/d/c
            if (
                not has_continue_cue
                or has_start_cue
                or has_stop_cue
                or has_hold_cue
                or has_change_cue
            ):
                med.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Medication action 'CONTINUE' for '{med.medication_name}' is not "
                            "supported by an explicit continue/take/resume cue in the medication's clause "
                            "(or conflicts with start/stop/hold/change cues)."
                        ),
                        item_id=med.id,
                        requires_human_review=True,
                        evidence_snippet=source_quote,
                    )
                )
        elif med.action == MedicationAction.NEW_START:
            if not has_start_cue or has_stop_cue or has_hold_cue:
                med.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Medication action 'NEW_START' for '{med.medication_name}' is not "
                            "supported by a start/begin/initiate/add cue in source evidence."
                        ),
                        item_id=med.id,
                        requires_human_review=True,
                        evidence_snippet=source_quote,
                    )
                )
        elif med.action == MedicationAction.STOP_DISCONTINUE:
            if not has_stop_cue:
                med.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Medication action 'STOP_DISCONTINUE' for '{med.medication_name}' is not "
                            "supported by a stop/discontinue/d/c cue in source evidence."
                        ),
                        item_id=med.id,
                        requires_human_review=True,
                        evidence_snippet=source_quote,
                    )
                )
        elif med.action == MedicationAction.HOLD_TEMPORARY:
            if not has_hold_cue:
                med.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Medication action 'HOLD_TEMPORARY' for '{med.medication_name}' is not "
                            "supported by a hold/withhold/pause cue in source evidence."
                        ),
                        item_id=med.id,
                        requires_human_review=True,
                        evidence_snippet=source_quote,
                    )
                )

    def _validate_follow_up(
        self, follow_up: FollowUpInstruction, evidence_verified: bool
    ) -> None:
        """Audit a FollowUpInstruction for construct anchor, qualifier-preserving timeframe, and word-boundary field support."""
        source_quote = (
            follow_up.source_evidence.source_text
            if follow_up.source_evidence is not None
            else None
        )
        ev_norm = (
            canonicalize_text(source_quote)
            if evidence_verified and source_quote
            else ""
        )

        # Requirement 9: Require an explicit follow-up/referral/appointment construct (not 'see'/'clinic'/'visit' alone)
        if evidence_verified and ev_norm:
            if not FOLLOWUP_CONSTRUCT_REGEX.search(ev_norm):
                follow_up.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.MISSING_SOURCE_EVIDENCE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Source evidence for follow-up '{follow_up.id}' does not contain "
                            "an explicit follow-up, referral, or appointment construct."
                        ),
                        item_id=follow_up.id,
                        requires_human_review=True,
                        evidence_snippet=source_quote,
                    )
                )

            # Requirement 2d: Specialty word-boundary matching
            if not _is_missing_or_unknown(follow_up.specialty):
                assert follow_up.specialty is not None
                if not _has_word_boundary_match(follow_up.specialty, ev_norm):
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

            # Requirement 2c: Timeframe support & qualifier preservation
            if not _is_missing_or_unknown(follow_up.timeframe):
                assert follow_up.timeframe is not None
                tf_norm = canonicalize_text(follow_up.timeframe)
                ext_qual = _extract_timeframe_qualifier(tf_norm)
                ev_qual = _extract_timeframe_qualifier(ev_norm)

                # Both verbatim word-boundary match AND qualifier consistency are required
                tf_verbatim = _has_word_boundary_match(tf_norm, ev_norm)
                qual_matches = ext_qual == ev_qual
                if not tf_verbatim or not qual_matches:
                    follow_up.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                            severity=HazardSeverity.HIGH,
                            message=(
                                f"Extracted timeframe '{follow_up.timeframe}' (qualifier={ext_qual}) "
                                f"on '{follow_up.id}' is not supported by source evidence (qualifier={ev_qual})."
                            ),
                            item_id=follow_up.id,
                            requires_human_review=True,
                            evidence_snippet=source_quote,
                        )
                    )

            # Provider name word-boundary matching
            if not _is_missing_or_unknown(follow_up.provider_name):
                assert follow_up.provider_name is not None
                if not _has_word_boundary_match(follow_up.provider_name, ev_norm):
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

            # Reason word-boundary and whitelist matching
            if not _is_missing_or_unknown(follow_up.reason):
                assert follow_up.reason is not None
                if not _has_word_boundary_match(follow_up.reason, ev_norm):
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

        # 1. Missing or unparseable/ambiguous follow-up timeframe (including APPROX like 'in about 2 weeks')
        canonical_tf = normalize_timeframe(follow_up.timeframe)
        if (
            _is_missing_or_unknown(follow_up.timeframe)
            or canonical_tf is None
            or canonical_tf.startswith("APPROX:")
        ):
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
            and canonicalize_text(follow_up.specialty) in VAGUE_SPECIALTY_TOKENS
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

    def _validate_warning_sign(
        self, warning: WarningSignInstruction, evidence_verified: bool
    ) -> None:
        """Requirement 8: Require ALL symptom content tokens in evidence and enforce minimum urgency cues."""
        source_quote = (
            warning.source_evidence.source_text
            if warning.source_evidence is not None
            else None
        )
        ev_norm = (
            canonicalize_text(source_quote)
            if evidence_verified and source_quote
            else ""
        )
        symptom_text = warning.symptom_description or ""
        symptom_norm = canonicalize_text(symptom_text)
        combined_text = f"{symptom_norm} {ev_norm}"

        if evidence_verified and ev_norm:
            stop_words = {"with", "from", "that", "this", "your", "have", "when", "noted", "and", "or", "for", "the", "of", "in", "to"}
            symp_words = [
                w
                for w in RE_WORD_TOKEN.findall(symptom_norm)
                if w not in stop_words
            ]
            # Requirement 8: Require ALL symptom content tokens to appear in evidence
            if not symp_words or not all(
                _has_word_boundary_match(w, ev_norm) for w in symp_words
            ):
                warning.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.MISSING_SOURCE_EVIDENCE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Source evidence for warning sign '{warning.id}' does not support "
                            f"all content tokens in symptom '{warning.symptom_description}'."
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
                            f"Symptom description '{warning.symptom_description}' contains tokens "
                            "not supported by the item's source evidence."
                        ),
                        item_id=warning.id,
                        requires_human_review=True,
                        evidence_snippet=source_quote,
                    )
                )

            if not _is_missing_or_unknown(warning.action_to_take):
                assert warning.action_to_take is not None
                if not _has_word_boundary_match(warning.action_to_take, ev_norm):
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

            # Requirement 8: Urgency must not be lower than the strongest cue in evidence
            required_min_urgency = UrgencyLevel.ROUTINE
            if EMERGENT_CUE_REGEX.search(ev_norm) or any(
                p.search(ev_norm) for p in CRITICAL_RED_FLAG_PATTERNS
            ):
                required_min_urgency = UrgencyLevel.EMERGENT_911
            elif URGENT_CUE_REGEX.search(ev_norm):
                required_min_urgency = UrgencyLevel.URGENT

            if URGENCY_RANK[warning.urgency] < URGENCY_RANK[required_min_urgency]:
                warning.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Warning sign '{warning.id}' urgency '{warning.urgency.value}' is lower "
                            f"than the urgency cue in source evidence ('{required_min_urgency.value}')."
                        ),
                        item_id=warning.id,
                        requires_human_review=True,
                        evidence_snippet=source_quote,
                    )
                )
                warning.safety_flags.append(
                    SafetyFlag(
                        code=HazardCode.WARNING_SIGN_ESCALATION,
                        severity=HazardSeverity.CRITICAL,
                        message=(
                            f"Under-triaged warning sign '{warning.id}': evidence requires at least "
                            f"{required_min_urgency.value}."
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

        if is_red_flag and not any(
            f.code == HazardCode.WARNING_SIGN_ESCALATION for f in warning.safety_flags
        ):
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
        self,
        task: CareTaskItem,
        payload: DischargeExtractionPayload,
        evidence_verified: bool,
    ) -> None:
        """Requirement 6 & 9: Verify CareTaskItem whitelist, due_window, and linked_item_id + category match."""
        # Verify linked_item_id exists and category matches
        linked_key = _canonical_id_key(task.linked_item_id)
        med_ids = {_canonical_id_key(m.id) for m in payload.medications}
        fu_ids = {_canonical_id_key(f.id) for f in payload.follow_ups}
        ws_ids = {_canonical_id_key(w.id) for w in payload.warning_signs}

        expected_category: str | None = None
        if linked_key in med_ids:
            expected_category = "MEDICATION"
        elif linked_key in fu_ids:
            expected_category = "FOLLOW_UP"
        elif linked_key in ws_ids:
            expected_category = "WARNING_SIGN"

        if expected_category is None:
            task.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                    severity=HazardSeverity.HIGH,
                    message=(
                        f"Task '{task.id}' references non-existent linked_item_id "
                        f"'{task.linked_item_id}'."
                    ),
                    item_id=task.id,
                    requires_human_review=True,
                )
            )
        elif task.category != expected_category and not (
            expected_category == "FOLLOW_UP"
            and task.category in ("FOLLOW_UP", "LAB_TEST")
        ):
            task.safety_flags.append(
                SafetyFlag(
                    code=HazardCode.UNSUPPORTED_FIELD_VALUE,
                    severity=HazardSeverity.HIGH,
                    message=(
                        f"Task '{task.id}' category '{task.category}' does not match "
                        f"linked item '{task.linked_item_id}' category ('{expected_category}')."
                    ),
                    item_id=task.id,
                    requires_human_review=True,
                )
            )

        if not evidence_verified or task.source_evidence is None:
            return
        ev_norm = canonicalize_text(task.source_evidence.source_text)

        if not _is_missing_or_unknown(task.due_window):
            assert task.due_window is not None
            due_norm = canonicalize_text(task.due_window)
            ext_qual = _extract_timeframe_qualifier(due_norm)
            ev_qual = _extract_timeframe_qualifier(ev_norm)
            if not _has_word_boundary_match(due_norm, ev_norm) or ext_qual != ev_qual:
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

        self._verify_free_text_whitelist(
            item=task,
            free_text=task.title,
            field_label="title",
            ev_norm=ev_norm,
            allow_template_vocab=True,
        )

    def _detect_duplicate_items(
        self, payload: DischargeExtractionPayload
    ) -> None:
        """Requirement 9: Case-insensitive ID uniqueness & semantic-duplicate detection across ALL item types."""
        all_items: list[ExtractedItemBase] = [
            *payload.medications,
            *payload.follow_ups,
            *payload.warning_signs,
            *payload.tasks,
        ]

        # 1. Group by case-insensitive canonical item ID
        by_id: dict[str, list[ExtractedItemBase]] = defaultdict(list)
        for item in all_items:
            by_id[_canonical_id_key(item.id)].append(item)

        for canon_id, group in by_id.items():
            if len(group) > 1:
                for dup in group:
                    dup.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.DUPLICATE_ITEM,
                            severity=HazardSeverity.HIGH,
                            message=(
                                f"Duplicate item ID '{dup.id}' (canonical '{canon_id}') detected "
                                f"({len(group)} occurrences). Human review required."
                            ),
                            item_id=dup.id,
                            requires_human_review=True,
                        )
                    )
                    self._finalize_item_status(dup)

        def _flag_dup_group(items_group: list[ExtractedItemBase], label: str) -> None:
            if len(items_group) <= 1:
                return
            for dup_item in items_group:
                if not any(
                    f.code == HazardCode.DUPLICATE_ITEM for f in dup_item.safety_flags
                ):
                    dup_item.safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.DUPLICATE_ITEM,
                            severity=HazardSeverity.HIGH,
                            message=(
                                f"Semantic duplicate {label} instruction ('{dup_item.id}') "
                                "detected. Preserving audit trail and escalating for review."
                            ),
                            item_id=dup_item.id,
                            requires_human_review=True,
                        )
                    )
                    self._finalize_item_status(dup_item)

        # 2. Medications semantic signature
        med_by_sig: dict[tuple[object, ...], list[ExtractedItemBase]] = defaultdict(list)
        for med in payload.medications:
            freq_cls, freq_mod = parse_frequency_classes(med.frequency or "")
            sig = (
                resolve_canonical_medication_name(med.medication_name),
                med.action.value,
                tuple(parse_doses(med.dosage or "")) or canonicalize_text(med.dosage or ""),
                (tuple(sorted(freq_cls)), tuple(sorted(freq_mod)))
                if freq_cls
                else canonicalize_text(med.frequency or ""),
            )
            med_by_sig[sig].append(med)
        for group in med_by_sig.values():
            _flag_dup_group(group, "medication")

        # 3. Follow-ups semantic signature (same specialty + identical raw timeframe + same provider + same reason)
        # Note: Distinct raw expressions of equivalent timeframes (e.g. 'within 1 week' vs 'within 7 days' in H32)
        # are tested for timeframe equivalence unless their raw timeframe text is also identical.
        fu_by_sig: dict[tuple[str, str, str, str], list[ExtractedItemBase]] = defaultdict(
            list
        )
        for fu in payload.follow_ups:
            sig_fu = (
                canonicalize_text(fu.specialty or ""),
                canonicalize_text(fu.timeframe or ""),
                canonicalize_text(fu.provider_name or ""),
                canonicalize_text(fu.reason or ""),
            )
            fu_by_sig[sig_fu].append(fu)
        for group in fu_by_sig.values():
            _flag_dup_group(group, "follow-up")

        # 4. Warning signs semantic signature
        ws_by_sig: dict[tuple[str, str, str], list[ExtractedItemBase]] = defaultdict(list)
        for ws in payload.warning_signs:
            sig_ws = (
                canonicalize_text(ws.symptom_description),
                ws.urgency.value,
                canonicalize_text(ws.action_to_take or ""),
            )
            ws_by_sig[sig_ws].append(ws)
        for group in ws_by_sig.values():
            _flag_dup_group(group, "warning-sign")

        # 5. Tasks semantic signature
        task_by_sig: dict[tuple[str, str, str, str], list[ExtractedItemBase]] = (
            defaultdict(list)
        )
        for t in payload.tasks:
            sig_t = (
                t.category,
                canonicalize_text(t.title),
                normalize_timeframe(t.due_window)
                or canonicalize_text(t.due_window or ""),
                _canonical_id_key(t.linked_item_id),
            )
            task_by_sig[sig_t].append(t)
        for group in task_by_sig.values():
            _flag_dup_group(group, "task")

    def _detect_cross_item_conflicts(
        self, payload: DischargeExtractionPayload
    ) -> None:
        """Detect conflicts across brand/generic medications and normalized follow-up timeframes."""
        meds_by_canonical: dict[str, list[MedicationInstruction]] = defaultdict(list)
        for med in payload.medications:
            key = resolve_canonical_medication_name(med.medication_name)
            meds_by_canonical[key].append(med)

        for canonical_name, group in meds_by_canonical.items():
            if len(group) < 2:
                continue
            actions = {m.action for m in group}
            dosages = {
                tuple(parse_doses(m.dosage))
                if parse_doses(m.dosage or "")
                else canonicalize_text(m.dosage or "")
                for m in group
                if not _is_missing_or_unknown(m.dosage)
            }
            frequencies = {
                parse_frequency_classes(m.frequency or "")[0]
                if parse_frequency_classes(m.frequency or "")[0]
                else canonicalize_text(m.frequency or "")
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
                    m.safety_status = SafetyStatus.CONFLICT_DETECTED
                    m.review_status = ReviewDecisionStatus.PENDING_HUMAN_REVIEW

        followups_by_specialty: dict[str, list[FollowUpInstruction]] = defaultdict(list)
        for fu in payload.follow_ups:
            if not _is_missing_or_unknown(fu.specialty):
                key = canonicalize_text(fu.specialty or "")
                followups_by_specialty[key].append(fu)

        for spec_name, group in followups_by_specialty.items():
            if len(group) < 2:
                continue
            canonical_timeframes = {
                normalize_timeframe(f.timeframe)
                or canonicalize_text(f.timeframe or "")
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
        """Requirement 4 & 5: Clause-bound omission detection (partial 'ev in sent' is rejected)."""
        all_items: list[ExtractedItemBase] = [
            *payload.medications,
            *payload.follow_ups,
            *payload.warning_signs,
            *payload.tasks,
        ]

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

        # Gather verified evidence snippets and extract covered drug identities per medication item
        med_evidence: list[str] = []
        extracted_med_canonicals: set[str] = set()
        extracted_med_equivalents: set[str] = set()
        for med in payload.medications:
            if med.source_evidence and not any(
                f.code == HazardCode.MISSING_SOURCE_EVIDENCE for f in med.safety_flags
            ):
                med_evidence.append(canonicalize_text(med.source_evidence.source_text))
                extracted_med_canonicals.add(
                    resolve_canonical_medication_name(med.medication_name)
                )
                extracted_med_equivalents.update(
                    get_medication_equivalents(med.medication_name)
                )

        fu_evidence: list[str] = []
        for fu in payload.follow_ups:
            if fu.source_evidence and not any(
                f.code == HazardCode.MISSING_SOURCE_EVIDENCE for f in fu.safety_flags
            ):
                fu_evidence.append(canonicalize_text(fu.source_evidence.source_text))
        for task in payload.tasks:
            if task.source_evidence and not any(
                f.code == HazardCode.MISSING_SOURCE_EVIDENCE for f in task.safety_flags
            ):
                fu_evidence.append(canonicalize_text(task.source_evidence.source_text))

        ws_evidence: list[str] = []
        for ws in payload.warning_signs:
            if ws.source_evidence and not any(
                f.code == HazardCode.MISSING_SOURCE_EVIDENCE for f in ws.safety_flags
            ):
                ws_evidence.append(canonicalize_text(ws.source_evidence.source_text))

        all_verified_evidence = med_evidence + fu_evidence + ws_evidence

        def _clause_is_covered(clause_norm: str, evidence_list: list[str]) -> bool:
            """Requirement 4: A clause is covered ONLY if an extracted item's evidence covers the whole clause."""
            c = clause_norm.strip(" .,;:-")
            if not c:
                return True
            return any(c in ev for ev in evidence_list)

        raw_lines = [line.strip() for line in raw_doc.splitlines() if line.strip()]
        sentences: list[tuple[str, bool]] = []
        for line in raw_lines:
            parts = [
                p.strip()
                for p in SENTENCE_SPLIT_REGEX.split(line)
                if p.strip()
            ]
            for part_idx, part in enumerate(parts):
                sentences.append((part, part_idx == 0))

        for sent, is_line_start in sentences:
            sent_norm = re.sub(
                r"^(?:[-*•]+|\d+[.)])\s*", "", canonicalize_text(sent)
            ).strip()
            if not sent_norm or ROUTINE_HEADER_REGEX.match(sent_norm):
                continue
            if "hemodynamically stable" in sent_norm:
                continue

            # Check uncertain negation in raw text
            if any(p.search(sent_norm) for p in UNCERTAIN_NEGATION_PATTERNS):
                if not _clause_is_covered(sent_norm, all_verified_evidence):
                    self._append_omission_flags(
                        payload,
                        f"Uncertain negation in discharge text was not reviewed: '{sent}'",
                        sent,
                    )
                continue

            # Check governed negation with an exception clause (e.g. 'No home medications were stopped except warfarin.')
            m_exc = re.search(
                r"\b(?:except(?:\s+for)?|other\s+than|unless|excluding)\s+([a-z0-9/-]+)",
                sent_norm,
            )
            if m_exc and (
                "medication" in sent_norm
                or any(p.search(sent_norm) for p in SENSITIVE_MED_OMISSION_PATTERNS)
                or bool(re.search(r"\b(?:continue|take|resume)\b", sent_norm))
            ):
                exc_drug = resolve_canonical_medication_name(m_exc.group(1))
                if not (
                    _clause_is_covered(sent_norm, med_evidence)
                    and exc_drug in extracted_med_canonicals
                ):
                    self._append_omission_flags(
                        payload,
                        f"Medication exception clause ('{m_exc.group(0)}') omitted from extraction: '{sent}'",
                        sent,
                    )
                    continue

            sent_stripped_neg = _strip_negated_clauses(sent_norm)
            if not sent_stripped_neg:
                continue

            # Split sentence into clauses on ; : 'and' 'then' 'but' 'except'
            clauses = _split_into_clauses(sent_stripped_neg)
            for clause in clauses:
                if not clause or ROUTINE_HEADER_REGEX.match(clause):
                    continue

                # 1. Check omitted medication instructions at the clause level
                # Includes bare medication list lines with dose + frequency (e.g. 'Medication list: metformin 1000 mg twice daily.')
                has_bare_med_regimen = bool(
                    parse_doses(clause) and parse_frequency_classes(clause)[0]
                )
                has_sensitive_med_verb = (
                    any(p.search(clause) for p in SENSITIVE_MED_OMISSION_PATTERNS)
                    or bool(re.search(r"\b(?:continue|take)\s+[a-z]{3,}", clause))
                    or has_bare_med_regimen
                    or (
                        "except" in sent_stripped_neg
                        and (
                            any(pat.search(clause) for _, pat in ALL_KNOWN_DRUG_PATTERNS)
                            or "home medication" not in clause
                        )
                    )
                )
                if has_sensitive_med_verb:
                    drugs_in_clause = {
                        SYNTHETIC_DRUG_ALIASES[alias]
                        for alias, pat in ALL_KNOWN_DRUG_PATTERNS
                        if pat.search(clause)
                    }
                    if drugs_in_clause:
                        drug_identity_covered = drugs_in_clause.issubset(
                            extracted_med_canonicals
                        )
                    elif "home medication" in clause:
                        drug_identity_covered = bool(extracted_med_equivalents)
                    else:
                        drug_identity_covered = any(
                            _has_word_boundary_match(eq, clause)
                            for eq in extracted_med_equivalents
                        )
                    clause_covered = (
                        _clause_is_covered(clause, med_evidence)
                        and drug_identity_covered
                    )
                    if not clause_covered:
                        self._append_omission_flags(
                            payload,
                            f"Clinically sensitive medication clause omitted from extraction: '{clause}' (in '{sent}')",
                            sent,
                        )
                        break

                # 2. Check omitted diagnostic tests / labs / imaging
                if TEST_ORDER_REGEX.search(clause):
                    if not _clause_is_covered(clause, all_verified_evidence):
                        self._append_omission_flags(
                            payload,
                            f"Diagnostic test or lab instruction omitted from extraction: '{clause}'",
                            sent,
                        )
                        break

                # 3. Check omitted warning signs / return precautions
                if WARNING_PRECAUTION_REGEX.search(clause) or any(
                    p.search(clause) for p in CRITICAL_RED_FLAG_PATTERNS
                ):
                    if not _clause_is_covered(clause, ws_evidence):
                        self._append_omission_flags(
                            payload,
                            f"Warning sign or return precaution omitted from extraction: '{clause}'",
                            sent,
                        )
                        break

                # 4. Check omitted follow-ups / referrals
                has_fu = bool(
                    re.search(
                        r"\b(?:refer(?:red|ral)?\s+to|consult(?:ation)?\s+with|appointment\s+with|schedule\s+(?:a\s+)?(?:visit|appointment|follow-?up))\b",
                        clause,
                    )
                    or (
                        re.search(r"\bfollow[\s-]?up\b", clause)
                        and (
                            is_line_start
                            or re.search(
                                r"\b(?:cardiology|nephrology|neurology|pulmonology|oncology|endocrinology|gastroenterology|orthopedics|urology|surgery|rheumatology|psychiatry|dermatology|ophthalmology|ent|hematology|infectious\s+disease|dr\.|clinic|hospital|for\s+[a-z]+)\b",
                                clause,
                            )
                        )
                    )
                )
                if has_fu:
                    if not _clause_is_covered(clause, fu_evidence):
                        self._append_omission_flags(
                            payload,
                            f"Follow-up or referral instruction omitted from extraction: '{clause}'",
                            sent,
                        )
                        break

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
                message=message[:1000],
                requires_human_review=True,
                evidence_snippet=snippet[:1000],
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
                    evidence_snippet=snippet[:1000],
                )
            )

    def _validate_provider_matches(
        self, payload: DischargeExtractionPayload
    ) -> None:
        """Requirement 7: Allow only template facts in claim_text; scan name/facility appropriately."""
        for provider in payload.provider_matches:
            # 1. If claim_text is non-null, it must strictly consist of closed template facts
            # (specialty, facility, distance, name, 'listed in synthetic directory') and zero evaluative claims.
            if provider.claim_text is not None:
                claim_norm = canonicalize_text(provider.claim_text)
                allowed_provider_tokens = (
                    set(PROVIDER_TEMPLATE_VOCABULARY)
                    | set(RE_WORD_TOKEN.findall(canonicalize_text(provider.specialty)))
                    | set(RE_WORD_TOKEN.findall(canonicalize_text(provider.facility)))
                    | set(RE_WORD_TOKEN.findall(canonicalize_text(provider.name)))
                    | {str(provider.distance_miles)}
                )
                claim_tokens = RE_WORD_TOKEN.findall(claim_norm)
                has_unlisted_token = not claim_tokens or any(
                    tok not in allowed_provider_tokens for tok in claim_tokens
                )
                has_evaluative_pattern = any(
                    p.search(provider.claim_text)
                    for p in UNSUPPORTED_PROVIDER_CLAIM_PATTERNS
                )
                if has_unlisted_token or has_evaluative_pattern:
                    payload.global_safety_flags.append(
                        SafetyFlag(
                            code=HazardCode.UNSUPPORTED_PROVIDER_CLAIM,
                            severity=HazardSeverity.CRITICAL,
                            message=(
                                f"Synthetic provider listing '{provider.provider_id}' field 'claim_text' "
                                "contains open-ended or unsupported evaluative/availability claims outside "
                                "permitted directory template facts."
                            ),
                            item_id=provider.provider_id,
                            requires_human_review=True,
                            evidence_snippet=provider.claim_text[:1000],
                        )
                    )

            # 2. Check name and facility for injected evaluative/availability guarantees
            # (Note: proper nouns like 'Bestway Health Clinic' do not match word-bounded '\bbest\b')
            for field_name, field_val in (
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
                            evidence_snippet=field_val[:1000],
                        )
                    )

            # 3. Verify mandatory non-guarantee disclaimer is intact
            disclaimer_lower = canonicalize_text(provider.verification_disclaimer)
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
        """Set safety_status, review_status, and release_blocked on an item based on its accumulated flags."""
        if any(
            f.code == HazardCode.CONFLICTING_INSTRUCTIONS for f in item.safety_flags
        ):
            item.safety_status = SafetyStatus.CONFLICT_DETECTED
            item.review_status = ReviewDecisionStatus.PENDING_HUMAN_REVIEW
            item.release_blocked = True
        elif any(f.requires_human_review for f in item.safety_flags):
            item.safety_status = SafetyStatus.NEEDS_REVIEW
            item.review_status = ReviewDecisionStatus.PENDING_HUMAN_REVIEW
            item.release_blocked = True
        else:
            item.safety_status = SafetyStatus.SAFE_ORGANIZED
            item.review_status = ReviewDecisionStatus.AUTO_APPROVED
            item.release_blocked = False


def validate_discharge_payload(
    payload: DischargeExtractionPayload,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> ValidationReport:
    """Convenience entrypoint to run the deterministic SafetyValidator."""
    validator = SafetyValidator(confidence_threshold=confidence_threshold)
    return validator.validate(payload)
