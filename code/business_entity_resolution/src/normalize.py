"""
normalize.py
Text normalization utilities for business names and addresses.
No external lookups -- pure string/regex transforms learned from the
noise patterns described in the problem statement.
"""
import re

# --- Legal-suffix / common-abbreviation expansion tables -------------------
# Keys and values are lowercase, whole-word tokens.
NAME_ABBREV = {
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
    "ltd": "limited",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "pvt": "private",
    "pvt.": "private",
    "intl": "international",
    "mfg": "manufacturing",
    "assoc": "associates",
    "bros": "brothers",
    "svcs": "services",
    "svc": "service",
    "grp": "group",
    "&": "and",
    "dept": "department",
    "natl": "national",
    "tech": "technology",
    "mkt": "market",
    "ent": "enterprises",
    "enterp": "enterprises",
}

ADDR_ABBREV = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "apt": "apartment",
    "fl": "floor",
    "flr": "floor",
    "bldg": "building",
    "nr": "near",
    "opp": "opposite",
    "no": "number",
    "hwy": "highway",
    "expy": "expressway",
    "sec": "sector",
    "ph": "phase",
    "ind": "industrial",
    "est": "estate",
    "twp": "township",
    "cplx": "complex",
    "stn": "station",
}

# Landmark / filler phrases that add noise but no matching signal.
ADDR_STOPPHRASES = [
    r"\bnear\b", r"\bopp(?:osite)?\b", r"\bbehind\b", r"\bnext to\b",
    r"\blandmark\b", r"\bbeside\b", r"\bclose to\b",
]

_punct_re = re.compile(r"[^\w\s]")
_ws_re = re.compile(r"\s+")
_postal_re = re.compile(r"\b(\d{5,6})\b")


def _expand_tokens(text: str, table: dict) -> str:
    tokens = text.split()
    out = []
    for t in tokens:
        out.append(table.get(t, t))
    return " ".join(out)


def basic_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = text.replace("&", " and ")
    text = _punct_re.sub(" ", text)
    text = _ws_re.sub(" ", text).strip()
    return text


def normalize_name(text: str) -> str:
    """Lowercase, strip punctuation, expand legal-form abbreviations."""
    text = basic_clean(text)
    text = _expand_tokens(text, NAME_ABBREV)
    return text


def core_name(text: str) -> str:
    """normalize_name() with trailing legal-suffix words removed, so
    'acme private limited' and 'acme pvt ltd' both collapse to 'acme'."""
    suffixes = {
        "corporation", "company", "incorporated", "limited",
        "limited liability company", "limited liability partnership",
        "private", "international", "group", "enterprises", "co",
        "ltd", "inc", "llc", "llp", "pvt",
    }
    norm = normalize_name(text)
    tokens = [t for t in norm.split() if t not in suffixes]
    return " ".join(tokens) if tokens else norm


def normalize_address(text: str) -> str:
    text = basic_clean(text)
    for pat in ADDR_STOPPHRASES:
        text = re.sub(pat, " ", text)
    text = _ws_re.sub(" ", text).strip()
    text = _expand_tokens(text, ADDR_ABBREV)
    return text


def extract_postal(text: str):
    if text is None:
        return None
    m = _postal_re.search(str(text))
    return m.group(1) if m else None


def tokenize(text: str) -> set:
    return set(t for t in text.split() if t)
