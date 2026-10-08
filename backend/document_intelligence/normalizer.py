"""Safe, provenance-preserving text normalization for hospital discharge documents.

Normalizes whitespace, line breaks, Unicode compatibility characters, curly quotes,
NBSP, and line-break word hyphenation while maintaining an exact character-offset
map back to raw_text.
"""

from __future__ import annotations

import unicodedata

from backend.document_intelligence.types import (
    NormalizedDocumentText,
    SuspiciousCharFinding,
)

SUSPICIOUS_FORMAT_CODEPOINTS: dict[str, str] = {
    "\u200b": "ZERO WIDTH SPACE",
    "\u200c": "ZERO WIDTH NON-JOINER",
    "\u200d": "ZERO WIDTH JOINER",
    "\ufeff": "ZERO WIDTH NO-BREAK SPACE (BOM)",
    "\u2060": "WORD JOINER",
    "\u00ad": "SOFT HYPHEN",
    "\u180e": "MONGOLIAN VOWEL SEPARATOR",
    "\u202a": "LEFT-TO-RIGHT EMBEDDING",
    "\u202b": "RIGHT-TO-LEFT EMBEDDING",
    "\u202c": "POP DIRECTIONAL FORMATTING",
    "\u202d": "LEFT-TO-RIGHT OVERRIDE",
    "\u202e": "RIGHT-TO-LEFT OVERRIDE",
    "\u2066": "LEFT-TO-RIGHT ISOLATE",
    "\u2067": "RIGHT-TO-LEFT ISOLATE",
    "\u2068": "FIRST STRONG ISOLATE",
    "\u2069": "POP DIRECTIONAL ISOLATE",
}

CURLY_SINGLE_QUOTES = {"\u2018", "\u2019", "\u201a", "\u201b", "\u2032"}
CURLY_DOUBLE_QUOTES = {"\u201c", "\u201d", "\u201e", "\u201f", "\u2033"}
UNICODE_SPACES = {
    "\u00a0",
    "\u1680",
    "\u2000",
    "\u2001",
    "\u2002",
    "\u2003",
    "\u2004",
    "\u2005",
    "\u2006",
    "\u2007",
    "\u2008",
    "\u2009",
    "\u200a",
    "\u202f",
    "\u205f",
    "\u3000",
    "\t",
}
UNICODE_DASHES = {"\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212"}

# Stems that legitimately keep their hyphen when split across a line break
HYPHENATED_COMPOUND_PREFIXES = {
    "follow",
    "self",
    "post",
    "pre",
    "non",
    "anti",
    "co",
    "re",
    "sub",
    "inter",
    "intra",
    "multi",
    "over",
    "under",
    "cross",
    "short",
    "long",
    "high",
    "low",
    "in",
    "out",
    "up",
    "off",
    "on",
    "check",
    "x",
    "beta",
    "alpha",
    "sacubitril",
    "amoxicillin",
    "piperacillin",
    "trimethoprim",
    "sulfamethoxazole",
}


def detect_suspicious_characters(raw_text: str) -> list[SuspiciousCharFinding]:
    """Scan raw_text for zero-width, soft-hyphen, or bidirectional control characters."""
    findings: list[SuspiciousCharFinding] = []
    for idx, ch in enumerate(raw_text):
        if ch in SUSPICIOUS_FORMAT_CODEPOINTS or unicodedata.category(ch) == "Cf":
            name = SUSPICIOUS_FORMAT_CODEPOINTS.get(
                ch, unicodedata.name(ch, "FORMAT CONTROL CHARACTER")
            )
            ctx_start = max(0, idx - 15)
            ctx_end = min(len(raw_text), idx + 16)
            snippet = raw_text[ctx_start:ctx_end].replace("\n", "\\n").replace("\r", "\\r")
            findings.append(
                SuspiciousCharFinding(
                    codepoint=f"U+{ord(ch):04X}",
                    character_name=name,
                    raw_offset=idx,
                    context_snippet=snippet or ch,
                )
            )
    return findings


def normalize_document_text(raw_text: str) -> NormalizedDocumentText:
    """Normalize raw discharge text while preserving a 1-to-1 character offset mapping to raw_text."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Discharge document raw_text must be a non-empty string.")

    suspicious = detect_suspicious_characters(raw_text)

    # Stage 1: Character-level normalization with (char, raw_start, raw_end) tuples
    stage1_chars: list[str] = []
    stage1_starts: list[int] = []
    stage1_ends: list[int] = []

    n = len(raw_text)
    i = 0
    while i < n:
        ch = raw_text[i]
        # Handle CRLF -> \n
        if ch == "\r":
            if i + 1 < n and raw_text[i + 1] == "\n":
                stage1_chars.append("\n")
                stage1_starts.append(i)
                stage1_ends.append(i + 2)
                i += 2
                continue
            stage1_chars.append("\n")
            stage1_starts.append(i)
            stage1_ends.append(i + 1)
            i += 1
            continue

        # Strip zero-width / format control characters (recorded in suspicious)
        if ch in SUSPICIOUS_FORMAT_CODEPOINTS or unicodedata.category(ch) == "Cf":
            i += 1
            continue

        if ch in CURLY_SINGLE_QUOTES:
            stage1_chars.append("'")
            stage1_starts.append(i)
            stage1_ends.append(i + 1)
        elif ch in CURLY_DOUBLE_QUOTES:
            stage1_chars.append('"')
            stage1_starts.append(i)
            stage1_ends.append(i + 1)
        elif ch in UNICODE_SPACES:
            stage1_chars.append(" ")
            stage1_starts.append(i)
            stage1_ends.append(i + 1)
        elif ch in UNICODE_DASHES:
            stage1_chars.append("-")
            stage1_starts.append(i)
            stage1_ends.append(i + 1)
        else:
            # NFKC normalize individual character (preserves offset for ligatures etc.)
            nfkc_str = unicodedata.normalize("NFKC", ch)
            for sub_ch in nfkc_str:
                if sub_ch in UNICODE_SPACES:
                    sub_ch = " "
                stage1_chars.append(sub_ch)
                stage1_starts.append(i)
                stage1_ends.append(i + 1)
        i += 1

    # Stage 2: Repair line-break word hyphenation (e.g., "disconti-\nnue" -> "discontinue")
    stage2_chars: list[str] = []
    stage2_starts: list[int] = []
    stage2_ends: list[int] = []
    hyphenations_repaired = 0

    m = len(stage1_chars)
    j = 0
    while j < m:
        if stage1_chars[j] == "-":
            # Check preceding lowercase word stem
            stem_chars: list[str] = []
            k = len(stage2_chars) - 1
            while k >= 0 and stage2_chars[k].isalpha():
                stem_chars.append(stage2_chars[k])
                k -= 1
            stem = "".join(reversed(stem_chars))

            # Check if followed by optional spaces, a single newline, optional spaces, and lowercase continuation
            look = j + 1
            while look < m and stage1_chars[look] == " ":
                look += 1
            if look < m and stage1_chars[look] == "\n":
                look += 1
                while look < m and stage1_chars[look] == " ":
                    look += 1
                cont_start = look
                cont_chars: list[str] = []
                while look < m and stage1_chars[look].isalpha():
                    cont_chars.append(stage1_chars[look])
                    look += 1
                continuation = "".join(cont_chars)

                if (
                    len(stem) >= 2
                    and len(continuation) >= 2
                    and stem[-1].islower()
                    and continuation[0].islower()
                ):
                    if stem.lower() in HYPHENATED_COMPOUND_PREFIXES:
                        # Keep the hyphen, drop only the newline and surrounding spaces
                        stage2_chars.append("-")
                        stage2_starts.append(stage1_starts[j])
                        stage2_ends.append(stage1_ends[j])
                        j = cont_start
                        hyphenations_repaired += 1
                        continue
                    else:
                        # Repair broken single word by eliding "-\n   "
                        j = cont_start
                        hyphenations_repaired += 1
                        continue

        stage2_chars.append(stage1_chars[j])
        stage2_starts.append(stage1_starts[j])
        stage2_ends.append(stage1_ends[j])
        j += 1

    # Stage 3: Collapse horizontal space runs, strip leading/trailing spaces per line,
    # and collapse >2 consecutive newlines to at most 2 newlines.
    final_chars: list[str] = []
    final_starts: list[int] = []
    final_ends: list[int] = []

    p = len(stage2_chars)
    idx = 0
    at_line_start = True
    while idx < p:
        ch = stage2_chars[idx]
        if ch == " ":
            # Look ahead across contiguous horizontal spaces
            run_end = idx + 1
            while run_end < p and stage2_chars[run_end] == " ":
                run_end += 1
            # Skip spaces at start of line or immediately before newline / end of text
            if at_line_start or run_end >= p or stage2_chars[run_end] == "\n":
                idx = run_end
                continue
            final_chars.append(" ")
            final_starts.append(stage2_starts[idx])
            final_ends.append(stage2_ends[run_end - 1])
            idx = run_end
        elif ch == "\n":
            # Count consecutive newlines
            nl_end = idx + 1
            while nl_end < p and stage2_chars[nl_end] in (" ", "\n"):
                if stage2_chars[nl_end] == " ":
                    nl_end += 1
                else:
                    nl_end += 1
            # Count actual newlines in stage2_chars[idx:nl_end]
            nl_indices = [x for x in range(idx, nl_end) if stage2_chars[x] == "\n"]
            if not final_chars:
                # Skip leading newlines at start of document
                idx = nl_end
                at_line_start = True
                continue
            if nl_end >= p:
                # Skip trailing newlines at end of document
                break
            emit_count = min(2, len(nl_indices))
            for e_i in range(emit_count):
                raw_nl_idx = nl_indices[e_i]
                final_chars.append("\n")
                final_starts.append(stage2_starts[raw_nl_idx])
                final_ends.append(stage2_ends[raw_nl_idx])
            idx = nl_end
            at_line_start = True
        else:
            final_chars.append(ch)
            final_starts.append(stage2_starts[idx])
            final_ends.append(stage2_ends[idx])
            at_line_start = False
            idx += 1

    normalized_str = "".join(final_chars)
    if not normalized_str:
        raise ValueError("Discharge document normalized to empty string.")

    return NormalizedDocumentText(
        raw_text=raw_text,
        normalized_text=normalized_str,
        norm_to_raw_start=final_starts,
        norm_to_raw_end=final_ends,
        suspicious_chars=suspicious,
        hyphenations_repaired=hyphenations_repaired,
    )
