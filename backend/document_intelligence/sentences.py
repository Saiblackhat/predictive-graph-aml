"""Deterministic sentence segmentation with exact character-offset provenance.

Splits detected sections into stable SegmentedSentence objects while protecting
clinical abbreviations (e.g., 'Dr.', 'q.d.', 'b.i.d.') and decimal dosages/thresholds
(e.g., '2.5 mg', '101.5 F'). Never creates synthetic evidence sentences.
"""

from __future__ import annotations

import re

from backend.document_intelligence.types import (
    DetectedSection,
    NormalizedDocumentText,
    SectionCategory,
    SegmentedSentence,
)

PROTECTED_ABBREVIATIONS = {
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "st.",
    "jr.",
    "sr.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "approx.",
    "dept.",
    "appt.",
    "meds.",
    "tab.",
    "tabs.",
    "cap.",
    "caps.",
    "mg.",
    "mcg.",
    "ml.",
    "q.d.",
    "b.i.d.",
    "t.i.d.",
    "q.i.d.",
    "p.r.n.",
    "p.o.",
    "i.v.",
    "i.m.",
    "s.c.",
}

LEADING_LIST_MARKER_RE = re.compile(r"^(?:[-*•▪‣]+|\d+[.)]|\([a-zA-Z0-9]+\))\s+")


def _is_sentence_boundary(text: str, punct_idx: int) -> bool:
    """Return True if punctuation at text[punct_idx] is a true sentence boundary."""
    ch = text[punct_idx]
    if ch in ("!", "?"):
        return True
    if ch != ".":
        return False

    n = len(text)
    # 1. Decimal number check: e.g., "2.5", "0.125", "101.5"
    if punct_idx > 0 and punct_idx + 1 < n:
        if text[punct_idx - 1].isdigit() and text[punct_idx + 1].isdigit():
            return False

    # 2. Must be followed by end-of-line or whitespace + uppercase/digit/bullet
    after_idx = punct_idx + 1
    while after_idx < n and text[after_idx] in ('"', "'", ")", "]"):
        after_idx += 1
    if after_idx >= n:
        return True
    if text[after_idx] != " ":
        return False

    next_non_space = after_idx
    while next_non_space < n and text[next_non_space] == " ":
        next_non_space += 1
    if next_non_space >= n:
        return True

    # 3. Check preceding token for protected abbreviations (e.g., "Dr.", "b.i.d.")
    token_start = punct_idx
    while token_start > 0 and text[token_start - 1] not in (" ", "\n", "(", "["):
        token_start -= 1
    candidate_abbrev = text[token_start : punct_idx + 1].lower()
    if candidate_abbrev in PROTECTED_ABBREVIATIONS:
        return False

    # Single initial (e.g., "Dr. A. Smith")
    if len(candidate_abbrev) == 2 and candidate_abbrev[0].isalpha() and candidate_abbrev[1] == ".":
        return False

    return True


def _split_line_into_norm_spans(line_text: str, line_norm_start: int) -> list[tuple[int, int]]:
    """Split a single normalized line into (norm_start, norm_end) sentence spans."""
    spans: list[tuple[int, int]] = []
    n = len(line_text)
    if n == 0:
        return spans

    # Strip leading bullet / numbering marker (e.g., "1. ", "- ", "• ")
    cur_start = 0
    m_lead = LEADING_LIST_MARKER_RE.match(line_text)
    if m_lead:
        cur_start = m_lead.end()

    i = cur_start
    while i < n:
        if line_text[i] in (".", "!", "?") and _is_sentence_boundary(line_text, i):
            end_i = i + 1
            while end_i < n and line_text[end_i] in ('"', "'", ")", "]"):
                end_i += 1
            # Trim leading/trailing spaces inside [cur_start, end_i)
            s = cur_start
            e = end_i
            while s < e and line_text[s] == " ":
                s += 1
            while e > s and line_text[e - 1] == " ":
                e -= 1
            if s < e:
                spans.append((line_norm_start + s, line_norm_start + e))
            cur_start = end_i
            while cur_start < n and line_text[cur_start] == " ":
                cur_start += 1
            # Also strip any nested list marker after a sentence boundary on the same line
            m_next = LEADING_LIST_MARKER_RE.match(line_text[cur_start:])
            if m_next:
                cur_start += m_next.end()
            i = cur_start
            continue
        i += 1

    # Trailing remainder without terminal punctuation
    s = cur_start
    e = n
    while s < e and line_text[s] == " ":
        s += 1
    while e > s and line_text[e - 1] == " ":
        e -= 1
    if s < e:
        spans.append((line_norm_start + s, line_norm_start + e))

    return spans


def segment_sentences(
    norm_doc: NormalizedDocumentText,
    sections: list[DetectedSection],
) -> list[SegmentedSentence]:
    """Segment each detected section into SegmentedSentence objects with verbatim raw_text spans."""
    text = norm_doc.normalized_text
    sentences: list[SegmentedSentence] = []

    for sec in sections:
        if sec.canonical_category == SectionCategory.HEADER_METADATA:
            continue
        c_start = sec.content_norm_start_char
        c_end = sec.content_norm_end_char
        if c_start >= c_end:
            continue

        sec_slice = text[c_start:c_end]
        rel_pos = 0
        for line in sec_slice.split("\n"):
            line_norm_start = c_start + rel_pos
            rel_pos += len(line) + 1  # +1 for '\n'
            if not line.strip():
                continue

            for norm_s, norm_e in _split_line_into_norm_spans(line, line_norm_start):
                norm_sent_str = text[norm_s:norm_e].strip()
                if not norm_sent_str:
                    continue
                raw_s, raw_e, raw_sub = norm_doc.map_norm_span_to_raw(norm_s, norm_e)
                if not raw_sub:
                    continue
                sent_id = f"sent-{len(sentences) + 1:03d}"
                sentences.append(
                    SegmentedSentence(
                        sentence_id=sent_id,
                        section_id=sec.section_id,
                        section=sec.section_name,
                        canonical_section=sec.canonical_category,
                        source_text=raw_sub,
                        normalized_text=norm_sent_str,
                        start_offset=raw_s,
                        end_offset=raw_e,
                        norm_start_offset=norm_s,
                        norm_end_offset=norm_e,
                    )
                )

    return sentences
