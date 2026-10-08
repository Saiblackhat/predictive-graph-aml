"""Deterministic section detection for hospital discharge summaries.

Detects canonical discharge sections and preserves unknown or unheaded sections
without discarding any document text.
"""

from __future__ import annotations

import re
from typing import Optional

from backend.document_intelligence.types import (
    DetectedSection,
    NormalizedDocumentText,
    SectionCategory,
)

# Ordered deterministic mapping from header patterns to SectionCategory
SECTION_HEADER_PATTERNS: list[tuple[re.Pattern[str], SectionCategory]] = [
    (
        re.compile(
            r"^(?:discharge\s+medications?|medications?\s+at\s+discharge|home\s+medications?\s+at\s+discharge|discharge\s+medication\s+list)$",
            re.IGNORECASE,
        ),
        SectionCategory.DISCHARGE_MEDICATIONS,
    ),
    (
        re.compile(
            r"^(?:medications?|medication\s+list|medication\s+instructions?|current\s+medications?|home\s+medications?)$",
            re.IGNORECASE,
        ),
        SectionCategory.MEDICATIONS,
    ),
    (
        re.compile(
            r"^(?:follow[\s-]?ups?\s+appointments?|appointments?|scheduled\s+appointments?|outpatient\s+appointments?|clinic\s+appointments?)$",
            re.IGNORECASE,
        ),
        SectionCategory.APPOINTMENTS,
    ),
    (
        re.compile(
            r"^(?:follow[\s-]?ups?|follow[\s-]?up\s+plan|follow[\s-]?up\s+instructions?|follow[\s-]?up\s+care)$",
            re.IGNORECASE,
        ),
        SectionCategory.FOLLOW_UP,
    ),
    (
        re.compile(
            r"^(?:tests?|pending\s+tests?|outpatient\s+tests?|diagnostic\s+tests?|laboratory\s+tests?|outpatient\s+labs?|labs?(?:\s+and\s+tests?|\s*&\s*imaging)?|imaging(?:\s+and\s+tests?)?)$",
            re.IGNORECASE,
        ),
        SectionCategory.TESTS,
    ),
    (
        re.compile(
            r"^(?:referrals?|outpatient\s+referrals?|specialty\s+referrals?|consults?|consultations?)$",
            re.IGNORECASE,
        ),
        SectionCategory.REFERRALS,
    ),
    (
        re.compile(
            r"^(?:wound\s+care|incision\s+care|dressing\s+care|surgical\s+site\s+care|wound\s+instructions?)$",
            re.IGNORECASE,
        ),
        SectionCategory.WOUND_CARE,
    ),
    (
        re.compile(
            r"^(?:activity|activity\s+restrictions?|physical\s+activity|exercise|rehabilitation|physical\s+therapy\s+instructions?)$",
            re.IGNORECASE,
        ),
        SectionCategory.ACTIVITY,
    ),
    (
        re.compile(
            r"^(?:diet|dietary\s+instructions?|nutrition|fluid\s+restrictions?|diet\s+and\s+nutrition)$",
            re.IGNORECASE,
        ),
        SectionCategory.DIET,
    ),
    (
        re.compile(
            r"^(?:when\s+to\s+seek\s+(?:immediate\s+)?(?:medical\s+)?(?:help|attention|care)|when\s+to\s+call(?:\s+the\s+doctor|\s+your\s+doctor)?|when\s+to\s+go\s+to\s+the\s+(?:er|emergency\s+department))$",
            re.IGNORECASE,
        ),
        SectionCategory.WHEN_TO_SEEK_HELP,
    ),
    (
        re.compile(
            r"^(?:warning\s+signs?|return\s+precautions?|red\s+flags?|emergency\s+warning\s+signs?|symptoms?\s+to\s+watch\s+for)$",
            re.IGNORECASE,
        ),
        SectionCategory.WARNING_SIGNS,
    ),
    (
        re.compile(
            r"^(?:instructions?|discharge\s+instructions?|patient\s+instructions?|care\s+instructions?|home\s+care\s+instructions?|general\s+instructions?)$",
            re.IGNORECASE,
        ),
        SectionCategory.INSTRUCTIONS,
    ),
    (
        re.compile(
            r"^(?:hospital\s+course|brief\s+hospital\s+course|clinical\s+course|summary\s+of\s+hospital\s+stay)$",
            re.IGNORECASE,
        ),
        SectionCategory.HOSPITAL_COURSE,
    ),
    (
        re.compile(
            r"^(?:discharge\s+summary(?:\s*[-—:]\s*[a-z0-9_-]+)?|patient\s+information|admission\s*\/\s*discharge\s+metadata)$",
            re.IGNORECASE,
        ),
        SectionCategory.HEADER_METADATA,
    ),
]

# Non-header clinical verbs that should never be mistaken for a section header
CLINICAL_INSTRUCTION_START_RE = re.compile(
    r"^(?:continue|start|begin|stop|discontinue|hold|take|resume|increase|decrease|change|switch|"
    r"follow\s+up\s+with|schedule\s+a|call\s+|return\s+to|go\s+to|seek\s+|monitor\s+for|avoid\s+|"
    r"keep\s+|wash\s+|change\s+dressing|weigh\s+|obtain\s+|repeat\s+|check\s+)\b",
    re.IGNORECASE,
)


def classify_header_title(header_title: str) -> SectionCategory:
    """Classify a stripped section header title into a canonical SectionCategory."""
    cleaned = header_title.strip().rstrip(":").strip()
    cleaned = re.sub(r"^(?:section\s+\d+[\s:.-]*|[ivx]+\.\s*)", "", cleaned, flags=re.IGNORECASE).strip()
    for pattern, category in SECTION_HEADER_PATTERNS:
        if pattern.match(cleaned):
            return category
    return SectionCategory.UNKNOWN


def _parse_header_line(line_str: str) -> Optional[tuple[str, SectionCategory, Optional[str]]]:
    """Check if a normalized line is a section header or inline 'Header: content' line.

    Returns (section_name, canonical_category, inline_content_or_None) if matched, else None.
    """
    stripped = line_str.strip()
    if not stripped:
        return None

    # 1. Standalone document title banner: "DISCHARGE SUMMARY - CASE-..."
    if re.match(r"^discharge\s+summary\b", stripped, re.IGNORECASE):
        return (stripped.rstrip(":"), SectionCategory.HEADER_METADATA, None)

    # 2. Line ending with ':' (e.g., "Discharge Medications:", "Custom Social Work Notes:")
    if stripped.endswith(":") and not stripped[0].isdigit():
        candidate_title = stripped[:-1].strip()
        # Ensure it's a concise header title (<= 6 words, <= 65 chars) and not a sentence
        words = candidate_title.split()
        if (
            1 <= len(words) <= 6
            and len(candidate_title) <= 65
            and "." not in candidate_title
            and not CLINICAL_INSTRUCTION_START_RE.match(candidate_title)
        ):
            cat = classify_header_title(candidate_title)
            return (candidate_title, cat, None)

    # 3. Standalone ALL-CAPS known header line without colon (e.g. "DISCHARGE MEDICATIONS")
    if stripped.isupper() and 1 <= len(stripped.split()) <= 5 and len(stripped) <= 55:
        cat = classify_header_title(stripped)
        if cat != SectionCategory.UNKNOWN:
            return (stripped.title(), cat, None)

    # 4. Inline header with colon + content on the same line (e.g., "Discharge medication list: metformin 1000 mg twice daily.")
    if ":" in stripped and not stripped[0].isdigit():
        prefix, remainder = stripped.split(":", 1)
        prefix_clean = prefix.strip()
        rem_clean = remainder.strip()
        if (
            rem_clean
            and 1 <= len(prefix_clean.split()) <= 5
            and len(prefix_clean) <= 50
            and not CLINICAL_INSTRUCTION_START_RE.match(prefix_clean)
        ):
            cat = classify_header_title(prefix_clean)
            if cat != SectionCategory.UNKNOWN:
                return (prefix_clean, cat, rem_clean)

    return None


def detect_sections(norm_doc: NormalizedDocumentText) -> list[DetectedSection]:
    """Detect all sections in a NormalizedDocumentText, preserving unknown sections."""
    text = norm_doc.normalized_text
    lines: list[tuple[int, int, str]] = []
    pos = 0
    for raw_line in text.split("\n"):
        line_start = pos
        line_end = pos + len(raw_line)
        lines.append((line_start, line_end, raw_line))
        pos = line_end + 1  # +1 for '\n'

    sections: list[DetectedSection] = []

    current_cat: SectionCategory = SectionCategory.UNKNOWN
    current_raw_header: Optional[str] = None
    current_name: str = "General Discharge Text"
    current_norm_start: Optional[int] = None
    current_content_start: Optional[int] = None
    current_norm_end: int = 0

    def _flush_current_section(end_norm_pos: int) -> None:
        nonlocal current_norm_start, current_content_start
        if current_norm_start is None:
            return
        c_start = current_content_start if current_content_start is not None else current_norm_start
        actual_end = max(current_norm_start, end_norm_pos)
        # Trim trailing newlines/spaces inside section span
        while actual_end > current_norm_start and text[actual_end - 1] in (" ", "\n"):
            actual_end -= 1
        c_end = max(c_start, actual_end)
        raw_s, raw_e, _ = norm_doc.map_norm_span_to_raw(
            current_norm_start, max(current_norm_start + 1, actual_end)
        )
        sec_id = f"sec-{len(sections) + 1:02d}"
        sections.append(
            DetectedSection(
                section_id=sec_id,
                canonical_category=current_cat,
                raw_header=current_raw_header,
                section_name=current_name,
                raw_start_char=raw_s,
                raw_end_char=raw_e,
                norm_start_char=current_norm_start,
                norm_end_char=actual_end,
                content_norm_start_char=min(c_start, c_end),
                content_norm_end_char=c_end,
            )
        )
        current_norm_start = None
        current_content_start = None

    for line_start, line_end, line_str in lines:
        if not line_str.strip():
            continue

        parsed = _parse_header_line(line_str)
        if parsed is not None:
            sec_name, sec_cat, inline_content = parsed
            # Flush any open section before starting the new section
            if current_norm_start is not None:
                _flush_current_section(current_norm_end)

            raw_h_start, raw_h_end, raw_h_text = norm_doc.map_norm_span_to_raw(
                line_start, line_end if inline_content is None else line_start + line_str.index(":") + 1
            )
            current_cat = sec_cat
            current_raw_header = raw_h_text.strip()
            current_name = sec_name
            current_norm_start = line_start
            if inline_content is not None:
                colon_offset = line_str.index(":") + 1
                while colon_offset < len(line_str) and line_str[colon_offset] == " ":
                    colon_offset += 1
                current_content_start = line_start + colon_offset
            else:
                current_content_start = min( len(text), line_end + 1 )
            current_norm_end = line_end
        else:
            if current_norm_start is None:
                current_cat = SectionCategory.UNKNOWN
                current_raw_header = None
                current_name = "General Discharge Text"
                current_norm_start = line_start
                current_content_start = line_start
            current_norm_end = line_end

    if current_norm_start is not None:
        _flush_current_section(current_norm_end)

    return sections
