"""Timeframe parsing and semantic preservation for Phase 5.

Distinguishes:
- Absolute calendar dates (e.g. 'March 18, 2027') without fabricating reference dates.
- Relative windows with strict operator preservation ('within 7 days' != 'after 7 days').
- Recurring schedules ('twice daily', 'every morning').
- Immediate actions ('now', 'immediately', 'today').
- Missing / unstated timeframes (never invented).
"""

from __future__ import annotations

import re
from typing import Optional

from backend.task_orchestration.types import ParsedTimeframe, TimeframeType

MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Regex for absolute dates like "March 18, 2027", "18 March 2027", "2027-03-18", "03/18/2027"
ABSOLUTE_DATE_MONTH_FIRST_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?,\s*(\d{4})\b",
    re.IGNORECASE,
)

ABSOLUTE_DATE_DAY_FIRST_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec),?\s*(\d{4})\b",
    re.IGNORECASE,
)

ABSOLUTE_DATE_ISO_RE = re.compile(
    r"\b(\d{4})-(\d{2})-(\d{2})\b"
)

ABSOLUTE_DATE_SLASH_RE = re.compile(
    r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"
)

RELATIVE_OPERATOR_RE = re.compile(
    r"\b(within|after|before|by|in)\s+(\d+(?:-\d+)?)\s*(days?|weeks?|months?|hours?)\b",
    re.IGNORECASE,
)

IMMEDIATE_CUES_RE = re.compile(
    r"\b(immediately|promptly|now|today|at\s+discharge|stat)\b",
    re.IGNORECASE,
)

RECURRING_CUES_RE = re.compile(
    r"\b(once\s+daily|twice\s+daily|three\s+times\s+daily|four\s+times\s+daily|daily|"
    r"every\s+(?:morning|evening|night|bedtime|\d+\s+hours?)|bid|tid|qid|qhs|qd|prn)\b",
    re.IGNORECASE,
)

VAGUE_TIMEFRAME_RE = re.compile(
    r"\b(when\s+(?:convenient|ready|possible)|as\s+needed\s+in\s+future|in\s+(?:the\s+)?future)\b",
    re.IGNORECASE,
)


def parse_clinical_timeframe(raw_timeframe: Optional[str]) -> ParsedTimeframe:
    """Deterministically parse clinical timeframe string preserving exact semantics.

    Never manufactures calendar dates from relative windows.
    Never collapses 'within 7 days' into 'after 7 days'.
    """
    if raw_timeframe is None or not raw_timeframe.strip():
        return ParsedTimeframe(
            raw_text=None,
            timeframe_type=TimeframeType.UNSPECIFIED,
            is_missing=True,
        )

    text = raw_timeframe.strip()
    lower = text.lower()

    if VAGUE_TIMEFRAME_RE.search(lower):
        return ParsedTimeframe(
            raw_text=text,
            timeframe_type=TimeframeType.UNSPECIFIED,
            is_missing=True,
        )

    # 1. Absolute Date Parsing
    iso_date: Optional[str] = None

    m_iso = ABSOLUTE_DATE_ISO_RE.search(text)
    if m_iso:
        y, m, d = m_iso.groups()
        iso_date = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

    if not iso_date:
        m_month_first = ABSOLUTE_DATE_MONTH_FIRST_RE.search(text)
        if m_month_first:
            month_str, day_str, year_str = m_month_first.groups()
            m_num = MONTH_MAP.get(month_str.lower(), 1)
            iso_date = f"{int(year_str):04d}-{m_num:02d}-{int(day_str):02d}"

    if not iso_date:
        m_day_first = ABSOLUTE_DATE_DAY_FIRST_RE.search(text)
        if m_day_first:
            day_str, month_str, year_str = m_day_first.groups()
            m_num = MONTH_MAP.get(month_str.lower(), 1)
            iso_date = f"{int(year_str):04d}-{m_num:02d}-{int(day_str):02d}"

    if not iso_date:
        m_slash = ABSOLUTE_DATE_SLASH_RE.search(text)
        if m_slash:
            m_str, d_str, y_str = m_slash.groups()
            iso_date = f"{int(y_str):04d}-{int(m_str):02d}-{int(d_str):02d}"

    if iso_date is not None:
        return ParsedTimeframe(
            raw_text=text,
            timeframe_type=TimeframeType.ABSOLUTE_DATE,
            absolute_date=iso_date,
            is_missing=False,
        )

    # 2. Relative Window Parsing with Strict Operator Preservation
    m_rel = RELATIVE_OPERATOR_RE.search(text)
    if m_rel:
        operator = m_rel.group(1).lower()
        num_str = m_rel.group(2)
        unit = m_rel.group(3).lower()

        # Handle ranges like "1-2" by taking the upper bound for safety window
        if "-" in num_str:
            num = int(num_str.split("-")[1])
        else:
            num = int(num_str)

        duration_days = num
        if "week" in unit:
            duration_days = num * 7
        elif "month" in unit:
            duration_days = num * 30
        elif "hour" in unit:
            duration_days = max(1, num // 24)

        return ParsedTimeframe(
            raw_text=text,
            timeframe_type=TimeframeType.RELATIVE_WINDOW,
            relative_window=text,
            relative_operator=operator,
            relative_duration_days=duration_days,
            is_missing=False,
        )

    # 3. Recurring Schedule
    if RECURRING_CUES_RE.search(lower):
        return ParsedTimeframe(
            raw_text=text,
            timeframe_type=TimeframeType.RECURRING,
            recurring_pattern=text,
            is_missing=False,
        )

    # 4. Immediate Cue
    if IMMEDIATE_CUES_RE.search(lower):
        return ParsedTimeframe(
            raw_text=text,
            timeframe_type=TimeframeType.IMMEDIATE,
            is_missing=False,
        )

    # 5. Fallback relative text without explicit regex unit (e.g. "next week")
    if "next week" in lower:
        return ParsedTimeframe(
            raw_text=text,
            timeframe_type=TimeframeType.RELATIVE_WINDOW,
            relative_window=text,
            relative_operator="in",
            relative_duration_days=7,
            is_missing=False,
        )

    return ParsedTimeframe(
        raw_text=text,
        timeframe_type=TimeframeType.UNSPECIFIED,
        is_missing=False,
    )
