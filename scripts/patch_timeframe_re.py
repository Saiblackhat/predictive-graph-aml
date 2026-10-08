#!/usr/bin/env python3
"""Patch TIMEFRAME_RE in extractor.py to capture optional at HH:MM [AM/PM] time suffix."""
import sys

path = "backend/document_intelligence/extractor.py"
content = open(path, encoding="utf-8").read()

# The exact 5-line block we want to replace (read from the repr above)
OLD = (
    "TIMEFRAME_RE = re.compile(\n"
    "    r\"\\b((?:within|in|after)\\s+(?:about\\s+|approximately\\s+)?\\d+(?:\\s*(?:to|-)\\s*\\d+)?\\s*(?:days?|weeks?|months?|hours?)|\"\n"
    "    r\"on\\s+(?:\\d{1,2}\\s+[A-Za-z]{3,9}\\s+\\d{4}|[A-Za-z]{3,9}\\s+\\d{1,2}(?:,\\s*\\d{4})?|\\d{4}-\\d{2}-\\d{2}))\\b\",\n"
    "    re.IGNORECASE,\n"
    ")"
)

# New regex: absolute-date branch now also optionally captures trailing "at HH:MM [AM/PM]"
# This ensures the extracted timeframe string always contains "at" when a time is present,
# matching the same qualifier ("IN") that _extract_timeframe_qualifier sees in the evidence sentence.
NEW = (
    "# Phase 4.1 fix: capture optional time-of-day suffix (\"at HH:MM [AM/PM]\") on\n"
    "# absolute-date patterns so the extracted timeframe contains 'at', matching\n"
    "# the 'IN' qualifier that _extract_timeframe_qualifier also sees in the source\n"
    "# evidence sentence (which contains \"at 10:30 AM\").  This eliminates the\n"
    "# spurious ON-vs-IN qualifier mismatch without modifying the frozen validator.\n"
    "TIMEFRAME_RE = re.compile(\n"
    "    r\"\\b((?:within|in|after)\\s+(?:about\\s+|approximately\\s+)?\\d+(?:\\s*(?:to|-)\\s*\\d+)?\\s*(?:days?|weeks?|months?|hours?)|\"\n"
    "    r\"on\\s+(?:\\d{1,2}\\s+[A-Za-z]{3,9}\\s+\\d{4}|[A-Za-z]{3,9}\\s+\\d{1,2}(?:,\\s*\\d{4})?|\\d{4}-\\d{2}-\\d{2})\"\n"
    "    r\"(?:\\s+at\\s+\\d{1,2}(?::\\d{2})?\\s*(?:[AP]M)?)?)\",\n"
    "    re.IGNORECASE,\n"
    ")"
)

if OLD not in content:
    print("ERROR: OLD block not found in file", file=sys.stderr)
    sys.exit(1)

content2 = content.replace(OLD, NEW, 1)
open(path, "w", encoding="utf-8").write(content2)
print("Patch applied successfully.")
