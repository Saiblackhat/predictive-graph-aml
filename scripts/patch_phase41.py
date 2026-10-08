#!/usr/bin/env python3
"""
Phase 4.1 patch:
1. Add appointment_time: Optional[str] to ExtractedFollowUpRecord (Phase 4 types — NOT frozen models)
2. Add a time-extraction regex to extractor.py
3. Populate appointment_time in _extract_followup_from_sentence
4. Store appointment_time in clinical_metadata inside orchestrator.py
"""
import sys
import re

# ─── STEP 1: Add appointment_time to ExtractedFollowUpRecord ─────────────────
types_path = "backend/document_intelligence/types.py"
content = open(types_path, encoding="utf-8").read()

old_rec = (
    "    appointment_scheduled: Optional[bool] = Field(default=None)\n"
    "    confidence: float = Field(..., ge=0.0, le=1.0)\n"
    "    is_ambiguous: bool = Field(default=False)\n"
    "    ambiguity_note: Optional[str] = Field(default=None, max_length=500)\n"
    "    evidence: EvidenceSpan"
)
new_rec = (
    "    appointment_scheduled: Optional[bool] = Field(default=None)\n"
    "    # Phase 4.1: time-of-day for absolute appointments, e.g. '10:30' or '14:00'.\n"
    "    # Stored separately from timeframe so downstream layers can consume it.\n"
    "    # None when no time is stated in the source text.\n"
    "    appointment_time: Optional[str] = Field(\n"
    "        default=None,\n"
    "        max_length=20,\n"
    "        description=\"Time-of-day for absolute appointment (24-h HH:MM), or None.\",\n"
    "    )\n"
    "    confidence: float = Field(..., ge=0.0, le=1.0)\n"
    "    is_ambiguous: bool = Field(default=False)\n"
    "    ambiguity_note: Optional[str] = Field(default=None, max_length=500)\n"
    "    evidence: EvidenceSpan"
)
if old_rec not in content:
    print("ERROR: ExtractedFollowUpRecord target block not found in types.py", file=sys.stderr)
    sys.exit(1)
content = content.replace(old_rec, new_rec, 1)
open(types_path, "w", encoding="utf-8").write(content)
print("[1] appointment_time field added to ExtractedFollowUpRecord in types.py")


# ─── STEP 2: Add APPT_TIME_RE to extractor.py and populate appointment_time ──
extractor_path = "backend/document_intelligence/extractor.py"
content = open(extractor_path, encoding="utf-8").read()

# 2a — Add the regex constant after TIMEFRAME_RE block
old_after_tf = "FOLLOWUP_REASON_RE = re.compile("
new_before_fr = (
    "# Phase 4.1: extracts time-of-day component from absolute appointment sentences.\n"
    "# Supports: '10:30 AM', '10 AM', '14:30', '2 PM'. Returns normalised 24-h string.\n"
    "APPT_TIME_RE = re.compile(\n"
    "    r\"\\bat\\s+(\\d{1,2})(?::(\\d{2}))?\\s*([AP]M)?\",\n"
    "    re.IGNORECASE,\n"
    ")\n"
    "\n"
    "FOLLOWUP_REASON_RE = re.compile("
)
if old_after_tf not in content:
    print("ERROR: FOLLOWUP_REASON_RE anchor not found in extractor.py", file=sys.stderr)
    sys.exit(1)
content = content.replace(old_after_tf, new_before_fr, 1)

# 2b — Populate appointment_time in _extract_followup_from_sentence
# Target: the block that builds ExtractedFollowUpRecord
old_return = (
    "    return ExtractedFollowUpRecord(\n"
    "        id=f\"fu-{item_index}\",\n"
    "        specialty=specialty,\n"
    "        provider_name=provider_name,\n"
    "        timeframe=timeframe,\n"
    "        reason=reason,\n"
    "        appointment_scheduled=appointment_scheduled,\n"
    "        confidence=confidence,\n"
    "        is_ambiguous=is_ambiguous,\n"
    "        ambiguity_note=\"; \".join(ambiguity_notes) if ambiguity_notes else None,\n"
    "        evidence=_make_evidence(sent),\n"
    "    )"
)
new_return = (
    "    # Phase 4.1: extract time-of-day only when an absolute date is present.\n"
    "    appointment_time: Optional[str] = None\n"
    "    if timeframe and re.match(r\"^on\\s+\", timeframe, re.IGNORECASE):\n"
    "        m_time = APPT_TIME_RE.search(norm)\n"
    "        if m_time:\n"
    "            hour = int(m_time.group(1))\n"
    "            minute = int(m_time.group(2)) if m_time.group(2) else 0\n"
    "            meridiem = (m_time.group(3) or \"\").upper()\n"
    "            if meridiem == \"PM\" and hour != 12:\n"
    "                hour += 12\n"
    "            elif meridiem == \"AM\" and hour == 12:\n"
    "                hour = 0\n"
    "            appointment_time = f\"{hour:02d}:{minute:02d}\"\n"
    "\n"
    "    return ExtractedFollowUpRecord(\n"
    "        id=f\"fu-{item_index}\",\n"
    "        specialty=specialty,\n"
    "        provider_name=provider_name,\n"
    "        timeframe=timeframe,\n"
    "        reason=reason,\n"
    "        appointment_scheduled=appointment_scheduled,\n"
    "        appointment_time=appointment_time,\n"
    "        confidence=confidence,\n"
    "        is_ambiguous=is_ambiguous,\n"
    "        ambiguity_note=\"; \".join(ambiguity_notes) if ambiguity_notes else None,\n"
    "        evidence=_make_evidence(sent),\n"
    "    )"
)
if old_return not in content:
    print("ERROR: ExtractedFollowUpRecord return block not found in extractor.py", file=sys.stderr)
    sys.exit(1)
content = content.replace(old_return, new_return, 1)
open(extractor_path, "w", encoding="utf-8").write(content)
print("[2] appointment_time extraction added to extractor.py")


# ─── STEP 3: Store appointment_time in clinical_metadata in orchestrator.py ──
orch_path = "backend/task_orchestration/orchestrator.py"
content = open(orch_path, encoding="utf-8").read()

old_meta = (
    "        clinical_metadata={\n"
    "            \"specialty\": fu.specialty,\n"
    "            \"provider_name\": fu.provider_name,\n"
    "            \"appointment_scheduled\": fu.appointment_scheduled,\n"
    "            \"reason\": fu.reason,\n"
    "        },"
)
new_meta = (
    "        clinical_metadata={\n"
    "            \"specialty\": fu.specialty,\n"
    "            \"provider_name\": fu.provider_name,\n"
    "            \"appointment_scheduled\": fu.appointment_scheduled,\n"
    "            \"reason\": fu.reason,\n"
    "            # Phase 4.1: 24-h time string 'HH:MM' or None\n"
    "            \"appointment_time\": getattr(fu, \"appointment_time\", None),\n"
    "        },"
)
if old_meta not in content:
    print("ERROR: clinical_metadata block not found in orchestrator.py", file=sys.stderr)
    sys.exit(1)
content = content.replace(old_meta, new_meta, 1)
open(orch_path, "w", encoding="utf-8").write(content)
print("[3] appointment_time stored in clinical_metadata in orchestrator.py")

print("\nAll Phase 4.1 patches applied successfully.")
