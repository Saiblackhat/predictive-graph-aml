"""Comprehensive unit and integration tests for Phase 4 Document Intelligence.

Covers:
1. Text normalization (normal text, line-break hyphenation, Unicode whitespace/quotes,
   zero-width/suspicious characters, bidirectional raw offset mapping)
2. Deterministic section detection (canonical sections + unknown section preservation)
3. Sentence segmentation (stable IDs, abbreviations, decimal dosages, verbatim spans)
4. Evidence-span provenance (no fabricated page numbers, exact raw_text[start:end] match)
5. Structured extraction across Medications, Follow-Ups, Tests, Referrals,
   Warning Signs, and Care Instructions (Wound Care, Diet, Activity, Rehabilitation, Other)
6. Missing-field non-invention (missing dosage/timeframe/specialty -> None)
7. Malformed-document & format boundary rejection
8. All 8 realistic synthetic discharge summary fixtures integrated with Phase 3 validator
"""

from __future__ import annotations

import unittest

from backend.document_intelligence import (
    ALL_PHASE4_SYNTHETIC_DOCUMENTS,
    DOC_SYNTH_001_NORMAL,
    DOC_SYNTH_002_MULTI,
    DOC_SYNTH_003_MISSING_AMBIGUOUS,
    DOC_SYNTH_004_CONFLICTING_MEDS,
    DOC_SYNTH_005_WARNING_SIGNS,
    DOC_SYNTH_006_TESTS_AND_REFERRALS,
    DOC_SYNTH_007_RICH_MULTISECTION,
    DOC_SYNTH_008_MESSY_UNICODE,
    DocumentFormat,
    MalformedDocumentError,
    SectionCategory,
    UnsupportedDocumentFormatError,
    detect_sections,
    detect_suspicious_characters,
    ingest_discharge_document,
    normalize_document_text,
    run_document_intelligence_pipeline,
    segment_sentences,
)
from backend.schemas.models import (
    HazardCode,
    MedicationAction,
    RawTextTooLargeError,
    SafetyStatus,
    UrgencyLevel,
)


class TestPhase4DocumentIntelligence(unittest.TestCase):
    """Test suite for Phase 4 Document Intelligence foundation."""

    # ------------------------------------------------------------------
    # 4.2 Text Normalization Tests
    # ------------------------------------------------------------------
    def test_normalization_normal_text_and_whitespace(self) -> None:
        raw = "  Discharge Medications:\r\n   Continue   aspirin   81 mg   once daily.  \r\n\r\n\r\n"
        norm = normalize_document_text(raw)
        self.assertEqual(
            norm.normalized_text,
            "Discharge Medications:\nContinue aspirin 81 mg once daily.",
        )
        self.assertEqual(len(norm.suspicious_chars), 0)
        self.assertEqual(norm.hyphenations_repaired, 0)
        # Check offset mapping back to raw_text
        idx_asp = norm.normalized_text.index("aspirin")
        raw_s, raw_e, sub = norm.map_norm_span_to_raw(idx_asp, idx_asp + len("aspirin"))
        self.assertEqual(sub, "aspirin")
        self.assertEqual(raw[raw_s:raw_e], "aspirin")

    def test_normalization_linebreak_hyphenation_vs_compound_words(self) -> None:
        raw = (
            "Disconti-\n"
            "   nue meto-\r\n"
            "prolol 25 mg daily.\n"
            "Follow-\n"
            "up with Cardiol-\n"
            "ogy in 7 days.\n"
            "Continue sacubitril-\n"
            "valsartan 49/51 mg twice daily."
        )
        norm = normalize_document_text(raw)
        self.assertIn("Discontinue metoprolol 25 mg daily.", norm.normalized_text)
        self.assertIn("Follow-up with Cardiology in 7 days.", norm.normalized_text)
        self.assertIn("Continue sacubitril-valsartan 49/51 mg twice daily.", norm.normalized_text)
        self.assertEqual(norm.hyphenations_repaired, 5)

    def test_normalization_unicode_whitespace_and_curly_quotes(self) -> None:
        raw = "Continue \u2018aspirin\u2019 81\u00a0mg\tonce\u202fdaily \u201cas before\u201d."
        norm = normalize_document_text(raw)
        self.assertEqual(
            norm.normalized_text,
            "Continue 'aspirin' 81 mg once daily \"as before\".",
        )

    def test_normalization_suspicious_zero_width_and_bidi_characters(self) -> None:
        raw = "Sto\u200bp lisin\u00adopril 10 mg\u202e daily."
        findings = detect_suspicious_characters(raw)
        self.assertEqual(len(findings), 3)
        codepoints = [f.codepoint for f in findings]
        self.assertEqual(codepoints, ["U+200B", "U+00AD", "U+202E"])

        norm = normalize_document_text(raw)
        self.assertEqual(norm.normalized_text, "Stop lisinopril 10 mg daily.")
        self.assertEqual(len(norm.suspicious_chars), 3)

    # ------------------------------------------------------------------
    # 4.3 Section Detection Tests
    # ------------------------------------------------------------------
    def test_section_detection_canonical_and_unknown_preserved(self) -> None:
        raw = (
            "DISCHARGE SUMMARY - TEST-SEC-01\n"
            "Discharge Medications:\n"
            "Continue aspirin 81 mg once daily.\n"
            "Follow-Up Appointments:\n"
            "Follow up with Cardiology within 7 days.\n"
            "Tests:\n"
            "Obtain basic metabolic panel within 5 days.\n"
            "Referrals:\n"
            "Referral to Physical Therapy within 7 days.\n"
            "Wound Care:\n"
            "Keep incision clean and dry.\n"
            "Activity:\n"
            "Avoid lifting more than 10 pounds.\n"
            "Diet:\n"
            "Follow a low-sodium diet.\n"
            "Warning Signs:\n"
            "Call 911 for chest pain.\n"
            "When to Seek Help:\n"
            "Call clinic if fever exceeds 101.5 F.\n"
            "Instructions:\n"
            "Weigh yourself every morning.\n"
            "Social Work and Transportation:\n"
            "Medical transport arranged for clinic visit.\n"
        )
        norm = normalize_document_text(raw)
        sections = detect_sections(norm)
        cats = [s.canonical_category for s in sections]
        self.assertEqual(
            cats,
            [
                SectionCategory.HEADER_METADATA,
                SectionCategory.DISCHARGE_MEDICATIONS,
                SectionCategory.APPOINTMENTS,
                SectionCategory.TESTS,
                SectionCategory.REFERRALS,
                SectionCategory.WOUND_CARE,
                SectionCategory.ACTIVITY,
                SectionCategory.DIET,
                SectionCategory.WARNING_SIGNS,
                SectionCategory.WHEN_TO_SEEK_HELP,
                SectionCategory.INSTRUCTIONS,
                SectionCategory.UNKNOWN,
            ],
        )
        # Verify unknown section is preserved with its original title
        unknown_sec = sections[-1]
        self.assertEqual(unknown_sec.section_name, "Social Work and Transportation")

    def test_section_detection_inline_header_and_unheaded_text(self) -> None:
        raw = (
            "Continue metformin 500 mg twice daily.\n\n"
            "Discharge medication list: metformin 1000 mg twice daily."
        )
        norm = normalize_document_text(raw)
        sections = detect_sections(norm)
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].canonical_category, SectionCategory.UNKNOWN)
        self.assertEqual(sections[1].canonical_category, SectionCategory.DISCHARGE_MEDICATIONS)

    # ------------------------------------------------------------------
    # 4.4 Sentence Segmentation & 4.6 Evidence-Span Tests
    # ------------------------------------------------------------------
    def test_sentence_segmentation_abbreviations_decimals_and_exact_spans(self) -> None:
        raw = (
            "Discharge Medications:\n"
            "1. Continue apixaban 2.5 mg by mouth twice daily.\n"
            "Follow-Up Appointments:\n"
            "- Follow up with Cardiology (Dr. Elena Rostova) within 7 days. "
            "Call Dr. A. Smith if temperature exceeds 101.5 F.\n"
        )
        norm = normalize_document_text(raw)
        sections = detect_sections(norm)
        sentences = segment_sentences(norm, sections)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0].sentence_id, "sent-001")
        self.assertEqual(
            sentences[0].source_text,
            "Continue apixaban 2.5 mg by mouth twice daily.",
        )
        self.assertEqual(
             sentences[1].source_text,
            "Follow up with Cardiology (Dr. Elena Rostova) within 7 days.",
        )
        self.assertEqual(
            sentences[2].source_text,
            "Call Dr. A. Smith if temperature exceeds 101.5 F.",
        )
        # Mathematical invariant: raw[start_offset:end_offset] == source_text
        for s in sentences:
            self.assertEqual(raw[s.start_offset : s.end_offset], s.source_text)

    def test_evidence_span_never_fabricates_page_numbers(self) -> None:
        res = run_document_intelligence_pipeline(DOC_SYNTH_001_NORMAL.raw_text)
        for med in res.medications:
            src_ev = med.evidence.to_source_evidence()
            self.assertIsNone(src_ev.source_page)
            self.assertFalse(src_ev.source_page_verified)
            self.assertIsNotNone(src_ev.start_char)
            self.assertIsNotNone(src_ev.end_char)
            self.assertEqual(
                DOC_SYNTH_001_NORMAL.raw_text[src_ev.start_char : src_ev.end_char],
                src_ev.source_text,
            )

    # ------------------------------------------------------------------
    # 4.5 & 4.6 Extraction & Missing-Field Non-Invention Tests
    # ------------------------------------------------------------------
    def test_missing_fields_are_never_invented(self) -> None:
        res = run_document_intelligence_pipeline(DOC_SYNTH_003_MISSING_AMBIGUOUS.raw_text)
        self.assertEqual(len(res.medications), 1)
        med = res.medications[0]
        self.assertEqual(med.medication_name, "Levothyroxine")
        self.assertIsNone(med.dosage)
        self.assertIsNone(med.route)
        self.assertTrue(med.is_ambiguous)

        self.assertEqual(len(res.follow_ups), 2)
        fu_missing_tf = res.follow_ups[0]
        self.assertEqual(fu_missing_tf.specialty, "Cardiology")
        self.assertIsNone(fu_missing_tf.timeframe)
        self.assertTrue(fu_missing_tf.is_ambiguous)

        fu_missing_spec = res.follow_ups[1]
        self.assertIsNone(fu_missing_spec.specialty)
        self.assertEqual(fu_missing_spec.timeframe, "within 14 days")
        self.assertTrue(fu_missing_spec.is_ambiguous)

        # Must be blocked by downstream validator
        self.assertFalse(res.validation_report.patient_releasable)
        self.assertEqual(
            res.validation_report.overall_safety_status,
            SafetyStatus.NEEDS_REVIEW,
        )
        self.assertIn(
            HazardCode.UNCLEAR_DOSAGE,
            res.validation_report.hazard_codes_detected,
        )
        self.assertIn(
            HazardCode.MISSING_FOLLOWUP_TIMEFRAME,
            res.validation_report.hazard_codes_detected,
        )
        self.assertIn(
            HazardCode.MISSING_SPECIALTY,
            res.validation_report.hazard_codes_detected,
        )

    def test_governed_negation_and_exception_extraction(self) -> None:
        # 1. Governed negation without exception -> does not invent medication
        raw_safe = (
            "Discharge Medications:\n"
            "No new medications were started.\n"
            "Follow-Up Appointments:\n"
            "Follow up with Primary Care within 7 days."
        )
        res_safe = run_document_intelligence_pipeline(raw_safe, document_id="DOC-NEG-1")
        self.assertEqual(len(res_safe.medications), 0)
        self.assertEqual(len(res_safe.follow_ups), 1)
        self.assertTrue(res_safe.validation_report.patient_releasable)

        # 2. Governed negation WITH exception -> extracts warfarin and escalates for review
        raw_exc = (
            "Discharge Medications:\n"
            "No home medications were stopped except warfarin.\n"
            "Follow-Up Appointments:\n"
            "Follow up with Endocrinology within 2 weeks."
        )
        res_exc = run_document_intelligence_pipeline(raw_exc, document_id="DOC-NEG-2")
        self.assertEqual(len(res_exc.medications), 1)
        self.assertEqual(res_exc.medications[0].medication_name, "warfarin")
        self.assertEqual(res_exc.medications[0].action, MedicationAction.STOP_DISCONTINUE)
        self.assertTrue(res_exc.medications[0].is_ambiguous)
        self.assertFalse(res_exc.validation_report.patient_releasable)

    # ------------------------------------------------------------------
    # 4.8 Synthetic Test Documents (All 8 Scenarios)
    # ------------------------------------------------------------------
    def test_all_8_synthetic_discharge_documents_end_to_end(self) -> None:
        self.assertEqual(len(ALL_PHASE4_SYNTHETIC_DOCUMENTS), 8)

        # 1. Normal uncomplicated discharge -> SAFE / ORGANIZED
        r1 = run_document_intelligence_pipeline(DOC_SYNTH_001_NORMAL.raw_text)
        self.assertEqual(r1.document.document_id, "DOC-SYNTH-001")
        self.assertEqual(r1.document.patient_identifier, "SYNTH-PT-401")
        self.assertEqual(len(r1.medications), 2)
        self.assertEqual(len(r1.follow_ups), 1)
        self.assertEqual(len(r1.warning_signs), 1)
        self.assertEqual(
            r1.validation_report.overall_safety_status,
            SafetyStatus.SAFE_ORGANIZED,
        )
        self.assertTrue(r1.validation_report.patient_releasable)

        # 2. Multiple medications + follow-ups -> SAFE / ORGANIZED
        r2 = run_document_intelligence_pipeline(DOC_SYNTH_002_MULTI.raw_text)
        self.assertEqual(len(r2.medications), 3)
        self.assertEqual(len(r2.follow_ups), 2)
        self.assertEqual(
            r2.validation_report.overall_safety_status,
            SafetyStatus.SAFE_ORGANIZED,
        )
        self.assertTrue(r2.validation_report.patient_releasable)

        # 3. Missing/ambiguous follow-up -> NEEDS REVIEW (Blocked)
        r3 = run_document_intelligence_pipeline(DOC_SYNTH_003_MISSING_AMBIGUOUS.raw_text)
        self.assertEqual(
            r3.validation_report.overall_safety_status,
            SafetyStatus.NEEDS_REVIEW,
        )
        self.assertFalse(r3.validation_report.patient_releasable)

        # 4. Conflicting medication instructions -> CONFLICT DETECTED (Blocked)
        r4 = run_document_intelligence_pipeline(DOC_SYNTH_004_CONFLICTING_MEDS.raw_text)
        self.assertEqual(len(r4.medications), 2)
        self.assertEqual(
            r4.validation_report.overall_safety_status,
            SafetyStatus.CONFLICT_DETECTED,
        )
        self.assertIn(
            HazardCode.CONFLICTING_INSTRUCTIONS,
            r4.validation_report.hazard_codes_detected,
        )
        self.assertFalse(r4.validation_report.patient_releasable)

        # 5. Warning signs (emergent 911 + numeric threshold) -> NEEDS REVIEW (WARNING_SIGN_ESCALATION)
        r5 = run_document_intelligence_pipeline(DOC_SYNTH_005_WARNING_SIGNS.raw_text)
        self.assertEqual(len(r5.warning_signs), 2)
        self.assertEqual(r5.warning_signs[0].urgency, UrgencyLevel.EMERGENT_911)
        self.assertEqual(r5.warning_signs[1].urgency, UrgencyLevel.SOON)
        self.assertIn(
            HazardCode.WARNING_SIGN_ESCALATION,
            r5.validation_report.hazard_codes_detected,
        )
        self.assertFalse(r5.validation_report.patient_releasable)

        # 6. Outpatient tests + specialty referrals -> SAFE / ORGANIZED
        r6 = run_document_intelligence_pipeline(DOC_SYNTH_006_TESTS_AND_REFERRALS.raw_text)
        self.assertEqual(len(r6.medications), 1)
        self.assertEqual(len(r6.follow_ups), 1)
        self.assertEqual(len(r6.tests), 2)
        self.assertEqual(len(r6.referrals), 1)
        self.assertEqual(r6.referrals[0].specialty_or_service, "Physical Therapy")
        self.assertEqual(
            r6.validation_report.overall_safety_status,
            SafetyStatus.SAFE_ORGANIZED,
        )
        self.assertTrue(r6.validation_report.patient_releasable)

        # 7. Rich multi-section discharge -> SAFE / ORGANIZED with Wound Care, Diet, Activity, Rehab, Equipment
        r7 = run_document_intelligence_pipeline(DOC_SYNTH_007_RICH_MULTISECTION.raw_text)
        self.assertEqual(len(r7.medications), 2)
        self.assertEqual(len(r7.follow_ups), 1)
        self.assertEqual(len(r7.tests), 1)
        self.assertEqual(len(r7.referrals), 1)
        self.assertEqual(len(r7.warning_signs), 1)
        self.assertEqual(len(r7.care_instructions), 5)
        care_types = [c.instruction_type for c in r7.care_instructions]
        self.assertIn("WOUND_CARE", care_types)
        self.assertIn("DIET", care_types)
        self.assertIn("ACTIVITY", care_types)
        self.assertIn("REHABILITATION", care_types)
        self.assertIn("OTHER_CARE_INSTRUCTION", care_types)
        self.assertEqual(
            r7.validation_report.overall_safety_status,
            SafetyStatus.SAFE_ORGANIZED,
        )
        self.assertTrue(r7.validation_report.patient_releasable)

        # 8. Messy formatting / Unicode / hyphenation / zero-width -> Blocked with SUSPICIOUS_TEXT_ENCODING + STOP_START
        r8 = run_document_intelligence_pipeline(DOC_SYNTH_008_MESSY_UNICODE.raw_text)
        self.assertEqual(r8.document.hyphenations_repaired, 3)
        self.assertEqual(len(r8.document.suspicious_chars), 1)
        self.assertEqual(len(r8.medications), 2)
        self.assertEqual(r8.medications[1].medication_name, "lisinopril")
        self.assertEqual(r8.medications[1].action, MedicationAction.STOP_DISCONTINUE)
        self.assertEqual(len(r8.follow_ups), 1)
        self.assertEqual(r8.follow_ups[0].specialty, "Cardiology")
        self.assertIn(
            HazardCode.SUSPICIOUS_TEXT_ENCODING,
            r8.validation_report.hazard_codes_detected,
        )
        self.assertFalse(r8.validation_report.patient_releasable)

    # ------------------------------------------------------------------
    # Malformed Documents & Ingestion Adapter Boundary Tests
    # ------------------------------------------------------------------
    def test_malformed_and_unsupported_document_inputs(self) -> None:
        with self.assertRaises(MalformedDocumentError):
            ingest_discharge_document("")
        with self.assertRaises(MalformedDocumentError):
            ingest_discharge_document("   \n\t  ")
        with self.assertRaises(MalformedDocumentError):
            ingest_discharge_document("Continue aspirin\x00 81 mg daily.")
        with self.assertRaises(RawTextTooLargeError):
            ingest_discharge_document("A" * 204_801)
        with self.assertRaises(UnsupportedDocumentFormatError):
            ingest_discharge_document(
                "Continue aspirin 81 mg daily.",
                source_format=DocumentFormat.PDF,
            )
        with self.assertRaises(UnsupportedDocumentFormatError):
            ingest_discharge_document(
                "Continue aspirin 81 mg daily.",
                source_format=DocumentFormat.DOCX,
            )

    def test_medication_actions_routes_frequencies_and_edge_cases(self) -> None:
        raw = (
            "DISCHARGE SUMMARY - DOC-EDGE-001\n"
            "Patient Identifier: SYNTH-PT-499\n\n"
            "Discharge Medications:\n"
            "1. Start Eliquis (apixaban) 5 mg by mouth three times daily for 7 days.\n"
            "2. Increase metoprolol from 25 mg to 50 mg orally four times daily.\n"
            "3. Decrease furosemide from 40 mg to 20 mg IV every 8 hours.\n"
            "4. Hold clopidogrel 75 mg daily until surgical follow-up.\n"
            "5. Take nitroglycerin 0.4 mg sublingual as needed for chest pain.\n"
            "6. Apply hydrocortisone 1% topical twice daily.\n\n"
            "Appointments:\n"
            "- Follow up with Cardiology scheduled for next week.\n"
            "- Please schedule a follow up with Pulmonology within 14 days.\n\n"
            "Warning Signs:\n"
            "- Call if you feel worse.\n"
            "- Go to the emergency room immediately for severe headache.\n"
        )
        res = run_document_intelligence_pipeline(raw)
        actions = [m.action for m in res.medications]
        self.assertIn(MedicationAction.NEW_START, actions)
        self.assertIn(MedicationAction.DOSE_CHANGE, actions)
        self.assertIn(MedicationAction.HOLD_TEMPORARY, actions)

        inc_med = next(m for m in res.medications if m.action == MedicationAction.DOSE_CHANGE and m.medication_name == "metoprolol")
        self.assertEqual(inc_med.previous_dosage, "25 mg")
        self.assertEqual(inc_med.dosage, "50 mg")
        self.assertEqual(inc_med.frequency, "four times daily")

        dec_med = next(m for m in res.medications if m.action == MedicationAction.DOSE_CHANGE and m.medication_name == "furosemide")
        self.assertEqual(dec_med.previous_dosage, "40 mg")
        self.assertEqual(dec_med.dosage, "20 mg")
        self.assertEqual(dec_med.route, "IV")

        fu_sched = [f.appointment_scheduled for f in res.follow_ups]
        self.assertIn(True, fu_sched)
        self.assertIn(False, fu_sched)

        ws_urgencies = [w.urgency for w in res.warning_signs]
        self.assertIn(UrgencyLevel.UNKNOWN, ws_urgencies)
        self.assertIn(UrgencyLevel.URGENT, ws_urgencies)

    def test_standalone_tests_and_narrative_only_document(self) -> None:
        raw_tests_only = (
            "Tests:\n"
            "- Obtain a lipid panel for cholesterol monitoring.\n"
        )
        res_t = run_document_intelligence_pipeline(raw_tests_only, document_id="DOC-TEST-ONLY", patient_identifier="PT-T1")
        self.assertEqual(len(res_t.tests), 1)
        self.assertTrue(res_t.tests[0].is_ambiguous)
        self.assertEqual(len(res_t.validator_payload.tasks), 1)

        raw_narrative = (
            "Hospital Course:\n"
            "Patient stabilized during hospitalization.\n"
        )
        res_n = run_document_intelligence_pipeline(raw_narrative, document_id="DOC-NARR", patient_identifier="PT-N1")
        self.assertEqual(res_n.extraction_completeness_score, 0.0)


if __name__ == "__main__":
    unittest.main()

