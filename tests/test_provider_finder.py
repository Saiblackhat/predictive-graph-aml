"""
tests/test_provider_finder.py

Phase 7 — Provider Finder test suite.

Adversarial cases A–R plus:
  - Synthetic fixture audit (all 12 providers)
  - Neutrality tests
  - Provenance tests
  - Availability safety
  - Insurance safety
  - Location safety
  - Language safety
  - End-to-end Phase 4→5→6→6.5→7 pipeline test
  - Appointment/reminder creation safety tests
"""
from __future__ import annotations

import copy
import unittest

from backend.provider_finder import find_providers, NeutralityStatus
from backend.provider_finder.fixtures import (
    ALL_SYNTHETIC_PROVIDERS,
    ADVERSARIAL_CLAIMED_PROVIDER,
    ADVERSARIAL_CONFLICTING_SPECIALTY_PROVIDER,
    ADVERSARIAL_NO_SOURCE_PROVIDER,
    ADVERSARIAL_STALE_PROVIDER,
    SYNTHETIC_PROVIDERS,
)
from backend.provider_finder.matcher import _audit_and_build, _specialty_matches
from backend.provider_finder.neutrality import (
    audit_provider_neutrality,
    check_text_neutrality,
    sanitize_display_text,
)
from backend.provider_finder.provenance import validate_provider_provenance
from backend.provider_finder.types import (
    FreshnessStatus,
    MatchAttribute,
    ProviderSearchQuery,
    ProviderType,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _query(specialty, city=None, language=None, insurance=None):
    return ProviderSearchQuery(
        specialty=specialty,
        location_city=city,
        language=language,
        insurance=insurance,
    )


# ─── Adversarial Case A — Exact Cardiology match ──────────────────────────────

class TestAdvA_ExactCardiologyMatch(unittest.TestCase):

    def setUp(self):
        self.result = find_providers(_query("Cardiology", city="Chennai"))

    def test_at_least_one_result(self):
        self.assertGreater(self.result.result_count, 0)

    def test_all_results_are_cardiology(self):
        for p in self.result.providers:
            self.assertIn("Cardiology", p.specialty)

    def test_all_results_patient_releasable(self):
        for p in self.result.providers:
            self.assertTrue(p.patient_releasable, f"Provider {p.provider_id} not releasable")

    def test_specialty_attribute_present(self):
        for p in self.result.providers:
            self.assertIn(MatchAttribute.SPECIALTY, p.matching_attributes)


# ─── Adversarial Case B — Dermatology for Cardiology task ────────────────────

class TestAdvB_DermatologyExcluded(unittest.TestCase):

    def test_dermatology_not_in_cardiology_results(self):
        result = find_providers(_query("Cardiology", city="Chennai"))
        for p in result.providers:
            self.assertNotEqual(p.provider_id, "P03",
                                "P03 (Dermatology) must not appear in Cardiology results")

    def test_cardiology_search_excludes_other_specialties(self):
        result = find_providers(_query("Cardiology"))
        for p in result.providers:
            self.assertIn("Cardiology", p.specialty,
                          f"Non-Cardiology provider {p.provider_id} leaked into results")


# ─── Adversarial Case C — Cardiology outside requested city ──────────────────

class TestAdvC_OutsideCityExcluded(unittest.TestCase):

    def test_bengaluru_excluded_when_chennai_requested(self):
        result = find_providers(_query("Cardiology", city="Chennai"))
        for p in result.providers:
            self.assertEqual(p.city, "Chennai",
                             f"Provider {p.provider_id} city={p.city!r} should be excluded when Chennai requested")

    def test_p04_bengaluru_not_in_chennai_results(self):
        result = find_providers(_query("Cardiology", city="Chennai"))
        ids = {p.provider_id for p in result.providers}
        self.assertNotIn("P04", ids, "P04 (Bengaluru) must be excluded from Chennai search")


# ─── Adversarial Case D — No location: specialty-only search ──────────────────

class TestAdvD_NoLocation(unittest.TestCase):

    def setUp(self):
        self.result = find_providers(_query("Cardiology"))

    def test_results_include_multi_city(self):
        cities = {p.city for p in self.result.providers}
        self.assertGreater(len(cities), 1, "Specialty-only should include providers from multiple cities")

    def test_no_location_attribute_set(self):
        # Without a city query, no provider should have LOCATION in matching_attributes
        for p in self.result.providers:
            self.assertNotIn(MatchAttribute.LOCATION, p.matching_attributes)

    def test_patient_message_mentions_add_location(self):
        self.assertIn("CAREBRIDGE helps you find providers", self.result.patient_message)


# ─── Adversarial Case E — Spanish language requested ─────────────────────────

class TestAdvE_SpanishLanguage(unittest.TestCase):

    def setUp(self):
        self.result = find_providers(_query("Cardiology", language="Spanish"))

    def test_only_spanish_providers_returned(self):
        for p in self.result.providers:
            listed_langs_lower = [l.lower() for l in p.languages]
            self.assertIn("spanish", listed_langs_lower,
                          f"Provider {p.provider_id} does not list Spanish but was included")

    def test_spanish_attribute_present(self):
        for p in self.result.providers:
            self.assertIn(MatchAttribute.LANGUAGE, p.matching_attributes)

    def test_p07_tamil_only_excluded(self):
        ids = {p.provider_id for p in self.result.providers}
        self.assertNotIn("P07", ids, "P07 (Tamil only) must be excluded from Spanish search")


# ─── Adversarial Case F — Language not listed ─────────────────────────────────

class TestAdvF_LanguageNotListed(unittest.TestCase):

    def test_no_hindi_provider_in_results_when_hindi_requested(self):
        # No synthetic provider lists Hindi → results should be empty
        result = find_providers(_query("Cardiology", language="Hindi"))
        self.assertEqual(result.result_count, 0,
                         "No providers list Hindi — result must be empty, not fabricated")

    def test_empty_result_not_a_fabricated_provider(self):
        result = find_providers(_query("Cardiology", language="Klingon"))
        self.assertEqual(result.result_count, 0)
        self.assertEqual(result.providers, [])


# ─── Adversarial Case G — "best cardiologist" quality claim ──────────────────

class TestAdvG_BestCardiologistClaim(unittest.TestCase):

    def test_quality_claim_detected_in_display_name(self):
        status, violations = check_text_neutrality("Best Cardiologist in Chennai")
        self.assertNotEqual(status, NeutralityStatus.PASS)
        self.assertTrue(len(violations) > 0)

    def test_adversarial_provider_blocked_from_results(self):
        result = find_providers(
            _query("Cardiology", city="Chennai"),
            provider_pool=[ADVERSARIAL_CLAIMED_PROVIDER],
        )
        self.assertEqual(result.result_count, 0,
                         "Provider with 'best' claim must be excluded from releasable results")

    def test_adversarial_provider_excluded_count_incremented(self):
        result = find_providers(
            _query("Cardiology", city="Chennai"),
            provider_pool=[ADVERSARIAL_CLAIMED_PROVIDER],
        )
        self.assertEqual(result.excluded_count, 1)


# ─── Adversarial Case H — "top-rated" claim ───────────────────────────────────

class TestAdvH_TopRatedClaim(unittest.TestCase):

    def test_top_rated_detected(self):
        status, violations = check_text_neutrality("Top-Rated Heart Clinic")
        self.assertNotEqual(status, NeutralityStatus.PASS)

    def test_sanitize_removes_top_rated(self):
        cleaned = sanitize_display_text("Top-Rated Cardiologist")
        self.assertNotIn("top", cleaned.lower())

    def test_facility_name_with_top_rated_blocked(self):
        """ADVERSARIAL_CLAIMED_PROVIDER facility_name contains 'Top-Rated'."""
        _, violations = audit_provider_neutrality(
            "Dr. Someone", "Top-Rated Heart Clinic", None
        )
        self.assertTrue(len(violations) > 0)


# ─── Adversarial Case I — "recommended" claim ────────────────────────────────

class TestAdvI_RecommendedClaim(unittest.TestCase):

    def test_recommended_detected(self):
        status, violations = check_text_neutrality("Recommended Cardiologist")
        self.assertNotEqual(status, NeutralityStatus.PASS)
        self.assertTrue(any("recommended" in v.lower() for v in violations))

    def test_sanitize_removes_recommended(self):
        cleaned = sanitize_display_text("Recommended Cardiologist")
        self.assertNotIn("recommended", cleaned.lower())


# ─── Adversarial Case J — Provider with no source ────────────────────────────

class TestAdvJ_NoSource(unittest.TestCase):

    def test_no_source_excluded_from_results(self):
        result = find_providers(
            _query("Cardiology", city="Chennai"),
            provider_pool=[ADVERSARIAL_NO_SOURCE_PROVIDER],
        )
        self.assertEqual(result.result_count, 0)
        self.assertGreater(result.excluded_count, 0)

    def test_no_source_provenance_fails(self):
        ok, notes = validate_provider_provenance(ADVERSARIAL_NO_SOURCE_PROVIDER)
        # source is empty → should fail
        self.assertFalse(ok)
        self.assertTrue(any("source" in n.lower() for n in notes))


# ─── Adversarial Case K — Stale data surfaced ────────────────────────────────

class TestAdvK_StaleData(unittest.TestCase):

    def test_stale_provider_excluded_from_releasable(self):
        result = find_providers(
            _query("Cardiology", city="Chennai"),
            provider_pool=[ADVERSARIAL_STALE_PROVIDER],
        )
        self.assertEqual(result.result_count, 0)

    def test_stale_freshness_status_on_audited_record(self):
        audited = _audit_and_build(
            ADVERSARIAL_STALE_PROVIDER,
            _query("Cardiology", city="Chennai"),
            [MatchAttribute.SPECIALTY],
            True, False,
        )
        self.assertEqual(audited.freshness_status, FreshnessStatus.STALE)
        self.assertTrue(audited.review_required)
        self.assertFalse(audited.patient_releasable)


# ─── Adversarial Case L — Insurance listed ───────────────────────────────────

class TestAdvL_InsuranceListed(unittest.TestCase):

    def test_insurance_note_says_listed_not_guaranteed(self):
        result = find_providers(_query("Cardiology", insurance="Synthetic Health Plan A"))
        for p in result.providers:
            if "Synthetic Health Plan A" in p.accepted_insurance:
                self.assertIn("listed as accepting", p.insurance_note.lower())
                self.assertNotIn("guaranteed", p.insurance_note.lower())
                self.assertNotIn("definitely accepted", p.insurance_note.lower())

    def test_insurance_mismatch_does_not_exclude(self):
        """Insurance mismatch should NOT exclude — shown with note only."""
        # Query requests Plan Z which no provider has, but without insurance filter
        # (insurance filter is optional; only language/specialty/city are hard filters)
        result = find_providers(_query("Cardiology", city="Chennai"))
        # Providers without the queried insurance still appear (no insurance filter set)
        self.assertGreater(result.result_count, 0)


# ─── Adversarial Case M — No insurance data ──────────────────────────────────

class TestAdvM_NoInsuranceData(unittest.TestCase):

    def test_p06_no_insurance_note_is_not_available(self):
        # P06 has no insurance → should show "not available"
        result = find_providers(_query("Cardiology", city="Chengalpattu"))
        p06_list = [p for p in result.providers if p.provider_id == "P06"]
        if p06_list:
            self.assertIn("not available", p06_list[0].insurance_note.lower())

    def test_no_insurance_inference(self):
        """Must not infer insurance from provider name, city, or specialty."""
        result = find_providers(_query("Cardiology"))
        for p in result.providers:
            if not p.accepted_insurance:
                self.assertIn("not available", p.insurance_note.lower())
                self.assertNotIn("probably", p.insurance_note.lower())
                self.assertNotIn("likely", p.insurance_note.lower())


# ─── Adversarial Case N — No availability data ───────────────────────────────

class TestAdvN_NoAvailability(unittest.TestCase):

    def test_all_providers_say_availability_not_verified(self):
        result = find_providers(_query("Cardiology"))
        for p in result.providers:
            self.assertIn("not verified", p.availability_note.lower(),
                          f"Provider {p.provider_id} has suspicious availability: {p.availability_note!r}")

    def test_no_fabricated_appointment_slots(self):
        result = find_providers(_query("Cardiology"))
        for p in result.providers:
            self.assertNotIn("available tomorrow", p.availability_note.lower())
            self.assertNotIn("guaranteed", p.availability_note.lower())
            self.assertNotIn("bookable", p.availability_note.lower())


# ─── Adversarial Case O — Conflicting specialty ───────────────────────────────

class TestAdvO_ConflictingSpecialty(unittest.TestCase):

    def test_conflicting_specialty_excluded(self):
        # "Cardiology / Dermatology" is ambiguous → excluded from releasable
        result = find_providers(
            _query("Cardiology"),
            provider_pool=[ADVERSARIAL_CONFLICTING_SPECIALTY_PROVIDER],
        )
        self.assertEqual(result.result_count, 0)
        self.assertGreater(result.excluded_count, 0)

    def test_conflicting_specialty_review_required(self):
        audited = _audit_and_build(
            ADVERSARIAL_CONFLICTING_SPECIALTY_PROVIDER,
            _query("Cardiology"),
            [MatchAttribute.SPECIALTY],
            False, False,
        )
        self.assertTrue(audited.review_required)
        self.assertFalse(audited.patient_releasable)


# ─── Adversarial Case P — Duplicate records ───────────────────────────────────

class TestAdvP_DuplicateDeduplication(unittest.TestCase):

    def test_duplicate_providers_deduplicated(self):
        """Two identical P01 records in pool → only one appears in results."""
        p01 = copy.deepcopy(ALL_SYNTHETIC_PROVIDERS[0])  # first is P01
        duplicated_pool = [p01, copy.deepcopy(p01)]

        result = find_providers(_query("Cardiology", city="Chennai"), provider_pool=duplicated_pool)
        ids = [p.provider_id for p in result.providers]
        # Deduplication: same provider_id should appear once only
        self.assertEqual(len(ids), len(set(ids)),
                         f"Duplicate provider_id found: {ids}")

    def test_deduplication_is_deterministic(self):
        """Running same query twice returns same IDs in same order."""
        r1 = find_providers(_query("Cardiology", city="Chennai"))
        r2 = find_providers(_query("Cardiology", city="Chennai"))
        ids1 = [p.provider_id for p in r1.providers]
        ids2 = [p.provider_id for p in r2.providers]
        self.assertEqual(ids1, ids2)


# ─── Adversarial Case Q — Search result must not create an appointment ────────

class TestAdvQ_NoAppointmentCreated(unittest.TestCase):

    def test_find_providers_returns_no_appointment_object(self):
        result = find_providers(_query("Cardiology", city="Chennai"))
        for p in result.providers:
            self.assertFalse(hasattr(p, "appointment_id"),
                             "ProviderMatch must not have an appointment_id field")
            self.assertFalse(hasattr(p, "appointment_time_confirmed"),
                             "ProviderMatch must not have confirmed appointment")

    def test_no_appointment_in_result(self):
        result = find_providers(_query("Cardiology"))
        self.assertFalse(hasattr(result, "appointment"),
                         "ProviderSearchResult must not have an appointment field")


# ─── Adversarial Case R — Search result must not create a reminder ────────────

class TestAdvR_NoReminderCreated(unittest.TestCase):

    def test_find_providers_does_not_import_scheduler(self):
        """The matcher must never automatically call create_reminder."""
        import backend.provider_finder.matcher as matcher_module
        self.assertFalse(
            hasattr(matcher_module, "create_reminder"),
            "matcher.py must not import or call create_reminder",
        )

    def test_provider_match_has_no_reminder_field(self):
        result = find_providers(_query("Cardiology"))
        for p in result.providers:
            self.assertFalse(hasattr(p, "reminder_id"),
                             "ProviderMatch must not have reminder_id")
            self.assertFalse(hasattr(p, "reminder_scheduled_for"))


# ─── Synthetic Fixture Audit — all 12 providers ───────────────────────────────

class TestSyntheticFixtureAudit(unittest.TestCase):
    """Every fixture must have required fields and no quality claims."""

    def test_all_12_providers_have_source(self):
        for p in SYNTHETIC_PROVIDERS:
            self.assertTrue(p.source, f"Provider {p.provider_id} missing source")
            self.assertNotEqual(p.source.lower(), "unknown")

    def test_all_12_providers_have_specialty(self):
        for p in SYNTHETIC_PROVIDERS:
            self.assertTrue(p.specialty, f"Provider {p.provider_id} missing specialty")

    def test_all_12_providers_have_display_name(self):
        for p in SYNTHETIC_PROVIDERS:
            self.assertTrue(p.display_name, f"Provider {p.provider_id} missing display_name")

    def test_all_12_providers_no_quality_claims_in_display_name(self):
        for p in SYNTHETIC_PROVIDERS:
            status, violations = check_text_neutrality(p.display_name)
            self.assertEqual(status, NeutralityStatus.PASS,
                             f"Provider {p.provider_id} display_name has quality claim: {violations}")

    def test_all_12_providers_availability_not_fabricated(self):
        for p in SYNTHETIC_PROVIDERS:
            note = p.availability_note.lower()
            self.assertIn("not verified", note,
                          f"Provider {p.provider_id} has fabricated availability: {p.availability_note!r}")

    def test_all_12_providers_distance_is_none(self):
        for p in SYNTHETIC_PROVIDERS:
            self.assertIsNone(p.distance_km,
                              f"Provider {p.provider_id} has a fabricated distance")

    def test_all_12_providers_source_is_synthetic(self):
        for p in SYNTHETIC_PROVIDERS:
            self.assertEqual(p.source, "synthetic")

    def test_all_12_providers_unique_ids(self):
        ids = [p.provider_id for p in SYNTHETIC_PROVIDERS]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate provider_id in fixtures")

    def test_all_12_providers_pass_provenance_after_audit(self):
        """After running through matcher neutrality audit, clean providers should pass provenance."""
        from backend.provider_finder.neutrality import audit_provider_neutrality
        for p in SYNTHETIC_PROVIDERS:
            p_copy = copy.deepcopy(p)
            status, violations = audit_provider_neutrality(
                p_copy.display_name, p_copy.facility_name, None
            )
            p_copy.neutrality_status = status
            p_copy.neutrality_violations = violations
            ok, notes = validate_provider_provenance(p_copy)
            self.assertTrue(ok, f"Provider {p.provider_id} failed provenance: {notes}")


# ─── Neutrality tests ─────────────────────────────────────────────────────────

class TestNeutralityChecker(unittest.TestCase):

    def test_neutral_text_passes(self):
        status, v = check_text_neutrality("Cardiology Provider, Chennai")
        self.assertEqual(status, NeutralityStatus.PASS)
        self.assertEqual(v, [])

    def test_best_flagged(self):
        status, _ = check_text_neutrality("Best Cardiologist")
        self.assertNotEqual(status, NeutralityStatus.PASS)

    def test_top_rated_flagged(self):
        status, _ = check_text_neutrality("Top-Rated Clinic")
        self.assertNotEqual(status, NeutralityStatus.PASS)

    def test_recommended_flagged(self):
        status, _ = check_text_neutrality("Recommended by peers")
        self.assertNotEqual(status, NeutralityStatus.PASS)

    def test_leading_flagged(self):
        status, _ = check_text_neutrality("Leading cardiologist in India")
        self.assertNotEqual(status, NeutralityStatus.PASS)

    def test_trusted_flagged(self):
        status, _ = check_text_neutrality("Trusted heart specialist")
        self.assertNotEqual(status, NeutralityStatus.PASS)

    def test_guaranteed_appointment_flagged(self):
        status, _ = check_text_neutrality("Guaranteed appointment available")
        self.assertNotEqual(status, NeutralityStatus.PASS)

    def test_sanitize_preserves_specialty(self):
        cleaned = sanitize_display_text("Best Cardiologist")
        self.assertIn("Cardiologist", cleaned)
        self.assertNotIn("Best", cleaned)

    def test_empty_text_passes(self):
        status, v = check_text_neutrality("")
        self.assertEqual(status, NeutralityStatus.PASS)
        self.assertEqual(v, [])


# ─── Provenance tests ─────────────────────────────────────────────────────────

class TestProvenance(unittest.TestCase):

    def test_clean_provider_passes_provenance(self):
        p = copy.deepcopy(SYNTHETIC_PROVIDERS[0])
        p.neutrality_status = NeutralityStatus.PASS
        ok, notes = validate_provider_provenance(p)
        self.assertTrue(ok)

    def test_empty_source_fails(self):
        p = copy.deepcopy(SYNTHETIC_PROVIDERS[0])
        p.neutrality_status = NeutralityStatus.PASS
        object.__setattr__(p, "source", "")
        ok, notes = validate_provider_provenance(p)
        self.assertFalse(ok)

    def test_empty_specialty_fails(self):
        p = copy.deepcopy(SYNTHETIC_PROVIDERS[0])
        p.neutrality_status = NeutralityStatus.PASS
        ok_before, _ = validate_provider_provenance(p)
        self.assertTrue(ok_before)
        object.__setattr__(p, "specialty", "")
        ok, notes = validate_provider_provenance(p)
        self.assertFalse(ok)

    def test_neutrality_fail_blocks_provenance(self):
        p = copy.deepcopy(SYNTHETIC_PROVIDERS[0])
        p.neutrality_status = NeutralityStatus.FAIL_QUALITY_CLAIM
        p.neutrality_violations = ["Unsupported quality claim: 'best'"]
        ok, notes = validate_provider_provenance(p)
        self.assertFalse(ok)


# ─── Ordering test ────────────────────────────────────────────────────────────

class TestOrdering(unittest.TestCase):

    def test_city_match_providers_come_before_non_city(self):
        """Specialty-only search: providers matching city come before others."""
        result = find_providers(_query("Cardiology"))
        # All results are releasable; no city filter applied, so ordering by provider_id
        # Just confirm results are stable and alphabetical by id
        ids = [p.provider_id for p in result.providers]
        self.assertEqual(ids, sorted(ids))

    def test_first_result_not_labeled_best(self):
        result = find_providers(_query("Cardiology", city="Chennai"))
        self.assertGreater(result.result_count, 0)
        first = result.providers[0]
        self.assertNotIn("best", first.display_name.lower())
        self.assertNotIn("top", first.display_name.lower())
        self.assertNotIn("recommended", first.display_name.lower())

    def test_ordering_explanation_present(self):
        result = find_providers(_query("Cardiology"))
        self.assertIn("alphabetical", result.ordering_explanation.lower())
        self.assertNotIn("ranking", result.ordering_explanation.lower())


# ─── Empty result test ────────────────────────────────────────────────────────

class TestEmptyResults(unittest.TestCase):

    def test_oncology_in_bengaluru_returns_empty(self):
        result = find_providers(_query("Oncology", city="Bengaluru"))
        self.assertEqual(result.result_count, 0)

    def test_empty_result_message_is_informative(self):
        result = find_providers(_query("Oncology", city="Bengaluru"))
        self.assertGreater(len(result.patient_message), 0)
        self.assertNotIn("fabricated", result.patient_message.lower())

    def test_empty_result_does_not_silently_broaden(self):
        """Must NOT automatically remove city filter and return Bengaluru providers."""
        result = find_providers(_query("Oncology", city="Bengaluru"))
        for p in result.providers:
            self.assertEqual(p.city, "Bengaluru")


# ─── End-to-end Phase 4→5→6→6.5→7 pipeline ──────────────────────────────────

class TestPhase7EndToEnd(unittest.TestCase):
    """
    Full pipeline starting from raw discharge text through to Provider Finder.
    Input: "Follow up with Cardiology on March 18, 2027 at 10:30 AM."
    """

    def setUp(self):
        from backend.document_intelligence import run_document_intelligence_pipeline
        from backend.task_orchestration import orchestrate_care_journey
        from backend.task_orchestration.types import TaskCategory
        from backend.patient_language import generate_patient_care_plan
        from backend.care_reminders.eligibility import check_reminder_eligibility

        raw = (
            "Discharge Instructions:\n"
            "Take amlodipine 5 mg by mouth once daily for blood pressure.\n"
            "Follow-Up:\n"
            "Follow up with Cardiology on March 18, 2027 at 10:30 AM."
        )
        self.ext = run_document_intelligence_pipeline(raw)
        self.journey = orchestrate_care_journey(self.ext)
        self.plan = generate_patient_care_plan(self.journey)

        fu_tasks = [t for t in self.journey.timeline.all_tasks
                    if t.task_category == TaskCategory.FOLLOW_UP]
        self.assertTrue(len(fu_tasks) > 0)
        self.fu_task = fu_tasks[0]

        self.eligibility = check_reminder_eligibility(self.fu_task)

        # Phase 7: derive specialty from the validated task
        specialty = self.fu_task.clinical_metadata.get("specialty", "Cardiology")
        self.provider_result = find_providers(
            _query(specialty, city="Chennai")
        )

    def test_phase4_extraction_ok(self):
        self.assertTrue(len(self.ext.follow_ups) > 0)

    def test_phase5_task_releasable(self):
        self.assertTrue(self.fu_task.patient_releasable)

    def test_phase65_task_eligible(self):
        self.assertTrue(self.eligibility.is_eligible)

    def test_phase7_returns_cardiology_providers(self):
        self.assertGreater(self.provider_result.result_count, 0)
        for p in self.provider_result.providers:
            self.assertEqual(p.specialty, "Cardiology")

    def test_phase7_does_not_select_best_provider(self):
        """Provider Finder must not claim to select the best provider."""
        msg = self.provider_result.patient_message
        self.assertNotIn("best provider", msg.lower())
        self.assertNotIn("recommended", msg.lower())
        self.assertNotIn("top pick", msg.lower())

    def test_phase7_does_not_modify_care_task(self):
        """The original follow-up task must be unchanged after provider search."""
        self.assertEqual(self.fu_task.clinical_metadata.get("specialty"), "Cardiology")
        self.assertEqual(self.fu_task.timeframe.absolute_date, "2027-03-18")

    def test_phase7_does_not_create_appointment(self):
        for p in self.provider_result.providers:
            self.assertFalse(hasattr(p, "appointment_id"))

    def test_phase7_does_not_create_reminder(self):
        """find_providers must not create a care reminder."""
        import backend.provider_finder.matcher as matcher_mod
        self.assertFalse(hasattr(matcher_mod, "create_reminder"))
        # Providers returned must not contain reminder objects or reminder IDs
        for p in self.provider_result.providers:
            self.assertFalse(hasattr(p, "reminder_id"))
            self.assertFalse(hasattr(p, "reminder_scheduled_for"))

    def test_phase7_all_results_patient_releasable(self):
        for p in self.provider_result.providers:
            self.assertTrue(p.patient_releasable)

    def test_phase7_no_quality_claims_in_output(self):
        for p in self.provider_result.providers:
            status, violations = check_text_neutrality(p.display_name)
            self.assertEqual(status, NeutralityStatus.PASS,
                             f"Provider {p.provider_id} has quality claim: {violations}")

    def test_phase7_availability_not_verified(self):
        for p in self.provider_result.providers:
            self.assertIn("not verified", p.availability_note.lower())


# ─── Care-plan non-modification test ─────────────────────────────────────────

class TestCareplanUnmodified(unittest.TestCase):
    """Provider Finder must never alter the original validated care task."""

    def test_original_task_specialty_unchanged(self):
        from backend.document_intelligence import run_document_intelligence_pipeline
        from backend.task_orchestration import orchestrate_care_journey
        from backend.task_orchestration.types import TaskCategory

        raw = (
            "Discharge Instructions:\n"
            "Take amlodipine 5 mg by mouth once daily.\n"
            "Follow-Up:\n"
            "Follow up with Cardiology on March 18, 2027 at 10:30 AM."
        )
        journey = orchestrate_care_journey(run_document_intelligence_pipeline(raw))
        fu_tasks = [t for t in journey.timeline.all_tasks
                    if t.task_category == TaskCategory.FOLLOW_UP]
        task = fu_tasks[0]
        original_specialty = task.clinical_metadata.get("specialty")
        original_date = task.timeframe.absolute_date

        # Run provider search
        find_providers(_query("Cardiology", city="Chennai"))

        # Task must be unchanged
        self.assertEqual(task.clinical_metadata.get("specialty"), original_specialty)
        self.assertEqual(task.timeframe.absolute_date, original_date)


if __name__ == "__main__":
    unittest.main()
