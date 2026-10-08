/**
 * Executable contract & boundary test suite for TypeScript schemas.
 * Can be run directly with Node 22 (`node --experimental-strip-types`).
 */

import {
  UI_DISTINCTION_BADGES,
  isValidExtractedItemBase,
  type FollowUpInstruction,
  type MedicationInstruction,
} from "./schema.ts";

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(`TypeScript Contract Assertion Failed: ${message}`);
  }
}

// 1. Verify UI status badges match exact clinical requirements
assert(
  UI_DISTINCTION_BADGES.SAFE_ORGANIZED === "SAFE / ORGANIZED",
  "SAFE_ORGANIZED badge mismatch"
);
assert(
  UI_DISTINCTION_BADGES.NEEDS_REVIEW === "NEEDS REVIEW",
  "NEEDS_REVIEW badge mismatch"
);
assert(
  UI_DISTINCTION_BADGES.CONFLICT_DETECTED === "CONFLICT DETECTED",
  "CONFLICT_DETECTED badge mismatch"
);
assert(
  UI_DISTINCTION_BADGES.SOURCE_EVIDENCE === "SOURCE EVIDENCE",
  "SOURCE_EVIDENCE badge mismatch"
);

// 2. Verify valid medication instruction passes runtime guard
const validMedication: MedicationInstruction = {
  id: "med-ts-1",
  medication_name: "Atorvastatin",
  action: "CONTINUE",
  dosage: "40 mg",
  frequency: "once daily",
  route: "by mouth",
  previous_dosage: null,
  indication_or_note: null,
  confidence: 0.98,
  safety_status: UI_DISTINCTION_BADGES.SAFE_ORGANIZED,
  review_status: "AUTO_APPROVED",
  safety_flags: [],
  patient_friendly_summary: "Keep taking Atorvastatin 40 mg once daily.",
  source_evidence: {
    badge: UI_DISTINCTION_BADGES.SOURCE_EVIDENCE,
    source_text: "Continue Atorvastatin 40 mg by mouth once daily.",
    source_section: "Discharge Medications",
    source_page: 1,
    start_char: 0,
    end_char: 49,
  },
};

assert(
  isValidExtractedItemBase(validMedication),
  "Valid MedicationInstruction should pass runtime guard"
);

// 3. Verify incomplete follow-up preserves explicit null for missing fields
const incompleteFollowUp: FollowUpInstruction = {
  id: "fu-ts-incomplete",
  specialty: null,
  provider_name: null,
  timeframe: null,
  reason: "post-discharge check",
  appointment_scheduled: null,
  provider_claim_note: null,
  confidence: 0.85,
  safety_status: UI_DISTINCTION_BADGES.NEEDS_REVIEW,
  review_status: "PENDING_HUMAN_REVIEW",
  safety_flags: [
    {
      code: "MISSING_FOLLOWUP_TIMEFRAME",
      severity: "HIGH",
      message: "Missing follow-up timeframe",
      item_id: "fu-ts-incomplete",
      requires_human_review: true,
      evidence_snippet: null,
    },
  ],
  patient_friendly_summary: null,
  source_evidence: null,
};

assert(
  isValidExtractedItemBase(incompleteFollowUp),
  "Incomplete FollowUpInstruction with explicit nulls should pass structural guard"
);

// 4. Intentionally attempt to break runtime guard with malformed inputs
assert(
  !isValidExtractedItemBase(null),
  "null must be rejected by runtime guard"
);
assert(
  !isValidExtractedItemBase({ ...validMedication, confidence: 1.5 }),
  "Confidence > 1.0 must be rejected"
);
assert(
  !isValidExtractedItemBase({ ...validMedication, confidence: -0.1 }),
  "Confidence < 0.0 must be rejected"
);
assert(
  !isValidExtractedItemBase({ ...validMedication, safety_status: "UNKNOWN_BADGE" }),
  "Invalid safety_status badge must be rejected"
);
assert(
  !isValidExtractedItemBase({
    ...validMedication,
    source_evidence: { source_text: "   " },
  }),
  "Blank source_evidence.source_text must be rejected"
);

console.log("PASS: All TypeScript schema contracts and adversarial guards verified.");
