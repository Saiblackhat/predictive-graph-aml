/**
 * Strict TypeScript data contracts and runtime schema guards for CAREBRIDGE.
 *
 * Mirrors backend/schemas/models.py 1-to-1.
 * Explicitly distinguishes:
 * - SAFE / ORGANIZED
 * - NEEDS REVIEW
 * - CONFLICT DETECTED
 * - SOURCE EVIDENCE
 */

export const UI_DISTINCTION_BADGES = {
  SAFE_ORGANIZED: "SAFE / ORGANIZED",
  NEEDS_REVIEW: "NEEDS REVIEW",
  CONFLICT_DETECTED: "CONFLICT DETECTED",
  SOURCE_EVIDENCE: "SOURCE EVIDENCE",
} as const;

export type SafetyStatus =
  | typeof UI_DISTINCTION_BADGES.SAFE_ORGANIZED
  | typeof UI_DISTINCTION_BADGES.NEEDS_REVIEW
  | typeof UI_DISTINCTION_BADGES.CONFLICT_DETECTED;

export type ReviewDecisionStatus =
  | "AUTO_APPROVED"
  | "PENDING_HUMAN_REVIEW"
  | "HUMAN_RESOLVED"
  | "ESCALATED_TO_CLINICIAN";

export type MedicationAction =
  | "CONTINUE"
  | "NEW_START"
  | "STOP_DISCONTINUE"
  | "DOSE_CHANGE"
  | "HOLD_TEMPORARY"
  | "UNKNOWN";

export type UrgencyLevel =
  | "ROUTINE"
  | "SOON"
  | "URGENT"
  | "EMERGENT_911"
  | "UNKNOWN";

export type HazardCode =
  | "MISSING_FOLLOWUP_TIMEFRAME"
  | "MISSING_SPECIALTY"
  | "AMBIGUOUS_MEDICATION_INSTRUCTION"
  | "MEDICATION_CHANGE_DETECTED"
  | "STOP_START_MEDICATION"
  | "UNCLEAR_DOSAGE"
  | "CONFLICTING_INSTRUCTIONS"
  | "UNCLEAR_SYMPTOMS"
  | "WARNING_SIGN_ESCALATION"
  | "UNSUPPORTED_PROVIDER_CLAIM"
  | "MISSING_SOURCE_EVIDENCE"
  | "PROHIBITED_CLINICAL_ACTION"
  | "LOW_EXTRACTION_CONFIDENCE"
  | "UNSUPPORTED_FIELD_VALUE"
  | "OMITTED_INSTRUCTION"
  | "INCOMPLETE_EXTRACTION"
  | "DUPLICATE_ITEM"
  | "SUSPICIOUS_TEXT_ENCODING";

export type HazardSeverity = "INFO" | "WARNING" | "HIGH" | "CRITICAL";

export type TaskCategory =
  | "MEDICATION"
  | "FOLLOW_UP"
  | "WARNING_SIGN"
  | "ACTIVITY_DIET"
  | "LAB_TEST";

export interface SourceEvidence {
  readonly badge: typeof UI_DISTINCTION_BADGES.SOURCE_EVIDENCE;
  readonly source_text: string;
  readonly source_section: string | null;
  readonly source_page: number | null;
  readonly start_char: number | null;
  readonly end_char: number | null;
  readonly span_status_note?: string;
  readonly source_page_verified?: boolean;
}

export interface SafetyFlag {
  readonly code: HazardCode;
  readonly severity: HazardSeverity;
  readonly message: string;
  readonly item_id: string | null;
  readonly requires_human_review: boolean;
  readonly evidence_snippet: string | null;
}

export interface ExtractedItemBase {
  readonly id: string;
  readonly source_evidence: SourceEvidence | null;
  readonly confidence: number;
  readonly safety_status: SafetyStatus;
  readonly review_status: ReviewDecisionStatus;
  readonly release_blocked?: boolean;
  readonly safety_flags: readonly SafetyFlag[];
  readonly patient_friendly_summary: string | null;
  readonly provider_claim_note?: string | null;
}

export interface MedicationInstruction extends ExtractedItemBase {
  readonly medication_name: string;
  readonly action: MedicationAction;
  readonly dosage: string | null;
  readonly frequency: string | null;
  readonly route: string | null;
  readonly previous_dosage: string | null;
  readonly indication_or_note: string | null;
}

export interface FollowUpInstruction extends ExtractedItemBase {
  readonly specialty: string | null;
  readonly provider_name: string | null;
  readonly timeframe: string | null;
  readonly reason: string | null;
  readonly appointment_scheduled: boolean | null;
  readonly provider_claim_note: string | null;
}

export interface WarningSignInstruction extends ExtractedItemBase {
  readonly symptom_description: string;
  readonly urgency: UrgencyLevel;
  readonly action_to_take: string | null;
}

export interface CareTaskItem extends ExtractedItemBase {
  readonly category: TaskCategory;
  readonly title: string;
  readonly due_window: string | null;
  readonly linked_item_id: string;
}

export interface SyntheticProviderMatch {
  readonly provider_id: string;
  readonly name: string;
  readonly specialty: string;
  readonly facility: string;
  readonly distance_miles: number;
  readonly accepts_new_patients_listed: boolean;
  readonly verification_disclaimer: string;
  readonly claim_text: string | null;
}

export interface DischargeExtractionPayload {
  readonly case_id: string;
  readonly patient_identifier: string;
  readonly raw_discharge_text: string;
  readonly medications: readonly MedicationInstruction[];
  readonly follow_ups: readonly FollowUpInstruction[];
  readonly warning_signs: readonly WarningSignInstruction[];
  readonly tasks: readonly CareTaskItem[];
  readonly provider_matches: readonly SyntheticProviderMatch[];
  readonly overall_safety_status: SafetyStatus;
  readonly requires_human_review: boolean;
  readonly global_safety_flags: readonly SafetyFlag[];
  readonly clinical_disclaimer: string;
}

export interface ValidationReport {
  readonly case_id: string;
  readonly overall_safety_status: SafetyStatus;
  readonly requires_human_review: boolean;
  readonly patient_releasable: boolean;
  readonly payload_sha256?: string;
  readonly total_flags: number;
  readonly hazard_codes_detected: readonly HazardCode[];
  readonly validated_payload: DischargeExtractionPayload;
}

const VALID_SAFETY_STATUSES = new Set<string>([
  UI_DISTINCTION_BADGES.SAFE_ORGANIZED,
  UI_DISTINCTION_BADGES.NEEDS_REVIEW,
  UI_DISTINCTION_BADGES.CONFLICT_DETECTED,
]);

/**
 * Runtime guard verifying untrusted JSON before entering the frontend state.
 */
export function isValidExtractedItemBase(
  candidate: unknown
): candidate is ExtractedItemBase {
  if (typeof candidate !== "object" || candidate === null) {
    return false;
  }
  const record = candidate as Record<string, unknown>;
  if (typeof record["id"] !== "string" || record["id"].trim().length === 0) {
    return false;
  }
  if (
    typeof record["confidence"] !== "number" ||
    Number.isNaN(record["confidence"]) ||
    record["confidence"] < 0 ||
    record["confidence"] > 1
  ) {
    return false;
  }
  if (
    typeof record["safety_status"] !== "string" ||
    !VALID_SAFETY_STATUSES.has(record["safety_status"])
  ) {
    return false;
  }
  if (!Array.isArray(record["safety_flags"])) {
    return false;
  }
  if (record["source_evidence"] !== null) {
    const ev = record["source_evidence"];
    if (typeof ev !== "object" || ev === null) {
      return false;
    }
    const evRec = ev as Record<string, unknown>;
    if (
      typeof evRec["source_text"] !== "string" ||
      evRec["source_text"].trim().length === 0
    ) {
      return false;
    }
  }
  return true;
}
