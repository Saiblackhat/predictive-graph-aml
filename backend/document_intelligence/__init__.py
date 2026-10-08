"""Phase 4 Document Intelligence package for CAREBRIDGE.

Pipeline architecture:
Synthetic discharge summary -> Document ingestion -> Text normalization ->
Section detection -> Sentence segmentation -> Evidence span extraction ->
Structured discharge extraction -> Existing Phase 3 safety validator ->
Care-plan-ready structured output.
"""

from backend.document_intelligence.extractor import (
    extract_discharge_intelligence,
    run_document_intelligence_pipeline,
)
from backend.document_intelligence.fixtures import (
    ALL_PHASE4_SYNTHETIC_DOCUMENTS,
    DOC_SYNTH_001_NORMAL,
    DOC_SYNTH_002_MULTI,
    DOC_SYNTH_003_MISSING_AMBIGUOUS,
    DOC_SYNTH_004_CONFLICTING_MEDS,
    DOC_SYNTH_005_WARNING_SIGNS,
    DOC_SYNTH_006_TESTS_AND_REFERRALS,
    DOC_SYNTH_007_RICH_MULTISECTION,
    DOC_SYNTH_008_MESSY_UNICODE,
    SyntheticDischargeDocumentFixture,
)
from backend.document_intelligence.ingestion import (
    BaseDocumentIngestor,
    DocxDocumentIngestorStub,
    MalformedDocumentError,
    PdfDocumentIngestorStub,
    TextDischargeIngestor,
    UnsupportedDocumentFormatError,
    ingest_discharge_document,
)
from backend.document_intelligence.normalizer import (
    detect_suspicious_characters,
    normalize_document_text,
)
from backend.document_intelligence.sections import (
    classify_header_title,
    detect_sections,
)
from backend.document_intelligence.sentences import segment_sentences
from backend.document_intelligence.types import (
    DetectedSection,
    DocumentFormat,
    DocumentIntelligenceResult,
    EvidenceSpan,
    ExtractedCareInstructionRecord,
    ExtractedFollowUpRecord,
    ExtractedMedicationRecord,
    ExtractedReferralRecord,
    ExtractedTestRecord,
    ExtractedWarningSignRecord,
    IngestedDocument,
    NormalizedDocumentText,
    SectionCategory,
    SegmentedSentence,
    SuspiciousCharFinding,
)

__all__ = [
    "ALL_PHASE4_SYNTHETIC_DOCUMENTS",
    "BaseDocumentIngestor",
    "DOC_SYNTH_001_NORMAL",
    "DOC_SYNTH_002_MULTI",
    "DOC_SYNTH_003_MISSING_AMBIGUOUS",
    "DOC_SYNTH_004_CONFLICTING_MEDS",
    "DOC_SYNTH_005_WARNING_SIGNS",
    "DOC_SYNTH_006_TESTS_AND_REFERRALS",
    "DOC_SYNTH_007_RICH_MULTISECTION",
    "DOC_SYNTH_008_MESSY_UNICODE",
    "DetectedSection",
    "DocumentFormat",
    "DocumentIntelligenceResult",
    "DocxDocumentIngestorStub",
    "EvidenceSpan",
    "ExtractedCareInstructionRecord",
    "ExtractedFollowUpRecord",
    "ExtractedMedicationRecord",
    "ExtractedReferralRecord",
    "ExtractedTestRecord",
    "ExtractedWarningSignRecord",
    "IngestedDocument",
    "MalformedDocumentError",
    "NormalizedDocumentText",
    "PdfDocumentIngestorStub",
    "SectionCategory",
    "SegmentedSentence",
    "SuspiciousCharFinding",
    "SyntheticDischargeDocumentFixture",
    "TextDischargeIngestor",
    "UnsupportedDocumentFormatError",
    "classify_header_title",
    "detect_sections",
    "detect_suspicious_characters",
    "extract_discharge_intelligence",
    "ingest_discharge_document",
    "normalize_document_text",
    "run_document_intelligence_pipeline",
    "segment_sentences",
]
