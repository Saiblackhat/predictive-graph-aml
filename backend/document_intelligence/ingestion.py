"""Document ingestion abstraction for CareBridge Phase 4 Document Intelligence.

Supports synthetic discharge summaries and plain-text documents out of the box,
with a clean adapter abstraction for future PDF and DOCX binary ingestion.
Never fabricates page numbers; provenance uses section, sentence, and character spans.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional

from backend.document_intelligence.normalizer import normalize_document_text
from backend.document_intelligence.sections import detect_sections
from backend.document_intelligence.sentences import segment_sentences
from backend.document_intelligence.types import DocumentFormat, IngestedDocument
from backend.schemas.models import MAX_RAW_DISCHARGE_TEXT_BYTES, RawTextTooLargeError


class UnsupportedDocumentFormatError(NotImplementedError):
    """Raised when a document format adapter is registered as a future extension stub."""


class MalformedDocumentError(ValueError):
    """Raised when an input document is empty, binary-corrupted, or exceeds size limits."""


class BaseDocumentIngestor(ABC):
    """Abstract base interface for format-specific discharge document ingestors."""

    supported_format: DocumentFormat

    @abstractmethod
    def ingest(
        self,
        raw_content: str,
        document_id: Optional[str] = None,
        patient_identifier: Optional[str] = None,
    ) -> IngestedDocument:
        """Ingest raw document content into an IngestedDocument with sections and sentences."""


HEADER_DOC_ID_RE = re.compile(
    r"DISCHARGE\s+SUMMARY\s*[-—:]\s*([A-Za-z0-9_.-]+)", re.IGNORECASE
)
HEADER_PATIENT_ID_RE = re.compile(
    r"(?:Patient(?:\s+ID|\s+Identifier)?|MRN)\s*:\s*([A-Za-z0-9_.-]+)", re.IGNORECASE
)


def _validate_raw_text(raw_content: str) -> str:
    """Validate that raw_content is a non-blank string within the 200 KB UTF-8 byte cap."""
    if not isinstance(raw_content, str):
        raise MalformedDocumentError("Document content must be a string.")
    if "\x00" in raw_content:
        raise MalformedDocumentError(
            "Document contains null bytes (binary corruption or unsupported binary stream)."
        )
    if len(raw_content.encode("utf-8")) > MAX_RAW_DISCHARGE_TEXT_BYTES:
        raise RawTextTooLargeError(
            f"Document exceeds maximum size of {MAX_RAW_DISCHARGE_TEXT_BYTES} bytes (200 KB)."
        )
    if not raw_content.strip():
        raise MalformedDocumentError("Document raw_text cannot be empty or whitespace-only.")
    return raw_content


class TextDischargeIngestor(BaseDocumentIngestor):
    """Deterministic ingestor for synthetic discharge summaries and plain-text notes."""

    def __init__(self, fmt: DocumentFormat = DocumentFormat.SYNTHETIC_SUMMARY) -> None:
        self.supported_format = fmt

    def ingest(
        self,
        raw_content: str,
        document_id: Optional[str] = None,
        patient_identifier: Optional[str] = None,
    ) -> IngestedDocument:
        validated_raw = _validate_raw_text(raw_content)

        # Infer document_id or patient_identifier from header if not explicitly supplied
        resolved_doc_id = (document_id or "").strip()
        if not resolved_doc_id:
            m_doc = HEADER_DOC_ID_RE.search(validated_raw)
            resolved_doc_id = m_doc.group(1) if m_doc else "DOC-SYNTH-UNNAMED"

        resolved_pt_id = (patient_identifier or "").strip()
        if not resolved_pt_id:
            m_pt = HEADER_PATIENT_ID_RE.search(validated_raw)
            resolved_pt_id = m_pt.group(1) if m_pt else "SYNTH-PT-UNKNOWN"

        norm_doc = normalize_document_text(validated_raw)
        sections = detect_sections(norm_doc)
        sentences = segment_sentences(norm_doc, sections)

        return IngestedDocument(
            document_id=resolved_doc_id,
            patient_identifier=resolved_pt_id,
            source_format=self.supported_format,
            raw_text=validated_raw,
            normalized_text=norm_doc.normalized_text,
            sections=sections,
            sentences=sentences,
            suspicious_chars=norm_doc.suspicious_chars,
            hyphenations_repaired=norm_doc.hyphenations_repaired,
        )


class PdfDocumentIngestorStub(BaseDocumentIngestor):
    """Future extension adapter for PDF discharge summaries."""

    supported_format = DocumentFormat.PDF

    def ingest(
        self,
        raw_content: str,
        document_id: Optional[str] = None,
        patient_identifier: Optional[str] = None,
    ) -> IngestedDocument:
        raise UnsupportedDocumentFormatError(
            "PDF binary ingestion is not yet enabled in Phase 4 foundation; use PLAIN_TEXT or SYNTHETIC_SUMMARY."
        )


class DocxDocumentIngestorStub(BaseDocumentIngestor):
    """Future extension adapter for DOCX discharge summaries."""

    supported_format = DocumentFormat.DOCX

    def ingest(
        self,
        raw_content: str,
        document_id: Optional[str] = None,
        patient_identifier: Optional[str] = None,
    ) -> IngestedDocument:
        raise UnsupportedDocumentFormatError(
            "DOCX binary ingestion is not yet enabled in Phase 4 foundation; use PLAIN_TEXT or SYNTHETIC_SUMMARY."
        )


INGESTOR_REGISTRY: dict[DocumentFormat, BaseDocumentIngestor] = {
    DocumentFormat.SYNTHETIC_SUMMARY: TextDischargeIngestor(DocumentFormat.SYNTHETIC_SUMMARY),
    DocumentFormat.PLAIN_TEXT: TextDischargeIngestor(DocumentFormat.PLAIN_TEXT),
    DocumentFormat.PDF: PdfDocumentIngestorStub(),
    DocumentFormat.DOCX: DocxDocumentIngestorStub(),
}


def ingest_discharge_document(
    raw_text: str,
    document_id: Optional[str] = None,
    patient_identifier: Optional[str] = None,
    source_format: DocumentFormat = DocumentFormat.SYNTHETIC_SUMMARY,
) -> IngestedDocument:
    """Ingest a discharge document through the registered format adapter."""
    ingestor = INGESTOR_REGISTRY.get(source_format)
    if ingestor is None:
        raise UnsupportedDocumentFormatError(f"Unsupported document format: {source_format}")
    return ingestor.ingest(
        raw_content=raw_text,
        document_id=document_id,
        patient_identifier=patient_identifier,
    )
