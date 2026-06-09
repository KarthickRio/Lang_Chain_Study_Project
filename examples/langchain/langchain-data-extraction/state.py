"""
State definition for the medical fax processing agent.

The state is the single source of truth that flows through every node.
Each node ADDS fields — it never removes what came before.
This gives you full traceability: you can always compare
raw extraction vs AI-corrected output.
"""

from typing import Annotated, Sequence, TypedDict, List, Dict, Any, Optional
from enum import Enum
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ── Enums ────────────────────────────────────────────────────────────────────

class PDFType(str, Enum):
    NATIVE  = "native"   # has a real text layer — pymupdf reads it directly
    SCANNED = "scanned"  # image-only pages — needs OCR
    UNKNOWN = "unknown"  # detection failed or mixed content


class DocType(str, Enum):
    REFERRAL       = "referral"
    LAB_RESULT     = "lab_result"
    PRESCRIPTION   = "prescription"
    INSURANCE_AUTH = "insurance_auth"
    UNKNOWN        = "unknown"


class ProcessingPhase(str, Enum):
    INGESTION             = "ingestion"
    NATIVE_PREPROCESSING  = "native_preprocessing"
    SCANNED_PREPROCESSING = "scanned_preprocessing"
    AI_EXTRACTION         = "ai_extraction"
    SYNTHESIS             = "synthesis"


# ── Main state ────────────────────────────────────────────────────────────────

class MedicalFaxState(TypedDict):
    """
    Grows as the document moves through the pipeline.

    Ingestion phase fills:     pdf_path, pdf_type, raw_text
    Preprocessing phase fills: doc_type, doc_type_confidence,
                               extracted_fields, confidence_score,
                               missing_fields
    AI phase fills:            ai_feedback, ai_refined_fields,
                               validation_passed
    Synthesis fills:           final_output
    """

    # ── LangGraph message thread (required by add_messages reducer) ──
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # ── Set once at startup ──────────────────────────────────────────
    pdf_path: str

    # ── Filled by pdf_ingestion_node ─────────────────────────────────
    pdf_type:        PDFType        # "native" | "scanned" | "unknown"
    raw_text:        str            # plain text extracted from the PDF
    page_count:      int            # total pages in the PDF
    text_per_page:   List[str]      # per-page text (useful for scanned QA)

    # ── Filled by preprocessing nodes ────────────────────────────────
    doc_type:             DocType   # classified document type
    doc_type_confidence:  float     # 0.0–1.0 — how sure we are of doc_type
    extracted_fields:     Dict[str, Any]   # e.g. {"patient_name": "John Doe", ...}
    missing_fields:       List[str]        # fields expected but not found
    confidence_score:     float     # % of expected fields that were found

    # ── Filled by AI extraction node ─────────────────────────────────
    ai_feedback:       List[str]           # what the LLM flagged or corrected
    ai_refined_fields: Dict[str, Any]      # LLM-filled versions of missing fields
    validation_passed: bool                # True if AI is satisfied with the result

    # ── Tracking ─────────────────────────────────────────────────────
    current_phase:  ProcessingPhase
    error_messages: List[str]       # non-fatal errors logged during processing

    # ── Final output ─────────────────────────────────────────────────
    final_output: Optional[Dict[str, Any]]