"""
state.py

Single source of truth for the entire pipeline.
Every node reads from state and adds to it — nothing is ever deleted.
This gives full traceability at synthesis: you can compare
raw regex output vs AI-corrected output vs tool-verified output.

Changes from previous version
──────────────────────────────
1. DocType: added RX_RENEWAL for US prescription refill/renewal faxes
2. ProcessingPhase: added TOOL_EXECUTION to track when LLM calls tools
3. MedicalFaxState: 
   - Renamed fields to US standard terminology
   - Added pharmacy vs prescriber NPI separation
   - Added DEA, rx_number, sig, last_filled, drug_prescribed, strength
   - Added phi_fields for HIPAA awareness
   - Added tool_results to track what each tool returned
   - Added ndc_code and npi_verified from tool calls
"""

from typing import Annotated, Sequence, TypedDict, List, Dict, Any, Optional
from enum import Enum
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ── Enums ─────────────────────────────────────────────────────────────────────

class PDFType(str, Enum):
    NATIVE  = "native"    # real text layer — pymupdf reads directly
    SCANNED = "scanned"   # image only — needs OCR
    UNKNOWN = "unknown"   # detection failed


class DocType(str, Enum):
    # ── Added RX_RENEWAL ──────────────────────────────────────────────
    # Separate from PRESCRIPTION because renewal faxes have different
    # fields: last_filled, eRx ID, NCPDP, pharmacy block + prescriber block
    # A standard PRESCRIPTION is a new Rx written by a provider directly
    # A RX_RENEWAL is a pharmacy requesting refill approval from prescriber
    RX_RENEWAL     = "rx_renewal"
    PRESCRIPTION   = "prescription"
    REFERRAL       = "referral"
    LAB_RESULT     = "lab_result"
    INSURANCE_AUTH = "insurance_auth"
    UNKNOWN        = "unknown"


class ProcessingPhase(str, Enum):
    INGESTION             = "ingestion"
    NATIVE_PREPROCESSING  = "native_preprocessing"
    SCANNED_PREPROCESSING = "scanned_preprocessing"
    # ── Added TOOL_EXECUTION ──────────────────────────────────────────
    # Distinct phase so logs show clearly when LLM is calling tools
    # vs when it is doing reasoning
    TOOL_EXECUTION        = "tool_execution"
    AI_EXTRACTION         = "ai_extraction"
    SYNTHESIS             = "synthesis"


# ── Main state ────────────────────────────────────────────────────────────────

class MedicalFaxState(TypedDict):
    """
    Grows as document moves through pipeline.

    Ingestion fills:      pdf_path, pdf_type, raw_text, page_count, text_per_page
    Preprocessing fills:  doc_type, doc_type_confidence, extracted_fields,
                          missing_fields, confidence_score
    Tool execution fills: tool_results, ndc_code, npi_verified
    AI extraction fills:  ai_feedback, ai_refined_fields, validation_passed
    Synthesis fills:      final_output
    """

    # ── LangGraph message thread ──────────────────────────────────────
    # Required by LangGraph — add_messages reducer appends instead of replacing
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # ── Set once at startup ───────────────────────────────────────────
    pdf_path: str

    # ── Filled by pdf_ingestion_node ──────────────────────────────────
    pdf_type:      PDFType       # native | scanned | unknown
    raw_text:      str           # full extracted text joined across pages
    page_count:    int
    text_per_page: List[str]     # per-page text — used to send first N pages to LLM

    # ── Filled by preprocessing nodes ─────────────────────────────────
    doc_type:            DocType
    doc_type_confidence: float           # 0.0–1.0
    extracted_fields:    Dict[str, Any]  # raw regex extraction result
    missing_fields:      List[str]       # fields expected but not found by regex
    confidence_score:    float           # fields_found / fields_expected

    # ── HIPAA awareness ───────────────────────────────────────────────
    # phi_fields tracks which extracted field names contain
    # Protected Health Information — useful for audit logging
    # and for knowing what NOT to log to console in production
    # PHI fields for prescriptions:
    #   patient_name, dob, address, phone, rx_number
    phi_fields: List[str]

    # ── Filled by tool execution (Day 1 additions) ────────────────────
    # tool_results: keyed by tool name, stores raw API response
    # so synthesis can include verification evidence in final output
    tool_results:  Dict[str, Any]   # {"verify_npi": {...}, "lookup_ndc": {...}}
    ndc_code:      str              # from lookup_ndc tool — uniquely IDs the drug
    npi_verified:  bool             # from verify_npi tool — is prescriber NPI real

    # ── Filled by memory_lookup_node (Day 2 addition) ──────────────────
    # List of hint strings from similar past documents, injected into
    # the AI extraction prompt so the agent learns from past mistakes
    memory_hints: List[str]

    # ── Filled by supervisor + specialist agents (Day 3 addition) ──────
    # specialists_needed: what Supervisor decided to run, e.g.
    #   ["patient_agent", "provider_agent"]  — drug_agent skipped because
    #   all drug fields were already fully found by regex
    specialists_needed: List[str]

    # Each specialist's raw output before merge — kept separate so you
    # can inspect exactly what each agent contributed, same audit
    # trail principle as tool_results in Day 1
    patient_agent_result:  Dict[str, Any]
    provider_agent_result: Dict[str, Any]
    drug_agent_result:     Dict[str, Any]

    # ── Filled by human_review_node (Day 4 addition) ────────────────────
    # Stores whatever the human typed in response to flagged fields,
    # keyed by field name. Empty dict means no corrections were given
    # (either nothing was flagged, or human pressed Enter to skip all)
    human_corrections: Dict[str, str]

    # ── Filled by AI extraction node ──────────────────────────────────
    ai_feedback:       List[str]        # list of issues flagged by LLM
    ai_refined_fields: Dict[str, Any]   # regex fields + AI gap fills merged
    validation_passed: bool

    # ── Tracking ──────────────────────────────────────────────────────
    current_phase:  ProcessingPhase
    error_messages: List[str]

    # ── Final output ──────────────────────────────────────────────────
    final_output: Optional[Dict[str, Any]]