"""
pdf_ingestion_node.py

Detects whether a PDF is native (has a real text layer) or scanned
(image-only pages that need OCR).

Detection strategy
──────────────────
We try to extract text from the first 3 pages using pymupdf.
- If we get a meaningful amount of text  → NATIVE
- If pages are empty or near-empty       → SCANNED

Why this beats checking PDF metadata:
  Metadata can lie. Some scanners embed a text layer with garbage
  characters to fool copy-paste. Probing the actual extracted text
  and measuring its length is far more reliable.

Threshold used:  MIN_CHARS_PER_PAGE = 100
  A real native page almost always has hundreds of characters.
  100 is a conservative lower bound — even a sparse cover page
  with just a date and a name exceeds this.
  A scanned-only page returns 0–10 characters (whitespace, stray
  characters from image noise), well below the threshold.
"""

import fitz  # pymupdf
from langchain_core.messages import AIMessage
from state import MedicalFaxState, PDFType, ProcessingPhase

MIN_CHARS_PER_PAGE = 100   # characters needed per page to call it "native"
PROBE_PAGES        = 3     # how many pages to sample for the detection


def pdf_ingestion_node(state: MedicalFaxState) -> dict:
    """
    Reads the PDF at state['pdf_path'].
    Saves:  pdf_type, raw_text, page_count, text_per_page
    Routes: to native_preprocessing OR scanned_preprocessing
    """
    print("\n=== PDF INGESTION ===")
    pdf_path = state["pdf_path"]
    errors   = list(state.get("error_messages", []))

    try:
        doc        = fitz.open(pdf_path)
        page_count = len(doc)
        print(f"  Opened: {pdf_path}  ({page_count} pages)")

        # ── Extract text from every page ──────────────────────────────
        text_per_page = []
        for page in doc:
            # get_text("text") gives clean plain text with newlines
            # get_text("blocks") gives layout-aware blocks — useful later
            text_per_page.append(page.get_text("text"))

        doc.close()

        # ── Probe first N pages to classify the PDF ────────────────────
        probe_pages   = text_per_page[:PROBE_PAGES]
        chars_by_page = [len(t.strip()) for t in probe_pages]
        avg_chars     = sum(chars_by_page) / max(len(chars_by_page), 1)

        print(f"  Characters per page (first {PROBE_PAGES}): {chars_by_page}")
        print(f"  Average: {avg_chars:.0f} chars  (threshold: {MIN_CHARS_PER_PAGE})")

        if avg_chars >= MIN_CHARS_PER_PAGE:
            pdf_type = PDFType.NATIVE
        else:
            pdf_type = PDFType.SCANNED

        # Combine all pages into a single raw_text string
        # (used by native preprocessing; scanned preprocessing re-does this via OCR)
        raw_text = "\n\n--- PAGE BREAK ---\n\n".join(text_per_page)

        print(f"  Detected: {pdf_type}")

        message = AIMessage(
            content=f"PDF ingestion complete. Type: {pdf_type}. "
                    f"Pages: {page_count}. Avg chars/page: {avg_chars:.0f}."
        )

        return {
            "messages":      [message],
            "pdf_type":      pdf_type,
            "raw_text":      raw_text,
            "page_count":    page_count,
            "text_per_page": text_per_page,
            "current_phase": ProcessingPhase.INGESTION,
            "error_messages": errors,
        }

    except FileNotFoundError:
        msg = f"PDF not found: {pdf_path}"
        print(f"  ERROR: {msg}")
        errors.append(msg)
        return {
            "messages":      [AIMessage(content=f"Error: {msg}")],
            "pdf_type":      PDFType.UNKNOWN,
            "raw_text":      "",
            "page_count":    0,
            "text_per_page": [],
            "current_phase": ProcessingPhase.INGESTION,
            "error_messages": errors,
        }

    except Exception as e:
        msg = f"PDF ingestion failed: {str(e)}"
        print(f"  ERROR: {msg}")
        errors.append(msg)
        return {
            "messages":      [AIMessage(content=f"Error: {msg}")],
            "pdf_type":      PDFType.UNKNOWN,
            "raw_text":      "",
            "page_count":    0,
            "text_per_page": [],
            "current_phase": ProcessingPhase.INGESTION,
            "error_messages": errors,
        }


# ── Conditional edge function ──────────────────────────────────────────────────

def route_after_ingestion(state: MedicalFaxState) -> str:
    """
    Called by LangGraph after pdf_ingestion_node.
    Returns the name of the next node.

    LangGraph uses this string to look up the next node in the graph.
    The dict passed to add_conditional_edges must have matching keys.
    """
    pdf_type = state.get("pdf_type", PDFType.UNKNOWN)

    if pdf_type == PDFType.NATIVE:
        print("  → routing to native_preprocessing")
        return "native_preprocessing"

    elif pdf_type == PDFType.SCANNED:
        print("  → routing to scanned_preprocessing")
        return "scanned_preprocessing"

    else:
        # UNKNOWN means ingestion errored — go to scanned as a safe fallback
        # (OCR on an empty/corrupt page returns empty string gracefully)
        print("  → routing to scanned_preprocessing (fallback for UNKNOWN)")
        return "scanned_preprocessing"