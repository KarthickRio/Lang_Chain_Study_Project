"""
scanned_preprocessing_node.py

Preprocessing for scanned (image-only) PDFs.

What happens here
──────────────────
1. Each PDF page is rendered to a PIL image using pymupdf
2. Tesseract OCR extracts text from each image
3. Pages are joined into a single text string — saved as raw_text
4. Then identical doc_type detection + field extraction as native path
5. Confidence is naturally lower for scanned docs — that's correct

Why Tesseract
─────────────
Tesseract is the industry-standard open-source OCR engine.
It handles printed text on clean backgrounds well.
For medical faxes (which are typically printed then faxed),
accuracy is usually 85–95% with default settings.

Requirements
────────────
  pip install pytesseract pillow pymupdf

  Tesseract binary must be installed at OS level:
  Ubuntu/Debian: sudo apt install tesseract-ocr
  macOS:         brew install tesseract
  Windows:       https://github.com/UB-Mannheim/tesseract/wiki

  pytesseract docs: https://github.com/madmaze/pytesseract

DPI note
────────
We render pages at 200 DPI. This is a balance:
  - 72 DPI (screen): too low, OCR misses characters
  - 300 DPI: better accuracy but 3× larger images, slower
  - 200 DPI: good accuracy for standard fax quality
For very poor quality scans, increase to 300.
"""

import io
import fitz          # pymupdf — for rendering pages to images
import pytesseract
from PIL import Image
from langchain_core.messages import AIMessage
from state import MedicalFaxState, ProcessingPhase
from doc_type_detector import detect_doc_type
from field_extractor import extract_fields

RENDER_DPI = 200


def _page_to_image(page: fitz.Page) -> Image.Image:
    """
    Render a single pymupdf page to a PIL Image.

    fitz.Matrix scales the page. 200 DPI = 200/72 ≈ 2.78× zoom.
    get_pixmap() rasterises the page at that scale.
    tobytes("png") converts to PNG bytes for PIL to open.
    """
    zoom   = RENDER_DPI / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix    = page.get_pixmap(matrix=matrix, alpha=False)
    img    = Image.open(io.BytesIO(pix.tobytes("png")))
    return img


def _ocr_image(image: Image.Image, page_num: int) -> str:
    """
    Run Tesseract OCR on a PIL image.

    pytesseract.image_to_string() calls the Tesseract binary
    and returns a string of recognised text.

    config explanation:
      --psm 3   : automatic page segmentation (default, good for faxes)
      --oem 1   : use LSTM neural net engine (more accurate than legacy)
    """
    try:
        text = pytesseract.image_to_string(
            image,
            config="--psm 3 --oem 1"
        )
        char_count = len(text.strip())
        print(f"  Page {page_num + 1}: OCR extracted {char_count} chars")
        return text
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract binary not found. "
            "Install with: sudo apt install tesseract-ocr  (Linux) "
            "or brew install tesseract  (macOS)"
        )
    except Exception as e:
        print(f"  Page {page_num + 1}: OCR failed — {e}")
        return ""


def scanned_preprocessing_node(state: MedicalFaxState) -> dict:
    """
    Process a scanned PDF: render each page → OCR → detect + extract.
    """
    print("\n=== SCANNED PREPROCESSING ===")

    pdf_path = state["pdf_path"]
    errors   = list(state.get("error_messages", []))

    try:
        # Step 1 — render each page and OCR it
        print(f"  Step 1: rendering pages at {RENDER_DPI} DPI and running OCR")
        doc           = fitz.open(pdf_path)
        text_per_page = []

        for i, page in enumerate(doc):
            image    = _page_to_image(page)
            ocr_text = _ocr_image(image, i)
            text_per_page.append(ocr_text)

        doc.close()

        raw_text = "\n\n--- PAGE BREAK ---\n\n".join(text_per_page)

        # Step 2 — same detection + extraction as native path
        print("  Step 2: document type detection")
        doc_type, type_confidence, _ = detect_doc_type(raw_text)

        print(f"  Step 3: field extraction for doc_type={doc_type}")
        extracted_fields, missing_fields, field_confidence = extract_fields(raw_text, doc_type)

        message_text = (
            f"Scanned preprocessing complete. "
            f"Doc type: {doc_type} (confidence: {type_confidence:.2f}). "
            f"Fields found: {len(extracted_fields)}. "
            f"Missing: {missing_fields}. "
            f"Field confidence: {field_confidence:.2f}."
        )
        print(f"  {message_text}")

        return {
            "messages":            [AIMessage(content=message_text)],
            "raw_text":            raw_text,       # overwrite — OCR is the real text
            "text_per_page":       text_per_page,
            "doc_type":            doc_type,
            "doc_type_confidence": type_confidence,
            "extracted_fields":    extracted_fields,
            "missing_fields":      missing_fields,
            "confidence_score":    field_confidence,
            "current_phase":       ProcessingPhase.SCANNED_PREPROCESSING,
            "error_messages":      errors,
        }

    except Exception as e:
        msg = f"Scanned preprocessing failed: {str(e)}"
        print(f"  ERROR: {msg}")
        errors.append(msg)
        return {
            "messages":            [AIMessage(content=msg)],
            "raw_text":            state.get("raw_text", ""),
            "text_per_page":       state.get("text_per_page", []),
            "doc_type":            "unknown",
            "doc_type_confidence": 0.0,
            "extracted_fields":    {},
            "missing_fields":      [],
            "confidence_score":    0.0,
            "current_phase":       ProcessingPhase.SCANNED_PREPROCESSING,
            "error_messages":      errors,
        }


# ── Conditional edge ────────────────────────────────────────────────────────

def route_after_preprocessing(state: MedicalFaxState) -> str:
    """
    After either preprocessing node, decide whether AI is needed.

    > 0.80  → high confidence, skip AI, go straight to synthesis
    ≤ 0.80  → send to AI extraction for gap filling and validation
    """
    confidence = state.get("confidence_score", 0.0)

    if confidence > 0.80:
        print(f"  → confidence {confidence:.2f} > 0.80, skipping AI")
        return "synthesis"
    else:
        print(f"  → confidence {confidence:.2f} ≤ 0.80, sending to AI extraction")
        return "ai_extraction"