"""
native_preprocessing_node.py

Preprocessing for native (text-layer) PDFs.

What happens here
──────────────────
1. raw_text is already in state (put there by pdf_ingestion_node)
2. We run doc_type detection via weighted keyword scoring
3. We run field extraction via regex — results saved to state
4. We compute confidence and populate missing_fields
5. Everything goes into state — nothing is lost

No LLM involved here. This is deterministic, fast, and cheap.
"""

from langchain_core.messages import AIMessage
from state import MedicalFaxState, ProcessingPhase
from doc_type_detector import detect_doc_type
from field_extractor import extract_fields


def native_preprocessing_node(state: MedicalFaxState) -> dict:
    """
    Process a native PDF through keyword detection + regex field extraction.
    """
    print("\n=== NATIVE PREPROCESSING ===")

    raw_text = state.get("raw_text", "")
    errors   = list(state.get("error_messages", []))

    if not raw_text.strip():
        msg = "Native preprocessing: raw_text is empty"
        errors.append(msg)
        return {
            "messages":          [AIMessage(content=msg)],
            "doc_type":          "unknown",
            "doc_type_confidence": 0.0,
            "extracted_fields":  {},
            "missing_fields":    [],
            "confidence_score":  0.0,
            "current_phase":     ProcessingPhase.NATIVE_PREPROCESSING,
            "error_messages":    errors,
        }

    # Step 1 — detect document type
    print("  Step 1: document type detection")
    doc_type, type_confidence, scores_detail = detect_doc_type(raw_text)

    # Step 2 — extract fields based on detected type
    print(f"  Step 2: field extraction for doc_type={doc_type}")
    extracted_fields, missing_fields, field_confidence, phi_fields_found = extract_fields(raw_text, doc_type)

    # Step 3 — save everything to state
    message_text = (
        f"Native preprocessing complete. "
        f"Doc type: {doc_type} (confidence: {type_confidence:.2f}). "
        f"Fields found: {len(extracted_fields)}. "
        f"Missing: {missing_fields}. "
        f"Field confidence: {field_confidence:.2f}."
    )
    print(f"  {message_text}")

    return {
        "messages":            [AIMessage(content=message_text)],
        "doc_type":            doc_type,
        "doc_type_confidence": type_confidence,
        "extracted_fields":    extracted_fields,
        "missing_fields":      missing_fields,
        "confidence_score":    field_confidence,
        "phi_fields":          phi_fields_found,
        "current_phase":       ProcessingPhase.NATIVE_PREPROCESSING,
        "error_messages":      errors,
    }