"""
synthesis_node.py

Final node in the pipeline.
Assembles everything in state into a clean final_output dict.

No LLM involved — this is purely mechanical assembly.
It reads whatever is in state and formats it for the consumer.

The final_output dict structure is what a downstream system
(EHR, database, API response) would actually receive.
"""

from langchain_core.messages import AIMessage
from state import MedicalFaxState, ProcessingPhase


def synthesis_node(state: MedicalFaxState) -> dict:
    """
    Assembles final_output from all state fields accumulated so far.
    Always runs — even if AI extraction failed or was skipped.
    """
    print("\n=== SYNTHESIS NODE ===")

    # ── Decide which fields to use ────────────────────────────────────
    # If AI ran:    use ai_refined_fields (regex + AI merged)
    # If AI skipped (high confidence path): use extracted_fields
    ai_refined   = state.get("ai_refined_fields")
    regex_only  = state.get("extracted_fields", {})
    final_fields = ai_refined if ai_refined else regex_only


    # ── Determine overall status ──────────────────────────────────────
    confidence       = state.get("confidence_score", 0.0)
    validation_passed = state.get("validation_passed")
    ai_feedback      = state.get("ai_feedback", [])
    errors           = state.get("error_messages", [])

    if errors:
        status = "ERROR"
    elif ai_refined:                          # AI actually ran and returned data
        status = "VALIDATED" if validation_passed else "NEEDS_REVIEW"
    else:
        status = "AUTO_EXTRACTED"             # AI was skipped — high confidence regex
    # ── Build final output ────────────────────────────────────────────
    final_output = {
        # Processing metadata
        "status":              status,
        "pdf_type":            str(state.get("pdf_type", "unknown")),
        "doc_type":            str(state.get("doc_type", "unknown")),
        "doc_type_confidence": state.get("doc_type_confidence", 0.0),
        "field_confidence":    confidence,
        "page_count":          state.get("page_count", 0),

        # The actual extracted data
        "fields":              final_fields,

        # Quality signals for downstream consumer
        "missing_fields":      state.get("missing_fields", []),
        "ai_feedback":         ai_feedback,
        "validation_passed":   validation_passed,

        # Traceability — raw regex output preserved separately
        "regex_extracted":     regex_only,
    }

    # ── Print summary ─────────────────────────────────────────────────
    print(f"  Status           : {status}")
    print(f"  Doc type         : {final_output['doc_type']} "
          f"(confidence: {final_output['doc_type_confidence']:.2f})")
    print(f"  Field confidence : {confidence:.2f}")
    print(f"  Fields in output : {list(final_fields.keys())}")
    if ai_feedback:
        print(f"  AI feedback      :")
        for fb in ai_feedback:
            print(f"    • {fb}")

    message_text = (
        f"Synthesis complete. Status: {status}. "
        f"Doc type: {final_output['doc_type']}. "
        f"Total fields: {len(final_fields)}."
    )

    return {
        "messages":      [AIMessage(content=message_text)],
        "final_output":  final_output,
        "current_phase": ProcessingPhase.SYNTHESIS,
    }