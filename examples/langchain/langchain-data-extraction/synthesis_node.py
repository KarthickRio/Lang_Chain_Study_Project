"""
synthesis_node.py

Final node — assembles everything in state into clean final_output.
No LLM involved. Pure mechanical assembly.

Fixes from previous version
─────────────────────────────
Bug 1 — fields: {} in output
  Cause: empty dict {} is not None so "ai_refined if ai_refined is not None"
         always picked the empty dict over regex_extracted
  Fix:   "if ai_refined" — empty dict is falsy, correctly falls back

Bug 2 — status NEEDS_REVIEW on AI-skipped path
  Cause: validation_passed initialised as False in main.py,
         never updated when AI was skipped (high confidence path)
  Fix:   status logic now checks whether ai_refined actually has data
         Empty dict = AI was skipped → AUTO_EXTRACTED
         Populated dict = AI ran → VALIDATED or NEEDS_REVIEW
"""

from langchain_core.messages import AIMessage
from state import MedicalFaxState, ProcessingPhase


def synthesis_node(state: MedicalFaxState) -> dict:
    """
    Assembles final_output from all accumulated state fields.
    Always runs — even if AI was skipped or errored.
    """
    print("\n=== SYNTHESIS NODE ===")

    regex_fields  = state.get("extracted_fields",    {})
    ai_refined    = state.get("ai_refined_fields",   {})
    ai_feedback   = state.get("ai_feedback",         [])
    tool_results  = state.get("tool_results",        {})
    errors        = state.get("error_messages",      [])
    confidence    = state.get("confidence_score",    0.0)
    validation    = state.get("validation_passed",   None)
    phi_fields    = state.get("phi_fields",          [])

    # ── Bug 1 fix: select final fields ────────────────────────────────
    # ai_refined is only populated when AI node actually ran and found fields
    # Empty dict {} means AI was skipped (high confidence path)
    # Use "if ai_refined" not "if ai_refined is not None"
    final_fields = ai_refined if ai_refined else regex_fields

    # ── Bug 2 fix: correct status logic ───────────────────────────────
    # ai_refined being populated means AI node ran
    # validation_passed tells us whether AI was satisfied
    if errors:
        status = "ERROR"
    elif ai_refined:
        # AI ran — check if it was satisfied
        status = "VALIDATED" if validation else "NEEDS_REVIEW"
    else:
        # AI was skipped — high confidence regex extraction
        status = "AUTO_EXTRACTED"

    # ── Assemble final output ──────────────────────────────────────────
    final_output = {
        # Processing metadata
        "status":              status,
        "pdf_type":            str(state.get("pdf_type",            "unknown")),
        "doc_type":            str(state.get("doc_type",            "unknown")),
        "doc_type_confidence": state.get("doc_type_confidence",     0.0),
        "field_confidence":    confidence,
        "page_count":          state.get("page_count",              0),

        # The extracted data — regex + AI merged
        "fields":              final_fields,

        # Quality signals
        "missing_fields":      state.get("missing_fields",          []),
        "ai_feedback":         ai_feedback,
        "validation_passed":   validation,

        # HIPAA tracking
        "phi_fields_present":  phi_fields,

        # Tool verification results
        "tool_results":        tool_results,
        "npi_verified":        state.get("npi_verified",            False),
        "ndc_code":            state.get("ndc_code",                ""),

        # Traceability — raw regex output preserved
        "regex_extracted":     regex_fields,
    }

    print(f"  Status           : {status}")
    print(f"  Doc type         : {final_output['doc_type']} "
          f"(confidence: {final_output['doc_type_confidence']:.2f})")
    print(f"  Field confidence : {confidence:.2f}")
    print(f"  Fields in output : {list(final_fields.keys())}")
    print(f"  PHI fields       : {phi_fields}")
    if ai_feedback:
        print(f"  AI feedback:")
        for fb in ai_feedback:
            print(f"    • {fb}")
    if tool_results:
        print(f"  Tool results     : {list(tool_results.keys())}")

    return {
        "messages":      [AIMessage(content=f"Synthesis complete. Status: {status}.")],
        "final_output":  final_output,
        "current_phase": ProcessingPhase.SYNTHESIS,
    }