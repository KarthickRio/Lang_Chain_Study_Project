"""
memory_save_node.py

Runs AFTER synthesis_node.
Saves the final result summary into memory for future lookups.

Why this runs after synthesis, not during
────────────────────────────────────────────
By the time synthesis completes, we have the FINAL truth about this
document — what was missing, what was flagged, whether validation
passed. Saving here means memory always reflects the final, settled
outcome, not an intermediate state.
"""

from langchain_core.messages import AIMessage
from state import MedicalFaxState
from memory_store import save_document_memory


def memory_save_node(state: MedicalFaxState) -> dict:
    """
    Saves this document's outcome into memory.
    Reads from final_output (set by synthesis_node).
    """
    print("\n=== MEMORY SAVE NODE ===")

    final_output = state.get("final_output") or {}

    doc_type           = final_output.get("doc_type", "unknown")
    missing_fields     = final_output.get("missing_fields", [])
    validation_passed  = final_output.get("validation_passed", None)
    final_fields        = final_output.get("fields", {})

    # Suspicious fields aren't in final_output directly — they're
    # embedded in ai_feedback strings. Extract them back into a dict
    # for clean storage.
    suspicious_fields = {}
    for fb in final_output.get("ai_feedback", []):
        if fb.startswith("SUSPICIOUS"):
            # Format: "SUSPICIOUS — field_name: reason"
            try:
                rest = fb.split("—", 1)[1].strip()
                field, reason = rest.split(":", 1)
                suspicious_fields[field.strip()] = reason.strip()
            except (IndexError, ValueError):
                continue

    record_id = save_document_memory(
        doc_type=doc_type,
        missing_fields=missing_fields,
        suspicious_fields=suspicious_fields,
        validation_passed=validation_passed,
        final_fields=final_fields,
    )

    message_text = f"Saved document outcome to memory (id: {record_id[:8]}...)"
    print(f"  {message_text}")

    return {
        "messages": [AIMessage(content=message_text)],
    }