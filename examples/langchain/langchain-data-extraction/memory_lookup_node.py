"""
memory_lookup_node.py

Runs BEFORE ai_extraction_node.
Searches memory for similar past documents and builds hint strings
that get injected into the AI extraction prompt.

Why this runs before AI extraction, not after
────────────────────────────────────────────────
The point of memory is to give the agent a head start — if a similar
document caused an NPI mismatch last time, the agent should watch out
for that BEFORE it makes the same mistake again, not learn about it
after the fact.
"""

from langchain_core.messages import AIMessage
from state import MedicalFaxState, ProcessingPhase
from memory_store import search_similar_documents


def memory_lookup_node(state: MedicalFaxState) -> dict:
    """
    Searches memory for documents similar to the current one.
    Saves memory_hints to state — a list of short hint strings.
    """
    print("\n=== MEMORY LOOKUP NODE ===")

    doc_type       = state.get("doc_type", "unknown")
    missing_fields = state.get("missing_fields", [])

    matches = search_similar_documents(
        doc_type=str(doc_type),
        missing_fields=missing_fields,
        n_results=2,
    )

    hints = []
    for match in matches:
        # Only surface hints from reasonably similar documents
        # distance is 0 = identical, higher = less similar
        # ChromaDB default embedding distances above ~1.5 are weak matches
        if match["distance"] > 1.5:
            continue

        if match["suspicious_fields"]:
            for field, reason in match["suspicious_fields"].items():
                hints.append(
                    f"A similar past {match['doc_type']} document had an issue "
                    f"with '{field}': {reason}. Watch for this in the current document."
                )

    if not hints:
        print("  No actionable hints from memory")
    else:
        print(f"  Generated {len(hints)} hint(s) for AI extraction:")
        for h in hints:
            print(f"    • {h}")

    message_text = f"Memory lookup complete. {len(hints)} hint(s) found."

    return {
        "messages":      [AIMessage(content=message_text)],
        "memory_hints":  hints,
        "current_phase": ProcessingPhase.AI_EXTRACTION,  # heading into AI next
    }