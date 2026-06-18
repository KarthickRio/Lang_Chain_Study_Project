"""
main.py

Entry point. Pass a PDF path as a command-line argument.

Usage:
    python main.py path/to/document.pdf

Day 4 change — human-in-the-loop
───────────────────────────────────
The agent can now PAUSE mid-run if human_review_node flags
suspicious fields. main.py must:
  1. Call agent.invoke() with a config containing a thread_id
     (LangGraph uses this to track which paused run you're resuming)
  2. Check if the result contains "__interrupt__" — if so, the
     graph is paused, not finished
  3. Show the question to the user, collect their corrections
  4. Resume with Command(resume=corrections)
  5. Repeat until the graph actually reaches END (no more interrupts)

Windows PowerShell:
    $env:OPENAI_API_KEY = "sk-..."
    python main.py "D:\\path\\to\\your.pdf"
"""

import sys
import json
import uuid
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from state import MedicalFaxState, PDFType, ProcessingPhase, DocType
from agent import create_agent


def _collect_human_corrections(interrupt_payload: dict) -> dict:
    """
    Displays the combined question to the user and collects one
    correction per flagged field. Pressing Enter on a field means
    "leave it flagged as-is" — not a blank value.
    """
    print(f"\n{'─'*55}")
    print("  ⏸  HUMAN REVIEW NEEDED — graph is paused")
    print(f"{'─'*55}")
    print(interrupt_payload.get("question", ""))

    suspicious_fields = interrupt_payload.get("suspicious_fields", {})
    corrections = {}

    for field in suspicious_fields:
        answer = input(f"  Correct value for '{field}' (Enter to keep flagged): ").strip()
        corrections[field] = answer

    print(f"{'─'*55}\n")
    return corrections


def run(pdf_path: str):
    agent = create_agent()

    # ── thread_id ties this run to its checkpointed state ──────────────
    # Required by the checkpointer so LangGraph knows which paused
    # execution to resume when we call invoke() again.
    thread_id = str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages":            [HumanMessage(content=f"Process PDF: {pdf_path}")],
        "pdf_path":            pdf_path,
        "pdf_type":            PDFType.UNKNOWN,
        "raw_text":            "",
        "page_count":          0,
        "text_per_page":       [],
        "doc_type":            DocType.UNKNOWN,
        "doc_type_confidence": 0.0,
        "extracted_fields":    {},
        "missing_fields":      [],
        "confidence_score":    0.0,
        "ai_feedback":         [],
        "ai_refined_fields":   {},
        "validation_passed":   False,
        "current_phase":       ProcessingPhase.INGESTION,
        "phi_fields":          [],
        "memory_hints":        [],
        "specialists_needed":   [],
        "patient_agent_result":  {},
        "provider_agent_result": {},
        "drug_agent_result":     {},
        "human_corrections":   {},
        "error_messages":      [],
        "final_output":        None,
    }

    print(f"\n{'='*55}")
    print(f"  MEDICAL FAX AGENT")
    print(f"  PDF: {pdf_path}")
    print(f"{'='*55}")

    # ── First call — starts the graph ───────────────────────────────────
    result = agent.invoke(initial_state, config)

    # ── Resume loop — keeps going until graph reaches END ───────────────
    # A single document could theoretically pause more than once if the
    # graph design allowed it; this loop handles that generally even
    # though our current design only pauses once at human_review_node.
    while "__interrupt__" in result:
        interrupt_obj     = result["__interrupt__"][0]
        interrupt_payload = interrupt_obj.value

        corrections = _collect_human_corrections(interrupt_payload)

        # Resume the graph — human_input inside human_review_node
        # becomes this corrections dict
        result = agent.invoke(Command(resume=corrections), config)

    final_state = result

    # ── Print final output ────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  FINAL OUTPUT")
    print(f"{'='*55}")
    output = final_state.get("final_output")
    if output:
        print(json.dumps(output, indent=2, default=str))
    else:
        print("  No final output — check error_messages:")
        print(f"  {final_state.get('error_messages')}")

    return final_state


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        print('Example: python main.py "D:\\docs\\fax.pdf"')
        sys.exit(1)

    run(sys.argv[1])