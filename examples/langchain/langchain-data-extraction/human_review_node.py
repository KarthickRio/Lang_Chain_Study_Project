"""
human_review_node.py

Pauses the graph using LangGraph's interrupt() mechanism when
merge_specialists flagged any field as suspicious.

Why interrupt() instead of just printing and continuing
───────────────────────────────────────────────────────────
Printing a warning and moving on means nobody actually looks at it —
that's exactly what happens today (NEEDS_REVIEW sits unread in JSON).
interrupt() genuinely PAUSES the graph. main.py's invoke() call
returns control to the terminal right here, waits for real human
input, then resumes the graph with that input available.

One combined interrupt, not one per field
────────────────────────────────────────────
Per your decision: if multiple fields are flagged, show all of them
in a single interrupt and let the human answer for all at once,
rather than pausing once per field.

What "no answer" means
────────────────────────
If the human just presses Enter for a field, we keep the AI's
flagged value as-is — pressing Enter means "I looked, leave it
flagged for now," not "delete this field."
"""

from langgraph.types import interrupt
from langchain_core.messages import AIMessage
from state import MedicalFaxState, ProcessingPhase


def human_review_node(state: MedicalFaxState) -> dict:
    """
    Checks ai_feedback for SUSPICIOUS entries.
    If any exist, pauses the graph and asks for human corrections.
    If none exist, passes through immediately — no pause.
    """
    print("\n=== HUMAN REVIEW NODE ===")

    ai_feedback       = state.get("ai_feedback", [])
    ai_refined_fields = state.get("ai_refined_fields", {})

    # Extract suspicious field names + reasons from ai_feedback
    # Format produced by merge_specialists_node: "SUSPICIOUS — field: reason"
    suspicious = {}
    for fb in ai_feedback:
        if fb.startswith("SUSPICIOUS"):
            try:
                rest = fb.split("—", 1)[1].strip()
                field, reason = rest.split(":", 1)
                suspicious[field.strip()] = reason.strip()
            except (IndexError, ValueError):
                continue

    if not suspicious:
        print("  No suspicious fields — skipping human review, continuing")
        return {
            "human_corrections": {},
            "current_phase":     ProcessingPhase.AI_EXTRACTION,
        }

    print(f"  {len(suspicious)} field(s) need human review — pausing graph")

    # ── Build one combined question covering all flagged fields ────────
    # This is what interrupt() sends back to main.py — main.py displays
    # it and collects input(), then resumes with that as the payload.
    question_lines = ["The following fields were flagged and need your review:\n"]
    for field, reason in suspicious.items():
        current_value = ai_refined_fields.get(field, "(no value)")
        question_lines.append(f"  Field:   {field}")
        question_lines.append(f"  Current: {current_value}")
        question_lines.append(f"  Issue:   {reason}")
        question_lines.append("")

    question_text = "\n".join(question_lines)

    # ── This is the actual pause ────────────────────────────────────────
    # Execution stops here. main.py's agent.invoke() call returns with
    # an __interrupt__ key containing this payload. When main.py resumes
    # with Command(resume=human_input), human_input becomes the return
    # value of this exact interrupt() call — code continues right here.
    human_input = interrupt({
        "question":            question_text,
        "suspicious_fields":   suspicious,
    })

    # ── human_input is expected to be a dict: {field_name: correction} ──
    # main.py is responsible for collecting one correction per field
    # and packaging them this way before resuming
    corrections = human_input if isinstance(human_input, dict) else {}

    # ── Apply corrections to ai_refined_fields ──────────────────────────
    updated_fields = dict(ai_refined_fields)
    for field, correction in corrections.items():
        if correction:   # non-empty string — human provided a real value
            updated_fields[field] = correction
            print(f"  ✏️  {field} corrected to: {correction}")
        else:
            print(f"  ⏭️  {field} left as flagged (no correction given)")

    return {
        "ai_refined_fields":  updated_fields,
        "human_corrections":  corrections,
        "current_phase":      ProcessingPhase.AI_EXTRACTION,
    }