"""
patient_agent_node.py

Specialist agent — owns ONLY patient_name and dob.
No tools needed. Pure text extraction reasoning.

Why separate from Provider/Drug agents
─────────────────────────────────────────
Patient demographics extraction is a completely different mental task
than verifying an NPI or looking up a drug code. Keeping this agent's
prompt narrow means it never has to think about anything except
"find the patient's name and birth date in this text" — less surface
area for the LLM to get distracted or confused.

Runs FIRST in the sequence (Patient → Provider → Drug)
so its result is available to the Provider Agent for context,
even though Provider Agent's own NPI cross-check uses its own
extracted prescriber_name, not the patient name.
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from state import MedicalFaxState, ProcessingPhase

MAX_TEXT_CHARS = 2000
PAGES_TO_SEND  = 2

SYSTEM_PROMPT = """You are a specialist focused ONLY on extracting patient demographic
information from a US medical fax document.

Your only job: find patient_name and dob (date of birth).
Do not look for any other field — prescriber, drug, pharmacy info is
handled by other specialists.

Respond ONLY with valid JSON, no markdown fences, no preamble:
{
  "found_fields": {
    "patient_name": "value if found",
    "dob": "value if found, format YYYY-MM-DD"
  },
  "reasoning": "one sentence: what you found and where"
}

Only include a field in found_fields if you are confident it is in the text.
Never invent a value not present in the document."""


def _parse_response(content: str) -> dict:
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*",     "", content)
    content = content.strip()
    start = content.find("{")
    end   = content.rfind("}") + 1
    if start != -1 and end > start:
        content = content[start:end]
    try:
        parsed = json.loads(content)
        return {
            "found_fields": parsed.get("found_fields", {}),
            "reasoning":    parsed.get("reasoning", ""),
        }
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse failed: {e}")
        return {"found_fields": {}, "reasoning": f"parse error: {e}"}


def patient_agent_node(state: MedicalFaxState) -> dict:
    """
    Extracts patient_name and dob only.
    Saves result to patient_agent_result in state.
    """
    print("\n=== PATIENT AGENT ===")

    missing       = [f for f in state.get("missing_fields", [])
                      if f in ("patient_name", "dob")]
    text_per_page = state.get("text_per_page", [])
    raw_text      = state.get("raw_text", "")

    context_text = (
        "\n\n--- PAGE BREAK ---\n\n".join(text_per_page[:PAGES_TO_SEND])
        if text_per_page else raw_text
    )
    if len(context_text) > MAX_TEXT_CHARS:
        context_text = context_text[:MAX_TEXT_CHARS] + "\n... [truncated]"

    print(f"  Missing fields to find: {missing}")

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=300)

        user_prompt = f"""Missing fields to find: {missing}

Raw document text:
{context_text}"""

        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        parsed = _parse_response(response.content)
        print(f"  found_fields: {parsed['found_fields']}")
        print(f"  reasoning:    {parsed['reasoning']}")

        return {
            "patient_agent_result": parsed,
            "current_phase":        ProcessingPhase.AI_EXTRACTION,
        }

    except Exception as e:
        print(f"  ERROR: Patient Agent failed: {e}")
        return {
            "patient_agent_result": {"found_fields": {}, "reasoning": f"error: {e}"},
            "current_phase":        ProcessingPhase.AI_EXTRACTION,
        }