"""
provider_agent_node.py

Specialist agent — owns prescriber_name, prescriber_npi, dea_number,
pharmacy_name, pharmacy_npi.

Tools: verify_npi, validate_field_value

This is the agent that does the NPI name cross-check from Day 2 —
but now it's the ONLY thing this agent thinks about. No drug lookups,
no patient name distractions in the same context window.

Runs SECOND in the sequence, after Patient Agent.
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from state import MedicalFaxState, ProcessingPhase
from tools import verify_npi, validate_field_value

MAX_TEXT_CHARS = 2000
PAGES_TO_SEND  = 2
MAX_ITERATIONS = 6

TOOL_REGISTRY = {
    "verify_npi":           verify_npi,
    "validate_field_value": validate_field_value,
}

SYSTEM_PROMPT = """You are a specialist focused ONLY on prescriber and pharmacy
information from a US medical fax document.

Your only job: find/verify prescriber_name, prescriber_npi, dea_number,
pharmacy_name, pharmacy_npi. Do not look for patient or drug information —
other specialists handle those.

TOOLS AVAILABLE:
  verify_npi(npi_number)
    → Call this for any prescriber_npi or pharmacy_npi you see or find
    → entity_type tells you Individual (prescriber) vs Organization (pharmacy)

  validate_field_value(field_name, value)
    → Call this on dea_number to confirm format
    → Call this on any suspicious value

CRITICAL RULE — NPI name matching:
  After verify_npi returns a valid result for prescriber_npi, you MUST
  check whether provider_name matches the prescriber_name you found.
  If they do NOT refer to the same person, flag it as suspicious —
  do not accept an NPI that belongs to someone else.

Respond ONLY with valid JSON, no markdown fences, no preamble, after
you finish all tool calls:
{
  "found_fields": {
    "field_name": "value you found for a missing field"
  },
  "suspicious_fields": {
    "field_name": "reason this value looks wrong"
  },
  "reasoning": "one sentence summary"
}

Never invent values not present in the document text."""


def _execute_tool(tool_call: dict) -> str:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_fn   = TOOL_REGISTRY.get(tool_name)
    if not tool_fn:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        return json.dumps(tool_fn.invoke(tool_args))
    except Exception as e:
        return json.dumps({"error": str(e)})


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
            "found_fields":      parsed.get("found_fields", {}),
            "suspicious_fields": parsed.get("suspicious_fields", {}),
            "reasoning":         parsed.get("reasoning", ""),
        }
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse failed: {e}")
        return {"found_fields": {}, "suspicious_fields": {}, "reasoning": f"parse error: {e}"}


def provider_agent_node(state: MedicalFaxState) -> dict:
    """
    Extracts and verifies prescriber/pharmacy fields.
    Saves result + raw tool results to provider_agent_result in state.
    """
    print("\n=== PROVIDER AGENT ===")

    provider_fields = {"prescriber_name", "prescriber_npi", "dea_number",
                        "pharmacy_name", "pharmacy_npi"}
    missing       = [f for f in state.get("missing_fields", []) if f in provider_fields]
    extracted     = {k: v for k, v in state.get("extracted_fields", {}).items()
                      if k in provider_fields}
    text_per_page = state.get("text_per_page", [])
    raw_text      = state.get("raw_text", "")

    context_text = (
        "\n\n--- PAGE BREAK ---\n\n".join(text_per_page[:PAGES_TO_SEND])
        if text_per_page else raw_text
    )
    if len(context_text) > MAX_TEXT_CHARS:
        context_text = context_text[:MAX_TEXT_CHARS] + "\n... [truncated]"

    print(f"  Already extracted: {extracted}")
    print(f"  Missing fields:    {missing}")

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=800).bind_tools(
            list(TOOL_REGISTRY.values())
        )

        extracted_str = "\n".join(f"  {k}: {v}" for k, v in extracted.items()) or "  (none)"
        missing_str   = "\n".join(f"  - {f}" for f in missing) or "  (none)"

        user_prompt = f"""Already extracted — verify, do not re-extract:
{extracted_str}

Missing fields to find:
{missing_str}

Raw document text:
{context_text}"""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        tool_results = {}
        iteration = 0

        while iteration < MAX_ITERATIONS:
            iteration += 1
            response = llm.invoke(messages)
            tool_calls = getattr(response, "tool_calls", []) or []

            if not tool_calls:
                print(f"  Finished after {iteration} iteration(s)")
                break

            messages.append(response)
            for tool_call in tool_calls:
                print(f"    → {tool_call['name']}({tool_call['args']})")
                result_str = _execute_tool(tool_call)
                tool_results[tool_call["name"]] = json.loads(result_str)
                print(f"    ← {result_str[:150]}")
                messages.append(ToolMessage(content=result_str, tool_call_id=tool_call["id"]))

        parsed = _parse_response(response.content)
        parsed["tool_results"] = tool_results

        print(f"  found_fields:      {parsed['found_fields']}")
        print(f"  suspicious_fields: {parsed['suspicious_fields']}")

        return {
            "provider_agent_result": parsed,
            "current_phase":         ProcessingPhase.AI_EXTRACTION,
        }

    except Exception as e:
        print(f"  ERROR: Provider Agent failed: {e}")
        return {
            "provider_agent_result": {
                "found_fields": {}, "suspicious_fields": {},
                "reasoning": f"error: {e}", "tool_results": {}
            },
            "current_phase": ProcessingPhase.AI_EXTRACTION,
        }