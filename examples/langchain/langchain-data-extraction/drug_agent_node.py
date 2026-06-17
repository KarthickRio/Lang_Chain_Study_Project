"""
drug_agent_node.py

Specialist agent — owns drug_prescribed, strength, sig, qty_prescribed.

Tools: lookup_ndc, validate_field_value

This agent's prompt is narrow on purpose — only thinking about drug
identity and dosage, separate from the patient/provider context that
caused noise in the single-agent Day 2 version.

Runs THIRD — last in the sequence.
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from state import MedicalFaxState, ProcessingPhase
from tools import lookup_ndc, validate_field_value

MAX_TEXT_CHARS = 2000
PAGES_TO_SEND  = 2
MAX_ITERATIONS = 6

TOOL_REGISTRY = {
    "lookup_ndc":            lookup_ndc,
    "validate_field_value":  validate_field_value,
}

SYSTEM_PROMPT = """You are a specialist focused ONLY on drug/medication information
from a US medical fax document.

Your only job: find/verify drug_prescribed, strength, sig, qty_prescribed.
Do not look for patient or prescriber information — other specialists
handle those.

TOOLS AVAILABLE:
  lookup_ndc(drug_name)
    → Call this for the drug_prescribed value
    → Returns FDA standard name and NDC code

  validate_field_value(field_name, value)
    → Call this on strength and qty_prescribed to confirm format

CRITICAL RULE — drug name sanity check:
  After lookup_ndc returns, compare standard_name against the drug
  name you extracted. If they describe clearly different drugs
  (not just a spelling/capitalization difference), flag drug_prescribed
  as suspicious — do not blindly trust the API's match. Use your own
  knowledge of common drug names to judge if the match looks wrong.

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


def drug_agent_node(state: MedicalFaxState) -> dict:
    """
    Extracts and verifies drug fields.
    Saves result + raw tool results to drug_agent_result in state.
    """
    print("\n=== DRUG AGENT ===")

    drug_fields   = {"drug_prescribed", "strength", "sig", "qty_prescribed"}
    missing       = [f for f in state.get("missing_fields", []) if f in drug_fields]
    extracted     = {k: v for k, v in state.get("extracted_fields", {}).items()
                      if k in drug_fields}
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
            "drug_agent_result": parsed,
            "current_phase":     ProcessingPhase.AI_EXTRACTION,
        }

    except Exception as e:
        print(f"  ERROR: Drug Agent failed: {e}")
        return {
            "drug_agent_result": {
                "found_fields": {}, "suspicious_fields": {},
                "reasoning": f"error: {e}", "tool_results": {}
            },
            "current_phase": ProcessingPhase.AI_EXTRACTION,
        }