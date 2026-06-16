"""
ai_extraction_node.py

AI extraction with autonomous tool calling — Day 1 upgrade.

What changed from previous version
────────────────────────────────────
OLD: llm.invoke(messages) once → parse JSON from response
     LLM is passive — one shot, no ability to try again

NEW: ReAct loop — Reason → Act (call tool) → Observe result → Reason again
     LLM decides which tools to call, in what order, how many times
     Loop runs until LLM stops making tool calls
     Only then do we ask for the final structured JSON

Why ReAct loop instead of LangChain AgentExecutor:
  AgentExecutor is a black box — you cannot see or control
  what happens between steps. The manual loop here is 10 more lines
  but gives full visibility into every decision the LLM makes.
  For learning, explicit > implicit.

Flow inside this node:
  1. LLM receives: system prompt + extracted fields + missing fields + raw text
  2. LLM reasons and calls tools (verify_npi, lookup_ndc, validate_field_value)
  3. Each tool result is fed back to LLM as a ToolMessage
  4. LLM reasons again — may call more tools or decide it is done
  5. When LLM stops calling tools → ask for final JSON
  6. Parse JSON → merge with regex fields → save to state

Max iterations: 10 — prevents infinite loops if LLM gets stuck
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from state import MedicalFaxState, ProcessingPhase
from tools import verify_npi, lookup_ndc, validate_field_value

# ── Config ────────────────────────────────────────────────────────────────────

MAX_TEXT_CHARS  = 2000
PAGES_TO_SEND   = 2
MAX_ITERATIONS  = 10   # safety cap on the ReAct loop

# ── Tool registry ─────────────────────────────────────────────────────────────
# Dict mapping tool name → callable
# Used in the ReAct loop to execute whatever tool the LLM chose

TOOL_REGISTRY = {
    "verify_npi":           verify_npi,
    "lookup_ndc":           lookup_ndc,
    "validate_field_value": validate_field_value,
}

# ── System prompt ─────────────────────────────────────────────────────────────
# This is what the LLM reads to understand its role and tools.
# Written as instructions to the LLM — not developer documentation.
# Tool descriptions in the @tool docstrings tell LLM WHEN to call each tool.
# This prompt tells LLM HOW to behave overall.

SYSTEM_PROMPT = """You are a US healthcare document field extractor processing medical fax documents.

You have been given partial extraction results from a regex pass.
Your job is to:
  1. Use your tools to verify and validate already-extracted fields
  2. Find values for missing fields by reading the raw document text
  3. Flag values that look like OCR errors or form labels, not real values

TOOLS AVAILABLE — use them proactively:
  verify_npi(npi_number)
    → Call this for every prescriber_npi you see or find
    → Confirms it belongs to a real US licensed provider
    → entity_type tells you if it is Individual (prescriber) or Organization (pharmacy)

  lookup_ndc(drug_name)
    → Call this for every drug_prescribed field
    → Returns FDA standard name and NDC code
    → Use the standard_name to correct inconsistent OCR drug names

  validate_field_value(field_name, value)
    → Call this on any value that looks suspicious
    → Call this on values YOU find before adding them to found_fields
    → Use the reason field to explain what is wrong

CRITICAL RULE — NPI name matching:
  After verify_npi returns a valid result, you MUST check whether
  the provider_name returned matches the prescriber_name found in
  the document. If they do NOT refer to the same person, the NPI
  belongs to someone else — do not accept it. Add it to
  suspicious_fields instead, explaining the name mismatch.
  Example: if verify_npi returns "Amanda Weindl" but the document
  says "Dr. Michael Anderson", these do not match — flag it.

DECISION RULES:
  - If prescriber_npi is in extracted fields → always call verify_npi on it
  - If drug_prescribed is in extracted fields → always call lookup_ndc on it
  - If a value is a single word like 'Other', 'Prescribed', 'Product' → validate it
  - If you find a value for a missing field → validate it before accepting it
  - Never invent values not present in the document text

After you finish all tool calls, produce a final JSON response.
The JSON must match this exact schema — no markdown fences, no preamble:
{
  "found_fields": {
    "field_name": "value you found for a missing field"
  },
  "suspicious_fields": {
    "field_name": "reason this extracted value looks wrong"
  },
  "validation_passed": true or false,
  "reasoning": "one sentence: what you found, what you flagged, what tools confirmed"
}

validation_passed = true ONLY when:
  suspicious_fields is empty AND all missing fields were found or confirmed unfindable"""


# ── User prompt builder ───────────────────────────────────────────────────────

def _build_user_prompt(state: MedicalFaxState) -> str:
    doc_type         = state.get("doc_type",         "unknown")
    extracted_fields = state.get("extracted_fields", {})
    missing_fields   = state.get("missing_fields",   [])
    text_per_page    = state.get("text_per_page",    [])
    raw_text         = state.get("raw_text",         "")
    memory_hints     = state.get("memory_hints",     [])

    # Send first N pages capped at MAX_TEXT_CHARS
    if text_per_page:
        context_text = "\n\n--- PAGE BREAK ---\n\n".join(text_per_page[:PAGES_TO_SEND])
    else:
        context_text = raw_text
    if len(context_text) > MAX_TEXT_CHARS:
        context_text = context_text[:MAX_TEXT_CHARS] + "\n... [truncated]"

    extracted_str = (
        "\n".join(f"  {k}: {v}" for k, v in extracted_fields.items())
        if extracted_fields else "  (none extracted)"
    )
    missing_str = (
        "\n".join(f"  - {f}" for f in missing_fields)
        if missing_fields else "  (none — all fields found by regex)"
    )

    # ── Memory hints section (Day 2 addition) ──────────────────────────
    # Only included when past similar documents had relevant issues.
    # Empty string when no hints — keeps prompt clean for first-ever runs.
    memory_section = ""
    if memory_hints:
        hints_str = "\n".join(f"  - {h}" for h in memory_hints)
        memory_section = f"""

LESSONS FROM SIMILAR PAST DOCUMENTS:
{hints_str}
Use these hints to double-check the current document for similar issues."""

    return f"""Document type: {doc_type}

Already extracted by regex — verify these, do NOT re-extract:
{extracted_str}

Missing fields to find in raw text:
{missing_str}{memory_section}

Raw document text (first {PAGES_TO_SEND} pages, max {MAX_TEXT_CHARS} chars):
{context_text}"""


# ── Tool executor ─────────────────────────────────────────────────────────────

def _execute_tool(tool_call: dict) -> str:
    """
    Execute a single tool call from the LLM.

    tool_call structure from LangChain:
      { "name": "verify_npi", "args": {"npi_number": "1699091280"}, "id": "..." }

    Returns the tool result as a JSON string.
    ToolMessage requires string content — we serialise the dict.
    """
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    tool_fn = TOOL_REGISTRY.get(tool_name)
    if not tool_fn:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        result = tool_fn.invoke(tool_args)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Final JSON parser (unchanged from previous version) ───────────────────────

def _parse_final_response(content: str) -> dict:
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*",     "", content)
    content = content.strip()
    start   = content.find("{")
    end     = content.rfind("}") + 1
    if start != -1 and end > start:
        content = content[start:end]
    try:
        parsed = json.loads(content)
        return {
            "found_fields":      parsed.get("found_fields",      {}),
            "suspicious_fields": parsed.get("suspicious_fields", {}),
            "validation_passed": parsed.get("validation_passed", False),
            "reasoning":         parsed.get("reasoning",         ""),
        }
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse failed: {e}")
        return {
            "found_fields":      {},
            "suspicious_fields": {"parse_error": str(e)},
            "validation_passed": False,
            "reasoning":         f"JSON parsing failed: {e}",
        }


# ── Main node ─────────────────────────────────────────────────────────────────

def ai_extraction_node(state: MedicalFaxState) -> dict:
    """
    ReAct loop: LLM calls tools autonomously until satisfied, then returns JSON.
    """
    print("\n=== AI EXTRACTION NODE (tool-calling) ===")

    errors    = list(state.get("error_messages", []))
    missing   = state.get("missing_fields",   [])
    extracted = state.get("extracted_fields", {})

    print(f"  extracted fields : {list(extracted.keys())}")
    print(f"  missing fields   : {missing}")

    try:
        # ── Bind tools to LLM ─────────────────────────────────────────
        # .bind_tools() tells the LLM what tools exist and how to call them.
        # LangChain reads the @tool docstrings and parameter types to build
        # the tool schema the LLM sees.
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=2000,
        ).bind_tools(list(TOOL_REGISTRY.values()))

        # ── Initialise message thread ──────────────────────────────────
        # This is the conversation history the LLM sees each iteration.
        # We append to it: AI message → ToolMessage(s) → AI message → ...
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_build_user_prompt(state)),
        ]

        # ── Tool results accumulated across all iterations ─────────────
        # Saved to state so synthesis can show what was verified
        tool_results_accumulated: dict = {}
        iteration = 0

        # ── ReAct loop ─────────────────────────────────────────────────
        print(f"\n  Starting ReAct loop (max {MAX_ITERATIONS} iterations)...")

        while iteration < MAX_ITERATIONS:
            iteration += 1
            print(f"\n  — Iteration {iteration} —")

            # LLM reasons over current message thread
            response = llm.invoke(messages)

            # Check if LLM wants to call any tools
            tool_calls = getattr(response, "tool_calls", []) or []

            if not tool_calls:
                # LLM made no tool calls — it is done reasoning
                # This response contains the final JSON answer
                print(f"  LLM finished tool calling after {iteration} iteration(s)")
                print(f"  Final response: {response.content[:300]}")
                break

            # LLM wants to call tools — add its message to thread
            messages.append(response)
            print(f"  LLM calling {len(tool_calls)} tool(s):")

            # Execute each tool the LLM requested
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id   = tool_call["id"]

                print(f"    → {tool_name}({tool_args})")
                result_str = _execute_tool(tool_call)

                # Store result for state
                tool_results_accumulated[tool_name] = json.loads(result_str)

                # Feed result back to LLM as ToolMessage
                # LangChain requires tool_call_id to match the call
                messages.append(
                    ToolMessage(
                        content=result_str,
                        tool_call_id=tool_id,
                    )
                )
                print(f"    ← result saved, fed back to LLM")

        else:
            print(f"  ⚠️  Hit max iterations ({MAX_ITERATIONS}) — forcing final response")
            # Ask LLM to produce final JSON without more tool calls
            messages.append(HumanMessage(
                content="You have reached the maximum tool call limit. "
                        "Now produce your final JSON response based on what you have found."
            ))
            response = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                max_tokens=1000,
            ).invoke(messages)

        # ── Parse final JSON from last LLM response ────────────────────
        parsed            = _parse_final_response(response.content)
        found_fields      = parsed["found_fields"]
        suspicious_fields = parsed["suspicious_fields"]
        validation_passed = parsed["validation_passed"]
        reasoning         = parsed["reasoning"]

        print(f"\n  Parsed result:")
        print(f"    found_fields      : {found_fields}")
        print(f"    suspicious_fields : {suspicious_fields}")
        print(f"    validation_passed : {validation_passed}")
        print(f"    reasoning         : {reasoning}")

        # ── Merge regex + AI findings ──────────────────────────────────
        # Regex is the base. AI fills gaps.
        # Suspicious values get flagged inline so synthesis sees them.
        ai_refined_fields = {**extracted, **found_fields}
        for field, reason in suspicious_fields.items():
            if field in ai_refined_fields:
                original = ai_refined_fields[field]
                ai_refined_fields[field] = f"[FLAGGED: {original}] {reason}"

        # ── Pull key tool findings into top-level state fields ─────────
        # ndc_code and npi_verified get their own state fields
        # because synthesis uses them directly in final_output
        ndc_code     = ""
        npi_verified = False

        if "lookup_ndc" in tool_results_accumulated:
            ndc_result = tool_results_accumulated["lookup_ndc"]
            if ndc_result.get("found"):
                ndc_code = ndc_result.get("ndc_code", "")
                # Also correct the drug name in refined fields
                std_name = ndc_result.get("standard_name", "")
                if std_name:
                    ai_refined_fields["drug_prescribed"] = std_name
                    ai_refined_fields["ndc_code"]        = ndc_code

        if "verify_npi" in tool_results_accumulated:
            npi_result   = tool_results_accumulated["verify_npi"]
            npi_verified = npi_result.get("valid", False)

            if npi_verified:
                verified_name   = npi_result.get("provider_name", "").lower()
                prescriber_name = (
                    found_fields.get("prescriber_name") or
                    extracted.get("prescriber_name") or
                    ""
                ).lower()

                # Check if any meaningful word from verified name
                # appears in the document's prescriber name
                name_words = [w for w in verified_name.split() if len(w) > 2]
                name_match = any(w in prescriber_name for w in name_words)

                if prescriber_name and not name_match:
                    # NPI is real but belongs to a different person
                    npi_verified = False
                    suspicious_fields["prescriber_npi"] = (
                        f"NPI verified as '{npi_result.get('provider_name')}' "
                        f"but document prescriber is '{prescriber_name}' — mismatch"
                    )
                else:
                    ai_refined_fields["prescriber_npi_verified_name"] = (
                        npi_result.get("provider_name", "")
                    )

        # ── Build ai_feedback list ─────────────────────────────────────
        ai_feedback = []
        for field, reason in suspicious_fields.items():
            ai_feedback.append(f"SUSPICIOUS — {field}: {reason}")
        for field in missing:
            if field not in found_fields:
                ai_feedback.append(f"NOT FOUND — {field}: could not locate in text")
        if npi_verified:
            name = tool_results_accumulated["verify_npi"].get("provider_name", "")
            ai_feedback.append(f"NPI VERIFIED — prescriber confirmed: {name}")
        if ndc_code:
            ai_feedback.append(f"NDC CONFIRMED — {ndc_code}")
        if not ai_feedback:
            ai_feedback.append("All fields validated successfully")

        message_text = (
            f"AI extraction complete. "
            f"Iterations: {iteration}. "
            f"Tools called: {list(tool_results_accumulated.keys())}. "
            f"Found {len(found_fields)} missing fields. "
            f"Flagged {len(suspicious_fields)} suspicious. "
            f"Validation passed: {validation_passed}."
        )
        print(f"\n  {message_text}")

        return {
            "messages":          [AIMessage(content=message_text)],
            "ai_refined_fields": ai_refined_fields,
            "ai_feedback":       ai_feedback,
            "validation_passed": validation_passed,
            "tool_results":      tool_results_accumulated,
            "ndc_code":          ndc_code,
            "npi_verified":      npi_verified,
            "current_phase":     ProcessingPhase.AI_EXTRACTION,
            "error_messages":    errors,
        }

    except Exception as e:
        msg = f"AI extraction failed: {str(e)}"
        print(f"  ERROR: {msg}")
        errors.append(msg)
        return {
            "messages":          [AIMessage(content=msg)],
            "ai_refined_fields": extracted,
            "ai_feedback":       [f"AI extraction error: {str(e)}"],
            "validation_passed": False,
            "tool_results":      {},
            "ndc_code":          "",
            "npi_verified":      False,
            "current_phase":     ProcessingPhase.AI_EXTRACTION,
            "error_messages":    errors,
        }


# ── Routing — unchanged ───────────────────────────────────────────────────────

def route_after_ai_extraction(state: MedicalFaxState) -> str:
    print(f"  → routing to synthesis "
          f"(validation_passed={state.get('validation_passed')})")
    return "synthesis"