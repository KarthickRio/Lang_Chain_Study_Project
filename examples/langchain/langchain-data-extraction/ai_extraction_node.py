"""
ai_extraction_node.py

Receives partial extraction results from preprocessing.
Does two jobs:
  1. Find values for missing fields from raw text
  2. Flag suspicious values in already-extracted fields

Returns structured JSON saved directly into state.

LLM used: claude-sonnet (via langchain-anthropic)
  pip install langchain-anthropic

Why not OpenAI here:
  You can swap the model easily — just change the import and
  model name. The prompt and parsing logic stays identical.
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from state import MedicalFaxState, ProcessingPhase
# ── Config ────────────────────────────────────────────────────────────────────

MAX_TEXT_CHARS = 2000   # cap on raw_text sent to LLM
PAGES_TO_SEND  = 2      # only send first N pages


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a medical document field extractor.
You receive partial extraction results from a medical fax document.

Your two jobs:
  1. Find values for the listed missing fields from the raw text
  2. Flag any already-extracted values that look suspicious
     (form labels mistaken as values, placeholders, garbage OCR text)

Respond ONLY with valid JSON. No explanation before or after. No markdown fences.

The JSON must match this exact schema:
{
  "found_fields": {
    "field_name": "value"
  },
  "suspicious_fields": {
    "field_name": "reason why this value looks wrong"
  },
  "validation_passed": true,
  "reasoning": "one sentence summary of what you found and flagged"
}

Rules:
- Only include a field in found_fields if you are confident it is in the text
- If you cannot find a missing field, do not include it in found_fields
- suspicious_fields should only flag values from already_extracted that look wrong
- validation_passed = true only when:
    suspicious_fields is empty AND all missing_fields were found
- Never invent or guess values not present in the raw text
- For dates, normalise to YYYY-MM-DD format
- For names, use Title Case"""


def _build_user_prompt(state: MedicalFaxState) -> str:
    """
    Build the user message from state fields.
    Sends only first PAGES_TO_SEND pages, capped at MAX_TEXT_CHARS.
    """
    doc_type         = state.get("doc_type", "unknown")
    extracted_fields = state.get("extracted_fields", {})
    missing_fields   = state.get("missing_fields", [])
    text_per_page    = state.get("text_per_page", [])
    raw_text         = state.get("raw_text", "")

    # ── Select text to send ───────────────────────────────────────────
    if text_per_page:
        # Send first PAGES_TO_SEND pages joined together
        selected_pages = text_per_page[:PAGES_TO_SEND]
        context_text   = "\n\n--- PAGE BREAK ---\n\n".join(selected_pages)
    else:
        context_text = raw_text

    # Apply character cap
    if len(context_text) > MAX_TEXT_CHARS:
        context_text = context_text[:MAX_TEXT_CHARS] + "\n... [truncated]"

    # ── Format missing fields as a clean list ─────────────────────────
    missing_list = "\n".join(f"  - {f}" for f in missing_fields) if missing_fields else "  (none)"

    # ── Format extracted fields as readable key: value ────────────────
    if extracted_fields:
        extracted_str = "\n".join(f"  {k}: {v}" for k, v in extracted_fields.items())
    else:
        extracted_str = "  (none extracted)"

    return f"""Document type: {doc_type}

Already extracted — do NOT re-extract these, only check if values look suspicious:
{extracted_str}

Missing fields to find in the raw text:
{missing_list}

Raw document text (first {PAGES_TO_SEND} pages, max {MAX_TEXT_CHARS} chars):
{context_text}"""


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_llm_response(content: str) -> dict:
    """
    Parse the LLM's JSON response safely.

    Handles two common failure modes:
      1. LLM wraps JSON in markdown fences  ```json ... ```
      2. LLM adds a preamble sentence before the JSON

    Returns a dict with safe defaults if parsing fails.
    """
    # Strip markdown fences if present
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*",     "", content)
    content = content.strip()

    # Find the JSON object — look for first { to last }
    start = content.find("{")
    end   = content.rfind("}") + 1
    if start != -1 and end > start:
        content = content[start:end]

    try:
        parsed = json.loads(content)

        # Ensure all expected keys exist with safe defaults
        return {
            "found_fields":      parsed.get("found_fields",      {}),
            "suspicious_fields": parsed.get("suspicious_fields", {}),
            "validation_passed": parsed.get("validation_passed", False),
            "reasoning":         parsed.get("reasoning",         "No reasoning provided"),
        }

    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse failed: {e}")
        print(f"  Raw response was: {content[:300]}")
        return {
            "found_fields":      {},
            "suspicious_fields": {"parse_error": str(e)},
            "validation_passed": False,
            "reasoning":         f"JSON parsing failed: {e}",
        }


# ── Node function ─────────────────────────────────────────────────────────────

def ai_extraction_node(state: MedicalFaxState) -> dict:
    """
    Calls the LLM with partial extraction results.
    Saves: ai_refined_fields, ai_feedback, validation_passed into state.
    """
    print("\n=== AI EXTRACTION NODE ===")

    errors       = list(state.get("error_messages", []))
    missing      = state.get("missing_fields", [])
    extracted    = state.get("extracted_fields", {})

    print(f"  Sending to LLM:")
    print(f"    extracted fields : {list(extracted.keys())}")
    print(f"    missing fields   : {missing}")
    print(f"    text pages sent  : first {PAGES_TO_SEND} (max {MAX_TEXT_CHARS} chars)")

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1000)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_build_user_prompt(state)),
        ]

        response = llm.invoke(messages)
        raw_content = response.content
        print(f"\n  LLM raw response:\n  {raw_content[:500]}")

        # ── Parse structured response ─────────────────────────────────
        parsed = _parse_llm_response(raw_content)

        found_fields      = parsed["found_fields"]
        suspicious_fields = parsed["suspicious_fields"]
        validation_passed = parsed["validation_passed"]
        reasoning         = parsed["reasoning"]

        print(f"\n  Parsed result:")
        print(f"    found_fields      : {found_fields}")
        print(f"    suspicious_fields : {suspicious_fields}")
        print(f"    validation_passed : {validation_passed}")
        print(f"    reasoning         : {reasoning}")

        # ── Merge: start with regex extraction, overlay AI findings ───
        # Regex results are the base. AI fills gaps and may correct
        # suspicious values. We keep both for traceability.
        ai_refined_fields = {**extracted, **found_fields}

        # Replace suspicious values with a flagged marker so synthesis
        # knows these need human review
        for field, reason in suspicious_fields.items():
            if field in ai_refined_fields:
                original = ai_refined_fields[field]
                ai_refined_fields[field] = f"[FLAGGED: {original}] {reason}"

        # ai_feedback as a flat list of strings for easy reading
        ai_feedback = []
        for field, reason in suspicious_fields.items():
            ai_feedback.append(f"SUSPICIOUS — {field}: {reason}")
        for field in missing:
            if field not in found_fields:
                ai_feedback.append(f"NOT FOUND — {field}: could not locate in text")
        if not ai_feedback:
            ai_feedback.append("All fields validated successfully")

        message_text = (
            f"AI extraction complete. "
            f"Found {len(found_fields)} missing fields. "
            f"Flagged {len(suspicious_fields)} suspicious values. "
            f"Validation passed: {validation_passed}."
        )

        return {
            "messages":          [AIMessage(content=message_text)],
            "ai_refined_fields": ai_refined_fields,
            "ai_feedback":       ai_feedback,
            "validation_passed": validation_passed,
            "current_phase":     ProcessingPhase.AI_EXTRACTION,
            "error_messages":    errors,
        }

    except Exception as e:
        msg = f"AI extraction failed: {str(e)}"
        print(f"  ERROR: {msg}")
        errors.append(msg)

        return {
            "messages":          [AIMessage(content=msg)],
            "ai_refined_fields": extracted,   # fall back to regex results
            "ai_feedback":       [f"AI extraction error: {str(e)}"],
            "validation_passed": False,
            "current_phase":     ProcessingPhase.AI_EXTRACTION,
            "error_messages":    errors,
        }


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_ai_extraction(state: MedicalFaxState) -> str:
    """
    Always goes to synthesis.
    validation_passed flag in state tells synthesis how confident to be.
    """
    print(f"  → routing to synthesis "
          f"(validation_passed={state.get('validation_passed')})")
    return "synthesis"