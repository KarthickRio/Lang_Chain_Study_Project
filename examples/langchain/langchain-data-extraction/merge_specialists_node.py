"""
merge_specialists_node.py

Combines Patient, Provider, and Drug agent results into the SAME
output shape the old single ai_extraction_node produced —
ai_refined_fields, ai_feedback, validation_passed, tool_results,
ndc_code, npi_verified.

Why match the old shape exactly
───────────────────────────────
synthesis_node.py was built to read these exact state fields.
By producing identical output shape here, synthesis needs ZERO
changes — multi-agent is a swap of HOW fields get filled, not WHAT
shape the result takes. This is good architecture: the contract
between extraction and synthesis stays stable even as extraction
internals change completely.
"""

from langchain_core.messages import AIMessage
from state import MedicalFaxState, ProcessingPhase


def merge_specialists_node(state: MedicalFaxState) -> dict:
    """
    Reads patient_agent_result, provider_agent_result, drug_agent_result.
    Produces the same combined output shape synthesis_node expects.
    """
    print("\n=== MERGE SPECIALISTS NODE ===")

    extracted = state.get("extracted_fields", {})

    patient_result  = state.get("patient_agent_result")  or {}
    provider_result = state.get("provider_agent_result") or {}
    drug_result     = state.get("drug_agent_result")      or {}

    # ── Collect found_fields from all 3 specialists ───────────────────
    all_found_fields = {}
    all_found_fields.update(patient_result.get("found_fields", {}))
    all_found_fields.update(provider_result.get("found_fields", {}))
    all_found_fields.update(drug_result.get("found_fields", {}))

    # ── Collect suspicious_fields from Provider and Drug agents ───────
    # (Patient Agent doesn't flag suspicious fields — no tools, no
    #  cross-check needed for plain demographic extraction)
    all_suspicious_fields = {}
    all_suspicious_fields.update(provider_result.get("suspicious_fields", {}))
    all_suspicious_fields.update(drug_result.get("suspicious_fields", {}))

    # ── Merge regex base + specialist findings ─────────────────────────
    ai_refined_fields = {**extracted, **all_found_fields}

    # ── Collect tool results early — needed below to recover suspicious
    #    values that were never placed in found_fields ───────────────────
    all_tool_results = {}
    all_tool_results.update(provider_result.get("tool_results", {}))
    all_tool_results.update(drug_result.get("tool_results", {}))

    for field, reason in all_suspicious_fields.items():
        if field in ai_refined_fields:
            # Normal case: field has a value already (from regex or
            # a specialist's found_fields) — wrap it with the flag
            original = ai_refined_fields[field]
            ai_refined_fields[field] = f"[FLAGGED: {original}] {reason}"
        else:
            # Bug fix (Day 4 testing): a specialist correctly did NOT
            # put a wrong value into found_fields, but that meant the
            # value disappeared from final output entirely instead of
            # staying visible as a flagged value. Recover the actual
            # value that was checked from tool_results so a human
            # reviewer can still see WHAT was found and rejected,
            # not just that something was wrong.
            recovered_value = "(value not recovered)"
            if field == "prescriber_npi" and "verify_npi" in all_tool_results:
                # The NPI that was checked isn't in the tool result itself
                # (verify_npi returns the REAL owner's info, not the
                # input NPI) — so we note that explicitly instead of
                # showing the wrong provider's NPI as if it were correct.
                wrong_owner = all_tool_results["verify_npi"].get("provider_name", "unknown")
                recovered_value = f"NPI on file belongs to: {wrong_owner}"
            elif field == "drug_prescribed" and "lookup_ndc" in all_tool_results:
                recovered_value = all_tool_results["lookup_ndc"].get("standard_name", "(unrecognised)")

            ai_refined_fields[field] = f"[FLAGGED: {recovered_value}] {reason}"

    # ── Tool results already collected above, before the flag-wrap loop ──

    # ── Pull ndc_code and npi_verified into top-level fields ───────────
    ndc_code     = ""
    npi_verified = False

    if "lookup_ndc" in all_tool_results:
        ndc_result = all_tool_results["lookup_ndc"]
        if ndc_result.get("found") and "drug_prescribed" not in all_suspicious_fields:
            ndc_code = ndc_result.get("ndc_code", "")
            std_name = ndc_result.get("standard_name", "")
            if std_name:
                ai_refined_fields["drug_prescribed"] = std_name
                ai_refined_fields["ndc_code"]        = ndc_code

    if "verify_npi" in all_tool_results:
        npi_result   = all_tool_results["verify_npi"]
        npi_verified = npi_result.get("valid", False) and "prescriber_npi" not in all_suspicious_fields
        if npi_verified:
            ai_refined_fields["prescriber_npi_verified_name"] = (
                npi_result.get("provider_name", "")
            )

    # ── Validation passed only if NOTHING was flagged suspicious ───────
    validation_passed = len(all_suspicious_fields) == 0

    # ── Build ai_feedback list — same format as Day 1/2 ────────────────
    ai_feedback = []
    for field, reason in all_suspicious_fields.items():
        ai_feedback.append(f"SUSPICIOUS — {field}: {reason}")
    missing = state.get("missing_fields", [])
    for field in missing:
        if field not in all_found_fields:
            ai_feedback.append(f"NOT FOUND — {field}: could not locate in text")
    if npi_verified:
        name = all_tool_results["verify_npi"].get("provider_name", "")
        ai_feedback.append(f"NPI VERIFIED — prescriber confirmed: {name}")
    if ndc_code:
        ai_feedback.append(f"NDC CONFIRMED — {ndc_code}")
    if not ai_feedback:
        ai_feedback.append("All fields validated successfully")

    specialists_run = state.get("specialists_needed", [])
    message_text = (
        f"Merge complete. Specialists run: {specialists_run}. "
        f"Found {len(all_found_fields)} missing fields. "
        f"Flagged {len(all_suspicious_fields)} suspicious. "
        f"Validation passed: {validation_passed}."
    )
    print(f"  {message_text}")

    return {
        "messages":          [AIMessage(content=message_text)],
        "ai_refined_fields": ai_refined_fields,
        "ai_feedback":       ai_feedback,
        "validation_passed": validation_passed,
        "tool_results":      all_tool_results,
        "ndc_code":          ndc_code,
        "npi_verified":      npi_verified,
        "current_phase":     ProcessingPhase.AI_EXTRACTION,
    }