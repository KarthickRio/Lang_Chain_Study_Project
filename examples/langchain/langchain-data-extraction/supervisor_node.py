"""
supervisor_node.py

The Supervisor decides which specialist agents actually need to run.
No LLM call here — this is pure routing logic, same style as
route_after_preprocessing from Day 1.

Why no LLM for the supervisor
───────────────────────────────
Deciding "is patient_name missing? then run Patient Agent" doesn't
need reasoning — it's a lookup against missing_fields. Using an LLM
here would be slower and more expensive for zero benefit. Reserve
LLM calls for steps that actually need judgment.

Skip logic (per your decision)
─────────────────────────────────
A specialist is skipped ONLY when ALL of its owned fields were
already found by regex. If even one field is missing, the specialist
runs and re-confirms everything in its domain — simpler than partial
re-runs, matches POC scope.
"""

from state import MedicalFaxState

# ── Field ownership map ────────────────────────────────────────────────────
# Which fields belong to which specialist — used to decide skip/run

PATIENT_FIELDS  = {"patient_name", "dob"}
PROVIDER_FIELDS = {"prescriber_name", "prescriber_npi", "dea_number",
                    "pharmacy_name", "pharmacy_npi"}
DRUG_FIELDS     = {"drug_prescribed", "strength", "sig", "qty_prescribed"}


def supervisor_node(state: MedicalFaxState) -> dict:
    """
    Reads missing_fields, decides which specialists need to run.
    Saves specialists_needed to state — used by conditional edges
    in agent.py to skip nodes that aren't needed.
    """
    print("\n=== SUPERVISOR NODE ===")

    missing_fields = set(state.get("missing_fields", []))

    specialists_needed = []

    # Patient Agent needed if ANY patient field is missing
    if missing_fields & PATIENT_FIELDS:
        specialists_needed.append("patient_agent")
        print(f"  Patient Agent  → NEEDED  (missing: {missing_fields & PATIENT_FIELDS})")
    else:
        print(f"  Patient Agent  → skipped (all patient fields found)")

    # Provider Agent needed if ANY provider field is missing
    if missing_fields & PROVIDER_FIELDS:
        specialists_needed.append("provider_agent")
        print(f"  Provider Agent → NEEDED  (missing: {missing_fields & PROVIDER_FIELDS})")
    else:
        print(f"  Provider Agent → skipped (all provider fields found)")

    # Drug Agent needed if ANY drug field is missing
    if missing_fields & DRUG_FIELDS:
        specialists_needed.append("drug_agent")
        print(f"  Drug Agent     → NEEDED  (missing: {missing_fields & DRUG_FIELDS})")
    else:
        print(f"  Drug Agent     → skipped (all drug fields found)")

    print(f"\n  Specialists to run, in order: {specialists_needed}")

    return {
        "specialists_needed": specialists_needed,
    }


# ── Conditional edge functions ───────────────────────────────────────────────
# Each specialist's "next step" depends on what Supervisor decided.
# These are called by LangGraph after each node to pick the next one.

def route_after_supervisor(state: MedicalFaxState) -> str:
    """First specialist to run, or skip straight to merge if none needed."""
    needed = state.get("specialists_needed", [])
    if "patient_agent" in needed:
        return "patient_agent"
    elif "provider_agent" in needed:
        return "provider_agent"
    elif "drug_agent" in needed:
        return "drug_agent"
    else:
        return "merge_specialists"


def route_after_patient_agent(state: MedicalFaxState) -> str:
    needed = state.get("specialists_needed", [])
    if "provider_agent" in needed:
        return "provider_agent"
    elif "drug_agent" in needed:
        return "drug_agent"
    else:
        return "merge_specialists"


def route_after_provider_agent(state: MedicalFaxState) -> str:
    needed = state.get("specialists_needed", [])
    if "drug_agent" in needed:
        return "drug_agent"
    else:
        return "merge_specialists"