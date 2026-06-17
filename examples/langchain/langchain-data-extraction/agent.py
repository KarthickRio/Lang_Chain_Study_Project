"""
agent.py

Wires all nodes into a LangGraph StateGraph.

Full flow (Day 3 — multi-agent specialists added)
──────────────────────────────────────────────────────
    pdf_ingestion
         ↓  route_after_ingestion
    native_preprocessing ──┐
    scanned_preprocessing ─┤
                           ↓  route_after_preprocessing
                    memory_lookup  ←── (if confidence ≤ 0.80)
                           ↓            or skip straight to:
                     supervisor          synthesis  ←──────┐
                           ↓  route_after_supervisor        │
                  patient_agent (if needed)                 │
                           ↓  route_after_patient_agent      │
                  provider_agent (if needed)                 │
                           ↓  route_after_provider_agent      │
                  drug_agent (if needed)                       │
                           ↓                                   │
                  merge_specialists                            │
                           ↓                                   │
                       synthesis ───────────────────────────────┘
                           ↓
                    memory_save
                           ↓
                          END

Why each specialist's conditional edge checks specialists_needed:
  Supervisor decided UP FRONT which specialists matter for this
  document. Each specialist's routing function reads that same
  decision to skip straight past agents that aren't needed —
  this is what makes Supervisor's routing actually save API calls,
  not just a label that gets ignored.
"""

from langgraph.graph import StateGraph, END
from state import MedicalFaxState

from pdf_ingestion_node         import pdf_ingestion_node,      route_after_ingestion
from native_preprocessing_node  import native_preprocessing_node
from scanned_preprocessing_node import scanned_preprocessing_node, route_after_preprocessing
from memory_lookup_node         import memory_lookup_node
from supervisor_node            import (
    supervisor_node, route_after_supervisor,
    route_after_patient_agent, route_after_provider_agent,
)
from patient_agent_node         import patient_agent_node
from provider_agent_node        import provider_agent_node
from drug_agent_node            import drug_agent_node
from merge_specialists_node     import merge_specialists_node
from synthesis_node             import synthesis_node
from memory_save_node           import memory_save_node


def create_agent():
    workflow = StateGraph(MedicalFaxState)

    # ── Register nodes ────────────────────────────────────────────────
    workflow.add_node("pdf_ingestion",          pdf_ingestion_node)
    workflow.add_node("native_preprocessing",   native_preprocessing_node)
    workflow.add_node("scanned_preprocessing",  scanned_preprocessing_node)
    workflow.add_node("memory_lookup",          memory_lookup_node)
    workflow.add_node("supervisor",             supervisor_node)
    workflow.add_node("patient_agent",          patient_agent_node)
    workflow.add_node("provider_agent",         provider_agent_node)
    workflow.add_node("drug_agent",             drug_agent_node)
    workflow.add_node("merge_specialists",      merge_specialists_node)
    workflow.add_node("synthesis",              synthesis_node)
    workflow.add_node("memory_save",            memory_save_node)

    # ── Entry point ───────────────────────────────────────────────────
    workflow.set_entry_point("pdf_ingestion")

    # ── Ingestion → native or scanned ──────────────────────────────────
    workflow.add_conditional_edges(
        "pdf_ingestion",
        route_after_ingestion,
        {
            "native_preprocessing":  "native_preprocessing",
            "scanned_preprocessing": "scanned_preprocessing",
        }
    )

    # ── Both preprocessing paths → memory_lookup or synthesis ─────────
    workflow.add_conditional_edges(
        "native_preprocessing",
        route_after_preprocessing,
        {
            "ai_extraction": "memory_lookup",
            "synthesis":     "synthesis",
        }
    )
    workflow.add_conditional_edges(
        "scanned_preprocessing",
        route_after_preprocessing,
        {
            "ai_extraction": "memory_lookup",
            "synthesis":     "synthesis",
        }
    )

    # ── memory_lookup always flows into supervisor ──────────────────────
    workflow.add_edge("memory_lookup", "supervisor")

    # ── Supervisor decides first specialist (or straight to merge) ─────
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "patient_agent":     "patient_agent",
            "provider_agent":    "provider_agent",
            "drug_agent":        "drug_agent",
            "merge_specialists": "merge_specialists",
        }
    )

    # ── Patient Agent → Provider, Drug, or merge ────────────────────────
    workflow.add_conditional_edges(
        "patient_agent",
        route_after_patient_agent,
        {
            "provider_agent":    "provider_agent",
            "drug_agent":        "drug_agent",
            "merge_specialists": "merge_specialists",
        }
    )

    # ── Provider Agent → Drug or merge ──────────────────────────────────
    workflow.add_conditional_edges(
        "provider_agent",
        route_after_provider_agent,
        {
            "drug_agent":        "drug_agent",
            "merge_specialists": "merge_specialists",
        }
    )

    # ── Drug Agent always flows into merge ──────────────────────────────
    workflow.add_edge("drug_agent", "merge_specialists")

    # ── merge_specialists always flows into synthesis ──────────────────
    workflow.add_edge("merge_specialists", "synthesis")

    # ── synthesis always flows into memory_save, then END ──────────────
    workflow.add_edge("synthesis",    "memory_save")
    workflow.add_edge("memory_save",  END)

    return workflow.compile()