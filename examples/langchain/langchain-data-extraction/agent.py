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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
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
from human_review_node          import human_review_node
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
    workflow.add_node("human_review",           human_review_node)
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

    # ── merge_specialists always flows into human_review ───────────────
    workflow.add_edge("merge_specialists", "human_review")

    # ── human_review always flows into synthesis ────────────────────────
    # (human_review_node itself decides internally whether to pause —
    #  if nothing is suspicious it returns immediately, no actual pause)
    workflow.add_edge("human_review", "synthesis")

    # ── synthesis always flows into memory_save, then END ──────────────
    workflow.add_edge("synthesis",    "memory_save")
    workflow.add_edge("memory_save",  END)

    # ── Compile WITH a checkpointer ─────────────────────────────────────
    # Required for interrupt() to work — LangGraph needs somewhere to
    # save in-progress state so it can resume after a pause.
    # MemorySaver keeps everything in memory for this script's lifetime —
    # enough for a POC. A real deployment would use a persistent
    # checkpointer (e.g. SqliteSaver or a database-backed one).
    #
    # allowed_msgpack_modules registers our custom str-Enum types
    # (PDFType, DocType, ProcessingPhase) explicitly with the
    # serializer. Without this, LangGraph prints a deprecation
    # warning on every checkpoint/resume and will outright block
    # these types in a future version — fixing it now avoids a
    # breaking change later.
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("state", "PDFType"),
            ("state", "DocType"),
            ("state", "ProcessingPhase"),
        ]
    )
    checkpointer = MemorySaver(serde=serde)
    return workflow.compile(checkpointer=checkpointer)