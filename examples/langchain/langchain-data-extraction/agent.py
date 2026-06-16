"""
agent.py

Wires all nodes into a LangGraph StateGraph.

Full flow (Day 2 — memory added)
────────────────────────────────────
    pdf_ingestion
         ↓  route_after_ingestion
    native_preprocessing ──┐
    scanned_preprocessing ─┤
                           ↓  route_after_preprocessing
                    memory_lookup  ←── (if confidence ≤ 0.80, before AI)
                           ↓            or skip straight to:
                    ai_extraction       synthesis  ←──────────┐
                           ↓                                  │
                       synthesis  ─────────────────────────────┘
                           ↓
                    memory_save   ← NEW — always runs last, saves outcome
                           ↓
                          END

Why memory_lookup sits BEFORE ai_extraction:
  The hint needs to be in the prompt when the LLM reasons — giving it
  context after the fact would be useless.

Why memory_save sits AFTER synthesis, always:
  Even high-confidence AUTO_EXTRACTED documents get saved — we want
  memory of clean documents too, not just the messy ones that needed AI.
"""

from langgraph.graph import StateGraph, END
from state import MedicalFaxState

from pdf_ingestion_node         import pdf_ingestion_node,      route_after_ingestion
from native_preprocessing_node  import native_preprocessing_node
from scanned_preprocessing_node import scanned_preprocessing_node, route_after_preprocessing
from memory_lookup_node         import memory_lookup_node
from ai_extraction_node         import ai_extraction_node,      route_after_ai_extraction
from synthesis_node             import synthesis_node
from memory_save_node           import memory_save_node


def create_agent():
    workflow = StateGraph(MedicalFaxState)

    # ── Register nodes ────────────────────────────────────────────────
    workflow.add_node("pdf_ingestion",          pdf_ingestion_node)
    workflow.add_node("native_preprocessing",   native_preprocessing_node)
    workflow.add_node("scanned_preprocessing",  scanned_preprocessing_node)
    workflow.add_node("memory_lookup",          memory_lookup_node)
    workflow.add_node("ai_extraction",          ai_extraction_node)
    workflow.add_node("synthesis",              synthesis_node)
    workflow.add_node("memory_save",            memory_save_node)

    # ── Entry point ───────────────────────────────────────────────────
    workflow.set_entry_point("pdf_ingestion")

    # ── Conditional edge: ingestion → native or scanned ───────────────
    workflow.add_conditional_edges(
        "pdf_ingestion",
        route_after_ingestion,
        {
            "native_preprocessing":  "native_preprocessing",
            "scanned_preprocessing": "scanned_preprocessing",
        }
    )

    # ── Conditional edge: both preprocessing paths → memory_lookup or synthesis ──
    # Same routing function as before — "ai_extraction" target renamed
    # to go through memory_lookup first
    workflow.add_conditional_edges(
        "native_preprocessing",
        route_after_preprocessing,
        {
            "ai_extraction": "memory_lookup",   # ← goes to memory first now
            "synthesis":     "synthesis",
        }
    )
    workflow.add_conditional_edges(
        "scanned_preprocessing",
        route_after_preprocessing,
        {
            "ai_extraction": "memory_lookup",   # ← goes to memory first now
            "synthesis":     "synthesis",
        }
    )

    # ── memory_lookup always flows into ai_extraction ──────────────────
    workflow.add_edge("memory_lookup", "ai_extraction")

    # ── AI extraction always goes to synthesis ────────────────────────
    workflow.add_conditional_edges(
        "ai_extraction",
        route_after_ai_extraction,
        {
            "synthesis": "synthesis",
        }
    )

    # ── synthesis always flows into memory_save, then END ──────────────
    workflow.add_edge("synthesis",    "memory_save")
    workflow.add_edge("memory_save",  END)

    return workflow.compile()