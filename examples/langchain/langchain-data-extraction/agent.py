"""
agent.py

Wires all nodes into a LangGraph StateGraph.

Full flow:
    pdf_ingestion
         ↓  route_after_ingestion
    native_preprocessing ──┐
    scanned_preprocessing ─┤
                           ↓  route_after_preprocessing
                    ai_extraction  ←── (if confidence ≤ 0.80)
                           ↓            or skip straight to:
                       synthesis  ←─────────────────────────
                           ↓
                          END
"""

from langgraph.graph import StateGraph, END
from state import MedicalFaxState

from pdf_ingestion_node      import pdf_ingestion_node,      route_after_ingestion
from native_preprocessing_node  import native_preprocessing_node
from scanned_preprocessing_node import scanned_preprocessing_node, route_after_preprocessing
from ai_extraction_node      import ai_extraction_node,      route_after_ai_extraction
from synthesis_node          import synthesis_node


def create_agent():
    workflow = StateGraph(MedicalFaxState)

    # ── Register nodes ────────────────────────────────────────────────
    workflow.add_node("pdf_ingestion",          pdf_ingestion_node)
    workflow.add_node("native_preprocessing",   native_preprocessing_node)
    workflow.add_node("scanned_preprocessing",  scanned_preprocessing_node)
    workflow.add_node("ai_extraction",          ai_extraction_node)
    workflow.add_node("synthesis",              synthesis_node)

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

    # ── Conditional edge: both preprocessing paths → AI or synthesis ──
    workflow.add_conditional_edges(
        "native_preprocessing",
        route_after_preprocessing,
        {
            "ai_extraction": "ai_extraction",
            "synthesis":     "synthesis",
        }
    )
    workflow.add_conditional_edges(
        "scanned_preprocessing",
        route_after_preprocessing,
        {
            "ai_extraction": "ai_extraction",
            "synthesis":     "synthesis",
        }
    )

    # ── AI extraction always goes to synthesis ────────────────────────
    workflow.add_conditional_edges(
        "ai_extraction",
        route_after_ai_extraction,
        {
            "synthesis": "synthesis",
        }
    )

    # ── Synthesis is the terminal node ────────────────────────────────
    workflow.add_edge("synthesis", END)

    return workflow.compile()