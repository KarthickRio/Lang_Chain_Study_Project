"""
Agent construction and workflow definition for the complex extraction agent
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from models import ComplexAgentState, ComplexityLevel
from nodes import (
    preprocessing_node, initial_extraction_node, validation_node, 
    refinement_node, synthesis_node, fallback_node,
    route_after_preprocessing, route_after_validation, 
    route_after_refinement, route_after_tool_node
)
from tools import analyze_transcript_structure, extract_with_confidence, validate_extraction, refine_extraction

def create_complex_agent():
    """Create and return the complex LangGraph agent"""
    
    workflow = StateGraph(ComplexAgentState)
    
    # Add nodes
    workflow.add_node("preprocessing", preprocessing_node)
    workflow.add_node("initial_extraction", initial_extraction_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("refinement", refinement_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("tool_node", ToolNode([
        analyze_transcript_structure,
        extract_with_confidence,
        validate_extraction,
        refine_extraction
    ]))
    
    # Set entry point
    workflow.set_entry_point("preprocessing")
    
    # Add edges
    workflow.add_conditional_edges(
        "preprocessing",
        route_after_preprocessing,
        {
            "tool_node": "tool_node",
            "initial_extraction": "initial_extraction"
        }
    )
    
    # Route tool_node back to the appropriate next step
    workflow.add_conditional_edges(
        "tool_node",
        route_after_tool_node,
        {
            "initial_extraction": "initial_extraction",
            "validation": "validation", 
            "synthesis": "synthesis"
        }
    )
    
    # Add conditional edges for nodes that might make tool calls
    workflow.add_conditional_edges(
        "initial_extraction",
        lambda state: "tool_node" if hasattr(state["messages"][-1], 'tool_calls') and state["messages"][-1].tool_calls else "validation",
        {
            "tool_node": "tool_node",
            "validation": "validation"
        }
    )
    
    workflow.add_conditional_edges(
        "validation", 
        lambda state: "tool_node" if hasattr(state["messages"][-1], 'tool_calls') and state["messages"][-1].tool_calls else "synthesis",
        {
            "tool_node": "tool_node",
            "synthesis": "synthesis"
        }
    )
    
    workflow.add_edge("refinement", "synthesis")
    
    workflow.add_edge("synthesis", END)
    workflow.add_edge("fallback", END)
    
    return workflow.compile() 