"""
Node functions and routing logic for the complex extraction agent
"""

import os
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from models import ComplexAgentState, ExtractionPhase
from tools import analyze_transcript_structure, extract_with_confidence, validate_extraction, refine_extraction
from models import ComplexityLevel
from dotenv import load_dotenv

load_dotenv()

def create_model():
    """Create and return the ChatOpenAI model with tools bound."""
    llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY")
    )
    
    return llm.bind_tools([
        analyze_transcript_structure,
        extract_with_confidence,
        validate_extraction,
        refine_extraction
    ])

def clean_messages_for_non_tool_nodes(messages):
    """Clean messages by removing complete tool call/response pairs"""
    clean_messages = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        
        # Check if this is an AI message with tool calls
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            # Look for the corresponding tool message(s)
            j = i + 1
            while j < len(messages) and hasattr(messages[j], 'type') and messages[j].type == 'tool':
                j += 1
            # Skip the entire tool call sequence (AI + tool messages)
            i = j
            continue
        
        # Keep non-tool messages
        clean_messages.append(msg)
        i += 1
    
    return clean_messages

# Node definitions
def preprocessing_node(state: ComplexAgentState) -> Dict[str, Any]:
    """Analyze transcript structure and determine processing approach"""
    print("\n=== PREPROCESSING PHASE ===")
    
    # Do the analysis directly without tools to avoid JSON issues
    transcript = state['transcript']
    lines = transcript.split('\n')
    participants = set()
    topics = []
    
    for line in lines:
        if ':' in line:
            speaker = line.split(':')[0].strip()
            participants.add(speaker)
            if any(keyword in line.lower() for keyword in ['decision', 'action', 'deadline', 'conflict']):
                topics.append(line.strip())
    
    complexity = ComplexityLevel.SIMPLE
    if len(participants) > 5 or len(topics) > 10:
        complexity = ComplexityLevel.COMPLEX
    elif len(participants) > 3 or len(topics) > 5:
        complexity = ComplexityLevel.MODERATE
    
    analysis_result = {
        "complexity_level": complexity,
        "participant_count": len(participants),
        "topic_count": len(topics),
        "analysis_confidence": 0.9,
        "recommended_approach": "detailed" if complexity == ComplexityLevel.COMPLEX else "standard"
    }
    
    # Create a simple response message
    from langchain_core.messages import AIMessage
    response = AIMessage(content=f"Transcript analysis complete. Complexity: {complexity}, Participants: {len(participants)}, Topics: {len(topics)}")
    
    return {
        "messages": [response],
        "current_phase": ExtractionPhase.INITIAL_EXTRACTION,
        "extraction_attempts": state["extraction_attempts"] + 1
    }

def initial_extraction_node(state: ComplexAgentState) -> Dict[str, Any]:
    """Perform initial extraction with confidence scoring"""
    print("\n=== INITIAL EXTRACTION PHASE ===")
    
    # Create model WITHOUT tools to avoid JSON issues
    llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY")
    )
    
    extraction_prompt = f"""
    Extract comprehensive information from this meeting transcript and provide a structured summary.
    
    Please extract and organize the following information:
    - Meeting title and type
    - Participants and their roles
    - Meeting duration and date
    - Key decisions made
    - Action items with assignees and deadlines
    - Conflicts and their resolutions
    - Key insights and risks identified
    - Follow-up meetings scheduled
    - Success metrics
    - Next steps
    
    Transcript:
    {state['transcript']}
    
    Provide a comprehensive, well-structured summary of all the extracted information.
    """
    
    # Clean messages by removing complete tool call/response pairs
    clean_messages = clean_messages_for_non_tool_nodes(state["messages"])
    
    messages = clean_messages + [HumanMessage(content=extraction_prompt)]
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "current_phase": ExtractionPhase.VALIDATION,
        "extraction_attempts": state["extraction_attempts"] + 1
    }

def validation_node(state: ComplexAgentState) -> Dict[str, Any]:
    """Validate extraction quality and provide feedback"""
    print("\n=== VALIDATION PHASE ===")
    
    # Create model WITHOUT tools to avoid JSON issues
    llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY")
    )
    
    validation_prompt = """
    Review the previous extraction and provide validation feedback.
    
    Check for:
    - Completeness of information extracted
    - Accuracy of details
    - Missing critical information
    - Quality of the summary
    
    Provide specific feedback on what was done well and what could be improved.
    """
    
    # Clean messages by removing complete tool call/response pairs
    clean_messages = clean_messages_for_non_tool_nodes(state["messages"])
    
    messages = clean_messages + [HumanMessage(content=validation_prompt)]
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "current_phase": ExtractionPhase.REFINEMENT
    }

def refinement_node(state: ComplexAgentState) -> Dict[str, Any]:
    """Refine extraction based on validation feedback"""
    print("\n=== REFINEMENT PHASE ===")
    
    model = create_model()
    refinement_prompt = f"""
    Refine the extraction based on validation feedback using the refine_extraction tool.
    Original transcript: {state['transcript'][:300]}...
    
    Focus on addressing the specific issues identified in validation.
    """
    
    # Clean messages by removing complete tool call/response pairs
    clean_messages = clean_messages_for_non_tool_nodes(state["messages"])
    
    messages = clean_messages + [HumanMessage(content=refinement_prompt)]
    response = model.invoke(messages)
    
    return {
        "messages": [response],
        "current_phase": ExtractionPhase.SYNTHESIS,
        "extraction_attempts": state["extraction_attempts"] + 1
    }

def synthesis_node(state: ComplexAgentState) -> Dict[str, Any]:
    """Synthesize final results"""
    print("\n=== SYNTHESIS PHASE ===")
    
    synthesis_prompt = """
    Synthesize the extraction results into a final, comprehensive summary.
    Include all key information in a well-structured format.
    """
    
    # Create model WITHOUT tools for synthesis
    llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Clean messages by removing complete tool call/response pairs
    clean_messages = clean_messages_for_non_tool_nodes(state["messages"])
    
    messages = clean_messages + [HumanMessage(content=synthesis_prompt)]
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "processing_complete": True,
        "current_phase": ExtractionPhase.SYNTHESIS
    }

def fallback_node(state: ComplexAgentState) -> Dict[str, Any]:
    """Fallback to simplified extraction if complex extraction fails"""
    print("\n=== FALLBACK PHASE ===")
    
    fallback_prompt = f"""
    Perform a simplified extraction from this transcript:
    {state['transcript']}
    
    Focus on basic information: participants, main topics, and key outcomes.
    Use a simpler structure if the complex extraction failed.
    """
    
    
    llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY")
    )
    messages = [HumanMessage(content=fallback_prompt)]
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "processing_complete": True,
        "current_phase": ExtractionPhase.FALLBACK
    }

# Conditional routing functions
def route_after_preprocessing(state: ComplexAgentState) -> str:
    """Route based on transcript complexity"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    
    # Default to initial extraction
    return "initial_extraction"

def route_after_validation(state: ComplexAgentState) -> str:
    """Route based on validation results"""
    # Check if validation passed or if we need refinement
    if state["extraction_attempts"] >= state["max_attempts"]:
        return "synthesis"  # Skip to synthesis if too many attempts
    
    # Check for validation results in messages
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    
    # Default routing logic - skip refinement for now
    return "synthesis"

def route_after_refinement(state: ComplexAgentState) -> str:
    """Route after refinement - either to synthesis or back to validation"""
    if state["extraction_attempts"] >= state["max_attempts"]:
        return "synthesis"
    
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    
    return "synthesis"

def route_after_tool_node(state: ComplexAgentState) -> str:
    """Route after tool execution based on current phase"""
    phase = state.get("current_phase", ExtractionPhase.PREPROCESSING)
    
    # Look for the most recent AIMessage with tool_calls to determine what tool was called
    for i in range(len(state["messages"]) - 1, -1, -1):
        msg = state["messages"][i]
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_name = msg.tool_calls[0]['name']
            if tool_name == 'analyze_transcript_structure':
                return "initial_extraction"
            elif tool_name == 'extract_with_confidence':
                return "validation"
            elif tool_name == 'validate_extraction':
                return "synthesis"
            elif tool_name == 'refine_extraction':
                return "synthesis"
            break
    
    # Fallback to phase-based routing
    if phase == ExtractionPhase.PREPROCESSING:
        return "initial_extraction"
    elif phase == ExtractionPhase.INITIAL_EXTRACTION:
        return "validation"
    elif phase == ExtractionPhase.VALIDATION:
        return "synthesis"
    elif phase == ExtractionPhase.REFINEMENT:
        return "synthesis"
    else:
        return "synthesis" 