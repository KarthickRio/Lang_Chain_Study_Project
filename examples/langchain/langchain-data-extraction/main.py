"""
Complex Extraction Agent using LangGraph
Demonstrates advanced workflow patterns including:
- Multi-stage processing with validation
- Conditional routing based on quality assessment
- Iterative refinement with feedback loops
- Fallback mechanisms for robustness
"""

import os
import sys
from langchain_core.messages import HumanMessage

# from pylangdb.langchain import init
# init()

# Import our modular components
from models import ComplexAgentState, ComplexityLevel
from agent import create_complex_agent
from transcript import get_complex_transcript

def main():
    """Run the complex extraction agent"""
    
    # Create the agent
    agent = create_complex_agent()
    
    # Get complex transcript
    transcript = get_complex_transcript()
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=f"Process this complex meeting transcript: {transcript}")],
        "transcript": transcript,
        "complexity_level": ComplexityLevel.COMPLEX,
        "extraction_attempts": 0,
        "max_attempts": 3,
        "confidence_scores": {},
        "validation_feedback": [],
        "extraction_data": {},
        "current_phase": "preprocessing",
        "requires_refinement": False,
        "processing_complete": False,
        "error_count": 0
    }
    
    print("=== COMPLEX EXTRACTION AGENT STARTING ===")
    print(f"Transcript length: {len(transcript)} characters")
    print(f"Estimated complexity: {ComplexityLevel.COMPLEX}")
    print("\n" + "="*60)
    
    # Run the agent
    try:
        phase_outputs = {}
        final_synthesis = None
        
        for output in agent.stream(initial_state):
            for key, value in output.items():
                if key == "__end__":
                    print("\n=== PROCESSING COMPLETE ===")
                    continue
                
                # Show phase transitions
                if key in ["preprocessing", "initial_extraction", "validation", "synthesis"]:
                    print(f"\n--- {key.upper()} PHASE ---")
                
                # Capture output from each phase
                if key in ["preprocessing", "initial_extraction", "validation", "synthesis"] and "messages" in value:
                    phase_content = ""
                    for msg in value["messages"]:
                        if hasattr(msg, 'content') and msg.content:
                            phase_content = msg.content
                            break
                    
                    if phase_content:
                        phase_outputs[key] = phase_content
                        print(f"\n{key.upper()} OUTPUT:")
                        print("-" * 40)
                        print(phase_content[:500] + "..." if len(phase_content) > 500 else phase_content)
                        print("-" * 40)
                
                # Capture synthesis output for final display
                if key == "synthesis" and "messages" in value:
                    for msg in value["messages"]:
                        if hasattr(msg, 'content') and msg.content:
                            final_synthesis = msg.content
                
                # Show tool node activity
                elif key == "tool_node" and "messages" in value:
                    for msg in value["messages"]:
                        if hasattr(msg, 'content') and msg.content:
                            if "Error:" in msg.content:
                                print(f"  ⚠️  Tool error: {msg.content[:100]}...")
                            else:
                                print(f"  ✅ Tool executed successfully")
        
        # Display final synthesis
        if final_synthesis:
            print("\n" + "="*60)
            print("=== FINAL EXTRACTION RESULTS ===")
            print("="*60)
            print(final_synthesis)
            print("="*60)
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to simplified extraction...")

if __name__ == "__main__":
    main()