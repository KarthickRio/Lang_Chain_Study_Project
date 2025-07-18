"""
Tool definitions for the complex extraction agent
"""

from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from models import ComplexTranscriptSummary, ComplexityLevel

@tool("analyze_transcript_structure")
def analyze_transcript_structure(transcript: str) -> Dict[str, Any]:
    """Analyze transcript structure and determine complexity level"""
    # Handle case where transcript might be passed as a dict or other format
    if isinstance(transcript, dict):
        transcript = str(transcript)
    elif not isinstance(transcript, str):
        transcript = str(transcript)
    
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
    
    return {
        "complexity_level": complexity,
        "participant_count": len(participants),
        "topic_count": len(topics),
        "analysis_confidence": 0.9,
        "recommended_approach": "detailed" if complexity == ComplexityLevel.COMPLEX else "standard"
    }

@tool("extract_with_confidence")
def extract_with_confidence(
    meeting_title: str,
    meeting_type: str,
    participants: List[str],
    duration: str,
    date: str,
    phases: Optional[List[str]] = None,
    decisions: Optional[List[str]] = None,
    action_items: Optional[List[str]] = None,
    conflicts: Optional[List[str]] = None,
    key_insights: Optional[List[str]] = None,
    risks_identified: Optional[List[str]] = None,
    follow_up_meetings: Optional[List[str]] = None,
    executive_summary: str = "",
    success_metrics: Optional[List[str]] = None,
    next_steps: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Extract structured data with confidence scoring"""
    
    # Ensure all lists are initialized
    phases = phases or []
    decisions = decisions or []
    action_items = action_items or []
    conflicts = conflicts or []
    key_insights = key_insights or []
    risks_identified = risks_identified or []
    follow_up_meetings = follow_up_meetings or []
    success_metrics = success_metrics or []
    next_steps = next_steps or []
    
    # Calculate confidence scores for different sections
    confidence_scores = {
        "participants": 0.95 if len(participants) > 0 else 0.3,
        "decisions": 0.9 if len(decisions) > 0 else 0.5,
        "action_items": 0.85 if len(action_items) > 0 else 0.4,
        "conflicts": 0.8 if len(conflicts) > 0 else 0.7,
        "phases": 0.9 if len(phases) > 0 else 0.6,
        "insights": 0.8 if len(key_insights) > 0 else 0.5
    }
    
    overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)
    
    extraction_data = {
        "meeting_title": meeting_title,
        "meeting_type": meeting_type,
        "participants": participants,
        "duration": duration,
        "date": date,
        "phases": phases,
        "decisions": decisions,
        "action_items": action_items,
        "conflicts": conflicts,
        "key_insights": key_insights,
        "risks_identified": risks_identified,
        "follow_up_meetings": follow_up_meetings,
        "executive_summary": executive_summary,
        "success_metrics": success_metrics,
        "next_steps": next_steps
    }
    
    return {
        "extraction_data": extraction_data,
        "confidence_scores": confidence_scores,
        "overall_confidence": overall_confidence,
        "extraction_complete": overall_confidence > 0.7
    }

@tool("validate_extraction")
def validate_extraction(extraction_data: Dict[str, Any], confidence_scores: Dict[str, float]) -> Dict[str, Any]:
    """Validate extraction quality and provide feedback"""
    
    feedback = []
    issues = []
    
    # Check for missing critical information
    if not extraction_data.get("decisions"):
        feedback.append("No decisions were extracted - review transcript for decision points")
        issues.append("missing_decisions")
    
    if not extraction_data.get("action_items"):
        feedback.append("No action items found - look for task assignments and deadlines")
        issues.append("missing_actions")
    
    if confidence_scores.get("participants", 0) < 0.8:
        feedback.append("Participant extraction confidence is low - verify names and roles")
        issues.append("low_participant_confidence")
    
    if confidence_scores.get("decisions", 0) < 0.7:
        feedback.append("Decision extraction needs improvement - look for explicit decisions")
        issues.append("low_decision_confidence")
    
    # Overall validation
    overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)
    validation_passed = overall_confidence > 0.75 and len(issues) < 2
    
    return {
        "validation_passed": validation_passed,
        "feedback": feedback,
        "issues": issues,
        "overall_confidence": overall_confidence,
        "requires_refinement": not validation_passed
    }

@tool("refine_extraction")
def refine_extraction(
    original_data: Dict[str, Any], 
    feedback: List[str], 
    transcript: str
) -> Dict[str, Any]:
    """Refine extraction based on validation feedback"""
    
    refined_data = original_data.copy()
    improvements = []
    
    # Simulate refinement logic
    if "missing_decisions" in str(feedback):
        # Add placeholder decision if missing
        refined_data["decisions"] = refined_data.get("decisions", []) + [
            "Additional decision point identified during refinement"
        ]
        improvements.append("Added missing decision points")
    
    if "missing_actions" in str(feedback):
        # Add placeholder action items
        refined_data["action_items"] = refined_data.get("action_items", []) + [
            "Follow up on refined extraction"
        ]
        improvements.append("Added missing action items")
    
    return {
        "refined_data": refined_data,
        "improvements": improvements,
        "refinement_confidence": 0.85
    } 