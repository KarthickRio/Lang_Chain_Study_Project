"""
Data models and state definitions for the complex extraction agent
"""

from typing import Annotated, Sequence, TypedDict, List, Dict, Any
from enum import Enum
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class ExtractionPhase(str, Enum):
    PREPROCESSING = "preprocessing"
    INITIAL_EXTRACTION = "initial_extraction"
    VALIDATION = "validation"
    REFINEMENT = "refinement"
    SYNTHESIS = "synthesis"
    FALLBACK = "fallback"

# Enhanced State Management
class ComplexAgentState(TypedDict):
    """Extended state for complex extraction workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    transcript: str
    complexity_level: ComplexityLevel
    extraction_attempts: int
    max_attempts: int
    confidence_scores: Dict[str, float]
    validation_feedback: List[str]
    extraction_data: Dict[str, Any]
    current_phase: ExtractionPhase
    requires_refinement: bool
    processing_complete: bool
    error_count: int

# Enhanced Transcript Data Models
class ParticipantRole(BaseModel):
    name: str = Field(..., description="Participant name")
    role: str = Field(..., description="Role in meeting (facilitator, stakeholder, expert, etc.)")
    department: str = Field(..., description="Department or team")

class DecisionItem(BaseModel):
    decision: str = Field(..., description="Decision made")
    rationale: str = Field(..., description="Reasoning behind decision")
    stakeholders: List[str] = Field(..., description="People involved in decision")
    timestamp: str = Field(..., description="When decision was made")

class ActionItem(BaseModel):
    task: str = Field(..., description="Task to be completed")
    assignee: str = Field(..., description="Person responsible")
    deadline: str = Field(..., description="Due date")
    priority: str = Field(..., description="Priority level")
    dependencies: List[str] = Field(default=[], description="Dependencies")

class ConflictResolution(BaseModel):
    issue: str = Field(..., description="Conflicting issue")
    parties: List[str] = Field(..., description="Parties involved")
    resolution: str = Field(..., description="How conflict was resolved")
    compromise: str = Field(..., description="What compromises were made")

class MeetingPhase(BaseModel):
    phase_name: str = Field(..., description="Name of meeting phase")
    duration: str = Field(..., description="Duration of phase")
    key_topics: List[str] = Field(..., description="Main topics discussed")
    outcome: str = Field(..., description="Outcome of this phase")

class ComplexTranscriptSummary(BaseModel):
    """Enhanced extraction schema with confidence scoring"""
    
    # Meeting metadata
    meeting_title: str = Field(..., description="Title of the meeting")
    meeting_type: str = Field(..., description="Type of meeting (planning, review, decision, etc.)")
    participants: List[ParticipantRole] = Field(..., description="Participants with roles")
    duration: str = Field(..., description="Meeting duration")
    date: str = Field(..., description="Meeting date")
    
    # Meeting structure
    phases: List[MeetingPhase] = Field(..., description="Different phases of the meeting")
    
    # Key outcomes
    decisions: List[DecisionItem] = Field(..., description="Decisions made during meeting")
    action_items: List[ActionItem] = Field(..., description="Tasks assigned")
    conflicts: List[ConflictResolution] = Field(..., description="Conflicts and resolutions")
    
    # Analysis
    key_insights: List[str] = Field(..., description="Important insights from discussion")
    risks_identified: List[str] = Field(..., description="Risks or concerns raised")
    follow_up_meetings: List[str] = Field(..., description="Required follow-up meetings")
    
    # Executive summary
    executive_summary: str = Field(..., description="High-level summary for leadership")
    success_metrics: List[str] = Field(..., description="How success will be measured")
    next_steps: List[str] = Field(..., description="Immediate next steps") 