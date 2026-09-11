"""
Shared Agent State Schema for OmniBrain Supervisor & Multi-Agent State Machine.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class TraceEvent(BaseModel):
    """Execution trace event emitted during multi-agent orchestration."""
    event_type: str  # 'thought', 'action', 'tool', 'result', 'state_update'
    agent: str       # 'Supervisor', 'SearchAgent', 'SQLAgent', 'VisionAgent', 'Synthesizer', 'GuardrailEvaluator'
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class SubTask(BaseModel):
    """Decomposed sub-task routed to a specialized agent."""
    id: str
    description: str
    target_agent: str  # 'SQLAgent' or 'SearchAgent'
    status: str = "pending"  # 'pending', 'running', 'completed', 'failed'
    result_summary: Optional[str] = None

class SupervisorState(BaseModel):
    """Shared state container passed between agents in the state machine."""
    query: str
    sub_tasks: List[SubTask] = Field(default_factory=list)
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    sql_results: List[Dict[str, Any]] = Field(default_factory=list)
    execution_trace: List[TraceEvent] = Field(default_factory=list)
    synthesized_memo: Optional[str] = None
    guardrail_report: Dict[str, Any] = Field(default_factory=dict)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    self_correction: Optional[Dict[str, Any]] = None
    execution_time_seconds: float = 0.0
    # Compact, cited turn summaries only. Full documents and raw images remain
    # in the RAG stores; they must never be copied into conversational memory.
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)

    def add_trace(self, event_type: str, agent: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Helper to append an execution trace event."""
        self.execution_trace.append(TraceEvent(
            event_type=event_type,
            agent=agent,
            content=content,
            metadata=metadata or {}
        ))
