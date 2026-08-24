from typing import Any, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated
import operator

class QueryAnalysis(BaseModel):
    """Supervisor's analysis of the incoming query"""
    query_type: Literal["semantic_search", "sql_query", "vision_analysis", "hybrid"]
    confidence: float = Field(ge=0, le=1)
    reasoning: str

class AgentState(TypedDict):
    """Central state managed by the LangGraph supervisor"""
    user_query: str
    query_analysis: QueryAnalysis
    search_results: Annotated[list[dict], operator.add]  # Vector DB results
    sql_results: Annotated[list[dict], operator.add]     # SQL results
    vision_results: Annotated[list[dict], operator.add]  # Vision analysis results
    intermediate_steps: Annotated[list[tuple[str, str]], operator.add]  # (agent, output)
    final_response: str
    errors: Annotated[list[str], operator.add]
    routing_decision: str