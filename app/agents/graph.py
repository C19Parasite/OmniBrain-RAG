from langgraph.graph import StateGraph, END
from typing import Optional, Literal
from app.schemas.state import AgentState
from app.agents.supervisor import supervisor_node
from app.agents.search_agent import search_agent_node
from app.agents.sql_agent import sql_agent_node
from app.agents.vision_agent import vision_agent_node
from app.llm import LLMManager  # Your existing LLMManager
import json

def build_agentic_graph(
    api_key: Optional[str] = None,
    model_name: str = "openai/gpt-oss-20b",
    provider: Literal["groq", "openai"] = "groq"
):
    """
    Constructs the LangGraph state machine with supervisor routing.
    Uses your existing LLMManager for all LLM operations.
    
    Args:
        api_key: LLM API key (uses env if not provided)
        model_name: Model to use
            - Groq (default): "mixtral-8x7b-32768", "llama-3.1-8b-instant"
            - OpenAI: "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"
        provider: "groq" (recommended, free) or "openai" (paid, high quality)
    
    Returns:
        Compiled LangGraph workflow
    """
    
    # Initialize graph
    workflow = StateGraph(AgentState)
    
    # Create supervisor wrapper with LLMManager
    def supervisor_wrapper(state: AgentState) -> dict:
        """Wrapper to pass LLM config to supervisor"""
        return supervisor_node(
            state,
            api_key=api_key,
            model_name=model_name,
            provider=provider
        )
    
    # Add supervisor as entry point
    workflow.add_node("supervisor", supervisor_wrapper)
    
    # Add sub-agents
    workflow.add_node("search_agent", search_agent_node)
    workflow.add_node("sql_agent", sql_agent_node)
    workflow.add_node("vision_agent", vision_agent_node)
    
    # Add synthesizer (combines all results)
    workflow.add_node("synthesizer", synthesizer_node)
    
    # Define routing logic
    def route_supervisor(state: AgentState) -> str:
        """Determine next node based on routing_decision"""
        routing = state["routing_decision"]
        
        if routing == "semantic_search":
            return "search_agent"
        elif routing == "sql_query":
            return "sql_agent"
        elif routing == "vision_analysis":
            return "vision_agent"
        elif routing == "hybrid":
            # For hybrid, start with search
            return "search_agent"
        else:
            return "search_agent"  # Default
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "search_agent": "search_agent",
            "sql_agent": "sql_agent",
            "vision_agent": "vision_agent"
        }
    )
    
    # All agents converge to synthesizer
    workflow.add_edge("search_agent", "synthesizer")
    workflow.add_edge("sql_agent", "synthesizer")
    workflow.add_edge("vision_agent", "synthesizer")
    
    # Synthesizer is final
    workflow.add_edge("synthesizer", END)
    
    # Compile and return
    return workflow.compile()


def synthesizer_node(
    state: AgentState,
    api_key: Optional[str] = None,
    model_name: str = "openai/gpt-oss-20b"
) -> dict:
    """
    Combines results from all agents into final response.
    Uses your LLMManager for synthesis.
    """
    
    # Initialize LLMManager for synthesis
    llm_manager = LLMManager(
        api_key=api_key,
        model_name=model_name
    )
    
    # Build synthesis prompt
    context_parts = []
    if state["search_results"]:
        context_parts.append(f"Vector DB Results:\n{json.dumps(state['search_results'], indent=2)}")
    if state["sql_results"]:
        context_parts.append(f"SQL Results:\n{json.dumps(state['sql_results'], indent=2)}")
    if state["vision_results"]:
        context_parts.append(f"Vision Analysis:\n{json.dumps(state['vision_results'], indent=2)}")
    
    system_prompt = """You are an expert financial analyst. 
Synthesize the retrieved information into a concise, cited investment memo.
Clearly distinguish between data from different sources.
Avoid hallucinations—only state what's explicitly provided."""
    
    prompt = f"""Original Query: {state['user_query']}

Available Data:
{chr(10).join(context_parts) if context_parts else "No results found."}

Generate a comprehensive but concise response based on the available data."""
    
    # Generate synthesis using LLMManager
    response = llm_manager.generate(
        prompt=prompt,
        system_prompt=system_prompt
    )
    
    state["final_response"] = response
    state["intermediate_steps"].append(("synthesizer", "Response synthesized"))
    
    return state



