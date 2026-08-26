from app.schemas.state import AgentState
from app.vector_store import VectorStoreManager  

def search_agent(state: AgentState) -> dict:
    """Queries the vector database using your existing VectorStoreManager"""
    query = state["user_query"]
    
    # Use your existing VectorStoreManager
    vector_store = VectorStoreManager()
    results = vector_store.search_similar(query, top_k=5)
    
    state["search_results"].extend(results)
    state["intermediate_steps"].append(("search_agent", f"Found {len(results)} results"))
    
    return state

def search_agent_node(state: AgentState) -> dict:
    """Only execute if routing decision is semantic_search or hybrid"""
    if state["routing_decision"] in ["semantic_search", "hybrid"]:
        return search_agent(state)
    return state