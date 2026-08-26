import json
import os
from typing import Optional, Literal
from app.schemas.state import AgentState, QueryAnalysis
from app.llm import LLMManager  


class QueryRouter:
    """Supervisor that analyzes queries and decides routing
    
    Uses your existing LLMManager for both Groq and OpenAI support.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "openai/gpt-oss-20b",
        provider: Literal["groq", "openai"] = "groq"
    ):
        """
        Initialize QueryRouter with your LLMManager
        
        Args:
            api_key: API key (uses env vars if not provided)
            model_name: Model to use
                - Groq: "mixtral-8x7b-32768", "llama-3.1-8b-instant"
                - OpenAI: "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"
            provider: "groq" or "openai"
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        
        # Initialize LLMManager (your existing class)
        self.llm_manager = LLMManager(
            api_key=api_key,
            model_name=model_name
        )
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyzes user query to determine routing:
        - semantic_search: keyword/contextual lookup in vector DB
        - sql_query: structured data query (stock prices, financial metrics)
        - vision_analysis: extract info from charts/tables
        - hybrid: combination of above
        """
        
        system_prompt = """You are a query router for a financial AI system.
Analyze the user's query and classify it as one of:
1. 'semantic_search' - textual/conceptual questions (e.g., "What does the CEO say about growth?")
2. 'sql_query' - structured data queries (e.g., "Show revenue growth 2021-2023", "Stock price on March 15?")
3. 'vision_analysis' - requires parsing charts/tables (e.g., "Analyze the P&L chart", "Extract table data")
4. 'hybrid' - requires multiple approaches

Respond in JSON format with keys: query_type, confidence (0-1), reasoning"""

        prompt = f"""Query: {query}

Respond ONLY with valid JSON in this format:
{{
    "query_type": "semantic_search|sql_query|vision_analysis|hybrid",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        try:
            # Use your existing LLMManager to generate response
            response_text = self.llm_manager.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            # Parse JSON from response
            response_text = response_text.strip()
            
            # Handle cases where LLM wraps JSON in markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(response_text)
            
            return QueryAnalysis(
                query_type=parsed["query_type"],
                confidence=float(parsed.get("confidence", 0.5)),
                reasoning=parsed.get("reasoning", "")
            )
        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            return self._fallback_analysis(query, str(e))

    @staticmethod
    def _fallback_analysis(query: str, error: str) -> QueryAnalysis:
        normalized_query = query.lower()
        vision_terms = ("chart", "table", "graph", "visual", "shown", "image")
        sql_terms = ("stock price", "revenue", "financial metric", "historical", "date")
        has_vision_terms = any(term in normalized_query for term in vision_terms)
        has_sql_terms = any(term in normalized_query for term in sql_terms)

        if has_vision_terms and has_sql_terms:
            query_type = "hybrid"
        elif has_vision_terms:
            query_type = "vision_analysis"
        elif has_sql_terms:
            query_type = "sql_query"
        else:
            query_type = "semantic_search"

        return QueryAnalysis(
            query_type=query_type,
            confidence=0.7,
            reasoning=f"Local fallback classification after LLM response error: {error}"
        )


def supervisor_node(
    state: AgentState,
    api_key: Optional[str] = None,
    model_name: str = "openai/gpt-oss-20b",
    provider: Literal["groq", "openai"] = "groq"
) -> dict:
    """
    Main supervisor node in the LangGraph state machine.
    Routes the query to the appropriate agents using your LLMManager.
    
    Args:
        state: Current agent state
        api_key: LLM API key (uses env if not provided)
        model_name: Model to use
        provider: "groq" or "openai"
    """
    
    # Initialize router using your LLMManager
    router = QueryRouter(
        api_key=api_key,
        model_name=model_name,
        provider=provider
    )
    
    # Analyze incoming query
    analysis = router.analyze_query(state["user_query"])
    
    # Update state
    state["query_analysis"] = analysis
    state["routing_decision"] = analysis.query_type
    state["intermediate_steps"].append(
        ("supervisor", f"Routed to {analysis.query_type} (provider: {provider}, model: {model_name})")
    )
    
    return state
