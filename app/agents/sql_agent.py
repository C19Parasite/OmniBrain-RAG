"""
SQL Agent for querying structured financial data.

Converts natural language queries to SQL and executes them against your database.
Integration point for your SQL database/ORM.
"""

from app.schemas.state import AgentState
from app.llm import LLMManager
from typing import Optional, List, Dict, Any
import json


class SQLQueryAgent:
    """
    Converts natural language to SQL and executes queries.
    
    Example queries:
    - "What was the stock price on March 15, 2023?"
    - "Show me revenue growth from 2021 to 2023"
    - "List all companies with revenue > $1B"
    """
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        database_connection: Optional[Any] = None,
        model_name: str = "openai/gpt-oss-20b"
    ):
        """
        Initialize SQL Agent
        
        Args:
            llm_manager: Your existing LLMManager (for text-to-SQL conversion)
            database_connection: Connection to your SQL database
            model_name: Model for SQL generation
        """
        self.llm_manager = llm_manager or LLMManager(model_name=model_name)
        self.database_connection = database_connection
        self.model_name = model_name
    
    def generate_sql(self, query: str, schema_info: str = "") -> str:
        """
        Convert natural language query to SQL using LLM.
        
        Args:
            query: Natural language query
            schema_info: Database schema information for context
            
        Returns:
            Generated SQL query
        """
        
        system_prompt = """You are a SQL query generator for a financial database.
Convert natural language queries to SQL.

IMPORTANT RULES:
1. Generate ONLY the SQL query, no explanations
2. Use standard SQL syntax
3. Include comments explaining the query
4. Handle date formats: YYYY-MM-DD
5. Be defensive about data types

Database tables available:
- stock_prices (date, symbol, open, close, high, low, volume)
- companies (symbol, name, sector, revenue, employees)
- financial_metrics (symbol, date, revenue, profit, margin)
- earnings (symbol, date, eps, guidance)

Return ONLY the SQL query as plain text, no markdown or backticks."""

        prompt = f"""Schema Information:
{schema_info if schema_info else "Use the standard financial schema provided above."}

Natural language query:
{query}

Generate the SQL query:"""

        try:
            # Use your LLMManager to generate SQL
            sql_query = self.llm_manager.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            # Clean up response
            sql_query = sql_query.strip()
            if sql_query.startswith("```"):
                sql_query = sql_query.split("```")[1]
            if sql_query.startswith("sql"):
                sql_query = sql_query[3:]
            
            return sql_query.strip()
        
        except Exception as e:
            return f"SELECT 'Error generating SQL: {str(e)}' as error"
    
    def execute_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """
        Execute SQL query against database.
        
        Integration point: Replace with your actual database connection
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            List of result dictionaries
        """
        
        # TODO: Replace this with YOUR actual database connection
        # Example with SQLAlchemy:
        # from sqlalchemy import text
        # connection = self.database_connection
        # result = connection.execute(text(sql_query))
        # return [dict(row) for row in result]
        
        # Example with psycopg2:
        # cursor = self.database_connection.cursor()
        # cursor.execute(sql_query)
        # results = cursor.fetchall()
        # return results
        
        # MOCK DATA for testing
        if "stock_price" in sql_query.lower() and "march 15" in sql_query.lower():
            return [
                {
                    "date": "2023-03-15",
                    "symbol": "AAPL",
                    "open": 143.50,
                    "close": 145.20,
                    "high": 147.80,
                    "low": 143.25,
                    "volume": 45000000
                }
            ]
        
        elif "revenue" in sql_query.lower() and "growth" in sql_query.lower():
            return [
                {"year": 2021, "revenue": 365.8, "growth": 0.00},
                {"year": 2022, "revenue": 394.3, "growth": 0.078},
                {"year": 2023, "revenue": 383.3, "growth": -0.028}
            ]
        
        else:
            # Generic mock response
            return [
                {"query_result": "Sample result from database", "value": 123.45}
            ]
    
    def query(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Process a natural language query end-to-end.
        
        Args:
            natural_language_query: User's natural language query
            
        Returns:
            Dictionary with SQL, results, and metadata
        """
        
        try:
            # Step 1: Generate SQL from natural language
            sql_query = self.generate_sql(natural_language_query)
            
            # Step 2: Execute SQL
            results = self.execute_query(sql_query)
            
            return {
                "success": True,
                "sql_query": sql_query,
                "results": results,
                "result_count": len(results),
                "error": None
            }
        
        except Exception as e:
            return {
                "success": False,
                "sql_query": None,
                "results": [],
                "result_count": 0,
                "error": str(e)
            }


def sql_agent_node(state: AgentState) -> dict:
    """
    LangGraph node for SQL agent execution.
    
    Only runs if routing decision is "sql_query" or "hybrid".
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with SQL results
    """
    
    # Only execute if routing says we need SQL
    if state["routing_decision"] not in ["sql_query", "hybrid"]:
        return state
    
    try:
        # Initialize SQL agent
        agent = SQLQueryAgent()
        
        # Execute query
        result = agent.query(state["user_query"])
        
        # Add results to state
        if result["success"]:
            state["sql_results"].extend(result["results"])
            state["intermediate_steps"].append(
                ("sql_agent", f"Executed SQL, found {result['result_count']} results")
            )
        else:
            state["errors"].append(f"SQL Error: {result['error']}")
            state["intermediate_steps"].append(
                ("sql_agent", f"SQL execution failed: {result['error']}")
            )
    
    except Exception as e:
        state["errors"].append(f"SQL Agent Error: {str(e)}")
        state["intermediate_steps"].append(
            ("sql_agent", f"Agent error: {str(e)}")
        )
    
    return state