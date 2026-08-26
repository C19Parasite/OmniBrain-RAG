import pytest
import os
from dotenv import load_dotenv
from app.agents.supervisor import QueryRouter
from app.schemas.state import AgentState, QueryAnalysis
from app.agents.graph import build_agentic_graph
from app.llm_config import LLMConfig

load_dotenv()


class TestQueryRouterWithGroq:
    """Test QueryRouter with Groq (free, fast)"""
    
    def setup_method(self):
        """Skip if Groq API key not available"""
        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not found")
    
    def test_supervisor_semantic_search_groq(self):
        """Test semantic_search routing with Groq"""
        router = QueryRouter(
            model_name="openai/gpt-oss-20b",
            provider="groq"
        )
        
        query = "What does the CEO say about future growth?"
        analysis = router.analyze_query(query)
        
        assert analysis.query_type == "semantic_search"
        assert analysis.confidence > 0.5
        print(f"✓ Groq - Semantic search: {analysis.reasoning}")
    
    def test_supervisor_sql_query_groq(self):
        """Test sql_query routing with Groq"""
        router = QueryRouter(
            model_name="openai/gpt-oss-20b",
            provider="groq"
        )
        
        query = "What was the stock price on March 15, 2023?"
        analysis = router.analyze_query(query)
        
        assert analysis.query_type == "sql_query"
        assert analysis.confidence > 0.5
        print(f"✓ Groq - SQL query: {analysis.reasoning}")
    
    def test_supervisor_vision_analysis_groq(self):
        """Test vision_analysis routing with Groq"""
        router = QueryRouter(
            model_name="openai/gpt-oss-20b",
            provider="groq"
        )
        
        query = "Analyze the profit margins shown in the P&L chart."
        analysis = router.analyze_query(query)
        
        assert analysis.query_type == "vision_analysis"
        assert analysis.confidence > 0.5
        print(f"✓ Groq - Vision analysis: {analysis.reasoning}")
    
    def test_supervisor_hybrid_groq(self):
        """Test hybrid routing with Groq"""
        router = QueryRouter(
            model_name="openai/gpt-oss-20b",
            provider="groq"
        )
        
        query = "Compare growth trends in the document with historical stock prices and explain visual patterns in the revenue chart."
        analysis = router.analyze_query(query)
        
        assert analysis.query_type == "hybrid"
        print(f"✓ Groq - Hybrid query: {analysis.reasoning}")


class TestQueryRouterWithOpenAI:
    """Test QueryRouter with OpenAI (high quality, paid)"""
    
    def setup_method(self):
        """Skip if OpenAI API key not available"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not found")
    
    def test_supervisor_semantic_search_openai(self):
        """Test semantic_search routing with OpenAI"""
        router = QueryRouter(
            model_name="gpt-4o",
            provider="openai"
        )
        
        query = "What does the CEO say about future growth?"
        analysis = router.analyze_query(query)
        
        assert analysis.query_type == "semantic_search"
        assert analysis.confidence > 0.7
        print(f"✓ OpenAI - Semantic search: {analysis.reasoning}")
    
    def test_supervisor_sql_query_openai(self):
        """Test sql_query routing with OpenAI"""
        router = QueryRouter(
            model_name="gpt-4o",
            provider="openai"
        )
        
        query = "What was the stock price on March 15, 2023?"
        analysis = router.analyze_query(query)
        
        assert analysis.query_type == "sql_query"
        assert analysis.confidence > 0.7
        print(f"✓ OpenAI - SQL query: {analysis.reasoning}")


class TestGraphBuilding:
    """Test LangGraph building with LLMManager"""
    
    def test_graph_build_with_groq(self):
        """Test building graph with Groq"""
        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not found")
        
        graph = build_agentic_graph(
            model_name="openai/gpt-oss-20b",
            provider="groq"
        )
        assert graph is not None
        print("✓ Graph built with Groq (LLMManager)")
    
    def test_graph_build_with_openai(self):
        """Test building graph with OpenAI"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not found")
        
        graph = build_agentic_graph(
            model_name="gpt-4o",
            provider="openai"
        )
        assert graph is not None
        print("✓ Graph built with OpenAI (LLMManager)")
    
    def test_full_graph_execution_groq(self):
        """Test complete graph execution with Groq"""
        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not found")
        
        graph = build_agentic_graph(
            model_name="openai/gpt-oss-20b",
            provider="groq"
        )
        
        initial_state = AgentState(
            user_query="What was the stock price on March 15, 2023?",
            query_analysis=None,
            search_results=[],
            sql_results=[],
            vision_results=[],
            intermediate_steps=[],
            final_response="",
            errors=[],
            routing_decision=""
        )
        
        result = graph.invoke(initial_state)
        
        assert result["routing_decision"] == "sql_query"
        assert result["final_response"] != ""
        print(f"✓ Full graph execution with Groq successful")
        print(f"  Routing: {result['routing_decision']}")
        print(f"  Steps: {[step[0] for step in result['intermediate_steps']]}")


class TestLLMConfig:
    """Test LLM configuration management"""
    
    def test_get_groq_config(self):
        """Test Groq config retrieval"""
        config = LLMConfig.get_config("groq")
        assert config["provider"] == "groq"
        assert config["model"] == "openai/gpt-oss-20b"
        print(f"✓ Groq config: {config['description']}")
    
    def test_get_openai_config(self):
        """Test OpenAI config retrieval"""
        config = LLMConfig.get_config("openai")
        assert config["provider"] == "openai"
        assert config["model"] == "gpt-4o"
        print(f"✓ OpenAI config: {config['description']}")
    
    def test_provider_availability(self):
        """Check which providers are available"""
        available = LLMConfig.get_available_providers()
        print(f"✓ Available providers: {available}")
        assert len(available) > 0, "At least one provider API key must be configured"
    
    def test_recommended_config(self):
        """Test getting recommended config"""
        try:
            config = LLMConfig.get_recommended_config()
            print(f"✓ Recommended config: {config['provider']} - {config['model']}")
            assert config["provider"] in ["groq", "openai"]
        except ValueError as e:
            pytest.skip(str(e))


class TestLLMManagerIntegration:
    """Test integration with your existing LLMManager"""
    
    def test_llm_manager_with_groq(self):
        """Test LLMManager initialization with Groq"""
        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not found")
        
        from app.llm import LLMManager
        
        llm = LLMManager(model_name="openai/gpt-oss-20b")
        assert llm is not None
        assert llm.model_name == "openai/gpt-oss-20b"
        print("✓ LLMManager initialized with Groq")
    
    def test_llm_manager_with_openai(self):
        """Test LLMManager initialization with OpenAI"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not found")
        
        from app.llm import LLMManager
        
        llm = LLMManager(model_name="gpt-4o")
        assert llm is not None
        assert llm.model_name == "gpt-4o"
        print("✓ LLMManager initialized with OpenAI")
    
    def test_llm_manager_generate_groq(self):
        """Test LLMManager.generate() with Groq"""
        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not found")
        
        from app.llm import LLMManager
        
        llm = LLMManager(model_name="openai/gpt-oss-20b")
        response = llm.generate(
            prompt="Classify this as 'positive' or 'negative': This is great!",
            system_prompt="You are a sentiment analyzer."
        )
        
        assert response is not None
        assert len(response) > 0
        print(f"✓ LLMManager.generate() works with Groq")
        print(f"  Response: {response[:100]}...")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing OmniBrain Agentic Orchestrator with LLMManager")
    print("=" * 70)
    
    # Check available providers
    available = LLMConfig.get_available_providers()
    print(f"\n📊 Available Providers: {available}\n")
    
    if not available:
        print("❌ No API keys found!")
        print("Please set up environment variables:")
        print("  - GROQ_API_KEY (free, recommended)")
        print("  - OPENAI_API_KEY (paid, optional)")
        exit(1)
    
    # Run pytest
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "-s"],
        cwd=os.path.dirname(__file__)
    )
    
    exit(result.returncode)
