"""
Reasoning Audit Test Suite for LangGraph Supervisor.
Proves that the Supervisor dynamically and accurately decides between:
1. Dense Vector DB Semantic Search (ChromaDB)
2. Structured Text-to-SQL Database Execution (SQLite)
3. Hybrid Multi-Agent Parallel Routing
"""
import unittest
from backend.app.agents.supervisor import SupervisorOrchestrator

class TestSupervisorReasoningAudit(unittest.TestCase):
    def setUp(self):
        self.supervisor = SupervisorOrchestrator()

    def test_pure_vector_rag_routing(self):
        """Prove qualitative/unstructured queries route to Vector DB (SearchAgent) only."""
        test_prompts = [
            "What are the primary risk factors described in the annual filing?",
            "Explain the leadership strategy and competitive advantages.",
            "What is written about environmental sustainability initiatives?"
        ]
        for prompt in test_prompts:
            with self.subTest(prompt=prompt):
                sub_tasks = self.supervisor._decompose_query(prompt)
                agents = [t.target_agent for t in sub_tasks]
                
                # Assert SearchAgent is invoked
                self.assertIn("SearchAgent", agents, f"Failed for prompt: '{prompt}'")
                # Assert SQLAgent is NOT invoked for non-database questions
                self.assertNotIn("SQLAgent", agents, f"SQLAgent wrongly triggered for prompt: '{prompt}'")

    def test_structured_sql_routing(self):
        """Prove quantitative database queries route to Text-to-SQL Agent."""
        test_prompts = [
            "What was NVDA quarterly revenue growth in the database?",
            "Show MSFT recent quarterly net income and gross margin metrics.",
            "Compare NVDA and MSFT revenue from database records."
        ]
        for prompt in test_prompts:
            with self.subTest(prompt=prompt):
                sub_tasks = self.supervisor._decompose_query(prompt)
                agents = [t.target_agent for t in sub_tasks]
                
                # Assert SQLAgent is explicitly assigned a sub-task
                self.assertIn("SQLAgent", agents, f"SQLAgent was not routed for quantitative query: '{prompt}'")

    def test_hybrid_multi_agent_routing(self):
        """Prove hybrid queries trigger parallel multi-agent execution."""
        hybrid_prompt = "Compare NVDA quarterly revenue from database and explain the reasons for gross margin expansion from the report"
        sub_tasks = self.supervisor._decompose_query(hybrid_prompt)
        agents = [t.target_agent for t in sub_tasks]
        
        self.assertIn("SQLAgent", agents, "SQLAgent missing from hybrid orchestration")
        self.assertIn("SearchAgent", agents, "SearchAgent missing from hybrid orchestration")
        self.assertGreaterEqual(len(sub_tasks), 2, "Supervisor did not formulate multi-step subtasks")

    def test_execution_trace_evidence(self):
        """Prove the supervisor generates auditable step-by-step reasoning traces."""
        state = self.supervisor.process_query("What was NVDA quarterly revenue from database?")
        trace_agents = [getattr(t, "agent", t.get("agent") if isinstance(t, dict) else "") for t in state.execution_trace]
        
        # Verify trace contains planning and execution evidence
        self.assertIn("Supervisor", trace_agents)
        self.assertIn("SQLAgent", trace_agents)
        self.assertIn("Synthesizer", trace_agents)
        self.assertIn("GuardrailEvaluator", trace_agents)

if __name__ == "__main__":
    unittest.main()
