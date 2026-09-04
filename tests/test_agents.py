import unittest
from backend.app.agents.sql_agent import TextToSQLAgent
from backend.app.agents.supervisor import SupervisorOrchestrator

class TestAgents(unittest.TestCase):
    def test_sql_agent_query(self):
        agent = TextToSQLAgent()
        res = agent.run("What was NVDA revenue in recent quarters?")
        self.assertTrue(res.get("is_valid"))
        self.assertGreater(len(res.get("rows", [])), 0)

    def test_supervisor_planning(self):
        orchestrator = SupervisorOrchestrator()
        sub_tasks = orchestrator._decompose_query("Compare NVDA quarterly revenue and explain AI growth")
        self.assertGreaterEqual(len(sub_tasks), 2)

if __name__ == "__main__":
    unittest.main()
