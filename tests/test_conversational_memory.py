"""
Unit and Integration Tests for Multi-Turn Contextual Memory and Co-Reference Resolution.
"""
import unittest
from uuid import uuid4
from fastapi.testclient import TestClient
from backend.app.agents.supervisor import SupervisorOrchestrator
from backend.app.main import app


class TestConversationalMemory(unittest.TestCase):
    def setUp(self):
        self.supervisor = SupervisorOrchestrator()
        self.client = TestClient(app)

    def test_heuristic_pronoun_resolution(self):
        """Prove pronouns ('their', 'its') are resolved to the prior turn's target subject."""
        history = [
            {
                "user_query": "What was NVIDIA revenue and gross margin in Q2 FY2025?",
                "assistant_memo": "NVIDIA reported record revenue of $30.04B with 75.1% gross margin."
            }
        ]
        resolved, was_resolved = self.supervisor._resolve_conversational_query(
            query="What drove their margin increase?",
            history=history
        )
        self.assertTrue(was_resolved)
        self.assertIn("NVIDIA", resolved)
        self.assertNotIn("their", resolved.lower())

    def test_heuristic_ellipsis_resolution(self):
        """Prove elliptical queries ('What about Microsoft?') incorporate metrics from prior turn."""
        history = [
            {
                "user_query": "Analyze NVIDIA revenue growth and data center performance in Q2 FY2025.",
                "assistant_memo": "NVIDIA revenue grew 122% YoY."
            }
        ]
        resolved, was_resolved = self.supervisor._resolve_conversational_query(
            query="What about Microsoft?",
            history=history
        )
        self.assertTrue(was_resolved)
        self.assertIn("Microsoft", resolved)
        self.assertTrue(any(term in resolved.lower() for term in ["revenue", "financial"]))

    def test_standalone_query_not_altered(self):
        """Prove independent queries without follow-up signals remain unchanged."""
        history = [
            {
                "user_query": "What was NVIDIA revenue?",
                "assistant_memo": "Revenue was $30B."
            }
        ]
        resolved, was_resolved = self.supervisor._resolve_conversational_query(
            query="Explain the difference between GAAP and non-GAAP accounting standards.",
            history=history
        )
        self.assertFalse(was_resolved)
        self.assertEqual(resolved, "Explain the difference between GAAP and non-GAAP accounting standards.")

    def test_supervisor_thread_memory_tracking(self):
        """Prove SupervisorOrchestrator maintains thread memory and resolves follow-ups."""
        thread_id = f"test_thread_{uuid4()}"

        # Turn 1
        state_turn1 = self.supervisor.process_query(
            query="What was NVDA quarterly revenue in the database?",
            thread_id=thread_id
        )
        self.assertIsNone(state_turn1.resolved_query)

        # Turn 2 (Follow-up)
        state_turn2 = self.supervisor.process_query(
            query="What was their gross margin?",
            thread_id=thread_id
        )
        self.assertIsNotNone(state_turn2.resolved_query)
        self.assertIn("NVDA", state_turn2.resolved_query.upper())
        self.assertEqual(len(state_turn2.conversation_history), 2)
        self.assertEqual(state_turn2.retrieval_mode, "hybrid")

    def test_api_multi_turn_endpoint(self):
        """Prove /api/query returns resolved_query and retrieval_mode across conversation turns."""
        thread_id = f"api_test_{uuid4()}"

        # Turn 1
        resp1 = self.client.post("/api/query", json={
            "query": "Show NVDA recent quarterly revenue metrics from database.",
            "thread_id": thread_id,
            "retrieval_mode": "hybrid"
        })
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.json()
        self.assertEqual(data1["retrieval_mode"], "hybrid")

        # Turn 2
        resp2 = self.client.post("/api/query", json={
            "query": "What drove their growth?",
            "thread_id": thread_id,
            "retrieval_mode": "hybrid"
        })
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertIsNotNone(data2.get("resolved_query"))
        self.assertIn("NVDA", data2["resolved_query"].upper())
        self.assertEqual(data2["retrieval_mode"], "hybrid")


if __name__ == "__main__":
    unittest.main()
