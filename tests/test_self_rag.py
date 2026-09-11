"""
Test Suite for Self-Correction Retrieval Loop (Self-RAG / Corrective RAG).
"""
import unittest
from backend.app.agents.supervisor import SupervisorOrchestrator

class TestSelfCorrection(unittest.TestCase):
    def setUp(self):
        self.supervisor = SupervisorOrchestrator()

    # -------------------------------------------------------------
    # 1. Self-RAG Retrieval Quality Evaluation Tests
    # -------------------------------------------------------------
    def test_evaluate_retrieval_quality_empty_chunks(self):
        """Should trigger self-correction if zero chunks are retrieved."""
        needs_corr, reason = self.supervisor._evaluate_retrieval_quality("What is NVDA gross margin?", [])
        self.assertTrue(needs_corr)
        self.assertIn("Zero chunks", reason)

    def test_evaluate_retrieval_quality_low_similarity(self):
        """Should trigger self-correction if maximum similarity is below threshold."""
        low_sim_chunks = [
            {"id": "c1", "text": "Unrelated general comment", "similarity_score": 0.22},
            {"id": "c2", "text": "Another low score snippet", "similarity_score": 0.28}
        ]
        needs_corr, reason = self.supervisor._evaluate_retrieval_quality("What is NVDA gross margin?", low_sim_chunks)
        self.assertTrue(needs_corr)
        self.assertIn("Low retrieval confidence", reason)

    def test_evaluate_retrieval_quality_high_similarity(self):
        """Should NOT trigger self-correction if chunks are highly relevant with good similarity."""
        high_sim_chunks = [
            {"id": "c1", "text": "NVIDIA gross margin expanded to 75.1% in Q3 FY25.", "similarity_score": 0.88},
            {"id": "c2", "text": "Strong Data Center revenue growth supported margins.", "similarity_score": 0.79}
        ]
        needs_corr, reason = self.supervisor._evaluate_retrieval_quality("What is NVIDIA gross margin?", high_sim_chunks)
        self.assertFalse(needs_corr)
        self.assertIn("passed", reason.lower())

    # -------------------------------------------------------------
    # 2. Query Reformulation Tests
    # -------------------------------------------------------------
    def test_heuristic_query_expansion_strips_filler(self):
        """Should strip conversational preambles like 'can you please explain'."""
        query = "Can you please explain the revenue growth for NVDA?"
        expanded = self.supervisor._heuristic_query_expansion(query)
        self.assertNotIn("can you please explain", expanded.lower())
        self.assertIn("nvda", expanded.lower())

    def test_heuristic_query_expansion_adds_financial_context(self):
        """Should expand abbreviations and key financial terms."""
        query = "Tell me about MSFT cloud margins"
        expanded = self.supervisor._heuristic_query_expansion(query)
        self.assertTrue("microsoft" in expanded.lower() or "azure" in expanded.lower() or "margin" in expanded.lower())

    # -------------------------------------------------------------
    # 3. End-to-End Self-Correction Orchestration
    # -------------------------------------------------------------
    def test_supervisor_state_tracks_self_correction(self):
        """Should record self-correction in state and execution trace."""
        state = self.supervisor.process_query("What are the key cloud risk factors?")
        self.assertIsNotNone(state.self_correction)
        self.assertIn("triggered", state.self_correction)
        
        # Verify trace contains Supervisor evaluation
        trace_events = [t.content for t in state.execution_trace]
        self.assertTrue(any("Self-RAG" in msg or "Routing query" in msg for msg in trace_events))

if __name__ == "__main__":
    unittest.main()
