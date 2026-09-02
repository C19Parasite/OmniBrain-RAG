import unittest
from backend.app.guardrails.evaluator import GuardrailEvaluator

class TestGuardrails(unittest.TestCase):
    def test_guardrail_evaluation(self):
        evaluator = GuardrailEvaluator()
        memo = "? NVDA revenue reached 30.0B in Q2 FY2025 [Source: NVDA-10Q.pdf, p.1]."
        search_results = [{
            "id": "c1",
            "source_document": "NVDA-10Q.pdf",
            "page_number": 1,
            "text": "NVIDIA reported record revenue of 30.0 billion for the second quarter."
        }]
        report = evaluator.evaluate_memo(memo, [], search_results)
        self.assertGreaterEqual(report["overall_score"], 0.8)

if __name__ == "__main__":
    unittest.main()
