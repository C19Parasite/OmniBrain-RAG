"""
Test Suite for Institutional Research Memo Export (Markdown & Print-Ready PDF/HTML).
"""
import unittest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.models.schemas import ExportMemoRequest
from backend.app.export import generate_memo_markdown, generate_memo_html

class TestMemoExport(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_generate_memo_markdown(self):
        """Test Markdown export format structure."""
        req = ExportMemoRequest(
            query="Analyze NVIDIA Q3 revenue segments",
            memo_markdown="NVIDIA reported $30.77B in Data Center revenue [Visual: tech_sector_performance.png, p.1].",
            guardrail_report={"overall_score": 1.0, "status": "PASSED", "grounded_claims": 1, "total_claims": 1, "ungrounded_claims": 0},
            citations=[{
                "citation_id": 1,
                "source_type": "visual",
                "source_name": "tech_sector_performance.png",
                "page_number": 1,
                "snippet": "Data Center Compute: $30.77B"
            }],
            format="markdown"
        )
        md = generate_memo_markdown(req)
        self.assertIn("# Institutional Research Memorandum", md)
        self.assertIn("NVIDIA reported $30.77B", md)
        self.assertIn("Factual Grounding & Hallucination Audit Report", md)
        self.assertIn("Verified Evidence Catalog (Citations)", md)
        self.assertIn("tech_sector_performance.png", md)

    def test_generate_memo_html(self):
        """Test Print-Ready HTML export format structure."""
        req = ExportMemoRequest(
            query="Analyze NVIDIA Q3 revenue segments",
            memo_markdown="NVIDIA reported $30.77B in Data Center revenue.",
            guardrail_report={"overall_score": 1.0, "status": "PASSED", "grounded_claims": 1, "total_claims": 1, "ungrounded_claims": 0},
            citations=[{
                "citation_id": 1,
                "source_type": "visual",
                "source_name": "tech_sector_performance.png",
                "page_number": 1,
                "snippet": "Data Center Compute: $30.77B"
            }],
            format="pdf_html"
        )
        html_doc = generate_memo_html(req)
        self.assertIn("<!DOCTYPE html>", html_doc)
        self.assertIn("@media print", html_doc)
        self.assertIn("OmniBrain Studio", html_doc)
        self.assertIn("window.print()", html_doc)
        self.assertIn("tech_sector_performance.png", html_doc)

    def test_export_api_endpoints(self):
        """Test /api/export/memo endpoint for both markdown and html."""
        payload = {
            "query": "Sample test query",
            "memo_markdown": "Test memo text [Source: doc.md, p.1].",
            "guardrail_report": {"overall_score": 1.0, "status": "PASSED"},
            "citations": [{"citation_id": 1, "source_type": "text", "source_name": "doc.md", "snippet": "Test snippet"}],
            "format": "markdown"
        }
        res_md = self.client.post("/api/export/memo", json=payload)
        self.assertEqual(res_md.status_code, 200)
        self.assertIn("text/markdown", res_md.headers.get("content-type", ""))

        payload["format"] = "pdf_html"
        res_html = self.client.post("/api/export/memo", json=payload)
        self.assertEqual(res_html.status_code, 200)
        self.assertIn("text/html", res_html.headers.get("content-type", ""))
        self.assertIn("Institutional Print Export", res_html.text)

if __name__ == "__main__":
    unittest.main()
