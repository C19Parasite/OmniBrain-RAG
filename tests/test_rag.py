import unittest
from pathlib import Path
from backend.app.rag.embeddings import TextEmbeddings
from backend.app.rag.document_parser import FinancialDocumentParser

class TestRAG(unittest.TestCase):
    def test_embeddings(self):
        embedder = TextEmbeddings()
        vecs = embedder.embed_documents(["Revenue grew 25% year-over-year.", "Balance sheet shows liquidity."])
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), 384)

    def test_document_chunking(self):
        parser = FinancialDocumentParser()
        temp_file = Path("data/uploads/temp_test.md")
        temp_file.write_text("# Section 1: Growth\n\nNVIDIA reported record quarterly revenue.\n\n## Section 2: Outlook\n\nGross margin expected at 75%.", encoding="utf-8")
        try:
            chunks = parser.parse_file(temp_file)
            self.assertGreaterEqual(len(chunks), 2)
        finally:
            temp_file.unlink(missing_ok=True)

if __name__ == "__main__":
    unittest.main()
