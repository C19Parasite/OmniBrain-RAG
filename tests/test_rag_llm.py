import unittest
from app.vector_store import VectorStoreManager
from app.rag_chain import RAGPipeline

class TestRAGLLMPipeline(unittest.TestCase):
    def test_rag_llm_pipeline(self):
        vdb = VectorStoreManager()
        vdb.add_chunks([{
            "chunk_id": "1",
            "text": "OmniBrain is a high-performance multi-modal RAG retrieval architecture.",
            "source": "data/sample.pdf",
            "page": 1
        }])

        pipeline = RAGPipeline(vector_store=vdb)
        response = pipeline.generate_response("What is OmniBrain?")

        self.assertIn("query", response)
        self.assertIn("context", response)
        self.assertIn("sources", response)

if __name__ == "__main__":
    unittest.main()
