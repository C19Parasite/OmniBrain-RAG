import unittest
import shutil
import os
from app.vector_store import VectorStoreManager

class TestVectorStoreManager(unittest.TestCase):

    def setUp(self):
        """Set up a temporary ChromaDB directory for testing."""
        self.test_db_path = "data/test_chroma_db"
        self.vector_manager = VectorStoreManager(collection_name="test_collection", db_path=self.test_db_path)

    def tearDown(self):
        """Clean up test database files."""
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

    def test_add_and_search_chunks(self):
        """Test adding sample chunks and retrieving them via similarity search."""
        sample_chunks = [
            {"chunk_id": 0, "page": 1, "text": "OmniBrain RAG handles PDF document processing.", "source": "doc1.pdf"},
            {"chunk_id": 1, "page": 2, "text": "Python unit testing guarantees software reliability.", "source": "doc2.pdf"}
        ]
        self.vector_manager.add_chunks(sample_chunks)
        
        results = self.vector_manager.search_similar("How does OmniBrain process PDFs?", top_k=1)
        self.assertTrue(len(results["documents"][0]) > 0)
        self.assertIn("OmniBrain", results["documents"][0][0])

if __name__ == "__main__":
    unittest.main()
