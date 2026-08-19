import unittest
import os
from app.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):

    def setUp(self):
        """Set up temporary test paths."""
        self.invalid_file_path = "data/non_existent_file.pdf"
        self.processor = DocumentProcessor("data/sample.pdf")

    def test_file_not_found(self):
        """Test that non-existent files raise FileNotFoundError."""
        processor = DocumentProcessor(self.invalid_file_path)
        with self.assertRaises(FileNotFoundError):
            processor.extract_text()

    def test_table_extraction_invalid_file(self):
        """Test table extraction handles missing files gracefully."""
        processor = DocumentProcessor(self.invalid_file_path)
        with self.assertRaises(FileNotFoundError):
            processor.extract_tables()

    def test_chunking_structure(self):
        """Test chunking outputs expected metadata format."""
        sample_pages = [
            {"page": 1, "content": "This is a sample document text used to test chunking functionality."}
        ]
        chunks = self.processor.create_chunks(sample_pages, chunk_size=20, overlap=5)
        self.assertTrue(len(chunks) > 0)
        self.assertIn("chunk_id", chunks[0])
        self.assertIn("page", chunks[0])
        self.assertIn("text", chunks[0])
        self.assertIn("source", chunks[0])

if __name__ == "__main__":
    unittest.main()
