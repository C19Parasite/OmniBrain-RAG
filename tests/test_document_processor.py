import unittest
import os
from app.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):

    def setUp(self):
        """Set up temporary test paths."""
        self.invalid_file_path = "data/non_existent_file.pdf"

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

if __name__ == "__main__":
    unittest.main()
