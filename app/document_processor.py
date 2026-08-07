import os
import pymupdf
import pdfplumber
import pytesseract
from PIL import Image

class DocumentProcessor:
    """
    Core Document Processing class for OmniBrain-RAG.
    Handles text extraction, table parsing, OCR processing, and chunking.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def extract_text(self) -> list[dict]:
        """Extract plain text page-by-page using PyMuPDF."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        doc = pymupdf.open(self.file_path)
        extracted_pages = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()
            extracted_pages.append({
                "page": page_num + 1,
                "content": text
            })
        return extracted_pages

    def extract_tables(self) -> list[dict]:
        """Extract structured tables page-by-page using pdfplumber."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        tables_data = []
        with pdfplumber.open(self.file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for idx, table in enumerate(tables):
                    tables_data.append({
                        "page": page_num + 1,
                        "table_id": idx + 1,
                        "data": table
                    })
        return tables_data

    def perform_ocr_on_image(self, image_path: str) -> str:
        """Perform OCR extraction on image files (PNG, JPG)."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
            
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()

    def create_chunks(self, text_pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
        """
        Splits extracted page text into smaller chunk units with metadata for RAG embeddings.
        """
        chunks = []
        chunk_id = 0
        
        for page_data in text_pages:
            page_num = page_data["page"]
            content = page_data["content"]
            
            start = 0
            while start < len(content):
                end = start + chunk_size
                chunk_text = content[start:end]
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_num,
                    "text": chunk_text,
                    "source": os.path.basename(self.file_path)
                })
                
                chunk_id += 1
                start += (chunk_size - overlap)
                
        return chunks

if __name__ == "__main__":
    print("Document Processor module with chunking updated successfully.")
