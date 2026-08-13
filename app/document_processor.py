import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Handles parsing PDF documents and splitting them into structured chunks for vector storage."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path

    def process(self) -> list[dict]:
        """Reads the PDF file page by page, extracts text, and breaks it down into chunks."""
        print(f"Reading file: {self.file_path}")
        
        pdf_reader = PdfReader(self.file_path)
        processed_document_chunks = []
        chunk_id_counter = 0
        
        for page_index, page_object in enumerate(pdf_reader.pages):
            extracted_page_text = page_object.extract_text()
            if extracted_page_text:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                split_text_segments = text_splitter.split_text(extracted_page_text)
                
                for text_segment in split_text_segments:
                    processed_document_chunks.append({
                        "text": text_segment,
                        "page": page_index + 1,
                        "source": os.path.basename(self.file_path),
                        "chunk_id": chunk_id_counter
                    })
                    chunk_id_counter += 1
                
        print(f"Generated {len(processed_document_chunks)} structured chunks.")
        return processed_document_chunks

if __name__ == "__main__":
    print("Document processor module ready.")
