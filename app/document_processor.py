import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def process(self) -> list[dict]:
        print(f"Reading file: {self.file_path}")
        
        # 1. Read PDF pages and extract text
        reader = PdfReader(self.file_path)
        chunks = []
        chunk_id_counter = 0
        
        for page_num, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted:
                # Split text page by page or use text splitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                page_chunks = splitter.split_text(extracted)
                
                for chunk_text in page_chunks:
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num + 1,
                        "source": os.path.basename(self.file_path),
                        "chunk_id": chunk_id_counter
                    })
                    chunk_id_counter += 1
                
        print(f"Generated {len(chunks)} structured chunks.")
        return chunks

if __name__ == "__main__":
    print("Document processor module ready.")
