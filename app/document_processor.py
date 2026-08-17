import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Handles parsing documents (PDF, TXT, MD) and splitting them into structured chunks."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def process(self) -> list[dict]:
        """Dispatches processing based on file extension."""
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == ".pdf":
            return self._process_pdf()
        elif ext in [".txt", ".md"]:
            return self._process_text()
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _process_pdf(self) -> list[dict]:
        print(f"Reading PDF file: {self.file_path}")
        pdf_reader = PdfReader(self.file_path)
        chunks = []
        chunk_id = 0

        for page_idx, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                segments = splitter.split_text(text)
                for segment in segments:
                    chunks.append({
                        "text": segment,
                        "page": page_idx + 1,
                        "source": os.path.basename(self.file_path),
                        "chunk_id": chunk_id
                    })
                    chunk_id += 1
        return chunks

    def _process_text(self) -> list[dict]:
        print(f"Reading text file: {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        chunks = []
        chunk_id = 0
        if text:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            segments = splitter.split_text(text)
            for segment in segments:
                chunks.append({
                    "text": segment,
                    "page": 1,
                    "source": os.path.basename(self.file_path),
                    "chunk_id": chunk_id
                })
                chunk_id += 1
        return chunks

    @classmethod
    def process_file(cls, file_path: str) -> list[dict]:
        return cls(file_path).process()

if __name__ == "__main__":
    print("Document processor module ready.")