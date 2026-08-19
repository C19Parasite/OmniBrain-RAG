class DocumentProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def _process_pdf(self) -> list[dict]:
        # ... existing code ...

        def _process_text(self) -> list[dict]:
            print(f"Reading text file: {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        if not text or not text.strip():
            raise ValueError("The text/markdown document is empty.")

        chunks = []
        chunk_id = 0
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

        def _process_text(self) -> list[dict]:
            print(f"Reading text file: {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        if not text or not text.strip():
            raise ValueError("The text/markdown document is empty.")

        chunks = []
        chunk_id = 0
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