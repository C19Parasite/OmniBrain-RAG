import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(file_path):
    print(f"Reading file: {file_path}")
    
    # 1. Read PDF pages and extract text
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
            
    print(f"Extracted {len(text)} characters total.")

    # 2. Split text into chunks for the vector store
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    
    print(f"Generated {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    print("Document processor module ready.")
