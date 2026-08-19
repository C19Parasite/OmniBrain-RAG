import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.document_processor import DocumentProcessor
from app.vector_store import VectorStoreManager
from app.rag_chain import RAGPipeline

app = FastAPI(
    title="OmniBrain RAG API",
    description="High-performance multi-modal RAG retrieval architecture",
    version="1.0.0"
)

# Initialize core persistent services
vdb = VectorStoreManager()
rag_pipeline = RAGPipeline(vector_store=vdb)

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    query: Optional[str] = None
    question: Optional[str] = None
    top_k: Optional[int] = 4

class QueryResponse(BaseModel):
    query: Optional[str] = None
    question: Optional[str] = None
    answer: str
    context: Optional[str] = None
    sources: List[dict] = []

@app.get("/")
def read_root():
    return {"status": "online", "message": "OmniBrain-RAG API is operational."}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg")):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please upload PDF, TXT, MD, or Image files."
        )

    file_path = os.path.join(UPLOAD_DIR, os.path.basename(file.filename))
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Initialize processor with file_path
        doc_processor = DocumentProcessor(file_path)
        chunks = doc_processor.process() if hasattr(doc_processor, 'process') else doc_processor.process_file()
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No extractable text found in document.")

        # Ingest chunks into vector store
        vdb.add_chunks(chunks)

        return {
            "filename": file.filename,
            "status": "success",
            "chunks_ingested": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/query")
def run_query(request: QueryRequest):
    try:
        # Support both 'query' and 'question' fields for frontend/backend contract alignment
        query_text = getattr(request, "query", None) or getattr(request, "question", None)
        top_k = getattr(request, "top_k", 3)
        
        response = rag_pipeline.generate_response(query_text, top_k=top_k)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")