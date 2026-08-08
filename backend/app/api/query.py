from fastapi import APIRouter
from pydantic import BaseModel
from app.vector_store import get_documents

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

@router.post("/query")
def query_documents(data: QueryRequest):
    docs = get_documents()

    if not docs:
        return {"answer": "No documents uploaded yet."}

    question = data.question.lower()

    for doc in docs:
        if question in doc["text"].lower():
            return {
                "answer": f"Found in {doc['filename']}",
                "document": doc["text"]
            }

    return {"answer": "No matching document found."}