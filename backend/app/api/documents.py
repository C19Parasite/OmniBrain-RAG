from fastapi import APIRouter
from app.vector_store import get_documents

router = APIRouter()

@router.get("/documents")
def list_documents():
    return get_documents()