from fastapi import APIRouter

router = APIRouter()

@router.get("/info")
def info():
    return {
        "project": "OmniBrain-RAG",
        "backend": "FastAPI",
        "version": "1.0"
    }