from fastapi import APIRouter, UploadFile, File
import os
from app.vector_store import add_document

router = APIRouter()

UPLOAD_DIR = "backend/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    content = await file.read()

    with open(file_path, "wb") as buffer:
        buffer.write(content)

    text = content.decode("utf-8")

    add_document(file.filename, text)

    return {
        "filename": file.filename,
        "saved_to": file_path,
        "content_preview": text[:200]
    }