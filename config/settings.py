import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

APP_NAME = os.getenv("APP_NAME", "OmniBrain Studio")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
SQLITE_DB_PATH = BASE_DIR / "data" / "financial_data.db"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
SAMPLE_DATA_DIR = BASE_DIR / "data" / "sample_data"
FRONTEND_DIR = BASE_DIR / "frontend"

TOP_K = int(os.getenv("TOP_K_CHUNKS", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
