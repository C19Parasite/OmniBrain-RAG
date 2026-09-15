from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    """Configuration settings for OmniBrain application."""
    
    # App Settings
    APP_NAME: str = "OmniBrain"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Paths
    BASE_DIR: Path = PROJECT_ROOT
    DATA_DIR: Path = BASE_DIR / "data"
    DB_PATH: Path = DATA_DIR / "financial_data.db"
    VECTOR_STORE_DIR: Path = BASE_DIR / "chroma_db"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    SAMPLE_DATA_DIR: Path = DATA_DIR / "sample_data"

    # API Keys
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    EMBEDDING_PROVIDER: str = "gemini"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    LOCAL_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Langfuse observability (leave keys empty to disable telemetry)
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_BASE_URL: str = "https://cloud.langfuse.com"
    LANGFUSE_ENVIRONMENT: str = "development"

    # Optional enforcement after the existing grounding evaluator runs.
    BLOCK_UNGROUNDED_MEMOS: bool = False
    MIN_GROUNDING_SCORE: float = 0.80

    # RAG / Model Hyperparameters
    TOP_K: int = 10
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    # Calibrated for normalized Gemini retrieval embeddings; lower scores are
    # broad topical associations rather than sufficiently grounded evidence.
    SIMILARITY_THRESHOLD: float = 0.75
    TEMPERATURE: float = 0.2

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
