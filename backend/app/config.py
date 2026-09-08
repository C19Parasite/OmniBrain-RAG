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

    # Langfuse observability (leave keys empty to disable telemetry)
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_BASE_URL: str = "https://cloud.langfuse.com"
    LANGFUSE_ENVIRONMENT: str = "development"

    # Optional enforcement after the existing grounding evaluator runs.
    BLOCK_UNGROUNDED_MEMOS: bool = False
    MIN_GROUNDING_SCORE: float = 0.80

    # RAG / Model Hyperparameters
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.35
    TEMPERATURE: float = 0.2

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
