from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Configuration settings for OmniBrain application."""
    
    # App Settings
    APP_NAME: str = "OmniBrain"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DB_PATH: Path = DATA_DIR / "financial_data.db"
    VECTOR_STORE_DIR: Path = DATA_DIR / "vector_store"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"

    # API Keys
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # RAG / Model Hyperparameters
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.35
    TEMPERATURE: float = 0.2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
