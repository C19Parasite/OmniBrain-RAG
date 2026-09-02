"""
OmniBrain Studio ? Backend Application Entrypoint.
Exposes the FastAPI application instance for Uvicorn and test clients.
"""
from backend.app.main import app

__all__ = ["app"]
