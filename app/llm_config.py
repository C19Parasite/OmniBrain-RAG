"""
LLM Configuration - Integrated with your existing LLMManager
"""

from typing import Literal
import os
from dotenv import load_dotenv

load_dotenv()


class LLMConfig:
    """Centralized LLM configuration for your existing LLMManager"""
    
    # ===== Groq Models (Fast, Free) =====
    GROQ_MODELS = {
        "openai/gpt-oss-20b": {
            "name": "openai/gpt-oss-20b",
            "cost": "free",
            "speed": "very-fast",
            "quality": "high",
            "recommended": True
        },
        "llama2-70b-4096": {
            "name": "llama2-70b-4096",
            "cost": "free",
            "speed": "fast",
            "quality": "high",
            "recommended": False
        },
    }
    
    # ===== OpenAI Models (Paid, High Quality) =====
    OPENAI_MODELS = {
        "gpt-4o": {
            "name": "gpt-4o",
            "cost": "~$0.015/1K tokens",
            "speed": "medium",
            "quality": "excellent",
            "recommended": True
        },
        "gpt-4-turbo": {
            "name": "gpt-4-turbo",
            "cost": "~$0.01/1K tokens",
            "speed": "medium",
            "quality": "very-high",
            "recommended": False
        },
        "gpt-3.5-turbo": {
            "name": "gpt-3.5-turbo",
            "cost": "~$0.001/1K tokens",
            "speed": "fast",
            "quality": "medium",
            "recommended": False
        },
    }
    
    # Default configuration
    DEFAULT_PROVIDER: Literal["groq", "openai"] = "groq"
    DEFAULT_MODEL = "openai/gpt-oss-20b"
    
    @staticmethod
    def get_config(provider: Literal["groq", "openai"] = None) -> dict:
        """Get config for a specific provider"""
        provider = provider or LLMConfig.DEFAULT_PROVIDER
        
        if provider == "openai":
            return {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key_env": "OPENAI_API_KEY",
                "available_models": list(LLMConfig.OPENAI_MODELS.keys()),
                "description": "OpenAI - Highest quality, slower, paid",
                "base_url": None
            }
        else:  # groq
            return {
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "api_key_env": "GROQ_API_KEY",
                "available_models": list(LLMConfig.GROQ_MODELS.keys()),
                "description": "Groq - Very fast, free, good quality",
                "base_url": "https://api.groq.com/openai/v1"
            }
    
    @staticmethod
    def validate_api_keys() -> dict:
        """Check if API keys are configured"""
        keys = {
            "openai": os.getenv("OPENAI_API_KEY") is not None,
            "groq": os.getenv("GROQ_API_KEY") is not None,
        }
        return keys
    
    @staticmethod
    def get_available_providers() -> list:
        """Get list of available providers based on API keys"""
        keys = LLMConfig.validate_api_keys()
        return [provider for provider, has_key in keys.items() if has_key]
    
    @staticmethod
    def get_recommended_config() -> dict:
        """Get recommended config based on available providers"""
        available = LLMConfig.get_available_providers()
        
        # Prefer Groq (free, fast)
        if "groq" in available:
            return LLMConfig.get_config("groq")
        # Fall back to OpenAI
        elif "openai" in available:
            return LLMConfig.get_config("openai")
        else:
            raise ValueError("No API keys configured. Set GROQ_API_KEY or OPENAI_API_KEY")


