"""Configuration settings for the RAG backend."""

from __future__ import annotations

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Database Configuration
    database_url: str = ""

    # RAG Configuration
    top_k: int = 3
    chunk_size: int = 300
    chunk_overlap: int = 50

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
