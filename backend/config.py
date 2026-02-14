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

    # Evaluation / Observability
    eval_judge_model: str = "gemini-2.0-flash"
    eval_judge_prompt_version: str = "v1"
    eval_judge_temperature: float = 0.0
    eval_feedback_enabled: bool = True
    eval_feedback_min_top_k: int = 5
    eval_feedback_groundedness_threshold: float = 0.85
    eval_feedback_hallucination_threshold: float = 0.08

    # Pricing snapshot as of February 14, 2026 (USD per 1M tokens).
    # Update these values when provider pricing changes.
    price_input_per_1m_tokens: float = 0.30
    price_output_per_1m_tokens: float = 2.50

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
