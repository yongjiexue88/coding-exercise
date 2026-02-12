"""Gemini LLM service with streaming support."""

from __future__ import annotations

from typing import AsyncIterator
import google.generativeai as genai
from config import settings


SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context documents.

RULES:
1. Answer the question using ONLY the information from the provided context.
2. If the context doesn't contain enough information to answer, say so honestly.
3. Keep the same tone as the source material — if it's humorous, be humorous in your response.
4. When referencing information, mention which source document it comes from.
5. Be concise but thorough.
"""


def _build_prompt(query: str, context_docs: list[dict]) -> str:
    """Build the augmented prompt with retrieved context."""
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        source = doc.get("source", "Unknown")
        content = doc.get("content", "")
        context_parts.append(f"--- Document {i} (Source: {source}) ---\n{content}")

    context_text = "\n\n".join(context_parts)

    return f"""Context Documents:
{context_text}

User Question: {query}

Please answer the question based on the context documents above."""


class LLMService:
    """LLM service using Google Gemini."""

    def __init__(self):
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)

    @property
    def current_model_name(self) -> str:
        return settings.gemini_model

    async def generate(self, query: str, context_docs: list[dict]) -> str:
        """Generate a complete response (non-streaming)."""
        prompt = _build_prompt(query, context_docs)
        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=SYSTEM_PROMPT,
        )
        response = model.generate_content(prompt)
        return response.text

    async def generate_stream(
        self, query: str, context_docs: list[dict]
    ) -> AsyncIterator[str]:
        """Generate a streaming response, yielding chunks of text."""
        prompt = _build_prompt(query, context_docs)
        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=SYSTEM_PROMPT,
        )
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
