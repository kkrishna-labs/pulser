"""LLM client for text generation.

Wraps OpenAI-compatible APIs (NVIDIA NIM, OpenAI, Ollama, etc.)
with lazy initialization and error handling.
"""

from __future__ import annotations

from typing import Any

from pulser.config import get_llm_config
from pulser.utils.logging import get_logger

log = get_logger(__name__)

_client: Any = None


def _get_client():
    global _client
    if _client is None:
        cfg = get_llm_config()
        if not cfg.enabled:
            log.warning("LLM not configured - set LLM_API_KEY to enable")
            return None
        from openai import OpenAI
        _client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    return _client


def generate(prompt: str, system: str = "", max_tokens: int | None = None) -> str | None:
    """Generate text using the configured LLM.

    Args:
        prompt: User prompt.
        system: Optional system prompt.
        max_tokens: Override default max_tokens.

    Returns:
        Generated text or None if LLM is unavailable.
    """
    client = _get_client()
    if client is None:
        return None

    cfg = get_llm_config()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=cfg.model,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=max_tokens or cfg.max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error("LLM generation failed: %s", e)
        return None


def summarize(texts: list[str], query: str = "") -> str | None:
    """Generate a summary from multiple texts."""
    combined = "\n\n---\n\n".join(texts[:10])
    if len(combined) > 8000:
        combined = combined[:8000]

    prompt = (
        f"Summarize the following content"
        + (f" in relation to: {query}" if query else "")
        + f"\n\nContent:\n{combined}"
    )

    system = (
        "You are a research assistant. Provide a concise, factual summary "
        "covering key points, trends, and notable findings. Use bullet points "
        "where appropriate."
    )

    return generate(prompt, system=system, max_tokens=1024)


def answer_question(question: str, context: str) -> str | None:
    """Answer a question given retrieved context."""
    prompt = (
        f"Based on the following context, answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    system = (
        "You are a research assistant. Answer questions accurately based on "
        "the provided context. If the context doesn't contain enough information, "
        "say so. Cite specific details from the context."
    )

    return generate(prompt, system=system, max_tokens=512)
