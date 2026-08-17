"""Central configuration for Pulser.

Loads from environment / .env and provides typed accessors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "output"))
RAW_DIR = OUTPUT_DIR / "raw"
PROCESSED_DIR = OUTPUT_DIR / "processed"
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", OUTPUT_DIR / "chroma_db"))

for _d in (OUTPUT_DIR, RAW_DIR, PROCESSED_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
REQUEST_DELAY_MIN = float(os.getenv("REQUEST_DELAY_MIN", "0.4"))
REQUEST_DELAY_MAX = float(os.getenv("REQUEST_DELAY_MAX", "1.5"))
USER_AGENT_ROTATION = os.getenv("USER_AGENT_ROTATION", "true").lower() == "true"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
]

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.1-70b-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = field(default_factory=lambda: LLM_API_KEY)
    base_url: str = field(default_factory=lambda: LLM_BASE_URL)
    model: str = field(default_factory=lambda: LLM_MODEL)
    temperature: float = field(default_factory=lambda: LLM_TEMPERATURE)
    max_tokens: int = field(default_factory=lambda: LLM_MAX_TOKENS)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


# ---------------------------------------------------------------------------
# NLP / Embeddings
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "512"))

SENTIMENT_LABELS = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def get_llm_config() -> LLMConfig:
    return LLMConfig()
