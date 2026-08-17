"""Data models for Pulser.

All pipeline stages speak the same vocabulary through these types.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any


class SourceCategory(str, Enum):
    NEWS = "news"
    SOCIAL = "social"
    FORUM = "forum"
    REVIEW = "review"
    BLOG = "blog"
    OTHER = "other"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class RawDocument:
    """A single piece of content scraped from the web."""

    id: str = ""
    url: str = ""
    title: str = ""
    content: str = ""
    author: str = ""
    source: str = ""
    category: SourceCategory = SourceCategory.OTHER
    timestamp: str = ""
    scraped_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            raw = f"{self.url}:{self.title}:{self.content[:200]}"
            self.id = hashlib.md5(raw.encode()).hexdigest()[:12]
        if not self.scraped_at:
            self.scraped_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


@dataclass
class ProcessedDocument:
    """A cleaned, scored document ready for NLP enrichment."""

    id: str = ""
    url: str = ""
    title: str = ""
    content: str = ""
    clean_content: str = ""
    author: str = ""
    source: str = ""
    category: SourceCategory = SourceCategory.OTHER
    timestamp: str = ""
    quality_score: float = 0.0
    word_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


@dataclass
class EnrichedDocument:
    """A fully analyzed document with NLP annotations."""

    id: str = ""
    url: str = ""
    title: str = ""
    content: str = ""
    clean_content: str = ""
    author: str = ""
    source: str = ""
    category: SourceCategory = SourceCategory.OTHER
    timestamp: str = ""
    quality_score: float = 0.0
    word_count: int = 0

    # NLP fields
    sentiment: Sentiment = Sentiment.NEUTRAL
    sentiment_score: float = 0.0
    topics: list[str] = field(default_factory=list)
    keyphrases: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    signal_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["sentiment"] = self.sentiment.value
        return d


@dataclass
class SearchTerms:
    """Generated search terms for a query."""

    query: str = ""
    terms: list[str] = field(default_factory=list)
    source: str = "fallback"


@dataclass
class PipelineResult:
    """Aggregated result of a full pipeline run."""

    query: str = ""
    raw_count: int = 0
    processed_count: int = 0
    enriched_count: int = 0
    sources_scraped: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    output_path: str = ""
