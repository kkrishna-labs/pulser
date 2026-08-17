"""Sentiment analysis using RoBERTa.

Wraps cardiffnlp/twitter-roberta-base-sentiment-latest with lazy loading
and batch processing.
"""

from __future__ import annotations

from typing import Any

from pulser.schema import Sentiment, ProcessedDocument, EnrichedDocument
from pulser.config import SENTIMENT_MODEL, BATCH_SIZE, SENTIMENT_LABELS
from pulser.utils.logging import get_logger

log = get_logger(__name__)

_pipeline: Any = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline as hf_pipeline

        log.info("Loading sentiment model: %s", SENTIMENT_MODEL)
        _pipeline = hf_pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            top_k=None,
        )
        log.info("Sentiment model loaded")
    return _pipeline


def analyze_sentiment(text: str) -> tuple[Sentiment, float]:
    """Classify sentiment of a single text.

    Returns (Sentiment, confidence).
    """
    pipe = _get_pipeline()
    # Truncate to 512 tokens (model limit)
    truncated = text[:1500]
    results = pipe(truncated)

    if not results or not results[0]:
        return Sentiment.NEUTRAL, 0.0

    # Get top prediction
    top = max(results[0], key=lambda x: x["score"])
    label = SENTIMENT_LABELS.get(top["label"], "neutral")

    return Sentiment(label), round(top["score"], 4)


def batch_sentiment(texts: list[str]) -> list[tuple[Sentiment, float]]:
    """Classify sentiment for a batch of texts."""
    pipe = _get_pipeline()
    truncated = [t[:1500] for t in texts]
    all_results = pipe(truncated, batch_size=BATCH_SIZE)

    results = []
    for preds in all_results:
        if not preds:
            results.append((Sentiment.NEUTRAL, 0.0))
            continue
        top = max(preds, key=lambda x: x["score"])
        label = SENTIMENT_LABELS.get(top["label"], "neutral")
        results.append((Sentiment(label), round(top["score"], 4)))

    return results


def enrich_with_sentiment(doc: ProcessedDocument) -> EnrichedDocument:
    """Add sentiment analysis to a single document."""
    text = doc.clean_content or doc.title
    sentiment, score = analyze_sentiment(text)

    return EnrichedDocument(
        id=doc.id,
        url=doc.url,
        title=doc.title,
        content=doc.content,
        clean_content=doc.clean_content,
        author=doc.author,
        source=doc.source,
        category=doc.category,
        timestamp=doc.timestamp,
        quality_score=doc.quality_score,
        word_count=doc.word_count,
        sentiment=sentiment,
        sentiment_score=score,
        metadata=doc.metadata,
    )
