"""NLP engine - topics, entities, relationships, and signal scoring.

Combines lightweight pattern-based extraction with sentiment to produce
fully enriched documents ready for vector storage and RAG.
"""

from __future__ import annotations

import re
from collections import Counter

from pulser.schema import (
    EnrichedDocument,
    ProcessedDocument,
    Sentiment,
)
from pulser.nlp.sentiment import enrich_with_sentiment, batch_sentiment
from pulser.utils.logging import get_logger
from pulser.utils.text import STOPWORDS, extract_keyphrases

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Topic taxonomy
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "technology": [
        "software", "hardware", "ai", "machine learning", "cloud", "data",
        "api", "algorithm", "computing", "digital", "cyber", "blockchain",
        "chip", "processor", "quantum", "robotics", "automation",
    ],
    "business": [
        "revenue", "profit", "market", "stock", "investment", "startup",
        "acquisition", "partnership", "growth", "forecast", "earnings",
        "valuation", "funding", "ipo", "merger", "strategy",
    ],
    "politics": [
        "government", "election", "policy", "regulation", "law", "congress",
        "senate", "president", "vote", "campaign", "legislation", "diplomacy",
        "sanctions", "trade war", "geopolitics",
    ],
    "health": [
        "health", "medical", "drug", "fda", "vaccine", "clinical",
        "patient", "treatment", "disease", "hospital", "mental health",
        "wellness", "therapy", "diagnosis", "pandemic",
    ],
    "science": [
        "research", "study", "discovery", "physics", "chemistry", "biology",
        "space", "nasa", "experiment", "journal", "peer review", "climate",
        "environment", "energy", "sustainability",
    ],
    "society": [
        "education", "culture", "social", "community", "equality", "rights",
        "privacy", "security", "ethics", "climate change", "housing",
        "inequality", "diversity", "activism",
    ],
}

# Entity patterns (capitalized multi-word names)
ENTITY_PATTERNS = [
    r"\b([A-Z][a-z]+ (?:[A-Z][a-z]+ )*(?:Inc|Corp|LLC|Ltd|Co|Group|Technologies|Labs|Systems))\b",
    r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b",  # Two-word proper names
    r"\b([A-Z]{2,})\b",  # Acronyms
]

# Relationship patterns
RELATIONSHIP_PATTERNS = [
    (r"(\w+) acquires (\w+)", "acquires"),
    (r"(\w+) acquires (\w+)", "acquired_by"),
    (r"(\w+) partners with (\w+)", "partners_with"),
    (r"(\w+) launches (\w+)", "launches"),
    (r"(\w+) faces (\w+)", "faces"),
    (r"(\w+) invests in (\w+)", "invests_in"),
    (r"(\w+) competes with (\w+)", "competes_with"),
]


def extract_topics(text: str) -> list[str]:
    """Extract topic labels based on keyword matching."""
    lower = text.lower()
    scores: dict[str, int] = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in lower)
        if count > 0:
            scores[topic] = count

    if not scores:
        return ["general"]

    # Return topics with score > 0, sorted by relevance
    return [t for t, _ in sorted(scores.items(), key=lambda x: -x[1])]


def extract_entities(text: str) -> list[str]:
    """Extract named entities using pattern matching."""
    entities: list[str] = []
    for pattern in ENTITY_PATTERNS:
        matches = re.findall(pattern, text)
        entities.extend(matches)

    # Dedup while preserving order
    seen = set()
    unique = []
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            unique.append(e)

    return unique[:15]  # Cap at 15


def extract_relationships(text: str) -> list[str]:
    """Extract entity relationships using pattern matching."""
    relationships: list[str] = []
    for pattern, rel_type in RELATIONSHIP_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for m in matches:
            relationships.append(f"{m.group(1)} --[{rel_type}]--> {m.group(2)}")
    return relationships[:10]


def signal_score(doc: EnrichedDocument) -> float:
    """Compute a 0-100 signal quality score.

    Factors: content richness, source diversity, sentiment confidence,
    topic coverage, entity density, relationship presence.
    """
    score = 50.0
    text = doc.clean_content
    wc = doc.word_count

    # Word count
    if wc > 200:
        score += 10
    elif wc > 100:
        score += 5
    elif wc < 20:
        score -= 15

    # URL validity
    if doc.url and "http" in doc.url:
        score += 5

    # Author present
    if doc.author:
        score += 3

    # Topic richness
    score += min(len(doc.topics) * 3, 12)

    # Entity density
    score += min(len(doc.entities) * 2, 10)

    # Relationships
    score += min(len(doc.relationships) * 3, 9)

    # Sentiment confidence
    score += doc.sentiment_score * 5

    # Keyphrase count
    score += min(len(doc.keyphrases) * 1, 5)

    return max(0.0, min(100.0, round(score, 1)))


# ---------------------------------------------------------------------------
# Main enrichment pipeline
# ---------------------------------------------------------------------------

def enrich_documents(
    docs: list[ProcessedDocument],
    use_batch: bool = True,
) -> list[EnrichedDocument]:
    """Enrich processed documents with full NLP analysis.

    Steps:
    1. Sentiment analysis
    2. Topic extraction
    3. Entity extraction
    4. Relationship extraction
    5. Keyphrase extraction
    6. Signal scoring
    """
    log.info("Enriching %d documents with NLP analysis", len(docs))

    enriched: list[EnrichedDocument] = []

    if use_batch and len(docs) > 5:
        # Batch sentiment analysis
        texts = [d.clean_content or d.title for d in docs]
        sentiments = batch_sentiment(texts)

        for doc, (sentiment, sent_score) in zip(docs, sentiments):
            text = doc.clean_content or doc.title
            ed = EnrichedDocument(
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
                sentiment_score=sent_score,
                topics=extract_topics(text),
                keyphrases=extract_keyphrases(text, top_n=8),
                entities=extract_entities(text),
                relationships=extract_relationships(text),
                metadata=doc.metadata,
            )
            ed.signal_score = signal_score(ed)
            enriched.append(ed)
    else:
        for doc in docs:
            ed = enrich_with_sentiment(doc)
            text = doc.clean_content or doc.title
            ed.topics = extract_topics(text)
            ed.keyphrases = extract_keyphrases(text, top_n=8)
            ed.entities = extract_entities(text)
            ed.relationships = extract_relationships(text)
            ed.signal_score = signal_score(ed)
            enriched.append(ed)

    log.info(
        "Enrichment complete: %d docs with avg signal %.1f",
        len(enriched),
        sum(e.signal_score for e in enriched) / max(len(enriched), 1),
    )
    return enriched
