"""Preprocessing pipeline.

Cleans, deduplicates, and quality-scores raw documents.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from difflib import SequenceMatcher

from pulser.schema import RawDocument, ProcessedDocument
from pulser.config import PROCESSED_DIR
from pulser.utils.logging import get_logger
from pulser.utils.text import normalize_text, word_count

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Quality signals
# ---------------------------------------------------------------------------

TEMPLATE_PATTERNS = [
    r"click here",
    r"read more",
    r"subscribe now",
    r"sign up for",
    r"terms and conditions",
    r"cookie policy",
    r"privacy policy",
    r"all rights reserved",
]

SPAM_PATTERNS = [
    r"(buy now|limited time|act fast|don't miss)",
    r"(earn money|make money|free money)",
    r"(100% free|no cost|risk free)",
]


def _is_template(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in TEMPLATE_PATTERNS)


def _is_spam(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in SPAM_PATTERNS)


def quality_score(doc: RawDocument) -> float:
    """Score document quality from 0.0 (junk) to 1.0 (high quality)."""
    text = doc.content or doc.title
    wc = word_count(text)
    score = 0.5

    # Word count adjustment
    if wc < 10:
        score -= 0.3
    elif wc < 30:
        score -= 0.1
    elif wc > 100:
        score += 0.1

    # URL present
    if doc.url:
        score += 0.05

    # Title present and meaningful
    if doc.title and len(doc.title) > 10:
        score += 0.05

    # Author present
    if doc.author:
        score += 0.05

    # Penalty for templates/spam
    if _is_template(text):
        score -= 0.3
    if _is_spam(text):
        score -= 0.4

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def _near_duplicate(a: str, b: str, threshold: float = 0.85) -> bool:
    """Fast near-duplicate check using SequenceMatcher."""
    if abs(len(a) - len(b)) > max(len(a), len(b)) * 0.3:
        return False
    # Compare first 500 chars for speed
    return SequenceMatcher(None, a[:500], b[:500]).ratio() >= threshold


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_documents(docs: list[RawDocument]) -> tuple[list[ProcessedDocument], list[dict]]:
    """Clean, deduplicate, and score documents.

    Returns:
        (accepted, rejected) where rejected contains rejection reasons.
    """
    seen_hashes: dict[str, str] = {}
    accepted: list[ProcessedDocument] = []
    rejected: list[dict] = []

    for doc in docs:
        text = doc.content or doc.title
        clean = normalize_text(text)

        # Quality scoring
        score = quality_score(doc)
        if score < 0.3:
            rejected.append({"id": doc.id, "url": doc.url, "reason": "low_quality", "score": score})
            continue

        # Exact dedup
        h = _content_hash(clean)
        if h in seen_hashes:
            rejected.append({"id": doc.id, "url": doc.url, "reason": "exact_dup", "dup_of": seen_hashes[h]})
            continue

        # Near-duplicate check (only against recent accepts for performance)
        is_dup = False
        for existing in accepted[-50:]:
            if _near_duplicate(clean, existing.clean_content):
                rejected.append({"id": doc.id, "url": doc.url, "reason": "near_dup", "dup_of": existing.id})
                is_dup = True
                break

        if is_dup:
            continue

        seen_hashes[h] = doc.id
        accepted.append(
            ProcessedDocument(
                id=doc.id,
                url=doc.url,
                title=doc.title,
                content=doc.content,
                clean_content=clean,
                author=doc.author,
                source=doc.source,
                category=doc.category,
                timestamp=doc.timestamp,
                quality_score=score,
                word_count=word_count(clean),
                metadata=doc.metadata,
            )
        )

    log.info(
        "Processed %d docs: %d accepted, %d rejected",
        len(docs),
        len(accepted),
        len(rejected),
    )
    return accepted, rejected


def save_processed(
    accepted: list[ProcessedDocument],
    rejected: list[dict],
    query: str,
) -> tuple[Path, Path]:
    """Save processed results to JSONL files."""
    from datetime import datetime

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() else "_" for c in query)[:50]

    accepted_path = PROCESSED_DIR / f"{safe_query}_{timestamp}_accepted.jsonl"
    rejected_path = PROCESSED_DIR / f"{safe_query}_{timestamp}_rejected.jsonl"

    with open(accepted_path, "w", encoding="utf-8") as f:
        for doc in accepted:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

    with open(rejected_path, "w", encoding="utf-8") as f:
        for item in rejected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    log.info("Saved %d accepted to %s", len(accepted), accepted_path)
    log.info("Saved %d rejected to %s", len(rejected), rejected_path)
    return accepted_path, rejected_path
