"""Parallel collection orchestrator.

Runs all registered source collectors concurrently and merges results.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from datetime import datetime

from pulser.scraping.base import BaseCollector
from pulser.scraping.sources.news import GoogleNewsCollector
from pulser.scraping.sources.reddit import RedditCollector
from pulser.scraping.sources.hackernews import HackerNewsCollector
from pulser.scraping.sources.medium import MediumCollector
from pulser.schema import RawDocument, SourceCategory
from pulser.config import RAW_DIR
from pulser.utils.logging import get_logger

log = get_logger(__name__)

# Registry of all available collectors
ALL_COLLECTORS: list[type[BaseCollector]] = [
    GoogleNewsCollector,
    RedditCollector,
    HackerNewsCollector,
    MediumCollector,
]


async def run_collector(
    collector_cls: type[BaseCollector],
    terms: list[str],
    max_results: int,
) -> list[RawDocument]:
    """Run a single collector, handling errors gracefully."""
    collector = collector_cls()
    try:
        docs = await collector.collect(terms, max_results=max_results)
        return docs
    except Exception as e:
        log.error("%s failed: %s", collector_cls.name, e)
        return []
    finally:
        await collector.close()


async def collect_all(
    terms: list[str],
    max_results_per_source: int = 20,
    sources: list[str] | None = None,
) -> list[RawDocument]:
    """Run all (or selected) collectors in parallel.

    Args:
        terms: Search terms to query across sources.
        max_results_per_source: Max documents per source collector.
        sources: Optional list of source names to use. None = all.

    Returns:
        Deduplicated list of RawDocuments.
    """
    collectors = ALL_COLLECTORS
    if sources:
        collectors = [c for c in ALL_COLLECTORS if c.name in sources]

    log.info(
        "Collecting from %d sources with %d terms: %s",
        len(collectors),
        len(terms),
        ", ".join(terms[:5]) + ("..." if len(terms) > 5 else ""),
    )

    tasks = [
        run_collector(cls, terms, max_results_per_source) for cls in collectors
    ]
    results = await asyncio.gather(*tasks)

    all_docs = []
    for docs in results:
        all_docs.extend(docs)

    # Dedup by URL
    seen_urls: set[str] = set()
    unique: list[RawDocument] = []
    for doc in all_docs:
        if doc.url and doc.url not in seen_urls:
            seen_urls.add(doc.url)
            unique.append(doc)
        elif not doc.url:
            unique.append(doc)

    log.info(
        "Collection complete: %d documents (%d unique)",
        len(all_docs),
        len(unique),
    )
    return unique


def save_raw_documents(docs: list[RawDocument], query: str) -> Path:
    """Persist raw documents to JSON."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() else "_" for c in query)[:50]
    filename = f"{safe_query}_{timestamp}.json"
    path = RAW_DIR / filename

    data = [doc.to_dict() for doc in docs]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %d raw documents to %s", len(docs), path)
    return path
