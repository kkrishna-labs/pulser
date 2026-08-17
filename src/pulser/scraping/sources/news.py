"""Google News RSS collector."""

from __future__ import annotations

import asyncio
from urllib.parse import quote_plus

import feedparser

from pulser.scraping.base import BaseCollector
from pulser.schema import RawDocument, SourceCategory
from pulser.utils.logging import get_logger
from pulser.utils.text import normalize_text

log = get_logger(__name__)


class GoogleNewsCollector(BaseCollector):
    name = "google_news"
    category = SourceCategory.NEWS
    base_url = "https://news.google.com/rss/search"

    async def collect(self, terms: list[str], max_results: int = 20) -> list[RawDocument]:
        docs = []
        per_term = max(3, max_results // len(terms)) if terms else 0

        for term in terms:
            url = f"{self.base_url}?q={quote_plus(term)}&hl=en-US&gl=US&ceid=US:en"
            text = await self.fetch(url)
            if not text:
                continue

            feed = feedparser.parse(text)
            for entry in feed.entries[:per_term]:
                content = ""
                if hasattr(entry, "summary"):
                    content = normalize_text(entry.summary)
                elif hasattr(entry, "description"):
                    content = normalize_text(entry.description)

                docs.append(
                    RawDocument(
                        url=entry.get("link", ""),
                        title=normalize_text(entry.get("title", "")),
                        content=content,
                        author=entry.get("author", ""),
                        source="google_news",
                        category=self.category,
                        timestamp=entry.get("published", ""),
                        metadata={"feed_term": term},
                    )
                )

        log.info("google_news: collected %d documents", len(docs))
        return docs
