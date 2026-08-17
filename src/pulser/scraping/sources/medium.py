"""Medium RSS + httpx collector."""

from __future__ import annotations

from urllib.parse import quote_plus

import feedparser

from pulser.scraping.base import BaseCollector
from pulser.schema import RawDocument, SourceCategory
from pulser.utils.logging import get_logger
from pulser.utils.text import normalize_text

log = get_logger(__name__)


class MediumCollector(BaseCollector):
    name = "medium"
    category = SourceCategory.BLOG
    base_url = "https://medium.com"

    async def collect(self, terms: list[str], max_results: int = 20) -> list[RawDocument]:
        docs = []
        per_term = max(3, max_results // len(terms)) if terms else 0

        for term in terms:
            # Medium's tag-based RSS
            tag = term.lower().replace(" ", "-").replace(".", "")
            url = f"https://medium.com/feed/tag/{tag}"
            text = await self.fetch(url)
            if not text:
                # Try search via Google News RSS for Medium specifically
                url = f"https://news.google.com/rss/search?q={quote_plus(term + ' site:medium.com')}&hl=en-US"
                text = await self.fetch(url)

            if not text:
                continue

            feed = feedparser.parse(text)
            for entry in feed.entries[:per_term]:
                content = ""
                if hasattr(entry, "summary"):
                    content = normalize_text(entry.summary)
                elif hasattr(entry, "content"):
                    content = normalize_text(entry.content[0].get("value", ""))

                docs.append(
                    RawDocument(
                        url=entry.get("link", ""),
                        title=normalize_text(entry.get("title", "")),
                        content=content,
                        author=entry.get("author", ""),
                        source="medium",
                        category=self.category,
                        timestamp=entry.get("published", ""),
                        metadata={"feed_term": term, "tag": tag},
                    )
                )

        log.info("medium: collected %d documents", len(docs))
        return docs
