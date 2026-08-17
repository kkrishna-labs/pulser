"""Hacker News Algolia API collector."""

from __future__ import annotations

from urllib.parse import quote_plus

from pulser.scraping.base import BaseCollector
from pulser.schema import RawDocument, SourceCategory
from pulser.utils.logging import get_logger
from pulser.utils.text import normalize_text

log = get_logger(__name__)


class HackerNewsCollector(BaseCollector):
    name = "hackernews"
    category = SourceCategory.FORUM
    base_url = "https://hn.algolia.com/api/v1"

    async def collect(self, terms: list[str], max_results: int = 20) -> list[RawDocument]:
        docs = []
        per_term = max(3, max_results // len(terms)) if terms else 0

        for term in terms:
            url = f"{self.base_url}/search?query={quote_plus(term)}&tags=story&hitsPerPage={per_term}"
            data = await self.fetch_json(url)
            if not data or not isinstance(data, dict):
                continue

            hits = data.get("hits", [])
            for hit in hits[:per_term]:
                title = normalize_text(hit.get("title", ""))
                url_val = hit.get("url", "")
                hn_url = f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"

                docs.append(
                    RawDocument(
                        url=url_val or hn_url,
                        title=title,
                        content=normalize_text(hit.get("story_text", "") or title),
                        author=hit.get("author", ""),
                        source="hackernews",
                        category=self.category,
                        timestamp=hit.get("created_at", ""),
                        metadata={
                            "points": hit.get("points", 0),
                            "num_comments": hit.get("num_comments", 0),
                            "hn_url": hn_url,
                            "feed_term": term,
                        },
                    )
                )

        log.info("hackernews: collected %d documents", len(docs))
        return docs
