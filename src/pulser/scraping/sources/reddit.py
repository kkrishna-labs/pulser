"""Reddit JSON API collector."""

from __future__ import annotations

from urllib.parse import quote_plus

from pulser.scraping.base import BaseCollector
from pulser.schema import RawDocument, SourceCategory
from pulser.utils.logging import get_logger
from pulser.utils.text import normalize_text

log = get_logger(__name__)


class RedditCollector(BaseCollector):
    name = "reddit"
    category = SourceCategory.SOCIAL
    base_url = "https://www.reddit.com"

    async def collect(self, terms: list[str], max_results: int = 20) -> list[RawDocument]:
        docs = []
        per_term = max(3, max_results // len(terms)) if terms else 0

        for term in terms:
            url = f"{self.base_url}/search.json?q={quote_plus(term)}&sort=relevance&limit={per_term}&t=month"
            data = await self.fetch_json(url)
            if not data or not isinstance(data, dict):
                continue

            children = data.get("data", {}).get("children", [])
            for child in children[:per_term]:
                post = child.get("data", {})
                title = normalize_text(post.get("title", ""))
                selftext = normalize_text(post.get("selftext", ""))
                content = selftext if selftext else title

                docs.append(
                    RawDocument(
                        url=f"https://reddit.com{post.get('permalink', '')}",
                        title=title,
                        content=content,
                        author=post.get("author", ""),
                        source="reddit",
                        category=self.category,
                        timestamp=str(post.get("created_utc", "")),
                        metadata={
                            "subreddit": post.get("subreddit", ""),
                            "score": post.get("score", 0),
                            "num_comments": post.get("num_comments", 0),
                            "feed_term": term,
                        },
                    )
                )

        log.info("reddit: collected %d documents", len(docs))
        return docs
