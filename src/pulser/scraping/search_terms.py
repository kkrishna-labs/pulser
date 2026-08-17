"""Search term generation.

Three-strategy approach:
1. LLM-powered expansion (if API key available)
2. Google News RSS discovery
3. Local fallback suffixes
"""

from __future__ import annotations

import asyncio
from urllib.parse import quote_plus

import feedparser

from pulser.config import get_llm_config
from pulser.schema import SearchTerms
from pulser.utils.logging import get_logger

log = get_logger(__name__)

FALLBACK_SUFFIXES = [
    "news",
    "reviews",
    "latest",
    "analysis",
    "opinion",
    "discussion",
    "trends",
    "2024",
]


async def _llm_expand(query: str) -> list[str] | None:
    """Use LLM to generate diverse search terms."""
    cfg = get_llm_config()
    if not cfg.enabled:
        return None

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

        prompt = (
            f"Generate 8 diverse search terms for web research about: \"{query}\"\n"
            "Return ONLY a JSON array of strings, nothing else.\n"
            'Example: ["term1", "term2", ...]'
        )

        response = await client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=256,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        import json
        terms = json.loads(raw)
        if isinstance(terms, list) and all(isinstance(t, str) for t in terms):
            log.info("LLM generated %d search terms", len(terms))
            return terms
    except Exception as e:
        log.warning("LLM term generation failed: %s", e)
    return None


async def _rss_discover(query: str) -> list[str] | None:
    """Discover related terms from Google News RSS feed titles."""
    try:
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        # Use aiohttp directly for a quick fetch
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return None
                text = await resp.text()

        feed = feedparser.parse(text)
        words: dict[str, int] = {}
        for entry in feed.entries[:15]:
            title = entry.get("title", "")
            for word in title.split():
                w = word.strip(".,!?\"'").lower()
                if len(w) > 3 and w not in _stopwords():
                    words[w] = words.get(w, 0) + 1

        # Top words become supplementary terms
        top = sorted(words, key=words.get, reverse=True)[:6]
        if top:
            log.info("RSS discovered terms: %s", top)
            return top
    except Exception as e:
        log.warning("RSS term discovery failed: %s", e)
    return None


def _fallback_expand(query: str) -> list[str]:
    """Generate terms using local suffixes."""
    terms = [query]
    for suffix in FALLBACK_SUFFIXES[:6]:
        terms.append(f"{query} {suffix}")
    log.info("Fallback generated %d terms", len(terms))
    return terms


def _stopwords() -> set[str]:
    return {
        "the", "and", "for", "that", "this", "with", "from", "are", "was",
        "has", "have", "been", "will", "would", "could", "about", "more",
        "than", "its", "not", "but", "what", "all", "new", "how", "can",
        "may", "one", "our", "out", "day", "had", "her", "his", "two",
        "way", "who", "did", "get", "let", "say", "she", "too", "use",
    }


async def generate_search_terms(query: str) -> SearchTerms:
    """Generate search terms using the best available strategy.

    Tries LLM -> RSS -> Fallback in order.
    """
    # Strategy 1: LLM
    llm_terms = await _llm_expand(query)
    if llm_terms:
        return SearchTerms(query=query, terms=llm_terms, source="llm")

    # Strategy 2: RSS discovery
    rss_terms = await _rss_discover(query)
    if rss_terms:
        all_terms = [query] + rss_terms
        return SearchTerms(query=query, terms=all_terms, source="rss")

    # Strategy 3: Fallback
    fallback = _fallback_expand(query)
    return SearchTerms(query=query, terms=fallback, source="fallback")
