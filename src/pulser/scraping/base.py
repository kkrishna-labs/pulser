"""Base collector with retry, rate-limiting, and user-agent rotation.

All source scrapers inherit from this. It handles the HTTP plumbing so
individual scrapers only need to implement `collect()`.
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from pulser.config import (
    MAX_CONCURRENT_REQUESTS,
    REQUEST_DELAY_MIN,
    REQUEST_DELAY_MAX,
    USER_AGENTS,
    USER_AGENT_ROTATION,
)
from pulser.schema import RawDocument, SourceCategory
from pulser.utils.logging import get_logger

log = get_logger(__name__)


class BaseCollector(ABC):
    """Abstract base for all web source collectors."""

    name: str = "base"
    category: SourceCategory = SourceCategory.OTHER
    base_url: str = ""

    def __init__(self):
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    def _headers(self) -> dict[str, str]:
        ua = random.choice(USER_AGENTS) if USER_AGENT_ROTATION else USER_AGENTS[0]
        return {"User-Agent": ua, "Accept": "text/html,application/xhtml+xml,application/json,*/*"}

    def _delay(self) -> float:
        return random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)

    async def fetch(self, url: str, **kwargs: Any) -> str | None:
        """Fetch a URL with rate-limiting, retry, and backoff."""
        async with self._semaphore:
            session = await self._get_session()
            for attempt in range(3):
                try:
                    await asyncio.sleep(self._delay())
                    async with session.get(url, **kwargs) as resp:
                        if resp.status == 429:
                            retry_after = int(resp.headers.get("Retry-After", 10))
                            log.warning("%s: rate-limited, waiting %ds", self.name, retry_after)
                            await asyncio.sleep(retry_after)
                            continue
                        if resp.status >= 400:
                            log.warning("%s: %d for %s", self.name, resp.status, url)
                            return None
                        return await resp.text()
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    log.warning("%s: attempt %d failed for %s: %s", self.name, attempt + 1, url, e)
                    await asyncio.sleep(2 ** attempt)
            return None

    async def fetch_json(self, url: str, **kwargs: Any) -> dict | list | None:
        """Fetch and parse JSON."""
        text = await self.fetch(url, **kwargs)
        if text is None:
            return None
        try:
            import json
            return json.loads(text)
        except Exception as e:
            log.warning("%s: JSON parse error for %s: %s", self.name, url, e)
            return None

    @abstractmethod
    async def collect(self, terms: list[str], max_results: int = 20) -> list[RawDocument]:
        """Collect documents matching the given search terms."""
        ...

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
