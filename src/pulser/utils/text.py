"""Text cleaning and normalization utilities."""

from __future__ import annotations

import html
import re
import unicodedata

# Common English stopwords (compact set)
STOPWORDS = frozenset(
    "a an the and or but if in on at to for of is it its that this these those "
    "was were be been being have has had do does did will would shall should may "
    "might can could am are is was were been being get got gets from by with as "
    "so no not nor too very just also than more most some any all each every "
    "both few many much own same other another such what which who whom whose "
    "when where why how i me my we our you your he him his she her they them "
    "their its about into through during before after above below between out "
    "off over under again further then once here there where why how all any "
    "both each few more most other some such no nor not only own same so than "
    "too very s t d ll m o re ve y don ain isn aren wasn weren hasn hadn "
    "doesn didn won wouldn shan shouldn couldn mustn needn".split()
)


def clean_html(text: str) -> str:
    """Strip HTML tags and decode entities."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to ASCII equivalents."""
    return unicodedata.normalize("NFKD", text)


def normalize_text(text: str) -> str:
    """Full normalization pipeline: unicode -> html -> whitespace."""
    text = normalize_unicode(text)
    text = clean_html(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(text.split())


def extract_keyphrases(text: str, top_n: int = 10) -> list[str]:
    """Extract keyphrases using n-gram frequency analysis."""
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS]

    # Bigrams
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    # Trigrams
    trigrams = [
        f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(len(tokens) - 2)
    ]

    from collections import Counter

    counts = Counter(bigrams + trigrams)
    return [phrase for phrase, _ in counts.most_common(top_n)]
