"""Knowledge Base introspection.

Lets you explore what data is currently held in the pipeline:
queries collected, topics, sources, sentiment breakdown, and
direct vector search over enriched documents.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pulser.config import RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, CHROMA_DIR
from pulser.utils.logging import get_logger

log = get_logger(__name__)


def _load_all_enriched() -> list[dict]:
    """Load all enriched documents across every run."""
    files = sorted(OUTPUT_DIR.glob("enriched_*.jsonl"))
    docs = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    docs.append(json.loads(line))
    return docs


def _load_all_raw_queries() -> list[dict]:
    """Extract query + term source from each raw collection file."""
    files = sorted(RAW_DIR.glob("*.json"))
    queries = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            data = json.loads(fh.read())
            if data:
                # Derive query from filename: quantum_computing_20260817_130039.json
                stem = f.stem
                # Strip trailing _YYYYMMDD_HHMMSS
                parts = stem.rsplit("_", 2)
                query = parts[0].replace("_", " ") if len(parts) >= 3 else stem
                queries.append({
                    "query": query,
                    "file": f.name,
                    "doc_count": len(data),
                    "sources": list({d.get("source", "?") for d in data}),
                })
    return queries


# ── Overview ────────────────────────────────────────────────────────────

def kb_overview() -> str:
    """Full overview of the knowledge base."""
    enriched = _load_all_enriched()
    queries = _load_all_raw_queries()

    if not enriched and not queries:
        return "Knowledge base is empty. Run `pulser collect \"<query>\"` to populate."

    lines = []
    lines.append("=" * 60)
    lines.append("KNOWLEDGE BASE OVERVIEW")
    lines.append("=" * 60)

    # Collection summary
    lines.append(f"\n  Queries collected:   {len(queries)}")
    lines.append(f"  Total documents:     {len(enriched)}")

    # Vector store
    chroma_chunks = 0
    if CHROMA_DIR.exists():
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            col = client.get_collection("pulser_docs")
            chroma_chunks = col.count()
        except Exception:
            pass
    lines.append(f"  Vector store chunks: {chroma_chunks}")

    # Source breakdown
    sources = Counter(d.get("source", "?") for d in enriched)
    lines.append(f"\n  Sources: {len(sources)}")
    for src, count in sources.most_common():
        lines.append(f"    {src:<20} {count:>4} docs")

    # Sentiment breakdown
    sentiments = Counter(d.get("sentiment", "?") for d in enriched)
    lines.append(f"\n  Sentiment:")
    for sent, count in sentiments.most_common():
        pct = count / len(enriched) * 100
        lines.append(f"    {sent:<10} {count:>4} ({pct:.0f}%)")

    # Topic breakdown
    all_topics = []
    for d in enriched:
        topics = d.get("topics", "")
        if isinstance(topics, str):
            all_topics.extend(t.strip() for t in topics.split(",") if t.strip())
        elif isinstance(topics, list):
            all_topics.extend(topics)
    topic_counts = Counter(all_topics)
    lines.append(f"\n  Top topics:")
    for topic, count in topic_counts.most_common(10):
        lines.append(f"    {topic:<20} {count:>4}")

    # Signal score stats
    scores = [d.get("signal_score", 0) for d in enriched]
    if scores:
        lines.append(f"\n  Signal scores: min={min(scores):.0f}  max={max(scores):.0f}  avg={sum(scores)/len(scores):.0f}")

    # Queries collected
    lines.append(f"\n  Queries:")
    for q in queries:
        lines.append(f"    \"{q['query']}\"  ({q['doc_count']} docs, sources: {', '.join(q['sources'])})")

    lines.append("")
    return "\n".join(lines)


# ── Topics ──────────────────────────────────────────────────────────────

def kb_topics() -> str:
    """List all topics across the knowledge base."""
    enriched = _load_all_enriched()
    if not enriched:
        return "No data. Run `pulser collect` first."

    all_topics = []
    for d in enriched:
        topics = d.get("topics", "")
        if isinstance(topics, str):
            all_topics.extend(t.strip() for t in topics.split(",") if t.strip())
        elif isinstance(topics, list):
            all_topics.extend(topics)

    counts = Counter(all_topics)
    lines = ["TOPICS IN KNOWLEDGE BASE", "=" * 40]
    for topic, count in counts.most_common():
        bar = "#" * min(count, 40)
        lines.append(f"  {topic:<20} {count:>4}  {bar}")
    lines.append(f"\n  Total unique topics: {len(counts)}")
    return "\n".join(lines)


# ── Sources ─────────────────────────────────────────────────────────────

def kb_sources() -> str:
    """List all sources and their document counts."""
    enriched = _load_all_enriched()
    if not enriched:
        return "No data. Run `pulser collect` first."

    counts = Counter(d.get("source", "?") for d in enriched)
    lines = ["SOURCES IN KNOWLEDGE BASE", "=" * 40]
    for src, count in counts.most_common():
        bar = "#" * min(count, 40)
        lines.append(f"  {src:<20} {count:>4}  {bar}")
    lines.append(f"\n  Total sources: {len(counts)}")
    return "\n".join(lines)


# ── Search ──────────────────────────────────────────────────────────────

def kb_search(query: str, top_k: int = 5) -> str:
    """Search enriched documents directly (vector similarity)."""
    from pulser.store.query import RAGQueryEngine

    engine = RAGQueryEngine()
    if not engine.ready:
        return "Vector store empty. Run `pulser collect` first."

    results = engine.search(query, n_results=top_k)
    if not results:
        return f"No results for \"{query}\"."

    lines = [f"SEARCH: \"{query}\"", "=" * 60]
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        lines.append(f"\n  {i}. {meta.get('title', '')[:70]}")
        lines.append(f"     Source: {meta.get('source', '?')}  |  Sentiment: {meta.get('sentiment', '?')}  |  Signal: {meta.get('signal_score', 0):.0f}")
        lines.append(f"     Relevance: {r['relevance']:.3f}")
        lines.append(f"     {meta.get('url', '')}")
    lines.append("")
    return "\n".join(lines)


# ── Queries ─────────────────────────────────────────────────────────────

def kb_queries() -> str:
    """Show all collected queries/topics."""
    queries = _load_all_raw_queries()
    if not queries:
        return "No queries collected yet. Run `pulser collect \"<query>\"` first."

    lines = ["COLLECTED QUERIES", "=" * 50]
    for i, q in enumerate(queries, 1):
        lines.append(f"\n  {i}. \"{q['query']}\"")
        lines.append(f"     Docs: {q['doc_count']}  |  Sources: {', '.join(q['sources'])}")
        lines.append(f"     File: {q['file']}")
    lines.append(f"\n  Total: {len(queries)} queries")
    return "\n".join(lines)
