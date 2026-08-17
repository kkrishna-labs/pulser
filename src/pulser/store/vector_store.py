"""Vector store builder.

Embeds enriched documents and stores them in ChromaDB for RAG queries.
"""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings

from pulser.schema import EnrichedDocument
from pulser.config import CHROMA_DIR, EMBEDDING_MODEL
from pulser.utils.logging import get_logger

log = get_logger(__name__)

_collection_name = "pulser_docs"


def _get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _get_or_create_collection() -> chromadb.Collection:
    client = _get_client()
    return client.get_or_create_collection(
        name=_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_text(text: str, max_len: int = 500) -> list[str]:
    """Split text into chunks for embedding."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    sentences = text.split(". ")
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_len:
            if current:
                chunks.append(current.strip())
            current = sentence
        else:
            current = current + ". " + sentence if current else sentence
    if current:
        chunks.append(current.strip())
    return chunks


def _compact_metadata(value: Any) -> str | int | float | bool:
    """Compact metadata values for ChromaDB (which requires scalar types)."""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return ", ".join(str(v) for v in value[:10])
    if isinstance(value, dict):
        return str(value)[:200]
    return str(value)


def build_vector_store(
    docs: list[EnrichedDocument],
    collection_name: str | None = None,
) -> int:
    """Build ChromaDB vector store from enriched documents.

    Each document is chunked and embedded. Metadata is preserved for filtering.

    Returns:
        Number of chunks stored.
    """
    global _collection_name
    if collection_name:
        _collection_name = collection_name

    collection = _get_or_create_collection()
    total_chunks = 0

    for doc in docs:
        text = doc.clean_content or doc.title
        if not text or len(text) < 20:
            continue

        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc.id}_chunk_{i}"
            metadata = {
                "doc_id": doc.id,
                "url": doc.url,
                "title": doc.title[:200],
                "source": doc.source,
                "category": doc.category.value if hasattr(doc.category, "value") else str(doc.category),
                "sentiment": doc.sentiment.value if hasattr(doc.sentiment, "value") else str(doc.sentiment),
                "sentiment_score": doc.sentiment_score,
                "signal_score": doc.signal_score,
                "topics": _compact_metadata(doc.topics),
                "entities": _compact_metadata(doc.entities),
                "keyphrases": _compact_metadata(doc.keyphrases),
                "quality_score": doc.quality_score,
                "word_count": doc.word_count,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }

            collection.upsert(
                ids=[chunk_id],
                documents=[chunk],
                metadatas=[metadata],
            )
            total_chunks += 1

    log.info(
        "Vector store built: %d chunks from %d documents in '%s'",
        total_chunks,
        len(docs),
        _collection_name,
    )
    return total_chunks


def get_vector_store_stats() -> dict[str, Any]:
    """Get statistics about the current vector store."""
    try:
        collection = _get_or_create_collection()
        count = collection.count()

        # Sample metadata
        sample = collection.peek(limit=min(5, count)) if count > 0 else None
        sources = set()
        sentiments = set()
        if sample and sample.get("metadatas"):
            for meta in sample["metadatas"]:
                sources.add(meta.get("source", ""))
                sentiments.add(meta.get("sentiment", ""))

        return {
            "total_chunks": count,
            "collection": _collection_name,
            "persist_dir": str(CHROMA_DIR),
            "sample_sources": list(sources),
            "sample_sentiments": list(sentiments),
        }
    except Exception as e:
        return {"error": str(e)}
