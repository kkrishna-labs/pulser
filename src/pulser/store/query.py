"""RAG query engine.

Combines vector search with LLM generation for question answering
over the collected and enriched document corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import chromadb

from pulser.config import CHROMA_DIR
from pulser.llm import answer_question, summarize
from pulser.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class QueryResult:
    """Result of a RAG query."""

    question: str = ""
    answer: str | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "rag"


class RAGQueryEngine:
    """Query the vector store with optional LLM augmentation."""

    def __init__(self, collection_name: str = "pulser_docs"):
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        try:
            self._collection = self._client.get_collection(name=collection_name)
        except Exception:
            log.warning("Collection '%s' not found. Run 'pulser build' first.", collection_name)
            self._collection = None

    @property
    def ready(self) -> bool:
        return self._collection is not None and self._collection.count() > 0

    def search(
        self,
        query: str,
        n_results: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over the vector store.

        Args:
            query: Search query.
            n_results: Number of results.
            filters: Optional ChromaDB where-clause filters.

        Returns:
            List of matching documents with metadata and scores.
        """
        if not self.ready:
            return []

        try:
            where = filters if filters else None
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            log.error("Vector search failed: %s", e)
            return []

        docs = []
        if results and results.get("documents"):
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                docs.append({
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                    "relevance": 1 - dist,  # Cosine similarity
                })

        return docs

    def query(
        self,
        question: str,
        n_results: int = 5,
        filters: dict[str, Any] | None = None,
        use_llm: bool = True,
    ) -> QueryResult:
        """Full RAG query: search + generate answer.

        Args:
            question: User question.
            n_results: Number of context chunks to retrieve.
            filters: Optional metadata filters.
            use_llm: Whether to use LLM for answer generation.

        Returns:
            QueryResult with answer and source documents.
        """
        results = self.search(question, n_results=n_results, filters=filters)

        if not results:
            return QueryResult(
                question=question,
                answer="No relevant documents found. Try different search terms or run 'pulser collect' first.",
                sources=[],
                mode="empty",
            )

        # Build context
        context_parts = []
        for r in results:
            meta = r["metadata"]
            header = f"[Source: {meta.get('source', 'unknown')} | {meta.get('title', '')[:80]}]"
            context_parts.append(f"{header}\n{r['content']}")
        context = "\n\n".join(context_parts)

        # Generate answer
        answer_text = None
        if use_llm:
            answer_text = answer_question(question, context)

        return QueryResult(
            question=question,
            answer=answer_text or "LLM not configured. Showing raw search results.",
            sources=results,
            mode="rag" if use_llm else "search",
        )

    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        if not self.ready:
            return {"status": "empty"}
        count = self._collection.count()
        peek = self._collection.peek(limit=min(3, count))
        return {
            "total_chunks": count,
            "sample_metadatas": peek.get("metadatas", [])[:3],
        }
