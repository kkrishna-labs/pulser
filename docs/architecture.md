# Pulser Architecture Notes

## 1. System Overview

Pulser implements a six-stage pipeline for transforming user queries into actionable intelligence:

```
Query -> Terms -> Collection -> Preprocessing -> NLP -> Vector Store -> Query/Dashboard
```

Each stage is designed as a composable unit with well-defined input/output contracts, enabling independent testing and replacement.

## 2. Design Decisions

### 2.1 Search Term Expansion

**Problem**: A single user query is insufficient for comprehensive web coverage.

**Solution**: Three-tier expansion strategy:

1. **LLM Expansion** (preferred): Uses an OpenAI-compatible API to generate diverse, semantically related search terms. This produces the highest-quality terms but requires an API key.

2. **RSS Discovery**: Parses Google News RSS feed titles for the query to discover frequently co-occurring words. These become supplementary search terms. This is free but noisier than LLM expansion.

3. **Fallback Suffixes**: Appends common informational suffixes ("news", "reviews", "analysis", etc.) to the original query. This is always available but produces the least diverse terms.

The strategy pattern ensures the system degrades gracefully: LLM -> RSS -> Fallback, each step only attempted if the previous failed.

### 2.2 Collection Architecture

**Problem**: Web scraping must handle diverse source formats (RSS, JSON APIs, HTML) while being respectful of rate limits.

**Solution**: Abstract `BaseCollector` class provides:

- Async HTTP with `aiohttp` for concurrent requests
- Semaphore-based concurrency control (`MAX_CONCURRENT_REQUESTS`)
- Exponential backoff retry with jitter
- User-agent rotation from a pool of 4 modern browser strings
- Rate-limit respect via `Retry-After` header parsing

Concrete collectors (`GoogleNewsCollector`, `RedditCollector`, etc.) only implement the `collect()` method, inheriting all HTTP infrastructure. This follows the Template Method pattern.

The orchestrator runs all collectors via `asyncio.gather()` and deduplicates results by URL.

### 2.3 Preprocessing

**Problem**: Raw scraped data contains duplicates, templates, spam, and low-quality content.

**Solution**: Multi-stage filtering:

1. **Quality Scoring**: Heuristic function considering word count, URL presence, author presence, template patterns, and spam patterns. Documents scoring below 0.3 are rejected.

2. **Exact Deduplication**: MD5 hash of normalized text content. Fast O(1) lookup.

3. **Near-Duplicate Detection**: `SequenceMatcher` ratio comparison on the first 500 characters. Threshold: 0.85. Only checked against the most recent 50 accepted documents (rolling window) to maintain O(n) performance.

### 2.4 NLP Enrichment

**Problem**: Extract structured intelligence from unstructured text.

**Solution**: Lightweight, composable extraction:

- **Sentiment**: RoBERTa transformer (`cardiffnlp/twitter-roberta-base-sentiment-latest`). Batch processing for efficiency. 3-class: positive/negative/neutral with confidence scores.

- **Topics**: Keyword taxonomy mapping to 6 categories (technology, business, politics, health, science, society). Simple but interpretable.

- **Entities**: Pattern-based extraction using regex for capitalized multi-word names, two-word proper names, and acronyms. Not as robust as spaCy NER but zero-dependency.

- **Relationships**: Regex patterns for common relationship types (acquires, partners_with, launches, etc.).

- **Signal Score**: Composite 0-100 score aggregating word count, URL validity, author presence, topic richness, entity density, relationship count, sentiment confidence, and keyphrase count.

**Trade-off**: We chose pattern-based extraction over ML-based NER to keep the dependency footprint small and the system fast. For production use, spaCy or a fine-tuned NER model would be more accurate.

### 2.5 Vector Store

**Problem**: Enable semantic search and RAG over the collected corpus.

**Solution**:

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional). Fast, small, good general-purpose performance.

- **Storage**: ChromaDB with HNSW index and cosine similarity. Persistent on disk.

- **Chunking**: Documents are split into ~500-character chunks at sentence boundaries. Each chunk is stored with full metadata (source, sentiment, topics, etc.) for filtered retrieval.

- **Metadata Filtering**: ChromaDB `where` clauses enable filtered search by source, sentiment, topic, etc.

### 2.6 RAG Query

**Problem**: Answer natural language questions over the collected corpus.

**Solution**: Standard Retrieve-then-Generate:

1. **Retrieve**: Vector similarity search returning top-k chunks
2. **Generate**: Construct prompt with retrieved context + question, send to LLM

The system gracefully degrades: if no LLM is configured, it returns raw search results with source links.

## 3. Data Models

The pipeline uses three progressively enriched document types:

```python
RawDocument          # Direct from scraper
    -> ProcessedDocument  # Cleaned, scored, deduplicated
        -> EnrichedDocument  # NLP annotations added
```

Each is a Python `dataclass` with a `to_dict()` method for serialization. The progression is unidirectional: data flows from Raw to Enriched, never backwards.

## 4. Extensibility Points

| Component | How to Extend |
|-----------|--------------|
| New source | Subclass `BaseCollector`, implement `collect()`, add to `ALL_COLLECTORS` |
| New NLP feature | Add extraction function to `nlp/engine.py`, add field to `EnrichedDocument` |
| New LLM provider | Modify `llm.py` (OpenAI-compatible APIs work out of the box) |
| New dashboard tab | Add function in `dashboard.py`, register in the `st.tabs()` block |

## 5. Performance Characteristics

| Stage | Complexity | Bottleneck |
|-------|-----------|------------|
| Search terms | O(1) or O(n_terms) | Network (LLM/RSS) |
| Collection | O(n_terms * n_sources) | Network I/O |
| Preprocessing | O(n_docs * n_recent) | CPU (near-dup check) |
| NLP | O(n_docs) | GPU/CPU (sentiment model) |
| Vector store | O(n_chunks) | Embedding generation |
| Query | O(log n_chunks) | Vector search + LLM |

## 6. Limitations

1. **Entity extraction** is pattern-based, not ML-based. Misses entities not matching capitalization patterns.
2. **Topic classification** is keyword-based, not probabilistic. May misclassify ambiguous texts.
3. **Near-duplicate detection** uses a rolling window of 50, not full corpus comparison. Some duplicates may slip through on large corpora.
4. **Relationship extraction** is limited to 7 predefined patterns. Custom relationships require code changes.
5. **No streaming**: The pipeline runs batch-style. Real-time streaming would require architectural changes.

## 7. Future Work

- spaCy integration for robust NER and dependency parsing
- Incremental vector store updates (avoid full rebuild)
- Multi-query expansion with query decomposition
- Source reliability scoring
- Temporal analysis (trend detection over time)
- Export to Neo4j for graph-based queries
- Kafka integration for real-time streaming
