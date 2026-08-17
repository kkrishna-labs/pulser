# Pulser

**Search-driven web intelligence: scrape, analyze, understand.**

Pulser is a modular pipeline that takes user-provided search terms, collects content from diverse web sources, applies NLP analysis (sentiment, topics, entities, relationships), and presents insights through a CLI, RAG query engine, or interactive dashboard.

## Architecture

```
User Query
    |
    v
[1] Search Term Generation
    LLM expansion -> RSS discovery -> Fallback suffixes
    |
    v
[2] Parallel Collection
    Google News | Reddit | HackerNews | Medium
    (asyncio + aiohttp with retry/rate-limit/UA rotation)
    |
    v
[3] Preprocessing
    Clean HTML -> Normalize text -> Quality scoring
    -> Exact dedup (MD5) -> Near-duplicate (SequenceMatcher)
    |
    v
[4] NLP Enrichment
    Sentiment (RoBERTa) -> Topics (keyword taxonomy)
    -> Entities (pattern matching) -> Relationships (regex)
    -> Signal scoring (0-100)
    |
    v
[5] Vector Store
    sentence-transformers/all-MiniLM-L6-v2 embeddings
    -> ChromaDB (HNSW, cosine similarity)
    |
    v
[6] Query & Visualization
    CLI: pulser query "..."
    Dashboard: Streamlit (overview, insights, Q&A)
```

## Installation

Requires Python 3.10+.

```bash
# Clone or navigate to the project
cd pulser

# Sync dependencies (creates .venv, installs everything)
python -m uv sync

# For development (adds pytest, ruff, uv itself)
python -m uv sync --dev

# Install Playwright browsers (for advanced scraping)
.venv/Scripts/python -m playwright install  # Windows
# .venv/bin/python -m playwright install    # Linux/Mac
```

To add/remove dependencies:

```bash
python -m uv add <package>            # add runtime dep
python -m uv add --dev <package>      # add dev dep
python -m uv remove <package>         # remove dep
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key settings:

- `LLM_API_KEY` - Optional. Enables AI-powered search term expansion and RAG answers. Supports OpenAI-compatible APIs (NVIDIA NIM, OpenAI, Ollama, etc.)
- `MAX_CONCURRENT_REQUESTS` - Scraping concurrency (default: 10)
- `EMBEDDING_MODEL` - Sentence transformer model (default: all-MiniLM-L6-v2)

## Usage

All commands go through `python -m uv run`:

### Full Pipeline

```bash
# Collect, preprocess, enrich, and build vector store for a topic
python -m uv run pulser collect "quantum computing"
```

### Query

```bash
# RAG-powered question answering
python -m uv run pulser query "what are the latest breakthroughs?"

# Search-only (no LLM)
python -m uv run pulser query "trends in AI" --no-llm
```

### Build

```bash
# Rebuild vector store from existing enriched data
python -m uv run pulser build
```

### Dashboard

```bash
# Launch Streamlit dashboard
python -m uv run pulser dashboard
```

### Stats

```bash
# View pipeline statistics
python -m uv run pulser stats
```

## Project Structure

```
pulser/
├── src/pulser/
│   ├── __init__.py
│   ├── config.py           # Central configuration
│   ├── schema.py           # Data models (RawDocument -> ProcessedDocument -> EnrichedDocument)
│   ├── preprocessing.py    # Cleaning, dedup, quality scoring
│   ├── llm.py              # LLM client (OpenAI-compatible)
│   ├── cli.py              # CLI entry point
│   ├── dashboard.py        # Streamlit dashboard
│   ├── scraping/
│   │   ├── base.py         # BaseCollector (retry, rate-limit, UA rotation)
│   │   ├── orchestrator.py # Parallel collection orchestrator
│   │   ├── search_terms.py # Search term generation (LLM/RSS/fallback)
│   │   └── sources/
│   │       ├── news.py     # Google News RSS
│   │       ├── reddit.py   # Reddit JSON API
│   │       ├── hackernews.py # HN Algolia API
│   │       └── medium.py   # Medium RSS
│   ├── nlp/
│   │   ├── sentiment.py    # RoBERTa sentiment analysis
│   │   └── engine.py       # Topics, entities, relationships, signal scoring
│   ├── store/
│   │   ├── vector_store.py # ChromaDB builder
│   │   └── query.py        # RAG query engine
│   └── utils/
│       ├── logging.py      # Structured logging
│       └── text.py         # Text cleaning, keyphrase extraction
├── docs/
│   └── architecture.md     # Detailed architecture notes
├── tests/
├── pyproject.toml
├── .env.example
└── .gitignore
```

## Data Flow

Each pipeline stage produces a typed output that feeds the next:

| Stage          | Input          | Output                                         |
| -------------- | -------------- | ---------------------------------------------- |
| Search Terms   | User query     | `SearchTerms` (terms + source)               |
| Collection     | Search terms   | `list[RawDocument]`                          |
| Preprocessing  | Raw docs       | `(list[ProcessedDocument], list[rejection])` |
| NLP Enrichment | Processed docs | `list[EnrichedDocument]`                     |
| Vector Store   | Enriched docs  | ChromaDB chunks                                |
| Query          | Question       | `QueryResult` (answer + sources)             |

## Design Principles

1. **Modularity** - Each stage is independently testable and replaceable
2. **Graceful degradation** - LLM features degrade gracefully when API keys are absent
3. **Type safety** - Dataclass-based schemas with explicit field contracts
4. **Async-first** - Collection layer uses asyncio for concurrent scraping
5. **Deterministic fallbacks** - Every LLM-powered feature has a local fallback

## Academic Notes

This project draws on established patterns in:

- **Information Retrieval**: TF-IDF inspired keyphrase extraction, cosine similarity for near-duplicate detection
- **Sentiment Analysis**: Transformer-based classification (RoBERTa fine-tuned on Twitter data)
- **RAG (Retrieval-Augmented Generation)**: Dense retrieval via sentence embeddings + ChromaDB, with LLM answer generation
- **Web Scraping**: Respectful scraping with rate limiting, user-agent rotation, and retry logic

## License

MIT
