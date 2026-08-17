"""Pulser CLI - Search-driven web intelligence.

Usage:
    pulser collect "quantum computing"          # Scrape + preprocess + enrich
    pulser query "what are the latest trends?"  # RAG query over collected data
    pulser build                                # Rebuild vector store from processed data
    pulser stats                                # Show pipeline statistics
    pulser dashboard                            # Launch Streamlit dashboard
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from pulser.config import RAW_DIR, PROCESSED_DIR, OUTPUT_DIR
from pulser.utils.logging import get_logger

log = get_logger("pulser")


def cmd_collect(args: argparse.Namespace) -> None:
    """Run the full collection pipeline."""
    from pulser.scraping.search_terms import generate_search_terms
    from pulser.scraping.orchestrator import collect_all, save_raw_documents
    from pulser.preprocessing import process_documents, save_processed
    from pulser.nlp.engine import enrich_documents
    from pulser.store.vector_store import build_vector_store
    from pulser.schema import RawDocument, ProcessedDocument

    query = args.query
    max_per_source = args.max_results
    sources = args.sources.split(",") if args.sources else None

    t0 = time.time()

    # Step 1: Generate search terms
    log.info("=" * 60)
    log.info("STEP 1: Generating search terms for '%s'", query)
    log.info("=" * 60)
    search_terms = asyncio.run(generate_search_terms(query))
    log.info("Terms (%s): %s", search_terms.source, ", ".join(search_terms.terms))

    # Step 2: Collect from web sources
    log.info("=" * 60)
    log.info("STEP 2: Collecting from web sources")
    log.info("=" * 60)
    raw_docs = asyncio.run(collect_all(search_terms.terms, max_per_source, sources))
    save_raw_documents(raw_docs, query)
    log.info("Raw documents: %d", len(raw_docs))

    # Step 3: Preprocess
    log.info("=" * 60)
    log.info("STEP 3: Preprocessing")
    log.info("=" * 60)
    accepted, rejected = process_documents(raw_docs)
    save_processed(accepted, rejected, query)
    log.info("Accepted: %d, Rejected: %d", len(accepted), len(rejected))

    # Step 4: NLP enrichment
    log.info("=" * 60)
    log.info("STEP 4: NLP enrichment")
    log.info("=" * 60)
    enriched = enrich_documents(accepted)

    # Save enriched docs
    enriched_path = OUTPUT_DIR / f"enriched_{query.replace(' ', '_')[:30]}.jsonl"
    with open(enriched_path, "w", encoding="utf-8") as f:
        for doc in enriched:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
    log.info("Saved enriched docs to %s", enriched_path)

    # Step 5: Build vector store
    log.info("=" * 60)
    log.info("STEP 5: Building vector store")
    log.info("=" * 60)
    chunks = build_vector_store(enriched)
    log.info("Vector store: %d chunks", chunks)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(
        "PIPELINE COMPLETE in %.1fs | Raw: %d | Processed: %d | Enriched: %d | Chunks: %d",
        elapsed,
        len(raw_docs),
        len(accepted),
        len(enriched),
        chunks,
    )
    log.info("=" * 60)


def cmd_query(args: argparse.Namespace) -> None:
    """Query the vector store with RAG."""
    from pulser.store.query import RAGQueryEngine

    engine = RAGQueryEngine()
    if not engine.ready:
        log.error("No data in vector store. Run 'pulser collect' first.")
        sys.exit(1)

    result = engine.query(
        args.question,
        n_results=args.top_k,
        use_llm=not args.no_llm,
    )

    print("\n" + "=" * 60)
    print(f"Q: {result.question}")
    print("=" * 60)
    print(f"\nA: {result.answer}\n")
    print(f"Sources ({len(result.sources)}):")
    for i, src in enumerate(result.sources[:5], 1):
        meta = src["metadata"]
        print(f"  {i}. [{meta.get('source', '?')}] {meta.get('title', '')[:70]}")
        print(f"     Relevance: {src['relevance']:.3f} | {meta.get('url', '')}")
    print()


def cmd_build(args: argparse.Namespace) -> None:
    """Rebuild vector store from existing processed data."""
    from pulser.store.vector_store import build_vector_store
    from pulser.nlp.engine import enrich_documents
    from pulser.preprocessing import process_documents
    from pulser.schema import RawDocument

    # Load most recent enriched file
    enriched_files = sorted(OUTPUT_DIR.glob("enriched_*.jsonl"), reverse=True)
    if not enriched_files:
        log.error("No enriched files found. Run 'pulser collect' first.")
        sys.exit(1)

    enriched_path = enriched_files[0]
    log.info("Loading enriched docs from %s", enriched_path)

    from pulser.schema import EnrichedDocument
    docs = []
    with open(enriched_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                data["category"] = data.get("category", "other")
                data["sentiment"] = data.get("sentiment", "neutral")
                docs.append(EnrichedDocument(**data))

    chunks = build_vector_store(docs)
    log.info("Vector store rebuilt: %d chunks", chunks)


def cmd_stats(args: argparse.Namespace) -> None:
    """Show pipeline statistics."""
    from pulser.store.vector_store import get_vector_store_stats

    print("\n" + "=" * 50)
    print("PULSER PIPELINE STATISTICS")
    print("=" * 50)

    # File counts
    raw_count = len(list(RAW_DIR.glob("*.json"))) if RAW_DIR.exists() else 0
    processed_count = len(list(PROCESSED_DIR.glob("*.jsonl"))) if PROCESSED_DIR.exists() else 0
    enriched_count = len(list(OUTPUT_DIR.glob("enriched_*.jsonl"))) if OUTPUT_DIR.exists() else 0

    print(f"\nRaw collections:     {raw_count} files")
    print(f"Processed files:     {processed_count} files")
    print(f"Enriched files:      {enriched_count} files")

    # Vector store
    stats = get_vector_store_stats()
    if "error" in stats:
        print(f"Vector store:        Error - {stats['error']}")
    else:
        print(f"Vector store chunks: {stats.get('total_chunks', 0)}")
        print(f"Collection:          {stats.get('collection', 'N/A')}")
        print(f"Persist dir:         {stats.get('persist_dir', 'N/A')}")

    print()


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / "dashboard.py"
    log.info("Launching dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


def main():
    parser = argparse.ArgumentParser(
        prog="pulser",
        description="Search-driven web intelligence: scrape, analyze, understand.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    # collect
    p_collect = subparsers.add_parser("collect", help="Run full collection pipeline")
    p_collect.add_argument("query", help="Search query / topic")
    p_collect.add_argument("--max-results", type=int, default=20, help="Max results per source")
    p_collect.add_argument("--sources", help="Comma-separated source names (default: all)")

    # query
    p_query = subparsers.add_parser("query", help="RAG query over collected data")
    p_query.add_argument("question", help="Question to answer")
    p_query.add_argument("--top-k", type=int, default=5, help="Number of context chunks")
    p_query.add_argument("--no-llm", action="store_true", help="Skip LLM, show search only")

    # build
    subparsers.add_parser("build", help="Rebuild vector store from processed data")

    # stats
    subparsers.add_parser("stats", help="Show pipeline statistics")

    # dashboard
    subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "collect": cmd_collect,
        "query": cmd_query,
        "build": cmd_build,
        "stats": cmd_stats,
        "dashboard": cmd_dashboard,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
