"""Pulser Dashboard - Streamlit interface.

Three tabs:
1. Overview - sentiment distribution, source breakdown, quality scores
2. Insights - topic analysis, entity cloud, signal scores
3. Q&A - RAG-powered question answering
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pulser.config import OUTPUT_DIR, PROCESSED_DIR
from pulser.schema import EnrichedDocument, Sentiment
from pulser.store.query import RAGQueryEngine
from pulser.llm import summarize

st.set_page_config(
    page_title="Pulser Dashboard",
    page_icon="🔍",
    layout="wide",
)


def load_enriched_data() -> list[EnrichedDocument]:
    """Load the most recent enriched dataset."""
    enriched_files = sorted(OUTPUT_DIR.glob("enriched_*.jsonl"), reverse=True)
    if not enriched_files:
        return []

    docs = []
    with open(enriched_files[0], encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                data["category"] = data.get("category", "other")
                data["sentiment"] = data.get("sentiment", "neutral")
                docs.append(EnrichedDocument(**data))
    return docs


def docs_to_df(docs: list[EnrichedDocument]) -> pd.DataFrame:
    records = []
    for d in docs:
        records.append({
            "id": d.id,
            "title": d.title[:100],
            "source": d.source,
            "category": d.category.value if hasattr(d.category, "value") else str(d.category),
            "sentiment": d.sentiment.value if hasattr(d.sentiment, "value") else str(d.sentiment),
            "sentiment_score": d.sentiment_score,
            "signal_score": d.signal_score,
            "quality_score": d.quality_score,
            "word_count": d.word_count,
            "topics": ", ".join(d.topics[:3]),
            "entities": ", ".join(d.entities[:5]),
            "url": d.url,
        })
    return pd.DataFrame(records)


def tab_overview(df: pd.DataFrame):
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Documents", len(df))
    col2.metric("Sources", df["source"].nunique())
    col3.metric("Avg Quality", f"{df['quality_score'].mean():.2f}")
    col4.metric("Avg Signal", f"{df['signal_score'].mean():.1f}")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Sentiment Distribution")
        sent_counts = df["sentiment"].value_counts()
        colors = {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}
        fig = px.pie(
            sent_counts,
            values=sent_counts.values,
            names=sent_counts.index,
            color=sent_counts.index,
            color_discrete_map=colors,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Source Breakdown")
        source_counts = df["source"].value_counts()
        fig = px.bar(
            x=source_counts.index,
            y=source_counts.values,
            labels={"x": "Source", "y": "Count"},
            color=source_counts.index,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Document Quality vs Signal Score")
    fig = px.scatter(
        df,
        x="quality_score",
        y="signal_score",
        color="sentiment",
        hover_data=["title", "source"],
        color_discrete_map={"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"},
    )
    st.plotly_chart(fig, use_container_width=True)


def tab_insights(df: pd.DataFrame):
    st.header("Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Topic Distribution")
        all_topics = []
        for t in df["topics"]:
            all_topics.extend([x.strip() for x in t.split(",") if x.strip()])
        if all_topics:
            topic_counts = pd.Series(all_topics).value_counts().head(15)
            fig = px.bar(
                x=topic_counts.index,
                y=topic_counts.values,
                labels={"x": "Topic", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topics extracted yet.")

    with col2:
        st.subheader("Top Entities")
        all_entities = []
        for e in df["entities"]:
            all_entities.extend([x.strip() for x in e.split(",") if x.strip()])
        if all_entities:
            ent_counts = pd.Series(all_entities).value_counts().head(15)
            fig = px.bar(
                x=ent_counts.index,
                y=ent_counts.values,
                labels={"x": "Entity", "y": "Count"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No entities extracted yet.")

    st.divider()
    st.subheader("Signal Score Distribution")
    fig = px.histogram(
        df,
        x="signal_score",
        nbins=20,
        color="sentiment",
        color_discrete_map={"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Top Documents by Signal Score")
    top_df = df.nlargest(10, "signal_score")[["title", "source", "sentiment", "signal_score", "url"]]
    for _, row in top_df.iterrows():
        st.markdown(f"**{row['title']}**")
        st.caption(f"Source: {row['source']} | Sentiment: {row['sentiment']} | Signal: {row['signal_score']:.1f} | {row['url']}")


def tab_qa():
    st.header("Ask Questions")

    engine = RAGQueryEngine()
    if not engine.ready:
        st.warning("No data in vector store. Run `pulser collect` first.")
        return

    question = st.text_input("Ask a question about your collected data:")
    col1, col2 = st.columns([1, 4])
    with col1:
        top_k = st.slider("Results", 3, 15, 5)
    with col2:
        use_llm = st.checkbox("Use LLM for answer generation", value=True)

    if question:
        with st.spinner("Searching..."):
            result = engine.query(question, n_results=top_k, use_llm=use_llm)

        if result.answer:
            st.success(result.answer)

        st.subheader("Sources")
        for i, src in enumerate(result.sources, 1):
            meta = src["metadata"]
            with st.expander(f"{i}. {meta.get('title', '')[:80]}"):
                st.markdown(f"**Source:** {meta.get('source', 'N/A')}")
                st.markdown(f"**Sentiment:** {meta.get('sentiment', 'N/A')} ({meta.get('sentiment_score', 0):.2f})")
                st.markdown(f"**Signal:** {meta.get('signal_score', 0):.1f}")
                st.markdown(f"**Relevance:** {src['relevance']:.3f}")
                st.markdown(f"**URL:** {meta.get('url', '')}")
                st.text_area(
                    "Content",
                    value=src["content"][:500],
                    height=100,
                    key=f"src_{i}",
                )


def main():
    st.title("Pulser Dashboard")
    st.caption("Search-driven web intelligence")

    docs = load_enriched_data()
    if not docs:
        st.warning(
            "No enriched data found. Run the pipeline first:\n\n"
            "```\npulser collect \"your search query\"\n```"
        )
        return

    df = docs_to_df(docs)

    tab1, tab2, tab3 = st.tabs(["Overview", "Insights", "Q&A"])

    with tab1:
        tab_overview(df)
    with tab2:
        tab_insights(df)
    with tab3:
        tab_qa()


if __name__ == "__main__":
    main()
