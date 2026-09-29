"""Minimal eval harness (v2): measure retrieval before you change it.

The rule of RAG work: never tune blind. Write 10-20 questions about YOUR
corpus where you know the answer, run this script, get a hit rate. Then
change chunk_size, try hybrid search, add a reranker, and re-run. If the
number goes up, keep the change. If not, revert.

A "hit" here is simple: at least one retrieved chunk contains one of the
expected keywords. Crude, but it catches regressions and it is honest.

Usage (run from backend/):
  python eval.py
"""
from __future__ import annotations

from app.config import load_settings
from app.embeddings import get_embedder
from app.retrieve import Retriever
from app.store import VectorStore

# TODO: replace these with questions about YOUR ingested corpus.
QUESTIONS = [
    {
        "question": "How does chunking work in this project?",
        "must_contain": ["chunk", "paragraph", "overlap"],
    },
    {
        "question": "Where are embeddings stored?",
        "must_contain": ["chroma", "vector"],
    },
    {
        "question": "How does the model cite its sources?",
        "must_contain": ["[1]", "cite", "source"],
    },
]


def run(top_k: int = 5) -> float:
    settings = load_settings()
    retriever = Retriever(get_embedder(settings), VectorStore(settings.chroma_dir), top_k)
    hits = 0
    for item in QUESTIONS:
        chunks = retriever.retrieve(item["question"], top_k=top_k)
        blob = " ".join(c.text for c in chunks).lower()
        hit = any(kw.lower() in blob for kw in item["must_contain"])
        hits += hit
        mark = "HIT " if hit else "MISS"
        print(f"[{mark}] {item['question']}")
        for c in chunks[:3]:
            print(f"       - {c.source} (score {c.score})")
    rate = hits / len(QUESTIONS)
    print(f"\nHit rate: {hits}/{len(QUESTIONS)} = {rate:.0%}")
    return rate


if __name__ == "__main__":
    run()
