"""Retrieval: turning a question into the most relevant chunks.

The core loop is three steps:
  1. embed the question with the SAME model used for the chunks
     (mixing models silently breaks everything, vectors are not portable),
  2. ask the vector store for the nearest chunk vectors,
  3. hand those chunks to the generator.

v2 ideas live here later: hybrid search (BM25 keyword scores fused with
vector scores), cross-encoder reranking of the top 20 down to the top 5,
and query rewriting ("what is X?" -> "X definition, X usage").
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    score: float  # cosine similarity, 0..1


class Retriever:
    def __init__(self, embedder, store, top_k: int = 5):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []
        query_vector = self.embedder.embed([query])[0]
        hits = self.store.search(query_vector, top_k or self.top_k)
        return [RetrievedChunk(**h) for h in hits]
