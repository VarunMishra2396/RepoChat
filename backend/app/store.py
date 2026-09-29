"""Vector store: where chunk embeddings live, indexed for fast search.

ChromaDB keeps every chunk's vector plus its text and metadata (source
file, chunk index) on disk under ./data/chroma. At query time we ask:
"which stored vectors are closest to the question vector?" Chroma uses an
HNSW index so this stays fast even with hundreds of thousands of chunks.

We store with cosine space, so Chroma returns cosine *distance*
(0 = identical direction, 2 = opposite). We convert to a friendlier
similarity score: score = 1 - distance.
"""
from __future__ import annotations

import os


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "repochat"):
        import chromadb

        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks, embeddings: list[list[float]]) -> None:
        """Add chunks with their precomputed embeddings. Ids are stable so
        re-ingesting the same file replaces its old chunks."""
        ids = [f"{c.source}::{c.index}" for c in chunks]
        # Remove stale chunks for these sources first (clean re-ingest).
        self.collection.delete(ids=ids)
        self.collection.add(
            ids=ids,
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"source": c.source, "chunk_index": c.index} for c in chunks],
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Return the top_k closest chunks as dicts with text, source,
        chunk_index, and similarity score (0..1, higher is better)."""
        total = self.collection.count()
        if total == 0:
            return []
        n = min(top_k, total)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
        )
        hits = []
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(
                {
                    "text": doc,
                    "source": meta["source"],
                    "chunk_index": meta["chunk_index"],
                    "score": round(1 - dist, 4),
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete everything. Useful when you change chunking or embedding
        settings and want a clean re-ingest."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name="repochat",
            metadata={"hnsw:space": "cosine"},
        )
