"""Embeddings: turning text into vectors.

An embedding model maps a piece of text to a long list of numbers (a
vector) positioned so that similar meanings sit close together. "How do I
stream on Twitch?" and "broadcast setup guide" end up near each other even
though they share no words. That is what makes semantic search possible:
we embed the question the same way and look up nearby chunk vectors.

We normalize embeddings to unit length, so cosine similarity is just a dot
product, and cosine *distance* is 1 - similarity.
"""
from __future__ import annotations

import os


class LocalEmbedder:
    """Free, offline embeddings via sentence-transformers. No API key."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # First run downloads the model (~90 MB) and caches it locally.
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]


class OpenAIEmbedder:
    """Optional paid alternative with (usually) higher quality vectors."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        resp = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in data]


def get_embedder(settings) -> LocalEmbedder | OpenAIEmbedder:
    if settings.use_openai_embeddings:
        return OpenAIEmbedder(model=settings.openai_embed_model)
    return LocalEmbedder(model_name=settings.embedding_model)
