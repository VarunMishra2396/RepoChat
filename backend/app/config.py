"""Central configuration for repochat.

Every important RAG lever lives here. Change a value, re-ingest, and feel
the difference in retrieval quality. That is the fastest way to learn.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # --- chunking ---
    chunk_size: int = 500      # target tokens per chunk (we approximate tokens as chars / 4)
    chunk_overlap: int = 50    # overlapping tokens between consecutive chunks

    # --- retrieval ---
    top_k: int = 5             # how many chunks to hand the LLM per question

    # --- embeddings ---
    # Local model: free, offline, no API key. Downloaded once on first run (~90 MB).
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Optional paid alternative (set REPOCHAT_USE_OPENAI_EMBEDDINGS=1 and OPENAI_API_KEY).
    openai_embed_model: str = "text-embedding-3-small"
    use_openai_embeddings: bool = False

    # --- generation ---
    llm_provider: str = "ollama"          # "ollama" (local, free) or "openai"
    ollama_model: str = "llama3.1"        # pull it once: ollama pull llama3.1
    ollama_url: str = "http://localhost:11434"
    openai_model: str = "gpt-4o-mini"

    # --- storage ---
    chroma_dir: str = "./data/chroma"
    collection_name: str = "repochat"


def _getenv_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def load_settings() -> Settings:
    """Build Settings, letting environment variables override defaults."""
    s = Settings()
    s.chunk_size = _getenv_int("REPOCHAT_CHUNK_SIZE", s.chunk_size)
    s.chunk_overlap = _getenv_int("REPOCHAT_CHUNK_OVERLAP", s.chunk_overlap)
    s.top_k = _getenv_int("REPOCHAT_TOP_K", s.top_k)
    s.embedding_model = os.environ.get("REPOCHAT_EMBED_MODEL", s.embedding_model)
    s.use_openai_embeddings = os.environ.get("REPOCHAT_USE_OPENAI_EMBEDDINGS") == "1"
    s.llm_provider = os.environ.get("REPOCHAT_LLM_PROVIDER", s.llm_provider)
    s.ollama_model = os.environ.get("REPOCHAT_OLLAMA_MODEL", s.ollama_model)
    s.chroma_dir = os.environ.get("REPOCHAT_CHROMA_DIR", s.chroma_dir)
    return s
