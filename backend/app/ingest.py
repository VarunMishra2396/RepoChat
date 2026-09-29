"""Ingestion: walking a directory and turning files into chunks.

Walks a folder (a codebase, a notes directory, anything), reads text files,
skips binaries and junk directories, and chunks everything. The output is a
list of Chunk objects ready for embedding and storage.
"""
from __future__ import annotations

import os

from .chunking import Chunk, chunk_documents

TEXT_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".rst",
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs",
    ".kt", ".kts", ".java", ".go", ".rs", ".rb", ".php",
    ".c", ".h", ".cpp", ".cs", ".swift",
    ".json", ".yaml", ".yml", ".toml", ".xml", ".html", ".css", ".sql",
}

SKIP_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", "__pycache__", ".venv", "venv", ".tox",
    "dist", "build", "out", "target", ".next", ".nuxt",
    "data",  # our own chroma storage, never ingest it
}

MAX_FILE_BYTES = 200 * 1024  # skip huge files (minified bundles, lockfiles)


def load_documents(root: str) -> list[tuple[str, str]]:
    """Return [(relative_path, text)] for every readable text file under root."""
    documents: list[tuple[str, str]] = []
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in TEXT_EXTENSIONS:
                continue
            full = os.path.join(dirpath, filename)
            if os.path.getsize(full) > MAX_FILE_BYTES:
                continue
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            if text.strip():
                rel = os.path.relpath(full, root)
                documents.append((rel, text))
    return documents


def ingest_directory(root: str, chunk_size: int = 500, chunk_overlap: int = 50) -> tuple[list[Chunk], int]:
    """Load and chunk a directory. Returns (chunks, file_count)."""
    documents = load_documents(root)
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    return chunks, len(documents)
