"""Chunking: how we cut documents into retrievable pieces.

Why this matters more than anything else in RAG: the retriever can only
return whole chunks, so chunk boundaries decide what the LLM gets to see.
Good chunks are self-contained ideas. Bad chunks split an idea in half,
and no embedding model can fix that later.

Strategy here (v0): greedy packing of paragraphs into a character budget,
splitting oversized paragraphs into sentences, with a small overlap so a
thought that straddles a boundary is not lost.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str   # file path the chunk came from
    index: int    # chunk number within that file


def _split_sentences(paragraph: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [p for p in parts if p]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    # Rough token estimate: ~4 chars per token for English/code.
    budget = chunk_size * 4
    overlap_chars = chunk_overlap * 4

    # 1. Break into paragraphs; split any paragraph bigger than the budget
    #    into sentences so nothing oversized survives.
    units: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= budget:
            units.append(paragraph)
        else:
            units.extend(_split_sentences(paragraph))

    # 2. Greedily pack units into chunks.
    raw_chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= budget:
            current = candidate
        else:
            if current:
                raw_chunks.append(current)
                # Overlap: carry the tail of the finished chunk forward so a
                # sentence split across the boundary still appears whole-ish.
                # (Crude: it can cut mid-sentence. A v2 improvement is to
                # carry whole trailing sentences instead.)
                tail = current[-overlap_chars:]
                current = f"{tail}\n\n{unit}" if tail else unit
            else:
                # A single unit larger than the budget: keep it whole rather
                # than dropping content.
                raw_chunks.append(unit)
                current = ""
    if current:
        raw_chunks.append(current)

    return [Chunk(text=c, source=source, index=i) for i, c in enumerate(raw_chunks)]


def chunk_documents(
    documents: list[tuple[str, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Chunk a list of (source_path, text) documents."""
    chunks: list[Chunk] = []
    for source, text in documents:
        chunks.extend(chunk_text(text, source, chunk_size, chunk_overlap))
    return chunks
