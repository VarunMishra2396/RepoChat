"""repochat CLI (v0): the whole RAG loop in your terminal, no UI needed.

Usage (run from backend/):
  python cli.py ingest /path/to/repo        # index a codebase or docs folder
  python cli.py ask "how does auth work?"   # ask; prints answer + sources
  python cli.py stats                       # how many chunks are stored
  python cli.py reset                       # wipe the store and start over
"""
from __future__ import annotations

import argparse

from app.config import load_settings
from app.embeddings import get_embedder
from app.generate import Generator
from app.ingest import ingest_directory
from app.retrieve import Retriever
from app.store import VectorStore


def main() -> None:
    settings = load_settings()
    embedder = get_embedder(settings)
    store = VectorStore(settings.chroma_dir, settings.collection_name)
    retriever = Retriever(embedder, store, settings.top_k)
    generator = Generator(settings)

    parser = argparse.ArgumentParser(prog="repochat")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="index a directory")
    p_ingest.add_argument("path", help="folder to ingest")

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question", help="your question")
    p_ask.add_argument("--top-k", type=int, default=None)

    sub.add_parser("stats", help="show stored chunk count")
    sub.add_parser("reset", help="wipe the vector store")

    args = parser.parse_args()

    if args.cmd == "ingest":
        chunks, file_count = ingest_directory(args.path, settings.chunk_size, settings.chunk_overlap)
        print(f"Loaded {file_count} files -> {len(chunks)} chunks. Embedding...")
        batch = 32
        for i in range(0, len(chunks), batch):
            piece = chunks[i:i + batch]
            store.add_chunks(piece, embedder.embed([c.text for c in piece]))
            print(f"  ...{min(i + batch, len(chunks))}/{len(chunks)}")
        print(f"Done. {store.count()} chunks in the store.")

    elif args.cmd == "ask":
        chunks = retriever.retrieve(args.question, top_k=args.top_k)
        answer = generator.generate(args.question, chunks)
        print(f"\n{answer}\n")
        print("Sources:")
        for i, c in enumerate(chunks, start=1):
            print(f"  [{i}] {c.source} (chunk {c.chunk_index}, score {c.score})")

    elif args.cmd == "stats":
        print(f"{store.count()} chunks stored in {settings.chroma_dir}")

    elif args.cmd == "reset":
        store.reset()
        print("Store wiped.")


if __name__ == "__main__":
    main()
