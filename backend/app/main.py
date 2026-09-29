"""repochat API: FastAPI server exposing the RAG pipeline over HTTP.

Endpoints:
  GET  /health          -> {"status": "ok", "chunks": N}
  POST /ingest          -> {"path": "<folder>"} ingests a directory
  POST /ask             -> {"question": "...", "top_k": 5} streams SSE:
                           {"type": "sources", ...}, {"type": "token", ...}, {"type": "done"}
  POST /reset           -> wipe the vector store (dev helper)

Run from backend/:  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import load_settings
from .embeddings import get_embedder
from .generate import Generator
from .ingest import ingest_directory
from .retrieve import Retriever
from .store import VectorStore

settings = load_settings()
embedder = get_embedder(settings)
store = VectorStore(settings.chroma_dir, settings.collection_name)
retriever = Retriever(embedder, store, settings.top_k)
generator = Generator(settings)

app = FastAPI(title="repochat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    path: str


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


@app.get("/health")
def health():
    return {"status": "ok", "chunks": store.count()}


@app.post("/ingest")
def ingest(req: IngestRequest):
    chunks, file_count = ingest_directory(req.path, settings.chunk_size, settings.chunk_overlap)
    # Embed in batches so a big repo does not blow up memory.
    batch = 32
    for i in range(0, len(chunks), batch):
        piece = chunks[i:i + batch]
        embeddings = embedder.embed([c.text for c in piece])
        store.add_chunks(piece, embeddings)
    return {"files": file_count, "chunks": len(chunks), "total_chunks": store.count()}


@app.post("/ask")
def ask(req: AskRequest):
    chunks = retriever.retrieve(req.question, top_k=req.top_k)

    def event_stream():
        sources = [
            {"text": c.text, "source": c.source, "chunk_index": c.chunk_index, "score": c.score}
            for c in chunks
        ]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        for token in generator.generate_stream(req.question, chunks):
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/reset")
def reset():
    store.reset()
    return {"status": "reset", "chunks": store.count()}
