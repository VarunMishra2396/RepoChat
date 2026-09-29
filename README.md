# repochat

Chat with your codebase. A retrieval-augmented generation (RAG) system built
from scratch, so every layer is visible and every layer teaches something.

Point it at any folder (a repo, your notes, a docs directory), ask questions
in plain English, and get answers with citations back to the exact files.

## How RAG works (the 60-second version)

An LLM answers from its training weights, which means it hallucinates about
things it never saw: your private code, your notes, yesterday's docs. RAG
fixes this with a cheat sheet:

1. **Ingest** your documents.
2. **Chunk** them into retrievable pieces.
3. **Embed** each chunk into a vector (a list of numbers capturing meaning).
4. **Store** the vectors in a vector database.
5. **Retrieve** the chunks closest to your question (cosine similarity).
6. **Generate** an answer from the question plus those chunks, with citations.

This repo implements each step as its own module, so you can read the
pipeline top to bottom and understand the whole thing in an afternoon.

## Quickstart

### v0: CLI (no UI, no API keys, fully local)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Index a codebase or docs folder
python cli.py ingest /path/to/your/repo

# Ask questions (answer + sources print in the terminal)
python cli.py ask "how does chunking work?"
python cli.py stats
```

Answers come from **Ollama** running locally. Install it and pull a model once:

```bash
ollama pull llama3.1
```

Prefer OpenAI? Copy `.env.example` to `.env`, add your key, and set
`REPOCHAT_LLM_PROVIDER=openai`.

### v1: API + web UI

```bash
# terminal 1: backend
cd backend && uvicorn app.main:app --reload --port 8000

# terminal 2: frontend
cd frontend && npm install && npm run dev
```

Open http://localhost:5173. Ask questions, watch tokens stream in, expand
the sources under any answer to see the exact chunks and similarity scores.

## Repo map (file -> concept)

| File | RAG concept it teaches |
|---|---|
| `backend/app/ingest.py` | ingestion: walking files, skipping junk, reading text |
| `backend/app/chunking.py` | chunking: the highest-leverage step in RAG |
| `backend/app/embeddings.py` | embeddings: text -> meaning vectors |
| `backend/app/store.py` | vector DB: ChromaDB, HNSW index, cosine distance |
| `backend/app/retrieve.py` | retrieval: question -> top-k chunks |
| `backend/app/generate.py` | augmented generation: prompt building, citations, streaming |
| `backend/app/config.py` | every tunable lever in one place |
| `backend/cli.py` | the whole loop in your terminal (v0) |
| `backend/app/main.py` | FastAPI + SSE streaming (v1) |
| `backend/eval.py` | measuring retrieval quality (v2) |
| `frontend/src/` | TypeScript chat UI with streamed tokens and sources |

## Roadmap

- **v0 (done):** CLI ingest + ask, local embeddings, Chroma, Ollama answers.
- **v1 (done):** FastAPI backend with SSE streaming, React chat UI with
  expandable cited sources.
- **v2 (next):** hybrid search (BM25 + vectors), cross-encoder reranking,
  query rewriting. Each change gets measured with `python eval.py`
  before it ships. The eval harness is scaffolded and waiting for your
  questions.
- **Stretch:** contextual retrieval (prepend chunk summaries at ingest),
  query expansion, self-reflective RAG (the model critiques its own answer
  and re-retrieves). A good catalog of these lives at
  https://github.com/prajjwal-mishra/ragflow

## Learning path

See [docs/LEARNING.md](docs/LEARNING.md) for exercises: each one changes a
single lever (chunk size, top-k, embedding model) and has you observe what
happens to answer quality. That loop is the entire skill of building RAG
systems.

## Notes

- First run downloads the embedding model (~90 MB) and `sentence-transformers`
  pulls in torch, so `pip install` takes a while. Worth it: everything then
  runs offline and free.
- The vector store lives in `backend/data/chroma` (gitignored). Changing
  chunk size or embedding model means re-ingesting: `python cli.py reset`
  then `python cli.py ingest ...` again.
- Embeddings from one model are not comparable to another. If you switch
  models, re-ingest everything.
