# Learning RAG with repochat

Each exercise changes one thing and asks you to observe the effect. That
observe-and-measure loop is the whole skill. Keep notes on what you find.

## Exercise 1: Feel the pipeline (30 min)

Ingest this repo itself:

```bash
cd backend
python cli.py ingest ..
python cli.py ask "how does chunking work?"
python cli.py ask "where are embeddings stored?"
```

Now read `app/chunking.py`, `app/embeddings.py`, `app/store.py`,
`app/retrieve.py`, `app/generate.py` in that order. For each file, answer:
what goes in, what comes out, and what would break if I removed it?

## Exercise 2: Chunking is the biggest lever (45 min)

In `app/config.py`, try `chunk_size = 150` and `chunk_size = 1500`.
Re-ingest each time (`python cli.py reset` first) and ask the same 3
questions. Notice:

- Small chunks: precise hits, but answers lose surrounding context.
- Large chunks: rich context, but the retriever returns noisier blocks.

Write down which size felt best and why. There is no universal answer;
chunk size is a property of your corpus.

## Exercise 3: top-k and the citation check (20 min)

Ask a question, then look at the similarity scores in the sources list.
Try `--top-k 2` vs `--top-k 10`. Notice when extra chunks add signal and
when they just add noise the model has to ignore. Watch for the model
citing [1] correctly; try to find a question where it cites the wrong
chunk, and figure out why.

## Exercise 4: Break it on purpose (30 min)

- Ask about something not in the docs. Confirm the model says it could not
  find it instead of hallucinating. (This is the system prompt doing its job.)
- Ingest a second, unrelated folder without resetting. Ask questions and
  see cross-corpus leakage. Then `reset` and re-ingest cleanly.
- Change the embedding model in config to something else from
  sentence-transformers without re-ingesting. Ask a question. The scores
  will be garbage. This teaches the hard rule: never mix embedding models
  between ingest and query time.

## Exercise 5: Measure before you tune (v2, 1 hour)

Fill in `eval.py` with 15 questions about your own corpus where you know
the right answers. Run it, record the hit rate. Now pick ONE improvement:

1. **Hybrid search:** add BM25 keyword scores (try the `rank-bm25` package),
   fuse with vector scores, re-run eval.
2. **Reranking:** retrieve top-20, rerank with a cross-encoder
   (`sentence-transformers` has `CrossEncoder`), keep top-5, re-run eval.
3. **Query rewriting:** before retrieval, ask the LLM to rewrite the
   question into a better search query, re-run eval.

Keep the change only if the number goes up. This discipline is what
separates demos from production systems.

## Exercise 6: Stretch goals

Pick one strategy from https://github.com/prajjwal-mishra/ragflow and
implement it as a toggle:

- **Contextual retrieval:** at ingest time, ask the LLM to write a 2-sentence
  summary of each chunk's document and prepend it before embedding.
- **Query expansion:** generate 3 variations of the question, retrieve for
  each, merge and dedupe.
- **Self-reflective RAG:** after generating, ask the model "is this fully
  supported by the context?" If not, retrieve more and try again.

Each one is a real, shippable feature and a great commit to show in the repo.
