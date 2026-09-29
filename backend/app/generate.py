"""Generation: answering the question using retrieved chunks as context.

This is the "augmented generation" half of RAG. The prompt has three parts:
  1. a system instruction telling the model to answer ONLY from the context
     and to cite sources like [1], [2],
  2. the retrieved chunks, numbered so citations map back to files,
  3. the user's question.

Two providers: Ollama (local, free, default) and OpenAI (paid, better).
Both are called with plain HTTP, no heavy SDKs.
"""
from __future__ import annotations

import json

SYSTEM_PROMPT = """You answer questions using ONLY the context below. \
Each context block is numbered: [1], [2], and so on.

Rules:
- If the answer is in the context, answer it and cite the block numbers you used, like [1] or [2][3].
- If the answer is not in the context, say you could not find it in the provided documents. Do not guess.
- Keep answers concise and concrete. Quote file paths when they help.
"""


def build_prompt(question: str, chunks) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {c.source})\n{c.text}")
    context = "\n\n".join(blocks)
    return f"{SYSTEM_PROMPT}\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"


class Generator:
    def __init__(self, settings):
        self.settings = settings

    # ---- Ollama (local) ----
    def _ollama(self, prompt: str, stream: bool):
        import httpx

        url = f"{self.settings.ollama_url}/api/generate"
        payload = {"model": self.settings.ollama_model, "prompt": prompt, "stream": stream}
        if not stream:
            resp = httpx.post(url, json=payload, timeout=180)
            resp.raise_for_status()
            return resp.json()["response"]
        # streaming: newline-delimited JSON, each line has {"response": "..."}
        with httpx.stream("POST", url, json=payload, timeout=180) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break

    # ---- OpenAI ----
    def _openai(self, prompt: str, stream: bool):
        import httpx, os

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": self.settings.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
        url = "https://api.openai.com/v1/chat/completions"
        if not stream:
            resp = httpx.post(url, headers=headers, json=payload, timeout=180)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        with httpx.stream("POST", url, headers=headers, json=payload, timeout=180) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0]["delta"]
                token = delta.get("content", "")
                if token:
                    yield token

    def _call(self, prompt: str, stream: bool):
        if self.settings.llm_provider == "openai":
            return self._openai(prompt, stream)
        return self._ollama(prompt, stream)

    def generate(self, question: str, chunks) -> str:
        """Non-streaming answer (used by the CLI)."""
        if not chunks:
            return "I could not find anything relevant in the ingested documents."
        return self._call(build_prompt(question, chunks), stream=False)

    def generate_stream(self, question: str, chunks):
        """Yield answer tokens one by one (used by the API for SSE)."""
        if not chunks:
            yield "I could not find anything relevant in the ingested documents."
            return
        yield from self._call(build_prompt(question, chunks), stream=True)
