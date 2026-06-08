"""
query.py  –  RAG query interface for project_RAG
=================================================
Drop this file in the project root (next to build_vector_index.py).

Flow:
  1. Load FAISS index  (vector_store/med_index.faiss)
  2. Load chunk metadata (extracted_chunks/chunks.json)
  3. Loop: accept user question
  4. Embed question with the same multilingual model
  5. Cosine-search top-10 chunks  (IndexFlatIP on L2-normalised vecs = cosine)
  6. Build a prompt that instructs the LLM to quote original text
  7. Call local Ollama (gemma4e4B:latest) – no API key needed
  8. Print the answer

Requirements:
    pip install faiss-cpu sentence-transformers requests

    Ollama must be running locally:
        https://ollama.com/download
        ollama pull gemma4e4B:latest   # or whichever tag you use
        ollama serve                   # starts the server on port 11434
"""

import json
import textwrap
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────
# CONFIGURATION  (edit here if your paths differ)
# ──────────────────────────────────────────────────────────────

CHUNKS_FILE  = "extracted_chunks/chunks.json"
INDEX_FILE   = "vector_store/med_index.faiss"
MODEL_NAME   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K        = 10          # number of nearest neighbours to retrieve

# LLM settings – local Ollama server.
# Change OLLAMA_MODEL to any model you have pulled, e.g. "llama3.2" or "mistral".
# Change OLLAMA_BASE_URL if Ollama is running on a different host/port.
LLM_PROVIDER     = "ollama"
OLLAMA_MODEL     = "deepseek-r1:8b" #"gemma4e4B:latest" 
OLLAMA_BASE_URL  = "http://localhost:11434"   # default Ollama address
LLM_TEMPERATURE  = 0.2

# ──────────────────────────────────────────────────────────────
# LOADING
# ──────────────────────────────────────────────────────────────

def load_resources():
    """
    Returns (chunks_list, faiss_index, embed_model).

    chunks_list is sorted by vector_id so that chunks_list[i]
    corresponds to the i-th row in the FAISS index.
    """
    print("  Loading chunk metadata …")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Sort by vector_id to guarantee alignment with the FAISS index rows
    chunks = sorted(raw, key=lambda c: c["vector_id"])

    print("  Loading FAISS index …")
    index = faiss.read_index(INDEX_FILE)

    print("  Loading embedding model …")
    model = SentenceTransformer(MODEL_NAME)

    assert index.ntotal == len(chunks), (
        f"Mismatch: FAISS has {index.ntotal} vectors "
        f"but chunks.json has {len(chunks)} records."
    )
    return chunks, index, model


# ──────────────────────────────────────────────────────────────
# RETRIEVAL  (cosine similarity via FAISS IndexFlatIP)
# ──────────────────────────────────────────────────────────────

def retrieve(query: str, chunks: list, index, model, top_k: int = TOP_K) -> list[dict]:
    """
    Embeds *query*, L2-normalises it (same as at index-build time),
    and returns the top_k most similar chunks with their scores.
    """
    q_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,   # must match build_vector_index.py
    ).astype("float32")              # FAISS expects float32

    # IndexFlatIP + normalised vectors → inner product = cosine similarity
    scores, indices = index.search(q_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:            # FAISS returns -1 when fewer results exist
            continue
        chunk = chunks[idx]
        results.append({
            "rank":      len(results) + 1,
            "score":     float(score),          # cosine ∈ [0, 1] for normalised vecs
            "vector_id": chunk["vector_id"],
            "medicine":  chunk["medicine_name"],
            "section":   chunk["section_id"],
            "text":      chunk["text"],
        })
    return results


# ──────────────────────────────────────────────────────────────
# PROMPT BUILDING
# ──────────────────────────────────────────────────────────────

def build_prompt(query: str, retrieved: list[dict]) -> str:
    """
    Formats the retrieved chunks into a prompt that instructs the LLM
    to answer grounded in the source text and use exact quotes.
    """
    excerpts = []
    for r in retrieved:
        header = (
            f"[Excerpt {r['rank']}]  "
            f"Medicine: {r['medicine']} | "
            f"Section: {r['section']} | "
            f"Cosine similarity: {r['score']:.4f}"
        )
        excerpts.append(f"{header}\n{r['text']}")

    context = "\n\n" + ("─" * 60) + "\n\n".join(excerpts) + "\n\n" + ("─" * 60)

    system = (
        "You are a precise medical information assistant. "
        "You answer questions ONLY from the provided excerpts of medication leaflets. "
        "Rules:\n"
        "1. Include verbatim quotes from the source text inside double quotation marks (\"…\").\n"
        "2. After each quote state which excerpt it comes from, e.g. [Excerpt 3].\n"
        "3. If the answer is not found in the excerpts, say so explicitly.\n"
        "4. Never invent or paraphrase medical facts – only quote and summarise what the excerpts say.\n"
        "5. Write in the same language as the user's question."
    )

    user = (
        f"RETRIEVED EXCERPTS FROM MEDICATION LEAFLETS:\n"
        f"{context}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"Answer the question in detail, supporting every claim with a direct "
        f"quote (in \"double quotes\") from the excerpts above:"
    )

    return system, user


# ──────────────────────────────────────────────────────────────
# LLM CALL  (local Ollama)
# ──────────────────────────────────────────────────────────────

import requests as _requests

def call_ollama(system: str, user: str) -> str:
    """
    POST to Ollama's /api/chat endpoint (streaming=True).
    Ollama merges a separate 'system' role transparently for models
    that support it (Gemma 3 does); for others it is prepended to
    the first user turn automatically.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": True,                     # get the whole reply at once
        "options": {
            "temperature": LLM_TEMPERATURE,
            "num_ctx": 8192,                 # context window; raise if chunks are large
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }

    try:
        resp = _requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
    except _requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Is 'ollama serve' running?"
        )
    except _requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama returned an error: {e}\n{resp.text}")

    data = resp.json()
    # Ollama /api/chat response shape: {"message": {"role": "assistant", "content": "..."}, ...}
    return data["message"]["content"]


def ask_llm(system: str, user: str) -> str:
    """Dispatch to the configured LLM provider."""
    if LLM_PROVIDER == "ollama":
        return call_ollama(system, user)

    # ── Fallback providers (uncomment + pip install as needed) ──
    # if LLM_PROVIDER == "openai":
    #     from openai import OpenAI
    #     client = OpenAI()
    #     r = client.chat.completions.create(
    #         model="gpt-4o-mini", temperature=LLM_TEMPERATURE,
    #         messages=[{"role":"system","content":system},{"role":"user","content":user}])
    #     return r.choices[0].message.content
    #
    # if LLM_PROVIDER == "anthropic":
    #     import anthropic
    #     client = anthropic.Anthropic()
    #     r = client.messages.create(
    #         model="claude-3-5-haiku-20241022", max_tokens=2048, system=system,
    #         messages=[{"role":"user","content":user}])
    #     return r.content[0].text

    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}")


# ──────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ──────────────────────────────────────────────────────────────

SEP = "═" * 70

def print_retrieved(results: list[dict]) -> None:
    print(f"\n{SEP}")
    print(f"  TOP {len(results)} RETRIEVED CHUNKS  (cosine similarity ↓)")
    print(SEP)
    for r in results:
        print(
            f"\n  [{r['rank']:>2}]  medicine={r['medicine']}  "
            f"section={r['section']}  score={r['score']:.4f}"
        )
        preview = r["text"][:280].replace("\n", " ")
        print(textwrap.fill(f"       {preview}…", width=78, subsequent_indent="       "))


def print_answer(answer: str) -> None:
    print(f"\n{SEP}")
    print("  LLM ANSWER")
    print(SEP)
    for line in answer.splitlines():
        print(textwrap.fill(line, width=78) if line.strip() else "")
    print()


# ──────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────

def main():
    print(f"\n{SEP}")
    print("  Medical RAG — initialising …")
    print(SEP)
    chunks, index, model = load_resources()
    print(f"\n  Ready.  {len(chunks)} chunks | {index.ntotal} FAISS vectors | top-k = {TOP_K}")
    print(f"  LLM provider: {LLM_PROVIDER}")
    print(f"  Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        # ── Step 1: vector search ───────────────────────────────
        results = retrieve(query, chunks, index, model, top_k=TOP_K)
        print_retrieved(results)

        # ── Step 2: LLM generation ──────────────────────────────
        print(f"\n  Calling {LLM_PROVIDER} …")
        system_prompt, user_prompt = build_prompt(query, results)
        answer = ask_llm(system_prompt, user_prompt)
        print_answer(answer)


if __name__ == "__main__":
    main()