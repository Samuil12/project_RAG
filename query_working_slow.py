import json
import textwrap
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────

CHUNKS_FILE  = "extracted_chunks/chunks.json"
INDEX_FILE   = "vector_store/med_index.faiss"
MODEL_NAME   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K        = 4          # As requested: 10 closest chunks

LLM_PROVIDER     = "ollama"
OLLAMA_MODEL     = "gemma4e4B:latest" 
#OLLAMA_MODEL     = "deepseek-r1:8b"
OLLAMA_BASE_URL  = "http://localhost:11434"
LLM_TEMPERATURE  = 0.1     # Lower temperature for higher factual accuracy

# ──────────────────────────────────────────────────────────────
# RESOURCE LOADING
# ──────────────────────────────────────────────────────────────

def load_resources():
    print("  [1/3] Loading chunk metadata...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    chunks = sorted(raw, key=lambda c: c["vector_id"])

    print("  [2/3] Loading FAISS index...")
    index = faiss.read_index(INDEX_FILE)

    print("  [3/3] Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    return chunks, index, model

# ──────────────────────────────────────────────────────────────
# RETRIEVAL
# ──────────────────────────────────────────────────────────────

def retrieve(query: str, chunks: list, index, model, top_k: int = TOP_K) -> list[dict]:
    # L2-normalize the query for Cosine Similarity (IndexFlatIP)
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    
    scores, indices = index.search(q_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0: continue
        results.append({
            "rank": len(results) + 1,
            "score": float(score),
            "medicine": chunks[idx]["medicine_name"],
            "section": chunks[idx]["section_id"],
            "text": chunks[idx]["text"],
        })
    return results

# ──────────────────────────────────────────────────────────────
# PROMPT BUILDING
# ──────────────────────────────────────────────────────────────

def build_prompt(query: str, retrieved: list[dict]):
    context_parts = []
    for r in retrieved:
        context_parts.append(
            f"SOURCE {r['rank']} (Medicine: {r['medicine']}, Section: {r['section']}):\n{r['text']}"
        )
    
    context_text = "\n\n".join(context_parts)

    system_prompt = (
        "Ти си медицински асистент и отговаряш на въпросите зададени от потребителя заедно с източниците.\n"
        "Изисквания:\n"
        "1. Use EXACT QUOTES from the text in double quotation marks (\"...\").\n"
        "2. Cite the source id number after each quote.\n"
        "3. If the sources do not contain the answer, say you don't know.\n"
        "4. Answer in the same language as the user query."
    )

    user_prompt = (
        f"SOURCES:\n{context_text}\n\n"
        f"USER QUESTION: {query}\n\n"
        f"Detailed Answer (with quotes):"
    )

    return system_prompt, user_prompt

# ──────────────────────────────────────────────────────────────
# LLM CALL (With Fix for Timeout)
# ──────────────────────────────────────────────────────────────

def call_ollama_streaming(system: str, user: str):
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "stream": True,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "num_ctx": 8192  # Increased context window for 10 chunks
        }
    }

    print("\n-- LLM Response --")
    full_response = []
    
    try:
        # timeout=None tells 'requests' to wait forever for the LLM to process
        with requests.post(url, json=payload, stream=True, timeout=None) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    print(content, end="", flush=True)
                    full_response.append(content)
                    if chunk.get("done"):
                        break
    except Exception as e:
        print(f"\n\nError connecting to Ollama: {e}")
    
    print("\n" + "="*50 + "\n")
    return "".join(full_response)

# ──────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────

def main():
    chunks, index, model = load_resources()
    print("\nSystem ready. Type your question or 'exit'.")

    while True:
        query = input("\nQuestion: ").strip()
        if not query or query.lower() in ["exit", "quit"]:
            break

        # 1. Retrieve
        retrieved = retrieve(query, chunks, index, model)
        
        # Print what we found for transparency
        print(f"\n[Retrieved {len(retrieved)} relevant segments]")
        for r in retrieved:
            print(f" - {r['medicine']} ({r['section']}) | Score: {r['score']:.4f}")

        # 2. Build Prompt
        sys_p, user_p = build_prompt(query, retrieved)

        # 3. Generate (Streamed)
        call_ollama_streaming(sys_p, user_p)

if __name__ == "__main__":
    main()