import json
import textwrap
import faiss
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
CHUNKS_FILE  = "extracted_chunks/chunks.json"
INDEX_FILE   = "vector_store/med_index.faiss"
EMBED_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Speed optimization: 
# We fetch 10 but only send the top 4 most relevant to the LLM 
# This makes the "prefill" phase 2-3x faster.
TOP_K_FETCH = 10 
TOP_K_LLM   = 4  

OLLAMA_MODEL = "gemma4e4B:latest" # Or "llama3.2:1b" for instant speed
OLLAMA_URL   = "http://localhost:11434/api/chat"

# ──────────────────────────────────────────────────────────────
# LOADING
# ──────────────────────────────────────────────────────────────

def load_resources():
    print("  Loading models (this happens once)...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = sorted(json.load(f), key=lambda c: c["vector_id"])
    
    index = faiss.read_index(INDEX_FILE)
    bi_encoder = SentenceTransformer(EMBED_MODEL)
    
    # Tiny reranker (~30MB) - Extremely fast, improves accuracy and speed
    reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
    
    return chunks, index, bi_encoder, reranker

# ──────────────────────────────────────────────────────────────
# RETRIEVAL + RERANKING (The Speed Secret)
# ──────────────────────────────────────────────────────────────

def retrieve_and_rank(query, chunks, index, bi_encoder, reranker):
    # 1. Fast Vector Search (Top 10)
    q_vec = bi_encoder.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_vec, TOP_K_FETCH)
    
    candidate_chunks = [chunks[i] for i in indices[0] if i >= 0]
    
    # 2. Reranking (Sort the 10 candidates by actual relevance to the question)
    # This ensures that even if the vector search is slightly off, the LLM gets the best info
    pairs = [[query, c['text']] for c in candidate_chunks]
    rerank_scores = reranker.predict(pairs)
    
    # Attach scores and sort
    for i, score in enumerate(rerank_scores):
        candidate_chunks[i]['rerank_score'] = score
        
    sorted_chunks = sorted(candidate_chunks, key=lambda x: x['rerank_score'], reverse=True)
    
    # Return only the top N to the LLM to save time
    return sorted_chunks[:TOP_K_LLM]

# ──────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────

def stream_answer(query, relevant_chunks):
    # Minimalist prompt to save tokens/time
    context = ""
    for i, c in enumerate(relevant_chunks):
        context += f"[Doc {i+1}] (Med: {c['medicine_name']}): {c['text']}\n\n"

    system_prompt = (
        "You are a medical assistant. Answer using the provided documents.\n"
        "1. Use direct quotes in \"double quotes\".\n"
        "2. Cite docs as [Doc X].\n"
        "3. If unknown, say 'Not found'."
    )
    
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\nQuestion: {query}"}
        ],
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
            "num_thread": 8  # Set to your CPU core count if no GPU
        }
    }

    print("\n  -- Answer --")
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=None) as res:
            for line in res.iter_lines():
                if line:
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    print(content, end="", flush=True)
    except Exception as e:
        print(f"\nError: {e}")
    print("\n" + "─"*50)

def main():
    chunks, index, bi_encoder, reranker = load_resources()
    print("\nReady! Type your question.")

    while True:
        query = input("\n> ").strip()
        if query.lower() in ["exit", "quit", ""]: break

        print("  Searching...")
        best_chunks = retrieve_and_rank(query, chunks, index, bi_encoder, reranker)
        
        stream_answer(query, best_chunks)

if __name__ == "__main__":
    main()