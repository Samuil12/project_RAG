import os
import json
import numpy as np
import faiss                                          # pip install faiss-cpu
from sentence_transformers import SentenceTransformer # pip install sentence-transformers

INPUT_FILE = "extracted_chunks/chunks.json"
INDEX_FILE = "vector_store/med_index.faiss"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_chunks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs("vector_store", exist_ok=True)

    # 1) Load chunk records
    chunks = load_chunks(INPUT_FILE)

    # 2) Extract only the text to embed
    texts = [c["text"] for c in chunks]

    # 3) Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # 4) Convert text -> vectors
    # normalize_embeddings=True lets us use cosine similarity with inner product
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # 5) Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(embeddings)

    # 6) Save index + metadata
    faiss.write_index(index, INDEX_FILE)


if __name__ == "__main__":
    main()