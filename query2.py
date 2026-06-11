import json
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

import faiss
import requests
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class MedicalRAGSystem:
    def __init__(
        self,
        chunks_file: str = "extracted_chunks/chunks.json",
        index_file: str = "vector_store/med_index.faiss",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "gemma3:4b",
        llm_temperature: float = 0.1,
        top_k: int = 5,
    ):
        self.chunks_file = chunks_file
        self.index_file = index_file
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.llm_temperature = llm_temperature
        self.top_k = top_k

        self.chunks: List[Dict[str, Any]] = []
        self.index: Any = None
        self.embed_model: Any = None
        self._medicine_names: set = set()   # populated on load, used for fast name lookup

    # ── resource loading ───────────────────────────────────────────────────────

    def load_resources(self) -> None:
        """Loads chunks, FAISS index, and the embedding model."""
        try:
            print("  [1/3] Зареждане на метаданните...")
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.chunks = sorted(raw, key=lambda c: c.get("vector_id", 0))
            self._medicine_names = {
                c.get("medicine_name", "") for c in self.chunks if c.get("medicine_name")
            }

            print("  [2/3] Зареждане на FAISS индекс...")
            self.index = faiss.read_index(self.index_file)

            print("  [3/3] Зареждане на embedding модел...")
            self.embed_model = SentenceTransformer(self.model_name)

            print(
                f"✅ Системата е инициализирана. "
                f"Заредени {len(self.chunks)} откъса от {len(self._medicine_names)} лекарства.\n"
            )
        except FileNotFoundError as e:
            logging.error(f"Не е намерен файл: {e.filename}. Проверете пътищата.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Грешка при зареждане: {e}")
            sys.exit(1)

    # ── internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        return text.lower().strip()

    def _get_chunks_for_medicine(self, medicine_name: str) -> List[Dict[str, Any]]:
        """Returns every chunk that belongs to the given medicine."""
        return [c for c in self.chunks if c.get("medicine_name") == medicine_name]

    def _chunks_to_results(
        self, chunks: List[Dict[str, Any]], search_type: str
    ) -> List[Dict[str, Any]]:
        """Converts raw chunk dicts into the standard result format."""
        results = []
        for chunk in chunks:
            results.append(
                {
                    "rank": len(results) + 1,
                    "score": 1.0,
                    "vector_id": chunk.get("vector_id", "N/A"),
                    "medicine": chunk.get("medicine_name", "Неизвестно"),
                    "section": chunk.get("section_id", "Неизвестна"),
                    "text": chunk.get("text", ""),
                    "search_type": search_type,
                }
            )
        return results

    # ── Strategy 1 — direct name match ────────────────────────────────────────

    def _find_medicine_in_query(self, query: str) -> Optional[str]:
        """
        Returns the best medicine name found inside the query, or None.

        Rules:
        • Exact match (query == medicine name) wins immediately.
        • Substring match: medicine name must be ≥ 4 chars and appear literally
          in the query.  Among all substring matches the longest name wins
          (avoids shadowing "Aspirin Cardio" with plain "Aspirin").
        """
        q = self._normalize(query)
        best_name: Optional[str] = None
        best_len = 0

        for name in self._medicine_names:
            n = self._normalize(name)

            if n == q:
                return name  # exact — stop immediately

            if len(n) >= 4 and n in q and len(n) > best_len:
                best_len = len(n)
                best_name = name

        return best_name

    # ── Strategy 2 — keyword search ───────────────────────────────────────────

    def _keyword_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Token-overlap keyword search.

        Score = (|query_tokens ∩ chunk_tokens| / |query_tokens|) × medicine_boost

        medicine_boost = 1.5  if any query token matches the medicine name
                         1.0  otherwise

        Returns top_k results sorted by score descending.
        """
        query_tokens = set(self._normalize(query).split())
        if not query_tokens:
            return []

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for chunk in self.chunks:
            text_tokens = set(self._normalize(chunk.get("text", "")).split())
            if not text_tokens:
                continue

            overlap = len(query_tokens & text_tokens) / len(query_tokens)
            if overlap == 0:
                continue

            med_tokens = set(self._normalize(chunk.get("medicine_name", "")).split())
            boost = 1.5 if query_tokens & med_tokens else 1.0
            scored.append((overlap * boost, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for score, chunk in scored[: self.top_k]:
            results.append(
                {
                    "rank": len(results) + 1,
                    "score": round(score, 4),
                    "vector_id": chunk.get("vector_id", "N/A"),
                    "medicine": chunk.get("medicine_name", "Неизвестно"),
                    "section": chunk.get("section_id", "Неизвестна"),
                    "text": chunk.get("text", ""),
                    "search_type": "keyword",
                }
            )
        return results

    # ── Strategy 3 — vector search (FAISS) ───────────────────────────────────

    def _vector_search(self, query: str) -> List[Dict[str, Any]]:
        """Cosine-similarity search via the FAISS index."""
        q_vec = self.embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(q_vec, self.top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "rank": len(results) + 1,
                    "score": float(score),
                    "vector_id": chunk.get("vector_id", "N/A"),
                    "medicine": chunk.get("medicine_name", "Неизвестно"),
                    "section": chunk.get("section_id", "Неизвестна"),
                    "text": chunk.get("text", ""),
                    "search_type": "vector",
                }
            )
        return results

    # ── Fusion — Reciprocal Rank Fusion ───────────────────────────────────────

    @staticmethod
    def _rrf_merge(
        *result_lists: List[Dict[str, Any]], k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF):

            score(doc) = Σᵢ  1 / (k + rankᵢ(doc))

        Deduplicates by vector_id and combines arbitrary many ranked lists.
        k=60 is the standard constant that balances precision vs. recall.
        """
        rrf_scores: Dict[Any, float] = {}
        chunk_map: Dict[Any, Dict[str, Any]] = {}

        for result_list in result_lists:
            for item in result_list:
                vid = item["vector_id"]
                rrf_scores[vid] = rrf_scores.get(vid, 0.0) + 1.0 / (k + item["rank"])
                if vid not in chunk_map:
                    chunk_map[vid] = item  # keep first-seen entry (richest)

        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final: List[Dict[str, Any]] = []
        for rank, (vid, score) in enumerate(merged, start=1):
            entry = chunk_map[vid].copy()
            entry["rank"] = rank
            entry["score"] = round(score, 6)
            entry["search_type"] = "hybrid"
            final.append(entry)
        return final

    # ── Public retrieval entry-point ──────────────────────────────────────────

    def retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Three-stage hybrid retrieval pipeline:

          Stage 1 — Name match
            If the query is (or contains) a medicine name, fetch that medicine's
            chunks directly.  No embeddings needed; fast and exact.

          Stage 2 — Hybrid fallback
            Run both vector search (semantic) and keyword search (lexical) in
            parallel, then merge with Reciprocal Rank Fusion so that chunks
            ranked highly by *either* strategy bubble to the top.

        Returns
        -------
        (results, strategy_label)
            results        : list of dicts ready for build_prompt
            strategy_label : human-readable label for the CLI log line
        """
        # ── Stage 1: direct name lookup ────────────────────────────────────
        matched_name = self._find_medicine_in_query(query)
        if matched_name:
            raw = self._get_chunks_for_medicine(matched_name)
            results = self._chunks_to_results(raw[: self.top_k], "name_match")
            return results, f"🏷️  Директно съвпадение: «{matched_name}»"

        # ── Stage 2: vector + keyword → RRF ────────────────────────────────
        vector_results  = self._vector_search(query)
        keyword_results = self._keyword_search(query)
        hybrid          = self._rrf_merge(vector_results, keyword_results)[: self.top_k]
        return hybrid, "🔀 Хибридно търсене (вектор + ключови думи)"

    # ── Prompt construction ───────────────────────────────────────────────────

    def build_prompt(
        self, query: str, retrieved: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        context_parts = [
            f"ИЗТОЧНИК [ID: {r['vector_id']} | Лекарство: {r['medicine']} | Раздел: {r['section']}]:\n{r['text']}"
            for r in retrieved
        ]
        context_text = "\n\n".join(context_parts)

        system_prompt = (
            "Ти си професионален медицински асистент. Отговаряй на въпросите САМО на базата на предоставените източници.\n"
            "Ако отговорът не присъства в контекста, кажи: „Не знам."
            "Цитирай използваните откъси и номерата им, например [цитат] [#3]."
            "Отговори с не повече от 200 думи."
        )

        user_prompt = (
            f"Контекст (ИЗТОЧНИЦИ):\n{context_text}\n\n"
            f"ВЪПРОС НА ПОТРЕБИТЕЛЯ: {query}\n\n"
            f"Моля, напиши своя подробен отговор (с цитати) тук:"
        )

        return system_prompt, user_prompt

    # ── Ollama streaming call ─────────────────────────────────────────────────

    def call_ollama_streaming(self, system: str, user: str) -> str:
        url = f"{self.ollama_base_url}/api/chat"
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": True,
            "keep_alive": "1h",
            "options": {
                "temperature": self.llm_temperature,
                "num_ctx": 4096,
                "stop": ["ВЪПРОС НА ПОТРЕБИТЕЛЯ", "ВЪПРОС НА ПОТРЕБИТЕЛЯ:", "ВЪПРОС:"],
            },
        }

        print("\n-- Отговор на асистента --")
        full_response: List[str] = []

        try:
            with requests.post(url, json=payload, stream=True, timeout=(5, None)) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        print(content, end="", flush=True)
                        full_response.append(content)
                        if chunk.get("done"):
                            break
        except requests.exceptions.ConnectionError:
            logging.error(
                "\nГрешка: Не може да се свърже с Ollama. "
                "Проверете дали услугата работи (http://localhost:11434)."
            )
        except Exception as e:
            logging.error(f"\nНеочаквана грешка: {e}")

        print("\n" + "=" * 60)
        return "".join(full_response)

    # ── Interactive CLI ───────────────────────────────────────────────────────

    def run_cli(self) -> None:
        self.load_resources()
        print("Въведете вашия въпрос (или 'exit' / 'quit' за изход).")

        while True:
            try:
                query = input("\nВъпрос: ").strip()
            except KeyboardInterrupt:
                break

            if not query or query.lower() in {"exit", "quit"}:
                break

            # ── Retrieval ──────────────────────────────────────────────────
            retrieved, strategy = self.retrieve(query)

            print(f"\n[{strategy} — намерени {len(retrieved)} откъса]")
            for r in retrieved[:3]:
                print(
                    f"  #{r['rank']} [{r['search_type']:10s}] "
                    f"ID:{r['vector_id']} | {r['medicine'][:11]}... ({r['section']}) "

                )
            if len(retrieved) > 3:
                print(f"   ... и още {len(retrieved) - 3} източника.")

            # ── Generate ───────────────────────────────────────────────────
            sys_p, user_p = self.build_prompt(query, retrieved)
            self.call_ollama_streaming(sys_p, user_p)


if __name__ == "__main__":
    rag_system = MedicalRAGSystem()
    rag_system.run_cli()