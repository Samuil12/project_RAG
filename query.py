import json
import logging
import sys
import faiss
import requests
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class MedicalRAGSystem:
    def __init__(
        self,
        chunks_file: str = "extracted_chunks/chunks.json",
        index_file: str = "vector_store/med_index.faiss",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "gemma3:4b", # "llama3.2:1b", 
        llm_temperature: float = 0.1,
        top_k: int = 5
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

    def load_resources(self) -> None:
        """Loads chunks, FAISS index, and the embedding model with error handling."""
        try:
            print("  [1/3] Зареждане на метаданните...")
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Ensure sorting matches FAISS index order
            self.chunks = sorted(raw, key=lambda c: c.get("vector_id", 0))

            print("  [2/3] Зареждане на FAISS индекс...")
            self.index = faiss.read_index(self.index_file)

            print("  [3/3] Зареждане на embedding модел...")
            self.embed_model = SentenceTransformer(self.model_name)
            
            print("✅ Системата е инициализирана успешно.\n")
        except FileNotFoundError as e:
            logging.error(f"Не е намерен файл: {e.filename}. Моля, проверете пътищата.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Грешка при зареждане на ресурсите: {e}")
            sys.exit(1)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieves top_k similar chunks from the FAISS index."""
        # L2-normalize the query for Cosine Similarity
        q_vec = self.embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        
        scores, indices = self.index.search(q_vec, self.top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks): 
                continue
            
            chunk = self.chunks[idx]
            results.append({
                "rank": len(results) + 1,
                "score": float(score),
                "vector_id": chunk.get("vector_id", "N/A"),
                "medicine": chunk.get("medicine_name", "Неизвестно"),
                "section": chunk.get("section_id", "Неизвестна"),
                "text": chunk.get("text", ""),
            })
        return results

    def build_prompt(self, query: str, retrieved: List[Dict[str, Any]]) -> Tuple[str, str]:
        context_parts = [
            f"ИЗТОЧНИК [ID: {r['vector_id']} | Лекарство: {r['medicine']} | Раздел: {r['section']}]:\n{r['text']}"
            for r in retrieved
        ]
        context_text = "\n\n".join(context_parts)

        system_prompt = (
            "Ти си професионален медицински асистент. Отговаряй на въпросите САМО на базата на предоставените източници.\n"
            "Ако отговорът не присъства в контекста, кажи: „Не знам.“."
            "Цитирай използваните откъси и номерата им, например [цитат] [#3]."
            "Отговори с не повече от 200 думи."
        )

        user_prompt = (
            f"Контекст (ИЗТОЧНИЦИ):\n{context_text}\n\n"
            f"ВЪПРОС НА ПОТРЕБИТЕЛЯ: {query}\n\n"
            f"Моля, напиши своя подробен отговор (с цитати) тук:"
        )

        return system_prompt, user_prompt

    def call_ollama_streaming(self, system: str, user: str) -> str:
        """Calls the Ollama API and streams the response."""
        url = f"{self.ollama_base_url}/api/chat"
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": True,
            "keep_alive": "1h", 
            "options": {
                "temperature": self.llm_temperature,
                "num_ctx": 4096,
                "stop": [
                    "ВЪПРОС НА ПОТРЕБИТЕЛЯ", 
                    "ВЪПРОС НА ПОТРЕБИТЕЛЯ:", 
                    "ВЪПРОС:"
                ]
            }
        }

        print("\n-- Отговор на асистента --")
        full_response = []
        
        try:
            with requests.post(url, json=payload, stream=True, timeout=(5, None)) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        print(content, end="", flush=True)
                        full_response.append(content)
                        if chunk.get("done"):
                            break
        except requests.exceptions.ConnectionError:
            logging.error("\nГрешка: Не може да се свърже с Ollama. Проверете дали услугата работи (http://localhost:11434).")
        except Exception as e:
            logging.error(f"\nНеочаквана грешка при комуникация с Ollama: {e}")
        
        print("\n" + "="*60)
        return "".join(full_response)

    def run_cli(self):
        """Runs the interactive command line interface."""
        self.load_resources()
        print("Въведете вашия въпрос (или 'exit'/'quit' за изход).")

        while True:
            try:
                query = input("\nВъпрос: ").strip()
            except KeyboardInterrupt:
                break # Graceful exit on Ctrl+C
            
            if not query or query.lower() in ["exit", "quit"]:
                break

            # 1. Retrieve
            retrieved = self.retrieve(query)
            
            # Print search context
            print(f"\n[Намерени {len(retrieved)} релевантни откъса]")
            for r in retrieved[:3]: # Limit to top 3 in logs to avoid terminal clutter
                print(f" - ID: {r['vector_id']} | {r['medicine']} ({r['section']}) | Сходство: {r['score']:.4f}")
            if len(retrieved) > 3:
                print(f"   ... и още {len(retrieved) - 3} източника.")

            # 2. Build Prompt
            sys_p, user_p = self.build_prompt(query, retrieved)

            # 3. Generate (Streamed)
            self.call_ollama_streaming(sys_p, user_p)

if __name__ == "__main__":
    rag_system = MedicalRAGSystem()
    rag_system.run_cli()


'''
"Ти си професионален медицински асистент. Отговаряй на въпросите САМО на базата на предоставените източници.\n"
            "Изисквания:\n"
            "1. Изчисти използвания текст от технически грешки, като запазиш точния смисъл.\n"
            "2. Цитирай източника в края на всяко твърдение в този точен формат: (Източник ID: [vector_id], Лекарство: [medicine]).\n"
            "3. Ако в източниците няма отговор, кажи ясно, че не разполагаш с тази информация.\n"
            "4. Отговаряй стриктно на български език.\n"
            "5. Бъди кратък и точен, използвай до 300 думи.\n"
            "6. ВАЖНО: Отговори САМО на текущия въпрос и СПРИ. В никакъв случай НЕ генерирай нови въпроси!"
'''