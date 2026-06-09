from query import MedicalRAGSystem
import numpy as np

# example questions and answers
evaluation_set = [
    {
        "question": "Как се прилага бактробан маз за нос?",
        "relevant_chunks": [7416]
    },

    {
        "question": "Къде се съхранява синакалцет акорд?",
        "relevant_chunks": [6065]
    },

    {
        "question": "За какво спомага кандесартан актавис?",
        "relevant_chunks": [12254]
    },
    {
        "question": "Какво представлява би-профенид?",
        "relevant_chunks": [6069]
    }
]

def evaluate_rag(rag: MedicalRAGSystem):
    '''Evaluates the metrics recall, MRR and the answer's semantic score'''
    n = len(evaluation_set)

    # 1. retrieval metrics
    total_mrr = 0.0
    total_recall = 0.0

    retrieved_sets = []

    for sample in evaluation_set:
        question = sample["question"]
        relevant = set(sample["relevant_chunks"])

        retrieved = rag.retrieve(question)
        retrieved_ids = [r["vector_id"] for r in retrieved]

        retrieved_sets.append(retrieved_ids)

        # recall at k
        hits = len(set(retrieved_ids) & relevant)
        total_recall += hits / len(relevant)

        # MRR
        rank = next(
            (i for i, cid in enumerate(retrieved_ids, start=1) if cid in relevant),
            None
        )

        if rank is not None:
            total_mrr += 1.0 / rank


    # results
    return {
        "mrr": total_mrr / n,
        "recall_at_k": total_recall / n
    }


if __name__ == "__main__":
    rag_system = MedicalRAGSystem()
    rag_system.load_resources()

    results = evaluate_rag(rag_system)

    print("mrr:", results["mrr"])
    print(f"recall at k = {rag_system.top_k}:", results["recall_at_k"])