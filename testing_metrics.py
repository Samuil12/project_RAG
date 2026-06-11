from query2 import MedicalRAGSystem

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
    },
    {
        "question": "Какво съдържа аклиеф?",
        "relevant_chunks": [10577]
    },
    {
        "question": "Кога да не прилагам цефуроксим генамид?",
        "relevant_chunks": [12762, 12743]
    },
    {
        "question": "Какви са страничните ефекти на бортезомиб сандоз?",
        "relevant_chunks": [6029, 13323]
    },
    {
        "question": "Какво да направя, ако пропусна приема на таблетка белуша?",
        "relevant_chunks": [17411]
    },
    {
        "question": "Какво може да предизвика консервантът от лекарството бримонидин тартарат?",
        "relevant_chunks": [16872]
    },
    {
        "question": "Кога не трябва да използвам бокотюр?",
        "relevant_chunks": [15971]
    },
    {
        "question": "Как да съхранявам инхалатора на бибекфо?",
        "relevant_chunks": [14650]
    },
    {
        "question": "Мога ли да приемам алифлузин при бременност?",
        "relevant_chunks": [13193]
    },
    {
        "question": "С какво помага кларитин?",
        "relevant_chunks": [5541]
    },
    {
        "question": "Агодеприн предизвиква ли алергични реакции?",
        "relevant_chunks": [5080]
    },
    {
        "question": "Цефасел изисква ли специални условия за съхранение?",
        "relevant_chunks": [4018]
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

        retrieved, _ = rag.retrieve(question)
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
    rag_system.top_k = 10
    rag_system.load_resources()

    results = evaluate_rag(rag_system)

    print("mrr:", results["mrr"])
    print(f"recall at k = {rag_system.top_k}:", results["recall_at_k"])