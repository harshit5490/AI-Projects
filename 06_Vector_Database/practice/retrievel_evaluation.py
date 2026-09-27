documents = {
    "doc_001": "FastAPI is a modern Python framework for building APIs.",
    "doc_002": "Python is widely used for machine learning and data science.",
    "doc_003": "PostgreSQL is a relational database management system.",
    "doc_004": "Transformers are neural network architectures widely used in NLP.",
    "doc_005": "Docker packages applications and their dependencies into containers.",
}


evaluation_data = [
    {
        "query": "How can I build a Python API?",
        "relevant_docs": ["doc_001"],
    },
    {
        "query": "Which technology is used for natural language processing?",
        "relevant_docs": ["doc_004"],
    },
    {
        "query": "What is used to containerize an application?",
        "relevant_docs": ["doc_005"],
    },
    {
        "query": "Which database is relational?",
        "relevant_docs": ["doc_003"],
    },
]


def hit_rate_at_k(retrieved_docs, relevant_docs, k):

    top_k = retrieved_docs[:k]

    return int(
        any(doc in relevant_docs for doc in top_k)
    )


def recall_at_k(retrieved_docs, relevant_docs, k):

    top_k = retrieved_docs[:k]

    relevant_retrieved = set(top_k) & set(relevant_docs)

    return len(relevant_retrieved) / len(relevant_docs)


def precision_at_k(retrieved_docs, relevant_docs, k):

    top_k = retrieved_docs[:k]

    relevant_retrieved = set(top_k) & set(relevant_docs)

    return len(relevant_retrieved) / k


def reciprocal_rank(retrieved_docs, relevant_docs):

    for rank, doc_id in enumerate(retrieved_docs, start=1):

        if doc_id in relevant_docs:
            return 1 / rank

    return 0


# Example retrieval results
retrieval_results = {
    "How can I build a Python API?":
        ["doc_001", "doc_002", "doc_003"],

    "Which technology is used for natural language processing?":
        ["doc_002", "doc_004", "doc_005"],

    "What is used to containerize an application?":
        ["doc_005", "doc_002", "doc_001"],

    "Which database is relational?":
        ["doc_003", "doc_001", "doc_002"],
}


K = 3

hit_scores = []
recall_scores = []
precision_scores = []
rr_scores = []


for item in evaluation_data:

    query = item["query"]
    relevant_docs = item["relevant_docs"]

    retrieved_docs = retrieval_results[query]

    hit = hit_rate_at_k(
        retrieved_docs,
        relevant_docs,
        K,
    )

    recall = recall_at_k(
        retrieved_docs,
        relevant_docs,
        K,
    )

    precision = precision_at_k(
        retrieved_docs,
        relevant_docs,
        K,
    )

    rr = reciprocal_rank(
        retrieved_docs,
        relevant_docs,
    )

    hit_scores.append(hit)
    recall_scores.append(recall)
    precision_scores.append(precision)
    rr_scores.append(rr)

    print("\nQuery:", query)

    print("Retrieved:", retrieved_docs)

    print("Relevant:", relevant_docs)

    print(f"Hit@{K}: {hit}")

    print(f"Recall@{K}: {recall:.3f}")

    print(f"Precision@{K}: {precision:.3f}")

    print(f"Reciprocal Rank: {rr:.3f}")


# Overall metrics

hit_rate = sum(hit_scores) / len(hit_scores)
recall = sum(recall_scores) / len(recall_scores)
precision = sum(precision_scores) / len(precision_scores)
mrr = sum(rr_scores) / len(rr_scores)


print("\n" + "=" * 50)
print("OVERALL RETRIEVAL EVALUATION")
print("=" * 50)

print(f"Hit@{K}:       {hit_rate:.3f}")
print(f"Recall@{K}:    {recall:.3f}")
print(f"Precision@{K}: {precision:.3f}")
print(f"MRR:           {mrr:.3f}")