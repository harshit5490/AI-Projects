import json

from app.embeddings.service import EmbeddingService
from app.retrieval.retriever import Retriever
from app.vectorstore.store import PineconeVectorStore


DOCUMENT_ID = "25c2a1e6-9561-4d42-ae71-c62af66e7fae"


def load_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def is_relevant(
    result: dict,
    expected_phrases: list[str],
) -> bool:
    text = result["text"].lower()

    matched_phrases = sum(
        phrase.lower() in text
        for phrase in expected_phrases
    )

    return matched_phrases >= 2


def reciprocal_rank(
    results: list[dict],
    expected_keywords: list[str],
) -> float:

    for rank, result in enumerate(results, start=1):
        if is_relevant(result, expected_keywords):
            return 1 / rank

    return 0.0


def hit_at_k(
    results: list[dict],
    expected_keywords: list[str],
    k: int,
) -> int:

    for result in results[:k]:
        if is_relevant(result, expected_keywords):
            return 1

    return 0


def main():

    questions = load_questions(
        "experiments/data/retrieval_questions.json"
    )

    embedding_service = EmbeddingService()

    vector_store = PineconeVectorStore(
        index_name="rag-document-index",
        namespace="rag-documents",
    )

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    total_mrr = 0.0
    total_hit_at_1 = 0
    total_hit_at_3 = 0
    total_hit_at_5 = 0

    for item in questions:

        question = item["question"]
        expected_keywords = item["expected_keywords"]

        results = retriever.retrieve(
            query=question,
            top_k=5,
            metadata_filter={
                "document_id": {
                    "$eq": DOCUMENT_ID
                }
            },
        )

        mrr = reciprocal_rank(
            results,
            expected_keywords,
        )

        hit_1 = hit_at_k(
            results,
            expected_keywords,
            1,
        )

        hit_3 = hit_at_k(
            results,
            expected_keywords,
            3,
        )

        hit_5 = hit_at_k(
            results,
            expected_keywords,
            5,
        )

        total_mrr += mrr
        total_hit_at_1 += hit_1
        total_hit_at_3 += hit_3
        total_hit_at_5 += hit_5

        print("\nQuestion:", question)

        print("Top result:")

        if results:
            print(
                f"  Score: {results[0]['score']:.4f}"
            )
            print(
                f"  Chunk: {results[0]['chunk_index']}"
            )

        print(f"  Hit@1: {hit_1}")
        print(f"  Hit@3: {hit_3}")
        print(f"  Hit@5: {hit_5}")
        print(f"  MRR: {mrr:.4f}")

    count = len(questions)

    print("\n" + "=" * 50)
    print("RETRIEVAL EVALUATION")
    print("=" * 50)

    print(
        f"Hit@1: {total_hit_at_1 / count:.4f}"
    )

    print(
        f"Hit@3: {total_hit_at_3 / count:.4f}"
    )

    print(
        f"Hit@5: {total_hit_at_5 / count:.4f}"
    )

    print(
        f"MRR: {total_mrr / count:.4f}"
    )


if __name__ == "__main__":
    main()