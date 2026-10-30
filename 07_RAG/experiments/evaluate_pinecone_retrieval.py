
import json
import os
from pathlib import Path

from app.embeddings.service import EmbeddingService
from app.retrieval.retriever import Retriever
from app.vectorstore.store import PineconeVectorStore
from experiments.evaluate_retrieval import build_retrieval_report, print_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_questions.json"
)

EVALUATION_NAMESPACE = "rag-evaluation"
TOP_K = 5


def main() -> None:
    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        questions = json.load(file)

    # Empty relevant sets are reserved for abstention evaluation.
    answerable_questions = [
        item for item in questions
        if item["answerable"] and item["relevant_chunk_ids"]
    ]

    if not answerable_questions:
        raise ValueError("No answerable evaluation questions found.")

    index_name = os.getenv(
        "PINECONE_INDEX_NAME",
        "rag-document-index",
    )

    embedding_service = EmbeddingService()

    vector_store = PineconeVectorStore(
        index_name=index_name,
        namespace=EVALUATION_NAMESPACE,
    )

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    rankings: list[list[str]] = []
    relevant_sets: list[set[str]] = []

    print(f"Evaluation namespace: {EVALUATION_NAMESPACE}")
    print(f"Questions evaluated: {len(answerable_questions)}")

    for item in answerable_questions:
        results = retriever.retrieve(
            query=item["question"],
            top_k=TOP_K,
            similarity_threshold=None,
        )

        retrieved_ids = [
            result["id"] for result in results
        ]
        relevant_ids = set(item["relevant_chunk_ids"])

        rankings.append(retrieved_ids)
        relevant_sets.append(relevant_ids)

        relevant_ranks = [
            rank
            for rank, chunk_id in enumerate(
                retrieved_ids, start=1
            )
            if chunk_id in relevant_ids
        ]

        print(f"\n{item['id']}: {item['question']}")
        print(f"Relevant chunk IDs: {sorted(relevant_ids)}")
        print(f"Retrieved chunk IDs: {retrieved_ids}")
        print(
            "First relevant rank:",
            min(relevant_ranks) if relevant_ranks else "Not found",
        )

    report = build_retrieval_report(
        rankings=rankings,
        relevant_chunk_ids=relevant_sets,
    )

    print_report(report)


if __name__ == "__main__":
    main()
