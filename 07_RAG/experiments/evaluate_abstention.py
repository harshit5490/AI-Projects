
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from app.embeddings.service import EmbeddingService
from app.vectorstore.store import PineconeVectorStore
from app.retrieval.retriever import Retriever
from app.evaluation.abstention_policy import should_abstain
from app.evaluation.abstention_metrics import abstention_metrics


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = PROJECT_ROOT / "data" / "evaluation" / "evaluation_questions.json"

THRESHOLD = 0.35
TOP_K = 5


def main() -> None:
    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    embedding_service = EmbeddingService()
    vector_store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME", "rag-document-index"),
        namespace="rag-evaluation",
    )
    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    answerable_labels = []
    abstention_decisions = []

    print(f"Abstention evaluation: threshold={THRESHOLD:.2f}, top_k={TOP_K}")
    print("-" * 72)

    for item in questions:
        question = item["question"]
        expected_answerable = item["answerable"]

        chunks = retriever.retrieve(
            query=question,
            top_k=TOP_K,
            similarity_threshold=None,
        )

        abstained = should_abstain(
            chunks,
            min_similarity=THRESHOLD,
        )

        answerable_labels.append(expected_answerable)
        abstention_decisions.append(abstained)

        if chunks:
            top_score = max(chunk["score"] for chunk in chunks)
            score_text = f"{top_score:.4f}"
        else:
            score_text = "no results"

        expected = "ANSWER" if expected_answerable else "ABSTAIN"
        actual = "ABSTAIN" if abstained else "ANSWER"
        outcome = "CORRECT" if expected == actual else "INCORRECT"

        print(f"Question: {question}")
        print(f"  Expected: {expected} | Decision: {actual} | {outcome}")
        print(f"  Retrieved chunks: {len(chunks)} | Highest score: {score_text}")

    print("\n" + "=" * 72)
    print("ABSTENTION METRICS")
    print("=" * 72)

    metrics = abstention_metrics(answerable_labels, abstention_decisions)

    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()
