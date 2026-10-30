
import json
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

from app.embeddings.service import EmbeddingService
from app.llm.huggingface import HuggingFaceClient
from app.retrieval.query_transformer import QueryTransformer
from app.retrieval.retriever import Retriever
from app.vectorstore.store import PineconeVectorStore


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
QUESTIONS_FILE = DATA_DIR / "chunking_eval_questions.json"

TOP_K = 3
SIMILARITY_THRESHOLD = 0.30

# Set this in .env to restrict evaluation to one document.
# Example: RAG_EVAL_DOCUMENT_ID=your-uploaded-document-id
DOCUMENT_ID = os.getenv("RAG_EVAL_DOCUMENT_ID")


# ---------------------------------------------------------
# Relevance evaluation
# ---------------------------------------------------------

def is_relevant(
    result: dict,
    relevant_phrases: list[str],
) -> bool:
    """Check whether a retrieved chunk contains a reference phrase."""

    chunk_text = str(result.get("text", "")).casefold()

    return any(
        phrase.casefold() in chunk_text
        for phrase in relevant_phrases
    )


def hit_at_k(
    results: list[dict],
    relevant_phrases: list[str],
    k: int,
) -> int:
    """Return 1 if any relevant chunk appears in the top-k results."""

    return int(
        any(
            is_relevant(result, relevant_phrases)
            for result in results[:k]
        )
    )


def reciprocal_rank(
    results: list[dict],
    relevant_phrases: list[str],
) -> float:
    """Return the reciprocal rank of the first relevant result."""

    for rank, result in enumerate(results, start=1):
        if is_relevant(result, relevant_phrases):
            return 1.0 / rank

    return 0.0


# ---------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------

def evaluate_retrieval(
    retriever: Retriever,
    query: str,
    relevant_phrases: list[str],
    transform_query: bool,
) -> dict:
    """Measure retrieval quality and end-to-end retrieval latency."""

    start = time.perf_counter()

    results = retriever.retrieve(
        query=query,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
        metadata_filter=(
            {"document_id": {"$eq": DOCUMENT_ID}}
            if DOCUMENT_ID
            else None
        ),
        transform_query=transform_query,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "results": results,
        "hit_at_3": hit_at_k(
            results, relevant_phrases, TOP_K
        ),
        "mrr_at_3": reciprocal_rank(
            results, relevant_phrases
        ),
        "latency_ms": elapsed_ms,
    }


def average(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main() -> None:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Questions file not found: {QUESTIONS_FILE}"
        )

    questions = json.loads(
        QUESTIONS_FILE.read_text(encoding="utf-8")
    )

    if not questions:
        raise ValueError("No evaluation questions found.")

    print("\nInitializing RAG components...")

    embedding_service = EmbeddingService()
    vector_store = PineconeVectorStore(index_name="rag-document-index")
    llm_client = HuggingFaceClient()
    query_transformer = QueryTransformer(llm_client)

    baseline_retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    transformed_retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        query_transformer=query_transformer,
    )

    print("Components initialized.")
    print(f"Questions: {len(questions)}")
    print(f"Top-k: {TOP_K}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(
        "Document filter: "
        + (DOCUMENT_ID if DOCUMENT_ID else "NONE — searching all documents")
    )

    baseline_rows = []
    transformed_rows = []

    print("\nQuery Transformation Evaluation")
    print("=" * 90)

    for item in questions:
        question = item["question"]
        phrases = item["relevant_phrases"]

        print(f"\nQuestion: {question}")

        # Original query: no LLM rewrite
        baseline = evaluate_retrieval(
            retriever=baseline_retriever,
            query=question,
            relevant_phrases=phrases,
            transform_query=False,
        )

        # Rewritten query: LLM rewrite enabled
        transformed = evaluate_retrieval(
            retriever=transformed_retriever,
            query=question,
            relevant_phrases=phrases,
            transform_query=True,
        )

        baseline_rows.append(baseline)
        transformed_rows.append(transformed)

        print(
            f"Original:   Hit@3={baseline['hit_at_3']} | "
            f"MRR@3={baseline['mrr_at_3']:.3f} | "
            f"Latency={baseline['latency_ms']:.1f} ms"
        )

        print(
            f"Transformed: Hit@3={transformed['hit_at_3']} | "
            f"MRR@3={transformed['mrr_at_3']:.3f} | "
            f"Latency={transformed['latency_ms']:.1f} ms"
        )

    # -----------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------

    baseline_hit = average(
        [row["hit_at_3"] for row in baseline_rows]
    )
    transformed_hit = average(
        [row["hit_at_3"] for row in transformed_rows]
    )

    baseline_mrr = average(
        [row["mrr_at_3"] for row in baseline_rows]
    )
    transformed_mrr = average(
        [row["mrr_at_3"] for row in transformed_rows]
    )

    baseline_latency = average(
        [row["latency_ms"] for row in baseline_rows]
    )
    transformed_latency = average(
        [row["latency_ms"] for row in transformed_rows]
    )

    print("\nAggregate Results")
    print("=" * 90)

    print(
        f"{'Metric':<25}"
        f"{'Original Query':>20}"
        f"{'Transformed Query':>22}"
    )
    print("-" * 70)

    print(
        f"{'Hit@3':<25}"
        f"{baseline_hit:>20.3f}"
        f"{transformed_hit:>22.3f}"
    )
    print(
        f"{'MRR@3':<25}"
        f"{baseline_mrr:>20.3f}"
        f"{transformed_mrr:>22.3f}"
    )
    print(
        f"{'Avg latency (ms)':<25}"
        f"{baseline_latency:>20.1f}"
        f"{transformed_latency:>22.1f}"
    )

    print("-" * 70)
    print(
        "Average latency difference: "
        f"{transformed_latency - baseline_latency:+.1f} ms"
    )
    print(
        "Hit@3 change: "
        f"{transformed_hit - baseline_hit:+.3f}"
    )
    print(
        "MRR@3 change: "
        f"{transformed_mrr - baseline_mrr:+.3f}"
    )

    # -----------------------------------------------------
    # Save detailed results
    # -----------------------------------------------------

    output_path = DATA_DIR / "query_transformation_results.json"

    report = {
        "configuration": {
            "top_k": TOP_K,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "document_id": DOCUMENT_ID,
            "question_count": len(questions),
        },
        "summary": {
            "original": {
                "hit_at_3": baseline_hit,
                "mrr_at_3": baseline_mrr,
                "average_latency_ms": baseline_latency,
            },
            "transformed": {
                "hit_at_3": transformed_hit,
                "mrr_at_3": transformed_mrr,
                "average_latency_ms": transformed_latency,
            },
        },
        "per_question": [
            {
                "question": item["question"],
                "original": {
                    "hit_at_3": baseline_rows[index]["hit_at_3"],
                    "mrr_at_3": baseline_rows[index]["mrr_at_3"],
                    "latency_ms": baseline_rows[index]["latency_ms"],
                },
                "transformed": {
                    "hit_at_3": transformed_rows[index]["hit_at_3"],
                    "mrr_at_3": transformed_rows[index]["mrr_at_3"],
                    "latency_ms": transformed_rows[index]["latency_ms"],
                },
            }
            for index, item in enumerate(questions)
        ],
    }

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
