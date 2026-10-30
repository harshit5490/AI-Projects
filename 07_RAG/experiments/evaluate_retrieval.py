from app.evaluation.retrieval_metrics import (
    mean_hit_at_k,
    mean_reciprocal_rank,
)


def build_retrieval_report(
    rankings: list[list[str]],
    relevant_chunk_ids: list[set[str]],
) -> dict[str, float]:
    """Build aggregate retrieval metrics for evaluated queries."""

    return {
        "hit_at_1": mean_hit_at_k(
            rankings, relevant_chunk_ids, k=1
        ),
        "hit_at_3": mean_hit_at_k(
            rankings, relevant_chunk_ids, k=3
        ),
        "hit_at_5": mean_hit_at_k(
            rankings, relevant_chunk_ids, k=5
        ),
        "mrr": mean_reciprocal_rank(
            rankings, relevant_chunk_ids
        ),
    }


def print_report(report: dict[str, float]) -> None:
    """Print metrics in a readable format."""

    print("\nRetrieval Evaluation Report")
    print("-" * 30)

    for metric, score in report.items():
        print(f"{metric}: {score:.3f}")


if __name__ == "__main__":
    # Synthetic example to verify the reporting logic.
    rankings = [
        ["chunk-a", "chunk-b", "chunk-c"],
        ["chunk-x", "chunk-y", "chunk-z"],
        ["chunk-m", "chunk-n", "chunk-o"],
    ]

    relevant_chunk_ids = [
        {"chunk-a"},
        {"chunk-y"},
        {"chunk-q"},
    ]

    report = build_retrieval_report(
        rankings,
        relevant_chunk_ids,
    )

    print_report(report)