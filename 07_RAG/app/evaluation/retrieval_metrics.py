
def hit_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """
    Return 1.0 if a relevant chunk appears in the top k results,
    otherwise return 0.0.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if not relevant_chunk_ids:
        raise ValueError(
            "relevant_chunk_ids cannot be empty."
        )

    top_k_ids = retrieved_chunk_ids[:k]

    return float(
        any(
            chunk_id in relevant_chunk_ids
            for chunk_id in top_k_ids
        )
    )


def reciprocal_rank(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
) -> float:
    """
    Return the reciprocal rank of the first relevant result.

    Returns 1/rank when a relevant chunk is found.
    Returns 0.0 when no relevant chunk is found.
    """

    if not relevant_chunk_ids:
        raise ValueError(
            "relevant_chunk_ids cannot be empty."
        )

    for rank, chunk_id in enumerate(
        retrieved_chunk_ids,
        start=1,
    ):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    rankings: list[list[str]],
    relevant_chunk_ids: list[set[str]],
) -> float:
    """Calculate MRR across multiple queries."""

    if not rankings:
        raise ValueError("rankings cannot be empty.")

    if len(rankings) != len(relevant_chunk_ids):
        raise ValueError(
            "Each ranking must have a matching relevant chunk set."
        )

    scores = [
        reciprocal_rank(ranking, relevant)
        for ranking, relevant in zip(
            rankings,
            relevant_chunk_ids,
        )
    ]

    return sum(scores) / len(scores)


def mean_hit_at_k(
    rankings: list[list[str]],
    relevant_chunk_ids: list[set[str]],
    k: int,
) -> float:
    """Calculate the average Hit@K score across queries."""

    if not rankings:
        raise ValueError("rankings cannot be empty.")

    if len(rankings) != len(relevant_chunk_ids):
        raise ValueError(
            "Each ranking must have a matching relevant chunk set."
        )

    scores = [
        hit_at_k(ranking, relevant, k)
        for ranking, relevant in zip(
            rankings,
            relevant_chunk_ids,
        )
    ]

    return sum(scores) / len(scores)
