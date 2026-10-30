
def should_abstain(
    retrieved_chunks: list[dict],
    min_similarity: float = 0.30,
) -> bool:
    """
    Abstain when no retrieved chunk meets the minimum similarity.

    This is a retrieval-based heuristic, not proof that
    the retrieved context contains sufficient evidence.
    """

    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError(
            "min_similarity must be between 0 and 1."
        )

    if not retrieved_chunks:
        return True

    return not any(
        chunk["score"] >= min_similarity
        for chunk in retrieved_chunks
    )
