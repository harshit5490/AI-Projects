import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.chunking.fixed import FixedSizeChunker
from app.chunking.recursive import RecursiveCharacterChunker
from app.chunking.semantic import SemanticChunker
from app.chunking.token import TokenChunker


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"

DOCUMENT_FILE = DATA_DIR / "chunking_eval.txt"
QUESTIONS_FILE = DATA_DIR / "retrieval_questions.json"


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve(
    query: str,
    chunks: list[str],
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[tuple[int, str, float]]:
    """
    Retrieve the top-k most similar chunks for a query.

    Returns:
        List of tuples:
        (
            chunk_index,
            chunk_text,
            similarity_score
        )
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not chunks:
        return []

    # Create embeddings for all chunks
    chunk_embeddings = model.encode(chunks)

    # Create embedding for the query
    query_embedding = model.encode([query])

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_embedding,
        chunk_embeddings,
    )[0]

    # Sort chunk indices from highest similarity
    # to lowest similarity
    ranked_indices = similarities.argsort()[::-1]

    # Keep only top-k results
    top_indices = ranked_indices[:top_k]

    results = []

    for index in top_indices:
        results.append(
            (
                int(index),
                chunks[index],
                float(similarities[index]),
            )
        )

    return results


# ---------------------------------------------------------
# Relevance evaluation
# ---------------------------------------------------------

def is_relevant(
    result: tuple[int, str, float],
    expected_chunk_index: int,
) -> bool:
    """
    Determine whether a retrieved result is the expected
    relevant chunk.

    We use the chunk index instead of keyword matching.

    This avoids false negatives caused by checking whether
    particular words or phrases happen to exist in a chunk.
    """

    chunk_index, _, _ = result

    return chunk_index == expected_chunk_index


# ---------------------------------------------------------
# Reciprocal Rank
# ---------------------------------------------------------

def reciprocal_rank(
    results: list[tuple[int, str, float]],
    expected_chunk_index: int,
) -> float:
    """
    Calculate Reciprocal Rank.

    If the relevant chunk is:
        rank 1 -> 1.0
        rank 2 -> 0.5
        rank 3 -> 0.333
        etc.

    If the relevant chunk is not retrieved:
        0.0
    """

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if is_relevant(
            result,
            expected_chunk_index,
        ):
            return 1 / rank

    return 0.0


# ---------------------------------------------------------
# Hit@K
# ---------------------------------------------------------

def hit_at_k(
    results: list[tuple[int, str, float]],
    expected_chunk_index: int,
    k: int,
) -> int:
    """
    Return 1 if the expected chunk appears
    within the top-k results.

    Otherwise return 0.
    """

    for result in results[:k]:
        if is_relevant(
            result,
            expected_chunk_index,
        ):
            return 1

    return 0


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def main() -> None:

    # -----------------------------------------------------
    # Load evaluation document
    # -----------------------------------------------------

    document = DOCUMENT_FILE.read_text(
        encoding="utf-8"
    )

    # -----------------------------------------------------
    # Load retrieval questions
    # -----------------------------------------------------

    questions = json.loads(
        QUESTIONS_FILE.read_text(
            encoding="utf-8"
        )
    )

    # -----------------------------------------------------
    # Load embedding model
    # -----------------------------------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    # -----------------------------------------------------
    # Create chunkers
    # -----------------------------------------------------

    fixed_chunker = FixedSizeChunker(
        chunk_size=500,
    )

    recursive_chunker = RecursiveCharacterChunker(
        chunk_size=500,
        chunk_overlap=50,
    )

    token_chunker = TokenChunker(
        model_name="bert-base-uncased",
        chunk_size=128,
        chunk_overlap=20,
    )

    semantic_chunker = SemanticChunker(
        model_name=EMBEDDING_MODEL,
        similarity_threshold=0.5,
    )

    # -----------------------------------------------------
    # Generate chunks using each strategy
    # -----------------------------------------------------

    strategies = {
        "Fixed-size": fixed_chunker.split(document),

        "Recursive": recursive_chunker.split(document),

        "Token": token_chunker.chunk(document),

        "Semantic": semantic_chunker.chunk(document),
    }

    # -----------------------------------------------------
    # Values of K that we want to evaluate
    # -----------------------------------------------------

    k_values = [1, 3, 5]

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    print("\nRetrieval Evaluation")
    print("=" * 70)

    # -----------------------------------------------------
    # Evaluate every chunking strategy
    # -----------------------------------------------------

    for strategy_name, chunks in strategies.items():

        print(f"\n{strategy_name}")
        print("-" * 70)

        print(
            f"Number of chunks: {len(chunks)}"
        )

        # -------------------------------------------------
        # Evaluate different K values
        # -------------------------------------------------

        for k in k_values:

            hits = 0

            reciprocal_ranks = []

            # ---------------------------------------------
            # Evaluate every question
            # ---------------------------------------------

            for item in questions:

                question = item["question"]

                expected_chunk_index = item[
                    "expected_chunk_index"
                ]

                # -----------------------------------------
                # Retrieve chunks
                # -----------------------------------------

                results = retrieve(
                    query=question,
                    chunks=chunks,
                    model=model,
                    top_k=k,
                )

                # -----------------------------------------
                # Calculate Hit@K
                # -----------------------------------------

                hits += hit_at_k(
                    results=results,
                    expected_chunk_index=expected_chunk_index,
                    k=k,
                )

                # -----------------------------------------
                # Calculate Reciprocal Rank
                # -----------------------------------------

                reciprocal_ranks.append(
                    reciprocal_rank(
                        results=results,
                        expected_chunk_index=expected_chunk_index,
                    )
                )

            # -------------------------------------------------
            # Calculate Hit Rate
            # -------------------------------------------------

            hit_rate = (
                hits / len(questions)
                if questions
                else 0.0
            )

            # -------------------------------------------------
            # Calculate MRR
            # -------------------------------------------------

            mrr = (
                sum(reciprocal_ranks)
                / len(reciprocal_ranks)
                if reciprocal_ranks
                else 0.0
            )

            # -------------------------------------------------
            # Print metrics
            # -------------------------------------------------

            print(
                f"Hit@{k}: {hit_rate:.2f} | "
                f"MRR@{k}: {mrr:.2f}"
            )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()