import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.chunking.recursive import RecursiveCharacterChunker


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"

DOCUMENT_PATH = DATA_DIR / "chunking_eval.txt"
QUESTIONS_PATH = DATA_DIR / "chunking_eval_questions.json"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TOP_K_VALUES = [1, 2, 3, 5]


# ---------------------------------------------------------
# Load document
# ---------------------------------------------------------

def load_document() -> str:

    with open(
        DOCUMENT_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return file.read()


# ---------------------------------------------------------
# Load evaluation questions
# ---------------------------------------------------------

def load_questions() -> list[dict]:

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve(
    query: str,
    chunks: list[str],
    model: SentenceTransformer,
    top_k: int,
) -> list[tuple[int, str, float]]:

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if not chunks:
        return []

    # Create embeddings for chunks
    chunk_embeddings = model.encode(
        chunks
    )

    # Create embedding for query
    query_embedding = model.encode(
        [query]
    )

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_embedding,
        chunk_embeddings,
    )[0]

    # Rank chunks from highest similarity
    # to lowest similarity
    ranked_indices = (
        similarities
        .argsort()[::-1]
    )

    # Select top-k
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
# Relevance
# ---------------------------------------------------------

def is_relevant(
    chunk: str,
    relevant_phrases: list[str],
) -> bool:

    normalized_chunk = chunk.lower()

    return any(
        phrase.lower() in normalized_chunk
        for phrase in relevant_phrases
    )


# ---------------------------------------------------------
# Hit@K
# ---------------------------------------------------------

def hit_at_k(
    results: list[tuple[int, str, float]],
    relevant_phrases: list[str],
) -> int:

    for _, chunk, _ in results:

        if is_relevant(
            chunk,
            relevant_phrases,
        ):
            return 1

    return 0


# ---------------------------------------------------------
# Reciprocal Rank
# ---------------------------------------------------------

def reciprocal_rank(
    results: list[tuple[int, str, float]],
    relevant_phrases: list[str],
) -> float:

    for rank, (_, chunk, _) in enumerate(
        results,
        start=1,
    ):

        if is_relevant(
            chunk,
            relevant_phrases,
        ):
            return 1 / rank

    return 0.0


# ---------------------------------------------------------
# Evaluate Top-K
# ---------------------------------------------------------

def evaluate_top_k(
    chunks: list[str],
    questions: list[dict],
    model: SentenceTransformer,
    top_k: int,
) -> dict:

    hits = []

    reciprocal_ranks = []

    for item in questions:

        results = retrieve(
            query=item["question"],
            chunks=chunks,
            model=model,
            top_k=top_k,
        )

        relevant_phrases = item[
            "relevant_phrases"
        ]

        # Hit@K
        hits.append(
            hit_at_k(
                results=results,
                relevant_phrases=relevant_phrases,
            )
        )

        # Reciprocal Rank
        reciprocal_ranks.append(
            reciprocal_rank(
                results=results,
                relevant_phrases=relevant_phrases,
            )
        )

    hit_rate = (
        sum(hits) / len(hits)
        if hits
        else 0.0
    )

    mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
        if reciprocal_ranks
        else 0.0
    )

    return {
        "hit": hit_rate,
        "mrr": mrr,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    document = load_document()

    questions = load_questions()

    # -----------------------------------------------------
    # Load embedding model
    # -----------------------------------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    # -----------------------------------------------------
    # Create our selected chunker
    # -----------------------------------------------------

    chunker = RecursiveCharacterChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = chunker.split(
        document
    )

    # -----------------------------------------------------
    # Print configuration
    # -----------------------------------------------------

    print("\nTop-K Retrieval Evaluation")
    print("=" * 70)

    print(
        f"Chunk size: {CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap: {CHUNK_OVERLAP}"
    )

    print(
        f"Number of chunks: {len(chunks)}"
    )

    # -----------------------------------------------------
    # Evaluate each Top-K
    # -----------------------------------------------------

    for top_k in TOP_K_VALUES:

        metrics = evaluate_top_k(
            chunks=chunks,
            questions=questions,
            model=model,
            top_k=top_k,
        )

        print("\n" + "-" * 70)

        print(
            f"Top-K: {top_k}"
        )

        print(
            f"Hit@{top_k}: "
            f"{metrics['hit']:.2f}"
        )

        print(
            f"MRR@{top_k}: "
            f"{metrics['mrr']:.2f}"
        )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()