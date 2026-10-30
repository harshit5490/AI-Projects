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
TOP_K = 3

THRESHOLDS = [
    None,
    0.20,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]


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
# Load questions
# ---------------------------------------------------------

def load_questions() -> list[dict]:

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ---------------------------------------------------------
# Retrieve
# ---------------------------------------------------------

def retrieve(
    query: str,
    chunks: list[str],
    model: SentenceTransformer,
    top_k: int,
    similarity_threshold: float | None = None,
) -> list[tuple[int, str, float]]:

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if not chunks:
        return []

    if (
        similarity_threshold is not None
        and not 0 <= similarity_threshold <= 1
    ):
        raise ValueError(
            "similarity_threshold must be "
            "between 0 and 1."
        )

    # -----------------------------------------------------
    # Create chunk embeddings
    # -----------------------------------------------------

    chunk_embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
    )

    # -----------------------------------------------------
    # Create query embedding
    # -----------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    # -----------------------------------------------------
    # Calculate cosine similarity
    # -----------------------------------------------------

    similarities = cosine_similarity(
        query_embedding,
        chunk_embeddings,
    )[0]

    # -----------------------------------------------------
    # Rank chunks
    # -----------------------------------------------------

    ranked_indices = (
        similarities
        .argsort()[::-1]
    )

    # -----------------------------------------------------
    # Select top-k
    # -----------------------------------------------------

    top_indices = ranked_indices[:top_k]

    results = []

    for index in top_indices:

        score = float(
            similarities[index]
        )

        # -------------------------------------------------
        # Apply similarity threshold
        # -------------------------------------------------

        if (
            similarity_threshold is not None
            and score < similarity_threshold
        ):
            continue

        results.append(
            (
                int(index),
                chunks[index],
                score,
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
# Hit
# ---------------------------------------------------------

def calculate_hit(
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

def calculate_reciprocal_rank(
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
# Evaluate threshold
# ---------------------------------------------------------

def evaluate_threshold(
    chunks: list[str],
    questions: list[dict],
    model: SentenceTransformer,
    threshold: float | None,
) -> dict:

    hits = []

    reciprocal_ranks = []

    retrieved_counts = []

    for item in questions:

        results = retrieve(
            query=item["question"],
            chunks=chunks,
            model=model,
            top_k=TOP_K,
            similarity_threshold=threshold,
        )

        relevant_phrases = item[
            "relevant_phrases"
        ]

        # ---------------------------------------------
        # Hit
        # ---------------------------------------------

        hits.append(
            calculate_hit(
                results=results,
                relevant_phrases=relevant_phrases,
            )
        )

        # ---------------------------------------------
        # Reciprocal rank
        # ---------------------------------------------

        reciprocal_ranks.append(
            calculate_reciprocal_rank(
                results=results,
                relevant_phrases=relevant_phrases,
            )
        )

        # ---------------------------------------------
        # Number of chunks returned
        # ---------------------------------------------

        retrieved_counts.append(
            len(results)
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

    average_retrieved = (
        sum(retrieved_counts)
        / len(retrieved_counts)
        if retrieved_counts
        else 0.0
    )

    return {
        "hit": hit_rate,
        "mrr": mrr,
        "average_retrieved": average_retrieved,
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
    # Create chunker
    # -----------------------------------------------------

    chunker = RecursiveCharacterChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = chunker.split(
        document
    )

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    print(
        "\nSimilarity Threshold Evaluation"
    )

    print("=" * 80)

    print(
        f"Chunk size: {CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap: {CHUNK_OVERLAP}"
    )

    print(
        f"Top-K: {TOP_K}"
    )

    print(
        f"Number of chunks: {len(chunks)}"
    )

    # -----------------------------------------------------
    # Evaluate thresholds
    # -----------------------------------------------------

    for threshold in THRESHOLDS:

        metrics = evaluate_threshold(
            chunks=chunks,
            questions=questions,
            model=model,
            threshold=threshold,
        )

        print("\n" + "-" * 80)

        if threshold is None:
            print(
                "Threshold: None"
            )
        else:
            print(
                f"Threshold: {threshold:.2f}"
            )

        print(
            f"Hit@{TOP_K}: "
            f"{metrics['hit']:.2f}"
        )

        print(
            f"MRR@{TOP_K}: "
            f"{metrics['mrr']:.2f}"
        )

        print(
            f"Average retrieved chunks: "
            f"{metrics['average_retrieved']:.2f}"
        )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()