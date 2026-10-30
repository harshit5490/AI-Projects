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
# Model
# ---------------------------------------------------------

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


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
) -> list[tuple[int, str, float]]:

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if not chunks:
        return []

    # Embed document chunks
    chunk_embeddings = model.encode(
        chunks
    )

    # Embed query
    query_embedding = model.encode(
        [query]
    )

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_embedding,
        chunk_embeddings,
    )[0]

    # Rank chunks by similarity
    ranked_indices = (
        similarities
        .argsort()[::-1]
        [:top_k]
    )

    results = []

    for index in ranked_indices:

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
# Hit@K
# ---------------------------------------------------------

def hit_at_k(
    results: list[tuple[int, str, float]],
    relevant_phrases: list[str],
    k: int,
) -> int:

    for _, chunk, _ in results[:k]:

        if is_relevant(
            chunk,
            relevant_phrases,
        ):

            return 1

    return 0


# ---------------------------------------------------------
# Evaluate configuration
# ---------------------------------------------------------

def evaluate_configuration(
    chunks: list[str],
    questions: list[dict],
    model: SentenceTransformer,
) -> dict:

    k_values = [1, 3, 5]

    metrics = {}

    for k in k_values:

        hits = []
        reciprocal_ranks = []

        for item in questions:

            results = retrieve(
                query=item["question"],
                chunks=chunks,
                model=model,
                top_k=k,
            )

            relevant_phrases = item[
                "relevant_phrases"
            ]

            hits.append(
                hit_at_k(
                    results=results,
                    relevant_phrases=relevant_phrases,
                    k=k,
                )
            )

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

        metrics[k] = {
            "hit": hit_rate,
            "mrr": mrr,
        }

    return metrics


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
    # Configurations
    # -----------------------------------------------------

    configurations = {
        "Small_300_30": (
            300,
            30,
        ),
        "Medium_500_50": (
            500,
            50,
        ),
        "Large_700_70": (
            700,
            70,
        ),
    }

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    print(
        "\nChunk Size & Overlap Experiment"
    )

    print("=" * 75)

    for name, (
        chunk_size,
        chunk_overlap,
    ) in configurations.items():

        # ---------------------------------------------
        # Create chunker
        # ---------------------------------------------

        chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # ---------------------------------------------
        # Create chunks
        # ---------------------------------------------

        chunks = chunker.split(
            document
        )

        # ---------------------------------------------
        # Evaluate
        # ---------------------------------------------

        metrics = evaluate_configuration(
            chunks=chunks,
            questions=questions,
            model=model,
        )

        # ---------------------------------------------
        # Average chunk length
        # ---------------------------------------------

        average_length = (
            sum(
                len(chunk)
                for chunk in chunks
            )
            / len(chunks)
            if chunks
            else 0
        )

        # ---------------------------------------------
        # Print configuration
        # ---------------------------------------------

        print(f"\n{name}")

        print("-" * 75)

        print(
            f"Chunk size: {chunk_size}"
        )

        print(
            f"Overlap: {chunk_overlap}"
        )

        print(
            f"Number of chunks: {len(chunks)}"
        )

        print(
            f"Average chunk length: "
            f"{average_length:.2f}"
        )

        # ---------------------------------------------
        # Print metrics
        # ---------------------------------------------

        for k in [1, 3, 5]:

            print(
                f"Hit@{k}: "
                f"{metrics[k]['hit']:.2f} | "
                f"MRR@{k}: "
                f"{metrics[k]['mrr']:.2f}"
            )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()