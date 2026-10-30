from pathlib import Path

from app.chunking.fixed import FixedSizeChunker
from app.chunking.recursive import RecursiveCharacterChunker
from app.chunking.semantic import SemanticChunker
from app.chunking.token import TokenChunker


DATA_FILE = Path(__file__).parent / "data" / "chunking_eval.txt"


def calculate_statistics(chunks: list[str]) -> dict:
    lengths = [len(chunk) for chunk in chunks]

    if not lengths:
        return {
            "chunk_count": 0,
            "average_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
        }

    return {
        "chunk_count": len(chunks),
        "average_chars": sum(lengths) / len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }


def print_result(name: str, statistics: dict) -> None:
    print(
        f"{name:<15}"
        f"{statistics['chunk_count']:<10}"
        f"{statistics['average_chars']:<15.2f}"
        f"{statistics['min_chars']:<10}"
        f"{statistics['max_chars']:<10}"
    )


def main() -> None:
    text = DATA_FILE.read_text(encoding="utf-8")

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
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold=0.5,
    )

    results = {
        "Fixed-size": fixed_chunker.split(text),
        "Recursive": recursive_chunker.split(text),
        "Token": token_chunker.chunk(text),
        "Semantic": semantic_chunker.chunk(text),
    }

    print("\nChunking Strategy Comparison")
    print("=" * 65)

    print(
        f"{'Strategy':<15}"
        f"{'Chunks':<10}"
        f"{'Avg Chars':<15}"
        f"{'Min':<10}"
        f"{'Max':<10}"
    )

    print("-" * 65)

    for name, chunks in results.items():
        statistics = calculate_statistics(chunks)
        print_result(name, statistics)

    print("=" * 65)

    print("\n\nActual Chunks")
    print("=" * 80)

    for name, chunks in results.items():
        print(f"\n{name}")
        print("-" * 80)

        for index, chunk in enumerate(chunks, start=1):
            print(f"\nChunk {index}:")
            print(chunk)    


if __name__ == "__main__":
    main()