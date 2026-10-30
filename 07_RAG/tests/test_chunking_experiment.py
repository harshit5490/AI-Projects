def calculate_statistics(chunks: list[str]) -> dict:
    lengths = [len(chunk) for chunk in chunks]

    return {
        "chunk_count": len(chunks),
        "average_characters": (
            sum(lengths) / len(lengths)
            if lengths
            else 0
        ),
        "min_characters": min(lengths) if lengths else 0,
        "max_characters": max(lengths) if lengths else 0,
    }