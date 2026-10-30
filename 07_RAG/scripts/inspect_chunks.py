from app.chunking.recursive import RecursiveCharacterChunker


def main():
    chunker = RecursiveCharacterChunker(
        chunk_size=500,
        chunk_overlap=50,
    )

    with open(
        "experiments/data/chunking_eval.txt",
        "r",
        encoding="utf-8",
    ) as file:
        text = file.read()

    chunks = chunker.split(text)

    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks):
        print("\n" + "=" * 80)
        print(f"Chunk {index}")
        print(f"Length: {len(chunk)}")
        print("=" * 80)
        print(chunk)


if __name__ == "__main__":
    main()