class FixedSizeChunker:

    def __init__(self, chunk_size: int):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        self.chunk_size = chunk_size

    def split(self, text: str) -> list[str]:
        chunks = []

        for start in range(0, len(text), self.chunk_size):
            chunk = text[start:start + self.chunk_size]

            if chunk.strip():
                chunks.append(chunk)

        return chunks