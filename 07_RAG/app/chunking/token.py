from transformers import AutoTokenizer


class TokenChunker:
    def __init__(
        self,
        model_name: str,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []

        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        chunks = []

        start = 0

        while start < len(tokens):
            end = start + self.chunk_size

            chunk_tokens = tokens[start:end]

            chunk_text = self.tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True,
            )

            if chunk_text.strip():
                chunks.append(chunk_text)

            start += self.chunk_size - self.chunk_overlap

        return chunks