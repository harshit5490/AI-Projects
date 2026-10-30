from sentence_transformers import SentenceTransformer


class EmbeddingService:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str):
        if not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        return self.model.encode(
            text,
            normalize_embeddings=True,
        )

    def embed_texts(self, texts: list[str]):
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError(
                "Texts cannot contain empty strings."
            )

        return self.model.encode(
            texts,
            normalize_embeddings=True,
        )