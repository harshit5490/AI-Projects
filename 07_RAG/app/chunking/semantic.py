from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticChunker:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
    ):
        if not 0 <= similarity_threshold <= 1:
            raise ValueError(
                "similarity_threshold must be between 0 and 1"
            )

        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []

        sentences = sent_tokenize(text)

        if not sentences:
            return []

        if len(sentences) == 1:
            return [sentences[0]]

        embeddings = self.model.encode(sentences)

        chunks = []
        current_chunk = [sentences[0]]

        for index in range(1, len(sentences)):
            previous_embedding = embeddings[index - 1]
            current_embedding = embeddings[index]

            similarity = cosine_similarity(
                [previous_embedding],
                [current_embedding],
            )[0][0]

            if similarity >= self.similarity_threshold:
                current_chunk.append(sentences[index])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[index]]

        chunks.append(" ".join(current_chunk))

        return chunks