from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL_NAME
from app.models import Document

MODEL_NAME = EMBEDDING_MODEL_NAME

class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def encode_documents(
            self,
            documents:list[Document]
    ) -> list[list[float]]:

        texts = [document.text for document in documents]

        embeddings = self.model.encode(texts,convert_to_numpy=True) 

        return embeddings.tolist()

    def encode_query(
        self,
        query: str,
    ) -> list[float]:

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
        )[0]

        return embedding.tolist()

    @property
    def dimension(self) -> int:
        return self.model.get_embedding_dimension()           