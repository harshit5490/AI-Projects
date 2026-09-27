from app.embeddings.model import EmbeddingModel
from app.ingestion.loader import load_documents
from app.vector_store.chroma import ChromaVectorStore


def initialize_vector_store(
    documents_directory: str,
    persist_directory: str,
) -> tuple[ChromaVectorStore, EmbeddingModel]:

    embedding_model = EmbeddingModel()

    vector_store = ChromaVectorStore(
        persist_directory=persist_directory
    )

    if vector_store.count() == 0:
        documents = load_documents(documents_directory)

        embeddings = embedding_model.encode_documents(
            documents
        )

        vector_store.add_documents(
            documents,
            embeddings,
        )

    return vector_store, embedding_model