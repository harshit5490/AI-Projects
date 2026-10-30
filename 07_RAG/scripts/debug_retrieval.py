from app.embeddings.service import EmbeddingService
from app.retrieval.retriever import Retriever
from app.vectorstore.store import PineconeVectorStore


DOCUMENT_ID = "25c2a1e6-9561-4d42-ae71-c62af66e7fae"


QUESTIONS = [
    "What is Python used for?",
    "What is FastAPI?",
    "What does a vector database store?",
    "What is Retrieval-Augmented Generation?",
    "What is the purpose of a RAG pipeline?",
]


def main():

    embedding_service = EmbeddingService()

    vector_store = PineconeVectorStore(
        index_name="rag-document-index",
        namespace="rag-documents",
    )

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    for question in QUESTIONS:

        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        results = retriever.retrieve(
            query=question,
            top_k=5,
            metadata_filter={
                "document_id": {
                    "$eq": DOCUMENT_ID
                }
            },
        )

        for rank, result in enumerate(results, start=1):

            print(f"\nRank: {rank}")
            print(f"Score: {result['score']:.4f}")
            print(f"Chunk index: {result['chunk_index']}")
            print(f"Page: {result['page']}")
            print(f"Filename: {result['filename']}")

            print("\nText:")
            print(result["text"])


if __name__ == "__main__":
    main()