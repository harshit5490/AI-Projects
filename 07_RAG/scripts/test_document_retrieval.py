from app.embeddings.service import EmbeddingService
from app.vectorstore.store import PineconeVectorStore


DOCUMENT_ID = "5b857984-8110-45cd-8393-c29753181684"


def main():
    embedding_service = EmbeddingService()

    vector_store = PineconeVectorStore(
        index_name="rag-document-index",
        namespace="rag-documents",
    )

    query = "What is FastAPI?"

    query_embedding = embedding_service.embed_text(query)

    results = vector_store.query_chunks(
        query_vector=query_embedding,
        top_k=5,
        metadata_filter={
            "document_id": {
                "$eq": DOCUMENT_ID
            }
        },
    )

    print(f"\nQuery: {query}")
    print(f"Results found: {len(results)}\n")

    for index, result in enumerate(results, start=1):
        print("=" * 70)
        print(f"Result #{index}")
        print(f"Score: {result['score']}")
        print(f"Filename: {result['filename']}")
        print(f"Chunk index: {result['chunk_index']}")
        print(f"Page: {result['page']}")
        print(f"Text:\n{result['text']}")
        print()


if __name__ == "__main__":
    main()