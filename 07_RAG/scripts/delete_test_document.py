from app.vectorstore.store import PineconeVectorStore


DOCUMENT_ID = "5b857984-8110-45cd-8393-c29753181684"


def main():

    vector_store = PineconeVectorStore(
        index_name="rag-document-index",
        namespace="rag-documents",
    )

    vector_store.delete_document(
        document_id=DOCUMENT_ID
    )

    print(
        f"Deleted document: {DOCUMENT_ID}"
    )


if __name__ == "__main__":
    main()