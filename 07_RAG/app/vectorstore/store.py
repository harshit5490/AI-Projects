from app.vectorstore.pinecone_client import PineconeClient


class PineconeVectorStore:

    def __init__(
        self,
        index_name: str,
        namespace: str = "rag-documents",
    ):
        pinecone_client = PineconeClient()

        self.client = pinecone_client.client
        self.index = self.client.Index(index_name)
        self.namespace = namespace

    def upsert(
        self,
        vectors: list[dict],
    ):
        if not vectors:
            return
        normalized_vectors = []        

        for vector in vectors:

                values = vector["values"]

                if hasattr(values, "tolist"):
                    values = values.tolist()

                normalized_vectors.append(
                    {
                        **vector,
                        "values": values,
                    }
                )

        self.index.upsert(
            vectors=normalized_vectors,
            namespace=self.namespace,
        )

    def query(
        self,
        query_vector,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ):
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()


        return self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
            filter=metadata_filter
        )

    def query_chunks(
        self,
        query_vector,
        top_k: int = 5,
        similarity_threshold: float | None = None,
        metadata_filter: dict | None = None,
    ) -> list[dict]:

        if similarity_threshold is not None:
            if not 0 <= similarity_threshold <= 1:
                raise ValueError(
                    "similarity_threshold must be between 0 and 1."
                )

        result = self.query(
            query_vector=query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter
        )

        chunks = []

        for match in result.matches:

            if (
                similarity_threshold is not None
                and match.score < similarity_threshold
            ):
                continue

            metadata = match.metadata

            chunks.append(
                {
                    "id": match.id,
                    "score": match.score,
                    "text": metadata["text"],
                    "document_id": metadata["document_id"],
                    "filename": metadata["filename"],
                    "file_type": metadata["file_type"],
                    "chunk_index": metadata["chunk_index"],
                    "page": metadata.get("page"),
                }
            )

        return chunks

    def delete_document(
        self,
        document_id: str,
    ):
        if not document_id.strip():
            raise ValueError(
                "document_id cannot be empty."
            )

        self.index.delete(
            filter={
                "document_id": {
                    "$eq": document_id
                }
            },
            namespace=self.namespace,
        )    