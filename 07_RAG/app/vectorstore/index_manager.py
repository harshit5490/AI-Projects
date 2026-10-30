from pinecone import ServerlessSpec

from app.vectorstore.pinecone_client import PineconeClient


class PineconeIndexManager:

    def __init__(
        self,
        index_name: str,
        dimension: int = 384,
        metric: str = "cosine",
    ):
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric

        pinecone_client = PineconeClient()
        self.client = pinecone_client.client

    def create_index(self):
        existing_indexes = self.client.list_indexes()

        if self.index_name in existing_indexes.names():
            return

        self.client.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )