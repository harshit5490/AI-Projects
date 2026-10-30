from app.vectorstore.index_manager import PineconeIndexManager


manager = PineconeIndexManager(
    index_name="rag-document-index",
    dimension=384,
    metric="cosine",
)

manager.create_index()

print("Pinecone index is ready.")