
import os
from dotenv import load_dotenv

from app.embeddings.service import EmbeddingService
from app.vectorstore.store import PineconeVectorStore
from app.retrieval.retriever import Retriever
from app.generation.context_builder import ContextBuilder

load_dotenv()

embedding_service = EmbeddingService()

vector_store = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME", "rag-document-index"),
    namespace="rag-evaluation",
)

retriever = Retriever(
    embedding_service=embedding_service,
    vector_store=vector_store,
)

question = "What is semantic search based on?"

chunks = retriever.retrieve(
    query=question,
    top_k=3,
    similarity_threshold=0.35,
)

print("\n--- RETRIEVED CHUNKS ---")
for chunk in chunks:
    print(f"Score: {chunk['score']:.4f}")
    print(f"Chunk index: {chunk['chunk_index']}")
    print(f"Text: {chunk['text']}")
    print("-" * 60)

context = ContextBuilder().build(chunks)

print("\n--- BUILT CONTEXT ---")
print(context)

print("\n--- EXPECTED EVIDENCE CHECK ---")
print("Contains 'meaning':", "meaning" in context.lower())
print("Contains 'keyword':", "keyword" in context.lower())
