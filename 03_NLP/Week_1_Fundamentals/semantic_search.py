import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")


documents = [
    "Python is a programming language used for AI and data science.",
    "FastAPI is a modern framework for building APIs with Python.",
    "Machine learning allows computers to learn patterns from data.",
    "PostgreSQL is a relational database management system.",
    "Deep learning uses neural networks with multiple layers."
]


document_embeddings = model.encode(documents)

print("Document embedding shape:")
print(document_embeddings.shape)


query = "What is Python used for?"

query_embedding = model.encode([query])


similarities = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]


best_index = np.argmax(similarities)

print("\nQuery:")
print(query)

print("\nMost relevant document:")
print(documents[best_index])

print("\nSimilarity score:")
print(similarities[best_index])