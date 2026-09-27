# import faiss
# import numpy as np


# vectors = np.array(
#     [
#         [1.0, 2.0],
#         [2.0, 4.0],
#         [10.0, 10.0],
#     ],
#     dtype=np.float32,
# )

# dimension = vectors.shape[1]

# index = faiss.IndexFlatL2(dimension)

# index.add(vectors)

# print("Number of vectors:", index.ntotal)

# query = np.array(
#     [[1.5, 3.0]],
#     dtype=np.float32,
# )

# distances, indices = index.search(query, k=2)

# print("Distances:")
# print(distances)

# print("Indices:")
# print(indices)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


documents = [
    "Python is used for machine learning.",
    "Machine learning models can be built using Python.",
    "Deep learning is a branch of artificial intelligence.",
    "I enjoy playing football.",
    "I like cooking Italian food.",
]


model = SentenceTransformer("all-MiniLM-L6-v2")


document_embeddings = model.encode(
    documents,
    convert_to_numpy=True,
)

document_embeddings = document_embeddings.astype(
    np.float32
)


dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(document_embeddings)


query = "How can Python be used in AI?"

query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
).astype(np.float32)


k = 3

distances, indices = index.search(
    query_embedding,
    k,
)


print("\nSearch results:\n")

for distance, index_id in zip(
    distances[0],
    indices[0],
):
    print(
        f"Distance: {distance:.4f} | "
        f"Document: {documents[index_id]}"
    )