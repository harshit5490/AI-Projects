# import chromadb

# client = chromadb.PersistentClient(
#     path="./chroma_data"
# )

# collection = client.get_or_create_collection(
#     name="test_collection"
# )

# collection.upsert(
#     ids=["doc_001"],
#     documents=["Python is useful for AI."],
# )

# print(collection.get())

import chromadb

client = chromadb.PersistentClient(
    path="./chroma_data"
)

collection = client.get_collection(
    name="test_collection"
)

print(collection.get())