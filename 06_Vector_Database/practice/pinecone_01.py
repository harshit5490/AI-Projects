import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)

print("Connected to Pinecone")

model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384

index_name = "semantic-search-demo"

existing_indexes = pc.list_indexes()

index_names = [index.name for index in existing_indexes]

if index_name not in index_names:

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        )
    )

    while not pc.describe_index(index_name).status["ready"]:
        print("Waiting for index to become ready...")
        time.sleep(2)

    print("Index created successfully.")

else:
    print("index already exist")


index = pc.Index(index_name)      

documents = [
    "Python is widely used for machine learning.",
    "Deep learning uses neural networks.",
    "FastAPI is useful for building APIs.",
    "PostgreSQL is a relational database.",
    "Transformers are widely used in modern NLP.",
]

metadata = [
    {
        "category": "AI",
        "source": "python.txt",
    },
    {
        "category": "AI",
        "source": "deep_learning.txt",
    },
    {
        "category": "Backend",
        "source": "fastapi.txt",
    },
    {
        "category": "Database",
        "source": "postgresql.txt",
    },
    {
        "category": "AI",
        "source": "transformers.txt",
    },
]

embeddings = model.encode(
    documents,
    convert_to_numpy=True
)

print(f"Generated embeddings shape: {embeddings.shape}")

vectors = []

for i,embedding in enumerate(embeddings):
    vector = {
        "id":f"doc_{i+1:03d}",
        "values":embedding.tolist(),
        "metadata":{
            **metadata[i],
            "text":documents[i]
        }
    }
    vectors.append(vector)

index.upsert(vectors = vectors)

query = "What technology is useful for building AI systems?"

query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
)[0]

print("\n" + "=" * 60)
print("SEMANTIC SEARCH")
print("=" * 60)

results = index.query(
    vector=query_embedding.tolist(),
    top_k=3,
    include_metadata=True,
)

for rank, match in enumerate(results["matches"], start=1):

    print(f"\nRank {rank}")
    print(f"ID:       {match['id']}")
    print(f"Score:    {match['score']:.4f}")

    match_metadata = match.get("metadata", {})

    print(f"Category: {match_metadata.get('category')}")
    print(f"Source:   {match_metadata.get('source')}")
    print(f"Text:     {match_metadata.get('text')}")

print("\n" + "=" * 60)
print("SEMANTIC SEARCH + METADATA FILTER")
print("=" * 60)

filtered_results = index.query(
    vector=query_embedding.tolist(),
    top_k=3,
    include_metadata=True,
    filter={
        "category": {
            "$eq": "AI"
        }
    },
)

for rank, match in enumerate(
    filtered_results["matches"],
    start=1,
):

    print(f"\nRank {rank}")
    print(f"ID:       {match['id']}")
    print(f"Score:    {match['score']:.4f}")

    match_metadata = match.get("metadata", {})

    print(f"Category: {match_metadata.get('category')}")
    print(f"Source:   {match_metadata.get('source')}")
    print(f"Text:     {match_metadata.get('text')}")