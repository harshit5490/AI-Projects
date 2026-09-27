# # import chromadb


# # client = chromadb.Client()

# # collection = client.create_collection(
# #     name="documents"
# # )

# # print("Collection created successfully.")

# import chromadb


# client = chromadb.Client()

# collection = client.create_collection(
#     name="documents"
# )


# documents = [
#     "Python is widely used for machine learning.",
#     "Deep learning uses neural networks.",
#     "Football is a popular sport.",
# ]

# ids = [
#     "doc_001",
#     "doc_002",
#     "doc_003",
# ]
# metadatas = [
#     {
#         "category": "AI",
#         "source": "python_guide.pdf",
#     },
#     {
#         "category": "AI",
#         "source": "deep_learning.pdf",
#     },
#     {
#         "category": "Sports",
#         "source": "sports_guide.pdf",
#     },
# ]


# collection.add(
#     documents=documents,
#     ids=ids,
#     metadatas=metadatas
# )


# print("Documents added.")

# results = collection.get()

# print(results)

# results = collection.query(
#     query_texts=[
#         "How is Python used in AI?"
#     ],
#     n_results=2,
# )

# print(f"query: {results}")

import chromadb


client = chromadb.Client()

collection = client.create_collection(
    name="semantic_search"
)


documents = [
    "Python is widely used for machine learning.",
    "Machine learning models can be built using Python.",
    "Deep learning uses neural networks.",
    "Football is a popular sport.",
    "Pizza is a popular Italian food.",
]


ids = [
    "doc_001",
    "doc_002",
    "doc_003",
    "doc_004",
    "doc_005",
]


metadatas = [
    {
        "category": "AI",
        "source": "ml_guide.pdf",
    },
    {
        "category": "AI",
        "source": "python_ml.pdf",
    },
    {
        "category": "AI",
        "source": "deep_learning.pdf",
    },
    {
        "category": "Sports",
        "source": "sports.pdf",
    },
    {
        "category": "Food",
        "source": "food.pdf",
    },
]


collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
)


results = collection.query(
    query_texts=[
        "How can Python be used in AI?"
    ],
    n_results=3,
)


print(results)