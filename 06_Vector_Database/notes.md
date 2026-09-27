# Vector Databases --- Complete Chapter Notes

> **AI Engineer Bootcamp --- Vector Databases**
>
> Purpose: Understand vector search and build the retrieval foundation
> needed for semantic search and later RAG systems.

------------------------------------------------------------------------

## 1. Chapter Objective

A vector database stores numerical vector representations of data and
makes it possible to search those vectors using similarity or distance.

The central pipeline is:

``` text
Documents
   ↓
Embedding Model
   ↓
Vectors
   ↓
Vector Store / Vector Database
   ↓
Similarity Search
   ↓
Top-K Results
```

For a query:

``` text
User Query
   ↓
Embedding Model
   ↓
Query Vector
   ↓
Vector Search
   ↓
Relevant Documents
```

The chapter covers:

1.  Why semantic search
2.  Embeddings → vectors
3.  Distance and similarity metrics
4.  Cosine similarity
5.  Dot product
6.  Euclidean distance
7.  FAISS
8.  ChromaDB
9.  Pinecone
10. Collections / indexes and metadata
11. Metadata filtering
12. Top-k retrieval
13. Embedding-model selection
14. Basic retrieval evaluation
15. Semantic Search project

------------------------------------------------------------------------

# 2. Why Semantic Search?

## 2.1 Keyword Search

Traditional keyword search looks for matching words.

Example:

``` text
Query:
"How can I build an API?"
```

A keyword-based system primarily looks for words such as:

``` text
build
API
```

This can fail when the query and document use different words but have
the same meaning.

Example:

``` text
Query:
"How can I create a Python web service?"

Document:
"FastAPI is a Python framework for building APIs."
```

The concepts are related even though the wording is different.

------------------------------------------------------------------------

## 2.2 Semantic Search

Semantic search attempts to retrieve content based on meaning rather
than only exact word matches.

The basic pipeline is:

``` text
Document
   ↓
Embedding Model
   ↓
Vector

Query
   ↓
Embedding Model
   ↓
Query Vector

Query Vector
   ↓
Similarity Search
   ↓
Most Similar Documents
```

Example:

``` text
"I enjoy programming in Python."

"I like developing software using Python."
```

These sentences use different wording but have similar meaning, so a
good embedding model should place their vectors relatively close
together.

------------------------------------------------------------------------

## 2.3 Semantic Search vs LLM

These components have different responsibilities.

### Embedding model

Converts text into vectors:

``` text
Text → Vector
```

### Vector database

Stores vectors and searches for similar vectors:

``` text
Vector → Similar Vectors
```

### LLM

Generates or transforms natural-language output:

``` text
Context + Prompt → Answer
```

A retrieval system can therefore exist without an LLM:

``` text
Query
 ↓
Embedding
 ↓
Vector Search
 ↓
Relevant Documents
```

A RAG system later adds an LLM:

``` text
Query
 ↓
Embedding
 ↓
Vector Search
 ↓
Retrieved Context
 ↓
LLM
 ↓
Answer
```

------------------------------------------------------------------------

# 3. Embeddings → Vectors

## 3.1 What is an Embedding?

An embedding is a numerical representation of an object such as:

-   text
-   image
-   audio
-   code

For this chapter we focus mainly on text embeddings.

Example:

``` text
"Python is used for machine learning."
              ↓
       Embedding Model
              ↓
[0.021, -0.183, 0.492, ..., 0.071]
```

The vector contains numerical information that can be used for
similarity calculations.

------------------------------------------------------------------------

## 3.2 Dense Vectors

Embedding models generally produce dense vectors.

Example:

``` text
[0.21, -0.18, 0.42, 0.07, ...]
```

Most dimensions contain numerical values.

This differs from a simple bag-of-words representation where many
dimensions may be zero.

------------------------------------------------------------------------

## 3.3 Distributed Representation

An embedding dimension generally does not correspond directly to one
specific word.

Instead, meaning is distributed across many dimensions.

Therefore, we normally do not interpret:

``` text
dimension 17 = Python
dimension 42 = API
```

as a simple one-to-one mapping.

The useful property is the geometry of the vectors.

------------------------------------------------------------------------

## 3.4 Embedding Dimension

A model determines the vector dimension.

For the model used throughout this chapter:

``` python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
```

the embedding dimension is:

``` text
384
```

Therefore:

``` text
One document → shape (384,)
N documents → shape (N, 384)
```

Example:

``` python
sentences = [
    "I enjoy programming in Python.",
    "I like developing software using Python.",
    "I enjoy eating pizza.",
]

embeddings = model.encode(sentences)

print(embeddings.shape)
```

Expected structure:

``` text
(number_of_sentences, 384)
```

------------------------------------------------------------------------

## 3.5 Normalization

Vector normalization changes the magnitude of a vector while preserving
its direction.

For a vector:

``` text
v
```

its L2-normalized form is:

``` text
v_normalized = v / ||v||
```

Normalization is important because some similarity methods are affected
by vector magnitude.

A key relationship:

``` text
For normalized vectors:

dot product ≈ cosine similarity
```

This is especially useful when choosing an index/metric combination.

------------------------------------------------------------------------

# 4. Similarity and Distance Metrics

Vector search needs a way to determine how close two vectors are.

Important metrics:

1.  Cosine similarity
2.  Dot product / inner product
3.  Euclidean distance

------------------------------------------------------------------------

# 5. Cosine Similarity

Cosine similarity measures the angle between two vectors.

Formula:

``` text
cos(A,B) = (A · B) / (||A|| ||B||)
```

Interpretation:

``` text
Higher cosine similarity
        ↓
More similar direction

Lower cosine similarity
        ↓
Less similar direction
```

The mathematical range is:

``` text
-1 to +1
```

For many modern text embedding models, useful semantic comparisons often
produce positive similarities.

Python implementation:

``` python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )
```

------------------------------------------------------------------------

# 6. Dot Product

Dot product is:

``` text
A · B = Σ(Ai × Bi)
```

Example:

``` text
A = [1, 2]
B = [3, 4]

A · B
= (1×3) + (2×4)
= 3 + 8
= 11
```

Dot product considers both direction and magnitude.

Therefore:

``` text
Large vector magnitude
        ↓
Can produce a larger dot product
```

When vectors are normalized:

``` text
dot product ≈ cosine similarity
```

------------------------------------------------------------------------

# 7. Euclidean Distance

Euclidean distance measures the straight-line distance between vectors.

Formula:

``` text
d(A,B) = sqrt(
    Σ(Ai - Bi)^2
)
```

Interpretation:

``` text
Smaller distance
      ↓
More similar / closer

Larger distance
      ↓
Less similar / farther
```

Python:

``` python
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)
```

------------------------------------------------------------------------

# 8. Similarity vs Distance

Remember the direction of the score:

  Metric               Better Match
  -------------------- --------------
  Cosine similarity    Higher
  Dot product          Higher
  Euclidean distance   Lower

This distinction is critical when sorting search results.

------------------------------------------------------------------------

# 9. Basic Semantic Retrieval

A simple semantic search system can be built without a vector database.

``` python
from sentence_transformers import SentenceTransformer
import numpy as np

documents = [
    "Python is used for machine learning.",
    "Machine learning models can be built using Python.",
    "Deep learning is a branch of artificial intelligence.",
    "I enjoy playing football.",
    "I like cooking Italian food.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = model.encode(documents)

query = "How can Python be used in AI?"
query_embedding = model.encode([query])[0]

scores = []

for i, document_embedding in enumerate(document_embeddings):
    score = np.dot(
        query_embedding,
        document_embedding
    )
    scores.append((i, score))

scores.sort(key=lambda x: x[1], reverse=True)

for index, score in scores:
    print(score, documents[index])
```

This works for small datasets but becomes inefficient as the number of
vectors grows.

That leads to vector-search libraries and databases.

------------------------------------------------------------------------

# 10. FAISS

## 10.1 What is FAISS?

FAISS stands for **Facebook AI Similarity Search**.

It is a library designed for efficient similarity search over vectors.

Important distinction:

``` text
FAISS ≠ Embedding Model
FAISS ≠ LLM
```

FAISS performs vector indexing and search.

Pipeline:

``` text
Text
 ↓
Embedding Model
 ↓
Vector
 ↓
FAISS
 ↓
Nearest Vectors
```

------------------------------------------------------------------------

# 11. FAISS IndexFlatL2

Example:

``` python
import faiss
import numpy as np

dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(
    document_embeddings.astype(np.float32)
)
```

`IndexFlatL2` performs exact search using L2 distance.

Important:

> FAISS `IndexFlatL2` returns **squared L2 distances**, not the
> square-root Euclidean distance.

------------------------------------------------------------------------

# 12. FAISS Search

``` python
distances, indices = index.search(
    query_embedding.astype(np.float32),
    3
)
```

Here:

``` text
3 = top-k
```

The result contains:

``` text
distances
indices
```

`indices` are integer IDs corresponding to the order in which vectors
were added.

Therefore, an application usually maintains a mapping:

``` text
FAISS ID
   ↓
Document ID
   ↓
Document / metadata
```

------------------------------------------------------------------------

# 13. FAISS IndexFlatIP

FAISS also supports inner product search:

``` python
index = faiss.IndexFlatIP(dimension)
```

For cosine similarity, a common approach is:

``` text
Normalize vectors
       ↓
IndexFlatIP
       ↓
Inner product
       ↓
Cosine-equivalent ranking
```

Example:

``` python
faiss.normalize_L2(document_embeddings)
faiss.normalize_L2(query_embedding)

index = faiss.IndexFlatIP(dimension)
index.add(document_embeddings)

distances, indices = index.search(
    query_embedding,
    3
)
```

------------------------------------------------------------------------

# 14. Exact Search vs Approximate Search

## Exact Search

Methods such as:

``` text
IndexFlatL2
IndexFlatIP
```

perform exact nearest-neighbor search.

Advantage:

``` text
High accuracy / exact result
```

Disadvantage:

``` text
Can become expensive at very large scale
```

------------------------------------------------------------------------

## Approximate Nearest Neighbor (ANN)

ANN methods trade some search accuracy for improved speed/scalability.

Conceptually:

``` text
Exact Search
Accuracy: very high
Speed: potentially slower

ANN
Accuracy: approximate
Speed: faster at scale
```

One FAISS family uses inverted-file indexing.

Conceptually:

``` text
Vectors
   ↓
Clusters / partitions
   ↓
Search relevant partitions
   ↓
Nearest vectors
```

Important parameters include concepts such as:

-   `nlist` --- number of partitions/clusters
-   `nprobe` --- number of partitions searched

Increasing search coverage can improve recall but can increase latency.

------------------------------------------------------------------------

# 15. Top-K Retrieval

Top-k means:

> Return the k highest-ranked results.

Example:

``` python
index.search(query_vector, 3)
```

means:

``` text
Top 3 results
```

Important:

``` text
Top-k ≠ guaranteed relevance
```

A vector database always returns the requested number of nearest
candidates when possible, even if some candidates are not useful.

This is why retrieval evaluation is necessary.

------------------------------------------------------------------------

# 16. FAISS Limitations

FAISS is primarily a vector similarity-search library.

Application features such as:

-   persistent metadata
-   document management
-   filtering
-   collections
-   multi-user data separation

usually need to be implemented around FAISS or handled by a higher-level
database system.

This leads to systems such as ChromaDB and Pinecone.

------------------------------------------------------------------------

# 17. ChromaDB

## 17.1 What is ChromaDB?

ChromaDB is a vector database/store designed for application-oriented
vector search.

It provides concepts such as:

-   client
-   collections
-   IDs
-   documents
-   embeddings
-   metadata
-   querying
-   filtering
-   persistence

Compared with raw FAISS, ChromaDB provides a higher-level database
abstraction.

------------------------------------------------------------------------

# 18. ChromaDB Collection

A collection is a logical container for related records.

Conceptually:

``` text
Chroma Client
     ↓
Collection
     ├── ID
     ├── Document
     ├── Embedding
     └── Metadata
```

Example:

``` python
import chromadb

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="documents"
)
```

------------------------------------------------------------------------

# 19. ChromaDB IDs

Every stored record should have an ID.

Example:

``` python
ids = [
    "doc_001",
    "doc_002",
    "doc_003",
]
```

IDs allow you to:

-   retrieve records
-   update records
-   delete records
-   identify search results

------------------------------------------------------------------------

# 20. Adding Documents

Example:

``` python
collection.add(
    ids=["doc_001", "doc_002"],
    documents=[
        "Python is used for machine learning.",
        "FastAPI is useful for building APIs.",
    ],
    metadatas=[
        {"category": "AI"},
        {"category": "Backend"},
    ],
)
```

Depending on configuration, Chroma can manage embedding generation
through its embedding-function mechanism. You can also explicitly
provide embeddings.

------------------------------------------------------------------------

# 21. ChromaDB Metadata

Metadata stores structured information associated with a vector.

Example:

``` python
{
    "category": "AI",
    "source": "ml.pdf"
}
```

Metadata is different from the embedding.

``` text
Embedding
→ numerical semantic representation

Metadata
→ structured attributes used for filtering/organization
```

Example:

``` text
Document
 ├── ID: doc_001
 ├── Text: ...
 ├── Vector: [0.12, ...]
 └── Metadata:
       category = AI
       source = ml.pdf
```

------------------------------------------------------------------------

# 22. ChromaDB Query

Example:

``` python
results = collection.query(
    query_texts=["How can Python be used in AI?"],
    n_results=3,
)
```

The database performs semantic retrieval and returns the nearest
records.

------------------------------------------------------------------------

# 23. Metadata Filtering

Semantic similarity alone may not be enough.

Suppose we have:

``` text
doc_001 → AI
doc_002 → Backend
doc_003 → Database
doc_004 → AI
```

We can restrict retrieval using metadata.

Conceptually:

``` python
results = collection.query(
    query_texts=["machine learning"],
    n_results=3,
    where={"category": "AI"},
)
```

Pipeline:

``` text
All Documents
     ↓
Metadata Filter
     ↓
Allowed Documents
     ↓
Similarity Search
     ↓
Top-K
```

Metadata filtering is useful for:

-   category
-   source
-   tenant
-   document type
-   date
-   access scope
-   other structured attributes

------------------------------------------------------------------------

# 24. ChromaDB Persistence

For persistent local storage:

``` python
client = chromadb.PersistentClient(
    path="./chroma_data"
)
```

Now vector data can persist on disk instead of existing only for the
current process.

This is important for applications that need to restart without
rebuilding their entire vector store.

------------------------------------------------------------------------

# 25. ChromaDB CRUD

Common operations include:

### Add

``` python
collection.add(...)
```

### Upsert

``` python
collection.upsert(...)
```

`upsert` generally means:

``` text
If ID doesn't exist → insert
If ID exists → update/replace
```

### Get

``` python
collection.get(
    ids=["doc_001"]
)
```

### Delete

``` python
collection.delete(
    ids=["doc_001"]
)
```

These operations allow the vector store to behave as part of an
application rather than just a static search index.

------------------------------------------------------------------------

# 26. FAISS vs ChromaDB

  Feature                FAISS                          ChromaDB
  ---------------------- ------------------------------ -----------------------------------
  Main purpose           Vector similarity search       Vector database/store
  Embedding generation   External                       Can integrate embedding functions
  Metadata               Usually application-managed    Built-in
  Filtering              Not a core database feature    Supported
  Persistence            Depends on index persistence   Supported
  Collections            Not a database abstraction     Supported
  Application CRUD       Limited/raw                    Higher-level
  Local development      Excellent                      Excellent

The key distinction:

``` text
FAISS
→ similarity-search library

ChromaDB
→ higher-level vector store/database
```

------------------------------------------------------------------------

# 27. Pinecone

## 27.1 What is Pinecone?

Pinecone is a managed/cloud vector database designed for production
vector-search workloads.

Conceptually:

``` text
Your Application
       ↓
Internet / API
       ↓
Pinecone
       ↓
Vector Search
```

Unlike a local FAISS index, Pinecone provides a managed service.

------------------------------------------------------------------------

# 28. Pinecone Terminology

Important terms:

### Index

A logical vector-search structure.

``` text
Index
 ├── vectors
 ├── metadata
 └── namespaces
```

### Vector

A numerical embedding.

``` text
[0.12, -0.04, ...]
```

### Metadata

Structured attributes associated with a vector.

### Namespace

A logical partition inside an index.

Namespaces can be useful for separating datasets or tenants.

### Upsert

Insert new vectors or replace/update vectors with the same IDs.

### Query

Search for nearest vectors.

------------------------------------------------------------------------

# 29. Pinecone Setup

Install:

``` bash
uv add pinecone python-dotenv
```

Environment variable:

``` env
PINECONE_API_KEY=your_api_key
```

Never commit the real API key.

`.gitignore`:

``` text
.env
```

`.env.example`:

``` text
PINECONE_API_KEY=
```

Load the key:

``` python
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)
```

------------------------------------------------------------------------

# 30. Pinecone Index

The vector dimension must match the embedding model.

Our model:

``` text
all-MiniLM-L6-v2
```

produces:

``` text
384 dimensions
```

Therefore the index must be configured for:

``` text
dimension = 384
```

A cosine index conceptually looks like:

``` python
pc.create_index(
    name="semantic-search",
    dimension=384,
    metric="cosine",
    ...
)
```

The exact cloud/specification parameters depend on the current Pinecone
SDK and service configuration.

------------------------------------------------------------------------

# 31. Pinecone Upsert

Example vector record:

``` python
{
    "id": "doc_001",
    "values": embedding.tolist(),
    "metadata": {
        "category": "AI",
        "source": "ml.pdf",
    },
}
```

Upsert means storing the vector and its metadata in the index.

------------------------------------------------------------------------

# 32. Pinecone Query

Example:

``` python
results = index.query(
    vector=query_embedding.tolist(),
    top_k=3,
    include_metadata=True,
)
```

The result contains matches with:

-   vector ID
-   similarity score
-   metadata when requested

------------------------------------------------------------------------

# 33. Pinecone Metadata Filtering

Example:

``` python
results = index.query(
    vector=query_embedding.tolist(),
    top_k=3,
    include_metadata=True,
    filter={
        "category": {
            "$eq": "AI"
        }
    },
)
```

Conceptually:

``` text
Query Vector
      ↓
Metadata filter
      ↓
Eligible vectors
      ↓
Similarity search
      ↓
Top-K
```

------------------------------------------------------------------------

# 34. Pinecone Namespaces

Namespaces provide logical separation inside an index.

Conceptually:

``` text
Index
│
├── namespace: company_a
│     ├── vectors
│
├── namespace: company_b
│     ├── vectors
│
└── namespace: company_c
      └── vectors
```

This can be useful for multi-tenant applications or logically separate
datasets.

------------------------------------------------------------------------

# 35. FAISS vs ChromaDB vs Pinecone

  --------------------------------------------------------------------------------
  Feature           FAISS                 ChromaDB               Pinecone
  ----------------- --------------------- ---------------------- -----------------
  Type              Search library        Vector store/database  Managed vector DB

  Local             Yes                   Yes                    Cloud service

  Metadata          Application-managed   Built-in               Built-in

  Filtering         Limited/raw           Supported              Supported

  Persistence       Index persistence     Built-in local         Managed
                                          persistence            

  Scaling           Application-managed   More                   Managed
                                          application-oriented   infrastructure

  Good learning use Excellent             Excellent              Excellent

  Production        Depends on            Suitable for many apps Managed
  architecture      application                                  production option
  --------------------------------------------------------------------------------

The important architectural distinction is:

``` text
FAISS → library
ChromaDB → database/store abstraction
Pinecone → managed vector database
```

------------------------------------------------------------------------

# 36. Collections, Indexes and Metadata

Different systems use different terminology.

### ChromaDB

``` text
Collection
    ↓
Documents + Embeddings + IDs + Metadata
```

### Pinecone

``` text
Index
    ↓
Vectors + Metadata
    ↓
Namespaces
```

### FAISS

``` text
Index
    ↓
Vectors
```

But the common conceptual model is:

``` text
Vector
+ ID
+ Metadata
      ↓
Searchable Record
```

------------------------------------------------------------------------

# 37. Metadata Filtering

Vector similarity answers:

> Which vectors are closest to my query?

Metadata filtering answers:

> Which vectors am I allowed/interested in searching?

These are complementary.

Example:

``` text
Query:
"How do I deploy an API?"

Filter:
category = "backend"
```

Conceptually:

``` text
All records
    ↓
category == backend
    ↓
Similarity search
    ↓
Top-K
```

This is especially important for:

-   multi-tenant systems
-   document categories
-   permissions/access scopes
-   dates
-   sources
-   document types

------------------------------------------------------------------------

# 38. Top-K Retrieval

Top-k means retrieving the highest-ranked k candidates.

Example:

``` python
top_k = 5
```

means:

``` text
Return the 5 nearest results.
```

Typical pipeline:

``` text
Query
 ↓
Embedding
 ↓
Vector Search
 ↓
Top-K
```

The value of `k` affects:

-   amount of context retrieved
-   latency
-   number of irrelevant candidates
-   downstream processing

Top-k should therefore be chosen based on the application's requirements
and evaluated rather than blindly selected.

------------------------------------------------------------------------

# 39. Similarity Threshold vs Top-K

Top-k says:

``` text
"Give me the best K results."
```

A threshold says:

``` text
"Give me results only if similarity is good enough."
```

These are different.

Example:

``` text
Query
 ↓
Results:

doc_1 → 0.91
doc_2 → 0.87
doc_3 → 0.43
doc_4 → 0.39
```

With:

``` text
top_k = 3
```

you still receive:

``` text
doc_1
doc_2
doc_3
```

even if `doc_3` is weak.

A threshold can be used to reject weak candidates.

However, the appropriate threshold depends on:

-   embedding model
-   similarity metric
-   data distribution
-   application
-   evaluation results

There is no universal similarity threshold.

------------------------------------------------------------------------

# 40. Embedding Model Selection

Embedding model selection is an engineering decision.

Do not choose a model simply because it has a large dimension or is
popular.

Important factors:

1.  Task
2.  Retrieval quality
3.  Embedding dimension
4.  Latency
5.  Model size
6.  RAM/GPU requirements
7.  Language support
8.  Domain suitability
9.  Query/document encoding behavior
10. Similarity metric compatibility
11. Deployment cost

------------------------------------------------------------------------

# 41. Factor: Task

Ask what the model is being used for:

``` text
Semantic similarity?
Semantic search?
Retrieval?
Clustering?
Classification?
```

A model trained specifically for retrieval can behave differently from
one primarily optimized for another task.

------------------------------------------------------------------------

# 42. Factor: Embedding Dimension

Example:

``` text
Model A → 384
Model B → 768
Model C → 1024
```

Higher dimension does **not** automatically mean better retrieval.

Higher dimensions can increase:

-   vector storage
-   memory usage
-   search computation

Therefore:

``` text
Dimension ≠ quality by itself
```

------------------------------------------------------------------------

# 43. Factor: Retrieval Quality

The most important practical question:

> Does the model retrieve the correct documents?

Test it using a labeled evaluation dataset.

Example:

``` text
Query → Relevant document(s)
```

Then compare models using:

-   Hit@K
-   Recall@K
-   Precision@K
-   MRR

------------------------------------------------------------------------

# 44. Factor: Latency

Embedding generation is part of application latency.

Pipeline:

``` text
User Query
 ↓
Embedding
 ↓
Vector Search
 ↓
Results
```

A larger model may provide different quality/latency trade-offs.

Measure actual inference time for your deployment environment.

------------------------------------------------------------------------

# 45. Factor: Model Size and Resources

Consider:

-   model download size
-   RAM
-   CPU/GPU availability
-   inference throughput
-   Docker image/deployment constraints

A model that is excellent but too expensive or slow for the target
infrastructure may not be appropriate for the application.

------------------------------------------------------------------------

# 46. Factor: Language Support

If your application is English-only, an English-focused model may be
sufficient.

For multilingual applications, evaluate multilingual models.

Example languages:

``` text
English
Hindi
Hinglish
Tamil
Telugu
Bengali
...
```

Language support should be evaluated on your actual queries and
documents.

------------------------------------------------------------------------

# 47. Factor: Domain

Consider the domain of the data:

``` text
General
Legal
Medical
Finance
Scientific
Code
Customer support
```

A model's general benchmark performance does not automatically guarantee
the same performance on a specialized domain.

------------------------------------------------------------------------

# 48. Factor: Query and Document Encoding

Some retrieval models distinguish between:

``` text
Query
```

and:

``` text
Document / Passage
```

Sentence Transformers provides:

``` python
model.encode_query(...)
model.encode_document(...)
```

for retrieval-oriented models where query/document-specific encoding is
useful.

Conceptually:

``` text
User Query
    ↓
Query Encoder

Document
    ↓
Document Encoder
```

The two representations are then compared in vector space.

------------------------------------------------------------------------

# 49. Factor: Similarity Metric Compatibility

Embedding model and similarity metric should be considered together.

Common metrics:

``` text
Cosine
Dot Product
Euclidean
```

Some retrieval models are designed/tuned around particular similarity
functions.

A common relationship:

``` text
Normalized vectors
       +
Dot Product
       ≈
Cosine Similarity
```

Do not arbitrarily change the metric without evaluating the impact.

------------------------------------------------------------------------

# 50. Our Baseline Model

Throughout the chapter we used:

``` python
SentenceTransformer(
    "all-MiniLM-L6-v2"
)
```

It provides a convenient lightweight baseline and produces:

``` text
384-dimensional embeddings
```

The reason for using it in the learning project is practical:

``` text
Small
↓
Fast
↓
Easy local inference
↓
Useful baseline
```

But a production system should validate the model against actual
retrieval requirements.

------------------------------------------------------------------------

# 51. Model Comparison

A proper model comparison is an experiment.

Example:

``` text
Model A
all-MiniLM-L6-v2

Model B
multi-qa-MiniLM-L6-dot-v1
```

Instead of:

``` text
"I think Model B is better."
```

use:

``` text
Same dataset
Same queries
Same ground truth
Same evaluation metrics
```

Then compare:

``` text
Retrieval quality
+
Latency
+
Resource usage
```

This produces evidence for the model decision.

------------------------------------------------------------------------

# 52. Public Benchmarks

Embedding models may be compared using public benchmarks such as MTEB.

Public benchmarks are useful for:

``` text
Shortlisting models
Understanding general capabilities
```

But they are not a substitute for testing your own data.

Recommended process:

``` text
Public benchmarks
        ↓
Shortlist
        ↓
Your dataset
        ↓
Retrieval evaluation
        ↓
Latency/resource evaluation
        ↓
Model selection
```

------------------------------------------------------------------------

# 53. Basic Retrieval Evaluation

A vector search system can run successfully while producing poor
results.

Therefore, we need ground truth and metrics.

Pipeline:

``` text
Queries
  +
Ground Truth
  +
Retrieved Results
        ↓
Evaluation Metrics
```

------------------------------------------------------------------------

# 54. Ground Truth

Ground truth defines which documents are relevant for each query.

Example:

``` python
ground_truth = {
    "How can I build a Python API?": ["doc_001"],
    "Which database is relational?": ["doc_003"],
}
```

For larger systems, a query can have multiple relevant documents:

``` python
{
    "query": "...",
    "relevant_docs": [
        "doc_001",
        "doc_007",
        "doc_013",
    ]
}
```

Ground truth quality strongly affects evaluation quality.

------------------------------------------------------------------------

# 55. Hit Rate@K

Question:

> Did at least one relevant document appear in the top K?

Formula:

``` text
Hit Rate@K =
Queries with a relevant result in top K
---------------------------------------
Total queries
```

Example:

``` text
4 queries
3 have a relevant document in top 3

Hit@3 = 3 / 4
      = 0.75
```

Interpretation:

``` text
75% of queries
found at least one relevant result
within the top 3.
```

------------------------------------------------------------------------

# 56. Recall@K

Question:

> How many of the relevant documents were retrieved?

Formula:

``` text
Recall@K =
Relevant documents retrieved in top K
-------------------------------------
Total relevant documents
```

Example:

``` text
Relevant:
doc_1
doc_4
doc_7

Top 5:
doc_2
doc_1
doc_5
doc_7
doc_8
```

Two of three relevant documents were retrieved:

``` text
Recall@5 = 2 / 3
         = 0.667
```

------------------------------------------------------------------------

# 57. Precision@K

Question:

> How many of the retrieved documents are relevant?

Formula:

``` text
Precision@K =
Relevant documents retrieved in top K
-------------------------------------
K
```

Example:

``` text
Top 5:
doc_1 ✓
doc_2 ✗
doc_3 ✗
doc_4 ✓
doc_5 ✗
```

Therefore:

``` text
Precision@5 = 2 / 5
            = 0.4
```

------------------------------------------------------------------------

# 58. Recall vs Precision

Recall:

``` text
Did I find the relevant information?
```

Precision:

``` text
How much of what I returned is relevant?
```

High recall can come with more irrelevant results.

High precision can come with fewer results and potentially missed
relevant documents.

The appropriate balance depends on the application.

------------------------------------------------------------------------

# 59. Reciprocal Rank

Reciprocal Rank focuses on the position of the first relevant result.

Formula:

``` text
RR = 1 / rank_of_first_relevant_result
```

Examples:

``` text
Relevant at rank 1
RR = 1/1 = 1.0

Relevant at rank 2
RR = 1/2 = 0.5

Relevant at rank 4
RR = 1/4 = 0.25
```

If no relevant document is retrieved:

``` text
RR = 0
```

------------------------------------------------------------------------

# 60. MRR --- Mean Reciprocal Rank

MRR is the average reciprocal rank across queries.

Formula:

``` text
MRR =
Σ Reciprocal Rank
-----------------
Number of Queries
```

Example:

``` text
Query 1 → rank 1 → 1.0
Query 2 → rank 2 → 0.5
Query 3 → rank 4 → 0.25
```

Therefore:

``` text
MRR = (1 + 0.5 + 0.25) / 3
    = 0.5833
```

MRR is useful when getting the first relevant result near the top is
important.

------------------------------------------------------------------------

# 61. Our Retrieval Evaluation Exercise

Evaluation data:

``` python
evaluation_data = [
    {
        "query": "How can I build a Python API?",
        "relevant_docs": ["doc_001"],
    },
    {
        "query": "Which technology is used for natural language processing?",
        "relevant_docs": ["doc_004"],
    },
    {
        "query": "What is used to containerize an application?",
        "relevant_docs": ["doc_005"],
    },
    {
        "query": "Which database is relational?",
        "relevant_docs": ["doc_003"],
    },
]
```

Metrics implemented:

``` python
hit_rate_at_k(...)
recall_at_k(...)
precision_at_k(...)
reciprocal_rank(...)
```

Overall:

``` python
hit_rate = sum(hit_scores) / len(hit_scores)
recall = sum(recall_scores) / len(recall_scores)
precision = sum(precision_scores) / len(precision_scores)
mrr = sum(rr_scores) / len(rr_scores)
```

------------------------------------------------------------------------

# 62. Our Evaluation Results

The completed exercise produced:

``` text
Hit@3:       1.000
Recall@3:    1.000
Precision@3: 0.333
MRR:         0.875
```

Interpretation:

### Hit@3

``` text
1.000 = 100%
```

All four test queries had a relevant document within the top three.

### Recall@3

``` text
1.000 = 100%
```

Because each test query had one labeled relevant document and it was
retrieved in the top three.

### Precision@3

``` text
0.333 = 33.3%
```

Each query returned three documents while only one was labeled relevant.

### MRR

``` text
0.875
```

The relevant document was usually ranked first. One query had its first
relevant result at rank 2.

Important:

> These numbers describe this small evaluation dataset. They should not
> be treated as production-level performance measurements.

------------------------------------------------------------------------

# 63. Important Evaluation Limitation

Our evaluation dataset is intentionally small.

Example:

``` text
1 query
1 relevant document
```

is easy to evaluate, but it may not represent the true relevance
structure of a real application.

A production evaluation set should contain:

-   diverse queries
-   realistic documents
-   multiple relevant documents where applicable
-   difficult queries
-   edge cases
-   representative languages/domains

Evaluation should also use the same retrieval pipeline that the
application actually uses.

------------------------------------------------------------------------

# 64. Complete Vector Search Architecture

At this point, the entire chapter can be understood as:

``` text
                         DOCUMENTS
                            │
                            ▼
                    Embedding Model
                            │
                            ▼
                         Vectors
                            │
                ┌───────────┴───────────┐
                │                       │
              FAISS                Vector DB
                                        │
                              ┌─────────┴─────────┐
                              │                   │
                          ChromaDB            Pinecone
                              │                   │
                              └─────────┬─────────┘
                                        │
                                        ▼
                                  Vector Search
                                        │
                                        ▼
                                      Top-K
                                        │
                              Metadata Filtering
                                        │
                                        ▼
                                  Search Results
                                        │
                                        ▼
                              Retrieval Evaluation
```

------------------------------------------------------------------------

# 65. Semantic Search Project

The roadmap project is a **Semantic Search application**.

Required project capabilities:

``` text
Document ingestion
        ↓
Embedding generation
        ↓
Vector storage
        ↓
Top-k retrieval
        ↓
Metadata filtering
        ↓
Live demo
        ↓
Docker
```

------------------------------------------------------------------------

# 66. Proposed Project Architecture

``` text
semantic_search/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   │
│   ├── ingestion/
│   │   └── loader.py
│   │
│   ├── embeddings/
│   │   └── model.py
│   │
│   ├── vector_store/
│   │   └── chroma.py
│   │
│   └── retrieval/
│       └── search.py
│
├── data/
│   └── documents/
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   └── test_search.py
│
├── .env
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

------------------------------------------------------------------------

# 67. Project Data Flow

``` text
              data/documents/
                      │
                      ▼
                Document Loader
                      │
                      ▼
                 Document Objects
                      │
                      ▼
                Embedding Model
                      │
                      ▼
                   Vectors
                      │
                      ▼
                   ChromaDB
                      │
                      │
              ┌───────┴────────┐
              │                │
          User Query      Metadata Filter
              │                │
              ▼                ▼
           Embedding       Filtering
              │                │
              └───────┬────────┘
                      ▼
                Similarity Search
                      │
                      ▼
                    Top-K
                      │
                      ▼
                Search Results
```

------------------------------------------------------------------------

# 68. Planned API

Health endpoint:

``` http
GET /health
```

Document ingestion:

``` http
POST /documents
```

Example:

``` json
{
    "documents": [
        {
            "id": "doc_001",
            "text": "FastAPI is a Python framework for building APIs.",
            "category": "backend",
            "source": "fastapi.txt"
        }
    ]
}
```

Search:

``` http
POST /search
```

Example:

``` json
{
    "query": "How can I build a Python API?",
    "top_k": 3
}
```

Search with metadata filtering:

``` json
{
    "query": "How can I build a Python API?",
    "top_k": 3,
    "filter": {
        "category": "backend"
    }
}
```

------------------------------------------------------------------------

# 69. Why We Separate the Components

We don't want one giant Python file.

Instead:

``` text
loader.py
→ document ingestion

model.py
→ embedding generation

chroma.py
→ vector storage

search.py
→ retrieval

main.py
→ API
```

Benefits:

-   easier testing
-   easier debugging
-   easier replacement of components
-   easier maintenance
-   cleaner architecture
-   better separation of responsibilities

For example, ChromaDB could later be replaced with another vector store
without rewriting ingestion or API logic.

------------------------------------------------------------------------

# 70. Security and Configuration

Secrets should never be hardcoded.

Bad:

``` python
PINECONE_API_KEY = "actual-secret-key"
```

Good:

``` env
PINECONE_API_KEY=...
```

and:

``` python
os.getenv("PINECONE_API_KEY")
```

`.env` should be excluded from Git.

``` text
.env
```

`.env.example` should contain only variable names/placeholders.

------------------------------------------------------------------------

# 71. Interview Questions

## Q1. What is a vector database?

A system designed to store and retrieve vector representations
efficiently using similarity or distance-based search.

------------------------------------------------------------------------

## Q2. What is an embedding?

A numerical vector representation of an object such as text that
captures useful semantic information.

------------------------------------------------------------------------

## Q3. Why do we need embeddings?

They allow semantic relationships to be represented geometrically so
similar content can be retrieved even when exact words differ.

------------------------------------------------------------------------

## Q4. What is cosine similarity?

A similarity measure based on the angle between vectors.

``` text
cos(A,B) = (A·B)/(||A||||B||)
```

Higher means more similar.

------------------------------------------------------------------------

## Q5. Cosine vs Euclidean?

Cosine compares vector direction.

Euclidean compares distance.

``` text
Cosine → higher is better
Euclidean → lower is better
```

------------------------------------------------------------------------

## Q6. What is top-k retrieval?

Returning the k highest-ranked candidate vectors for a query.

------------------------------------------------------------------------

## Q7. What is metadata filtering?

Restricting vector search to records matching structured metadata
conditions.

------------------------------------------------------------------------

## Q8. FAISS vs ChromaDB?

FAISS is primarily a vector similarity-search library.

ChromaDB is a higher-level vector store/database abstraction with
application-oriented features such as metadata and persistence.

------------------------------------------------------------------------

## Q9. ChromaDB vs Pinecone?

ChromaDB can be used locally and provides a vector-store abstraction.

Pinecone is a managed/cloud vector database.

------------------------------------------------------------------------

## Q10. Does higher embedding dimension mean better quality?

No.

Model training, task suitability, domain, retrieval performance,
latency, and resource requirements all matter.

------------------------------------------------------------------------

## Q11. Why evaluate retrieval?

Because a technically functioning vector search can still retrieve poor
results.

------------------------------------------------------------------------

## Q12. What is Recall@K?

The fraction of relevant documents that appear in the top K retrieved
results.

------------------------------------------------------------------------

## Q13. What is Precision@K?

The fraction of top-K retrieved results that are relevant.

------------------------------------------------------------------------

## Q14. What is MRR?

Mean Reciprocal Rank, the average reciprocal rank of the first relevant
result across queries.

------------------------------------------------------------------------

## Q15. Why can Top-K return irrelevant results?

Because top-k guarantees a number of nearest candidates, not that every
returned candidate passes a relevance threshold.

------------------------------------------------------------------------

# 72. Common Mistakes

### Mistake 1

Assuming:

``` text
Higher vector dimension = better
```

Not necessarily.

### Mistake 2

Confusing:

``` text
Embedding model
```

with:

``` text
Vector database
```

They perform different jobs.

### Mistake 3

Using the wrong vector dimension.

``` text
Embedding dimension ≠ Index dimension
```

can cause indexing/query problems.

### Mistake 4

Mixing similarity direction.

``` text
Cosine → higher
Dot product → higher
Euclidean → lower
```

### Mistake 5

Assuming top-k means all results are relevant.

It doesn't.

### Mistake 6

Ignoring metadata.

Metadata enables structured filtering and organization.

### Mistake 7

Choosing an embedding model only from a public benchmark.

Always validate on representative application data.

### Mistake 8

Hardcoding API keys.

Use environment variables.

### Mistake 9

Evaluating with too little or unrealistic data.

A small evaluation can be useful for learning but does not establish
production performance.

------------------------------------------------------------------------

# 73. Chapter Completion Checklist

## Concepts

-   [x] Why semantic search
-   [x] Embeddings → vectors
-   [x] Dense vectors
-   [x] Embedding dimensions
-   [x] Normalization
-   [x] Cosine similarity
-   [x] Dot product
-   [x] Euclidean distance
-   [x] Similarity vs distance

## FAISS

-   [x] FAISS overview
-   [x] IndexFlatL2
-   [x] IndexFlatIP
-   [x] Vector insertion
-   [x] Search
-   [x] Top-k
-   [x] ID mapping
-   [x] Float32 requirement
-   [x] Exact search
-   [x] ANN concepts
-   [x] `nlist`
-   [x] `nprobe`
-   [x] Cosine with normalized vectors + inner product

## ChromaDB

-   [x] Client
-   [x] Collections
-   [x] IDs
-   [x] Documents
-   [x] Embeddings
-   [x] Metadata
-   [x] Query
-   [x] Top-k
-   [x] Metadata filtering
-   [x] Persistence
-   [x] Add
-   [x] Upsert
-   [x] Get
-   [x] Delete
-   [x] Update workflow

## Pinecone

-   [x] Managed vector database concept
-   [x] Index
-   [x] Vector
-   [x] Dimension
-   [x] Metric
-   [x] Metadata
-   [x] Namespace
-   [x] Upsert
-   [x] Query
-   [x] Top-k
-   [x] Metadata filtering
-   [x] Environment variables / API key handling

## Embedding Model Selection

-   [x] Task
-   [x] Retrieval quality
-   [x] Dimension
-   [x] Latency
-   [x] Model size
-   [x] Resource requirements
-   [x] Language support
-   [x] Domain suitability
-   [x] Query/document encoding
-   [x] Similarity metric compatibility
-   [x] Public benchmarks
-   [x] Application-specific evaluation
-   [x] Model comparison

## Retrieval Evaluation

-   [x] Ground truth
-   [x] Top-k evaluation
-   [x] Hit Rate@K
-   [x] Recall@K
-   [x] Precision@K
-   [x] Reciprocal Rank
-   [x] MRR
-   [x] Evaluation limitations

## Hands-on Exercises

-   [x] Embedding generation
-   [x] Similarity calculations
-   [x] Semantic retrieval
-   [x] FAISS search
-   [x] ChromaDB CRUD/search/filtering/persistence
-   [x] Pinecone indexing/upsert/query/filtering
-   [x] Embedding model comparison
-   [x] Retrieval evaluation

## Project

-   [ ] Document ingestion
-   [ ] Embedding generation
-   [ ] Vector storage
-   [ ] Top-k retrieval
-   [ ] Metadata filtering
-   [ ] Tests
-   [ ] API
-   [ ] Docker
-   [ ] Live demo

------------------------------------------------------------------------

# 74. Final Mental Model

If you remember only one architecture from this chapter, remember this:

``` text
                         DOCUMENTS
                            │
                            ▼
                    EMBEDDING MODEL
                            │
                            ▼
                         VECTORS
                            │
                            ▼
                 VECTOR DATABASE
                            │
                            ▼
                       SIMILARITY
                         SEARCH
                            │
                            ▼
                          TOP-K
                            │
                    ┌───────┴───────┐
                    │               │
              Metadata Filter   Evaluation
                    │               │
                    └───────┬───────┘
                            ▼
                     RELEVANT RESULTS
```

And for a complete AI application:

``` text
User Query
    ↓
Query Embedding
    ↓
Vector Search
    ↓
Metadata Filtering
    ↓
Top-K Documents
    ↓
(Optional later)
LLM
    ↓
Answer
```

This vector retrieval layer is the foundation we will use for the
Semantic Search project and later RAG systems.
