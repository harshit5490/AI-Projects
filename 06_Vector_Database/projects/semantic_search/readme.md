# Semantic Search API

A production-oriented semantic search application built with **Python, FastAPI, Sentence Transformers, and ChromaDB**.

The system converts documents and user queries into vector embeddings and retrieves the most semantically relevant documents using vector similarity search. It also supports metadata filtering, automated testing, Docker deployment, and persistent vector storage.

---

## Features

* Document ingestion from `.txt` files
* Semantic embedding generation
* Vector storage using ChromaDB
* Cosine similarity search
* Top-K retrieval
* Metadata filtering
* FastAPI REST API
* Swagger/OpenAPI documentation
* Pydantic request/response validation
* Automated tests with pytest
* Dockerized deployment
* Persistent ChromaDB storage using Docker volumes
* CPU-only PyTorch configuration for lightweight deployment

---

## Architecture

```text
                Documents
                    │
                    ▼
             Document Loader
                    │
                    ▼
          Sentence Transformer
          all-MiniLM-L6-v2
                    │
                    ▼
              Embeddings
                    │
                    ▼
                ChromaDB
                    │
                    │
User Query ──► Embedding Model
                    │
                    ▼
             Vector Search
                    │
             ┌──────┴──────┐
             │             │
           Top-K      Metadata Filter
             │             │
             └──────┬──────┘
                    ▼
              Search Results
                    │
                    ▼
                FastAPI
                    │
                    ▼
              REST Response
```

---

## Technology Stack

| Component         | Technology            |
| ----------------- | --------------------- |
| Language          | Python                |
| API               | FastAPI               |
| Embedding Model   | `all-MiniLM-L6-v2`    |
| Vector Database   | ChromaDB              |
| Validation        | Pydantic              |
| Testing           | pytest                |
| Containerization  | Docker                |
| Similarity Metric | Cosine                |
| Python Runtime    | Python 3.12 in Docker |

---

## Project Structure

```text
semantic_search/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── model.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── pipeline.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── search.py
│   │
│   └── vector_store/
│       ├── __init__.py
│       └── chroma.py
│
├── data/
│   └── documents/
│       ├── python.txt
│       ├── fastapi.txt
│       ├── machine_learning.txt
│       ├── transformers.txt
│       └── docker.txt
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_search.py
│   ├── test_retrieval.py
│   └── test_api.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# How It Works

## 1. Document Ingestion

Documents are loaded from:

```text
data/documents/
```

Each document is represented using:

* ID
* Text
* Source
* Category

Example:

```text
python.txt
    ↓
ID: python
Source: python.txt
Category: programming
```

---

## 2. Embedding Generation

The application uses:

```text
all-MiniLM-L6-v2
```

Each document is converted into a **384-dimensional vector**.

```text
Document
   ↓
Embedding Model
   ↓
384-dimensional vector
```

The same embedding model is used to convert incoming search queries into vectors.

---

## 3. Vector Storage

Embeddings are stored in **ChromaDB**.

The collection uses cosine distance for similarity search.

Each vector is associated with metadata:

```json
{
  "source": "python.txt",
  "category": "programming"
}
```

---

## 4. Semantic Search

For a query such as:

```text
How can I build a Python API?
```

the query is converted into an embedding.

ChromaDB compares the query vector against stored document vectors and returns the closest results.

The API returns the requested number of results using `top_k`.

---

## 5. Metadata Filtering

Search can optionally be restricted by category.

Example:

```json
{
  "query": "What technologies are used in AI?",
  "top_k": 3,
  "category": "AI"
}
```

This combines:

```text
Semantic similarity
        +
Metadata filtering
```

---

# API

## Health Check

### `GET /health`

Example response:

```json
{
  "status": "healthy",
  "documents": 5
}
```

---

## Semantic Search

### `POST /api/v1/search`

Request:

```json
{
  "query": "How can I build a Python API?",
  "top_k": 3
}
```

The response contains:

* document ID
* document text
* source
* category
* distance

---

## Metadata Filter

Request:

```json
{
  "query": "What technologies are used in AI?",
  "top_k": 3,
  "category": "AI"
}
```

Only documents matching the requested category are considered.

---

# Running Locally

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run pytest
```

Start the API:

```bash
uv run uvicorn app.main:app --reload
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

---

# Running with Docker

Build the image:

```bash
docker build -t semantic-search-api .
```

Run the container:

```bash
docker run -p 8000:8000 \
  --name semantic-search-container \
  semantic-search-api
```

The Docker image supports the `PORT` environment variable.

Default:

```text
8000
```

For example:

```bash
docker run -p 8000:8000 \
  -e PORT=8000 \
  --name semantic-search-container \
  semantic-search-api
```

---

# Persistent ChromaDB Storage

To persist the vector database outside the container, create a Docker volume:

```bash
docker volume create semantic-search-data
```

Run:

```bash
docker run -p 8000:8000 \
  -v semantic-search-data:/app/chroma_data \
  --name semantic-search-container \
  semantic-search-api
```

The volume allows ChromaDB data to survive container recreation.

---

# Testing

The project includes tests covering:

* Document ingestion
* Document fields
* Embedding dimensions
* Document embeddings
* Query embeddings
* Vector-store insertion
* Vector-store search
* Top-K retrieval
* Metadata filtering
* Retrieval service
* Query validation
* API endpoints
* API validation

Current test result:

```text
26 passed
```

---

# Design Decisions

### Embedding Model

`all-MiniLM-L6-v2` was selected because it provides a practical balance between:

* Embedding quality
* Model size
* Inference speed
* CPU compatibility
* 384-dimensional embeddings

### Vector Database

ChromaDB was selected for the project because it provides a simple application-oriented interface for:

* Vector storage
* Similarity search
* Metadata
* Filtering
* Persistence

### Similarity Metric

Cosine similarity/distance is appropriate for comparing semantic embeddings because the direction of the vectors is more important than their raw magnitude.

### Docker

The application uses CPU-only PyTorch because this semantic-search application does not require GPU inference.

---

# Limitations

This project is intentionally focused on the vector-search foundation.

Current limitations include:

* Text documents only
* No PDF/DOCX ingestion
* Small demonstration dataset
* No authentication
* No rate limiting
* No production monitoring
* No distributed vector database
* Retrieval evaluation uses a small manually created evaluation set

These are potential extensions rather than requirements for the current project.

---

# Future Improvements

Possible extensions include:

1. PDF and DOCX document ingestion
2. Document chunking
3. Hybrid keyword + semantic search
4. Reranking models
5. Larger evaluation datasets
6. Retrieval quality monitoring
7. Authentication
8. Rate limiting
9. Background document ingestion
10. Production cloud vector database
11. RAG integration with an LLM

---

# Learning Outcomes

This project demonstrates understanding of:

* Semantic search
* Embeddings
* Vector representations
* Cosine similarity
* Vector databases
* ChromaDB
* Metadata filtering
* Top-K retrieval
* Embedding model selection
* Retrieval evaluation
* FastAPI integration
* Docker deployment
* Persistent vector storage

---

## Project Status

**Status: Complete**

The project satisfies the Vector Database module project requirements:

* Document ingestion
* Embedding generation
* Vector storage
* Top-K retrieval
* Metadata filtering
* Live API demonstration
* Docker deployment
