# Extractive Question Answering API

A production-oriented Extractive Question Answering system built using
RoBERTa, Hugging Face Transformers, FastAPI, and Docker.

## Project Overview

This project provides an API that accepts a question and a context and
returns the most likely answer span from the given context.

The system also supports no-answer detection using a SQuAD 2.0-trained
RoBERTa model.

## Model

Model used:

deepset/roberta-base-squad2

The model is designed for extractive question answering and supports
answerable as well as unanswerable questions.

## Architecture

Client
   |
   v
FastAPI
   |
   v
QA Prediction Module
   |
   +--> Tokenization
   |
   +--> RoBERTa
   |
   +--> Start/End Logits
   |
   +--> Best Answer Span
   |
   +--> No-Answer Detection
   |
   v
JSON Response

## API Endpoints

### GET /

Returns the API status.

### GET /health

Health-check endpoint.

Example response:

{
    "status": "healthy",
    "model": "MiniLM
deepset/minilm-uncased-squad2"
    
}

### POST /predict

Accepts:

{
    "question": "Who created Python?",
    "context": "Python was created by Guido van Rossum and first released in 1991."
}

Example response:

{
    "question": "Who created Python?",
    "answer": "Guido van Rossum",
    "has_answer": true,
    "best_span_score": 16.28,
    "null_score": 3.30,
    "score_difference": -12.98
}

## Project Structure

api/
|
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .dockerignore
├── README.md
|
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── qa_model.py
|
└── tests/
    └── test_api.py

## Testing

The project contains automated API tests using pytest.

Current tests:

- Health endpoint
- Answerable question
- Unanswerable question

All three tests pass successfully.

## Docker

The application is containerized using Docker.

Build:

docker build -t qa-system-api .

Run:

docker run -p 8000:8000 --name qa-system-container qa-system-api

API documentation:

http://localhost:8000/docs

## Docker Image Optimization

The initial Docker image used a CUDA-enabled PyTorch installation even
though the application performs CPU inference.

Initial image:

8.23 GB disk usage

The container reported:

CUDA available: False

Therefore, GPU dependencies were unnecessary for the current deployment.

The PyTorch dependency was changed to the CPU-only build:

torch==2.13.0+cpu

After optimization:

1.68 GB disk usage

This resulted in approximately an 80% reduction in Docker disk usage.

## Production Decisions

### CPU-only inference

The application currently performs CPU inference, so GPU/CUDA
dependencies were removed from the production image.

### Separate production and development dependencies

Production dependencies are stored in:

requirements.txt

Testing dependencies are stored in:

requirements-dev.txt

This prevents development-only packages from unnecessarily increasing
the production image size.

### Model loaded once

The model is loaded when the application starts rather than for every
API request. This avoids repeatedly loading the model into memory.

### Health endpoint

The `/health` endpoint allows deployment platforms and monitoring systems
to verify that the API is running.

## Future Improvements

- Cloud deployment
- Public API endpoint
- Authentication
- Request logging
- Performance monitoring
- Better QA evaluation dataset
- Automated CI/CD
- Model optimization