
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

# Install system utilities needed during installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install the RAG project's dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# Download NLTK sentence tokenizer data
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab

# Copy the RAG application
COPY app/ ./app/

EXPOSE 8000

# Render assigns PORT; use 8000 for local Docker testing
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
