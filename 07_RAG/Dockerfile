
FROM python:3.12-slim

# Python configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES="" \
    NLTK_DATA=/usr/local/share/nltk_data \
    HF_HOME=/tmp/huggingface

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy project dependencies
COPY requirements.txt .

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install CPU-only PyTorch
RUN pip install torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install -r requirements.txt

# Download NLTK tokenizer resources
RUN python -m nltk.downloader \
    -d /usr/local/share/nltk_data \
    punkt punkt_tab

# Copy application source code
COPY app/ ./app/

# Expose application port
EXPOSE 8000

# Start FastAPI
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
