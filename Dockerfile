FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake FastEmbed local model during build-time (stored in /app/.cache)
ENV HF_HOME=/app/.cache
ENV FASTEMBED_CACHE_PATH=/app/.cache
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Copy project files
COPY . .

# Create non-root user and ensure ownership of workspace + pre-downloaded cache
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Bind to port 8000
ENV PORT=8000

# Run uvicorn on port 8000
CMD uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
