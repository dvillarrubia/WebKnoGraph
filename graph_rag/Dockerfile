# WebKnoGraph Graph-RAG Dockerfile (with Crawl4AI + Playwright)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PLAYWRIGHT_BROWSERS_PATH=/home/appuser/.cache/ms-playwright

# Set work directory
WORKDIR /app

# Install system dependencies (including Playwright/Chromium deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    # Playwright/Chromium dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    libwayland-client0 \
    fonts-liberation \
    fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-rag.txt .

# Install Python dependencies + crawl4ai extras
RUN pip install --no-cache-dir -r requirements-rag.txt \
    crawl4ai \
    aiohttp

# Create non-root user before installing browsers
RUN useradd --create-home --shell /bin/bash appuser

# Create HuggingFace cache directory with correct ownership
RUN mkdir -p /home/appuser/.cache/huggingface/hub && \
    chown -R appuser:appuser /home/appuser/.cache

# Install Playwright Chromium as appuser
USER appuser
RUN python -m playwright install chromium
USER root

# Copy application code
COPY graph_rag/ ./graph_rag/
COPY scripts/ ./scripts/
COPY ontologia/ ./ontologia/

# Fix permissions
RUN chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "graph_rag.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
