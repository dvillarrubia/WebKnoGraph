# Crawler Service Dockerfile
# Separate container for Crawl4AI + Playwright/Chromium
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Create non-root user FIRST (before installing playwright)
RUN useradd --create-home --shell /bin/bash appuser

# System dependencies for Playwright/Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY crawler_service/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright system deps as root, then browsers as appuser
RUN playwright install-deps chromium

# Switch to appuser for browser install (goes to ~/.cache/ms-playwright)
USER appuser
RUN playwright install chromium

# Switch back to root for file operations
USER root

# Copy crawler service code
COPY crawler_service/ ./crawler_service/

# Create data directory and set ownership
RUN mkdir -p /app/data/crawl4ai_data && \
    chown -R appuser:appuser /app

# Run as non-root
USER appuser

# Expose port
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Run the crawler service
CMD ["python", "-m", "uvicorn", "crawler_service.main:app", "--host", "0.0.0.0", "--port", "8081"]
