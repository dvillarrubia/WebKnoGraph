# WebKnoGraph - Production Deployment

Deploy WebKnoGraph (API + Crawler) in Docker. The databases (PostgreSQL/pgvector and Neo4j) must already be running as containers on the same Docker network.

## Prerequisites

- Docker and Docker Compose installed
- A Docker network named `webknograph-network` already created
- PostgreSQL with pgvector extension running on `webknograph-network`
- Neo4j running on `webknograph-network`

```bash
# Create the network if it doesn't exist
docker network create webknograph-network
```

## Setup

### 1. Copy the `.dockerignore` to the project root

The build context is the project root. Copy the optimized `.dockerignore` to reduce build context size:

```bash
cp deploy/.dockerignore ../
```

### 2. Configure environment variables

```bash
cp .env.production.example .env.production
```

Edit `.env.production` with your actual values:
- Database connection details (use container names as hosts)
- API keys (OpenAI, Gemini)
- Adjust `WEB_CONCURRENCY` based on available CPU cores

### 3. Build images

```bash
docker compose -f docker-compose.prod.yml build
```

### 4. Start services

```bash
docker compose -f docker-compose.prod.yml up -d
```

The API will take ~2 minutes to become healthy while the embedding model loads for the first time. Subsequent starts will be faster thanks to the `huggingface_cache` volume.

## Verification

```bash
# Check service status
docker compose -f docker-compose.prod.yml ps

# API health check
curl http://localhost:8080/api/v1/health

# Crawler health check (internal only, from within the network)
docker compose -f docker-compose.prod.yml exec crawler-service curl http://localhost:8081/health

# View logs
docker compose -f docker-compose.prod.yml logs -f
docker compose -f docker-compose.prod.yml logs -f graph-rag-api
docker compose -f docker-compose.prod.yml logs -f crawler-service
```

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │          webknograph-network             │
                    │                                         │
 :8080 ───────────► │  graph-rag-api ──► crawler-service      │
                    │       │                   │              │
                    │       ▼                   ▼              │
                    │   supabase-db     crawl_data (volume)    │
                    │   neo4j                                  │
                    └─────────────────────────────────────────┘
```

- **graph-rag-api**: Main API (FastAPI + Gunicorn). Exposed on port 8080.
- **crawler-service**: Crawler with Playwright/Chromium (1 worker). Internal only.
- Databases are external containers on the same network.

## Troubleshooting

**API not becoming healthy**
- Check logs: `docker compose -f docker-compose.prod.yml logs graph-rag-api`
- The embedding model download can take several minutes on first run
- Ensure database containers are reachable on the network

**Crawler not starting**
- Playwright/Chromium requires sufficient memory (~512MB minimum)
- Check logs: `docker compose -f docker-compose.prod.yml logs crawler-service`

**Database connection errors**
- Verify container names in `.env.production` match actual DB container names
- Ensure all containers are on `webknograph-network`: `docker network inspect webknograph-network`

## Updating

```bash
# Pull latest code, then rebuild and restart
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
```

## Stopping

```bash
docker compose -f docker-compose.prod.yml down

# To also remove volumes (caution: deletes cached models and crawl data)
docker compose -f docker-compose.prod.yml down -v
```
