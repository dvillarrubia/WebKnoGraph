# CLAUDE.md - AI Assistant Guide for WebKnoGraph

This document provides essential context for AI assistants working with the WebKnoGraph codebase.

## Project Overview

**WebKnoGraph** is a Graph-RAG (Retrieval Augmented Generation) system that combines:
- Web crawling and content extraction
- Vector embeddings for semantic search (pgvector)
- Graph-based context expansion via internal link structure (Neo4j)
- Multi-tenant support with isolated data per client
- LLM-powered question answering (OpenAI GPT-4o)

**Core Goal**: Optimize website internal linking through AI-driven graph analysis and link prediction using vector embeddings and graph neural networks.

**License**: Apache 2.0

## Directory Structure

```
WebKnoGraph/
├── graph_rag/                    # Main RAG service (primary development focus)
│   ├── api/                      # FastAPI endpoints & Pydantic models
│   │   ├── main.py               # FastAPI application entry point
│   │   ├── routes.py             # 68+ API endpoints (admin, client, dashboard)
│   │   ├── models.py             # Request/response schemas
│   │   └── dependencies.py       # Dependency injection & auth
│   ├── config/
│   │   └── settings.py           # Environment-based configuration
│   ├── db/                       # Database clients
│   │   ├── supabase_client.py    # PostgreSQL/pgvector async client
│   │   ├── neo4j_client.py       # Neo4j graph database client
│   │   ├── migrations/           # PostgreSQL schema (001_create_schema.sql)
│   │   └── neo4j/                # Neo4j constraints (001_constraints.cypher)
│   ├── services/                 # Core business logic
│   │   ├── rag_service.py        # RAG query orchestration
│   │   ├── embedding_service.py  # Vector embedding generation
│   │   ├── ingest_service.py     # Data ingestion pipeline
│   │   ├── crawler_service.py    # Web crawl job management
│   │   ├── manual_cleaner_service.py  # Content cleaning
│   │   └── chunking_service.py   # Semantic text chunking
│   ├── scripts/                  # Database migration scripts
│   ├── static/                   # Dashboard SPA (index.html)
│   └── Dockerfile                # Container definition (Python 3.11)
│
├── src/                          # Original WebKnoGraph modules (legacy)
│   └── backend/
│       ├── services/             # Crawler, embeddings, PageRank, GraphSAGE
│       ├── graph/                # Graph algorithms
│       └── models/               # ML models
│
├── scripts/                      # Standalone CLI scripts
│   ├── crawl4ai_advanced.py      # Advanced web crawler
│   └── smart_cleaner.py          # Content cleaning utility
│
├── tests/                        # Test suite (unittest-based)
│   └── backend/services/         # Service unit tests
│
├── notebooks/                    # Jupyter/Gradio UI notebooks
│
├── data/                         # Runtime data (gitignored)
├── results/                      # Experiment results
└── assets/                       # Project assets
```

## Quick Start Commands

```bash
# Install dependencies
pip install -r requirements-rag.txt

# Run the API server (development)
PYTHONPATH=. python -m uvicorn graph_rag.api.main:app --host 0.0.0.0 --port 8080

# Run with Docker
docker-compose -f docker-compose.rag.yml up -d

# Run tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/backend/services/test_crawler_service.py -v
```

## Key Technologies

| Category | Technologies |
|----------|-------------|
| **API Framework** | FastAPI, Uvicorn, Pydantic |
| **Databases** | PostgreSQL (pgvector), Neo4j |
| **AI/ML** | OpenAI GPT-4o, sentence-transformers (multilingual-e5-large), PyTorch |
| **Data Processing** | Pandas, PyArrow, DuckDB |
| **Graph Algorithms** | NetworkX, PyTorch Geometric (GraphSAGE) |
| **Web Crawling** | Crawl4AI, Trafilatura, BeautifulSoup4 |

## Configuration

Environment variables are loaded from `.env` files. Key settings:

```bash
# Database
DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# AI/Embeddings
OPENAI_API_KEY
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-large
EMBEDDING_DIMENSION=1024

# RAG Settings
RAG_TOP_K_VECTORS=10
RAG_GRAPH_HOPS=2
RAG_MAX_CONTEXT_PAGES=15
RAG_MIN_SIMILARITY=0.5
```

See `graph_rag/.env.example` for all available options.

## Code Conventions

### Python Style
- **Type hints**: Required on all functions and class methods
- **Async/await**: Use for all I/O operations (database, HTTP)
- **Docstrings**: Required on classes and public methods
- **Imports**: Group by standard lib, third-party, local

### Formatting & Linting
Pre-commit hooks enforce:
- **Ruff**: Linting and formatting (Black-compatible)
- No trailing whitespace
- Newline at end of files
- No debug statements (`print`, `pdb`)
- Max file size: 5MB

Run manually:
```bash
pre-commit run --all-files
```

### Naming Conventions
- **Files**: snake_case (`crawler_service.py`)
- **Classes**: PascalCase (`CrawlerService`)
- **Functions/Variables**: snake_case (`get_pages`, `page_count`)
- **Database tables**: snake_case with `rag_` prefix (`rag_pages`, `rag_clients`)
- **API routes**: RESTful paths (`/api/v1/resource`)

### Service Pattern
```python
class XyzService:
    def __init__(self, settings: Settings, db_client: DatabaseClient):
        self.settings = settings
        self.db = db_client

    async def operation(self, param: str) -> ResultType:
        """Docstring explaining the operation."""
        # Implementation
        return result
```

### Multi-tenancy
Always filter by `client_id` when querying data:
```python
# Correct
pages = await db.get_pages(client_id=client_id, limit=10)

# Wrong - missing client_id
pages = await db.get_pages(limit=10)
```

## API Structure

### Authentication
- Admin endpoints: No authentication required (internal use)
- Client endpoints: Require `X-API-Key` header

### Endpoint Categories
- `/api/v1/health` - Health check
- `/api/v1/admin/clients/*` - Client CRUD (admin)
- `/api/v1/query`, `/api/v1/search` - RAG queries (client)
- `/api/v1/dashboard/*` - Crawler, cleaner, ingest (dashboard)

### Response Format
All responses use Pydantic models with consistent structure:
```python
class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    context_stats: dict
```

## Database Schema

### PostgreSQL (pgvector)
- `rag_clients`: Multi-tenant client management
- `rag_pages`: Pages with vector embeddings
- `rag_links`: Internal links (edge list)
- `rag_conversations`: Chat sessions
- `rag_messages`: Conversation messages

### Neo4j
- `:Page` nodes with properties: `client_id`, `url`, `title`, `pagerank`, `hub_score`, `authority_score`
- `:Client` nodes for multi-tenancy
- `:LINKS_TO` relationships for internal links

## Testing

Tests use `unittest.TestCase` with mocking:

```python
class TestCrawlerService(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.service = CrawlerService(self.mock_config)

    def test_crawl_page(self):
        # Test implementation
```

Test file locations mirror source structure:
- `tests/backend/services/test_*.py`

## Common Development Tasks

### Adding a New API Endpoint
1. Define Pydantic models in `graph_rag/api/models.py`
2. Add route function in `graph_rag/api/routes.py`
3. Implement business logic in appropriate service
4. Add tests

### Adding a New Service
1. Create `graph_rag/services/new_service.py`
2. Follow the service pattern with dependency injection
3. Use async/await for I/O
4. Add to service exports in `__init__.py`

### Database Changes
1. Add migration file in `graph_rag/db/migrations/` (PostgreSQL) or `graph_rag/db/neo4j/` (Neo4j)
2. Update corresponding client in `graph_rag/db/`
3. Run migration script

## Data Pipeline Flow

1. **Crawling**: `CrawlerService` uses Crawl4AI → Parquet files
2. **Cleaning**: `ManualCleanerService` detects patterns → cleaned markdown
3. **Ingestion**: `IngestService` → PostgreSQL + Neo4j
4. **Graph Analysis**: Auto-calculated PageRank/HITS post-ingest
5. **RAG Query**: Vector search + graph expansion + LLM generation

## Important Files

| File | Purpose |
|------|---------|
| `graph_rag/api/main.py` | FastAPI app entry point |
| `graph_rag/config/settings.py` | Configuration management |
| `graph_rag/services/rag_service.py` | Core RAG orchestration |
| `graph_rag/db/supabase_client.py` | PostgreSQL/pgvector operations |
| `graph_rag/db/neo4j_client.py` | Neo4j graph operations |
| `docker-compose.rag.yml` | Docker deployment |
| `requirements-rag.txt` | Python dependencies |

## Troubleshooting

### Common Issues

**Import errors**: Ensure `PYTHONPATH=.` is set when running from project root

**Database connection failures**: Check `.env` configuration and ensure services are running

**Embedding model loading**: First run downloads ~2GB model; check disk space and network

**Neo4j memory errors**: Increase heap size in Neo4j configuration for large graphs

### Logs
- API logs: stdout/stderr (uvicorn)
- Service-level logging with Python `logging` module
- Debug mode: Set `DEBUG=true` in `.env`

## Additional Documentation

- `README.md` - Project overview and quick start
- `HOW-IT-WORKS.md` - Detailed architecture and data flow
- `graph_rag/README.md` - Graph-RAG specific documentation
- `graph_rag/TODO_FEATURES.md` - Feature implementation status
- `/docs` endpoint - Swagger API documentation (when running)
