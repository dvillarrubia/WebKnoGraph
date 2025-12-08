"""
FastAPI Application for Graph-RAG.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from graph_rag.api.routes import public_router, admin_router, client_router, dashboard_router
from graph_rag.api.dependencies import shutdown_clients
from graph_rag.config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings = get_settings()
    print(f"Starting {settings.app_name} v{settings.app_version}")
    yield
    # Shutdown
    await shutdown_clients()
    print("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
# WebKnoGraph Graph-RAG API

Multi-tenant RAG system that combines:
- **Vector Search** (pgvector) - Semantic similarity search
- **Graph Expansion** (Neo4j) - Context enrichment via link structure
- **LLM Generation** (OpenAI) - Natural language responses

## Authentication

Most endpoints require an API key passed in the `X-API-Key` header.
API keys are generated when creating a client via the admin endpoints.

## Endpoints

### Public
- `GET /api/v1/health` - Health check

### Admin (Client Management)
- `POST /api/v1/admin/clients` - Create new client
- `GET /api/v1/admin/clients` - List all clients
- `GET /api/v1/admin/clients/{id}` - Get client details
- `POST /api/v1/admin/clients/{id}/migrate` - Migrate data
- `GET /api/v1/admin/clients/{id}/stats` - Get client stats

### Client (Requires API Key)
- `POST /api/v1/query` - Execute RAG query
- `POST /api/v1/search` - Vector similarity search
- `POST /api/v1/related` - Get related pages via graph
- `POST /api/v1/path` - Find path between pages
        """,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(public_router)
    app.include_router(admin_router)
    app.include_router(client_router)
    app.include_router(dashboard_router)

    # Static files for dashboard
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Root route serves the dashboard
    @app.get("/")
    async def serve_dashboard():
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "Dashboard not found. Visit /docs for API documentation."}

    return app


# Application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "graph_rag.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
