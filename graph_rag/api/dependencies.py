"""
FastAPI Dependencies for Graph-RAG.
Handles dependency injection and authentication.
"""

from typing import Optional
from fastapi import Header, HTTPException, Depends

from graph_rag.config.settings import Settings, get_settings
from graph_rag.db.supabase_client import SupabaseClient
from graph_rag.db.neo4j_client import Neo4jClient
from graph_rag.services.embedding_service import EmbeddingService
from graph_rag.services.reranking_service import RerankingService
from graph_rag.services.rag_service import RAGService
from graph_rag.services.migration_service import MigrationService


# Singletons
_supabase_client: Optional[SupabaseClient] = None
_neo4j_client: Optional[Neo4jClient] = None
_embedding_service: Optional[EmbeddingService] = None
_reranking_service: Optional[RerankingService] = None
_rag_service: Optional[RAGService] = None
_migration_service: Optional[MigrationService] = None


async def get_supabase_client() -> SupabaseClient:
    """Get Supabase client singleton."""
    global _supabase_client
    if _supabase_client is None:
        settings = get_settings()
        _supabase_client = SupabaseClient(settings)
        await _supabase_client.connect()
    return _supabase_client


async def get_neo4j_client() -> Neo4jClient:
    """Get Neo4j client singleton."""
    global _neo4j_client
    if _neo4j_client is None:
        settings = get_settings()
        _neo4j_client = Neo4jClient(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        await _neo4j_client.connect()
    return _neo4j_client


def get_embedding_service() -> EmbeddingService:
    """Get embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        settings = get_settings()
        _embedding_service = EmbeddingService(settings)
    return _embedding_service


def get_reranking_service() -> Optional[RerankingService]:
    """Get reranking service singleton (lazy loaded)."""
    global _reranking_service
    if _reranking_service is None:
        settings = get_settings()
        if settings.rag_use_reranking:
            _reranking_service = RerankingService(
                model_name=settings.reranking_model,
            )
    return _reranking_service


async def get_rag_service() -> RAGService:
    """Get RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        settings = get_settings()
        supabase = await get_supabase_client()
        neo4j = await get_neo4j_client()
        embedding = get_embedding_service()
        reranking = get_reranking_service()
        _rag_service = RAGService(settings, supabase, neo4j, embedding, reranking)
    return _rag_service


async def get_migration_service() -> MigrationService:
    """Get migration service singleton."""
    global _migration_service
    if _migration_service is None:
        settings = get_settings()
        supabase = await get_supabase_client()
        neo4j = await get_neo4j_client()
        embedding = get_embedding_service()
        _migration_service = MigrationService(settings, supabase, neo4j, embedding)
    return _migration_service


async def verify_api_key(
    x_api_key: str = Header(..., description="Client API key"),
    supabase: SupabaseClient = Depends(get_supabase_client),
) -> dict:
    """
    Verify API key and return client info.
    Used as a dependency for authenticated endpoints.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")

    client = await supabase.get_client_by_api_key(x_api_key)
    if not client:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not client.get("is_active", False):
        raise HTTPException(status_code=403, detail="Client is inactive")

    return client


async def shutdown_clients():
    """Shutdown all clients on app shutdown."""
    global _supabase_client, _neo4j_client

    if _supabase_client:
        await _supabase_client.disconnect()
        _supabase_client = None

    if _neo4j_client:
        await _neo4j_client.disconnect()
        _neo4j_client = None
