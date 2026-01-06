"""
WebKnoGraph Graph-RAG Configuration
Multi-tenant RAG system with Supabase + Neo4j + OpenAI
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==========================================================================
    # APP CONFIG
    # ==========================================================================
    app_name: str = "WebKnoGraph Graph-RAG"
    app_version: str = "1.0.0"
    debug: bool = False

    # ==========================================================================
    # SUPABASE (pgvector)
    # ==========================================================================
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_key: str = Field(default="", alias="SUPABASE_SERVICE_KEY")  # Service role key (optional)
    supabase_db_host: str = Field(default="localhost", env="SUPABASE_DB_HOST")
    supabase_db_port: int = Field(default=54322, env="SUPABASE_DB_PORT")
    supabase_db_name: str = Field(default="postgres", env="SUPABASE_DB_NAME")
    supabase_db_user: str = Field(default="postgres", env="SUPABASE_DB_USER")
    supabase_db_password: str = Field(..., env="SUPABASE_DB_PASSWORD")

    # ==========================================================================
    # NEO4J
    # ==========================================================================
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")

    # ==========================================================================
    # OPENAI
    # ==========================================================================
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-large", env="OPENAI_EMBEDDING_MODEL")

    # ==========================================================================
    # EMBEDDING MODEL (Local - Spanish optimized)
    # Using hiiamsid/sentence_similarity_spanish_es for better Spanish results
    # ==========================================================================
    embedding_model_name: str = Field(
        default="hiiamsid/sentence_similarity_spanish_es",
        env="EMBEDDING_MODEL_NAME"
    )
    embedding_dimension: int = Field(default=768, env="EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")

    # ==========================================================================
    # RAG SETTINGS
    # ==========================================================================
    rag_top_k_vectors: int = Field(default=10, env="RAG_TOP_K_VECTORS")
    rag_graph_hops: int = Field(default=2, env="RAG_GRAPH_HOPS")
    rag_max_context_pages: int = Field(default=15, env="RAG_MAX_CONTEXT_PAGES")
    rag_min_similarity: float = Field(default=0.5, env="RAG_MIN_SIMILARITY")
    rag_context_max_tokens: int = Field(default=8000, env="RAG_CONTEXT_MAX_TOKENS")

    # ==========================================================================
    # API CONFIG
    # ==========================================================================
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8080, env="API_PORT")

    class Config:
        # Look for .env in multiple locations
        env_file = (
            Path(__file__).parent.parent / ".env",
            ".env",
            "graph_rag/.env",
        )
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
