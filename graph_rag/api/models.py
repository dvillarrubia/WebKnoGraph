"""
Pydantic models for API request/response.
"""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


# =============================================================================
# CLIENT MODELS
# =============================================================================

class ClientCreate(BaseModel):
    """Request to create a new client."""
    name: str = Field(..., min_length=1, max_length=255)
    domain: str = Field(..., min_length=1, max_length=255)


class ClientResponse(BaseModel):
    """Client information response."""
    id: str
    name: str
    domain: str
    api_key: Optional[str] = None  # Only returned on creation
    is_active: bool = True
    created_at: Optional[datetime] = None


class ClientListResponse(BaseModel):
    """List of clients."""
    clients: list[ClientResponse]
    total: int


# =============================================================================
# RAG MODELS
# =============================================================================

class QueryRequest(BaseModel):
    """RAG query request."""
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_history: Optional[list[dict]] = None
    top_k_vectors: Optional[int] = Field(default=None, ge=1, le=50)
    graph_hops: Optional[int] = Field(default=None, ge=0, le=3)
    max_context_pages: Optional[int] = Field(default=None, ge=1, le=30)
    min_similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SourceInfo(BaseModel):
    """Source page information."""
    url: str
    title: Optional[str] = None
    pagerank: float = 0.0
    similarity: float = 0.0
    from_graph: bool = False


class QueryResponse(BaseModel):
    """RAG query response."""
    answer: str
    sources: list[SourceInfo]
    context_stats: dict
    tokens_used: int


# =============================================================================
# SEARCH MODELS
# =============================================================================

class SearchRequest(BaseModel):
    """Vector search request."""
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)
    min_pagerank: float = Field(default=0.0, ge=0.0)


class SearchResult(BaseModel):
    """Single search result."""
    url: str
    title: Optional[str] = None
    content_preview: Optional[str] = None
    pagerank: float = 0.0
    similarity: float = 0.0


class SearchResponse(BaseModel):
    """Search results response."""
    results: list[SearchResult]
    total: int


# =============================================================================
# GRAPH MODELS
# =============================================================================

class RelatedPagesRequest(BaseModel):
    """Request for related pages."""
    url: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=50)


class RelatedPagesResponse(BaseModel):
    """Related pages response."""
    source_url: str
    related: list[dict]
    total: int


class PathRequest(BaseModel):
    """Request for path between pages."""
    source_url: str = Field(..., min_length=1)
    target_url: str = Field(..., min_length=1)


class PathResponse(BaseModel):
    """Path response."""
    source_url: str
    target_url: str
    path: Optional[list[str]] = None
    path_length: Optional[int] = None


# =============================================================================
# MIGRATION MODELS
# =============================================================================

class MigrationRequest(BaseModel):
    """Migration request."""
    data_path: str = Field(..., description="Path to WebKnoGraph data folder")
    regenerate_embeddings: bool = Field(default=True, description="Re-generate embeddings with new model")
    batch_size: int = Field(default=100, ge=10, le=1000)


class MigrationResponse(BaseModel):
    """Migration result."""
    pages_migrated: int
    links_migrated: int
    embeddings_generated: int
    errors: list[str]


# =============================================================================
# STATS MODELS
# =============================================================================

class ClientStats(BaseModel):
    """Client statistics."""
    client_id: str
    pages_count: int
    links_count: int
    pages_with_embeddings: int
