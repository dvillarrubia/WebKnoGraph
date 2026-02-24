"""
FastAPI Routes for Graph-RAG.
"""

from fastapi import APIRouter, Depends, HTTPException

from graph_rag.api.models import (
    ClientCreate,
    ClientResponse,
    ClientListResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
    SearchRequest,
    SearchResponse,
    SearchResult,
    RelatedPagesRequest,
    RelatedPagesResponse,
    PathRequest,
    PathResponse,
    MigrationRequest,
    MigrationResponse,
    ClientStats,
)
from graph_rag.api.dependencies import (
    verify_api_key,
    get_supabase_client,
    get_neo4j_client,
    get_rag_service,
    get_migration_service,
    get_embedding_service,
)
from graph_rag.db.supabase_client import SupabaseClient
from graph_rag.db.neo4j_client import Neo4jClient
from graph_rag.services.rag_service import RAGService
from graph_rag.services.migration_service import MigrationService
from graph_rag.services.embedding_service import EmbeddingService
from graph_rag.services.community_service import CommunityService


# =============================================================================
# ROUTERS
# =============================================================================

# Public router (no auth)
public_router = APIRouter(prefix="/api/v1", tags=["public"])

# Admin router (for client management)
admin_router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

# Client router (requires API key)
client_router = APIRouter(prefix="/api/v1", tags=["client"])


# =============================================================================
# PUBLIC ENDPOINTS
# =============================================================================

@public_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "graph-rag"}


# =============================================================================
# ADMIN ENDPOINTS (Client Management)
# =============================================================================

@admin_router.post("/clients", response_model=ClientResponse)
async def create_client(
    request: ClientCreate,
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    """Create a new client. Returns the API key (only shown once)."""
    try:
        client = await supabase.create_client(request.name, request.domain)
        return ClientResponse(
            id=str(client["id"]),
            name=client["name"],
            domain=client["domain"],
            api_key=client["api_key"],
            created_at=client["created_at"],
        )
    except Exception as e:
        if "unique" in str(e).lower():
            raise HTTPException(status_code=400, detail="Domain already exists")
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.get("/clients", response_model=ClientListResponse)
async def list_clients(
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    """List all clients."""
    clients = await supabase.list_clients()
    return ClientListResponse(
        clients=[
            ClientResponse(
                id=str(c["id"]),
                name=c["name"],
                domain=c["domain"],
                is_active=c["is_active"],
                created_at=c["created_at"],
            )
            for c in clients
        ],
        total=len(clients),
    )


@admin_router.get("/clients/{client_id}", response_model=ClientResponse)
async def get_client(
    client_id: str,
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    """Get client by ID."""
    client = await supabase.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    return ClientResponse(
        id=str(client["id"]),
        name=client["name"],
        domain=client["domain"],
        api_key=client["api_key"],
        is_active=client["is_active"],
        created_at=client["created_at"],
    )


@admin_router.post("/clients/{client_id}/migrate", response_model=MigrationResponse)
async def migrate_data(
    client_id: str,
    request: MigrationRequest,
    migration: MigrationService = Depends(get_migration_service),
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    """Migrate data from WebKnoGraph data folder to the client."""
    # Verify client exists
    client = await supabase.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    # Run migration
    stats = await migration.migrate_from_webknograph(
        client_id=client_id,
        data_path=request.data_path,
        regenerate_embeddings=request.regenerate_embeddings,
        batch_size=request.batch_size,
    )

    return MigrationResponse(**stats)


@admin_router.get("/clients/{client_id}/stats", response_model=ClientStats)
async def get_client_stats(
    client_id: str,
    supabase: SupabaseClient = Depends(get_supabase_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """Get statistics for a client."""
    # Verify client exists
    client = await supabase.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    pages_count = await supabase.get_pages_count(client_id)
    neo4j_stats = await neo4j.get_client_stats(client_id)

    return ClientStats(
        client_id=client_id,
        pages_count=pages_count,
        links_count=neo4j_stats["link_count"],
        pages_with_embeddings=pages_count,  # Simplified
    )


# =============================================================================
# CLIENT ENDPOINTS (Require API Key)
# =============================================================================

@client_router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    client: dict = Depends(verify_api_key),
    rag: RAGService = Depends(get_rag_service),
):
    """
    Execute a RAG query.
    Combines vector search + graph expansion + LLM generation.
    """
    response = await rag.query(
        client_id=str(client["id"]),
        question=request.question,
        conversation_history=request.conversation_history,
        top_k_vectors=request.top_k_vectors,
        graph_hops=request.graph_hops,
        max_context_pages=request.max_context_pages,
        min_similarity=request.min_similarity,
    )

    return QueryResponse(
        answer=response.answer,
        sources=[
            SourceInfo(
                url=s["url"],
                title=s.get("title"),
                pagerank=s.get("pagerank", 0),
                similarity=s.get("similarity", 0),
                from_graph=s.get("from_graph", False),
            )
            for s in response.sources
        ],
        context_stats={
            "vector_results": response.context.vector_results,
            "graph_expanded": response.context.graph_expanded,
            "total_pages": len(response.context.pages),
            "context_tokens": response.context.total_tokens,
        },
        tokens_used=response.tokens_used,
    )


@client_router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    client: dict = Depends(verify_api_key),
    supabase: SupabaseClient = Depends(get_supabase_client),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """
    Vector similarity search without LLM generation.
    Useful for finding relevant pages.
    """
    # Generate query embedding
    query_embedding = embedding_service.embed_query(request.query)

    # Search
    results = await supabase.search_similar_pages(
        client_id=str(client["id"]),
        query_embedding=query_embedding,
        limit=request.limit,
        min_similarity=request.min_similarity,
        min_pagerank=request.min_pagerank,
    )

    return SearchResponse(
        results=[
            SearchResult(
                url=r["url"],
                title=r.get("title"),
                content_preview=r.get("content", "")[:300] + "..." if r.get("content") else None,
                pagerank=r.get("pagerank", 0),
                similarity=r.get("similarity", 0),
            )
            for r in results
        ],
        total=len(results),
    )


@client_router.post("/related", response_model=RelatedPagesResponse)
async def get_related_pages(
    request: RelatedPagesRequest,
    client: dict = Depends(verify_api_key),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """
    Get pages related to a given URL via internal links.
    Uses graph traversal.
    """
    related = await neo4j.get_linked_pages(
        client_id=str(client["id"]),
        url=request.url,
        hops=2,
        limit=request.limit,
    )

    return RelatedPagesResponse(
        source_url=request.url,
        related=related,
        total=len(related),
    )


@client_router.post("/path", response_model=PathResponse)
async def find_path(
    request: PathRequest,
    client: dict = Depends(verify_api_key),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """
    Find the shortest path between two pages.
    """
    path = await neo4j.get_shortest_path(
        client_id=str(client["id"]),
        source_url=request.source_url,
        target_url=request.target_url,
    )

    return PathResponse(
        source_url=request.source_url,
        target_url=request.target_url,
        path=path,
        path_length=len(path) - 1 if path else None,
    )


@client_router.get("/pages/{url:path}")
async def get_page(
    url: str,
    client: dict = Depends(verify_api_key),
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    """Get page details by URL."""
    page = await supabase.get_page_by_url(str(client["id"]), url)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")
    return page


# =============================================================================
# DASHBOARD ENDPOINTS (No API Key - for internal dashboard use)
# =============================================================================

dashboard_router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@dashboard_router.get("/clients")
async def dashboard_list_clients(
    supabase: SupabaseClient = Depends(get_supabase_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """List all clients with stats for the dashboard."""
    clients = await supabase.list_clients()
    result = []
    for c in clients:
        client_id = str(c["id"])
        pages_count = await supabase.get_pages_count(client_id)
        neo4j_stats = await neo4j.get_client_stats(client_id)
        result.append({
            "id": client_id,
            "name": c["name"],
            "domain": c["domain"],
            "is_active": c.get("is_active", True),
            "pages_count": pages_count,
            "links_count": neo4j_stats.get("link_count", 0),
            "created_at": c.get("created_at"),
        })
    return result


@dashboard_router.post("/query")
async def dashboard_query(
    request: dict,
    supabase: SupabaseClient = Depends(get_supabase_client),
    rag: RAGService = Depends(get_rag_service),
):
    """Execute a RAG query from the dashboard (no API key required)."""
    client_id = request.get("client_id")
    question = request.get("query") or request.get("question")
    session_id = request.get("session_id")
    conversation_id = request.get("conversation_id")

    if not client_id:
        raise HTTPException(status_code=400, detail="client_id is required")
    if not question:
        raise HTTPException(status_code=400, detail="query is required")

    # Verify client exists
    client = await supabase.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    # Handle conversation history
    conversation_history = []
    if conversation_id:
        # Get existing conversation history
        messages = await supabase.get_conversation_messages(conversation_id, limit=20)
        conversation_history = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]
    elif session_id:
        # Create new conversation for this session
        conversation_id = await supabase.create_conversation(client_id, session_id)

    response = await rag.query(
        client_id=client_id,
        question=question,
        conversation_history=conversation_history if conversation_history else None,
        top_k_vectors=request.get("top_k", 15),
        graph_hops=request.get("graph_hops", 2),
        max_context_pages=request.get("max_context_pages", request.get("top_k", 15)),
        min_similarity=request.get("min_similarity", 0.3),
        use_reranking=request.get("use_reranking", True),
    )

    # Save messages to conversation
    if conversation_id:
        # Save user message
        context_pages = [s["url"] for s in response.sources[:5]]
        await supabase.add_message(
            conversation_id=conversation_id,
            role="user",
            content=question,
            context_pages=context_pages,
        )
        # Save assistant response
        await supabase.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response.answer,
            tokens_used=response.tokens_used,
        )

    return {
        "answer": response.answer,
        "conversation_id": conversation_id,
        "sources": [
            {
                "url": s["url"],
                "title": s.get("title"),
                "pagerank": s.get("pagerank", 0),
                "similarity": s.get("similarity", 0),
            }
            for s in response.sources
        ],
        "context_stats": {
            "vector_results": response.context.vector_results,
            "graph_expanded": response.context.graph_expanded,
            "total_chunks": len(response.context.chunks),
            "reranked": response.context.reranked,
        },
    }


# =============================================================================
# CONVERSATION HISTORY ENDPOINTS
# =============================================================================

@dashboard_router.get("/conversations/{client_id}")
async def dashboard_list_conversations(
    client_id: str,
    limit: int = 20,
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    """List recent conversations for a client."""
    client = await supabase.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    conversations = await supabase.get_client_conversations(client_id, limit=limit)
    return {
        "conversations": [
            {
                "id": str(c["id"]),
                "session_id": c["session_id"],
                "created_at": c["created_at"].isoformat() if c["created_at"] else None,
                "message_count": c["message_count"],
                "last_message_at": c["last_message_at"].isoformat() if c["last_message_at"] else None,
                "first_query": c["first_query"][:100] + "..." if c["first_query"] and len(c["first_query"]) > 100 else c["first_query"],
            }
            for c in conversations
        ],
        "total": len(conversations),
    }


@dashboard_router.get("/conversation/{conversation_id}")
async def dashboard_get_conversation(
    conversation_id: str,
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    """Get all messages from a conversation."""
    messages = await supabase.get_conversation_messages(conversation_id, limit=100)

    return {
        "conversation_id": conversation_id,
        "messages": [
            {
                "id": str(m["id"]),
                "role": m["role"],
                "content": m["content"],
                "context_pages": m["context_pages"],
                "tokens_used": m["tokens_used"],
                "created_at": m["created_at"].isoformat() if m["created_at"] else None,
            }
            for m in messages
        ],
        "total": len(messages),
    }


@dashboard_router.delete("/conversation/{conversation_id}")
async def dashboard_delete_conversation(
    conversation_id: str,
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    """Delete a conversation and all its messages."""
    deleted = await supabase.delete_conversation(conversation_id)
    return {
        "deleted": deleted,
        "conversation_id": conversation_id,
    }


@dashboard_router.post("/conversation/new")
async def dashboard_new_conversation(
    request: dict,
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    """Create a new conversation for a client."""
    client_id = request.get("client_id")
    session_id = request.get("session_id")

    if not client_id:
        raise HTTPException(status_code=400, detail="client_id is required")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    # Verify client exists
    client = await supabase.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    conversation_id, is_new = await supabase.get_or_create_conversation(client_id, session_id)

    return {
        "conversation_id": conversation_id,
        "is_new": is_new,
        "session_id": session_id,
    }


@dashboard_router.post("/search")
async def dashboard_search(
    request: dict,
    supabase: SupabaseClient = Depends(get_supabase_client),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """Vector search from the dashboard using chunks (no API key required)."""
    client_id = request.get("client_id")
    query = request.get("query")

    if not client_id:
        raise HTTPException(status_code=400, detail="client_id is required")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    # Generate query embedding
    query_embedding = embedding_service.embed_query(query)

    # Search in chunks (more granular than pages)
    # Lower min_similarity to 0.2 to get more results, then filter by normalized score
    results = await supabase.search_similar_chunks(
        client_id=client_id,
        query_embedding=query_embedding,
        limit=request.get("top_k", 10),
        min_similarity=request.get("min_similarity", 0.2),
    )

    def normalize_similarity(raw_sim: float) -> float:
        """
        Normalize cosine similarity to a more intuitive 0-100% scale.
        For Spanish sentence-transformers models, typical useful range is 0.25-0.55.
        We map: 0.25 -> 0%, 0.4 -> 50%, 0.55+ -> 100%
        This gives more intuitive scores where 0.47 (match exacto) -> ~73%
        """
        min_threshold = 0.25
        max_threshold = 0.55
        if raw_sim <= min_threshold:
            return 0.0
        if raw_sim >= max_threshold:
            return 1.0
        return (raw_sim - min_threshold) / (max_threshold - min_threshold)

    # Apply URL filter if provided
    url_filter = request.get("url_filter")
    if url_filter:
        results = [r for r in results if url_filter.lower() in r.get("url", "").lower()]

    return {
        "results": [
            {
                "chunk_id": r.get("id"),
                "chunk_index": r.get("chunk_index"),
                "chunk_content": r.get("content", ""),
                "heading_context": r.get("heading_context"),
                "page_id": r.get("page_id"),
                "url": r["url"],
                "title": r.get("title"),
                "pagerank": r.get("pagerank", 0),
                "similarity": normalize_similarity(r.get("similarity", 0)),
                "raw_similarity": r.get("similarity", 0),
            }
            for r in results
        ],
        "total": len(results),
    }


@dashboard_router.post("/compare")
async def dashboard_compare_text(
    request: dict,
    supabase: SupabaseClient = Depends(get_supabase_client),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """Compare text against indexed chunks or pages (no API key required)."""
    client_id = request.get("client_id")
    text = request.get("text")
    mode = request.get("mode", "chunks")  # 'chunks' or 'pages'
    top_k = request.get("top_k", 10)

    if not client_id:
        raise HTTPException(status_code=400, detail="client_id is required")
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    # Generate embedding for the input text
    text_embedding = embedding_service.embed_query(text)

    if mode == "pages":
        # Search against page embeddings
        results = await supabase.search_similar_pages(
            client_id=client_id,
            query_embedding=text_embedding,
            limit=top_k,
            min_similarity=0.15,
        )
        return {
            "results": [
                {
                    "url": r["url"],
                    "title": r.get("title"),
                    "content": (r.get("content") or "")[:500],
                    "similarity": r.get("similarity", 0),
                    "pagerank": r.get("pagerank", 0),
                }
                for r in results
            ],
            "mode": "pages",
        }
    else:
        # Search against chunk embeddings (default)
        results = await supabase.search_similar_chunks(
            client_id=client_id,
            query_embedding=text_embedding,
            limit=top_k,
            min_similarity=0.15,
        )
        return {
            "results": [
                {
                    "url": r["url"],
                    "title": r.get("title"),
                    "chunk_content": (r.get("content") or "")[:500],
                    "chunk_index": r.get("chunk_index"),
                    "heading_context": r.get("heading_context"),
                    "similarity": r.get("similarity", 0),
                }
                for r in results
            ],
            "mode": "chunks",
        }


@dashboard_router.post("/related")
async def dashboard_related_pages(
    request: dict,
    supabase: SupabaseClient = Depends(get_supabase_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """Get related pages via graph traversal (no API key required)."""
    client_id = request.get("client_id")
    url = request.get("url")

    if not client_id:
        raise HTTPException(status_code=400, detail="client_id is required")
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    # Verify client exists
    client = await supabase.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    # Get related pages from graph
    related = await neo4j.get_linked_pages(
        client_id=client_id,
        url=url,
        hops=request.get("hops", 2),
        limit=request.get("limit", 20),
    )

    # Get incoming links (backlinks)
    backlinks = await neo4j.get_incoming_links(
        client_id=client_id,
        url=url,
        limit=request.get("limit", 20),
    )

    # Get outgoing links
    outlinks = await neo4j.get_outgoing_links(
        client_id=client_id,
        url=url,
        limit=request.get("limit", 20),
    )

    return {
        "source_url": url,
        "related": related,
        "backlinks": backlinks,
        "outlinks": outlinks,
        "total_related": len(related),
        "total_backlinks": len(backlinks),
        "total_outlinks": len(outlinks),
    }


@dashboard_router.post("/path")
async def dashboard_find_path(
    request: dict,
    supabase: SupabaseClient = Depends(get_supabase_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """Find shortest path between two pages (no API key required)."""
    client_id = request.get("client_id")
    source_url = request.get("source_url")
    target_url = request.get("target_url")

    if not client_id:
        raise HTTPException(status_code=400, detail="client_id is required")
    if not source_url:
        raise HTTPException(status_code=400, detail="source_url is required")
    if not target_url:
        raise HTTPException(status_code=400, detail="target_url is required")

    # Verify client exists
    client = await supabase.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    # Find path
    path = await neo4j.get_shortest_path(
        client_id=client_id,
        source_url=source_url,
        target_url=target_url,
    )

    return {
        "source_url": source_url,
        "target_url": target_url,
        "path": path,
        "path_length": len(path) - 1 if path else None,
        "found": path is not None and len(path) > 0,
    }


@dashboard_router.post("/interlinking")
async def dashboard_interlinking_suggestions(
    request: dict,
    supabase: SupabaseClient = Depends(get_supabase_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """
    Get interlinking suggestions for a URL.
    Finds semantically similar pages that are NOT currently linked.
    """
    client_id = request.get("client_id")
    url = request.get("url")

    if not client_id:
        raise HTTPException(status_code=400, detail="client_id is required")
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    # Verify client exists
    client = await supabase.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    # Get current page content for embedding
    page = await supabase.get_page_by_url(client_id, url)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    # Get pages already linked (to exclude)
    outlinks = await neo4j.get_outgoing_links(client_id, url, limit=100)
    linked_urls = {link["url"] for link in outlinks}
    linked_urls.add(url)  # Exclude self

    # Generate embedding from page content
    content = page.get("content", "") or page.get("title", "")
    if not content:
        return {"suggestions": [], "total": 0, "message": "Page has no content"}

    query_embedding = embedding_service.embed_query(content[:2000])

    # Search for similar pages
    similar_pages = await supabase.search_similar_pages(
        client_id=client_id,
        query_embedding=query_embedding,
        limit=30,
        min_similarity=request.get("min_similarity", 0.5),
    )

    # Filter out already linked pages
    suggestions = [
        {
            "url": p["url"],
            "title": p.get("title"),
            "similarity": p.get("similarity", 0),
            "pagerank": p.get("pagerank", 0),
            "content_preview": p.get("content", "")[:200] + "..." if p.get("content") else None,
        }
        for p in similar_pages
        if p["url"] not in linked_urls
    ][:request.get("limit", 10)]

    return {
        "source_url": url,
        "source_title": page.get("title"),
        "suggestions": suggestions,
        "total": len(suggestions),
        "already_linked": len(linked_urls) - 1,  # Exclude self
    }


@dashboard_router.get("/pages")
async def dashboard_list_pages(
    client_id: str,
    limit: int = 50,
    offset: int = 0,
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    """List pages for a client (for URL autocomplete)."""
    if not client_id:
        raise HTTPException(status_code=400, detail="client_id is required")

    # Get pages from Supabase
    pages = await supabase.list_pages(client_id, limit=limit, offset=offset)

    return {
        "pages": [
            {
                "url": p["url"],
                "title": p.get("title"),
                "pagerank": p.get("pagerank", 0),
            }
            for p in pages
        ],
        "total": len(pages),
    }


# =============================================================================
# CRAWLER ENDPOINTS (Dashboard)
# =============================================================================

@dashboard_router.post("/crawler/start")
async def dashboard_start_crawl(request: dict):
    """
    Start a new crawl job.

    Options:
    - url: Required. The URL to crawl (used as base domain).
    - max_pages: Max pages to crawl (0 = unlimited, default 0).
    - delay: Delay between requests in seconds (default 0.5).
    - use_sitemap: Use sitemap for URL discovery (default true).
    - content_filter: Filter boilerplate content (default true).
    - resume: Resume from previous crawl, skip already crawled URLs (default false).
    - force_sitemap: With resume, force re-fetch sitemap to find new URLs (default false).
    - urls_list: List of specific URLs to crawl (optional). Can be array or newline-separated string.
    - urls_only: Only crawl URLs from urls_list, don't discover new links (default false).
    - exclude_selectors: Additional CSS selectors to exclude (cookies, popups, etc.). Can be array or comma-separated string.
    - respect_robots: Respect robots.txt rules (default true).
    - skip_noindex: Skip pages with noindex meta tag (default true).
    - sitemap_only: Only crawl URLs from sitemap, don't follow discovered links (default false).
    """
    from graph_rag.services.crawler_service import get_crawler_service
    from dataclasses import asdict

    url = request.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    # Handle urls_list - can be array or newline-separated string
    urls_list = request.get("urls_list", [])
    if isinstance(urls_list, str):
        # Parse newline-separated string
        urls_list = [u.strip() for u in urls_list.split("\n") if u.strip() and not u.strip().startswith("#")]

    # Handle exclude_selectors - can be array or comma-separated string
    exclude_selectors = request.get("exclude_selectors", [])
    if isinstance(exclude_selectors, str):
        # Parse comma-separated string
        exclude_selectors = [s.strip() for s in exclude_selectors.split(",") if s.strip()]

    crawler = get_crawler_service()

    try:
        job = await crawler.start_crawl(
            url=url,
            max_pages=request.get("max_pages", 0),  # 0 = unlimited
            delay=request.get("delay", 0.5),
            use_sitemap=request.get("use_sitemap", True),
            content_filter=request.get("content_filter", True),
            resume=request.get("resume", False),
            force_sitemap=request.get("force_sitemap", False),
            urls_list=urls_list,
            urls_only=request.get("urls_only", False),
            exclude_selectors=exclude_selectors if exclude_selectors else None,
            respect_robots=request.get("respect_robots", True),
            skip_noindex=request.get("skip_noindex", True),
            sitemap_only=request.get("sitemap_only", False),
        )
        return asdict(job)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@dashboard_router.post("/crawler/stop")
async def dashboard_stop_crawl():
    """Stop the current crawl job."""
    from graph_rag.services.crawler_service import get_crawler_service
    from dataclasses import asdict

    crawler = get_crawler_service()
    job = await crawler.stop_crawl()

    if job:
        return asdict(job)
    return {"status": "no_active_crawl"}


@dashboard_router.get("/crawler/status")
async def dashboard_crawler_status():
    """Get current crawler status."""
    from graph_rag.services.crawler_service import get_crawler_service
    from dataclasses import asdict

    crawler = get_crawler_service()
    job = crawler.get_status()

    if job:
        return asdict(job)
    return {"status": "idle"}


@dashboard_router.get("/crawler/logs")
async def dashboard_crawler_logs(last_n: int = 100):
    """
    Get crawler logs from the log file.

    Args:
        last_n: Number of last log entries to return (default 100)
    """
    from graph_rag.services.crawler_service import get_crawler_service
    from pathlib import Path
    import json

    crawler = get_crawler_service()
    job = crawler.get_status()

    if not job or not job.output_dir:
        return {"logs": [], "count": 0}

    # Log file is in the base output directory (parent of domain-specific folder)
    log_file = Path(job.output_dir).parent / ".crawl_log.jsonl"

    if not log_file.exists():
        return {"logs": [], "count": 0}

    try:
        with open(log_file, "r") as f:
            lines = f.readlines()

        # Get last N lines
        logs = []
        for line in lines[-last_n:]:
            try:
                logs.append(json.loads(line.strip()))
            except:
                pass

        return {"logs": logs, "count": len(lines)}
    except Exception as e:
        return {"logs": [], "count": 0, "error": str(e)}


@dashboard_router.get("/crawler/crawls")
async def dashboard_list_crawls():
    """List available crawl data directories."""
    from graph_rag.services.crawler_service import get_crawler_service

    crawler = get_crawler_service()
    return {"crawls": crawler.list_available_crawls()}


@dashboard_router.get("/crawler/crawls/{crawl_name}/pages")
async def dashboard_list_crawl_pages(
    crawl_name: str,
    limit: int = 100,
    offset: int = 0,
    search: str = None,
    min_words: int = 0,
):
    """
    List pages from a crawl's parquet files with filtering options.

    Args:
        crawl_name: Name of the crawl directory (e.g., 'www_ilerna_es')
        limit: Max pages to return (default 100)
        offset: Pagination offset
        search: Search filter for URL or title
        min_words: Minimum word count filter
    """
    import pandas as pd
    from pathlib import Path
    from glob import glob

    base_dir = Path("data/crawl4ai_data") / crawl_name
    if not base_dir.exists():
        raise HTTPException(status_code=404, detail=f"Crawl '{crawl_name}' not found")

    # Load all pages from parquet files
    pages_pattern = str(base_dir / "pages" / "**" / "*.parquet")
    pages_files = glob(pages_pattern, recursive=True)

    if not pages_files:
        return {"pages": [], "total": 0, "filtered": 0}

    all_pages = []
    for pf in pages_files:
        try:
            df = pd.read_parquet(pf)
            all_pages.append(df)
        except Exception as e:
            continue

    if not all_pages:
        return {"pages": [], "total": 0, "filtered": 0}

    df_all = pd.concat(all_pages, ignore_index=True)
    df_all = df_all.drop_duplicates(subset='url')
    total = len(df_all)

    # Apply filters
    if min_words > 0:
        df_all = df_all[df_all['word_count'] >= min_words]

    if search:
        search_lower = search.lower()
        df_all = df_all[
            df_all['url'].str.lower().str.contains(search_lower, na=False) |
            df_all['title'].str.lower().str.contains(search_lower, na=False)
        ]

    filtered = len(df_all)

    # Sort by word count (more content first) and paginate
    df_all = df_all.sort_values('word_count', ascending=False)
    df_page = df_all.iloc[offset:offset + limit]

    pages = []
    for _, row in df_page.iterrows():
        pages.append({
            "url": row.get("url", ""),
            "title": row.get("title", "")[:100] if row.get("title") else "",
            "word_count": int(row.get("word_count", 0)),
            "content_preview": (row.get("markdown", "")[:200] + "...") if row.get("markdown") else "",
            "crawled_at": row.get("crawled_at", ""),
        })

    return {
        "pages": pages,
        "total": total,
        "filtered": filtered,
        "limit": limit,
        "offset": offset,
    }


@dashboard_router.get("/crawler/crawls/{crawl_name}/page")
async def dashboard_get_crawl_page(crawl_name: str, url: str):
    """
    Get full content of a specific page from crawl data.
    Prioritizes cleaned content (pages_clean/) over raw markdown (pages/).
    Used for previewing scraping quality.
    """
    import pandas as pd
    from pathlib import Path
    from glob import glob

    base_dir = Path("data/crawl4ai_data") / crawl_name
    if not base_dir.exists():
        raise HTTPException(status_code=404, detail=f"Crawl '{crawl_name}' not found")

    # First, try to find in manual_clean/ (manual cleaner results)
    manual_clean_pattern = str(base_dir / "manual_clean" / "*.parquet")
    manual_clean_files = glob(manual_clean_pattern)

    for mcf in manual_clean_files:
        try:
            df = pd.read_parquet(mcf)
            match = df[df['url'] == url]
            if not match.empty:
                row = match.iloc[0]
                clean_text = row.get("markdown_clean", "")
                if clean_text and len(str(clean_text).strip()) > 0:
                    word_count = len(clean_text.split()) if clean_text else 0
                    return {
                        "url": row.get("url", ""),
                        "title": row.get("title", ""),
                        "markdown": clean_text,
                        "word_count": int(word_count),
                        "content_hash": "",
                        "crawled_at": "",
                        "links_count": 0,
                        "is_cleaned": True,
                        "template_group": row.get("template_id", ""),
                    }
        except Exception:
            continue

    # Second, try auto_clean/ (auto cleaner results)
    auto_clean_file = base_dir / "auto_clean" / "all_pages.parquet"
    if auto_clean_file.exists():
        try:
            df = pd.read_parquet(auto_clean_file)
            match = df[df['url'] == url]
            if not match.empty:
                row = match.iloc[0]
                clean_text = row.get("markdown_clean", "")
                if clean_text and len(str(clean_text).strip()) > 0:
                    word_count = len(clean_text.split()) if clean_text else 0
                    return {
                        "url": row.get("url", ""),
                        "title": row.get("title", ""),
                        "markdown": clean_text,
                        "word_count": int(word_count),
                        "content_hash": "",
                        "crawled_at": "",
                        "links_count": 0,
                        "is_cleaned": True,
                        "template_group": "auto_clean",
                    }
        except Exception:
            pass

    # Third, try pages_clean/ (smart cleaner results)
    clean_pattern = str(base_dir / "pages_clean" / "*.parquet")
    clean_files = sorted(glob(clean_pattern), reverse=True)

    for cf in clean_files:
        try:
            df = pd.read_parquet(cf)
            match = df[df['url'] == url]
            if not match.empty:
                row = match.iloc[0]
                clean_text = row.get("clean_text", "")
                if clean_text and len(str(clean_text).strip()) > 0:
                    word_count = row.get("clean_word_count", len(clean_text.split()) if clean_text else 0)
                    return {
                        "url": row.get("url", ""),
                        "title": row.get("title", ""),
                        "markdown": clean_text,
                        "word_count": int(word_count),
                        "content_hash": "",
                        "crawled_at": str(row.get("crawl_date", "")),
                        "links_count": 0,
                        "is_cleaned": True,
                        "template_group": row.get("template_group", ""),
                    }
        except Exception:
            continue

    # Fallback: Search in raw pages (pages/)
    pages_pattern = str(base_dir / "pages" / "**" / "*.parquet")
    pages_files = glob(pages_pattern, recursive=True)

    for pf in pages_files:
        try:
            df = pd.read_parquet(pf)
            match = df[df['url'] == url]
            if not match.empty:
                row = match.iloc[0]
                return {
                    "url": row.get("url", ""),
                    "title": row.get("title", ""),
                    "markdown": row.get("markdown", ""),
                    "word_count": int(row.get("word_count", 0)),
                    "content_hash": row.get("content_hash", ""),
                    "crawled_at": str(row.get("crawled_at", "")),
                    "links_count": int(row.get("links_count", 0)) if "links_count" in row else 0,
                    "is_cleaned": False,
                }
        except Exception:
            continue

    raise HTTPException(status_code=404, detail="Page not found in crawl data")


@dashboard_router.get("/crawler/crawls/{crawl_name}/links")
async def dashboard_get_crawl_links(
    crawl_name: str,
    limit: int = 100,
    source_url: str = None,
    location: str = None
):
    """
    Get links extracted during crawl with their locations and weights.
    Used for previewing link extraction quality.

    Args:
        crawl_name: Name of the crawl directory
        limit: Max links to return (default 100)
        source_url: Filter by source URL (optional)
        location: Filter by link location: 'content', 'nav', 'sidebar', 'footer' (optional)
    """
    import pandas as pd
    from pathlib import Path
    from glob import glob

    base_dir = Path("data/crawl4ai_data") / crawl_name
    if not base_dir.exists():
        raise HTTPException(status_code=404, detail=f"Crawl '{crawl_name}' not found")

    # Load all links from parquet files
    links_pattern = str(base_dir / "links" / "**" / "*.parquet")
    links_files = glob(links_pattern, recursive=True)

    if not links_files:
        return {"links": [], "total": 0, "by_location": {}}

    all_links = []
    for lf in links_files:
        try:
            df = pd.read_parquet(lf)
            all_links.append(df)
        except Exception:
            continue

    if not all_links:
        return {"links": [], "total": 0, "by_location": {}}

    df_all = pd.concat(all_links, ignore_index=True)

    # Filter by source URL if provided
    if source_url:
        df_all = df_all[df_all['source_url'] == source_url]

    # Count by location (before filtering by location)
    by_location = {}
    if 'link_location' in df_all.columns:
        by_location = df_all['link_location'].value_counts().to_dict()

    total = len(df_all)

    # Filter by location if provided
    if location and 'link_location' in df_all.columns:
        df_all = df_all[df_all['link_location'] == location]

    filtered_total = len(df_all)

    # Get links (no prioritization when filtering by location)
    df_sample = df_all.head(limit)

    links = []
    for _, row in df_sample.iterrows():
        links.append({
            "source_url": row.get("source_url", ""),
            "target_url": row.get("target_url", ""),
            "anchor_text": row.get("anchor_text", ""),
            "link_location": row.get("link_location", "content"),
            "link_weight": float(row.get("link_weight", 1.0)),
        })

    return {
        "links": links,
        "total": total,
        "filtered_total": filtered_total,
        "by_location": by_location,
        "location_filter": location,
    }


@dashboard_router.delete("/crawler/crawls/{crawl_name}/page")
async def dashboard_delete_crawl_page(crawl_name: str, url: str):
    """
    Delete a specific page from crawl parquet files.
    """
    import pandas as pd
    from pathlib import Path
    from glob import glob

    base_dir = Path("data/crawl4ai_data") / crawl_name
    if not base_dir.exists():
        raise HTTPException(status_code=404, detail=f"Crawl '{crawl_name}' not found")

    # Search and delete from parquet files
    pages_pattern = str(base_dir / "pages" / "**" / "*.parquet")
    pages_files = glob(pages_pattern, recursive=True)

    deleted = False
    for pf in pages_files:
        try:
            df = pd.read_parquet(pf)
            if url in df['url'].values:
                df_filtered = df[df['url'] != url]
                if len(df_filtered) > 0:
                    df_filtered.to_parquet(pf, index=False)
                else:
                    # If file is empty, delete it
                    Path(pf).unlink()
                deleted = True
                break
        except Exception as e:
            continue

    if not deleted:
        raise HTTPException(status_code=404, detail="Page not found in crawl data")

    # Also try to delete from links parquet
    links_pattern = str(base_dir / "links" / "**" / "*.parquet")
    links_files = glob(links_pattern, recursive=True)

    for lf in links_files:
        try:
            df = pd.read_parquet(lf)
            # Remove links where this URL is source
            df_filtered = df[df['source_url'] != url]
            if len(df_filtered) > 0:
                df_filtered.to_parquet(lf, index=False)
            elif len(df_filtered) == 0 and len(df) > 0:
                Path(lf).unlink()
        except Exception:
            continue

    return {"status": "deleted", "url": url}


@dashboard_router.delete("/crawler/crawls/{crawl_name}")
async def dashboard_delete_crawl(crawl_name: str):
    """
    Delete entire crawl data directory.
    """
    import shutil
    from pathlib import Path

    base_dir = Path("data/crawl4ai_data") / crawl_name
    if not base_dir.exists():
        raise HTTPException(status_code=404, detail=f"Crawl '{crawl_name}' not found")

    try:
        shutil.rmtree(base_dir)
        return {"status": "deleted", "crawl_name": crawl_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting crawl: {str(e)}")


@dashboard_router.post("/crawler/crawls/{crawl_name}/delete-pages")
async def dashboard_delete_multiple_pages(crawl_name: str, request: dict):
    """
    Delete multiple pages from crawl parquet files.

    Body: {"urls": ["url1", "url2", ...]}
    """
    import pandas as pd
    from pathlib import Path
    from glob import glob

    urls_to_delete = set(request.get("urls", []))
    if not urls_to_delete:
        raise HTTPException(status_code=400, detail="No URLs provided")

    base_dir = Path("data/crawl4ai_data") / crawl_name
    if not base_dir.exists():
        raise HTTPException(status_code=404, detail=f"Crawl '{crawl_name}' not found")

    # Delete from pages parquet files
    pages_pattern = str(base_dir / "pages" / "**" / "*.parquet")
    pages_files = glob(pages_pattern, recursive=True)

    deleted_count = 0
    for pf in pages_files:
        try:
            df = pd.read_parquet(pf)
            original_len = len(df)
            df_filtered = df[~df['url'].isin(urls_to_delete)]
            deleted_in_file = original_len - len(df_filtered)

            if deleted_in_file > 0:
                deleted_count += deleted_in_file
                if len(df_filtered) > 0:
                    df_filtered.to_parquet(pf, index=False)
                else:
                    Path(pf).unlink()
        except Exception:
            continue

    # Also delete from links
    links_pattern = str(base_dir / "links" / "**" / "*.parquet")
    links_files = glob(links_pattern, recursive=True)

    for lf in links_files:
        try:
            df = pd.read_parquet(lf)
            df_filtered = df[~df['source_url'].isin(urls_to_delete)]
            if len(df_filtered) < len(df):
                if len(df_filtered) > 0:
                    df_filtered.to_parquet(lf, index=False)
                else:
                    Path(lf).unlink()
        except Exception:
            continue

    return {"status": "deleted", "deleted_count": deleted_count}


# =============================================================================
# INGEST ENDPOINTS (Dashboard)
# =============================================================================

@dashboard_router.post("/ingest")
async def dashboard_ingest_data(
    request: dict,
    supabase: SupabaseClient = Depends(get_supabase_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """
    Ingest crawled data into the database.

    Options:
    - crawl_path: Required. Path to crawl data directory.
    - client_name: Required. Name for the client.
    - client_domain: Required. Domain for the client.
    - ingest_mode: How to handle existing data:
        - "new_only": Only insert pages that don't exist (default, safest)
        - "skip_existing": Same as new_only - skip if URL exists
        - "overwrite": Update existing pages with new data
        - "full_refresh": Delete all existing data and re-import
    - delete_after: Delete crawl data after ingestion (default false).
    - min_word_count: Minimum word count for pages (default 50).
    - embedding_model: Model name for embeddings (default: hiiamsid/sentence_similarity_spanish_es)
    - embedding_dimension: Embedding vector dimension (default: 768)
    """
    from graph_rag.services.ingest_service import IngestService
    from graph_rag.config.settings import get_settings, Settings

    crawl_path = request.get("crawl_path")
    client_name = request.get("client_name")
    client_domain = request.get("client_domain")
    ingest_mode = request.get("ingest_mode", "new_only")
    embedding_model = request.get("embedding_model", "hiiamsid/sentence_similarity_spanish_es")
    embedding_dimension = request.get("embedding_dimension", 768)

    if not crawl_path:
        raise HTTPException(status_code=400, detail="crawl_path is required")
    if not client_name:
        raise HTTPException(status_code=400, detail="client_name is required")
    if not client_domain:
        raise HTTPException(status_code=400, detail="client_domain is required")

    valid_modes = ["new_only", "skip_existing", "overwrite", "full_refresh"]
    if ingest_mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"ingest_mode must be one of: {', '.join(valid_modes)}"
        )

    # Get base settings and create custom embedding service with selected model
    settings = get_settings()

    # Create a custom settings object with the selected embedding model
    custom_settings = Settings(
        supabase_url=settings.supabase_url,
        supabase_db_password=settings.supabase_db_password,
        neo4j_password=settings.neo4j_password,
        openai_api_key=settings.openai_api_key,
        embedding_model_name=embedding_model,
        embedding_dimension=embedding_dimension,
    )

    # Create embedding service with the selected model
    embedding_service = EmbeddingService(custom_settings)

    ingest_service = IngestService(
        settings=settings,
        supabase_client=supabase,
        neo4j_client=neo4j,
        embedding_service=embedding_service,
    )

    try:
        result = await ingest_service.ingest_crawl_data(
            crawl_path=crawl_path,
            client_name=client_name,
            client_domain=client_domain,
            ingest_mode=ingest_mode,
            delete_after=request.get("delete_after", False),
            min_word_count=request.get("min_word_count", 50),
        )

        return {
            "client_id": result.client_id,
            "client_name": result.client_name,
            "pages_migrated": result.pages_migrated,
            "pages_skipped": result.pages_skipped,
            "pages_updated": result.pages_updated,
            "embeddings_generated": result.embeddings_generated,
            "chunks_created": result.chunks_created,
            "links_migrated": result.links_migrated,
            "ingest_mode": ingest_mode,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "errors": result.errors,
            "error_messages": result.error_messages,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SCORES CALCULATION ENDPOINTS
# =============================================================================

@dashboard_router.post("/clients/{client_id}/calculate-scores")
async def dashboard_calculate_scores(
    client_id: str,
    request: dict = None,
    supabase: SupabaseClient = Depends(get_supabase_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """
    Calculate PageRank and HITS scores for a client.
    This updates scores in both Neo4j and Supabase.
    """
    request = request or {}

    # Verify client exists
    client = await supabase.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    try:
        # Calculate scores in Neo4j
        stats = await neo4j.calculate_all_scores(
            client_id=client_id,
            pagerank_iterations=request.get("pagerank_iterations", 20),
            hits_iterations=request.get("hits_iterations", 20),
        )

        # Sync scores to Supabase
        top_pages = await neo4j.get_top_pages_by_pagerank(client_id, limit=10000)
        if top_pages:
            await supabase.update_scores_batch(client_id, top_pages)

        return {
            "client_id": client_id,
            "stats": stats,
            "synced_to_supabase": len(top_pages),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@dashboard_router.get("/clients/{client_id}/top-pages")
async def dashboard_top_pages(
    client_id: str,
    limit: int = 20,
    supabase: SupabaseClient = Depends(get_supabase_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """Get top pages by PageRank for a client."""
    # Verify client exists
    client = await supabase.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    pages = await neo4j.get_top_pages_by_pagerank(client_id, limit=limit)
    return {"pages": pages, "total": len(pages)}


# =============================================================================
# CLIENT MANAGEMENT ENDPOINTS
# =============================================================================

@dashboard_router.delete("/clients/{client_id}")
async def dashboard_delete_client(
    client_id: str,
    supabase: SupabaseClient = Depends(get_supabase_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """Delete a client and all their data."""
    # Verify client exists
    client = await supabase.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    try:
        # Delete from Neo4j first
        await neo4j.delete_client_data(client_id)

        # Then delete from Supabase
        deleted = await supabase.delete_client(client_id)

        return {
            "deleted": deleted,
            "client_id": client_id,
            "message": "Client and all data deleted successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@dashboard_router.post("/clients/{client_id}/deactivate")
async def dashboard_deactivate_client(
    client_id: str,
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    """Deactivate a client (soft delete)."""
    client = await supabase.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    deactivated = await supabase.deactivate_client(client_id)
    return {
        "deactivated": deactivated,
        "client_id": client_id,
    }


@dashboard_router.post("/clients/{client_id}/rotate-key")
async def dashboard_rotate_api_key(
    client_id: str,
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    """Generate a new API key for a client."""
    client = await supabase.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    new_key = await supabase.regenerate_api_key(client_id)
    if not new_key:
        raise HTTPException(status_code=500, detail="Failed to regenerate API key")

    return {
        "client_id": client_id,
        "new_api_key": new_key,
        "message": "API key regenerated. Save this key - it won't be shown again!",
    }


@dashboard_router.post("/clients/{client_id}/regenerate-embeddings")
async def dashboard_regenerate_embeddings(
    client_id: str,
    supabase: SupabaseClient = Depends(get_supabase_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """Regenerate all embeddings for a client's pages."""
    from graph_rag.services.ingest_service import IngestService
    from graph_rag.config.settings import get_settings

    # Verify client exists
    client = await supabase.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    settings = get_settings()
    ingest_service = IngestService(
        settings=settings,
        supabase_client=supabase,
        neo4j_client=neo4j,
        embedding_service=embedding_service,
    )

    try:
        result = await ingest_service.regenerate_embeddings(client_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# GRAPH EXPLORER ENDPOINTS
# =============================================================================

@dashboard_router.get("/graph/{client_id}")
async def dashboard_get_graph_data(
    client_id: str,
    limit: int = 100,
    supabase: SupabaseClient = Depends(get_supabase_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """
    Get graph data for visualization.
    Returns nodes and edges in a format suitable for vis.js or D3.js.
    """
    # Verify client exists
    client = await supabase.get_client_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    # Get top pages by PageRank as nodes
    async with neo4j.get_session() as session:
        # Get nodes
        nodes_result = await session.run(
            """
            MATCH (p:Page {client_id: $client_id})
            RETURN
                p.url AS id,
                p.title AS label,
                p.pagerank AS pagerank,
                p.hub_score AS hub,
                p.authority_score AS authority
            ORDER BY p.pagerank DESC
            LIMIT $limit
            """,
            client_id=client_id,
            limit=limit,
        )
        nodes_data = await nodes_result.data()

        # Get edges between these nodes (excluding self-loops)
        node_urls = [n["id"] for n in nodes_data]
        edges_result = await session.run(
            """
            MATCH (source:Page {client_id: $client_id})-[r:LINKS_TO]->(target:Page {client_id: $client_id})
            WHERE source.url IN $urls AND target.url IN $urls
              AND source.url <> target.url
            RETURN DISTINCT
                source.url AS from,
                target.url AS to,
                r.anchor_text AS label
            LIMIT 10000
            """,
            client_id=client_id,
            urls=node_urls,
        )
        edges_data = await edges_result.data()

    # Format for vis.js
    nodes = [
        {
            "id": n["id"],
            "label": (n["label"] or n["id"].split("/")[-1])[:30],
            "title": n["label"] or n["id"],
            "value": max(1, int((n["pagerank"] or 0) * 100)),
            "pagerank": n["pagerank"] or 0,
            "hub": n["hub"] or 0,
            "authority": n["authority"] or 0,
        }
        for n in nodes_data
    ]

    edges = [
        {
            "from": e["from"],
            "to": e["to"],
            "title": e["label"] if e["label"] else None,
            "arrows": "to",
        }
        for e in edges_data
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }


# =============================================================================
# CLEANING ENDPOINTS (Dashboard)
# =============================================================================

@dashboard_router.get("/cleaner/crawls")
async def dashboard_cleaner_list_crawls():
    """List crawls available for cleaning."""
    from graph_rag.services.cleaning_service import get_cleaning_service

    cleaning = get_cleaning_service()
    return {"crawls": cleaning.list_available_crawls()}


@dashboard_router.get("/pipeline/status")
async def dashboard_pipeline_status(
    supabase: SupabaseClient = Depends(get_supabase_client),
):
    """
    Get complete pipeline status for all crawls.
    Shows: crawl status, cleaning status, ingestion status.
    """
    from graph_rag.services.cleaning_service import get_cleaning_service
    from pathlib import Path
    from glob import glob
    import pandas as pd

    cleaning = get_cleaning_service()
    crawls_data = cleaning.list_available_crawls()

    # Get ingested clients/domains
    ingested_domains = set()
    try:
        clients = await supabase.list_clients()
        for client in clients:
            if client.get("domain"):
                # Normalize domain for comparison: www.ilerna.es -> www_ilerna_es
                domain_normalized = client["domain"].replace(".", "_")
                ingested_domains.add(domain_normalized)
                # Also add without www prefix
                if domain_normalized.startswith("www_"):
                    ingested_domains.add(domain_normalized[4:])
                else:
                    ingested_domains.add("www_" + domain_normalized)
    except Exception as e:
        print(f"Error fetching clients: {e}")

    # Enrich crawl data with pipeline status
    pipeline_data = []
    for crawl in crawls_data:
        name = crawl["name"]

        # Check if ingested (domain matches a client)
        is_ingested = name in ingested_domains or name.replace("www_", "") in ingested_domains

        pipeline_data.append({
            "name": name,
            "domain": name.replace("_", "."),
            "path": crawl["path"],
            "pages": crawl["pages"],
            "has_html_content": crawl["has_html_content"],
            "is_cleaned": crawl["is_cleaned"],
            "is_ingested": is_ingested,
            "created": crawl["created"],
        })

    return {"crawls": pipeline_data}


@dashboard_router.post("/cleaner/start")
async def dashboard_cleaner_start(request: dict):
    """
    Start a cleaning job for a crawl.

    Options:
    - crawl_name: Required. Name of the crawl directory.
    - use_ai: Use AI to discover CSS selectors (default true).
    """
    from graph_rag.services.cleaning_service import get_cleaning_service
    from dataclasses import asdict

    crawl_name = request.get("crawl_name")
    if not crawl_name:
        raise HTTPException(status_code=400, detail="crawl_name is required")

    use_ai = request.get("use_ai", True)

    cleaning = get_cleaning_service()

    try:
        job = await cleaning.start_cleaning(
            crawl_name=crawl_name,
            use_ai=use_ai,
        )
        return asdict(job)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@dashboard_router.get("/cleaner/status")
async def dashboard_cleaner_status():
    """Get current cleaning job status."""
    from graph_rag.services.cleaning_service import get_cleaning_service
    from dataclasses import asdict

    cleaning = get_cleaning_service()
    job = cleaning.get_status()

    if job:
        return asdict(job)
    return {"status": "idle"}


@dashboard_router.get("/cleaner/rules/{crawl_name}")
async def dashboard_cleaner_get_rules(crawl_name: str):
    """Get cleaning rules discovered for a crawl."""
    from graph_rag.services.cleaning_service import get_cleaning_service

    cleaning = get_cleaning_service()
    rules = cleaning.get_cleaning_rules(crawl_name)

    if rules is None:
        raise HTTPException(status_code=404, detail="No cleaning rules found for this crawl")

    return {"crawl_name": crawl_name, "rules": rules}


@dashboard_router.get("/cleaner/preview/{crawl_name}")
async def dashboard_cleaner_preview(
    crawl_name: str,
    url: str,
    selector: str = None,
):
    """
    Preview cleaning result for a specific page using ScrapeGraphAI.
    Shows both AI-cleaned and simple-cleaned versions for comparison.
    """
    from graph_rag.services.cleaning_service import preview_cleaning
    import pandas as pd
    from pathlib import Path
    from glob import glob

    base_dir = Path("data/crawl4ai_data") / crawl_name
    if not base_dir.exists():
        raise HTTPException(status_code=404, detail=f"Crawl '{crawl_name}' not found")

    # Find the page
    pages_pattern = str(base_dir / "pages" / "**" / "*.parquet")
    pages_files = glob(pages_pattern, recursive=True)

    for pf in pages_files:
        try:
            df = pd.read_parquet(pf)
            match = df[df["url"] == url]
            if not match.empty:
                row = match.iloc[0]
                html = row.get("html_content", "")

                if not html:
                    raise HTTPException(status_code=400, detail="Page has no html_content")

                # Preview cleaning with both methods
                result = preview_cleaning(html, url)
                result["original_title"] = row.get("title", "")

                return result
        except HTTPException:
            raise
        except Exception:
            continue

    raise HTTPException(status_code=404, detail="Page not found in crawl data")


# =============================================================================
# MARKDOWN CLEANER ENDPOINTS (LLM-based pattern cleaning)
# =============================================================================

@dashboard_router.get("/markdown-cleaner/crawls")
async def dashboard_markdown_cleaner_list_crawls():
    """List crawls available for markdown cleaning."""
    from graph_rag.services.markdown_cleaner_service import get_markdown_cleaner_service

    cleaner = get_markdown_cleaner_service()
    return {"crawls": cleaner.list_crawls()}


@dashboard_router.post("/markdown-cleaner/start")
async def dashboard_markdown_cleaner_start(request: dict):
    """
    Start a markdown cleaning job using LLM-generated patterns.

    This process:
    1. Clusters pages by URL template
    2. For each template, generates cleaning patterns via LLM
    3. Applies patterns to all pages in template (no additional LLM calls)

    Options:
    - crawl_name: Required. Name of the crawl directory.
    """
    from graph_rag.services.markdown_cleaner_service import get_markdown_cleaner_service

    crawl_name = request.get("crawl_name")
    if not crawl_name:
        raise HTTPException(status_code=400, detail="crawl_name is required")

    cleaner = get_markdown_cleaner_service()

    try:
        job = await cleaner.start_cleaning(crawl_name=crawl_name)
        return job
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@dashboard_router.get("/markdown-cleaner/status")
async def dashboard_markdown_cleaner_status():
    """Get current markdown cleaning job status."""
    from graph_rag.services.markdown_cleaner_service import get_markdown_cleaner_service

    cleaner = get_markdown_cleaner_service()
    status = cleaner.get_status()

    if status:
        return status
    return {"status": "idle"}


@dashboard_router.get("/markdown-cleaner/patterns/{crawl_name}")
async def dashboard_markdown_cleaner_get_patterns(crawl_name: str):
    """Get the LLM-generated patterns for a crawl."""
    from pathlib import Path
    import json

    patterns_file = Path("data/crawl4ai_data") / crawl_name / "cleaning_patterns.json"
    if not patterns_file.exists():
        raise HTTPException(status_code=404, detail="No patterns found for this crawl")

    with open(patterns_file) as f:
        patterns = json.load(f)

    return {"crawl_name": crawl_name, "patterns": patterns}


@dashboard_router.get("/markdown-cleaner/preview/{crawl_name}")
async def dashboard_markdown_cleaner_preview(crawl_name: str, url: str):
    """
    Preview markdown cleaning result for a specific page.
    Shows original markdown, cleaned markdown, and patterns applied.
    """
    from graph_rag.services.markdown_cleaner_service import get_markdown_cleaner_service
    import pandas as pd
    from pathlib import Path
    from glob import glob
    import json

    base_dir = Path("data/crawl4ai_data") / crawl_name
    if not base_dir.exists():
        raise HTTPException(status_code=404, detail=f"Crawl '{crawl_name}' not found")

    cleaner = get_markdown_cleaner_service()

    # Load patterns if available
    patterns_file = base_dir / "cleaning_patterns.json"
    all_patterns = {}
    if patterns_file.exists():
        with open(patterns_file) as f:
            data = json.load(f)
            all_patterns = data.get("templates", {})

    # Find the page
    pages_pattern = str(base_dir / "pages" / "**" / "*.parquet")
    pages_files = glob(pages_pattern, recursive=True)

    for pf in pages_files:
        try:
            df = pd.read_parquet(pf)
            match = df[df["url"] == url]
            if not match.empty:
                row = match.iloc[0]
                markdown = row.get("markdown", "")

                if not markdown:
                    raise HTTPException(status_code=400, detail="Page has no markdown content")

                # Get template for this URL
                template = cleaner._get_url_template(url)
                patterns = all_patterns.get(template, [])

                # Apply patterns
                clean_markdown = cleaner.apply_patterns(markdown, patterns)

                return {
                    "url": url,
                    "title": row.get("title", ""),
                    "template": template,
                    "patterns_count": len(patterns),
                    "patterns": patterns,
                    "original_markdown": markdown,
                    "original_length": len(markdown),
                    "clean_markdown": clean_markdown,
                    "clean_length": len(clean_markdown),
                    "reduction_pct": round((1 - len(clean_markdown) / len(markdown)) * 100, 1) if markdown else 0,
                }
        except HTTPException:
            raise
        except Exception as e:
            continue

    raise HTTPException(status_code=404, detail="Page not found in crawl data")


# =============================================================================
# MANUAL CLEANER ENDPOINTS (Human-controlled cleaning)
# =============================================================================

@dashboard_router.get("/manual-cleaner/analyze/{crawl_name}")
async def manual_cleaner_analyze(crawl_name: str):
    """
    Analyze a crawl to detect templates using HTML fingerprinting.
    Returns list of templates with page counts and sample URLs.
    """
    from graph_rag.services.manual_cleaner_service import get_manual_cleaner_service

    cleaner = get_manual_cleaner_service()
    try:
        return cleaner.analyze_templates(crawl_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@dashboard_router.get("/manual-cleaner/template/{crawl_name}/{template_id}")
async def manual_cleaner_get_template(crawl_name: str, template_id: str, sample_index: int = 0):
    """
    Get a sample page from a template for manual inspection.
    Returns original markdown, current patterns, and preview.
    """
    from graph_rag.services.manual_cleaner_service import get_manual_cleaner_service

    cleaner = get_manual_cleaner_service()
    try:
        return cleaner.get_template_sample(crawl_name, template_id, sample_index)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@dashboard_router.post("/manual-cleaner/pattern/{crawl_name}/{template_id}")
async def manual_cleaner_add_pattern(crawl_name: str, template_id: str, request: dict):
    """
    Add a cleaning pattern to a template.

    Request body:
    - pattern_type: "exact", "prefix", "contains", "regex", or "line_range"
    - value: The pattern value (for line_range: "start_text|||end_text")
    - description: Optional description
    """
    from graph_rag.services.manual_cleaner_service import get_manual_cleaner_service

    pattern_type = request.get("pattern_type")
    value = request.get("value")
    description = request.get("description", "")

    if not pattern_type or not value:
        raise HTTPException(status_code=400, detail="pattern_type and value are required")

    cleaner = get_manual_cleaner_service()
    try:
        return cleaner.add_pattern(crawl_name, template_id, pattern_type, value, description)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@dashboard_router.delete("/manual-cleaner/pattern/{crawl_name}/{template_id}/{pattern_id}")
async def manual_cleaner_remove_pattern(crawl_name: str, template_id: str, pattern_id: str):
    """Remove a pattern from a template."""
    from graph_rag.services.manual_cleaner_service import get_manual_cleaner_service

    cleaner = get_manual_cleaner_service()
    try:
        return cleaner.remove_pattern(crawl_name, template_id, pattern_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@dashboard_router.get("/manual-cleaner/available-patterns/{crawl_name}/{template_id}")
async def manual_cleaner_available_patterns(crawl_name: str, template_id: str):
    """Get patterns from other templates that can be reused."""
    from graph_rag.services.manual_cleaner_service import get_manual_cleaner_service

    cleaner = get_manual_cleaner_service()
    try:
        return cleaner.get_available_patterns(crawl_name, template_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@dashboard_router.get("/manual-cleaner/preview/{crawl_name}/{template_id}")
async def manual_cleaner_preview(crawl_name: str, template_id: str):
    """
    Preview cleaning results for all pages in a template.
    Shows stats without saving.
    """
    from graph_rag.services.manual_cleaner_service import get_manual_cleaner_service

    cleaner = get_manual_cleaner_service()
    try:
        return cleaner.preview_cleaning(crawl_name, template_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@dashboard_router.post("/manual-cleaner/apply/{crawl_name}/{template_id}")
async def manual_cleaner_apply(crawl_name: str, template_id: str, request: dict = None):
    """
    Apply cleaning patterns to all pages in a template and save.

    Optional body (for auto-cleaning):
    - auto_clean_options: Dict with toggleable options:
        - extract_from_first_heading: bool (default True) - Extraer desde primer H1/H2
        - remove_footer_content: bool (default True) - Eliminar contenido de footer
        - remove_empty_lines: bool (default True) - Limpiar líneas vacías excesivas
        - remove_nav_patterns: bool (default True) - Eliminar patrones de navegación
        - min_heading_level: int (default 1) - Nivel mínimo de heading
    """
    from graph_rag.services.manual_cleaner_service import get_manual_cleaner_service

    cleaner = get_manual_cleaner_service()
    auto_clean_options = None
    if request:
        auto_clean_options = request.get("auto_clean_options")

    try:
        return cleaner.apply_cleaning(crawl_name, template_id, auto_clean_options)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@dashboard_router.get("/manual-cleaner/status/{crawl_name}")
async def manual_cleaner_status(crawl_name: str):
    """Get overall cleaning status for a crawl."""
    from graph_rag.services.manual_cleaner_service import get_manual_cleaner_service

    cleaner = get_manual_cleaner_service()
    try:
        return cleaner.get_cleaning_status(crawl_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# =============================================================================
# AUTO-CLEAN ENDPOINTS (Quick cleaning without manual patterns)
# =============================================================================

@dashboard_router.post("/manual-cleaner/auto-clean/{crawl_name}")
async def manual_cleaner_auto_clean_all(crawl_name: str, request: dict = None):
    """
    Apply auto-cleaning to ALL pages in a crawl without manual patterns.
    This is a quick way to clean an entire crawl with sensible defaults.

    Optional body:
    - extract_from_first_heading: bool (default True) - Extraer desde primer H1/H2
    - remove_footer_content: bool (default True) - Eliminar contenido de footer
    - remove_empty_lines: bool (default True) - Limpiar líneas vacías excesivas
    - remove_nav_patterns: bool (default True) - Eliminar patrones de navegación
    - min_heading_level: int (default 1) - Nivel mínimo de heading (1=H1, 2=H2)
    """
    from graph_rag.services.manual_cleaner_service import get_manual_cleaner_service

    cleaner = get_manual_cleaner_service()
    auto_clean_options = request if request else None

    try:
        return cleaner.auto_clean_all(crawl_name, auto_clean_options)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@dashboard_router.get("/manual-cleaner/auto-clean/preview/{crawl_name}")
async def manual_cleaner_preview_auto_clean(
    crawl_name: str,
    sample_url: str = None,
    extract_from_first_heading: bool = True,
    remove_footer_content: bool = True,
    remove_empty_lines: bool = True,
    remove_nav_patterns: bool = True,
    min_heading_level: int = 1,
):
    """
    Preview auto-cleaning on a sample page before applying to all.

    Query params:
    - sample_url: Optional URL to preview (uses first page if not provided)
    - extract_from_first_heading: bool (default True)
    - remove_footer_content: bool (default True)
    - remove_empty_lines: bool (default True)
    - remove_nav_patterns: bool (default True)
    - min_heading_level: int (default 1)

    Returns original and cleaned content for comparison.
    """
    from graph_rag.services.manual_cleaner_service import get_manual_cleaner_service

    cleaner = get_manual_cleaner_service()
    auto_clean_options = {
        "extract_from_first_heading": extract_from_first_heading,
        "remove_footer_content": remove_footer_content,
        "remove_empty_lines": remove_empty_lines,
        "remove_nav_patterns": remove_nav_patterns,
        "min_heading_level": min_heading_level,
    }

    try:
        return cleaner.preview_auto_clean(crawl_name, sample_url, auto_clean_options)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@dashboard_router.get("/manual-cleaner/auto-clean/options")
async def manual_cleaner_auto_clean_options():
    """
    Get available auto-clean options with descriptions.
    Useful for building UI forms.
    """
    return {
        "options": [
            {
                "name": "extract_from_first_heading",
                "type": "bool",
                "default": True,
                "description": "Extraer contenido desde el primer heading (H1, H2...). Elimina todo el contenido anterior al primer título.",
            },
            {
                "name": "remove_footer_content",
                "type": "bool",
                "default": True,
                "description": "Eliminar contenido de footer. Detecta patrones como '©', 'política de privacidad', 'síguenos', etc.",
            },
            {
                "name": "remove_empty_lines",
                "type": "bool",
                "default": True,
                "description": "Limpiar líneas vacías excesivas. Máximo 2 líneas vacías seguidas.",
            },
            {
                "name": "remove_nav_patterns",
                "type": "bool",
                "default": True,
                "description": "Eliminar patrones de navegación. Breadcrumbs, menús, 'Ir al contenido', etc.",
            },
            {
                "name": "min_heading_level",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 6,
                "description": "Nivel mínimo de heading para extract_from_first_heading. 1=H1, 2=H2, etc.",
            },
        ],
        "nav_patterns": [
            "Inicio >", "Home >", "Breadcrumb", "Ir al contenido",
            "Skip to content", "Menú principal", "Main menu", "Buscar", "Search",
        ],
        "footer_patterns": [
            "## navegación", "## footer", "© ", "política de privacidad",
            "aviso legal", "términos y condiciones", "síguenos", "redes sociales",
        ],
    }


@dashboard_router.get("/cleaner/cleaned/{crawl_name}/pages")
async def dashboard_cleaner_list_cleaned_pages(
    crawl_name: str,
    limit: int = 100,
    offset: int = 0,
    search: str = None,
    min_words: int = 0,
):
    """List cleaned pages from a crawl."""
    import pandas as pd
    from pathlib import Path
    from glob import glob

    base_dir = Path("data/crawl4ai_data") / crawl_name / "pages_clean"
    if not base_dir.exists():
        raise HTTPException(status_code=404, detail=f"No cleaned data found for '{crawl_name}'")

    # Load cleaned pages
    pages_pattern = str(base_dir / "*.parquet")
    pages_files = glob(pages_pattern)

    if not pages_files:
        return {"pages": [], "total": 0, "filtered": 0}

    all_pages = []
    for pf in pages_files:
        try:
            df = pd.read_parquet(pf)
            all_pages.append(df)
        except:
            continue

    if not all_pages:
        return {"pages": [], "total": 0, "filtered": 0}

    df_all = pd.concat(all_pages, ignore_index=True)
    df_all = df_all.drop_duplicates(subset="url")
    total = len(df_all)

    # Apply filters
    if min_words > 0:
        df_all = df_all[df_all["clean_word_count"] >= min_words]

    if search:
        search_lower = search.lower()
        df_all = df_all[
            df_all["url"].str.lower().str.contains(search_lower, na=False) |
            df_all["title"].str.lower().str.contains(search_lower, na=False)
        ]

    filtered = len(df_all)

    # Sort and paginate
    df_all = df_all.sort_values("clean_word_count", ascending=False)
    df_page = df_all.iloc[offset:offset + limit]

    pages = []
    for _, row in df_page.iterrows():
        pages.append({
            "url": row.get("url", ""),
            "title": row.get("title", "")[:100] if row.get("title") else "",
            "clean_word_count": int(row.get("clean_word_count", 0)),
            "template_group": row.get("template_group", ""),
            "clean_preview": (row.get("clean_text", "")[:200] + "...") if row.get("clean_text") else "",
        })

    return {
        "pages": pages,
        "total": total,
        "filtered": filtered,
        "limit": limit,
        "offset": offset,
    }


# =============================================================================
# COMMUNITY DETECTION ENDPOINTS
# =============================================================================

@dashboard_router.post("/communities/detect/{client_id}")
async def dashboard_detect_communities(
    client_id: str,
    resolution: float = 1.0,
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """
    Detect communities in the page graph using Louvain algorithm.
    
    Args:
        client_id: Client ID to analyze.
        resolution: Resolution parameter (higher = more communities).
    """
    community_service = CommunityService(neo4j)
    result = await community_service.detect_communities(client_id, resolution)
    return result


@dashboard_router.get("/communities/{client_id}")
async def dashboard_get_communities(
    client_id: str,
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """Get community information for a client."""
    async with neo4j.get_session() as session:
        # Get community distribution
        result = await session.run(
            """
            MATCH (p:Page)
            WHERE p.client_id = $client_id AND p.community_id IS NOT NULL
            RETURN p.community_id AS community_id, COUNT(*) AS size,
                   COLLECT(p.title)[0..3] AS sample_titles
            ORDER BY size DESC
            """,
            client_id=client_id,
        )
        records = await result.data()
        
        communities = []
        for r in records:
            communities.append({
                "id": r["community_id"],
                "size": r["size"],
                "sample_titles": r["sample_titles"],
            })
        
        return {"communities": communities, "total": len(communities)}
