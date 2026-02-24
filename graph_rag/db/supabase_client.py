"""
Supabase client with pgvector support for Graph-RAG.
Handles vector similarity search and page/client management.
"""

import asyncpg
from typing import Optional
from contextlib import asynccontextmanager

from graph_rag.config.settings import Settings


class SupabaseClient:
    """Async Supabase/PostgreSQL client with pgvector support."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Initialize connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=self.settings.supabase_db_host,
                port=self.settings.supabase_db_port,
                database=self.settings.supabase_db_name,
                user=self.settings.supabase_db_user,
                password=self.settings.supabase_db_password,
                min_size=2,
                max_size=10,
            )

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        if self._pool is None:
            await self.connect()
        async with self._pool.acquire() as conn:
            yield conn

    # =========================================================================
    # CLIENT OPERATIONS
    # =========================================================================

    async def create_client(self, name: str, domain: str) -> dict:
        """Create a new client and return client info with API key."""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO rag_clients (name, domain)
                VALUES ($1, $2)
                RETURNING id, name, domain, api_key, created_at
                """,
                name,
                domain,
            )
            return dict(row)

    async def get_client_by_api_key(self, api_key: str) -> Optional[dict]:
        """Get client by API key for authentication."""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, name, domain, is_active
                FROM rag_clients
                WHERE api_key = $1 AND is_active = true
                """,
                api_key,
            )
            return dict(row) if row else None

    async def get_client_by_id(self, client_id: str) -> Optional[dict]:
        """Get client by ID."""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, name, domain, api_key, is_active, created_at
                FROM rag_clients
                WHERE id = $1
                """,
                client_id,
            )
            return dict(row) if row else None

    async def list_clients(self) -> list[dict]:
        """List all clients."""
        async with self.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, domain, is_active, created_at
                FROM rag_clients
                ORDER BY created_at DESC
                """
            )
            return [dict(row) for row in rows]

    # =========================================================================
    # PAGE OPERATIONS
    # =========================================================================

    async def upsert_page(
        self,
        client_id: str,
        url: str,
        title: Optional[str] = None,
        meta_description: Optional[str] = None,
        content: Optional[str] = None,
        content_hash: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        pagerank: float = 0.0,
        hub_score: float = 0.0,
        authority_score: float = 0.0,
        folder_depth: int = 0,
    ) -> dict:
        """Insert or update a page."""
        async with self.get_connection() as conn:
            # Convert embedding list to pgvector format
            embedding_str = None
            if embedding:
                embedding_str = f"[{','.join(map(str, embedding))}]"

            row = await conn.fetchrow(
                """
                INSERT INTO rag_pages (
                    client_id, url, title, meta_description, content, content_hash, embedding,
                    pagerank, hub_score, authority_score, folder_depth, last_crawled_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8, $9, $10, $11, NOW())
                ON CONFLICT (client_id, url) DO UPDATE SET
                    title = COALESCE(EXCLUDED.title, rag_pages.title),
                    meta_description = COALESCE(EXCLUDED.meta_description, rag_pages.meta_description),
                    content = COALESCE(EXCLUDED.content, rag_pages.content),
                    content_hash = COALESCE(EXCLUDED.content_hash, rag_pages.content_hash),
                    embedding = COALESCE(EXCLUDED.embedding, rag_pages.embedding),
                    pagerank = EXCLUDED.pagerank,
                    hub_score = EXCLUDED.hub_score,
                    authority_score = EXCLUDED.authority_score,
                    folder_depth = EXCLUDED.folder_depth,
                    last_crawled_at = NOW(),
                    updated_at = NOW()
                RETURNING id, url, title, meta_description, pagerank
                """,
                client_id,
                url,
                title,
                meta_description,
                content,
                content_hash,
                embedding_str,
                pagerank,
                hub_score,
                authority_score,
                folder_depth,
            )
            return dict(row)

    async def upsert_pages_batch(self, client_id: str, pages: list[dict]) -> int:
        """Batch upsert pages for efficiency."""
        async with self.get_connection() as conn:
            # Prepare data
            records = []
            for page in pages:
                embedding_str = None
                if page.get("embedding"):
                    embedding_str = f"[{','.join(map(str, page['embedding']))}]"
                records.append((
                    client_id,
                    page["url"],
                    page.get("title"),
                    page.get("meta_description"),
                    page.get("content"),
                    page.get("content_hash"),
                    embedding_str,
                    page.get("pagerank", 0.0),
                    page.get("hub_score", 0.0),
                    page.get("authority_score", 0.0),
                    page.get("folder_depth", 0),
                ))

            # Use copy for bulk insert (faster)
            await conn.executemany(
                """
                INSERT INTO rag_pages (
                    client_id, url, title, meta_description, content, content_hash, embedding,
                    pagerank, hub_score, authority_score, folder_depth, last_crawled_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8, $9, $10, $11, NOW())
                ON CONFLICT (client_id, url) DO UPDATE SET
                    title = COALESCE(EXCLUDED.title, rag_pages.title),
                    meta_description = COALESCE(EXCLUDED.meta_description, rag_pages.meta_description),
                    content = COALESCE(EXCLUDED.content, rag_pages.content),
                    content_hash = COALESCE(EXCLUDED.content_hash, rag_pages.content_hash),
                    embedding = COALESCE(EXCLUDED.embedding, rag_pages.embedding),
                    pagerank = EXCLUDED.pagerank,
                    hub_score = EXCLUDED.hub_score,
                    authority_score = EXCLUDED.authority_score,
                    folder_depth = EXCLUDED.folder_depth,
                    last_crawled_at = NOW(),
                    updated_at = NOW()
                """,
                records,
            )
            return len(records)

    async def get_page_by_url(self, client_id: str, url: str) -> Optional[dict]:
        """Get a page by URL."""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, url, title, content, pagerank, hub_score,
                       authority_score, folder_depth, last_crawled_at
                FROM rag_pages
                WHERE client_id = $1 AND url = $2
                """,
                client_id,
                url,
            )
            return dict(row) if row else None

    async def get_pages_count(self, client_id: str) -> int:
        """Get total page count for a client."""
        async with self.get_connection() as conn:
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM rag_pages WHERE client_id = $1",
                client_id,
            )
            return result

    async def list_pages(
        self,
        client_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """List pages for a client with pagination."""
        async with self.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT url, title, pagerank, hub_score, authority_score, folder_depth
                FROM rag_pages
                WHERE client_id = $1
                ORDER BY pagerank DESC, title ASC
                LIMIT $2 OFFSET $3
                """,
                client_id,
                limit,
                offset,
            )
            return [dict(row) for row in rows]

    # =========================================================================
    # VECTOR SEARCH
    # =========================================================================

    async def search_similar_pages(
        self,
        client_id: str,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.0,
        min_pagerank: float = 0.0,
    ) -> list[dict]:
        """Search pages by vector similarity using pgvector."""
        async with self.get_connection() as conn:
            embedding_str = f"[{','.join(map(str, query_embedding))}]"

            rows = await conn.fetch(
                """
                SELECT
                    id,
                    url,
                    title,
                    content,
                    pagerank,
                    hub_score,
                    authority_score,
                    folder_depth,
                    1 - (embedding <=> $2::vector) AS similarity
                FROM rag_pages
                WHERE client_id = $1
                  AND embedding IS NOT NULL
                  AND pagerank >= $4
                ORDER BY embedding <=> $2::vector
                LIMIT $3
                """,
                client_id,
                embedding_str,
                limit,
                min_pagerank,
            )

            results = []
            for row in rows:
                row_dict = dict(row)
                if row_dict["similarity"] >= min_similarity:
                    results.append(row_dict)

            return results

    # =========================================================================
    # LINK OPERATIONS
    # =========================================================================

    async def upsert_link(
        self,
        client_id: str,
        source_page_id: str,
        target_page_id: str,
        anchor_text: Optional[str] = None,
    ) -> None:
        """Insert or update a link."""
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO rag_links (client_id, source_page_id, target_page_id, anchor_text)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (client_id, source_page_id, target_page_id) DO UPDATE SET
                    anchor_text = COALESCE(EXCLUDED.anchor_text, rag_links.anchor_text)
                """,
                client_id,
                source_page_id,
                target_page_id,
                anchor_text,
            )

    async def upsert_links_batch(self, client_id: str, links: list[dict]) -> int:
        """Batch upsert links."""
        async with self.get_connection() as conn:
            records = [
                (client_id, link["source_page_id"], link["target_page_id"], link.get("anchor_text"))
                for link in links
            ]
            await conn.executemany(
                """
                INSERT INTO rag_links (client_id, source_page_id, target_page_id, anchor_text)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (client_id, source_page_id, target_page_id) DO NOTHING
                """,
                records,
            )
            return len(records)

    async def get_page_id_by_url(self, client_id: str, url: str) -> Optional[str]:
        """Get page ID by URL (used for link migration)."""
        async with self.get_connection() as conn:
            result = await conn.fetchval(
                "SELECT id FROM rag_pages WHERE client_id = $1 AND url = $2",
                client_id,
                url,
            )
            return str(result) if result else None

    async def get_url_to_page_id_mapping(self, client_id: str) -> dict[str, str]:
        """Get mapping of URL to page ID for a client."""
        async with self.get_connection() as conn:
            rows = await conn.fetch(
                "SELECT url, id FROM rag_pages WHERE client_id = $1",
                client_id,
            )
            return {row["url"]: str(row["id"]) for row in rows}

    # =========================================================================
    # CONVERSATION OPERATIONS
    # =========================================================================

    async def create_conversation(self, client_id: str, session_id: str) -> str:
        """Create a new conversation and return its ID."""
        async with self.get_connection() as conn:
            result = await conn.fetchval(
                """
                INSERT INTO rag_conversations (client_id, session_id)
                VALUES ($1, $2)
                RETURNING id
                """,
                client_id,
                session_id,
            )
            return str(result)

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        context_pages: Optional[list[str]] = None,
        tokens_used: Optional[int] = None,
    ) -> str:
        """Add a message to a conversation."""
        async with self.get_connection() as conn:
            result = await conn.fetchval(
                """
                INSERT INTO rag_messages (conversation_id, role, content, context_pages, tokens_used)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
                """,
                conversation_id,
                role,
                content,
                context_pages,
                tokens_used,
            )
            return str(result)

    async def get_conversation_messages(
        self, conversation_id: str, limit: int = 50
    ) -> list[dict]:
        """Get messages from a conversation."""
        async with self.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT id, role, content, context_pages, tokens_used, created_at
                FROM rag_messages
                WHERE conversation_id = $1
                ORDER BY created_at ASC
                LIMIT $2
                """,
                conversation_id,
                limit,
            )
            return [dict(row) for row in rows]

    async def get_client_conversations(
        self, client_id: str, limit: int = 20
    ) -> list[dict]:
        """Get recent conversations for a client."""
        async with self.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.id,
                    c.session_id,
                    c.created_at,
                    COUNT(m.id) as message_count,
                    MAX(m.created_at) as last_message_at,
                    (SELECT content FROM rag_messages
                     WHERE conversation_id = c.id AND role = 'user'
                     ORDER BY created_at ASC LIMIT 1) as first_query
                FROM rag_conversations c
                LEFT JOIN rag_messages m ON m.conversation_id = c.id
                WHERE c.client_id = $1
                GROUP BY c.id, c.session_id, c.created_at
                ORDER BY MAX(m.created_at) DESC NULLS LAST
                LIMIT $2
                """,
                client_id,
                limit,
            )
            return [dict(row) for row in rows]

    async def get_or_create_conversation(
        self, client_id: str, session_id: str
    ) -> tuple[str, bool]:
        """Get existing conversation for session or create new one."""
        async with self.get_connection() as conn:
            # Try to find existing conversation
            existing = await conn.fetchval(
                """
                SELECT id FROM rag_conversations
                WHERE client_id = $1 AND session_id = $2
                ORDER BY created_at DESC
                LIMIT 1
                """,
                client_id,
                session_id,
            )
            if existing:
                return str(existing), False

            # Create new
            new_id = await conn.fetchval(
                """
                INSERT INTO rag_conversations (client_id, session_id)
                VALUES ($1, $2)
                RETURNING id
                """,
                client_id,
                session_id,
            )
            return str(new_id), True

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages."""
        async with self.get_connection() as conn:
            await conn.execute(
                "DELETE FROM rag_messages WHERE conversation_id = $1",
                conversation_id,
            )
            result = await conn.execute(
                "DELETE FROM rag_conversations WHERE id = $1",
                conversation_id,
            )
            return "DELETE 1" in result

    # =========================================================================
    # SCORE UPDATES (sync from Neo4j)
    # =========================================================================

    async def update_page_scores(
        self,
        client_id: str,
        url: str,
        pagerank: float,
        hub_score: float,
        authority_score: float,
    ) -> None:
        """Update PageRank and HITS scores for a page."""
        async with self.get_connection() as conn:
            await conn.execute(
                """
                UPDATE rag_pages
                SET pagerank = $3,
                    hub_score = $4,
                    authority_score = $5,
                    updated_at = NOW()
                WHERE client_id = $1 AND url = $2
                """,
                client_id,
                url,
                pagerank,
                hub_score,
                authority_score,
            )

    async def update_scores_batch(
        self,
        client_id: str,
        scores: list[dict],
    ) -> int:
        """Batch update scores for multiple pages."""
        async with self.get_connection() as conn:
            records = [
                (
                    client_id,
                    s["url"],
                    s.get("pagerank", 0.0),
                    s.get("hub_score", 0.0),
                    s.get("authority_score", 0.0),
                )
                for s in scores
            ]
            await conn.executemany(
                """
                UPDATE rag_pages
                SET pagerank = $3,
                    hub_score = $4,
                    authority_score = $5,
                    updated_at = NOW()
                WHERE client_id = $1 AND url = $2
                """,
                records,
            )
            return len(records)

    # =========================================================================
    # DELETE OPERATIONS
    # =========================================================================

    async def delete_client(self, client_id: str) -> bool:
        """Delete a client and all their data (pages, links, conversations)."""
        async with self.get_connection() as conn:
            # Delete in order due to foreign keys
            # 1. Delete messages (via conversations)
            await conn.execute(
                """
                DELETE FROM rag_messages
                WHERE conversation_id IN (
                    SELECT id FROM rag_conversations WHERE client_id = $1
                )
                """,
                client_id,
            )
            # 2. Delete conversations
            await conn.execute(
                "DELETE FROM rag_conversations WHERE client_id = $1",
                client_id,
            )
            # 3. Delete links
            await conn.execute(
                "DELETE FROM rag_links WHERE client_id = $1",
                client_id,
            )
            # 4. Delete pages
            await conn.execute(
                "DELETE FROM rag_pages WHERE client_id = $1",
                client_id,
            )
            # 5. Delete client
            result = await conn.execute(
                "DELETE FROM rag_clients WHERE id = $1",
                client_id,
            )
            return "DELETE 1" in result

    async def deactivate_client(self, client_id: str) -> bool:
        """Deactivate a client (soft delete)."""
        async with self.get_connection() as conn:
            result = await conn.execute(
                """
                UPDATE rag_clients
                SET is_active = false
                WHERE id = $1
                """,
                client_id,
            )
            return "UPDATE 1" in result

    async def regenerate_api_key(self, client_id: str) -> Optional[str]:
        """Generate a new API key for a client."""
        async with self.get_connection() as conn:
            # Generate new key using PostgreSQL
            row = await conn.fetchrow(
                """
                UPDATE rag_clients
                SET api_key = gen_random_uuid()::text
                WHERE id = $1
                RETURNING api_key
                """,
                client_id,
            )
            return row["api_key"] if row else None

    # =========================================================================
    # CHUNK OPERATIONS
    # =========================================================================

    async def upsert_chunk(
        self,
        page_id: str,
        chunk_index: int,
        content: str,
        heading_context: Optional[str] = None,
        char_start: Optional[int] = None,
        char_end: Optional[int] = None,
        embedding: Optional[list[float]] = None,
    ) -> dict:
        """Insert or update a chunk."""
        async with self.get_connection() as conn:
            embedding_str = None
            if embedding:
                embedding_str = f"[{','.join(map(str, embedding))}]"

            row = await conn.fetchrow(
                """
                INSERT INTO rag_chunks (
                    page_id, chunk_index, content, heading_context,
                    char_start, char_end, embedding
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7::vector)
                ON CONFLICT (page_id, chunk_index) DO UPDATE SET
                    content = EXCLUDED.content,
                    heading_context = EXCLUDED.heading_context,
                    char_start = EXCLUDED.char_start,
                    char_end = EXCLUDED.char_end,
                    embedding = COALESCE(EXCLUDED.embedding, rag_chunks.embedding)
                RETURNING id, page_id, chunk_index
                """,
                page_id,
                chunk_index,
                content,
                heading_context,
                char_start,
                char_end,
                embedding_str,
            )
            return dict(row)

    async def upsert_chunks_batch(
        self,
        page_id: str,
        chunks: list[dict],
    ) -> int:
        """Batch upsert chunks for a page."""
        async with self.get_connection() as conn:
            # First delete existing chunks for this page
            await conn.execute(
                "DELETE FROM rag_chunks WHERE page_id = $1",
                page_id,
            )

            # Prepare records
            records = []
            for chunk in chunks:
                embedding_str = None
                if chunk.get("embedding"):
                    embedding_str = f"[{','.join(map(str, chunk['embedding']))}]"
                records.append((
                    page_id,
                    chunk["chunk_index"],
                    chunk["content"],
                    chunk.get("heading_context"),
                    chunk.get("char_start"),
                    chunk.get("char_end"),
                    embedding_str,
                ))

            # Insert all chunks
            await conn.executemany(
                """
                INSERT INTO rag_chunks (
                    page_id, chunk_index, content, heading_context,
                    char_start, char_end, embedding
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7::vector)
                """,
                records,
            )
            return len(records)

    async def get_chunks_by_page(self, page_id: str) -> list[dict]:
        """Get all chunks for a page."""
        async with self.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT id, chunk_index, content, heading_context, char_start, char_end
                FROM rag_chunks
                WHERE page_id = $1
                ORDER BY chunk_index
                """,
                page_id,
            )
            return [dict(row) for row in rows]

    async def search_similar_chunks(
        self,
        client_id: str,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[dict]:
        """Search chunks by vector similarity using pgvector."""
        async with self.get_connection() as conn:
            embedding_str = f"[{','.join(map(str, query_embedding))}]"

            rows = await conn.fetch(
                """
                SELECT
                    c.id,
                    c.page_id,
                    c.chunk_index,
                    c.content,
                    c.heading_context,
                    p.url,
                    p.title,
                    p.pagerank,
                    p.hub_score,
                    p.authority_score,
                    1 - (c.embedding <=> $2::vector) AS similarity
                FROM rag_chunks c
                JOIN rag_pages p ON p.id = c.page_id
                WHERE p.client_id = $1
                  AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> $2::vector
                LIMIT $3
                """,
                client_id,
                embedding_str,
                limit,
            )

            results = []
            for row in rows:
                row_dict = dict(row)
                if row_dict["similarity"] >= min_similarity:
                    results.append(row_dict)

            return results

    async def get_chunks_count(self, client_id: str) -> int:
        """Get total chunk count for a client."""
        async with self.get_connection() as conn:
            result = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM rag_chunks c
                JOIN rag_pages p ON p.id = c.page_id
                WHERE p.client_id = $1
                """,
                client_id,
            )
            return result

    async def delete_chunks_by_page(self, page_id: str) -> int:
        """Delete all chunks for a page."""
        async with self.get_connection() as conn:
            result = await conn.execute(
                "DELETE FROM rag_chunks WHERE page_id = $1",
                page_id,
            )
            # Extract count from result like "DELETE 5"
            count = int(result.split()[-1]) if result else 0
            return count
