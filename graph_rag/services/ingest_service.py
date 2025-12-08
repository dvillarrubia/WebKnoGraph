"""
Ingest Service for Graph-RAG.
Handles importing crawled data into the database.
"""

import asyncio
import shutil
from pathlib import Path
from glob import glob
from typing import Optional
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from graph_rag.config.settings import Settings, get_settings
from graph_rag.db.supabase_client import SupabaseClient
from graph_rag.db.neo4j_client import Neo4jClient
from graph_rag.services.embedding_service import EmbeddingService
from graph_rag.services.chunking_service import ChunkingService


@dataclass
class IngestResult:
    """Result of an ingestion operation."""
    client_id: str
    client_name: str
    pages_migrated: int
    pages_skipped: int  # Pages skipped (already exist)
    pages_updated: int  # Pages updated (overwrite mode)
    embeddings_generated: int
    chunks_created: int
    links_migrated: int
    errors: int
    error_messages: list[str]
    ingest_mode: str  # Mode used for ingestion


class IngestService:
    """Service to handle data ingestion from crawl data to databases."""

    def __init__(
        self,
        settings: Settings,
        supabase_client: SupabaseClient,
        neo4j_client: Neo4jClient,
        embedding_service: EmbeddingService,
        chunk_size: int = 512,
    ):
        self.settings = settings
        self.supabase = supabase_client
        self.neo4j = neo4j_client
        self.embedding_service = embedding_service
        self.chunking_service = ChunkingService(chunk_size=chunk_size, min_chunk_size=50)

    async def ingest_crawl_data(
        self,
        crawl_path: str,
        client_name: str,
        client_domain: str,
        ingest_mode: str = "new_only",
        delete_after: bool = False,
        min_word_count: int = 50,
    ) -> IngestResult:
        """
        Ingest crawled data into Graph-RAG databases.

        Args:
            crawl_path: Path to crawl data directory
            client_name: Name for the client
            client_domain: Domain for the client
            ingest_mode: How to handle existing data:
                - "new_only": Only insert pages that don't exist (default, safest)
                - "skip_existing": Same as new_only - skip if URL exists
                - "overwrite": Update existing pages with new data
                - "full_refresh": Delete all existing data and re-import
            delete_after: Whether to delete crawl data after ingestion
            min_word_count: Minimum word count for pages to include

        Returns:
            IngestResult with statistics
        """
        # Validate ingest_mode
        valid_modes = {"new_only", "skip_existing", "overwrite", "full_refresh"}
        if ingest_mode not in valid_modes:
            raise ValueError(f"Invalid ingest_mode: {ingest_mode}. Must be one of {valid_modes}")
        crawl_dir = Path(crawl_path)
        errors = []

        # 1. Create or get client
        try:
            client = await self.supabase.create_client(client_name, client_domain)
            client_id = str(client["id"])
        except Exception as e:
            if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                clients = await self.supabase.list_clients()
                client = next((c for c in clients if c["domain"] == client_domain), None)
                if not client:
                    raise ValueError(f"Client with domain {client_domain} not found")
                client_id = str(client["id"])
            else:
                raise

        # 2. Handle full_refresh mode - delete all existing data for this client
        if ingest_mode == "full_refresh":
            try:
                # Delete all pages and chunks for this client from Supabase
                async with self.supabase.get_connection() as conn:
                    # First delete chunks (they reference pages)
                    await conn.execute(
                        """
                        DELETE FROM rag_chunks
                        WHERE page_id IN (SELECT id FROM rag_pages WHERE client_id = $1)
                        """,
                        client_id,
                    )
                    # Then delete pages
                    await conn.execute(
                        "DELETE FROM rag_pages WHERE client_id = $1",
                        client_id,
                    )
                # Delete from Neo4j
                await self.neo4j.delete_client_data(client_id)
            except Exception as e:
                errors.append(f"Error during full_refresh cleanup: {e}")

        # 3. Load existing URLs for skip/overwrite logic
        existing_urls: set[str] = set()
        existing_pages: dict[str, str] = {}  # url -> page_id mapping
        if ingest_mode in ("new_only", "skip_existing", "overwrite"):
            try:
                async with self.supabase.get_connection() as conn:
                    rows = await conn.fetch(
                        "SELECT id, url FROM rag_pages WHERE client_id = $1",
                        client_id,
                    )
                    for row in rows:
                        existing_urls.add(row["url"])
                        existing_pages[row["url"]] = str(row["id"])
            except Exception as e:
                errors.append(f"Error loading existing URLs: {e}")

        # 4. Find page files
        pages_pattern = str(crawl_dir / "pages" / "**" / "*.parquet")
        pages_files = glob(pages_pattern, recursive=True)

        if not pages_files:
            # Try alternative structure
            pages_pattern = str(crawl_dir / "**" / "*.parquet")
            pages_files = [f for f in glob(pages_pattern, recursive=True) if "links" not in f]

        # 4b. Load manually cleaned data if available (url -> markdown_clean mapping)
        manual_clean_map: dict[str, str] = {}
        manual_clean_dir = crawl_dir / "manual_clean"
        if manual_clean_dir.exists():
            for mc_file in manual_clean_dir.glob("*.parquet"):
                try:
                    mc_df = pd.read_parquet(mc_file)
                    for _, mc_row in mc_df.iterrows():
                        mc_url = mc_row.get("url", "")
                        mc_clean = mc_row.get("markdown_clean", "")
                        if mc_url and mc_clean:
                            manual_clean_map[mc_url] = mc_clean
                except Exception as e:
                    errors.append(f"Error reading manual_clean file {mc_file}: {e}")

        pages_migrated = 0
        pages_skipped = 0
        pages_updated = 0
        embeddings_generated = 0
        chunks_created = 0

        # 5. Process pages
        for pf in pages_files:
            try:
                df = pd.read_parquet(pf)
            except Exception as e:
                errors.append(f"Error reading {pf}: {e}")
                continue

            for _, row in df.iterrows():
                url = row.get("url", "")
                # Use manually cleaned markdown if available, otherwise use original
                markdown = manual_clean_map.get(url) or row.get("markdown", "")
                title = row.get("title", "")
                content_hash = row.get("content_hash", "")
                word_count = row.get("word_count", 0)

                if not url:
                    continue

                # Skip pages with very little content
                if word_count < min_word_count:
                    continue

                # Handle skip/overwrite logic based on ingest_mode
                is_existing = url in existing_urls
                is_update = False

                if is_existing:
                    if ingest_mode in ("new_only", "skip_existing"):
                        # Skip this page - it already exists
                        pages_skipped += 1
                        continue
                    elif ingest_mode == "overwrite":
                        # Will update this page
                        is_update = True

                # Generate page-level embedding (from truncated content)
                page_embedding = None
                if markdown and len(markdown) > 100:
                    try:
                        text_for_page_embedding = markdown[:8000]
                        page_embedding = self.embedding_service.embed_document(text_for_page_embedding)
                    except Exception as e:
                        errors.append(f"Page embedding error for {url}: {e}")

                # Insert page into Supabase (with full content and page-level embedding)
                page_id = None
                try:
                    page_result = await self.supabase.upsert_page(
                        client_id=client_id,
                        url=url,
                        title=title[:500] if title else None,
                        content=markdown if markdown else None,  # Full content, no truncation
                        content_hash=content_hash,
                        embedding=page_embedding,  # Page-level embedding for fallback searches
                        pagerank=0.0,
                        hub_score=0.0,
                        authority_score=0.0,
                        folder_depth=url.count('/') - 2,
                    )
                    page_id = str(page_result["id"])
                    if is_update:
                        pages_updated += 1
                        # Delete old chunks for this page before creating new ones
                        async with self.supabase.get_connection() as conn:
                            await conn.execute(
                                "DELETE FROM rag_chunks WHERE page_id = $1",
                                page_id,
                            )
                    else:
                        pages_migrated += 1
                except Exception as e:
                    errors.append(f"DB insert error for {url}: {e}")
                    continue

                # Generate semantic chunks and their embeddings
                if markdown and len(markdown) > 100 and page_id:
                    try:
                        # Get semantic chunks
                        chunks = self.chunking_service.chunk_with_metadata(
                            text=markdown,
                            title=title,
                            url=url,
                        )

                        # Generate embeddings for each chunk
                        chunks_with_embeddings = []
                        for chunk in chunks:
                            try:
                                embedding = self.embedding_service.embed_document(chunk["content"])
                                chunk["embedding"] = embedding
                                embeddings_generated += 1
                            except Exception as e:
                                errors.append(f"Chunk embedding error for {url} chunk {chunk['chunk_index']}: {e}")
                                chunk["embedding"] = None

                            chunks_with_embeddings.append(chunk)

                        # Store chunks in database
                        if chunks_with_embeddings:
                            await self.supabase.upsert_chunks_batch(page_id, chunks_with_embeddings)
                            chunks_created += len(chunks_with_embeddings)

                    except Exception as e:
                        errors.append(f"Chunking error for {url}: {e}")

                # Insert to Neo4j
                try:
                    await self.neo4j.upsert_page(
                        client_id=client_id,
                        url=url,
                        title=title[:500] if title else None,
                        pagerank=0.0,
                        hub_score=0.0,
                        authority_score=0.0,
                        folder_depth=url.count('/') - 2,
                    )
                except Exception as e:
                    errors.append(f"Neo4j insert error for {url}: {e}")

        # 4. Process links
        links_pattern = str(crawl_dir / "links" / "**" / "*.parquet")
        links_files = glob(links_pattern, recursive=True)

        links_migrated = 0
        links_batch = []

        for lf in links_files:
            try:
                df = pd.read_parquet(lf)
            except Exception as e:
                errors.append(f"Error reading {lf}: {e}")
                continue

            for _, row in df.iterrows():
                source_url = row.get("source_url", "")
                target_url = row.get("target_url", "")
                anchor_text = row.get("anchor_text", "")

                if not source_url or not target_url:
                    continue

                links_batch.append({
                    "source_url": source_url,
                    "target_url": target_url,
                    "anchor_text": anchor_text[:200] if anchor_text else None,
                })

                if len(links_batch) >= 500:
                    try:
                        await self.neo4j.create_links_batch(client_id, links_batch)
                        links_migrated += len(links_batch)
                    except Exception as e:
                        errors.append(f"Neo4j batch error: {e}")
                    links_batch = []

        # Remaining links
        if links_batch:
            try:
                await self.neo4j.create_links_batch(client_id, links_batch)
                links_migrated += len(links_batch)
            except:
                pass

        # 5. Calculate PageRank and HITS scores
        if links_migrated > 0:
            try:
                await self.neo4j.calculate_all_scores(
                    client_id=client_id,
                    pagerank_iterations=20,
                    hits_iterations=20,
                )
                # Sync scores to Supabase
                top_pages = await self.neo4j.get_top_pages_by_pagerank(client_id, limit=10000)
                if top_pages:
                    await self.supabase.update_scores_batch(client_id, top_pages)
            except Exception as e:
                errors.append(f"Score calculation error: {e}")

        # 7. Cleanup if requested
        total_processed = pages_migrated + pages_updated
        if delete_after and total_processed > 0:
            try:
                shutil.rmtree(crawl_dir)
            except:
                pass

        return IngestResult(
            client_id=client_id,
            client_name=client_name,
            pages_migrated=pages_migrated,
            pages_skipped=pages_skipped,
            pages_updated=pages_updated,
            embeddings_generated=embeddings_generated,
            chunks_created=chunks_created,
            links_migrated=links_migrated,
            errors=len(errors),
            error_messages=errors[:20],  # First 20 errors
            ingest_mode=ingest_mode,
        )

    async def regenerate_embeddings(
        self,
        client_id: str,
        batch_size: int = 50,
    ) -> dict:
        """
        Regenerate embeddings for all pages of a client.
        Useful when changing embedding model or fixing issues.

        Returns:
            Dict with stats about the operation
        """
        errors = []
        updated = 0
        skipped = 0

        # Get all pages that have content
        async with self.supabase.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT id, url, content
                FROM rag_pages
                WHERE client_id = $1 AND content IS NOT NULL AND length(content) > 100
                ORDER BY url
                """,
                client_id,
            )

        total = len(rows)

        for i, row in enumerate(rows):
            url = row["url"]
            content = row["content"]

            try:
                # Generate new embedding
                text_for_embedding = content[:8000]
                embedding = self.embedding_service.embed_document(text_for_embedding)

                # Update in database
                embedding_str = f"[{','.join(map(str, embedding))}]"
                async with self.supabase.get_connection() as conn:
                    await conn.execute(
                        """
                        UPDATE rag_pages
                        SET embedding = $2::vector, updated_at = NOW()
                        WHERE id = $1
                        """,
                        row["id"],
                        embedding_str,
                    )
                updated += 1
            except Exception as e:
                errors.append(f"Embedding error for {url}: {e}")
                skipped += 1

        return {
            "client_id": client_id,
            "total_pages": total,
            "updated": updated,
            "skipped": skipped,
            "errors": len(errors),
            "error_messages": errors[:10],
        }
