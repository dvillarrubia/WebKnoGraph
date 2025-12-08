"""
Migration Service for Graph-RAG.
Migrates data from WebKnoGraph's existing files to Supabase + Neo4j.
"""

import os
import hashlib
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from graph_rag.config.settings import Settings
from graph_rag.db.supabase_client import SupabaseClient
from graph_rag.db.neo4j_client import Neo4jClient
from graph_rag.services.embedding_service import EmbeddingService


class MigrationService:
    """
    Migrates existing WebKnoGraph data to the new multi-tenant Graph-RAG system.

    Source data (from WebKnoGraph):
    - data/crawled_data_parquet/     -> Page content
    - data/url_embeddings/           -> Existing embeddings (will re-generate with new model)
    - data/link_graph_edges.csv      -> Internal links
    - data/url_analysis_results.csv  -> PageRank, HITS scores
    """

    def __init__(
        self,
        settings: Settings,
        supabase_client: SupabaseClient,
        neo4j_client: Neo4jClient,
        embedding_service: EmbeddingService,
    ):
        self.settings = settings
        self.supabase = supabase_client
        self.neo4j = neo4j_client
        self.embedding_service = embedding_service

    async def create_client_from_domain(self, name: str, domain: str) -> dict:
        """Create a new client for a domain."""
        return await self.supabase.create_client(name, domain)

    async def migrate_from_webknograph(
        self,
        client_id: str,
        data_path: str,
        regenerate_embeddings: bool = True,
        batch_size: int = 100,
    ) -> dict:
        """
        Full migration from WebKnoGraph data folder to Graph-RAG.

        Args:
            client_id: Target client ID
            data_path: Path to WebKnoGraph data folder
            regenerate_embeddings: If True, regenerate with multilingual-e5-large
            batch_size: Batch size for DB operations

        Returns:
            Migration statistics
        """
        data_path = Path(data_path)
        stats = {
            "pages_migrated": 0,
            "links_migrated": 0,
            "embeddings_generated": 0,
            "errors": [],
        }

        # 1. Load analysis results (PageRank, HITS, folder depth)
        analysis_path = data_path / "url_analysis_results.csv"
        if analysis_path.exists():
            analysis_df = pd.read_csv(analysis_path)
            print(f"Loaded {len(analysis_df)} URL analysis records")
        else:
            analysis_df = pd.DataFrame()
            stats["errors"].append("url_analysis_results.csv not found")

        # Create lookup dict for metrics
        url_metrics = {}
        if not analysis_df.empty:
            for _, row in analysis_df.iterrows():
                url_metrics[row["URL"]] = {
                    "pagerank": row.get("PageRank", 0.0),
                    "hub_score": row.get("Hub_Score", 0.0),
                    "authority_score": row.get("Authority_Score", 0.0),
                    "folder_depth": row.get("Folder_Depth", 0),
                }

        # 2. Load crawled content
        crawled_path = data_path / "crawled_data_parquet"
        content_df = None
        if crawled_path.exists():
            try:
                content_df = pd.read_parquet(crawled_path)
                print(f"Loaded {len(content_df)} crawled pages")
            except Exception as e:
                stats["errors"].append(f"Error loading parquet: {e}")

        # 3. Migrate pages to Supabase and Neo4j
        if content_df is not None and not content_df.empty:
            pages_batch = []

            for idx, row in tqdm(content_df.iterrows(), total=len(content_df), desc="Processing pages"):
                url = row.get("URL", "")
                content = row.get("Content", "") or row.get("clean_text", "")

                if not url:
                    continue

                # Get metrics from analysis
                metrics = url_metrics.get(url, {})

                # Generate content hash
                content_hash = hashlib.sha256(content.encode()).hexdigest() if content else None

                # Extract title from content (simple heuristic)
                title = self._extract_title(content)

                # Generate embedding if requested
                embedding = None
                if regenerate_embeddings and content and len(content) > 100:
                    try:
                        # Truncate content to avoid token limits
                        truncated_content = content[:8000]
                        embedding = self.embedding_service.embed_document(truncated_content)
                        stats["embeddings_generated"] += 1
                    except Exception as e:
                        stats["errors"].append(f"Embedding error for {url}: {e}")

                page_data = {
                    "url": url,
                    "title": title,
                    "content": content[:50000] if content else None,  # Limit content size
                    "content_hash": content_hash,
                    "embedding": embedding,
                    "pagerank": metrics.get("pagerank", 0.0),
                    "hub_score": metrics.get("hub_score", 0.0),
                    "authority_score": metrics.get("authority_score", 0.0),
                    "folder_depth": metrics.get("folder_depth", 0),
                }
                pages_batch.append(page_data)

                # Batch insert
                if len(pages_batch) >= batch_size:
                    await self._insert_pages_batch(client_id, pages_batch)
                    stats["pages_migrated"] += len(pages_batch)
                    pages_batch = []

            # Insert remaining
            if pages_batch:
                await self._insert_pages_batch(client_id, pages_batch)
                stats["pages_migrated"] += len(pages_batch)

        # 4. Migrate links
        links_path = data_path / "link_graph_edges.csv"
        if links_path.exists():
            links_df = pd.read_csv(links_path)
            print(f"Loaded {len(links_df)} links")

            # Get URL to page ID mapping from Supabase
            url_to_id = await self.supabase.get_url_to_page_id_mapping(client_id)

            links_batch_supabase = []
            links_batch_neo4j = []

            for _, row in tqdm(links_df.iterrows(), total=len(links_df), desc="Processing links"):
                source_url = row.get("Source", "") or row.get("FROM", "")
                target_url = row.get("Target", "") or row.get("TO", "")

                if not source_url or not target_url:
                    continue

                source_id = url_to_id.get(source_url)
                target_id = url_to_id.get(target_url)

                # Only add if both pages exist
                if source_id and target_id:
                    links_batch_supabase.append({
                        "source_page_id": source_id,
                        "target_page_id": target_id,
                        "anchor_text": row.get("Anchor_Text"),
                    })

                links_batch_neo4j.append({
                    "source_url": source_url,
                    "target_url": target_url,
                    "anchor_text": row.get("Anchor_Text"),
                })

                # Batch insert
                if len(links_batch_neo4j) >= batch_size:
                    await self.supabase.upsert_links_batch(client_id, links_batch_supabase)
                    await self.neo4j.create_links_batch(client_id, links_batch_neo4j)
                    stats["links_migrated"] += len(links_batch_neo4j)
                    links_batch_supabase = []
                    links_batch_neo4j = []

            # Insert remaining
            if links_batch_neo4j:
                await self.supabase.upsert_links_batch(client_id, links_batch_supabase)
                await self.neo4j.create_links_batch(client_id, links_batch_neo4j)
                stats["links_migrated"] += len(links_batch_neo4j)

        return stats

    async def _insert_pages_batch(self, client_id: str, pages: list[dict]) -> None:
        """Insert pages to both Supabase and Neo4j."""
        # Supabase (with embeddings)
        await self.supabase.upsert_pages_batch(client_id, pages)

        # Neo4j (without embeddings, just graph structure)
        neo4j_pages = [
            {
                "url": p["url"],
                "title": p["title"],
                "pagerank": p["pagerank"],
                "hub_score": p["hub_score"],
                "authority_score": p["authority_score"],
                "folder_depth": p["folder_depth"],
            }
            for p in pages
        ]
        await self.neo4j.upsert_pages_batch(client_id, neo4j_pages)

    def _extract_title(self, content: str) -> Optional[str]:
        """Extract title from HTML content."""
        if not content:
            return None

        # Try to find <title> tag
        import re
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", content, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()[:500]

        # Try <h1>
        h1_match = re.search(r"<h1[^>]*>([^<]+)</h1>", content, re.IGNORECASE)
        if h1_match:
            return h1_match.group(1).strip()[:500]

        return None

    async def migrate_only_embeddings(
        self,
        client_id: str,
        batch_size: int = 50,
    ) -> dict:
        """
        Re-generate embeddings for existing pages with the new model.
        Useful when changing embedding models.
        """
        stats = {"updated": 0, "errors": []}

        # This would require a new method in supabase_client to get pages without embeddings
        # or all pages that need re-embedding
        # For now, this is a placeholder

        return stats
