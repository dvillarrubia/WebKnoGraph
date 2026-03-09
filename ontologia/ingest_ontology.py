"""
SEOntology Ingestion Service — FUSED with graph_rag.

FUSION STRATEGY:
- Existing :Page nodes get an additional :SeoWebPage label (dual-label)
- seovoc properties are SET on the SAME node that already has pagerank/hub/authority
- All graph_rag queries (MATCH (p:Page)...) keep working unchanged
- New ontologia queries (MATCH (p:SeoWebPage)...) also work
- :LINKS_TO relationships (used by PageRank) are preserved, new rels are additive

Sprint 1: WebPage, Chunk, URL, Link, LinkGroup, AnchorText
Sprint 2: Schema, Thing (from HTML parsing)
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from neo4j import AsyncDriver


@dataclass
class SeoIngestResult:
    client_id: str
    webpages: int = 0
    chunks: int = 0
    urls: int = 0
    links: int = 0
    link_groups: int = 0
    anchor_texts: int = 0
    relationships: int = 0
    errors: list[str] = field(default_factory=list)


# =============================================================================
# XSD type mapping for Neo4j property storage
# =============================================================================
XSD_TO_NEO4J = {
    "string": "str",
    "integer": "int",
    "long": "int",
    "double": "float",
    "float": "float",
    "decimal": "float",
    "boolean": "bool",
    "dateTime": "str",  # stored as ISO string in Neo4j
}


class SeoOntologyIngestor:
    """Ingests WebKnoGraph data into SEOntology Neo4j nodes."""

    def __init__(
        self,
        driver: AsyncDriver,
        client_id: str,
        domain: str = "www.uoc.edu",
        embedding_model: str = "hiiamsid/sentence_similarity_spanish_es",
        language: str = "es",
    ):
        self.driver = driver
        self.client_id = client_id
        self.domain = domain
        self.embedding_model = embedding_model
        self.language = language

    def _is_internal(self, url: str) -> bool:
        """Check if URL belongs to the same domain."""
        try:
            parsed = urlparse(url)
            return self.domain in (parsed.netloc or "")
        except Exception:
            return False

    # =========================================================================
    # INGEST FROM PARQUET (Sprint 1 - main entry point)
    # =========================================================================

    async def ingest_from_parquet(
        self,
        crawl_dir: str | Path,
        supabase_pages: list[dict] | None = None,
        supabase_chunks: list[dict] | None = None,
    ) -> SeoIngestResult:
        """
        Ingest crawl data from parquet files into SEOntology Neo4j nodes.

        Args:
            crawl_dir: Path to crawl data directory (contains pages/ and links/)
            supabase_pages: Optional pre-loaded page data from Supabase
                            (with id, url, title, content, meta_description, etc.)
            supabase_chunks: Optional pre-loaded chunk data from Supabase

        Returns:
            SeoIngestResult with counts.
        """
        crawl_dir = Path(crawl_dir)
        result = SeoIngestResult(client_id=self.client_id)

        # Load parquet data
        pages_df = self._load_parquet(crawl_dir / "pages")
        links_df = self._load_parquet(crawl_dir / "links")

        if pages_df is None or pages_df.empty:
            result.errors.append(f"No pages found in {crawl_dir / 'pages'}")
            return result

        # Step 1: Create SeoWebPage + SeoURL nodes
        await self._ingest_webpages(pages_df, supabase_pages, result)

        # Step 2: Create SeoChunk nodes (from Supabase if available)
        if supabase_chunks:
            await self._ingest_chunks_from_supabase(supabase_chunks, result)

        # Step 3: Create SeoLinkGroup + SeoLink + SeoAnchorText nodes
        if links_df is not None and not links_df.empty:
            await self._ingest_links(links_df, result)

        # Step 4: Create LINKS_TO_PAGE relationships between SeoLink and target SeoWebPage
        if links_df is not None and not links_df.empty:
            await self._create_link_page_relationships(links_df, result)

        return result

    # =========================================================================
    # INGEST FROM SUPABASE (alternative: read directly from DB)
    # =========================================================================

    async def ingest_from_supabase(
        self,
        pages: list[dict],
        chunks: list[dict],
        links_df: pd.DataFrame | None = None,
    ) -> SeoIngestResult:
        """
        Ingest from Supabase query results (no parquet needed).

        Args:
            pages: List of dicts from rag_pages query
            chunks: List of dicts from rag_chunks query
            links_df: Optional links DataFrame from parquet
        """
        result = SeoIngestResult(client_id=self.client_id)

        # Build a DataFrame-like structure from Supabase pages
        pages_data = []
        for p in pages:
            pages_data.append({
                "url": p["url"],
                "title": p.get("title", ""),
                "meta_description": p.get("meta_description", ""),
                "markdown": p.get("content", ""),
                "word_count": len(p.get("content", "").split()) if p.get("content") else 0,
                "folder_depth": p.get("folder_depth", 0),
            })
        pages_df = pd.DataFrame(pages_data) if pages_data else pd.DataFrame()

        if not pages_df.empty:
            await self._ingest_webpages(pages_df, pages, result)

        if chunks:
            await self._ingest_chunks_from_supabase(chunks, result)

        if links_df is not None and not links_df.empty:
            await self._ingest_links(links_df, result)
            await self._create_link_page_relationships(links_df, result)

        return result

    # =========================================================================
    # PRIVATE: Node creation methods
    # =========================================================================

    def _load_parquet(self, directory: Path) -> pd.DataFrame | None:
        """Load all parquet files from a directory (including partitioned)."""
        if not directory.exists():
            return None
        parquet_files = list(directory.rglob("*.parquet"))
        if not parquet_files:
            return None
        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True)

    async def _ingest_webpages(
        self,
        pages_df: pd.DataFrame,
        supabase_pages: list[dict] | None,
        result: SeoIngestResult,
    ) -> None:
        """
        Fuse seovoc properties onto existing :Page nodes (dual-label Page:SeoWebPage).

        If :Page nodes already exist (from graph_rag ingest), we MERGE on them
        and ADD the :SeoWebPage label + seovoc properties. PageRank/HITS stay.
        If :Page doesn't exist yet, we create Page:SeoWebPage from scratch.
        """
        # Build lookup for Supabase enrichment
        sb_lookup = {}
        if supabase_pages:
            for p in supabase_pages:
                sb_lookup[p["url"]] = p

        batch = []
        for _, row in pages_df.iterrows():
            url = str(row.get("url", ""))
            if not url:
                continue

            title = str(row.get("title", ""))[:500]
            meta_desc = str(row.get("meta_description", ""))[:1000]
            markdown = str(row.get("markdown", ""))
            word_count = int(row.get("word_count", 0))
            click_depth = int(row.get("folder_depth", url.count("/") - 2))

            # Enrich from Supabase if available
            sb = sb_lookup.get(url, {})
            if not markdown and sb.get("content"):
                markdown = sb["content"]
            if not meta_desc and sb.get("meta_description"):
                meta_desc = sb["meta_description"]

            batch.append({
                "url": url,
                "title": title,
                "metaDescription": meta_desc,
                "markdownText": markdown[:50000],  # cap for Neo4j
                "embeddingModel": self.embedding_model,
                "clickDepth": click_depth,
                "isCrawlable": True,
                "inLanguage": self.language,
                "wordCount": word_count,
                # Preserve folder_depth for graph_rag compat
                "folder_depth": click_depth,
            })

        if not batch:
            return

        async with self.driver.session() as session:
            # FUSION: MERGE on :Page (graph_rag's label), then ADD :SeoWebPage label
            # This way:
            #   - If :Page exists (from ingest_service.py): we enrich it
            #   - If :Page doesn't exist: we create it with both labels
            # Existing properties (pagerank, hub_score, authority_score) are NOT overwritten
            await session.run(
                """
                UNWIND $pages AS page
                MERGE (wp:Page {client_id: $client_id, url: page.url})
                SET wp:SeoWebPage,
                    wp.title = coalesce(page.title, wp.title),
                    wp.metaDescription = page.metaDescription,
                    wp.markdownText = page.markdownText,
                    wp.embeddingModel = page.embeddingModel,
                    wp.clickDepth = page.clickDepth,
                    wp.folder_depth = coalesce(wp.folder_depth, page.folder_depth),
                    wp.isCrawlable = page.isCrawlable,
                    wp.inLanguage = page.inLanguage,
                    wp.wordCount = page.wordCount,
                    wp.updated_at = datetime()
                """,
                client_id=self.client_id,
                pages=batch,
            )
            result.webpages = len(batch)

            # Create SeoURL nodes + HAS_URL relationship
            await session.run(
                """
                UNWIND $pages AS page
                MERGE (u:SeoURL {client_id: $client_id, value: page.url})
                WITH u, page
                MATCH (wp:Page {client_id: $client_id, url: page.url})
                MERGE (wp)-[:HAS_URL]->(u)
                """,
                client_id=self.client_id,
                pages=batch,
            )
            result.urls = len(batch)
            result.relationships += len(batch)

    async def _ingest_chunks_from_supabase(
        self,
        chunks: list[dict],
        result: SeoIngestResult,
    ) -> None:
        """Create SeoChunk nodes from Supabase chunk data."""
        batch = []
        for c in chunks:
            page_url = c.get("url", "")  # joined from rag_pages
            if not page_url:
                continue
            batch.append({
                "page_url": page_url,
                "chunkText": str(c.get("content", ""))[:10000],
                "chunkPosition": int(c.get("chunk_index", 0)),
                "chunkSetName": str(c.get("heading_context", "")),
                "chunkStrategy": "semantic",
                "start": c.get("char_start"),
                "end": c.get("char_end"),
            })

        if not batch:
            return

        async with self.driver.session() as session:
            await session.run(
                """
                UNWIND $chunks AS chunk
                MERGE (c:SeoChunk {
                    client_id: $client_id,
                    page_url: chunk.page_url,
                    chunkPosition: chunk.chunkPosition
                })
                SET c.chunkText = chunk.chunkText,
                    c.chunkSetName = chunk.chunkSetName,
                    c.chunkStrategy = chunk.chunkStrategy,
                    c.start = chunk.start,
                    c.end = chunk.end,
                    c.inLanguage = $language,
                    c.updated_at = datetime()
                WITH c, chunk
                MATCH (wp:Page {client_id: $client_id, url: chunk.page_url})
                MERGE (wp)-[:HAS_CHUNK]->(c)
                """,
                client_id=self.client_id,
                chunks=batch,
                language=self.language,
            )
            result.chunks = len(batch)
            result.relationships += len(batch)

    async def _ingest_links(
        self,
        links_df: pd.DataFrame,
        result: SeoIngestResult,
    ) -> None:
        """Create SeoLinkGroup, SeoLink, and SeoAnchorText nodes."""
        # Group links by source_url + link_location to create LinkGroups
        groups = links_df.groupby(["source_url", "link_location"])

        link_group_batch = []
        link_batch = []
        anchor_batch = []

        for (source_url, location), group_df in groups:
            source_url = str(source_url)
            location = str(location) if pd.notna(location) else "content"

            # Map location to human-readable name
            location_names = {
                "nav": "Main Navigation",
                "footer": "Footer Links",
                "content": "Content Links",
                "sidebar": "Sidebar Links",
            }
            group_name = location_names.get(location, location.title() + " Links")

            link_group_batch.append({
                "source_url": source_url,
                "name": group_name,
                "identifier": f"{source_url}::{location}",
                "location": location,
            })

            for idx, (_, row) in enumerate(group_df.iterrows()):
                target_url = str(row.get("target_url", ""))
                anchor_text = str(row.get("anchor_text", ""))[:200]
                weight = float(row.get("link_weight", 0.5))
                link_type = "Inbound" if self._is_internal(target_url) else "Outbound"
                link_id = f"{source_url}::{target_url}::{idx}"

                link_batch.append({
                    "source_url": source_url,
                    "target_url": target_url,
                    "location": location,
                    "link_id": link_id,
                    "linkType": link_type,
                    "weight": weight,
                    "position": idx,
                })

                if anchor_text.strip():
                    anchor_batch.append({
                        "link_id": link_id,
                        "anchorValue": anchor_text,
                    })

        async with self.driver.session() as session:
            # Create SeoLinkGroup nodes + HAS_LINK_GROUP from SeoWebPage
            if link_group_batch:
                await session.run(
                    """
                    UNWIND $groups AS grp
                    MERGE (lg:SeoLinkGroup {
                        client_id: $client_id,
                        page_url: grp.source_url,
                        name: grp.name
                    })
                    SET lg.identifier = grp.identifier,
                        lg.location = grp.location,
                        lg.updated_at = datetime()
                    WITH lg, grp
                    MATCH (wp:Page {client_id: $client_id, url: grp.source_url})
                    MERGE (wp)-[:HAS_LINK_GROUP]->(lg)
                    """,
                    client_id=self.client_id,
                    groups=link_group_batch,
                )
                result.link_groups = len(link_group_batch)
                result.relationships += len(link_group_batch)

            # Create SeoLink nodes + HAS_LINK from SeoLinkGroup
            if link_batch:
                await session.run(
                    """
                    UNWIND $links AS link
                    MERGE (l:SeoLink {client_id: $client_id, link_id: link.link_id})
                    SET l.linkType = link.linkType,
                        l.weight = link.weight,
                        l.position = link.position,
                        l.target_url = link.target_url,
                        l.source_url = link.source_url,
                        l.updated_at = datetime()
                    WITH l, link
                    MATCH (lg:SeoLinkGroup {
                        client_id: $client_id,
                        page_url: link.source_url,
                        name: CASE link.location
                            WHEN 'nav' THEN 'Main Navigation'
                            WHEN 'footer' THEN 'Footer Links'
                            WHEN 'sidebar' THEN 'Sidebar Links'
                            ELSE 'Content Links'
                        END
                    })
                    MERGE (lg)-[:HAS_LINK]->(l)
                    """,
                    client_id=self.client_id,
                    links=link_batch,
                )
                result.links = len(link_batch)
                result.relationships += len(link_batch)

            # Create SeoAnchorText nodes
            if anchor_batch:
                await session.run(
                    """
                    UNWIND $anchors AS anchor
                    MERGE (at:SeoAnchorText {
                        client_id: $client_id,
                        anchorValue: anchor.anchorValue,
                        link_id: anchor.link_id
                    })
                    SET at.updated_at = datetime()
                    WITH at, anchor
                    MATCH (l:SeoLink {client_id: $client_id, link_id: anchor.link_id})
                    MERGE (l)-[:HAS_ANCHOR_TEXT]->(at)
                    """,
                    client_id=self.client_id,
                    anchors=anchor_batch,
                )
                result.anchor_texts = len(anchor_batch)
                result.relationships += len(anchor_batch)

    async def _create_link_page_relationships(
        self,
        links_df: pd.DataFrame,
        result: SeoIngestResult,
    ) -> None:
        """Create LINKS_TO_PAGE from SeoLink to target SeoWebPage (if target exists)."""
        # Only create relationships for internal links (target pages we have)
        targets = []
        for _, row in links_df.iterrows():
            target_url = str(row.get("target_url", ""))
            source_url = str(row.get("source_url", ""))
            if target_url and self._is_internal(target_url):
                targets.append({
                    "source_url": source_url,
                    "target_url": target_url,
                })

        if not targets:
            return

        async with self.driver.session() as session:
            res = await session.run(
                """
                UNWIND $targets AS t
                MATCH (l:SeoLink {client_id: $client_id, source_url: t.source_url, target_url: t.target_url})
                MATCH (wp:Page {client_id: $client_id, url: t.target_url})
                MERGE (l)-[:LINKS_TO_PAGE]->(wp)
                RETURN count(*) AS created
                """,
                client_id=self.client_id,
                targets=targets,
            )
            record = await res.single()
            if record:
                result.relationships += record["created"]
