"""
Neo4j client for Graph-RAG.
Handles graph traversal and link-based context expansion.
"""

from typing import Optional
from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase, AsyncDriver


class Neo4jClient:
    """Async Neo4j client for graph operations."""

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver: Optional[AsyncDriver] = None

    async def connect(self) -> None:
        """Initialize driver connection."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            # Verify connectivity
            await self._driver.verify_connectivity()

    async def disconnect(self) -> None:
        """Close driver connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    @asynccontextmanager
    async def get_session(self):
        """Get a session from the driver."""
        if self._driver is None:
            await self.connect()
        async with self._driver.session() as session:
            yield session

    # =========================================================================
    # SCHEMA SETUP
    # =========================================================================

    async def setup_constraints(self) -> None:
        """Create constraints and indexes."""
        async with self.get_session() as session:
            # Unique constraint for Page
            await session.run(
                """
                CREATE CONSTRAINT page_unique_url IF NOT EXISTS
                FOR (p:Page) REQUIRE (p.client_id, p.url) IS UNIQUE
                """
            )
            # Index for client_id lookups
            await session.run(
                """
                CREATE INDEX page_client_id IF NOT EXISTS
                FOR (p:Page) ON (p.client_id)
                """
            )
            # Index for pagerank
            await session.run(
                """
                CREATE INDEX page_pagerank IF NOT EXISTS
                FOR (p:Page) ON (p.client_id, p.pagerank)
                """
            )

    # =========================================================================
    # PAGE OPERATIONS
    # =========================================================================

    async def upsert_page(
        self,
        client_id: str,
        url: str,
        title: Optional[str] = None,
        pagerank: float = 0.0,
        hub_score: float = 0.0,
        authority_score: float = 0.0,
        folder_depth: int = 0,
    ) -> None:
        """Insert or update a page node."""
        async with self.get_session() as session:
            await session.run(
                """
                MERGE (p:Page {client_id: $client_id, url: $url})
                SET p.title = $title,
                    p.pagerank = $pagerank,
                    p.hub_score = $hub_score,
                    p.authority_score = $authority_score,
                    p.folder_depth = $folder_depth,
                    p.updated_at = datetime()
                """,
                client_id=client_id,
                url=url,
                title=title,
                pagerank=pagerank,
                hub_score=hub_score,
                authority_score=authority_score,
                folder_depth=folder_depth,
            )

    async def upsert_pages_batch(self, client_id: str, pages: list[dict]) -> int:
        """Batch upsert pages using UNWIND."""
        async with self.get_session() as session:
            await session.run(
                """
                UNWIND $pages AS page
                MERGE (p:Page {client_id: $client_id, url: page.url})
                SET p.title = page.title,
                    p.pagerank = page.pagerank,
                    p.hub_score = page.hub_score,
                    p.authority_score = page.authority_score,
                    p.folder_depth = page.folder_depth,
                    p.updated_at = datetime()
                """,
                client_id=client_id,
                pages=pages,
            )
            return len(pages)

    # =========================================================================
    # LINK OPERATIONS
    # =========================================================================

    async def create_link(
        self,
        client_id: str,
        source_url: str,
        target_url: str,
        anchor_text: Optional[str] = None,
    ) -> None:
        """Create a LINKS_TO relationship between two pages."""
        async with self.get_session() as session:
            await session.run(
                """
                MATCH (source:Page {client_id: $client_id, url: $source_url})
                MATCH (target:Page {client_id: $client_id, url: $target_url})
                MERGE (source)-[r:LINKS_TO]->(target)
                SET r.anchor_text = $anchor_text
                """,
                client_id=client_id,
                source_url=source_url,
                target_url=target_url,
                anchor_text=anchor_text,
            )

    async def create_links_batch(self, client_id: str, links: list[dict]) -> int:
        """Batch create links using UNWIND."""
        async with self.get_session() as session:
            await session.run(
                """
                UNWIND $links AS link
                MATCH (source:Page {client_id: $client_id, url: link.source_url})
                MATCH (target:Page {client_id: $client_id, url: link.target_url})
                MERGE (source)-[r:LINKS_TO]->(target)
                SET r.anchor_text = link.anchor_text
                """,
                client_id=client_id,
                links=links,
            )
            return len(links)

    # =========================================================================
    # GRAPH TRAVERSAL (for RAG context expansion)
    # =========================================================================

    async def get_linked_pages(
        self,
        client_id: str,
        url: str,
        hops: int = 1,
        limit: int = 20,
    ) -> list[dict]:
        """
        Get pages linked from/to a source URL within N hops.
        Used for graph-based context expansion in RAG.
        """
        # Validate hops to prevent injection (Neo4j doesn't allow params in variable-length patterns)
        hops = max(1, min(int(hops), 5))

        async with self.get_session() as session:
            result = await session.run(
                f"""
                MATCH (source:Page {{client_id: $client_id, url: $url}})
                      -[:LINKS_TO*1..{hops}]-(related:Page)
                WHERE related.client_id = $client_id
                RETURN DISTINCT
                    related.url AS url,
                    related.title AS title,
                    related.pagerank AS pagerank,
                    related.hub_score AS hub_score,
                    related.authority_score AS authority_score,
                    related.folder_depth AS folder_depth
                ORDER BY related.pagerank DESC
                LIMIT $limit
                """,
                client_id=client_id,
                url=url,
                limit=limit,
            )
            records = await result.data()
            return records

    async def get_outgoing_links(
        self,
        client_id: str,
        url: str,
        limit: int = 50,
    ) -> list[dict]:
        """Get pages that a source URL links TO."""
        async with self.get_session() as session:
            result = await session.run(
                """
                MATCH (source:Page {client_id: $client_id, url: $url})
                      -[:LINKS_TO]->(target:Page)
                WHERE target.client_id = $client_id
                RETURN target.url AS url,
                       target.title AS title,
                       target.pagerank AS pagerank
                ORDER BY target.pagerank DESC
                LIMIT $limit
                """,
                client_id=client_id,
                url=url,
                limit=limit,
            )
            records = await result.data()
            return records

    async def get_incoming_links(
        self,
        client_id: str,
        url: str,
        limit: int = 50,
    ) -> list[dict]:
        """Get pages that link TO a target URL (backlinks)."""
        async with self.get_session() as session:
            result = await session.run(
                """
                MATCH (source:Page)-[:LINKS_TO]->(target:Page {client_id: $client_id, url: $url})
                WHERE source.client_id = $client_id
                RETURN source.url AS url,
                       source.title AS title,
                       source.pagerank AS pagerank
                ORDER BY source.pagerank DESC
                LIMIT $limit
                """,
                client_id=client_id,
                url=url,
                limit=limit,
            )
            records = await result.data()
            return records

    async def get_shortest_path(
        self,
        client_id: str,
        source_url: str,
        target_url: str,
    ) -> Optional[list[str]]:
        """Find shortest path between two pages."""
        async with self.get_session() as session:
            result = await session.run(
                """
                MATCH path = shortestPath(
                    (a:Page {client_id: $client_id, url: $source_url})
                    -[:LINKS_TO*]->
                    (b:Page {client_id: $client_id, url: $target_url})
                )
                RETURN [node IN nodes(path) | node.url] AS path_urls
                """,
                client_id=client_id,
                source_url=source_url,
                target_url=target_url,
            )
            record = await result.single()
            return record["path_urls"] if record else None

    async def expand_context_from_urls(
        self,
        client_id: str,
        urls: list[str],
        hops: int = 1,
        limit_per_url: int = 5,
    ) -> list[dict]:
        """
        Expand context by getting linked pages from multiple source URLs.
        Used to enrich RAG context with graph neighborhood.
        """
        # Neo4j doesn't allow parameters in relationship patterns, so we build the query
        hops = min(max(1, hops), 5)  # Limit hops between 1 and 5 for safety
        query = f"""
            UNWIND $urls AS source_url
            MATCH (source:Page {{client_id: $client_id, url: source_url}})
                  -[:LINKS_TO*1..{hops}]-(related:Page)
            WHERE related.client_id = $client_id
              AND NOT related.url IN $urls
            RETURN DISTINCT
                related.url AS url,
                related.title AS title,
                related.pagerank AS pagerank,
                related.hub_score AS hub_score,
                related.authority_score AS authority_score,
                related.folder_depth AS folder_depth
            ORDER BY related.pagerank DESC
            LIMIT $total_limit
        """
        async with self.get_session() as session:
            result = await session.run(
                query,
                client_id=client_id,
                urls=urls,
                total_limit=limit_per_url * len(urls),
            )
            records = await result.data()
            return records

    # =========================================================================
    # STATS & INFO
    # =========================================================================

    async def get_client_stats(self, client_id: str) -> dict:
        """Get statistics for a client's graph."""
        async with self.get_session() as session:
            result = await session.run(
                """
                MATCH (p:Page {client_id: $client_id})
                OPTIONAL MATCH (p)-[r:LINKS_TO]->()
                RETURN
                    COUNT(DISTINCT p) AS page_count,
                    COUNT(r) AS link_count
                """,
                client_id=client_id,
            )
            record = await result.single()
            return {
                "page_count": record["page_count"],
                "link_count": record["link_count"],
            }

    async def delete_client_data(self, client_id: str) -> None:
        """Delete all data for a client."""
        async with self.get_session() as session:
            await session.run(
                """
                MATCH (p:Page {client_id: $client_id})
                DETACH DELETE p
                """,
                client_id=client_id,
            )

    # =========================================================================
    # PAGERANK & HITS CALCULATION
    # =========================================================================

    async def calculate_pagerank(
        self,
        client_id: str,
        iterations: int = 20,
        damping_factor: float = 0.85,
    ) -> dict:
        """
        Calculate PageRank for all pages of a client.
        Uses iterative algorithm since Neo4j Community doesn't have GDS.

        Returns:
            Dict with stats about the calculation
        """
        async with self.get_session() as session:
            # Step 1: Count pages and initialize PageRank
            result = await session.run(
                """
                MATCH (p:Page {client_id: $client_id})
                WITH count(p) as total_pages
                MATCH (p:Page {client_id: $client_id})
                SET p.pagerank = 1.0 / total_pages
                RETURN total_pages
                """,
                client_id=client_id,
            )
            record = await result.single()
            total_pages = record["total_pages"] if record else 0

            if total_pages == 0:
                return {"pages": 0, "iterations": 0, "status": "no_pages"}

            # Step 2: Iterative PageRank calculation
            for i in range(iterations):
                await session.run(
                    """
                    // Calculate new PageRank for each page
                    MATCH (p:Page {client_id: $client_id})

                    // Get incoming links and their source PageRanks
                    OPTIONAL MATCH (source:Page {client_id: $client_id})-[:LINKS_TO]->(p)

                    // Count outgoing links from each source
                    WITH p, source,
                         CASE WHEN source IS NOT NULL
                              THEN size([(source)-[:LINKS_TO]->() | 1])
                              ELSE 1 END as out_degree

                    // Sum contributions from incoming links
                    WITH p,
                         sum(CASE WHEN source IS NOT NULL
                                  THEN source.pagerank / out_degree
                                  ELSE 0 END) as incoming_rank

                    // Apply damping factor formula
                    SET p.new_pagerank = (1.0 - $damping) / $total_pages + $damping * incoming_rank
                    """,
                    client_id=client_id,
                    damping=damping_factor,
                    total_pages=total_pages,
                )

                # Copy new_pagerank to pagerank
                await session.run(
                    """
                    MATCH (p:Page {client_id: $client_id})
                    SET p.pagerank = COALESCE(p.new_pagerank, p.pagerank)
                    REMOVE p.new_pagerank
                    """,
                    client_id=client_id,
                )

            # Step 3: Normalize PageRank values (optional, for better readability)
            await session.run(
                """
                MATCH (p:Page {client_id: $client_id})
                WITH max(p.pagerank) as max_pr, min(p.pagerank) as min_pr
                MATCH (p:Page {client_id: $client_id})
                SET p.pagerank = CASE
                    WHEN max_pr = min_pr THEN 0.5
                    ELSE (p.pagerank - min_pr) / (max_pr - min_pr)
                END
                """,
                client_id=client_id,
            )

            return {
                "pages": total_pages,
                "iterations": iterations,
                "damping_factor": damping_factor,
                "status": "completed",
            }

    async def calculate_hits(
        self,
        client_id: str,
        iterations: int = 20,
    ) -> dict:
        """
        Calculate HITS (Hyperlink-Induced Topic Search) scores.
        Computes hub_score and authority_score for each page.

        Hub: Page that links to many authorities
        Authority: Page that is linked by many hubs

        Returns:
            Dict with stats about the calculation
        """
        async with self.get_session() as session:
            # Step 1: Initialize hub and authority scores
            result = await session.run(
                """
                MATCH (p:Page {client_id: $client_id})
                SET p.hub_score = 1.0, p.authority_score = 1.0
                RETURN count(p) as total_pages
                """,
                client_id=client_id,
            )
            record = await result.single()
            total_pages = record["total_pages"] if record else 0

            if total_pages == 0:
                return {"pages": 0, "iterations": 0, "status": "no_pages"}

            # Step 2: Iterative HITS calculation
            for i in range(iterations):
                # Update authority scores (sum of hub scores of pages pointing to this page)
                await session.run(
                    """
                    MATCH (p:Page {client_id: $client_id})
                    OPTIONAL MATCH (source:Page {client_id: $client_id})-[:LINKS_TO]->(p)
                    WITH p, sum(COALESCE(source.hub_score, 0)) as new_auth
                    SET p.new_authority = new_auth
                    """,
                    client_id=client_id,
                )

                # Update hub scores (sum of authority scores of pages this page points to)
                await session.run(
                    """
                    MATCH (p:Page {client_id: $client_id})
                    OPTIONAL MATCH (p)-[:LINKS_TO]->(target:Page {client_id: $client_id})
                    WITH p, sum(COALESCE(target.new_authority, target.authority_score, 0)) as new_hub
                    SET p.new_hub = new_hub
                    """,
                    client_id=client_id,
                )

                # Normalize and apply new scores
                await session.run(
                    """
                    // Normalize authority scores
                    MATCH (p:Page {client_id: $client_id})
                    WITH sqrt(sum(p.new_authority * p.new_authority)) as auth_norm
                    MATCH (p:Page {client_id: $client_id})
                    SET p.authority_score = CASE
                        WHEN auth_norm > 0 THEN p.new_authority / auth_norm
                        ELSE 0
                    END
                    REMOVE p.new_authority
                    """,
                    client_id=client_id,
                )

                await session.run(
                    """
                    // Normalize hub scores
                    MATCH (p:Page {client_id: $client_id})
                    WITH sqrt(sum(p.new_hub * p.new_hub)) as hub_norm
                    MATCH (p:Page {client_id: $client_id})
                    SET p.hub_score = CASE
                        WHEN hub_norm > 0 THEN p.new_hub / hub_norm
                        ELSE 0
                    END
                    REMOVE p.new_hub
                    """,
                    client_id=client_id,
                )

            return {
                "pages": total_pages,
                "iterations": iterations,
                "status": "completed",
            }

    async def calculate_all_scores(
        self,
        client_id: str,
        pagerank_iterations: int = 20,
        hits_iterations: int = 20,
    ) -> dict:
        """
        Calculate both PageRank and HITS scores for a client.
        Returns combined stats.
        """
        pr_stats = await self.calculate_pagerank(
            client_id=client_id,
            iterations=pagerank_iterations,
        )
        hits_stats = await self.calculate_hits(
            client_id=client_id,
            iterations=hits_iterations,
        )

        return {
            "pagerank": pr_stats,
            "hits": hits_stats,
            "status": "completed",
        }

    async def get_top_pages_by_pagerank(
        self,
        client_id: str,
        limit: int = 20,
    ) -> list[dict]:
        """Get top pages ordered by PageRank."""
        async with self.get_session() as session:
            result = await session.run(
                """
                MATCH (p:Page {client_id: $client_id})
                RETURN p.url AS url,
                       p.title AS title,
                       p.pagerank AS pagerank,
                       p.hub_score AS hub_score,
                       p.authority_score AS authority_score
                ORDER BY p.pagerank DESC
                LIMIT $limit
                """,
                client_id=client_id,
                limit=limit,
            )
            return await result.data()
