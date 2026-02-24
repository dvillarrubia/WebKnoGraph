"""
Community Detection Service for Graph-RAG.
Uses Louvain algorithm to detect communities in the page link graph.
"""

import logging
from typing import Optional
import networkx as nx
import community as community_louvain

from graph_rag.db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class CommunityService:
    """
    Service for detecting and managing page communities.

    Uses the Louvain algorithm to partition the page graph into
    thematic communities based on link structure.
    """

    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client

    async def detect_communities(
        self,
        client_id: str,
        resolution: float = 1.0,
    ) -> dict:
        """
        Detect communities in the page graph using Louvain algorithm.

        Args:
            client_id: Client ID to analyze.
            resolution: Resolution parameter for Louvain (higher = more communities).

        Returns:
            Dict with community statistics.
        """
        logger.info(f"Detecting communities for client {client_id}")

        # 1. Extract graph from Neo4j
        graph = await self._extract_graph(client_id)

        if graph.number_of_nodes() == 0:
            logger.warning(f"No nodes found for client {client_id}")
            return {"communities": 0, "nodes": 0}

        # 2. Run Louvain algorithm
        partition = community_louvain.best_partition(
            graph,
            resolution=resolution,
            random_state=42,
        )

        # 3. Count communities and get statistics
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)

        # 4. Update Neo4j with community assignments
        await self._update_communities(client_id, partition)

        # 5. Generate community labels based on top pages
        community_info = await self._generate_community_info(client_id, communities)

        logger.info(f"Detected {len(communities)} communities for {graph.number_of_nodes()} pages")

        return {
            "communities": len(communities),
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "community_sizes": {k: len(v) for k, v in communities.items()},
            "community_info": community_info,
        }

    async def _extract_graph(self, client_id: str) -> nx.Graph:
        """Extract the page link graph from Neo4j."""
        graph = nx.Graph()  # Undirected for community detection

        async with self.neo4j.get_session() as session:
            # Get all pages and links
            result = await session.run(
                """
                MATCH (p1:Page)-[:LINKS_TO]->(p2:Page)
                WHERE p1.client_id = $client_id AND p2.client_id = $client_id
                RETURN p1.url AS source, p2.url AS target, p1.title AS source_title, p2.title AS target_title
                """,
                client_id=client_id,
            )
            records = await result.data()

            for record in records:
                source = record["source"]
                target = record["target"]
                # Add nodes with title attribute
                graph.add_node(source, title=record["source_title"])
                graph.add_node(target, title=record["target_title"])
                # Add edge
                graph.add_edge(source, target)

            # Also add isolated pages (no links)
            result = await session.run(
                """
                MATCH (p:Page)
                WHERE p.client_id = $client_id
                AND NOT (p)-[:LINKS_TO]-()
                AND NOT ()-[:LINKS_TO]->(p)
                RETURN p.url AS url, p.title AS title
                """,
                client_id=client_id,
            )
            records = await result.data()
            for record in records:
                graph.add_node(record["url"], title=record["title"])

        return graph

    async def _update_communities(
        self,
        client_id: str,
        partition: dict[str, int],
    ) -> None:
        """Update Neo4j nodes with community assignments."""
        async with self.neo4j.get_session() as session:
            # Batch update in chunks
            items = list(partition.items())
            batch_size = 100

            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                await session.run(
                    """
                    UNWIND $updates AS update
                    MATCH (p:Page {client_id: $client_id, url: update.url})
                    SET p.community_id = update.community_id
                    """,
                    client_id=client_id,
                    updates=[{"url": url, "community_id": comm_id} for url, comm_id in batch],
                )

    async def _generate_community_info(
        self,
        client_id: str,
        communities: dict[int, list[str]],
    ) -> dict:
        """Generate descriptive info for each community based on top pages."""
        info = {}

        async with self.neo4j.get_session() as session:
            for comm_id, urls in communities.items():
                # Get top pages by PageRank in this community
                result = await session.run(
                    """
                    MATCH (p:Page)
                    WHERE p.client_id = $client_id AND p.url IN $urls
                    RETURN p.url, p.title, p.pagerank
                    ORDER BY p.pagerank DESC
                    LIMIT 3
                    """,
                    client_id=client_id,
                    urls=urls,
                )
                records = await result.data()

                top_pages = [
                    {"url": r["p.url"], "title": r["p.title"], "pagerank": r["p.pagerank"]}
                    for r in records
                ]

                # Extract common terms from titles for a label
                titles = [r["p.title"] or "" for r in records]
                label = self._extract_community_label(titles)

                info[comm_id] = {
                    "size": len(urls),
                    "top_pages": top_pages,
                    "label": label,
                }

        return info

    def _extract_community_label(self, titles: list[str]) -> str:
        """Extract a label for the community based on common terms in titles."""
        if not titles:
            return "General"

        # Simple approach: find most common significant words
        stop_words = {"de", "en", "la", "el", "los", "las", "un", "una", "y", "o", "|", "-", "a", "para", "con"}
        word_counts = {}

        for title in titles:
            words = title.lower().split()
            for word in words:
                word = word.strip(".,;:()[]")
                if len(word) > 2 and word not in stop_words:
                    word_counts[word] = word_counts.get(word, 0) + 1

        if not word_counts:
            return titles[0][:30] if titles else "General"

        # Return most common word(s)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[0][0].capitalize() if sorted_words else "General"

    async def get_community_pages(
        self,
        client_id: str,
        community_id: int,
        limit: int = 10,
    ) -> list[dict]:
        """Get pages in a specific community."""
        async with self.neo4j.get_session() as session:
            result = await session.run(
                """
                MATCH (p:Page)
                WHERE p.client_id = $client_id AND p.community_id = $community_id
                RETURN p.url, p.title, p.pagerank, p.hub_score, p.authority_score
                ORDER BY p.pagerank DESC
                LIMIT $limit
                """,
                client_id=client_id,
                community_id=community_id,
                limit=limit,
            )
            records = await result.data()
            return [dict(r) for r in records]

    async def get_related_communities(
        self,
        client_id: str,
        url: str,
    ) -> list[dict]:
        """Get communities related to a specific page via links."""
        async with self.neo4j.get_session() as session:
            # Get the page's community
            result = await session.run(
                """
                MATCH (p:Page {client_id: $client_id, url: $url})
                RETURN p.community_id AS community_id
                """,
                client_id=client_id,
                url=url,
            )
            records = await result.data()
            if not records or records[0]["community_id"] is None:
                return []

            page_community = records[0]["community_id"]

            # Find linked communities
            result = await session.run(
                """
                MATCH (p1:Page {client_id: $client_id, url: $url})-[:LINKS_TO]-(p2:Page)
                WHERE p2.community_id <> $page_community
                RETURN DISTINCT p2.community_id AS community_id, COUNT(*) AS link_count
                ORDER BY link_count DESC
                LIMIT 5
                """,
                client_id=client_id,
                url=url,
                page_community=page_community,
            )
            records = await result.data()
            return [dict(r) for r in records]
