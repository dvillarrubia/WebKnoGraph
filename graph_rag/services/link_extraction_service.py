"""
Link Extraction Service — Extracts internal links from HTML in parquets
and creates :LINKS_TO relationships in Neo4j.

Also calculates PageRank, Hub and Authority scores.
"""

import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import pyarrow.parquet as pq
from bs4 import BeautifulSoup
from neo4j import AsyncDriver

from graph_rag.db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class LinkExtractionResult:
    client_id: str
    pages_processed: int = 0
    links_extracted: int = 0
    links_to_created: int = 0
    pagerank_calculated: bool = False
    errors: list = field(default_factory=list)


def _classify_link_location(a_tag, soup) -> str:
    """Classify link location based on DOM position and parent elements."""
    # Walk up the DOM tree to find context
    for parent in a_tag.parents:
        if parent is None or parent.name is None:
            break
        tag_name = parent.name.lower()
        classes = " ".join(parent.get("class", [])).lower()
        element_id = (parent.get("id") or "").lower()
        role = (parent.get("role") or "").lower()

        # Nav detection
        if tag_name == "nav" or role == "navigation":
            return "nav"
        if any(kw in classes for kw in ("menu", "nav", "navbar", "header-menu", "main-menu", "breadcrumb")):
            return "nav"
        if any(kw in element_id for kw in ("menu", "nav", "navbar", "header")):
            return "nav"

        # Header detection
        if tag_name == "header":
            return "nav"
        if any(kw in classes for kw in ("header", "top-bar", "site-header")):
            return "nav"

        # Footer detection
        if tag_name == "footer":
            return "footer"
        if any(kw in classes for kw in ("footer", "site-footer", "bottom")):
            return "footer"
        if any(kw in element_id for kw in ("footer",)):
            return "footer"

        # Sidebar detection
        if tag_name == "aside":
            return "sidebar"
        if any(kw in classes for kw in ("sidebar", "aside", "widget")):
            return "sidebar"

    return "content"


def _get_link_weight(location: str) -> float:
    """Assign weight based on link location (for PageRank)."""
    weights = {
        "content": 1.0,
        "sidebar": 0.4,
        "nav": 0.3,
        "footer": 0.3,
    }
    return weights.get(location, 0.5)


def _extract_links_from_html(html: str, source_url: str, domain: str) -> list[dict]:
    """Extract internal links from HTML, classify by location."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    seen = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue

        # Resolve relative URLs
        absolute_url = urljoin(source_url, href)

        # Parse and normalize
        parsed = urlparse(absolute_url)
        if not parsed.scheme or not parsed.netloc:
            continue

        # Only internal links (same domain)
        if domain not in parsed.netloc:
            continue

        # Normalize: strip fragment and trailing slash for dedup
        target_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            target_url += f"?{parsed.query}"
        target_url = target_url.rstrip("/")
        source_normalized = source_url.rstrip("/")

        # Skip self-links
        if target_url == source_normalized:
            continue

        # Dedup within same page
        dedup_key = (source_normalized, target_url)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # Classify location
        location = _classify_link_location(a_tag, soup)
        anchor_text = a_tag.get_text(strip=True)[:200]

        links.append({
            "source_url": source_url,
            "target_url": target_url,
            "anchor_text": anchor_text,
            "location": location,
            "weight": _get_link_weight(location),
        })

    return links


def _read_and_extract_links(crawl_dir: str, domain: str, date_partition: Optional[str] = None) -> list[dict]:
    """Read parquets and extract all internal links."""
    pages_dir = Path(crawl_dir) / "pages"
    all_links = []

    for partition_dir in sorted(pages_dir.iterdir()):
        if not partition_dir.is_dir():
            continue
        if date_partition and date_partition not in partition_dir.name:
            continue

        for parquet_file in sorted(partition_dir.glob("*.parquet")):
            try:
                table = pq.ParquetFile(str(parquet_file)).read()
                if "html_content" not in table.column_names:
                    continue

                urls = table.column("url").to_pylist()
                htmls = table.column("html_content").to_pylist()

                for url, html in zip(urls, htmls):
                    if html and len(html) > 100:
                        links = _extract_links_from_html(html, url, domain)
                        all_links.extend(links)
            except Exception as e:
                logger.warning(f"Error reading {parquet_file}: {e}")

    return all_links


async def _create_links_to(
    driver: AsyncDriver,
    client_id: str,
    links: list[dict],
    batch_size: int = 500,
) -> int:
    """Create :LINKS_TO relationships in Neo4j (only between existing :Page nodes)."""
    total = 0
    for i in range(0, len(links), batch_size):
        batch = links[i: i + batch_size]
        async with driver.session() as session:
            result = await session.run(
                """
                UNWIND $batch AS link
                MATCH (source:Page {client_id: $client_id, url: link.source_url})
                MATCH (target:Page {client_id: $client_id, url: link.target_url})
                MERGE (source)-[r:LINKS_TO]->(target)
                SET r.location = link.location,
                    r.weight = link.weight,
                    r.anchor_text = link.anchor_text
                RETURN count(r) AS cnt
                """,
                client_id=client_id,
                batch=batch,
            )
            record = await result.single()
            cnt = record["cnt"] if record else 0
            total += cnt
    return total


async def _calculate_pagerank(driver: AsyncDriver, client_id: str) -> int:
    """Calculate PageRank using iterative Cypher (no GDS required)."""
    # Initialize all pages with equal rank
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (p:Page {client_id: $client_id})
            WITH count(p) AS total
            MATCH (p:Page {client_id: $client_id})
            SET p.pagerank = 1.0 / total
            RETURN total AS pages
            """,
            client_id=client_id,
        )
        record = await result.single()
        num_pages = record["pages"] if record else 0

    if num_pages == 0:
        return 0

    damping = 0.85
    iterations = 20

    for i in range(iterations):
        async with driver.session() as session:
            await session.run(
                """
                MATCH (p:Page {client_id: $client_id})
                OPTIONAL MATCH (source:Page {client_id: $client_id})-[r:LINKS_TO]->(p)
                WITH p,
                     CASE WHEN count(source) > 0
                          THEN sum(
                              source.pagerank * COALESCE(r.weight, 1.0) /
                              CASE WHEN size([(source)-[:LINKS_TO]->() | 1]) > 0
                                   THEN size([(source)-[:LINKS_TO]->() | 1])
                                   ELSE 1 END
                          )
                          ELSE 0 END AS incoming_rank
                SET p.pagerank = (1 - $damping) / $num_pages + $damping * incoming_rank
                """,
                client_id=client_id,
                damping=damping,
                num_pages=num_pages,
            )

    return num_pages


async def _calculate_hits(driver: AsyncDriver, client_id: str) -> int:
    """Calculate HITS (Hub/Authority) scores."""
    # Initialize
    async with driver.session() as session:
        await session.run(
            """
            MATCH (p:Page {client_id: $client_id})
            SET p.hub_score = 1.0, p.authority_score = 1.0
            """,
            client_id=client_id,
        )

    iterations = 20
    for i in range(iterations):
        async with driver.session() as session:
            # Update authority: sum of hub scores of pages linking TO this page
            await session.run(
                """
                MATCH (p:Page {client_id: $client_id})
                OPTIONAL MATCH (source:Page {client_id: $client_id})-[:LINKS_TO]->(p)
                WITH p, CASE WHEN count(source) > 0 THEN sum(source.hub_score) ELSE 0 END AS auth
                SET p.authority_score = auth
                """,
                client_id=client_id,
            )
            # Update hub: sum of authority scores of pages this page links TO
            await session.run(
                """
                MATCH (p:Page {client_id: $client_id})
                OPTIONAL MATCH (p)-[:LINKS_TO]->(target:Page {client_id: $client_id})
                WITH p, CASE WHEN count(target) > 0 THEN sum(target.authority_score) ELSE 0 END AS hub
                SET p.hub_score = hub
                """,
                client_id=client_id,
            )
            # Normalize
            await session.run(
                """
                MATCH (p:Page {client_id: $client_id})
                WITH max(p.authority_score) AS max_auth, max(p.hub_score) AS max_hub
                MATCH (p:Page {client_id: $client_id})
                SET p.authority_score = CASE WHEN max_auth > 0 THEN p.authority_score / max_auth ELSE 0 END,
                    p.hub_score = CASE WHEN max_hub > 0 THEN p.hub_score / max_hub ELSE 0 END
                """,
                client_id=client_id,
            )

    async with driver.session() as session:
        result = await session.run(
            "MATCH (p:Page {client_id: $client_id}) WHERE p.authority_score > 0 RETURN count(p) AS cnt",
            client_id=client_id,
        )
        record = await result.single()
        return record["cnt"] if record else 0


async def run_link_extraction(
    neo4j: Neo4jClient,
    client_id: str,
    crawl_dir: str,
    domain: str,
    date_partition: Optional[str] = None,
    calculate_scores: bool = True,
) -> dict:
    """
    Main entrypoint: extract links from HTML, create LINKS_TO, calculate PageRank/HITS.

    Args:
        neo4j: Neo4jClient instance
        client_id: Client UUID
        crawl_dir: Path to crawl data directory (inside container)
        domain: Site domain for filtering internal links
        date_partition: Optional date filter
        calculate_scores: Whether to calculate PageRank and HITS
    """
    driver = neo4j._driver
    if driver is None:
        await neo4j.connect()
        driver = neo4j._driver

    result = LinkExtractionResult(client_id=client_id)

    try:
        # 1. Extract links from HTML
        all_links = _read_and_extract_links(crawl_dir, domain, date_partition)
        result.links_extracted = len(all_links)
        logger.info(f"Extracted {len(all_links)} internal links from HTML")

        # Count unique source pages
        source_pages = {l["source_url"] for l in all_links}
        result.pages_processed = len(source_pages)

        if not all_links:
            result.errors.append("No internal links found in HTML")
            return asdict(result)

        # 2. Remove existing LINKS_TO for this client
        async with driver.session() as session:
            await session.run(
                "MATCH (p:Page {client_id: $client_id})-[r:LINKS_TO]->() DELETE r",
                client_id=client_id,
            )

        # 3. Create LINKS_TO relationships
        result.links_to_created = await _create_links_to(driver, client_id, all_links)
        logger.info(f"Created {result.links_to_created} LINKS_TO relationships")

        # 4. Calculate PageRank and HITS
        if calculate_scores and result.links_to_created > 0:
            pages_ranked = await _calculate_pagerank(driver, client_id)
            logger.info(f"PageRank calculated for {pages_ranked} pages")

            pages_hits = await _calculate_hits(driver, client_id)
            logger.info(f"HITS calculated, {pages_hits} pages with authority > 0")

            result.pagerank_calculated = True

    except Exception as e:
        logger.error(f"Link extraction error: {e}", exc_info=True)
        result.errors.append(str(e))

    return asdict(result)
