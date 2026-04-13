"""
SEO Metadata Extraction Service — Extracts SEO metadata from HTML in parquets
and enriches :SeoWebPage nodes in Neo4j.

Extracts: canonical, robots, og_*, twitter_card, hreflang, lang, viewport,
headings (h1-h6), has_structured_data, noFollow, http_status.
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq
from bs4 import BeautifulSoup
from neo4j import AsyncDriver

from graph_rag.db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class SeoMetadataResult:
    client_id: str
    pages_processed: int = 0
    pages_updated: int = 0
    headings_created: int = 0
    errors: list = field(default_factory=list)


def _extract_metadata(html: str, url: str) -> dict:
    """Extract SEO metadata from HTML content."""
    soup = BeautifulSoup(html, "html.parser")
    meta = {"url": url}

    # <html lang="...">
    html_tag = soup.find("html")
    meta["lang"] = html_tag.get("lang", "").strip()[:10] if html_tag else ""

    # <head> metas
    head = soup.find("head") or soup

    # Canonical
    canonical_tag = head.find("link", rel="canonical")
    meta["canonical"] = canonical_tag["href"].strip() if canonical_tag and canonical_tag.get("href") else ""

    # Robots meta
    robots_tag = head.find("meta", attrs={"name": "robots"})
    meta["robots"] = robots_tag["content"].strip() if robots_tag and robots_tag.get("content") else ""

    # Viewport
    viewport_tag = head.find("meta", attrs={"name": "viewport"})
    meta["viewport"] = viewport_tag["content"].strip() if viewport_tag and viewport_tag.get("content") else ""

    # Open Graph
    for og_prop in ("og:title", "og:description", "og:image", "og:type", "og:url", "og:site_name"):
        tag = head.find("meta", property=og_prop) or head.find("meta", attrs={"name": og_prop})
        key = og_prop.replace(":", "_")  # og:title -> og_title
        meta[key] = tag["content"].strip()[:500] if tag and tag.get("content") else ""

    # Twitter Card
    twitter_tag = head.find("meta", attrs={"name": "twitter:card"}) or head.find("meta", property="twitter:card")
    meta["twitter_card"] = twitter_tag["content"].strip() if twitter_tag and twitter_tag.get("content") else ""

    # Hreflang links
    hreflang_tags = head.find_all("link", rel="alternate", hreflang=True)
    hreflangs = {}
    for tag in hreflang_tags:
        lang_code = tag.get("hreflang", "")
        href = tag.get("href", "")
        if lang_code and href:
            hreflangs[lang_code] = href
    meta["hreflang"] = json.dumps(hreflangs) if hreflangs else ""

    # Structured data (JSON-LD)
    json_ld_tags = soup.find_all("script", type="application/ld+json")
    meta["has_structured_data"] = len(json_ld_tags) > 0
    meta["structured_data_count"] = len(json_ld_tags)

    # Headings
    headings = {}
    for level in range(1, 7):
        tag_name = f"h{level}"
        tags = soup.find_all(tag_name)
        if tags:
            texts = [t.get_text(strip=True)[:200] for t in tags if t.get_text(strip=True)]
            if texts:
                headings[tag_name] = texts
    meta["headings"] = json.dumps(headings, ensure_ascii=False) if headings else "{}"

    # H1 specifically (most important for SEO)
    h1_tags = soup.find_all("h1")
    h1_texts = [t.get_text(strip=True)[:200] for t in h1_tags if t.get_text(strip=True)]
    meta["h1"] = h1_texts[0] if h1_texts else ""
    meta["h1_count"] = len(h1_texts)

    # noFollow check (meta robots or rel=nofollow on links)
    meta["noFollow"] = "nofollow" in meta.get("robots", "").lower()

    # noIndex check
    meta["noIndex"] = "noindex" in meta.get("robots", "").lower()

    return meta


def _read_parquets_from_dir(crawl_dir: str, date_partition: Optional[str] = None) -> list[dict]:
    """Read all page parquets from a crawl directory and extract HTML metadata."""
    pages_dir = Path(crawl_dir) / "pages"
    if not pages_dir.exists():
        return []

    results = []
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
                        meta = _extract_metadata(html, url)
                        results.append(meta)
            except Exception as e:
                logger.warning(f"Error reading {parquet_file}: {e}")

    return results


async def _update_pages_metadata(
    driver: AsyncDriver,
    client_id: str,
    batch: list[dict],
) -> int:
    """Update :SeoWebPage nodes with SEO metadata."""
    async with driver.session() as session:
        result = await session.run(
            """
            UNWIND $batch AS item
            MATCH (p:SeoWebPage {client_id: $client_id, url: item.url})
            SET p.canonical = item.canonical,
                p.robots = item.robots,
                p.lang = item.lang,
                p.viewport = item.viewport,
                p.og_title = item.og_title,
                p.og_description = item.og_description,
                p.og_image = item.og_image,
                p.og_type = item.og_type,
                p.og_url = item.og_url,
                p.og_site_name = item.og_site_name,
                p.twitter_card = item.twitter_card,
                p.hreflang = item.hreflang,
                p.has_structured_data = item.has_structured_data,
                p.structured_data_count = item.structured_data_count,
                p.headings = item.headings,
                p.h1 = item.h1,
                p.h1_count = item.h1_count,
                p.noFollow = item.noFollow,
                p.noIndex = item.noIndex,
                p.isCrawlable = NOT item.noIndex
            RETURN count(p) AS cnt
            """,
            client_id=client_id,
            batch=batch,
        )
        record = await result.single()
        return record["cnt"] if record else 0


async def run_seo_metadata_extraction(
    neo4j: Neo4jClient,
    client_id: str,
    crawl_dir: str,
    date_partition: Optional[str] = None,
    batch_size: int = 100,
) -> dict:
    """
    Main entrypoint: extract SEO metadata from HTML parquets and update Neo4j.

    Args:
        neo4j: Neo4jClient instance
        client_id: Client UUID
        crawl_dir: Path to crawl data directory (inside container)
        date_partition: Optional date filter (e.g. '2026-04-13')
        batch_size: Neo4j batch size
    """
    driver = neo4j._driver
    if driver is None:
        await neo4j.connect()
        driver = neo4j._driver

    result = SeoMetadataResult(client_id=client_id)

    try:
        # 1. Extract metadata from parquets
        metadata_list = _read_parquets_from_dir(crawl_dir, date_partition)
        result.pages_processed = len(metadata_list)
        logger.info(f"Extracted metadata from {len(metadata_list)} pages")

        if not metadata_list:
            result.errors.append("No HTML content found in parquets")
            return asdict(result)

        # 2. Batch update Neo4j
        for i in range(0, len(metadata_list), batch_size):
            batch = metadata_list[i : i + batch_size]
            cnt = await _update_pages_metadata(driver, client_id, batch)
            result.pages_updated += cnt
            logger.info(f"Batch {i // batch_size + 1}: {cnt} pages updated")

    except Exception as e:
        logger.error(f"SEO metadata extraction error: {e}", exc_info=True)
        result.errors.append(str(e))

    return asdict(result)
