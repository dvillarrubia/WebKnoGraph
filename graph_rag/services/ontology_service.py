"""
SEOntology Service — Wrapper async for ontologia module.
Provides ontology ingest, schema setup, cleanup, and stats for the dashboard.
"""

import logging
from dataclasses import asdict

from graph_rag.db.neo4j_client import Neo4jClient
from ontologia.ingest_ontology import SeoOntologyIngestor, SeoIngestResult
from ontologia.neo4j_schema import setup_seo_schema, cleanup_seo_data

logger = logging.getLogger(__name__)


async def run_seo_ingest(
    neo4j: Neo4jClient,
    client_id: str,
    crawl_dir: str,
    domain: str,
    refresh: bool = False,
    with_chunks: bool = False,
    supabase_client=None,
) -> dict:
    """
    Run SEOntology ingest pipeline.

    Args:
        neo4j: Neo4jClient instance
        client_id: Client UUID
        crawl_dir: Path to crawl data directory
        domain: Site domain (e.g. www.uoc.edu)
        refresh: If True, cleanup existing SEO data before ingest
        with_chunks: If True, also ingest chunk data from Supabase
        supabase_client: Optional SupabaseClient for chunk data
    """
    driver = neo4j._driver
    if driver is None:
        await neo4j.connect()
        driver = neo4j._driver

    # Cleanup first if refresh requested
    if refresh:
        await cleanup_seo_data(driver, client_id)
        logger.info(f"Cleaned up SEO data for client {client_id}")

    # Load Supabase data if chunks requested
    supabase_chunks = None
    if with_chunks and supabase_client:
        try:
            async with supabase_client.get_connection() as conn:
                chunks_result = await conn.fetch(
                    """
                    SELECT c.id, c.content, c.chunk_index, c.heading_context,
                           c.char_start, c.char_end, p.url
                    FROM rag_chunks c
                    JOIN rag_pages p ON c.page_id = p.id
                    WHERE p.client_id = $1
                    """,
                    client_id,
                )
            supabase_chunks = [dict(r) for r in chunks_result]
            logger.info(f"Loaded {len(supabase_chunks)} chunks from Supabase")
        except Exception as e:
            logger.warning(f"Could not load chunks from Supabase: {e}")

    # Run ingest
    ingestor = SeoOntologyIngestor(
        driver=driver,
        client_id=client_id,
        domain=domain,
    )

    result: SeoIngestResult = await ingestor.ingest_from_parquet(
        crawl_dir=crawl_dir,
        supabase_chunks=supabase_chunks,
    )

    return asdict(result)


async def run_schema_setup(neo4j: Neo4jClient) -> dict:
    """Create all SEOntology constraints and indexes."""
    driver = neo4j._driver
    if driver is None:
        await neo4j.connect()
        driver = neo4j._driver

    return await setup_seo_schema(driver)


async def run_cleanup(neo4j: Neo4jClient, client_id: str) -> dict:
    """Remove SEOntology enrichment for a client."""
    driver = neo4j._driver
    if driver is None:
        await neo4j.connect()
        driver = neo4j._driver

    return await cleanup_seo_data(driver, client_id)


async def get_seo_stats(neo4j: Neo4jClient, client_id: str) -> dict:
    """Get counts of SEOntology nodes for a client."""
    labels = [
        "SeoWebPage", "SeoChunk", "SeoURL", "SeoLink",
        "SeoLinkGroup", "SeoAnchorText", "SeoSchema", "SeoThing",
    ]
    stats = {}
    async with neo4j.get_session() as session:
        for label in labels:
            result = await session.run(
                f"MATCH (n:{label} {{client_id: $client_id}}) RETURN count(n) AS cnt",
                client_id=client_id,
            )
            record = await result.single()
            stats[label] = record["cnt"] if record else 0
    return stats
