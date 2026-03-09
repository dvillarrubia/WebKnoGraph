"""
CLI runner for SEOntology ingestion.

Usage:
    # Parse and print ontology summary
    python -m ontologia.run_ingest --parse-only

    # Ingest UOC data into Neo4j SEOntology nodes
    python -m ontologia.run_ingest --ingest --crawl-dir data/crawl4ai_data/www_uoc_edu

    # Full refresh (delete existing Seo* nodes first)
    python -m ontologia.run_ingest --ingest --crawl-dir data/crawl4ai_data/www_uoc_edu --refresh

    # Ingest with Supabase chunks
    python -m ontologia.run_ingest --ingest --crawl-dir data/crawl4ai_data/www_uoc_edu --with-chunks

    # Extract JSON-LD from HTML (Sprint 2)
    python -m ontologia.run_ingest --extract-schema --crawl-dir data/crawl4ai_data/www_uoc_edu
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import AsyncGraphDatabase

from ontologia.ontology_parser import parse_seovoc, print_summary
from ontologia.neo4j_schema import setup_seo_schema, cleanup_seo_data
from ontologia.ingest_ontology import SeoOntologyIngestor


async def run_parse_only():
    """Parse seovoc.ttl and print summary."""
    onto = parse_seovoc()
    print_summary(onto)
    return onto


async def run_ingest(
    crawl_dir: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    client_id: str,
    domain: str,
    refresh: bool = False,
    with_chunks: bool = False,
    supabase_dsn: str | None = None,
):
    """Run SEOntology ingestion pipeline."""
    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        await driver.verify_connectivity()
        print(f"Connected to Neo4j at {neo4j_uri}")

        # Setup schema
        print("\n[1/4] Setting up SEOntology schema...")
        schema_result = setup_seo_schema(driver)
        schema_result = await schema_result
        print(f"  Created {schema_result['created']} constraints/indexes")
        if schema_result["errors"]:
            for err in schema_result["errors"]:
                print(f"  WARNING: {err}")

        # Optionally clean existing SEO enrichment (preserves :Page + PageRank)
        if refresh:
            print(f"\n[2/4] Cleaning SEO enrichment for client {client_id}...")
            print(f"       (preserves :Page nodes, :LINKS_TO, PageRank/HITS)")
            cleanup = await cleanup_seo_data(driver, client_id)
            print(f"  Cleaned {cleanup['cleaned']} steps")
            if cleanup["errors"]:
                for err in cleanup["errors"]:
                    print(f"  WARNING: {err}")
        else:
            print("\n[2/4] Skipping cleanup (use --refresh to strip SEO labels)")

        # Load chunks from Supabase if requested
        chunks_data = None
        if with_chunks and supabase_dsn:
            print("\n[3/4] Loading chunks from Supabase...")
            chunks_data = await _load_chunks_from_supabase(supabase_dsn, client_id)
            print(f"  Loaded {len(chunks_data)} chunks")
        else:
            print("\n[3/4] Skipping Supabase chunks (use --with-chunks --supabase-dsn)")

        # Run ingestion
        print(f"\n[4/4] Ingesting from {crawl_dir}...")
        ingestor = SeoOntologyIngestor(
            driver=driver,
            client_id=client_id,
            domain=domain,
        )
        result = await ingestor.ingest_from_parquet(
            crawl_dir=crawl_dir,
            supabase_chunks=chunks_data,
        )

        # Print results
        print(f"\n{'='*50}")
        print(f"SEOntology Ingestion Complete (FUSED)")
        print(f"{'='*50}")
        print(f"  Page:SeoWebPage (fused): {result.webpages}")
        print(f"  SeoURL nodes:            {result.urls}")
        print(f"  SeoChunk nodes:          {result.chunks}")
        print(f"  SeoLink nodes:           {result.links}")
        print(f"  SeoLinkGroup nodes:      {result.link_groups}")
        print(f"  SeoAnchorText nodes:     {result.anchor_texts}")
        print(f"  Relationships:           {result.relationships}")
        if result.errors:
            print(f"\n  Errors ({len(result.errors)}):")
            for err in result.errors[:10]:
                print(f"    - {err}")

    finally:
        await driver.close()


async def run_extract_schema(crawl_dir: str):
    """Extract JSON-LD schema from HTML content in parquet."""
    from ontologia.extractors.schema_extractor import extract_schemas_from_parquet

    crawl_path = Path(crawl_dir)
    results = extract_schemas_from_parquet(crawl_path / "pages")

    print(f"\nJSON-LD Extraction Results:")
    print(f"  Pages processed: {results['pages_processed']}")
    print(f"  Schemas found:   {results['schemas_found']}")

    for url, schemas in results.get("schemas", {}).items():
        print(f"\n  {url}:")
        for s in schemas:
            s_type = s.get("@type", "unknown")
            print(f"    - @type: {s_type}")


async def _load_chunks_from_supabase(dsn: str, client_id: str) -> list[dict]:
    """Load chunks with page URLs from Supabase."""
    import asyncpg
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT c.chunk_index, c.content, c.heading_context,
                   c.char_start, c.char_end, p.url
            FROM rag_chunks c
            JOIN rag_pages p ON p.id = c.page_id
            WHERE p.client_id = $1
            ORDER BY p.url, c.chunk_index
            """,
            client_id,
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(description="SEOntology Ingestion CLI")

    # Actions
    parser.add_argument("--parse-only", action="store_true",
                        help="Only parse seovoc.ttl and print summary")
    parser.add_argument("--ingest", action="store_true",
                        help="Run ingestion pipeline")
    parser.add_argument("--extract-schema", action="store_true",
                        help="Extract JSON-LD from HTML (Sprint 2)")

    # Data source
    parser.add_argument("--crawl-dir", type=str,
                        default="data/crawl4ai_data/www_uoc_edu",
                        help="Path to crawl data directory")

    # Neo4j connection
    parser.add_argument("--neo4j-uri", type=str,
                        default="bolt://localhost:7687",
                        help="Neo4j URI")
    parser.add_argument("--neo4j-user", type=str, default="neo4j")
    parser.add_argument("--neo4j-password", type=str, default="")

    # Client info
    parser.add_argument("--client-id", type=str, required=False,
                        help="Client UUID from rag_clients")
    parser.add_argument("--domain", type=str, default="www.uoc.edu",
                        help="Domain for internal link detection")

    # Options
    parser.add_argument("--refresh", action="store_true",
                        help="Delete existing Seo* nodes before ingestion")
    parser.add_argument("--with-chunks", action="store_true",
                        help="Load and ingest chunks from Supabase")
    parser.add_argument("--supabase-dsn", type=str,
                        help="Supabase/PostgreSQL DSN for chunk loading")

    args = parser.parse_args()

    if args.parse_only:
        asyncio.run(run_parse_only())
    elif args.extract_schema:
        asyncio.run(run_extract_schema(args.crawl_dir))
    elif args.ingest:
        if not args.neo4j_password:
            # Try loading from .env
            try:
                from graph_rag.config.settings import get_settings
                settings = get_settings()
                args.neo4j_uri = settings.neo4j_uri
                args.neo4j_user = settings.neo4j_user
                args.neo4j_password = settings.neo4j_password
            except Exception:
                parser.error("--neo4j-password required (or configure .env)")

        if not args.client_id:
            # Use a default for UOC
            args.client_id = "uoc-seo-ontology"
            print(f"Using default client_id: {args.client_id}")

        asyncio.run(run_ingest(
            crawl_dir=args.crawl_dir,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            client_id=args.client_id,
            domain=args.domain,
            refresh=args.refresh,
            with_chunks=args.with_chunks,
            supabase_dsn=args.supabase_dsn,
        ))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
