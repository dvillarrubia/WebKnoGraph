#!/usr/bin/env python3
"""
Ingest Crawl4AI data into Graph-RAG system.
- Reads parquet files from crawl
- Generates embeddings with multilingual-e5-large
- Inserts into Supabase (pages + embeddings)
- Inserts into Neo4j (link graph)
- Deletes parquet files after successful migration
"""

import asyncio
import argparse
import shutil
import sys
from pathlib import Path
from glob import glob

import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / "graph_rag" / ".env")


async def ingest_crawl_data(
    crawl_path: str,
    client_name: str,
    client_domain: str,
    delete_after: bool = True,
    batch_size: int = 10,
):
    """Ingest crawled data into Graph-RAG databases."""

    from graph_rag.config.settings import get_settings
    from graph_rag.db.supabase_client import SupabaseClient
    from graph_rag.db.neo4j_client import Neo4jClient
    from graph_rag.services.embedding_service import EmbeddingService

    settings = get_settings()
    crawl_dir = Path(crawl_path)

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║              Graph-RAG Data Ingestion                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Client:        {client_name:<50} ║
║  Domain:        {client_domain:<50} ║
║  Crawl Path:    {str(crawl_dir):<50} ║
║  Delete After:  {str(delete_after):<50} ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # =========================================================================
    # 1. Connect to databases
    # =========================================================================
    print("[1/6] Connecting to databases...")

    supabase = SupabaseClient(settings)
    await supabase.connect()
    print("  ✓ Supabase connected")

    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    await neo4j.connect()
    print("  ✓ Neo4j connected")

    # =========================================================================
    # 2. Create or get client
    # =========================================================================
    print(f"\n[2/6] Setting up client '{client_name}'...")

    try:
        client = await supabase.create_client(client_name, client_domain)
        client_id = str(client["id"])
        print(f"  ✓ Client created")
        print(f"  → ID: {client_id}")
        print(f"  → API Key: {client['api_key']}")
        print(f"\n  ⚠️  SAVE THIS API KEY - IT WON'T BE SHOWN AGAIN!\n")
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            print(f"  Client already exists, retrieving...")
            clients = await supabase.list_clients()
            client = next((c for c in clients if c["domain"] == client_domain), None)
            if not client:
                raise ValueError(f"Client with domain {client_domain} not found")
            client_id = str(client["id"])
            print(f"  → Using existing client ID: {client_id}")
        else:
            raise

    # =========================================================================
    # 3. Load embedding model
    # =========================================================================
    print(f"\n[3/6] Loading embedding model...")
    print(f"  Model: {settings.embedding_model_name}")

    embedding_service = EmbeddingService(settings)
    # Warm up
    _ = embedding_service.embed_query("test")
    print("  ✓ Model loaded")

    # =========================================================================
    # 4. Load and process pages
    # =========================================================================
    print(f"\n[4/6] Processing pages...")

    pages_pattern = str(crawl_dir / "pages" / "**" / "*.parquet")
    pages_files = glob(pages_pattern, recursive=True)

    if not pages_files:
        # Try alternative structure
        pages_pattern = str(crawl_dir / "**" / "*.parquet")
        pages_files = [f for f in glob(pages_pattern, recursive=True) if "links" not in f]

    print(f"  Found {len(pages_files)} page files")

    pages_migrated = 0
    embeddings_generated = 0
    errors = []

    for pf in pages_files:
        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            errors.append(f"Error reading {pf}: {e}")
            continue

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {Path(pf).name}"):
            url = row.get("url", "")
            markdown = row.get("markdown", "")
            title = row.get("title", "")
            content_hash = row.get("content_hash", "")
            word_count = row.get("word_count", 0)

            if not url:
                continue

            # Skip pages with very little content
            if word_count < 50:
                continue

            # Generate embedding
            embedding = None
            if markdown and len(markdown) > 100:
                try:
                    # Truncate for embedding (max ~8000 chars)
                    text_for_embedding = markdown[:8000]
                    embedding = embedding_service.embed_document(text_for_embedding)
                    embeddings_generated += 1
                except Exception as e:
                    errors.append(f"Embedding error for {url}: {e}")

            # Insert into Supabase
            try:
                await supabase.upsert_page(
                    client_id=client_id,
                    url=url,
                    title=title[:500] if title else None,
                    content=markdown[:50000] if markdown else None,  # Store full markdown
                    content_hash=content_hash,
                    embedding=embedding,
                    pagerank=0.0,  # Will be calculated later
                    hub_score=0.0,
                    authority_score=0.0,
                    folder_depth=url.count('/') - 2,  # Approximate depth
                )
                pages_migrated += 1
            except Exception as e:
                errors.append(f"DB insert error for {url}: {e}")

            # Also insert to Neo4j (just the node)
            try:
                await neo4j.upsert_page(
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

    print(f"\n  ✓ Pages migrated: {pages_migrated}")
    print(f"  ✓ Embeddings generated: {embeddings_generated}")

    # =========================================================================
    # 5. Load and process links
    # =========================================================================
    print(f"\n[5/6] Processing links...")

    links_pattern = str(crawl_dir / "links" / "**" / "*.parquet")
    links_files = glob(links_pattern, recursive=True)

    print(f"  Found {len(links_files)} link files")

    # Get URL to page ID mapping
    url_to_id = await supabase.get_url_to_page_id_mapping(client_id)
    print(f"  URL mappings available: {len(url_to_id)}")

    links_migrated = 0
    links_batch = []

    for lf in links_files:
        try:
            df = pd.read_parquet(lf)
        except Exception as e:
            errors.append(f"Error reading {lf}: {e}")
            continue

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {Path(lf).name}"):
            source_url = row.get("source_url", "")
            target_url = row.get("target_url", "")
            anchor_text = row.get("anchor_text", "")

            if not source_url or not target_url:
                continue

            source_id = url_to_id.get(source_url)
            target_id = url_to_id.get(target_url)

            # Insert to Supabase if both pages exist
            if source_id and target_id:
                try:
                    await supabase.upsert_link(
                        client_id, source_id, target_id, anchor_text[:200] if anchor_text else None
                    )
                except:
                    pass

            # Batch for Neo4j
            links_batch.append({
                "source_url": source_url,
                "target_url": target_url,
                "anchor_text": anchor_text[:200] if anchor_text else None,
            })

            if len(links_batch) >= 500:
                try:
                    await neo4j.create_links_batch(client_id, links_batch)
                    links_migrated += len(links_batch)
                except Exception as e:
                    errors.append(f"Neo4j batch error: {e}")
                links_batch = []

    # Remaining links
    if links_batch:
        try:
            await neo4j.create_links_batch(client_id, links_batch)
            links_migrated += len(links_batch)
        except:
            pass

    print(f"\n  ✓ Links migrated: {links_migrated}")

    # =========================================================================
    # 6. Cleanup
    # =========================================================================
    if delete_after and pages_migrated > 0:
        print(f"\n[6/6] Cleaning up crawl data...")
        try:
            shutil.rmtree(crawl_dir)
            print(f"  ✓ Deleted {crawl_dir}")
        except Exception as e:
            print(f"  ⚠ Could not delete {crawl_dir}: {e}")
    else:
        print(f"\n[6/6] Keeping crawl data at {crawl_dir}")

    # Disconnect
    await supabase.disconnect()
    await neo4j.disconnect()

    # Summary
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    INGESTION COMPLETE                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Client ID:       {client_id:<47} ║
║  Pages migrated:  {pages_migrated:<47} ║
║  Embeddings:      {embeddings_generated:<47} ║
║  Links migrated:  {links_migrated:<47} ║
║  Errors:          {len(errors):<47} ║
╚══════════════════════════════════════════════════════════════════╝
""")

    if errors:
        print(f"\nFirst 10 errors:")
        for err in errors[:10]:
            print(f"  - {err}")

    return {
        "client_id": client_id,
        "pages_migrated": pages_migrated,
        "embeddings_generated": embeddings_generated,
        "links_migrated": links_migrated,
        "errors": len(errors),
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest Crawl4AI data into Graph-RAG")
    parser.add_argument("--crawl-path", required=True, help="Path to crawled data")
    parser.add_argument("--client-name", required=True, help="Client name")
    parser.add_argument("--client-domain", required=True, help="Client domain")
    parser.add_argument("--keep-files", action="store_true", help="Don't delete parquet files after ingestion")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for operations")

    args = parser.parse_args()

    result = asyncio.run(ingest_crawl_data(
        crawl_path=args.crawl_path,
        client_name=args.client_name,
        client_domain=args.client_domain,
        delete_after=not args.keep_files,
        batch_size=args.batch_size,
    ))

    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
