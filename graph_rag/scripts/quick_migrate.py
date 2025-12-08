#!/usr/bin/env python3
"""
Quick migration script for WebKnoGraph data to Graph-RAG.
"""

import asyncio
import hashlib
import re
import os
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


async def main():
    from graph_rag.config.settings import get_settings
    from graph_rag.db.supabase_client import SupabaseClient
    from graph_rag.db.neo4j_client import Neo4jClient
    from graph_rag.services.embedding_service import EmbeddingService

    settings = get_settings()

    print("=" * 60)
    print("WebKnoGraph -> Graph-RAG Migration")
    print("=" * 60)

    # Connect to databases
    print("\n1. Connecting to databases...")
    supabase = SupabaseClient(settings)
    await supabase.connect()
    print("   Supabase: OK")

    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    await neo4j.connect()
    print("   Neo4j: OK")

    # Create client
    print("\n2. Creating client...")
    try:
        client = await supabase.create_client("Kalicube", "kalicube.com")
        client_id = str(client["id"])
        print(f"   Client ID: {client_id}")
        print(f"   API Key: {client['api_key']}")
        print("\n   *** SAVE THIS API KEY ***\n")
    except Exception as e:
        if "unique" in str(e).lower():
            print("   Client already exists, getting ID...")
            clients = await supabase.list_clients()
            client = next((c for c in clients if c["domain"] == "kalicube.com"), None)
            client_id = str(client["id"])
            print(f"   Client ID: {client_id}")
        else:
            raise

    # Load embedding model
    print("\n3. Loading embedding model (this may take a while first time)...")
    embedding_service = EmbeddingService(settings)
    # Warm up model
    _ = embedding_service.embed_query("test")
    print(f"   Model: {settings.embedding_model_name}")
    print("   Model loaded OK")

    # Load PageRank data
    print("\n4. Loading PageRank data...")
    analysis_df = pd.read_csv(project_root / "data" / "url_analysis_results.csv")
    url_metrics = {}
    for _, row in analysis_df.iterrows():
        url_metrics[row["URL"]] = {
            "pagerank": row.get("PageRank", 0.0),
            "folder_depth": int(row.get("Folder_Depth", 0)),
        }
    print(f"   Loaded {len(url_metrics)} URL metrics")

    # Load and process crawled content
    print("\n5. Loading crawled content...")
    parquet_dir = project_root / "data" / "crawled_data_parquet" / "crawl_date=2025-06-28"
    parquet_files = sorted(glob(str(parquet_dir / "*.parquet")))
    print(f"   Found {len(parquet_files)} parquet files")

    # Process in batches
    pages_migrated = 0
    embeddings_generated = 0
    batch_size = 10  # Small batches for progress visibility

    print("\n6. Migrating pages with embeddings...")

    all_urls = set()  # Track all URLs for link migration

    for pq_file in tqdm(parquet_files, desc="Processing files"):
        try:
            df = pd.read_parquet(pq_file)
        except Exception as e:
            print(f"   Error reading {pq_file}: {e}")
            continue

        for _, row in df.iterrows():
            url = row.get("URL", "")
            content = row.get("Content", "")

            if not url or not content:
                continue

            all_urls.add(url)

            # Get metrics
            metrics = url_metrics.get(url, {"pagerank": 0.0, "folder_depth": 0})

            # Extract title
            title = None
            title_match = re.search(r"<title[^>]*>([^<]+)</title>", content, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()[:500]

            # Extract text for embedding (simple approach)
            from trafilatura import extract
            clean_text = extract(content) or ""

            if len(clean_text) < 100:
                clean_text = re.sub(r'<[^>]+>', ' ', content)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            # Generate embedding
            embedding = None
            if len(clean_text) > 100:
                try:
                    truncated = clean_text[:8000]
                    embedding = embedding_service.embed_document(truncated)
                    embeddings_generated += 1
                except Exception as e:
                    print(f"   Embedding error for {url}: {e}")

            # Content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Insert page
            try:
                await supabase.upsert_page(
                    client_id=client_id,
                    url=url,
                    title=title,
                    content=clean_text[:50000] if clean_text else None,
                    content_hash=content_hash,
                    embedding=embedding,
                    pagerank=metrics["pagerank"],
                    hub_score=0.0,
                    authority_score=0.0,
                    folder_depth=metrics["folder_depth"],
                )

                # Also insert to Neo4j
                await neo4j.upsert_page(
                    client_id=client_id,
                    url=url,
                    title=title,
                    pagerank=metrics["pagerank"],
                    hub_score=0.0,
                    authority_score=0.0,
                    folder_depth=metrics["folder_depth"],
                )

                pages_migrated += 1
            except Exception as e:
                print(f"   Error inserting {url}: {e}")

    print(f"\n   Pages migrated: {pages_migrated}")
    print(f"   Embeddings generated: {embeddings_generated}")

    # Load and migrate links
    print("\n7. Migrating links...")
    links_df = pd.read_csv(project_root / "data" / "link_graph_edges.csv")
    print(f"   Found {len(links_df)} links")

    # Get URL to page ID mapping
    url_to_id = await supabase.get_url_to_page_id_mapping(client_id)
    print(f"   URL mappings: {len(url_to_id)}")

    links_migrated = 0
    links_batch = []

    for _, row in tqdm(links_df.iterrows(), total=len(links_df), desc="Migrating links"):
        source_url = row.get("FROM", "")
        target_url = row.get("TO", "")

        if not source_url or not target_url:
            continue

        source_id = url_to_id.get(source_url)
        target_id = url_to_id.get(target_url)

        # Insert to Supabase if both exist
        if source_id and target_id:
            try:
                await supabase.upsert_link(client_id, source_id, target_id)
            except:
                pass

        # Insert to Neo4j (will skip if nodes don't exist)
        links_batch.append({
            "source_url": source_url,
            "target_url": target_url,
            "anchor_text": None,
        })

        if len(links_batch) >= 500:
            try:
                await neo4j.create_links_batch(client_id, links_batch)
            except:
                pass
            links_migrated += len(links_batch)
            links_batch = []

    # Remaining links
    if links_batch:
        try:
            await neo4j.create_links_batch(client_id, links_batch)
        except:
            pass
        links_migrated += len(links_batch)

    print(f"   Links migrated: {links_migrated}")

    # Summary
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"Client ID: {client_id}")
    print(f"Pages: {pages_migrated}")
    print(f"Embeddings: {embeddings_generated}")
    print(f"Links: {links_migrated}")
    print("=" * 60)

    # Cleanup
    await supabase.disconnect()
    await neo4j.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
