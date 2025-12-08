#!/usr/bin/env python3
"""
Data migration script for Graph-RAG.
Migrates existing WebKnoGraph data to the new multi-tenant system.
"""

import asyncio
import argparse
from pathlib import Path


async def main():
    """Main migration function."""
    import os
    import sys
    from dotenv import load_dotenv

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    # Load environment variables
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Migrate WebKnoGraph data to Graph-RAG")
    parser.add_argument(
        "--name",
        required=True,
        help="Client name (e.g., 'Mi Empresa')",
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Client domain (e.g., 'miempresa.com')",
    )
    parser.add_argument(
        "--data-path",
        default=str(project_root / "data"),
        help="Path to WebKnoGraph data folder",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip regenerating embeddings (use existing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for DB operations",
    )
    args = parser.parse_args()

    # Import after path setup
    from graph_rag.config.settings import get_settings
    from graph_rag.db.supabase_client import SupabaseClient
    from graph_rag.db.neo4j_client import Neo4jClient
    from graph_rag.services.embedding_service import EmbeddingService
    from graph_rag.services.migration_service import MigrationService

    settings = get_settings()

    # Initialize clients
    print("Connecting to databases...")
    supabase = SupabaseClient(settings)
    await supabase.connect()

    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    await neo4j.connect()

    # Initialize embedding service
    print(f"Loading embedding model: {settings.embedding_model_name}")
    embedding_service = EmbeddingService(settings)

    # Initialize migration service
    migration_service = MigrationService(
        settings=settings,
        supabase_client=supabase,
        neo4j_client=neo4j,
        embedding_service=embedding_service,
    )

    # Create client
    print(f"\nCreating client: {args.name} ({args.domain})")
    try:
        client = await migration_service.create_client_from_domain(
            name=args.name,
            domain=args.domain,
        )
        print(f"Client created with ID: {client['id']}")
        print(f"API Key: {client['api_key']}")
        print("\n*** SAVE THIS API KEY - IT WON'T BE SHOWN AGAIN ***\n")
    except Exception as e:
        if "unique" in str(e).lower():
            print(f"Client for domain {args.domain} already exists")
            # Get existing client
            clients = await supabase.list_clients()
            client = next((c for c in clients if c["domain"] == args.domain), None)
            if not client:
                raise
            print(f"Using existing client ID: {client['id']}")
        else:
            raise

    # Run migration
    print(f"\nMigrating data from: {args.data_path}")
    print(f"Regenerate embeddings: {not args.skip_embeddings}")
    print(f"Batch size: {args.batch_size}")

    stats = await migration_service.migrate_from_webknograph(
        client_id=str(client["id"]),
        data_path=args.data_path,
        regenerate_embeddings=not args.skip_embeddings,
        batch_size=args.batch_size,
    )

    # Print results
    print("\n" + "=" * 50)
    print("MIGRATION COMPLETE")
    print("=" * 50)
    print(f"Pages migrated:       {stats['pages_migrated']}")
    print(f"Links migrated:       {stats['links_migrated']}")
    print(f"Embeddings generated: {stats['embeddings_generated']}")

    if stats["errors"]:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats["errors"][:10]:
            print(f"  - {error}")
        if len(stats["errors"]) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more")

    # Cleanup
    await supabase.disconnect()
    await neo4j.disconnect()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
