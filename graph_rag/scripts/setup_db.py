#!/usr/bin/env python3
"""
Database setup script for Graph-RAG.
Runs SQL migrations on Supabase and Cypher constraints on Neo4j.
"""

import asyncio
import asyncpg
from pathlib import Path
from neo4j import AsyncGraphDatabase


async def setup_supabase(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
):
    """Run SQL migrations on Supabase/PostgreSQL."""
    print("Setting up Supabase schema...")

    conn = await asyncpg.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
    )

    # Read and execute migration file
    migration_path = Path(__file__).parent.parent / "db" / "migrations" / "001_create_schema.sql"
    sql = migration_path.read_text()

    try:
        await conn.execute(sql)
        print("Supabase schema created successfully!")
    except Exception as e:
        print(f"Error creating schema: {e}")
        raise
    finally:
        await conn.close()


async def setup_neo4j(uri: str, user: str, password: str):
    """Run Cypher constraints on Neo4j."""
    print("Setting up Neo4j constraints...")

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    # Read constraints file
    constraints_path = Path(__file__).parent.parent / "db" / "neo4j" / "001_constraints.cypher"
    cypher_content = constraints_path.read_text()

    # Extract individual statements (skip comments)
    statements = []
    for line in cypher_content.split("\n"):
        line = line.strip()
        if line and not line.startswith("//"):
            statements.append(line)

    async with driver.session() as session:
        for stmt in statements:
            if stmt.startswith("CREATE"):
                try:
                    await session.run(stmt)
                    print(f"  Executed: {stmt[:60]}...")
                except Exception as e:
                    print(f"  Warning: {e}")

    await driver.close()
    print("Neo4j constraints created successfully!")


async def main():
    """Main setup function."""
    import os
    from dotenv import load_dotenv

    # Load environment variables
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    # Supabase setup
    await setup_supabase(
        host=os.getenv("SUPABASE_DB_HOST", "localhost"),
        port=int(os.getenv("SUPABASE_DB_PORT", 54322)),
        database=os.getenv("SUPABASE_DB_NAME", "postgres"),
        user=os.getenv("SUPABASE_DB_USER", "postgres"),
        password=os.getenv("SUPABASE_DB_PASSWORD", ""),
    )

    # Neo4j setup
    await setup_neo4j(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
    )

    print("\nDatabase setup complete!")


if __name__ == "__main__":
    asyncio.run(main())
