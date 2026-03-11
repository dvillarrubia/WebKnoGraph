"""
Neo4j Schema Setup for SEOntology nodes.

FUSION STRATEGY:
- :Page nodes (from graph_rag) get an ADDITIONAL :SeoWebPage label
- Same node holds both PageRank/HITS (graph_rag) and seovoc properties (ontologia)
- All existing graph_rag queries on :Page keep working
- New ontologia queries on :SeoWebPage also work
- Other seovoc classes (Chunk, Link, etc.) are new labels only used by ontologia
"""

from neo4j import AsyncDriver


# Neo4j label mapping: seovoc class -> Neo4j label
# NOTE: WebPage maps to dual-label Page:SeoWebPage (fusion with existing)
SEOVOC_LABELS = {
    "WebPage": "Page:SeoWebPage",  # DUAL LABEL — fused with graph_rag :Page
    "Chunk": "SeoChunk",
    "URL": "SeoURL",
    "Link": "SeoLink",
    "LinkGroup": "SeoLinkGroup",
    "AnchorText": "SeoAnchorText",
    "Query": "SeoQuery",
    "Schema": "SeoSchema",
    "Persona": "SeoPersona",
    "PageGroup": "SeoPageGroup",
    "Thing": "SeoThing",
}

# Relationship types mapping seovoc object properties -> Neo4j relationship types
# NOTE: LINKS_TO is kept from graph_rag (used by PageRank). New rels are additive.
SEOVOC_RELATIONSHIPS = {
    "hasURL": "HAS_URL",
    "hasChunk": "HAS_CHUNK",
    "hasLinkGroup": "HAS_LINK_GROUP",
    "hasLink": "HAS_LINK",
    "link": "HAS_DIRECT_LINK",
    "hasQuery": "HAS_QUERY",
    "hasPrimaryQuery": "HAS_PRIMARY_QUERY",
    "hasSchemaMarkup": "HAS_SCHEMA_MARKUP",
    "hasPersona": "HAS_PERSONA",
    "hasPage": "HAS_PAGE",
    "about": "ABOUT",
    "mentions": "MENTIONS",
    "anchorResource": "ANCHOR_RESOURCE",
    "influencedByQuery": "INFLUENCED_BY_QUERY",
    "inLanguage": "IN_LANGUAGE",
    # Existing (graph_rag) — kept as-is, not duplicated
    # "linksTo": "LINKS_TO",
}


SETUP_QUERIES = [
    # === Existing :Page constraints (graph_rag creates these, but ensure they exist) ===
    """CREATE CONSTRAINT page_unique_url IF NOT EXISTS
       FOR (p:Page) REQUIRE (p.client_id, p.url) IS UNIQUE""",

    """CREATE INDEX page_client_id IF NOT EXISTS
       FOR (p:Page) ON (p.client_id)""",

    """CREATE INDEX page_pagerank IF NOT EXISTS
       FOR (p:Page) ON (p.client_id, p.pagerank)""",

    # === New SEOntology constraints ===
    """CREATE CONSTRAINT seo_url_value IF NOT EXISTS
       FOR (n:SeoURL) REQUIRE (n.client_id, n.value) IS UNIQUE""",

    """CREATE CONSTRAINT seo_chunk_id IF NOT EXISTS
       FOR (n:SeoChunk) REQUIRE (n.client_id, n.page_url, n.chunkPosition) IS UNIQUE""",

    """CREATE CONSTRAINT seo_linkgroup_id IF NOT EXISTS
       FOR (n:SeoLinkGroup) REQUIRE (n.client_id, n.page_url, n.name) IS UNIQUE""",

    """CREATE CONSTRAINT seo_anchortext_id IF NOT EXISTS
       FOR (n:SeoAnchorText) REQUIRE (n.client_id, n.anchorValue, n.link_id) IS UNIQUE""",

    # === Indexes on :SeoWebPage (secondary label on :Page) ===
    """CREATE INDEX seo_webpage_title IF NOT EXISTS
       FOR (n:SeoWebPage) ON (n.title)""",

    """CREATE INDEX seo_webpage_language IF NOT EXISTS
       FOR (n:SeoWebPage) ON (n.inLanguage)""",

    # === Indexes on new labels ===
    """CREATE INDEX seo_chunk_client IF NOT EXISTS
       FOR (n:SeoChunk) ON (n.client_id)""",

    """CREATE INDEX seo_link_client IF NOT EXISTS
       FOR (n:SeoLink) ON (n.client_id)""",

    """CREATE INDEX seo_link_type IF NOT EXISTS
       FOR (n:SeoLink) ON (n.linkType)""",

    """CREATE INDEX seo_linkgroup_name IF NOT EXISTS
       FOR (n:SeoLinkGroup) ON (n.name)""",

    """CREATE INDEX seo_url_client IF NOT EXISTS
       FOR (n:SeoURL) ON (n.client_id)""",

    # === SeoSchema (Sprint 2) ===
    """CREATE CONSTRAINT seo_schema_id IF NOT EXISTS
       FOR (n:SeoSchema) REQUIRE (n.client_id, n.page_url, n.schemaType, n.position) IS UNIQUE""",

    """CREATE INDEX seo_schema_client IF NOT EXISTS
       FOR (n:SeoSchema) ON (n.client_id)""",

    """CREATE INDEX seo_schema_type IF NOT EXISTS
       FOR (n:SeoSchema) ON (n.schemaType)""",

    # === SeoThing (Sprint 2) ===
    """CREATE CONSTRAINT seo_thing_id IF NOT EXISTS
       FOR (n:SeoThing) REQUIRE (n.client_id, n.name, n.thingType) IS UNIQUE""",

    """CREATE INDEX seo_thing_client IF NOT EXISTS
       FOR (n:SeoThing) ON (n.client_id)""",

    """CREATE INDEX seo_thing_type IF NOT EXISTS
       FOR (n:SeoThing) ON (n.thingType)""",
]

# Cleanup removes ONLY the seovoc enrichment, preserving :Page nodes and :LINKS_TO
CLEANUP_QUERIES = [
    # Remove SeoWebPage label from :Page nodes (keeps :Page + pagerank/HITS intact)
    "MATCH (n:SeoWebPage {client_id: $client_id}) REMOVE n:SeoWebPage REMOVE n.metaDescription, n.markdownText, n.embeddingModel, n.isCrawlable, n.inLanguage, n.wordCount, n.clickDepth, n.publishingDate, n.metaTitle",
    # Delete satellite nodes (these are purely ontologia)
    "MATCH (n:SeoChunk {client_id: $client_id}) DETACH DELETE n",
    "MATCH (n:SeoURL {client_id: $client_id}) DETACH DELETE n",
    "MATCH (n:SeoLink {client_id: $client_id}) DETACH DELETE n",
    "MATCH (n:SeoLinkGroup {client_id: $client_id}) DETACH DELETE n",
    "MATCH (n:SeoAnchorText {client_id: $client_id}) DETACH DELETE n",
    "MATCH (n:SeoSchema {client_id: $client_id}) DETACH DELETE n",
    "MATCH (n:SeoThing {client_id: $client_id}) DETACH DELETE n",
    "MATCH (n:SeoPersona {client_id: $client_id}) DETACH DELETE n",
    # Remove HAS_URL/HAS_CHUNK/HAS_LINK_GROUP rels from :Page (keeps :LINKS_TO)
    """MATCH (p:Page {client_id: $client_id})-[r]->()
       WHERE type(r) IN ['HAS_URL','HAS_CHUNK','HAS_LINK_GROUP','HAS_SCHEMA_MARKUP','ABOUT','MENTIONS']
       DELETE r""",
]


async def setup_seo_schema(driver: AsyncDriver) -> dict:
    """Create all SEOntology constraints and indexes in Neo4j."""
    results = {"created": 0, "errors": []}
    async with driver.session() as session:
        for query in SETUP_QUERIES:
            try:
                await session.run(query)
                results["created"] += 1
            except Exception as e:
                results["errors"].append(f"{query[:60]}...: {e}")
    return results


async def cleanup_seo_data(driver: AsyncDriver, client_id: str) -> dict:
    """
    Remove SEOntology enrichment for a client.
    PRESERVES :Page nodes, :LINKS_TO relationships, and PageRank/HITS scores.
    Only strips Seo* labels and deletes satellite nodes (SeoChunk, SeoLink, etc).
    """
    results = {"cleaned": 0, "errors": []}
    async with driver.session() as session:
        for query in CLEANUP_QUERIES:
            try:
                await session.run(query, client_id=client_id)
                results["cleaned"] += 1
            except Exception as e:
                results["errors"].append(str(e))
    return results
