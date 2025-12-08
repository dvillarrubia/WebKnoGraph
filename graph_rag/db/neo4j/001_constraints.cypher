// ============================================================================
// WebKnoGraph Graph-RAG: Neo4j Multi-tenant Schema
// ============================================================================

// ============================================================================
// CONSTRAINTS
// ============================================================================

// Unique constraint for Page nodes per client
CREATE CONSTRAINT page_unique_url IF NOT EXISTS
FOR (p:Page) REQUIRE (p.client_id, p.url) IS UNIQUE;

// Unique constraint for Client nodes
CREATE CONSTRAINT client_unique_id IF NOT EXISTS
FOR (c:Client) REQUIRE c.id IS UNIQUE;

// ============================================================================
// INDEXES
// ============================================================================

// Index for fast client_id lookups on Pages
CREATE INDEX page_client_id IF NOT EXISTS
FOR (p:Page) ON (p.client_id);

// Index for PageRank-based queries
CREATE INDEX page_pagerank IF NOT EXISTS
FOR (p:Page) ON (p.client_id, p.pagerank);

// Index for folder depth filtering
CREATE INDEX page_folder_depth IF NOT EXISTS
FOR (p:Page) ON (p.client_id, p.folder_depth);

// Composite index for common query pattern
CREATE INDEX page_client_metrics IF NOT EXISTS
FOR (p:Page) ON (p.client_id, p.pagerank, p.authority_score);

// ============================================================================
// SAMPLE QUERIES FOR REFERENCE
// ============================================================================

// -- Get pages linked from a source page (1-hop neighbors)
// MATCH (source:Page {client_id: $client_id, url: $source_url})-[:LINKS_TO]->(target:Page)
// RETURN target.url, target.title, target.pagerank
// ORDER BY target.pagerank DESC
// LIMIT 10

// -- Get 2-hop neighborhood for context expansion
// MATCH (source:Page {client_id: $client_id, url: $source_url})-[:LINKS_TO*1..2]-(related:Page)
// WHERE related.client_id = $client_id
// RETURN DISTINCT related.url, related.title, related.pagerank
// ORDER BY related.pagerank DESC
// LIMIT 20

// -- Find pages that link TO a target (backlinks)
// MATCH (source:Page)-[:LINKS_TO]->(target:Page {client_id: $client_id, url: $target_url})
// WHERE source.client_id = $client_id
// RETURN source.url, source.title, source.pagerank
// ORDER BY source.pagerank DESC

// -- Get hub pages (pages that link to many authority pages)
// MATCH (hub:Page {client_id: $client_id})-[:LINKS_TO]->(auth:Page)
// WHERE hub.hub_score > 0.1
// RETURN hub.url, hub.hub_score, COUNT(auth) as outlinks
// ORDER BY hub.hub_score DESC
// LIMIT 10

// -- Shortest path between two pages
// MATCH path = shortestPath(
//   (a:Page {client_id: $client_id, url: $url_a})-[:LINKS_TO*]-(b:Page {client_id: $client_id, url: $url_b})
// )
// RETURN path
