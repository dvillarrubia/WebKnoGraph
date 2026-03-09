# WebKnoGraph + SEOntology Project Memory

## Project Structure
- **Root**: `D:/KGRAG UOC/WebKnoGraph/`
- **Ontologia module**: `WebKnoGraph/ontologia/` - SEOntology integration
- **SEOvoc source**: `WebKnoGraph/ontologia/seontology/seovoc.ttl`
- **Graph RAG**: `WebKnoGraph/graph_rag/` - Main API/services
- **UOC crawl data**: `WebKnoGraph/data/crawl4ai_data/www_uoc_edu/`

## SEOntology (seovoc.ttl) Stats
- 13 classes, 21 object properties, 77 data properties
- Key classes: WebPage, Chunk, URL, Link, LinkGroup, AnchorText, Query, Schema, Persona, PageGroup, Thing
- Parsed with rdflib 7.6.0

## Ontologia Module Files
- `ontology_parser.py` - Parses seovoc.ttl with rdflib → ParsedOntology dataclass
- `neo4j_schema.py` - Neo4j labels (Seo* prefix), constraints, indexes
- `ingest_ontology.py` - Maps parquet/Supabase data → Neo4j Seo* nodes
- `run_ingest.py` - CLI: --parse-only, --ingest, --extract-schema
- `export_rdf.py` - Export Neo4j Seo* nodes → uoc_ontology.ttl
- `extractors/schema_extractor.py` - JSON-LD extraction from HTML

## UOC Data Available (as of 2026-03-09)
- 3 pages crawled, 96 links, 7 JSON-LD blocks
- JSON-LD types: CollegeOrUniversity, ItemList, NewsArticle
- Link locations: content, footer (weights 0.3-1.0)
- Languages: es, ca

## Neo4j Label Mapping (FUSED)
- WebPage → `:Page:SeoWebPage` (DUAL LABEL — fused with graph_rag :Page)
- Chunk→SeoChunk, URL→SeoURL, Link→SeoLink, LinkGroup→SeoLinkGroup
- AnchorText→SeoAnchorText, Query→SeoQuery, Schema→SeoSchema, Persona→SeoPersona
- :LINKS_TO preserved (PageRank/HITS), new rels are additive (HAS_URL, HAS_CHUNK, etc)
- Cleanup strips Seo* labels/satellites but PRESERVES :Page + pagerank + :LINKS_TO

## Sprint Progress
- Sprint 1 (Foundations): DONE + FUSED with graph_rag (dual-label Page:SeoWebPage)
- Sprint 2 (HTML enrichment): schema_extractor.py created, needs Neo4j integration
- Sprint 3 (LLM enrichment): Not started
- Sprint 4 (GSC integration): Not started

## Key Patterns
- Embedding model: `hiiamsid/sentence_similarity_spanish_es` (768-dim)
- Client ID pattern: UUID from rag_clients table
- Parquet structure: pages/ and links/ subdirs with crawl_date partition
- Docker services: docker-compose.local.yml (local dev), docker-compose.rag.yml (prod)

## Docker / Workflow
- **Siempre trabajamos contra containers Docker** — no ejecución local
- docker-compose.local.yml: wkg-crawler (8081), wkg-app (8080), wkg-postgres, wkg-neo4j
- Crawler code mounted as volume: `./crawler_service:/app/crawler_service` → solo restart, no rebuild
- Rebuild solo necesario si cambian requirements.txt o Dockerfile
- **Cambios en crawler deben ir en AMBOS**: `scripts/crawl4ai_advanced.py` + `crawler_service/crawl4ai_advanced.py`
- Restart crawler: `docker-compose -f docker-compose.local.yml restart wkg-crawler`

## Bugs Corregidos
- **urls_only ignorado**: `to_visit` incluía sitemap+start_url+discovered aunque urls_only=True. Fix: si urls_only, to_visit solo usa urls_from_file
- **NoneType not subscriptable en crawl**: result.metadata o result.markdown pueden ser None en algunas páginas. Fix: try/except en accesos a metadata y markdown
