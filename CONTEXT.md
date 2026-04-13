# CONTEXT.md — Estado del proyecto para continuidad entre máquinas

> Generado 2026-04-13. Leer esto al retomar trabajo en cualquier máquina.
> Decir "retomo desde el portátil" o "retomo" para que Claude lo use.

---

## Estado actual (commit 81bf068)

**Rama:** `feature/crawler-docker-separation`
**Pushed a:** GitHub (origin) + GitLab (gitlab)

### Grafo Neo4j — 14K nodos, 14K relaciones

| Label | Count | Descripción |
|-------|------:|-------------|
| SeoQuery | 12,397 | Datos GSC (3 ventanas: 7d/28d/3m) |
| SeoChunk | 945 | Fragmentos de texto con embeddings |
| SeoSchema | 306 | JSON-LD (CollegeOrUniversity, ItemList, NewsArticle...) |
| Page:SeoWebPage | 153 | Páginas crawleadas de www.uoc.edu/es/ |
| SeoURL | 153 | URLs canónicas |

| Relación | Count |
|----------|------:|
| HAS_QUERY | 12,397 |
| HAS_CHUNK | 945 |
| LINKS_TO | 381 |
| HAS_SCHEMA_MARKUP | 306 |
| HAS_URL | 153 |
| HAS_PRIMARY_QUERY | 131 |

### Campos poblados en :SeoWebPage (23+)

| Campo | Fuente | Cobertura |
|-------|--------|-----------|
| canonical, robots, lang, viewport | HTML parquet | 96% |
| h1, headings (JSON h1-h6) | HTML parquet | 81-96% |
| og_title, og_description, og_image, og_type | HTML parquet | 12-96% |
| twitter_card | HTML parquet | 96% |
| hreflang (JSON) | HTML parquet | 90% |
| has_structured_data, noFollow, noIndex | HTML parquet | 96% |
| isCrawlable | robots real (no hardcoded) | 96% |
| clicks, impressions, ctr, position | GSC 28d | 86% |
| pagerank, hub_score, authority_score | LINKS_TO calc | 100% |

### Campos de :SeoQuery (26 propiedades)

clicks/impressions/ctr/position × 3 ventanas (7Days, 28Days, 3Months) + 8 trends + queryText, page_url, inLanguage, dateCreated, updated_at, client_id

---

## Servicios implementados (2026-04-13)

### 1. GSC Sync — `graph_rag/services/gsc_service.py`
- **Endpoint:** `POST /api/v1/dashboard/gsc/sync`
- **Qué hace:** Descarga datos de Google Search Console, crea :SeoQuery nodes, asigna primary queries, agrega métricas a páginas
- **Parámetros:** client_id, site_url (`sc-domain:uoc.edu`), credentials_path (`/app/credentials/uoc-marketing-ff7d855ed00c.json`), url_filter (`www.uoc.edu`), refresh, language
- **Incremental:** Solo importa queries de URLs que existen en Neo4j

### 2. SEO Metadata — `graph_rag/services/seo_metadata_service.py`
- **Endpoint:** `POST /api/v1/dashboard/seo-metadata/extract`
- **Qué hace:** Lee HTML de parquets, extrae canonical/robots/OG/hreflang/headings/viewport, actualiza :SeoWebPage
- **Parámetros:** client_id, crawl_dir, date_partition

### 3. Link Extraction — `graph_rag/services/link_extraction_service.py`
- **Endpoint:** `POST /api/v1/dashboard/links/extract`
- **Qué hace:** Extrae links internos del HTML, clasifica por ubicación DOM, crea :LINKS_TO, calcula PageRank (20 iter, damping 0.85) y HITS (Hub/Authority)
- **Parámetros:** client_id, crawl_dir, domain, date_partition, calculate_scores

---

## Docker Stack UOC

```
Containers: wkguoc-app (8090), wkguoc-crawler (8082), wkguoc-postgres (54323), wkguoc-neo4j (7475/7688)
Compose: docker-compose.local.yml
Project name: webknograph-uoc
```

**Credenciales:**
- Neo4j: `neo4j` / `neo4j123`
- Postgres: `postgres` / `postgres123`
- GSC: `credentials/uoc-marketing-ff7d855ed00c.json` (NO en git, copiar manualmente)

**Client ID UOC:** `0c4e3765-2960-4988-85f8-b480f04abf96`

**Volúmenes mount (docker-compose.local.yml):**
- `./graph_rag:/app/graph_rag` (hot reload código)
- `./ontologia:/app/ontologia`
- `./credentials:/app/credentials:ro` (credenciales GSC)

---

## Restaurar DBs en otra máquina

Dumps en `data/db_dumps/` (10 MB total):

```bash
# 1. Levantar stack
docker compose -f docker-compose.local.yml up -d

# 2. Restaurar Postgres
docker cp data/db_dumps/postgres_dump.backup wkguoc-postgres:/tmp/
docker exec wkguoc-postgres bash -c "pg_restore -U postgres -d postgres --clean --if-exists /tmp/postgres_dump.backup"

# 3. Restaurar Neo4j
docker stop wkguoc-neo4j
docker run --rm -v wkguoc-neo4jdata:/data -v ./data/db_dumps:/dumps neo4j:5.20-community bash -c "neo4j-admin database load neo4j --from-path=/dumps/ --overwrite-destination"
docker start wkguoc-neo4j

# 4. Verificar
docker exec wkguoc-neo4j cypher-shell -u neo4j -p neo4j123 "MATCH (n) UNWIND labels(n) AS l RETURN l, count(n) ORDER BY count(n) DESC"
# Esperado: SeoQuery 12397, SeoChunk 945, SeoSchema 306, Page 153, SeoWebPage 153, SeoURL 153
```

Si los dumps no están, re-ejecutar en orden:
1. Crawl → 2. Ingest (full_refresh) → 3. Ontology setup + ingest (with_chunks) → 4. GSC sync → 5. SEO metadata extract → 6. Links extract

---

## Decisiones GSC (5/5 cerradas)

1. **Auth:** Service account JSON por cliente en `rag_clients.gsc_credentials`
2. **Propiedad UOC:** `sc-domain:uoc.edu` (domain, siteFullUser, cubre subdominios)
3. **Ventanas:** 7d, 28d, 3m (seovoc)
4. **URLs sin match:** Filtrar y saltar (incremental, se amplía con cada crawl)
5. **Score:** Vacío por ahora (sin fórmula de negocio)

---

## Bugs conocidos (no repetir)

- **Neo4j NULL ORDER DESC:** NULL ordena primero en DESC. Usar `COALESCE(field, 0)`
- **Crawl browser restart hang:** Crawl4AI se cuelga en reinicio de browser tras ~200 URLs. Usar stop manual si se atasca.
- **HF PermissionError:** mkdir+chown en Dockerfile antes de `USER appuser`
- **urls_only ignorado:** Con urls_only, construir to_visit solo con urls_from_file
- **NoneType en crawl:** result.metadata/markdown pueden ser None

---

## Reglas del proyecto

- **Docker SIEMPRE.** Nunca nativo. Ver CLAUDE.md para reglas completas.
- **Push on green:** Tras cada hito, push a GitHub Y GitLab.
- **Quiron NO tocar:** Stack paralelo en `D:\Quiron\WebKnoGraph\` con containers `wkg-*`.
- **Crawler dual sync:** Cambios en `scripts/` Y `crawler_service/`.

---

## Próximos pasos (Sprint 3)

- [ ] `queryType` / `queryCategory` en :SeoQuery — LLM classifier
- [ ] GLiNER entity persistence → :SeoThing nodes
- [ ] Más crawl depth (154 de 6,314 URLs /es/ disponibles)
- [ ] Orquestador: `POST /dashboard/full-ingest` (cadena todos los pasos)
- [ ] Verificador: `GET /dashboard/ingest/verify` (compara contadores vs DBs reales)
- [ ] Batch embeddings (`embed_documents_batch` en vez de per-chunk)

---

## Ontología seovoc.ttl (referencia)

- 13 classes, 21 object properties, 77 data properties
- Cobertura actual: ~65% (antes 42%)
- Mayor gap restante: Query (queryType/queryCategory), Persona, PageGroup
- Canónica: `D:/KGRAG UOC/ontologia/seontology/seovoc.ttl` v0.0.1
- Coverage doc: `SEOVOC_COVERAGE.md`
