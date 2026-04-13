# WebKnoGraph — Status (Source of Truth)

**Last updated:** 2026-04-08
**Version:** `wkg-v0.1.12`
**Branch:** `feature/crawler-docker-separation` (29 commits ahead of origin — sin push)

---

## Servicios (docker-compose.local.yml)

**Compose project:** `webknograph-uoc` — renombrado para coexistir en paralelo con el stack del cliente **Quiron** (`wkg-*`, project `webknograph`). Ambos stacks corren simultáneamente sin colisiones.

| Container | Puerto host | Estado |
|---|---|---|
| `wkguoc-app` (dashboard) | **8090** → 8080 | healthy |
| `wkguoc-crawler` | **8082** → 8081 | healthy |
| `wkguoc-neo4j` | **7475** / **7688** | healthy (`neo4j` / `neo4j123`) |
| `wkguoc-postgres` | **54323** → 5432 | healthy (`postgres` / `postgres123`) |

**Dashboard UOC:** http://localhost:8090
**Neo4j browser UOC:** http://localhost:7475

> ⚠️ Quiron corre en los puertos clásicos (8080/8081/7474/7687/54322) con containers `wkg-*`. **No tocar Quiron desde este repo.**

## Datos (cliente UOC `e72610f5-23a6-44af-9118-4eb22fbfb56e`)

Volúmenes recién creados bajo project `webknograph-uoc` — **DBs vacías**. Los 356 pages / 10.244 chunks / 297 links que había antes estaban en volúmenes del stack Quiron (contaminación histórica). Pendiente **re-ingestar desde los parquets** en `data/crawl4ai_data/www_uoc_edu/`.

Origen de la re-ingesta:
- `data/crawl4ai_data/www_uoc_edu/pages/crawl_date=*/` (parquets)
- `data/crawl4ai_data/www_uoc_edu/links/crawl_date=*/` (parquets)

Embedding model: `hiiamsid/sentence_similarity_spanish_es` (768-dim).

---

## Avances desde `v0.1.4` (último registro de memoria)

Commits relevantes en `feature/crawler-docker-separation`:

- `de004a3` **feat: integrate SEOntology + GLiNER pipelines in dashboard** (v0.1.7) — pendiente histórico resuelto
- `91d8501` fix: crawler robustness + API_URL auto-detect
- `5307a0f` fix: add `CRAWLER_SERVICE_URL` and healthcheck to prod compose (v0.1.8)
- `223094f` fix: crawler UI detects running crawls on page load (v0.1.9)
- `8601618` fix: unify crawler UI status into single `checkCrawlerStatus()` (v0.1.10)
- `041f2e5` fix: prevent crawler UI flickering from concurrent status checks (v0.1.11 → v0.1.12)

**Foco reciente:** estabilización del crawler UI + integración SEOntology/GLiNER en dashboard.

---

## Sprints SEOntology

| Sprint | Descripción | Estado |
|---|---|---|
| 1 | Foundations (fusión con graph_rag) | ✅ DONE |
| 2 | HTML enrichment (SeoSchema + SeoThing) | ⚠️ A re-verificar tras re-ingesta |
| 3 | LLM enrichment | ⏳ Not started |
| 4 | GSC integration | ⏳ Not started |

---

## Pendientes (prioridad)

### P0 — Housekeeping
- [x] ~~`git push` de los 29 commits locales~~ (hecho 2026-04-08)
- [x] ~~Separar stack UOC de Quiron~~ (hecho 2026-04-08, project `webknograph-uoc`)
- [ ] **Re-ingestar datos UOC** desde parquets `data/crawl4ai_data/www_uoc_edu/`
- [ ] Verificar post-ingesta si `SeoThing` se crea o sigue a 0
- [ ] Limpiar PNGs sueltos en root: `client_modal_test.png`, `modal_fixed.png`, `modal_open.png`
- [ ] Decidir si `rag_links` se elimina de Postgres (single source = Neo4j)

### P1 — Performance / Quality
- [ ] **Batch embeddings** en ingest (`embed_documents_batch` en vez de `embed_document` por chunk)
- [ ] Más profundidad de crawl (solo 18 links content sobre 297 totales — 278 nav, 1 footer)
- [ ] Tuning RAG (reranking, context window)

### P2 — Features
- [ ] SEOntology Sprint 3 (LLM enrichment)
- [ ] SEOntology Sprint 4 (GSC integration)
- [ ] Interlinking: anchor text suggestions, silo filtering

---

## Documentación relacionada

- **`INGESTION_PIPELINE.md` — auditoría end-to-end del pipeline (2026-04-08). Bugs P0: `links_migrated` optimista, `rag_links` nunca poblado, falta endpoint de verificación.**
- `workplan.md` — plan de trabajo detallado
- `HOW-IT-WORKS.md` — arquitectura
- `INSTALL.md` — setup
- `crawlerlimpieza.md` — notas de limpieza del crawler
- `devops/README.md` — CI/CD GitLab
- memoria Claude: `feedback_deploy_gitlab.md`, `project_ingestion_pipelines.md`

---

## Git remotes

- `origin` → https://github.com/dvillarrubia/WebKnoGraph.git
- `gitlab` → https://gitlab.lin3s.com/dvillarrubia/WebKnoGraph.git
- Rama CI/CD: `gitlab/1-ci-cd`
