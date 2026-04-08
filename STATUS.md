# WebKnoGraph — Status (Source of Truth)

**Last updated:** 2026-04-08
**Version:** `wkg-v0.1.12`
**Branch:** `feature/crawler-docker-separation` (29 commits ahead of origin — sin push)

---

## Servicios (docker-compose.local.yml)

| Container | Puerto | Estado |
|---|---|---|
| `wkg-app` | 8080 | healthy |
| `wkg-crawler` | 8081 | healthy |
| `wkg-neo4j` | 7474 / 7687 | healthy (`neo4j` / `neo4j123`) |
| `wkg-postgres` | 54322 | healthy (`postgres` / default) |

## Datos reales (cliente UOC `e72610f5-23a6-44af-9118-4eb22fbfb56e`)

### Postgres (`public.rag_*`)
| Tabla | Filas |
|---|---|
| `rag_pages` | 356 |
| `rag_chunks` | 10.244 |
| `rag_links` | **0** ⚠️ (los links viven solo en Neo4j) |
| `rag_clients` | 1 |
| `rag_conversations` | 0 |
| `rag_messages` | 0 |

### Neo4j (labels activos)
`Page`, `SeoWebPage`, `SeoURL`, `SeoChunk`, `SeoLink`, `SeoLinkGroup`, `SeoAnchorText`, `SeoSchema`, `SeoThing`

| Entidad | Cantidad |
|---|---|
| `:Page` (dual `:SeoWebPage`) | 356 |
| `:SeoChunk` | 10.244 |
| `:LINKS_TO` (con `location`, `weight`, `anchor_text`) | 297 |
| `:SeoSchema` | **859** |
| `:SeoThing` | **0** ⚠️ |

> Embeddings: modelo `hiiamsid/sentence_similarity_spanish_es` (768-dim).

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
| 2 | HTML enrichment (SeoSchema + SeoThing) | ⚠️ PARCIAL — 859 SeoSchema pero **0 SeoThing** |
| 3 | LLM enrichment | ⏳ Not started |
| 4 | GSC integration | ⏳ Not started |

---

## Pendientes (prioridad)

### P0 — Housekeeping
- [ ] `git push` de los 29 commits locales a `origin` y `gitlab`
- [ ] Limpiar PNGs sueltos en root: `client_modal_test.png`, `modal_fixed.png`, `modal_open.png`
- [ ] Investigar por qué `SeoThing = 0` (¿pipeline Sprint 2 nunca corrió con datos reales?)
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
