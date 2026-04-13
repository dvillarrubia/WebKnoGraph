# WORKPLAN — SEO metadata completo + GSC integration (seovoc-compliant)

> **Creado:** 2026-04-08 · **Actualizado:** 2026-04-08 con auditoría exhaustiva seovoc
> **Origen:** auditoría `INGESTION_PIPELINE.md` + `SEOVOC_COVERAGE.md` (cobertura real: 42% data props, 67% object props, 64% classes).
> **Objetivo:** cerrar el gap entre lo que seovoc modela y lo que extraemos/persistimos.
> **Gap individual más grande:** clase `Query` con **41 data properties al 0%** (53% del total de data props de seovoc).

---

## Contexto — gaps detectados

### SEO HTML metadata (crawler)
Capturado hoy: `title`, `meta_description`, `markdown` (con H1–H6 embebidos), `html_content`, JSON-LD (vía `ontologia/extractors/schema_extractor.py`).

**NO capturado** (el HTML está en `html_content` pero no se parsea):
- `<link rel="canonical">`, meta robots, meta keywords, meta author
- OpenGraph (`og:title`, `og:description`, `og:image`, `og:type`)
- Twitter Card (`twitter:*`)
- `hreflang` / `<link rel="alternate">`
- `<html lang="...">`
- Viewport, favicon
- H1/H2/H3 **estructurados** (solo existen embebidos en markdown, no como lista indexable)
- Microdata / RDFa (solo JSON-LD parseado)

### seovoc:WebPage — props sin implementar
- `metaTitle` (diferente de `title` en seovoc) — hoy mezclado
- `inLanguage` — hoy **hardcoded `es`** (`ingest_ontology.py`)
- `isCrawlable` — hoy **hardcoded `true`**
- `sourceCode`, `status`, `hasSchemaMarkup` — no se setean
- `clickDepth` — parcial (se calcula como `folder_depth`)

### GSC / Query — gap grande
`neo4j_schema.py:24` mapea `:Query → :SeoQuery` pero **no existe** `gsc_service.py`, ni endpoint, ni ingestor. Nodos `:SeoQuery` jamás se crean.

Relaciones seovoc implicadas:
- `WebPage -hasQuery-> Query`
- `WebPage -hasPrimaryQuery-> Query` (functional)
- `AnchorText -influencedByQuery-> Query`

Data properties de `:Query` (todas vacías hoy, 30+):
- `clicks`, `clicks7Days`, `clicks28Days`, `clicks3Months` + 3 trends
- `impressions` + 3 ventanas + 3 trends
- `ctr` + 3 ventanas + 3 trends
- `position` + 3 ventanas + 3 trends
- `dateCreated`

No hay `:Keyword` separada — todo se modela como `:Query`.

---

## FIX-1 — Crawler: extraer SEO HTML metadata

**Archivos:** `crawler_service/crawl4ai_advanced.py` + gemelo `scripts/crawl4ai_advanced.py` (mantener paridad, regla CLAUDE.md).

**Acción:**
Añadir parser BeautifulSoup sobre `html_content` (ya disponible en `result.html`). Extraer:

| Campo parquet | Origen HTML |
|---|---|
| `canonical` | `<link rel="canonical" href>` |
| `robots` | `<meta name="robots" content>` |
| `lang` | `<html lang>` |
| `og_title` | `<meta property="og:title">` |
| `og_description` | `<meta property="og:description">` |
| `og_image` | `<meta property="og:image">` |
| `og_type` | `<meta property="og:type">` |
| `twitter_card` | `<meta name="twitter:card">` |
| `hreflang` | JSON list de `<link rel="alternate" hreflang>` |
| `headings` | JSON dict `{h1:[], h2:[], h3:[], h4:[], h5:[], h6:[]}` |
| `viewport` | `<meta name="viewport">` |
| `has_structured_data` | bool — hay ≥1 JSON-LD válido |

**Contrato:** solo añade columnas, no rompe schema existente. Todas opcionales (string vacío o `None`).

**Tests:** una página fixture con HTML completo, asserts sobre cada columna nueva.

**Requiere re-crawl** de UOC para poblar las nuevas columnas (DB está vacía, momento perfecto).

**Estado:** ⏳ pendiente

---

## FIX-2 — SEOntology: propagar SEO metadata a `:SeoWebPage`

**Archivos:** `ontologia/ingest_ontology.py`, `ontologia/neo4j_schema.py`.

**Acción:**
- Mapear columnas nuevas del parquet a props de `:SeoWebPage`:
  - `lang` → `seo:inLanguage` (eliminar hardcode `'es'`)
  - `robots` → derivar `isCrawlable` (`false` si contiene `noindex`)
  - `canonical` → prop `canonical` (seovoc `:link` es ambiguo — usar custom `canonical` hasta decidir)
  - `og_*`, `twitter_card` → props directas (prefijadas `og_`, `tw_`)
  - `hreflang` → crear nodos `:SeoURL` alternate con rel `:HAS_ALTERNATE`
  - `has_structured_data` → `hasSchemaMarkup` (boolean)
- **Headings como nodos** `:SeoHeading {level, order, text}` con rel `(:SeoWebPage)-[:HAS_HEADING]->(:SeoHeading)`. Permite queries "páginas con H1 conteniendo X". *(Decidido en conversación — opción más rica sobre props JSON)*
- Actualizar `neo4j_schema.py` con constraints para `:SeoHeading`.

**Estado:** ⏳ pendiente — depende de FIX-1

---

## FIX-3 — GSC importer (servicio nuevo)

**Archivo nuevo:** `graph_rag/services/gsc_service.py`
**Endpoint nuevo:** `POST /api/v1/dashboard/gsc/sync` en `graph_rag/api/routes.py`

**Auth:**
- OAuth2 service account o refresh_token almacenado por cliente en `rag_clients` (nueva columna `gsc_credentials` JSONB)
- Alternativa inicial: leer de env var `GSC_SERVICE_ACCOUNT_JSON` (1 cuenta para todos los clientes UOC)

**Body:** `{client_id, site_url, windows: ["7d","28d","3m"]}`

**Flujo:**
1. Llamar `searchanalytics.query` para cada ventana (`startDate`/`endDate` calculados)
2. Dimensiones: `query`, `page`
3. Para cada row:
   - MERGE `:SeoQuery {text: row.query}` — ID estable por texto normalizado
   - MATCH `:SeoWebPage {url: row.page}`
   - MERGE `(page)-[:HAS_QUERY]->(query)`
   - Set props ventana-específicas: `clicks7Days`, `impressions7Days`, `ctr7Days`, `position7Days` (y equivalentes para 28d, 3m)
4. Segunda pasada — calcular trends:
   - `clicks7DaysTo28DaysTrend = clicks7Days / clicks28Days` (normalizado)
   - idem para impressions, ctr, position (28d→3m, 7d→3m)
5. Tercera pasada — determinar `hasPrimaryQuery` por page = query con más `clicks28Days`. Borrar `:HAS_PRIMARY_QUERY` previo, crear nuevo.
6. Set `dateCreated` en `:SeoQuery` la primera vez que se ve.

**Idempotencia:** tabla nueva `rag_gsc_sync_log` en Postgres con `(client_id, site_url, window, synced_at, rows_imported)`. Migration SQL en `graph_rag/db/migrations/`.

**Contadores devueltos:** `queries_created`, `queries_updated`, `relationships_created`, `pages_matched`, `pages_missing` (URLs de GSC que no existen aún en `:SeoWebPage`).

**Estado:** ⏳ pendiente — independiente de FIX-1/2, puede correr sobre DB actual

---

## FIX-4 — Dashboard UI para GSC

**Archivo:** `graph_rag/static/index.html`

**Acción:**
- Nueva sección "GSC Sync" debajo de la de ontology
- Form: site_url (autocompletar con client.domain), multicheck ventanas 7d/28d/3m, botón "Sync"
- Muestra counts post-ejecución + link a Neo4j Browser con query sugerida (`MATCH (q:SeoQuery) RETURN q LIMIT 25`)
- Muestra última sync desde `rag_gsc_sync_log`

**Estado:** ⏳ pendiente — depende de FIX-3

---

## FIX-3 ampliado — GSC importer debe cubrir 41 properties de :SeoQuery

Tras auditoría exhaustiva `SEOVOC_COVERAGE.md`, la clase `Query` tiene **41 data properties** (no las ~10 que asumí originalmente). Desglose:

- **Clicks (6):** `clicks7Days`, `clicks28Days`, `clicks3Months` + 3 trends (`7DTo28D`, `28DTo3M`, `7DTo3M`)
- **Impressions (6):** idem estructura
- **CTR (6):** idem estructura
- **Position (6):** idem estructura
- **Score (3):** `score7Days`, `score28Days`, `score3Months` — fórmula **no definida en TTL**. Preguntar al usuario. Propuesta: `(clicks * ctr) / position` normalizado.
- **Meta (2):** `dateCreated`, `keywordType` (primary/secondary — derivar de ranking por clicks)
- **Clasificación enum (2):** `queryType` (13 valores), `queryCategory` (4 valores). **Requiere clasificador LLM — diferir a Sprint 3.**
- **Lang (1):** `schema:inLanguage`

**Relaciones asociadas:**
- `(:SeoWebPage)-[:HAS_QUERY]->(:SeoQuery)` — N:N
- `(:SeoWebPage)-[:HAS_PRIMARY_QUERY]->(:SeoQuery)` — **FunctionalProperty**, borrar previa antes de crear
- `(:SeoAnchorText)-[:INFLUENCED_BY_QUERY]->(:SeoQuery)` — FIX-8 (diferido)

**Valores enum de `queryType`:** Popular, Long-tail, Dynamic, Multi-hop, Analytical, Commonsense, Causal, Exploratory, Instructive, Recommendation, Spatio-temporal, Lifestyle, Cultural, Philosophical
**Valores enum de `queryCategory`:** Advice, Explanations, Facts, Planning

---

## FIX-6 (nuevo) — Propiedades de auditoría en todos los nodos

seovoc define en `owl:Thing`:
- `importID` — ID externo (ej. GSC row_id, crawl_id)
- `importHash` — hash del payload para detectar cambios → **re-ingesta incremental**
- `importOrigin` — fuente (`crawl4ai_2026-04-08`, `gsc_2026-04-08`, `gliner_2026-04-08`)

**Acción:** añadir estas 3 props a TODO MERGE en `ingest_ontology.py`, `gsc_service.py`, `gliner_service.py`. Habilita:
- Dedup por hash
- Tracking de origen (qué etapa creó qué nodo)
- Updates incrementales sin re-crear el grafo entero

**Estado:** ⏳ pendiente — transversal, tocar cada MERGE

---

## FIX-7 (nuevo, baja prio) — `anchorResource` Link → Thing

`Link -anchorResource-> (Thing | LinkGroup)`. Conecta un link a la entidad Thing extraída del JSON-LD del target page.

**Acción:** cuando un link apunta a una página ya ingestada con `:SeoThing` extraído, crear `(:SeoLink)-[:ANCHOR_RESOURCE]->(:SeoThing)`. Útil para queries "¿qué entidades son enlazadas desde X?".

**Estado:** ⏳ baja prioridad

---

## FIX-8 (nuevo, Sprint 3) — `influencedByQuery` AnchorText → Query

`AnchorText -influencedByQuery-> Query`. Modela que el anchor text fue elegido para rankear en ciertas queries. Análisis tipo "link baiting".

**Acción:** NLP matching anchor↔query (embedding similarity). Requiere FIX-3 completado.

**Estado:** ⏳ Sprint 3

---

## FIX-9 (stub, baja prio) — Persona y PageGroup

Clases enteras sin implementar:
- `:SeoPersona` (3 props: `ageRange`, `preferredDevice`, `role`) + rel `hasPersona`
- `:SeoPageGroup` (2 props: `groupName`, `groupType`) + rel `hasPage`

**Decisión para UOC:** documentar gap, NO implementar hasta petición del cliente. Requieren fuentes externas (GA4 demographics, test config).

**Estado:** ⏳ parked

---

## Micro-fixes (añadir dentro de FIX-1/2)

De la auditoría de cobertura, pequeñas props fáciles que no justifican un FIX propio:

- **FIX-1:** añadir `noFollow` (extraer `rel="nofollow"` del `<a>`), `status` HTTP code (ya lo da Crawl4AI)
- **FIX-2:**
  - `tokenCount` en Chunk (usar tokenizer del embedding model)
  - Eliminar hardcodes: `isCrawlable=true` → leer robots meta, `inLanguage='es'` → leer `<html lang>`, `chunkStrategy='semantic'` → configurable

---

## FIX-5 — Actualizar STATUS y memoria

- STATUS.md: sprint 4 GSC `Not started` → `In progress`
- Añadir entrada en memoria: `project_gsc_integration.md`
- Cerrar este workplan cuando FIX-1..4 estén en verde

**Estado:** ⏳ pendiente — último

---

## Orden de ejecución recomendado

Tras consulta al usuario, decidir entre:

**Opción A — SEO primero (re-crawl necesario):**
1. FIX-1 (crawler) → rebuild container crawler
2. Re-crawl UOC completo
3. FIX-2 (SEOntology propagation) → re-ingesta
4. FIX-3 (GSC) → sync
5. FIX-4 (UI)
6. FIX-5 (docs)

**Opción B — GSC primero (sin re-crawl):**
1. FIX-3 (GSC) → funciona sobre `:SeoWebPage` actuales (aunque sin `lang`/`canonical` correctos)
2. FIX-4 (UI)
3. FIX-1 + FIX-2 (SEO metadata) — más tarde, requiere re-crawl
4. FIX-5 (docs)

**Recomendación:** **Opción A**. Las DBs están vacías ahora (momento ideal para re-crawl), y GSC se beneficia de tener `canonical` correcto para matchear URLs.

---

## Preguntas abiertas (bloqueantes para FIX-3)

1. **GSC auth**: ¿service account ya creada o hay que pedirla? ¿dónde guardamos credenciales por cliente?
2. **Site URL UOC en GSC**: ¿`https://www.uoc.edu/` o `sc-domain:uoc.edu`?
3. **Ventanas confirmadas**: las 3 de seovoc (7d/28d/3m) o simplificamos a una?
4. **URLs de GSC sin match en Neo4j**: ¿crear `:SeoWebPage` stub o solo loggear y saltarlas?

---

## Referencias

- Ontología canónica: `D:/KGRAG UOC/ontologia/seontology/seovoc.ttl` (v0.0.1)
- **Auditoría cobertura seovoc:** `D:/KGRAG UOC/WebKnoGraph/SEOVOC_COVERAGE.md` (77 data props + 21 object props cruzadas contra código)
- Auditoría ingesta: `D:/KGRAG UOC/WebKnoGraph/INGESTION_PIPELINE.md`
- Mapeo actual Neo4j: `ontologia/neo4j_schema.py`
- Ingestor SEOntology: `ontologia/ingest_ontology.py`
- Schema extractor (JSON-LD): `ontologia/extractors/schema_extractor.py`
- Reglas proyecto: `CLAUDE.md` (paridad `scripts/` ↔ `crawler_service/`)
