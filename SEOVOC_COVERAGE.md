# SEOVOC Coverage Audit — qué mapeamos y qué no

> **Fecha:** 2026-04-08
> **Ontología auditada:** `D:/KGRAG UOC/ontologia/seontology/seovoc.ttl` v0.0.1
> **Código auditado:** `D:/KGRAG UOC/WebKnoGraph/` (ingest_ontology.py, neo4j_schema.py, schema_extractor.py, crawl4ai_advanced.py)
> **Método:** lectura exhaustiva del TTL + cross-check contra implementación real

---

## Cobertura global

| Categoría | Total seovoc | Implementado | % |
|---|---|---|---|
| **Classes** | 11 (+2 de schema.org) | 7 | **64%** |
| **Object Properties** | 21 | 14 | **67%** |
| **Data Properties** | 77 | 32 | **42%** |

> La clase **`Query` sola aporta 41 data properties (53% del total)** y está al 0%. Es el gap individual más grande.

---

## Cobertura por clase

| Clase seovoc | Label Neo4j | Nodos creados | Data props | % | Sprint |
|---|---|---|---|---|---|
| WebPage | `:Page:SeoWebPage` | ✅ | 11/19 | 58% | 1 |
| Chunk | `:SeoChunk` | ✅ | 7/8 | 88% | 1 |
| URL | `:SeoURL` | ✅ | 1/1 | 100% | 1 |
| Link | `:SeoLink` | ✅ | 4/6 | 67% | 1 |
| LinkGroup | `:SeoLinkGroup` | ✅ | 2/3 | 67% | 1 |
| AnchorText | `:SeoAnchorText` | ✅ | 1/1 | 100% | 1 |
| Schema | `:SeoSchema` | ✅ | 1/1 | 100% | 2 |
| Thing | `:SeoThing` | ✅ | 2/2 | 100% | 2 |
| **Query** | `:SeoQuery` | ❌ | **0/41** | **0%** | ❌ no iniciado |
| **Persona** | `:SeoPersona` | ❌ | **0/3** | **0%** | ❌ no iniciado |
| **PageGroup** | `:SeoPageGroup` | ❌ | **0/2** | **0%** | ❌ no iniciado |
| schema:Language | — | ❌ (solo string) | — | — | — |

---

## Gap table — Data Properties faltantes

### WebPage — 8 props faltantes (de 19)

| Property | Tipo | Estado | Nota |
|---|---|---|---|
| `bounceRate` | decimal | ❌ | Métrica GA4, necesita integración GA |
| `clicks` | long | ❌ | Agregado a nivel página desde GSC |
| `ctr` | double | ❌ | Agregado GSC |
| `impressions` | long | ❌ | Agregado GSC |
| `position` | double | ❌ | Agregado GSC |
| `status` | int | ❌ | HTTP status — capturar en crawler |
| `intent` | string | ❌ | Clasificación página (nav/transactional/info) — LLM Sprint 3 |
| `embedding` | string | ❌ | Vive en Supabase pgvector, no replicado a Neo4j (OK) |
| `embeddingText` | string | ❌ | No almacenado |
| `sourceCode` | string | 🟡 | HTML en parquet pero no en Neo4j (por tamaño) |
| `menu` | boolean | ❌ | ¿La página es un menú? No implementado |
| `isCrawlable` | boolean | 🟡 | **HARDCODED `true`** — arreglar con robots meta en FIX-1/2 |

### Chunk — 1 prop faltante (de 8)

| Property | Tipo | Estado | Nota |
|---|---|---|---|
| `tokenCount` | integer | ❌ | Requiere tokenizer post-chunking |
| `chunkStrategy` | string | 🟡 | HARDCODED `"semantic"` |
| `schema:inLanguage` | string | 🟡 | HARDCODED |

### Link — 3 props faltantes (de 6)

| Property | Tipo | Estado | Nota |
|---|---|---|---|
| `schema:keywords` | string | ❌ | Keywords asociadas al link |
| `schema:thumbnailUrl` | string | ❌ | Para rich previews |
| `noFollow` | boolean | ❌ | **Fácil de extraer** — `rel="nofollow"` en crawler |

### LinkGroup — 1 prop faltante (de 3)

| Property | Tipo | Estado | Nota |
|---|---|---|---|
| `schema:inLanguage` | string | ❌ | Union domain no asignado |

### Query — 41 props faltantes (de 41) — **CLASE ENTERA SIN IMPLEMENTAR**

**Clicks (6 props):** `clicks7Days`, `clicks28Days`, `clicks3Months`, `clicks7DaysTo28DaysTrend`, `clicks28DaysTo3MonthsTrend`, `clicks7DaysTo3MonthsTrend`

**Impressions (6 props):** `impressions7Days`, `impressions28Days`, `impressions3Months`, `impressions7DaysTo28DaysTrend`, `impressions28DaysTo3MonthsTrend`, `impressions7DaysTo3MonthsTrend`

**CTR (6 props):** `ctr7Days`, `ctr28Days`, `ctr3Months`, `ctr7DaysTo28DaysTrend`, `ctr28DaysTo3MonthsTrend`, `ctr7DaysTo3MonthsTrend`

**Position (6 props):** `position7Days`, `position28Days`, `position3Months`, `position7DaysTo28DaysTrend`, `position28DaysTo3MonthsTrend`, `position7DaysTo3MonthsTrend`

**Score (3 props):** `score7Days`, `score28Days`, `score3Months` — overall score, fórmula no definida en TTL

**Meta (2 props):** `dateCreated`, `keywordType` (primario/secundario)

**Clasificación (2 props con enums):**
- `queryType` (13 valores): `Popular`, `Long-tail`, `Dynamic`, `Multi-hop`, `Analytical`, `Commonsense`, `Causal`, `Exploratory`, `Instructive`, `Recommendation`, `Spatio-temporal`, `Lifestyle`, `Cultural`, `Philosophical`
- `queryCategory` (4 valores): `Advice`, `Explanations`, `Facts`, `Planning`

**Lang:** `schema:inLanguage`

> **Implicación:** GSC importer (FIX-3 del workplan) debe crear `:SeoQuery` con estas 41 props. Los `queryType`/`queryCategory` requieren clasificador (LLM Sprint 3 puede cubrirlo).

### Persona — 3 props faltantes (de 3) — **CLASE ENTERA SIN IMPLEMENTAR**

| Property | Tipo | Estado |
|---|---|---|
| `ageRange` | string | ❌ |
| `preferredDevice` | string | ❌ |
| `role` | string | ❌ |

> Modelaría segmentación de audiencia. Requiere fuente de datos (GA4 demographics o definición manual).

### PageGroup — 2 props faltantes (de 2) — **CLASE ENTERA SIN IMPLEMENTAR**

| Property | Tipo | Estado |
|---|---|---|
| `groupName` | string | ❌ |
| `groupType` | string | ❌ |

> Modelaría A/B testing. Baja prioridad para UOC.

---

## Gap table — Object Properties faltantes (7 de 21)

| # | Domain | Property | Range | Estado | Razón |
|---|---|---|---|---|---|
| 1 | WebPage | `hasQuery` | Query | ❌ | SeoQuery no existe |
| 2 | WebPage | `hasPrimaryQuery` | Query | ❌ | SeoQuery no existe. **Functional** |
| 3 | WebPage | `hasPersona` | Persona | ❌ | SeoPersona no existe |
| 4 | WebPage | `link` | Link | 🟡 | Existe `:LINKS_TO` pero no relación directa `:HAS_DIRECT_LINK` |
| 5 | Link | `anchorResource` | Thing/LinkGroup | ❌ | No implementado — apunta a entidad del target |
| 6 | AnchorText | `influencedByQuery` | Query | ❌ | SeoQuery no existe |
| 7 | PageGroup | `hasPage` | WebPage | ❌ | SeoPageGroup no existe |
| 8 | Persona | `isPersonaFor` | WebPage | ❌ | Inversa — SeoPersona no existe |
| 9 | Query | `isPrimaryQueryOf` | WebPage | ❌ | Inversa — **InverseFunctional** |

---

## Hallazgos no contemplados en el WORKPLAN actual

### 🔴 HALLAZGO-A — La clase `Query` es gigante y no tiene plan completo
El workplan menciona "FIX-3 GSC importer" pero **no contaba con**:
- 41 properties distintas (no 4 como asumí)
- 6 trends calculados (ratios entre ventanas) — no datos crudos
- 3 scores "overall" sin fórmula definida en TTL
- 2 enums de clasificación (`queryType` con 13 valores, `queryCategory` con 4) que requieren clasificador LLM
- `keywordType` (primario/secundario) — decisión automatizable desde clicks28Days
- `dateCreated` — timestamp de importación

**Acción:** ampliar FIX-3 en WORKPLAN_SEO_GSC.md con las 41 props explícitas + fase separada para clasificador.

### 🔴 HALLAZGO-B — Propiedades de auditoría en `owl:Thing` (importID, importHash, importOrigin)
seovoc define 3 props de trazabilidad aplicables a cualquier nodo:
- `importID` — ID externo (ej. GSC query_id)
- `importHash` — hash de datos para detectar cambios → **útil para re-ingesta incremental**
- `importOrigin` — fuente (`crawl4ai_2026-04-08`, `gsc_2026-04-08`, `gliner_2026-04-08`)

**Acción:** añadir FIX-6 (nuevo) — propiedades de auditoría en todos los nodos para ingesta incremental.

### 🔴 HALLAZGO-C — `hasPrimaryQuery` es FunctionalProperty
seovoc declara `hasPrimaryQuery` como `owl:FunctionalProperty` y su inversa como `InverseFunctional`. Cada WebPage tiene EXACTAMENTE UNA query primaria, y cada query primaria pertenece a EXACTAMENTE una WebPage.

**Acción:** en FIX-3, al crear `:HAS_PRIMARY_QUERY` borrar previa antes de crear nueva. Ya contemplado en el workplan actual pero no estaba justificado por el TTL.

### 🟡 HALLAZGO-D — Relación `anchorResource` desconocida
```
Link -anchorResource-> (Thing | LinkGroup)
```
Un Link puede apuntar a la entidad Thing que representa el target (no solo la URL). Esto permite queries como "¿qué Things son enlazadas desde X?". Requiere:
1. Extraer Thing del target page
2. MERGE con el SeoThing ya existente por JSON-LD

**Acción:** añadir FIX-7 (nuevo) — conectar Links a Things cuando el target tiene JSON-LD. Baja prioridad.

### 🟡 HALLAZGO-E — Relación `influencedByQuery` (AnchorText → Query)
Modela que el anchor text de un link fue elegido para rankear en ciertas queries. Análisis tipo "link baiting":
```cypher
MATCH (a:SeoAnchorText)-[:INFLUENCED_BY_QUERY]->(q:SeoQuery)
WHERE q.clicks28Days > 100
RETURN a.anchorValue, q.text
```
**Acción:** FIX-8 (nuevo) — anchor text optimization. Requiere GSC + NLP matching anchor↔query. Sprint 3/4.

### 🟡 HALLAZGO-F — Clase `schema:Language` nunca se materializa
El TTL importa `schema:Language` de schema.org y la usa como range de `inLanguage`. Nosotros guardamos el idioma como **string** en `:SeoWebPage.inLanguage`, no como nodo `:Language`.

**Decisión:** mantener como string (más eficiente) — no vale la pena crear nodos Language. Documentar como divergencia consciente del TTL.

### 🟡 HALLAZGO-G — Clase `Persona` (3 props) y `PageGroup` (2 props)
Ambas clases completas sin implementar:
- `Persona` (ageRange, preferredDevice, role) — segmentación audiencia
- `PageGroup` (groupName, groupType) — A/B testing

**Decisión para UOC:** ambas baja prioridad. Añadir como FIX-9 (stub) para documentar el gap pero sin implementar hasta que el cliente lo pida.

### 🟡 HALLAZGO-H — Props de Link faltantes fáciles de cubrir
- `noFollow` (boolean) — trivial de extraer del `rel` attribute del `<a>` tag. **Incluir en FIX-1.**
- `schema:keywords` — más complejo, requiere contexto
- `schema:thumbnailUrl` — si el link apunta a imagen u og:image, capturable

**Acción:** añadir `noFollow` a FIX-1 (crawler metadata extraction).

### 🟡 HALLAZGO-I — `WebPage.status` (HTTP status code)
Crawl4AI ya devuelve el status HTTP pero no lo guardamos. Trivial.
**Acción:** añadir al parquet en FIX-1 y propagar en FIX-2.

### 🟡 HALLAZGO-J — `Chunk.tokenCount`
No se guarda. Al usar `sentence_similarity_spanish_es` (BERT-based) tenemos tokenizer disponible. Una línea.
**Acción:** añadir al chunking service. Micro-FIX.

### 🟡 HALLAZGO-K — Hardcodes a limpiar
Cross-check rápido de valores hardcoded en `ingest_ontology.py`:
- `isCrawlable=true` → leer robots meta (FIX-1/2)
- `inLanguage='es'` → leer `<html lang>` (FIX-1/2)
- `chunkStrategy='semantic'` → configurable por chunking_service
- `embeddingModel` → ya OK, viene de config

---

## Propuesta de ampliación del WORKPLAN

Añadir a `WORKPLAN_SEO_GSC.md`:

- **FIX-3 ampliado:** GSC importer con **41 properties** de `:SeoQuery`, clasificador `queryType`/`queryCategory` diferido a Sprint 3 LLM, fórmula de `scoreXDays` por definir con el usuario.
- **FIX-6 (nuevo):** Propiedades de auditoría `importID`, `importHash`, `importOrigin` en todos los nodos. Permite re-ingesta incremental.
- **FIX-7 (nuevo, baja prio):** Relación `anchorResource` — Link→Thing cuando target tiene JSON-LD.
- **FIX-8 (nuevo, Sprint 3):** Relación `influencedByQuery` — AnchorText→Query via NLP matching.
- **FIX-9 (stub, baja prio):** Clases `Persona` y `PageGroup` — documentar gap, no implementar hasta petición.
- **Micro-fixes** (añadir a FIX-1/2):
  - `noFollow` en Link
  - `status` HTTP en WebPage
  - `tokenCount` en Chunk
  - Eliminar hardcodes `isCrawlable`, `inLanguage`, `chunkStrategy`

---

## Divergencias conscientes de seovoc (no implementar)

| Item | Razón |
|---|---|
| `schema:Language` como nodo | Guardar como string es más eficiente y no perdemos queries |
| `embedding` en WebPage (Neo4j) | Vive en Supabase pgvector, replicarlo a Neo4j no aporta |
| `sourceCode` (HTML completo) en Neo4j | HTML crudo en parquet, no en grafo (tamaño) |
