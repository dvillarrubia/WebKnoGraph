# WebKnoGraph - Guia de despliegue en Portainer

## Indice

1. [Resumen de arquitectura](#resumen-de-arquitectura)
2. [Que esta desactualizado](#que-esta-desactualizado)
3. [Servicios del stack completo](#servicios-del-stack-completo)
4. [Imagenes a construir en CI/CD](#imagenes-a-construir-en-cicd)
5. [Volumenes y datos persistentes](#volumenes-y-datos-persistentes)
6. [Variables de entorno](#variables-de-entorno)
7. [Docker Compose para Portainer](#docker-compose-para-portainer)
8. [init_db.sql (PostgreSQL bootstrap)](#init_dbsql)
9. [Orden de arranque y healthchecks](#orden-de-arranque-y-healthchecks)
10. [Post-despliegue: verificaciones](#post-despliegue-verificaciones)
11. [Notas sobre recursos y hardware](#notas-sobre-recursos-y-hardware)

---

## Resumen de arquitectura

```
                         +-----------------+
                         |   Dashboard UI  |
                         |  (embebido en   |
                         |  graph-rag-api) |
                         +--------+--------+
                                  |
                           :8080 (HTTP)
                                  |
              +-------------------+--------------------+
              |                                        |
    +---------v----------+               +-------------v-----------+
    |   graph-rag-api    |  HTTP :8081   |   crawler-service       |
    |   (FastAPI +       +-------------->|   (FastAPI + Crawl4AI   |
    |    Gunicorn)       |               |    + Playwright/Chrome) |
    +----+----------+----+               +------------+------------+
         |          |                                  |
    bolt:|     SQL  |                                  | (parquet)
    :7687|    :5432 |                                  |
         |          |                         +--------v--------+
   +-----v---+ +---v-----------+              |  Vol: crawl_data |
   |  Neo4j  | |  PostgreSQL   |              |  (compartido)    |
   |  5.20   | |  16 + pgvector|              +---------+--------+
   +---------+ +---------------+                        |
                                              (ambos containers leen/escriben)
```

**4 servicios, 4 volumenes, 1 red:**

| Servicio | Funcion | Puerto |
|----------|---------|--------|
| `wkg-app` | API principal + Dashboard + RAG + Embeddings | 8080 |
| `wkg-crawler` | Crawl4AI + Playwright/Chromium | 8081 (interno) |
| `wkg-postgres` | PostgreSQL 16 + pgvector (vector search) | 5432 (interno) |
| `wkg-neo4j` | Neo4j 5.20 (grafo de enlaces, PageRank) | 7687 (interno) |

---

## Que esta desactualizado

### Nomenclatura "Supabase"

El codigo usa variables como `SUPABASE_URL`, `SUPABASE_DB_HOST`, etc. pero **NO se usa Supabase real**. Es simplemente **PostgreSQL + pgvector**. Las variables mantienen el nombre por legado, pero la conexion es directa via `asyncpg` a PostgreSQL:

- `docker-compose.rag.yml` referencia `seo-supabase-db` y `seo-supabase-api` (PostgREST) que **no se usan** - la API conecta directo a PostgreSQL
- `docker-compose.local.yml` ya corrigio esto: usa `pgvector/pgvector:pg16` directamente
- `SUPABASE_URL` y `SUPABASE_SERVICE_KEY` tienen default vacio en `settings.py` — **no hace falta definirlos**
- La conexion real usa solo `SUPABASE_DB_HOST`, `SUPABASE_DB_PORT`, etc. via `asyncpg`

**Conclusion:** No necesitas Supabase ni PostgREST. Solo PostgreSQL con la extension pgvector.

### docker-compose.rag.yml

- Referencia servicios externos (`seo-supabase-db`, `seo-supabase-api`, `seo-neo4j`) que asume ya estan corriendo
- Usa `version: '3.8'` (deprecated en Docker Compose V2)
- No incluye las DBs en el stack - no es autocontenido

### deploy/docker-compose.prod.yml

- Solo tiene los 2 containers de aplicacion, sin DBs
- Depende de `webknograph-network` externa
- No incluye `init_db.sql` ni healthchecks de DBs

### Embedding dimension (CORREGIDO)

- Habia un mismatch: `init_db.sql` usaba `vector(768)`, la migracion `001_create_schema.sql` usaba `vector(1024)`
- **Ya corregido:** todo unificado a `vector(768)` con modelo espanol por defecto

### Dependencias produccion (CORREGIDO)

- `requirements-rag.txt` no incluia `networkx` ni `python-louvain` (usados por `community_service.py`)
- **Ya corregido:** ambos paquetes anadidos a `requirements-rag.txt`
- El Dockerfile de desarrollo (`graph_rag/Dockerfile`) los instalaba aparte, por eso funcionaba en local

### settings.py (CORREGIDO)

- `SUPABASE_URL` era campo required (`Field(...)`) pero no se usa en las conexiones
- **Ya corregido:** cambiado a `Field(default="")` — no hace falta definirlo en el stack

### setup_db.py (CORREGIDO)

- El parser de Cypher parseaba linea a linea, truncando sentencias multilinea de Neo4j
- La migracion SQL no era idempotente (CREATE POLICY fallaba en re-ejecucion)
- **Ya corregido:** parser junta bloques multilinea, policies usan DROP IF EXISTS previo

### READMEs

- Mencionan Supabase como si fuera un servicio separado
- El CLAUDE.md es la documentacion mas actualizada y fiable

---

## Servicios del stack completo

### 1. PostgreSQL 16 + pgvector (`wkg-postgres`)

- **Imagen:** `pgvector/pgvector:pg16`
- **Proposito:** Almacena paginas, chunks, embeddings vectoriales, enlaces, clientes multi-tenant, conversaciones
- **Tablas creadas por `init_db.sql`:**
  - `rag_clients` - Tenants (UUID, api_key, domain)
  - `rag_pages` - Paginas con embedding vector(768) + pagerank + HITS scores
  - `rag_chunks` - Chunks semanticos con embedding vector(768)
  - `rag_links` - Enlaces entre paginas
  - `rag_conversations` / `rag_messages` - Historial de chat
- **Extensiones:** `vector` (pgvector), `pgcrypto`
- **Indices:** HNSW para busqueda vectorial (cosine similarity)
- **Recurso:** ~256MB RAM minimo, mas con datasets grandes

### 2. Neo4j 5.20 Community (`wkg-neo4j`)

- **Imagen:** `neo4j:5.20-community`
- **Proposito:** Grafo de enlaces web para PageRank, HITS, expansion de contexto, pathfinding
- **Nodos:** `Page` (url, title, pagerank, hub_score, authority_score, community_id)
- **Relaciones:** `LINKS_TO` (anchor_text, location, weight)
- **Algoritmos:** PageRank, HITS, Louvain community detection (via NetworkX)
- **Recurso:** ~512MB RAM minimo (configurable con `NEO4J_server_memory_heap_initial__size`)

### 3. Crawler Service (`wkg-crawler`)

- **Imagen:** Build propio (CI/CD) desde `deploy/Dockerfile.crawler`
- **Base:** `python:3.11-slim` + Playwright + Chromium
- **Proposito:** Crawling web con rendering JS, extraccion de contenido y enlaces
- **API interna (solo accesible desde la red Docker):**
  - `POST /crawl/start` - Iniciar crawl
  - `POST /crawl/stop` - Parar crawl
  - `GET /crawl/status` - Estado actual
  - `GET /crawl/logs` - Logs JSONL
  - `GET /crawl/crawls` - Listar crawls disponibles
  - `GET /health` - Healthcheck
- **Output:** Archivos Parquet en volumen compartido `crawl_data`
- **Recurso:** ~1GB RAM (Chromium es pesado), 1 worker unico (`-w 1`), timeout 300s

### 4. Graph-RAG API (`wkg-app`)

- **Imagen:** Build propio (CI/CD) desde `deploy/Dockerfile.api`
- **Base:** `python:3.11-slim` + Gunicorn + sentence-transformers + torch
- **Proposito:** API REST + Dashboard web + motor RAG completo
- **Pipeline RAG:** Query → Embedding local → pgvector search → Neo4j graph expansion → Reranking → OpenAI LLM → Respuesta
- **Dashboard:** Servido como ficheros estaticos en `/` (SPA embebido)
- **API docs:** `/docs` (Swagger) y `/redoc`
- **~40 endpoints:** Admin, client (con API key), dashboard, crawler proxy
- **Modelos ML cargados en memoria:**
  - Embedding: `hiiamsid/sentence_similarity_spanish_es` (~500MB)
  - Reranker (opcional): `BAAI/bge-reranker-v2-m3` (~1.1GB)
- **Recurso:** ~2-4GB RAM (modelos ML), 2 workers Gunicorn

---

## Imagenes a construir en CI/CD

Solo 2 imagenes custom. Las DBs usan imagenes oficiales.

### Image 1: `gitlab.lin3s.com:5050/dvillarrubia/webknograph/api`

- **Dockerfile:** `deploy/Dockerfile.api`
- **Contiene:** FastAPI + Gunicorn + sentence-transformers + torch + graph_rag/
- **Tags generados:** `wkg-v0.1.0`, `latest`, `wkg-v0.1.0-prod`, `prod`

### Image 2: `gitlab.lin3s.com:5050/dvillarrubia/webknograph/crawler`

- **Dockerfile:** `deploy/Dockerfile.crawler`
- **Contiene:** FastAPI + Gunicorn + Crawl4AI + Playwright + Chromium + crawler_service/
- **Tags generados:** `wkg-v0.1.0`, `latest`, `wkg-v0.1.0-prod`, `prod`

### Flujo de release

```bash
# 1. Crear tag de version (interactivo: patch/minor/major)
bash tools/release.sh

# 2. Pushear - esto dispara el pipeline automaticamente
git push && git push --tags

# 3. (Opcional) Promover a PROD desde GitLab UI → deploy-prod (manual)
```

Ver `.gitlab-ci.yml` para detalles del pipeline.

---

## Volumenes y datos persistentes

| Directorio host | Montado en | Proposito | Compartido |
|-----------------|-----------|-----------|------------|
| `default/pgdata` | `/var/lib/postgresql/data` | Datos PostgreSQL + pgvector | No (solo postgres) |
| `default/neo4jdata` | `/data` | Datos Neo4j | No (solo neo4j) |
| `default/crawl-data` | `/app/data/crawl4ai_data` | Parquets de crawls | Si (crawler + app) |
| `default/hf-cache` | `/home/appuser/.cache/huggingface` | Modelos ML descargados | No (solo app) |

Todos los datos persisten en `/home/lin3s/webknograph/default/` en el host.

**Importante sobre `hf-cache`:** Sin este directorio, cada restart de `wkg-app` descargaria ~500MB-1.6GB de modelos de HuggingFace.

**Importante sobre `crawl-data`:** Compartido entre crawler y app. El crawler escribe Parquets, la app los lee para ingest. Ambos necesitan read-write.

---

## Variables de entorno

### Secretos (configurar en Portainer como environment o .env)

| Variable | Obligatoria | Ejemplo | Descripcion |
|----------|-------------|---------|-------------|
| `OPENAI_API_KEY` | Si (para RAG queries) | `sk-proj-...` | API key de OpenAI para generar respuestas |
| `POSTGRES_PASSWORD` | Si | `un-password-seguro` | Password de PostgreSQL |
| `NEO4J_PASSWORD` | Si | `otro-password-seguro` | Password de Neo4j |
| `GEMINI_API_KEY` | No | `AIza...` | Fallback a Google Gemini (opcional) |

### Configuracion (valores por defecto sensatos, ajustables)

| Variable | Default | Descripcion |
|----------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | `hiiamsid/sentence_similarity_spanish_es` | Modelo de embeddings local |
| `EMBEDDING_DIMENSION` | `768` | Dimension de vectores (**debe coincidir con init_db.sql**) |
| `OPENAI_MODEL` | `gpt-4o` | Modelo LLM para respuestas |
| `RAG_USE_RERANKING` | `true` | Habilitar reranking (usa ~1GB RAM extra) |
| `RERANKING_MODEL` | `BAAI/bge-reranker-v2-m3` | Modelo de reranking |
| `WEB_CONCURRENCY` | `2` | Workers de Gunicorn en wkg-app |
| `RAG_TOP_K_VECTORS` | `10` | Resultados vectoriales a recuperar |
| `RAG_GRAPH_HOPS` | `2` | Saltos de grafo para expandir contexto |
| `RAG_MAX_CONTEXT_PAGES` | `15` | Max paginas en contexto LLM |
| `RAG_MIN_SIMILARITY` | `0.5` | Umbral de similitud coseno |
| `DEBUG` | `false` | Modo debug (no usar en produccion) |

---

## Docker Compose para Portainer

Copiar el YAML como stack en Portainer y configurar el `.env` en la seccion "Environment variables" > "Load from .env file".

> **Registry privado:** Antes de desplegar, configurar las credenciales en
> **Registries** > **Add registry** > **GitLab** con URL `gitlab.lin3s.com:5050`.

### Fichero `.env` (pegar en Portainer > Environment variables)

```env
# --- Secretos (OBLIGATORIO rellenar) ---
POSTGRES_PASSWORD=pon-tu-password-postgres
NEO4J_PASSWORD=pon-tu-password-neo4j
OPENAI_API_KEY=sk-proj-tu-clave-openai

# --- Embeddings ---
EMBEDDING_MODEL_NAME=hiiamsid/sentence_similarity_spanish_es
EMBEDDING_DIMENSION=768
EMBEDDING_BATCH_SIZE=32

# --- LLM ---
OPENAI_MODEL=gpt-4o

# --- RAG ---
RAG_TOP_K_VECTORS=10
RAG_GRAPH_HOPS=2
RAG_MAX_CONTEXT_PAGES=15
RAG_MIN_SIMILARITY=0.5
RAG_CONTEXT_MAX_TOKENS=8000
RAG_USE_RERANKING=true
RERANKING_MODEL=BAAI/bge-reranker-v2-m3

# --- App ---
WEB_CONCURRENCY=2
DEBUG=false
```

### Docker Compose (pegar en Portainer > Web editor)

```yaml
# =============================================================================
# WebKnoGraph - Stack completo para Portainer
# Todas las variables sensibles y configurables vienen del .env
# =============================================================================

services:

  # --------------------------------------------------------------------------
  # PostgreSQL 16 + pgvector (Vector DB)
  # --------------------------------------------------------------------------
  wkg-postgres:
    image: pgvector/pgvector:pg16
    container_name: wkg-postgres
    networks:
      - wkg-internal
    environment:
      POSTGRES_DB: postgres
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - /home/lin3s/webknograph/default/pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # --------------------------------------------------------------------------
  # Neo4j 5.20 Community (Graph DB)
  # --------------------------------------------------------------------------
  wkg-neo4j:
    image: neo4j:5.20-community
    container_name: wkg-neo4j
    networks:
      - wkg-internal
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_server_memory_heap_initial__size: 256m
      NEO4J_server_memory_heap_max__size: 512m
    volumes:
      - /home/lin3s/webknograph/default/neo4jdata:/data
    healthcheck:
      test: ["CMD-SHELL", "wget --no-verbose --tries=1 --spider http://localhost:7474 || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 5
      start_period: 30s
    restart: unless-stopped

  # --------------------------------------------------------------------------
  # Crawler Service (Crawl4AI + Playwright/Chromium)
  # --------------------------------------------------------------------------
  wkg-crawler:
    image: gitlab.lin3s.com:5050/dvillarrubia/webknograph/crawler:latest
    container_name: wkg-crawler
    networks:
      - wkg-internal
    environment:
      CRAWL_OUTPUT_DIR: /app/data/crawl4ai_data
    volumes:
      - /home/lin3s/webknograph/default/crawl-data:/app/data/crawl4ai_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped

  # --------------------------------------------------------------------------
  # Graph-RAG API + Dashboard (Main Application)
  # --------------------------------------------------------------------------
  wkg-app:
    image: gitlab.lin3s.com:5050/dvillarrubia/webknograph/api:latest
    container_name: wkg-app
    networks:
      - wkg-internal
      - proxy         # Red externa para reverse proxy (Traefik/Nginx)
    ports:
      - "8080:8080"
    environment:
      # --- App ---
      DEBUG: ${DEBUG}
      WEB_CONCURRENCY: ${WEB_CONCURRENCY}

      # --- Crawler (container interno) ---
      CRAWLER_SERVICE_URL: http://wkg-crawler:8081

      # --- PostgreSQL (prefijo SUPABASE_ por legado, NO usa Supabase real) ---
      SUPABASE_DB_HOST: wkg-postgres
      SUPABASE_DB_PORT: "5432"
      SUPABASE_DB_NAME: postgres
      SUPABASE_DB_USER: postgres
      SUPABASE_DB_PASSWORD: ${POSTGRES_PASSWORD}

      # --- Neo4j ---
      NEO4J_URI: bolt://wkg-neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: ${NEO4J_PASSWORD}

      # --- OpenAI ---
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      OPENAI_MODEL: ${OPENAI_MODEL}

      # --- Embeddings ---
      EMBEDDING_MODEL_NAME: ${EMBEDDING_MODEL_NAME}
      EMBEDDING_DIMENSION: ${EMBEDDING_DIMENSION}
      EMBEDDING_BATCH_SIZE: ${EMBEDDING_BATCH_SIZE}

      # --- RAG ---
      RAG_TOP_K_VECTORS: ${RAG_TOP_K_VECTORS}
      RAG_GRAPH_HOPS: ${RAG_GRAPH_HOPS}
      RAG_MAX_CONTEXT_PAGES: ${RAG_MAX_CONTEXT_PAGES}
      RAG_MIN_SIMILARITY: ${RAG_MIN_SIMILARITY}
      RAG_CONTEXT_MAX_TOKENS: ${RAG_CONTEXT_MAX_TOKENS}
      RAG_USE_RERANKING: ${RAG_USE_RERANKING}
      RERANKING_MODEL: ${RERANKING_MODEL}
    volumes:
      - /home/lin3s/webknograph/default/crawl-data:/app/data/crawl4ai_data
      - /home/lin3s/webknograph/default/hf-cache:/home/appuser/.cache/huggingface
    depends_on:
      wkg-postgres:
        condition: service_healthy
      wkg-neo4j:
        condition: service_healthy
      wkg-crawler:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
    restart: unless-stopped

networks:
  wkg-internal:
    # Red interna del stack (postgres, neo4j, crawler, app)
  proxy:
    external: true
    # Red externa pre-existente para reverse proxy (Traefik/Nginx)
    # Debe existir antes de desplegar: docker network create proxy
```

---

## Inicializacion de bases de datos

**La app NO auto-inicializa las DBs.** Hay que ejecutar el setup manualmente tras el primer despliegue.

### Script de setup (inicializa AMBAS DBs de un golpe)

```bash
# Ejecutar desde dentro del container wkg-app (que ya tiene las env vars y dependencias):
docker exec wkg-app python -m graph_rag.scripts.setup_db
```

Esto ejecuta `graph_rag/scripts/setup_db.py`, que:

1. **PostgreSQL** - Ejecuta `graph_rag/db/migrations/001_create_schema.sql`:
   - Extension `vector` (pgvector)
   - Tablas: `rag_clients`, `rag_pages`, `rag_chunks`, `rag_links`, `rag_conversations`, `rag_messages`
   - Indices HNSW para busqueda vectorial
   - Funcion `search_pages_by_similarity()`
   - Triggers de `updated_at`
   - RLS policies

2. **Neo4j** - Ejecuta `graph_rag/db/neo4j/001_constraints.cypher`:
   - Constraint `page_unique_url` (client_id + url)
   - Constraint `client_unique_id`
   - Indices: `page_client_id`, `page_pagerank`, `page_folder_depth`, `page_client_metrics`

> **Nota:** El `init_db.sql` de la raiz es una copia legacy para docker-compose.local.yml.
> La fuente de verdad es `graph_rag/db/migrations/001_create_schema.sql`.

### Dimension de vectores

La migracion y el docker-compose estan alineados en `vector(768)` con el modelo
espanol `hiiamsid/sentence_similarity_spanish_es`. Si necesitas usar el modelo
multilingue (`intfloat/multilingual-e5-large`, 1024 dims), debes cambiar:
1. `EMBEDDING_DIMENSION` en el docker-compose a `1024`
2. Las columnas `vector(768)` en la migracion SQL a `vector(1024)`
3. Regenerar todos los embeddings

---

## Orden de arranque y healthchecks

```
1. wkg-postgres   (healthcheck: pg_isready, ~10s)
2. wkg-neo4j      (healthcheck: wget http://localhost:7474, ~30s)
3. wkg-crawler    (healthcheck: curl /health, ~60s por Chromium)
4. wkg-app        (healthcheck: curl /health, ~120s por carga de modelos ML)
```

El `depends_on` con `condition: service_healthy` garantiza el orden.

**Primer arranque:** `wkg-app` tardara 2-5 minutos extra descargando modelos de HuggingFace al volumen `wkg-hf-cache`. Los siguientes arranques son rapidos (~30s).

---

## Post-despliegue: verificaciones

### 1. Verificar que todos los containers estan healthy

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
# Esperar a ver "healthy" en los 4 containers
```

### 2. Inicializar las bases de datos (solo primer despliegue)

```bash
# Inicializa PostgreSQL (tablas + indices + pgvector) y Neo4j (constraints + indices)
docker exec wkg-app python -m graph_rag.scripts.setup_db
```

### 3. Verificar health de la API

```bash
curl http://localhost:8080/api/v1/health
# Deberia devolver {"status":"ok",...}
```

### 4. Acceder al Dashboard

Abrir `http://tu-servidor:8080/` en el navegador.

### 5. Crear primer cliente (tenant)

```bash
curl -X POST http://localhost:8080/api/v1/admin/clients \
  -H "Content-Type: application/json" \
  -d '{"name":"Mi Sitio","domain":"www.misitio.com"}'
# Devuelve un api_key que necesitaras para queries
```

### 6. Lanzar primer crawl

```bash
curl -X POST http://localhost:8080/api/v1/dashboard/crawler/start \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.misitio.com","max_pages":10,"delay":0.5}'
```

### 7. Ingestar datos (tras crawl completado)

Desde el Dashboard UI o via API:
```bash
curl -X POST http://localhost:8080/api/v1/dashboard/ingest \
  -H "Content-Type: application/json" \
  -d '{"crawl_name":"www_misitio_com","client_name":"Mi Sitio","client_domain":"www.misitio.com"}'
```

---

## Notas sobre recursos y hardware

### Requisitos minimos

| Recurso | Minimo | Recomendado |
|---------|--------|-------------|
| RAM total | 4 GB | 8 GB |
| CPU | 2 cores | 4 cores |
| Disco | 10 GB | 50 GB (crawls + modelos) |
| GPU | No necesaria | Opcional (acelera embeddings) |

### Desglose de RAM por servicio

| Servicio | RAM estimada |
|----------|-------------|
| `wkg-postgres` | 256MB - 1GB |
| `wkg-neo4j` | 512MB - 1GB |
| `wkg-crawler` | 512MB - 1.5GB (Chromium) |
| `wkg-app` (sin reranking) | 1.5GB - 2.5GB |
| `wkg-app` (con reranking) | 2.5GB - 4GB |

### Para reducir consumo de RAM

- Deshabilitar reranking: `RAG_USE_RERANKING=false` (ahorra ~1GB)
- Reducir workers: `WEB_CONCURRENCY=1` (ahorra ~500MB)
- Limitar Neo4j heap: `NEO4J_server_memory_heap_max__size=256m`

### Puertos expuestos

Solo se expone el puerto **8080** (API + Dashboard). Las DBs y el crawler son internos a la red Docker.

Si necesitas acceso directo a Neo4j Browser o PostgreSQL para debug:

```yaml
# Anadir temporalmente a wkg-neo4j:
    ports:
      - "7474:7474"  # Neo4j Browser
      - "7687:7687"  # Bolt protocol

# Anadir temporalmente a wkg-postgres:
    ports:
      - "54322:5432"  # PostgreSQL
```

---

## Troubleshooting

| Problema | Causa probable | Solucion |
|----------|---------------|----------|
| wkg-app no arranca (timeout healthcheck) | Descargando modelos ML | Esperar 5min, revisar logs: `docker logs wkg-app` |
| Error "relation rag_pages does not exist" | setup_db no ejecutado | `docker exec wkg-app python -m graph_rag.scripts.setup_db` |
| Error "vector type does not exist" | Extension pgvector no cargada | Verificar imagen `pgvector/pgvector:pg16` |
| Crawl se cuelga | `networkidle` (no deberia pasar con imagen oficial) | Revisar logs: `docker logs wkg-crawler` |
| "OPENAI_API_KEY not set" | Variable no configurada | Configurar en environment de Portainer |
| wkg-app consume mucha RAM | Reranking model cargado | `RAG_USE_RERANKING=false` para deshabilitar |
| Neo4j healthcheck falla | Neo4j aun arrancando | Esperar 30s+, revisar logs: `docker logs wkg-neo4j` |
| Embedding dimension mismatch | Migracion SQL no coincide con env | `EMBEDDING_DIMENSION` y `vector(N)` en SQL deben coincidir |
| Neo4j password no funciona | Caracteres especiales en password | Usar solo alfanumericos en `NEO4J_AUTH` |
