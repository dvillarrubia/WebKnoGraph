# WebKnoGraph — Guía de Instalación y Desarrollo

## Requisitos Previos

- **Docker** >= 24.0 y **Docker Compose** >= 2.20
- **Git**
- ~4 GB RAM libre (el modelo de embeddings carga en memoria)
- ~2 GB disco para imágenes Docker + modelos HuggingFace

## Instalación Rápida

### 1. Clonar el repositorio

```bash
git clone https://github.com/dvillarrubia/WebKnoGraph.git
cd WebKnoGraph
git checkout feature/crawler-docker-separation
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
```

Edita `.env` y añade tu API key de OpenAI (necesaria para el chat RAG):

```
OPENAI_API_KEY=sk-proj-tu-clave-aqui
```

### 3. Levantar los servicios

```bash
docker-compose -f docker-compose.local.yml up --build -d
```

Esto levanta 4 containers:

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| **wkg-app** | [localhost:8080](http://localhost:8080) | Dashboard + API Graph-RAG |
| **wkg-crawler** | [localhost:8081](http://localhost:8081) | Servicio de crawling (Crawl4AI) |
| **wkg-postgres** | localhost:54322 | PostgreSQL 16 + pgvector |
| **wkg-neo4j** | [localhost:7474](http://localhost:7474) | Neo4j 5.20 (browser) / bolt:7687 |

### 4. Verificar que todo funciona

```bash
# Health check
curl http://localhost:8080/api/v1/health
# Debería devolver: {"status":"healthy","service":"graph-rag"}

# Ver estado de containers
docker-compose -f docker-compose.local.yml ps
# Los 4 deben estar "healthy"
```

### 5. Abrir el Dashboard

Navega a [http://localhost:8080](http://localhost:8080)

---

## Flujo de Trabajo Completo

### Paso 1: Crawlear un sitio web

Desde el Dashboard > **Crawler**:
1. Introduce la URL del sitio (ej: `https://www.uoc.edu`)
2. Configura: max páginas, delay, profundidad
3. Haz clic en "Iniciar Crawl"
4. Espera a que termine (monitoriza el progreso en la pestaña)

O vía API:
```bash
curl -X POST http://localhost:8080/api/v1/dashboard/crawler/start \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com","max_pages":50,"delay":0.5}'
```

### Paso 2: Ingestar los datos

Desde el Dashboard > **Ingest**:
1. Selecciona el crawl completado
2. Pon nombre y dominio del cliente
3. Selecciona modelo de embeddings (por defecto: español 768-dim)
4. Haz clic en "Iniciar Ingestión"

Esto genera embeddings para cada página y chunk, crea nodos en Neo4j, y calcula PageRank.

> **Nota**: La primera ingestión descarga el modelo de embeddings (~500MB). Siguientes ejecuciones usan cache.

### Paso 3: Usar las funcionalidades

- **Chat RAG**: Pregunta sobre el contenido del sitio web
- **Búsqueda**: Busca páginas por similitud semántica
- **Graph Explorer**: Visualiza el grafo de enlaces con PageRank
- **Interlinking**: Encuentra oportunidades de enlazado interno

---

## Estructura del Proyecto

```
WebKnoGraph/
├── docker-compose.local.yml    # Stack de desarrollo
├── docker-compose.rag.yml      # Stack de producción
├── init_db.sql                 # Schema PostgreSQL (pgvector)
├── .env.example                # Template de variables
│
├── graph_rag/                  # API + Dashboard
│   ├── Dockerfile
│   ├── api/
│   │   ├── main.py             # FastAPI app
│   │   ├── routes.py           # Todos los endpoints
│   │   ├── models.py           # Pydantic models
│   │   └── dependencies.py     # Inyección de dependencias
│   ├── config/settings.py      # Configuración
│   ├── db/
│   │   ├── supabase_client.py  # PostgreSQL + pgvector
│   │   └── neo4j_client.py     # Neo4j driver
│   ├── services/
│   │   ├── rag_service.py      # RAG pipeline
│   │   ├── ingest_service.py   # Parquet → DBs
│   │   ├── embedding_service.py
│   │   ├── chunking_service.py
│   │   ├── agentic_rag_service.py
│   │   └── community_service.py
│   └── static/index.html       # Dashboard UI
│
├── crawler_service/            # Crawler container
│   ├── Dockerfile
│   ├── main.py                 # FastAPI wrapper
│   └── crawl4ai_advanced.py    # Crawler script
│
├── scripts/                    # Scripts locales (mirror de crawler_service)
│   └── crawl4ai_advanced.py
│
├── ontologia/                  # SEOntology module
│   ├── ontology_parser.py      # Parse seovoc.ttl
│   ├── neo4j_schema.py         # Neo4j schema
│   ├── ingest_ontology.py      # Ingest ontology data
│   └── seontology/seovoc.ttl   # SEO ontology definition
│
└── data/crawl4ai_data/         # Datos de crawl (gitignored)
```

---

## Desarrollo

### Modificar el Dashboard / API

Los archivos de `graph_rag/` se copian al container en build. Para ver cambios:

```bash
# Rebuild y recrear solo wkg-app
docker-compose -f docker-compose.local.yml build wkg-app
docker-compose -f docker-compose.local.yml up -d wkg-app
```

### Modificar el Crawler

El directorio `crawler_service/` está montado como volumen, así que basta con restart:

```bash
docker-compose -f docker-compose.local.yml restart wkg-crawler
```

> **Importante**: Los cambios en el crawler deben hacerse en AMBOS archivos:
> - `crawler_service/crawl4ai_advanced.py`
> - `scripts/crawl4ai_advanced.py`

### Acceso directo a las bases de datos

**PostgreSQL**:
```bash
psql -h localhost -p 54322 -U postgres -d postgres
# Password: postgres123
```

**Neo4j Browser**: [http://localhost:7474](http://localhost:7474)
- User: `neo4j`
- Password: `neo4j123`

### Ver logs

```bash
# Todos los servicios
docker-compose -f docker-compose.local.yml logs -f

# Solo la app
docker-compose -f docker-compose.local.yml logs -f wkg-app

# Solo errores
docker-compose -f docker-compose.local.yml logs wkg-app 2>&1 | grep -i error
```

### Reset completo (borrar datos)

```bash
docker-compose -f docker-compose.local.yml down -v
docker-compose -f docker-compose.local.yml up --build -d
```

> Esto borra todos los volúmenes (PostgreSQL, Neo4j, crawl data, HuggingFace cache).

---

## API Endpoints Principales

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/dashboard/clients` | Listar clientes |
| POST | `/api/v1/dashboard/crawler/start` | Iniciar crawl |
| GET | `/api/v1/dashboard/crawler/status` | Estado del crawl |
| POST | `/api/v1/dashboard/ingest` | Ingestar datos |
| POST | `/api/v1/dashboard/interlinking` | Sugerencias de interlinking |
| POST | `/api/v1/dashboard/related` | Backlinks/outlinks de una URL |
| GET | `/api/v1/dashboard/graph/{client_id}` | Datos del grafo |
| POST | `/api/v1/query` | Chat RAG (requiere X-API-Key) |
| POST | `/api/v1/search` | Búsqueda semántica (requiere X-API-Key) |

---

## Troubleshooting

| Problema | Solución |
|----------|----------|
| Container no arranca | `docker-compose -f docker-compose.local.yml logs wkg-app` |
| "PermissionError" en embeddings | Rebuild: `docker-compose -f docker-compose.local.yml build wkg-app` |
| Chat RAG no responde | Verificar OPENAI_API_KEY en `.env` y recrear: `docker-compose up -d wkg-app` |
| Ingest muy lento | Normal en CPU (~10-20 min por 100 páginas). Primera vez descarga modelo. |
| Neo4j no tiene datos | Ejecutar ingest desde Dashboard > Ingest |
| Puerto ocupado | Cambiar puertos en docker-compose.local.yml |
