# WebKnoGraph

Sistema completo de **Graph-RAG** para crawling, limpieza y consulta de contenido web con búsqueda híbrida (vectorial + grafo).

## Características

- **Web Crawler** - Extracción de contenido con Crawl4AI, soporte para JavaScript rendering
- **Manual Cleaner** - Limpieza de markdown basada en patrones con detección de plantillas HTML
- **Graph-RAG** - Búsqueda híbrida combinando vectores (pgvector) y grafo (Neo4j)
- **Dashboard UI** - Interfaz web única para gestionar todo el pipeline
- **Multi-tenant** - Soporte para múltiples clientes con aislamiento de datos

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dashboard UI (Single Page)                    │
├─────────────────────────────────────────────────────────────────┤
│                         FastAPI Backend                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   Supabase   │  │    Neo4j     │  │      Crawl4AI          │ │
│  │  + pgvector  │  │   (Graph)    │  │   (Web Crawler)        │ │
│  │              │  │              │  │                        │ │
│  │ - Embeddings │  │ - Links      │  │ - HTML + Markdown      │ │
│  │ - Contenido  │  │ - PageRank   │  │ - JavaScript render    │ │
│  │ - Chunks     │  │ - HITS       │  │ - State persistence    │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Clonar e instalar dependencias

```bash
git clone https://github.com/dvillarrubia/WebKnoGraph.git
cd WebKnoGraph
pip install -r requirements-rag.txt
playwright install chromium
```

### 2. Configurar variables de entorno

```bash
cp graph_rag/.env.example graph_rag/.env
# Editar .env con tus credenciales de Supabase, Neo4j y OpenAI
```

### 3. Levantar servicios con Docker

```bash
docker-compose -f docker-compose.rag.yml up -d
```

### 4. Iniciar la API y Dashboard

```bash
PYTHONPATH=. python -m uvicorn graph_rag.api.main:app --host 0.0.0.0 --port 8080
```

Accede al dashboard en: **http://localhost:8080**

## Flujo de Trabajo

### 1. Crawling
```
URL → Crawler → pages/*.parquet (HTML + Markdown + Links)
```

### 2. Limpieza Manual
```
pages/*.parquet → Template Detection → Pattern Matching → manual_clean/*.parquet
```

### 3. Ingesta
```
manual_clean/*.parquet → Embeddings → Supabase (pgvector) + Neo4j (grafo)
```

### 4. Consulta RAG
```
Query → Vector Search → Graph Expansion → LLM → Respuesta con fuentes
```

## Módulos

### Crawler (`scripts/crawl4ai_advanced.py`)
- Extracción asíncrona con Crawl4AI
- Guarda HTML original para limpieza posterior
- Extrae enlaces internos para grafo
- Estado persistente para resumir crawls

### Manual Cleaner (`graph_rag/services/manual_cleaner_service.py`)
- Agrupa páginas por fingerprint HTML (misma plantilla)
- Patrones: `contains`, `prefix`, `line_range`, `exact`, `regex`
- Preview antes de aplicar cambios
- Guarda markdown limpio separado del original

### Ingest Service (`graph_rag/services/ingest_service.py`)
- Modos: `new_only`, `overwrite`, `full_refresh`
- Genera embeddings con `multilingual-e5-large` (1024 dims)
- Chunking semántico para búsqueda granular
- Calcula PageRank y HITS en Neo4j

### RAG Service (`graph_rag/services/rag_service.py`)
- Búsqueda vectorial en chunks (pgvector)
- Expansión por grafo N-hops (Neo4j)
- Reranking por PageRank + similitud
- Generación con GPT-4o

## Estructura del Proyecto

```
WebKnoGraph/
├── graph_rag/                   # Servicio RAG principal
│   ├── api/
│   │   ├── main.py              # FastAPI app
│   │   └── routes.py            # Endpoints (68+ rutas)
│   ├── services/
│   │   ├── crawler_service.py   # Integración Crawl4AI
│   │   ├── manual_cleaner_service.py
│   │   ├── ingest_service.py
│   │   ├── rag_service.py
│   │   └── embedding_service.py
│   ├── db/
│   │   ├── supabase_client.py
│   │   ├── neo4j_client.py
│   │   └── migrations/
│   ├── static/
│   │   └── index.html           # Dashboard SPA
│   └── config/
│       └── settings.py
├── src/                         # Módulos originales (legacy)
│   └── backend/
│       ├── services/            # Crawler, embeddings, PageRank, GraphSAGE
│       ├── graph/               # Algoritmos de grafo
│       └── models/              # Modelos ML
├── scripts/
│   └── crawl4ai_advanced.py     # Crawler standalone
├── tests/                       # Suite de tests
│   └── backend/services/        # Tests unitarios
├── notebooks/                   # Jupyter/Gradio UIs
│   ├── crawler_ui.ipynb
│   ├── embeddings_ui.ipynb
│   ├── link_prediction_ui.ipynb
│   └── pagerank_ui.ipynb
├── data/
│   └── crawl4ai_data/           # Datos de crawls (gitignored)
├── docker-compose.rag.yml
├── requirements-rag.txt
└── README.md
```

## API Endpoints

### Dashboard
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/v1/health` | Health check |

### Crawler
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/v1/dashboard/crawler/start` | Iniciar crawl |
| GET | `/api/v1/dashboard/crawler/status` | Estado del crawl |
| POST | `/api/v1/dashboard/crawler/stop` | Detener crawl |

### Manual Cleaner
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/v1/dashboard/manual-cleaner/analyze/{crawl}` | Detectar plantillas |
| GET | `/api/v1/dashboard/manual-cleaner/template/{crawl}/{template}` | Ver muestra |
| POST | `/api/v1/dashboard/manual-cleaner/pattern/{crawl}/{template}` | Añadir patrón |
| POST | `/api/v1/dashboard/manual-cleaner/apply/{crawl}/{template}` | Aplicar limpieza |

### Ingesta
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/v1/dashboard/ingest` | Ingestar crawl |
| GET | `/api/v1/dashboard/clients` | Listar clientes |

### RAG
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/v1/query` | Consulta RAG completa |
| POST | `/api/v1/search` | Solo búsqueda vectorial |

## Ejemplo: Consulta RAG

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "X-API-Key: tu-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cuáles son los requisitos de admisión?",
    "top_k_vectors": 10,
    "graph_hops": 2
  }'
```

## Tecnologías

- **Backend**: FastAPI, Python 3.10+
- **Crawler**: Crawl4AI, Playwright
- **Vector DB**: Supabase + pgvector
- **Graph DB**: Neo4j 5.x
- **Embeddings**: intfloat/multilingual-e5-large (1024 dims)
- **LLM**: OpenAI GPT-4o
- **Frontend**: Bootstrap 5, Vanilla JS

## Documentación Adicional

| Documento | Descripción |
|-----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Guía para asistentes IA trabajando con el código |
| [HOW-IT-WORKS.md](HOW-IT-WORKS.md) | Arquitectura detallada y flujo de datos |
| [graph_rag/README.md](graph_rag/README.md) | Documentación específica de Graph-RAG |
| [graph_rag/TODO_FEATURES.md](graph_rag/TODO_FEATURES.md) | Estado de implementación de features |
| `/docs` | Swagger API docs (cuando el servidor está corriendo) |

## Tests

```bash
# Ejecutar todos los tests
python -m pytest tests/ -v

# Test específico
python -m pytest tests/backend/services/test_crawler_service.py -v
```

## Licencia

Apache License 2.0
