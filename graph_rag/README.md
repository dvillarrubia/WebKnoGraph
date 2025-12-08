# WebKnoGraph Graph-RAG

Sistema RAG multi-tenant que combina busqueda vectorial, expansion por grafo y generacion con LLM.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Supabase    │  │    Neo4j     │  │     OpenAI       │  │
│  │  + pgvector  │  │   (Graph)    │  │    (GPT-4o)      │  │
│  │              │  │              │  │                  │  │
│  │ - Embeddings │  │ - Links      │  │ - Respuestas     │  │
│  │ - Contenido  │  │ - PageRank   │  │ - Contexto       │  │
│  │ - Clientes   │  │ - Expansion  │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Modelo de Embeddings

Usa `intfloat/multilingual-e5-large` optimizado para espanol:
- Dimension: 1024
- Multilingue con excelente rendimiento en castellano
- Requiere prefijos: `query:` para consultas, `passage:` para documentos

## Setup Rapido

### 1. Configurar variables de entorno

```bash
cp graph_rag/.env.example graph_rag/.env
# Editar .env con tus credenciales
```

### 2. Crear schema en bases de datos

```bash
# Instalar dependencias
pip install -r requirements-rag.txt

# Ejecutar migraciones
python -m graph_rag.scripts.setup_db
```

### 3. Migrar datos existentes

```bash
python -m graph_rag.scripts.migrate_data \
  --name "Mi Cliente" \
  --domain "micliente.com" \
  --data-path ./data
```

Esto:
- Crea un nuevo cliente con API key
- Migra paginas con contenido
- Regenera embeddings con multilingual-e5-large
- Migra enlaces al grafo Neo4j

### 4. Iniciar el servicio

```bash
# Desarrollo
python -m graph_rag.api.main

# Docker
docker-compose -f docker-compose.rag.yml up graph-rag-api
```

## API Endpoints

### Autenticacion

Todos los endpoints de cliente requieren header `X-API-Key`.

### Endpoints principales

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/admin/clients` | Crear cliente |
| POST | `/api/v1/admin/clients/{id}/migrate` | Migrar datos |
| POST | `/api/v1/query` | Consulta RAG |
| POST | `/api/v1/search` | Busqueda vectorial |
| POST | `/api/v1/related` | Paginas relacionadas |

### Ejemplo: Consulta RAG

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "X-API-Key: tu-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Como funciona el proceso de compra?",
    "top_k_vectors": 10,
    "graph_hops": 2
  }'
```

Respuesta:
```json
{
  "answer": "El proceso de compra funciona...",
  "sources": [
    {
      "url": "https://ejemplo.com/compras",
      "title": "Proceso de compra",
      "pagerank": 0.0234,
      "similarity": 0.89,
      "from_graph": false
    }
  ],
  "context_stats": {
    "vector_results": 10,
    "graph_expanded": 8,
    "total_pages": 15,
    "context_tokens": 4500
  },
  "tokens_used": 1234
}
```

## Flujo RAG

```
Query del usuario
       │
       ▼
┌─────────────────┐
│ 1. Embedding    │ ← multilingual-e5-large
│    "query: ..." │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. pgvector     │ ← Busqueda por similitud coseno
│    top-K        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Neo4j        │ ← Expansion N-hops via enlaces
│    expansion    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. OpenAI       │ ← GPT-4o con contexto
│    GPT-4o       │
└─────────────────┘
```

## Multi-tenancy

Cada cliente tiene:
- ID unico (UUID)
- API key para autenticacion
- Datos aislados (filtro por client_id en todas las queries)

```sql
-- Todas las tablas filtran por client_id
SELECT * FROM rag_pages WHERE client_id = $1
```

```cypher
// Neo4j tambien filtra por client_id
MATCH (p:Page {client_id: $client_id})
```

## Estructura de carpetas

```
graph_rag/
├── api/
│   ├── main.py          # FastAPI app
│   ├── routes.py        # Endpoints
│   ├── models.py        # Pydantic schemas
│   └── dependencies.py  # DI y auth
├── config/
│   └── settings.py      # Configuracion
├── db/
│   ├── migrations/      # SQL para Supabase
│   ├── neo4j/           # Cypher constraints
│   ├── supabase_client.py
│   └── neo4j_client.py
├── services/
│   ├── embedding_service.py
│   ├── rag_service.py
│   └── migration_service.py
├── scripts/
│   ├── setup_db.py
│   └── migrate_data.py
├── Dockerfile
└── .env.example
```

## Notas

- El modelo de embeddings se descarga la primera vez (~2GB)
- La migracion regenera todos los embeddings con el nuevo modelo
- Neo4j almacena solo estructura de grafo, no embeddings
- pgvector usa indice HNSW para busqueda eficiente
