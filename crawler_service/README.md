# Crawler Service

Servicio independiente de crawling web basado en Crawl4AI, diseñado para ejecutarse como un contenedor Docker separado de la aplicación principal Graph-RAG.

## Arquitectura

```
graph-rag-api (:8080)  ──HTTP──▶  crawler-service (:8081)
       │                                  │
       └──── volumen Docker compartido ───┘
              data/crawl4ai_data/
```

- **Control del crawler**: via HTTP (start, stop, status, logs)
- **Datos de salida**: Parquet en volumen Docker compartido
- **Compatibilidad**: si `CRAWLER_SERVICE_URL` no está definido, la app principal usa subprocess local (mismo comportamiento anterior)

## API Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/crawl/start` | POST | Iniciar crawl (mismos params que el dashboard) |
| `/crawl/stop` | POST | Detener crawl activo |
| `/crawl/status` | GET | Estado del crawl actual |
| `/crawl/logs?last_n=100` | GET | Últimas N entradas del log |
| `/crawl/crawls` | GET | Listar crawls disponibles |

## Parámetros de `/crawl/start`

```json
{
  "url": "https://example.com",
  "max_pages": 0,
  "delay": 0.5,
  "use_sitemap": true,
  "content_filter": true,
  "resume": false,
  "force_sitemap": false,
  "urls_list": [],
  "urls_only": false,
  "exclude_selectors": [],
  "respect_robots": true,
  "skip_noindex": true,
  "sitemap_only": false
}
```

## Docker

### Build individual

```bash
docker build -t crawler-service -f crawler_service/Dockerfile .
```

### Run individual

```bash
docker run -d \
  --name crawler-service \
  -p 8081:8081 \
  -v webknograph-crawl-data:/app/data/crawl4ai_data \
  crawler-service
```

### Con Docker Compose (recomendado)

```bash
docker-compose -f docker-compose.rag.yml up -d crawler-service
```

## Variables de entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `CRAWL_OUTPUT_DIR` | `/app/data/crawl4ai_data` | Directorio de salida para datos de crawl |

## Estructura de salida

```
/app/data/crawl4ai_data/
├── .crawl_status.json          # Estado del crawl actual
├── .crawl_log.jsonl            # Logs detallados (JSONL)
└── {dominio}/                  # ej: www_example_com
    ├── pages/
    │   └── crawl_date=YYYY-MM-DD/*.parquet
    └── links/
        └── crawl_date=YYYY-MM-DD/*.parquet
```

## Contenido del contenedor

- Python 3.11
- Crawl4AI + Playwright + Chromium
- FastAPI + Uvicorn
- pandas + pyarrow (para Parquet)
- BeautifulSoup + lxml (para HTML parsing)
