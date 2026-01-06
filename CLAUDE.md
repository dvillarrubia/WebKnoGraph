# CLAUDE.md - Reglas del Proyecto WebKnoGraph

## Comandos Útiles

```bash
# Levantar servicios
PYTHONPATH=. python -m uvicorn graph_rag.api.routes:app --host 0.0.0.0 --port 8080 --reload

# Dashboard
http://localhost:8080/

# Test crawl rápido
curl -X POST http://localhost:8080/api/v1/dashboard/crawler/start \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com","max_pages":10}'
```

---

## Arquitectura del Sistema

```
CRAWLER → CLEANER → INGEST → RAG API
(Crawl4AI)  (3 opts)  (Embed)   (Search)

Parquet → Parquet → Supabase + Neo4j → Vector + Graph Search
```

**Archivos principales:**
- `scripts/crawl4ai_advanced.py` - Crawler script
- `graph_rag/services/crawler_service.py` - Gestión de jobs
- `graph_rag/services/manual_cleaner_service.py` - Limpieza manual
- `graph_rag/services/ingest_service.py` - Ingesta a DBs
- `graph_rag/api/routes.py` - Todos los endpoints
- `graph_rag/static/index.html` - Dashboard UI

---

## REGLAS CRÍTICAS - NO ROMPER

### 1. Crawler - Configuración de Crawl4AI

```python
# EN: scripts/crawl4ai_advanced.py

# CORRECTO - No se cuelga
crawler_cfg = CrawlerRunConfig(
    wait_until="load",  # NO usar "networkidle" - se cuelga con websockets/chat
    page_timeout=60000,
    delay_before_return_html=2.0,  # Espera para JS
)

# OBLIGATORIO - Timeout explícito
result = await asyncio.wait_for(
    crawler.arun(url=url, config=crawler_cfg),
    timeout=90.0  # Safety net absoluto
)
```

**Por qué:** `networkidle` espera a que no haya tráfico de red, pero páginas con chat widgets, analytics o websockets nunca llegan a "idle" y el crawler se cuelga indefinidamente.

### 2. Crawler - Usar fit_markdown, NO raw_markdown

```python
# EN: scripts/crawl4ai_advanced.py

md_result = result.markdown
if hasattr(md_result, 'raw_markdown'):
    markdown_raw = md_result.raw_markdown or ""
    markdown_fit = md_result.fit_markdown or ""
    # USAR fit_markdown como contenido principal
    markdown = markdown_fit if markdown_fit else markdown_raw
```

**Por qué:** `raw_markdown` contiene menús, footers, promos. `fit_markdown` está filtrado por `PruningContentFilter` y tiene solo el contenido relevante.

### 3. Crawler - PruningContentFilter menos agresivo

```python
# EN: scripts/crawl4ai_advanced.py

pruning_filter = PruningContentFilter(
    threshold=0.25,  # NO subir a 0.4+ (elimina contenido válido)
    threshold_type="dynamic",
    min_word_threshold=15,  # NO subir mucho
)
```

**Por qué:** Con threshold alto (0.4+) el filtro elimina demasiado contenido y las páginas quedan con 0-1 palabras.

### 4. Crawler - NO shadowing de variables

```python
# INCORRECTO - content_filter se sobreescribe
def crawl(..., content_filter: bool = True):
    content_filter = PruningContentFilter(...)  # Shadowing!
    if content_filter and word_count < 50:  # Ahora es el objeto, no bool

# CORRECTO
def crawl(..., content_filter: bool = True):
    pruning_filter = PruningContentFilter(...)  # Nombre diferente
    if content_filter and word_count < 50:  # Sigue siendo bool
```

### 5. Links - Preservar location y weight para PageRank

```python
# EN: scripts/crawl4ai_advanced.py - extract_links_with_location()

# Los links DEBEN tener:
{
    "href": url,
    "text": anchor_text,
    "location": "content" | "nav" | "footer" | "sidebar",
    "weight": 1.0 | 0.5 | 0.3 | 0.4  # Para PageRank
}
```

**Por qué:** Los weights se usan en Neo4j para calcular PageRank ponderado. Links en content valen más que en footer.

### 6. Parquet Schema - Campos obligatorios

```python
# Pages parquet DEBE tener:
{
    "url": str,
    "title": str,
    "markdown": str,      # Contenido limpio (fit_markdown)
    "markdown_raw": str,  # Original sin filtrar
    "html_content": str,  # Para CSS selectors en cleaner
    "word_count": int,
    "links_count": int,
}

# Links parquet DEBE tener:
{
    "source_url": str,
    "target_url": str,
    "anchor_text": str,
    "link_location": str,  # nav/footer/content/sidebar
    "link_weight": float,  # 0.3-1.0
}
```

**Por qué:** El ingest service espera estos campos. El cleaner necesita html_content para CSS selectors.

### 7. Cleaner - Prioridad de fuentes en Ingest

```python
# EN: graph_rag/services/ingest_service.py

# Orden de prioridad para leer markdown:
# 1. manual_clean/*.parquet (si existe)
# 2. markdown_clean/*.parquet (si existe)
# 3. pages_clean/*.parquet (si existe)
# 4. pages/*.parquet (original)
```

**Por qué:** El contenido más limpio tiene prioridad. El usuario puede refinar con manual cleaner.

### 8. Embeddings - Modelo y dimensiones

```python
# Default model: hiiamsid/sentence_similarity_spanish_es
# Dimensión: 768

# NO cambiar dimensión sin migrar datos
# Supabase pgvector está configurado para 768 dims
```

### 9. API - Log file location

```python
# EN: graph_rag/api/routes.py - dashboard_crawler_logs()

# El log está en el PARENT del output_dir
log_file = Path(job.output_dir).parent / ".crawl_log.jsonl"
# NO en job.output_dir directamente
```

---

## Estructura de Directorios

```
data/crawl4ai_data/
├── .crawl_status.json      # Estado del crawl actual
├── .crawl_log.jsonl        # Logs detallados
└── {domain}/               # ej: www_ilerna_es
    ├── pages/
    │   └── crawl_date=YYYY-MM-DD/*.parquet
    ├── links/
    │   └── crawl_date=YYYY-MM-DD/*.parquet
    ├── manual_clean/       # Output del manual cleaner
    ├── markdown_clean/     # Output del markdown cleaner
    ├── pages_clean/        # Output del simple cleaner
    └── manual_patterns.json
```

---

## Errores Conocidos y Soluciones

| Síntoma | Causa | Solución |
|---------|-------|----------|
| Crawl se cuelga sin avanzar | `wait_until="networkidle"` | Cambiar a `"load"` |
| Páginas con 0-1 palabras | Usando `raw_markdown` | Usar `fit_markdown` |
| Resume falla con datos corruptos | Status file stale | Borrar directorio y empezar fresh |
| Exclusiones CSS no funcionan | Se aplican a raw, no fit | Revisar PruningContentFilter |
| Logs no aparecen en UI | Path incorrecto | Usar `.parent / ".crawl_log.jsonl"` |

---

## Testing

```bash
# Test crawl de 5 páginas
curl -X POST http://localhost:8080/api/v1/dashboard/crawler/start \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.ilerna.es","max_pages":5,"delay":0.5}'

# Verificar contenido limpio
python3 -c "
import pandas as pd
df = pd.read_parquet('data/crawl4ai_data/www_ilerna_es/pages/crawl_date=2025-01-01/pages_*.parquet')
print(df[['url', 'word_count']].head())
print(df.iloc[0]['markdown'][:500])
"

# Verificar que NO tiene basura
python3 -c "
import pandas as pd
df = pd.read_parquet('data/crawl4ai_data/www_ilerna_es/pages/crawl_date=2025-01-01/*.parquet')
problems = ['SORTEAMOS', 'Llámanos gratis', 'FP Online', 'Centros de FP']
for p in problems:
    if any(p in str(m) for m in df['markdown']):
        print(f'ERROR: {p} encontrado en markdown')
    else:
        print(f'OK: {p} no está')
"
```

---

## Exclusiones CSS para Ilerna

```
#header-mobile-tablet, #header-desktop, #menu-mobile-tablet,
#Cookiebot, #bloque-promo-cintillo, .menu, .main-menu,
.footer, .subfooter, .form, .chatwith, .modal__close,
.breadcrumbs, .toc, .cta-mobile, .solicitar-info-float-btn,
header, footer, nav, aside, form, script, style, noscript, iframe
```

Archivo completo: `/Users/dvillarrubia/webknoGraph/exclusiones_ilerna.md`
