# WebKnoGraph - Sistema de Crawler y Limpieza de Contenido

Este documento describe el proceso completo de crawling y limpieza de contenido web implementado en WebKnoGraph. El sistema está diseñado para extraer contenido de sitios web, procesarlo y prepararlo para sistemas RAG (Retrieval-Augmented Generation).

---

## Tabla de Contenidos

1. [Arquitectura General](#arquitectura-general)
2. [Sistema de Crawling](#sistema-de-crawling)
3. [Sistema de Limpieza](#sistema-de-limpieza)
4. [API Endpoints](#api-endpoints)
5. [Flujo de Trabajo Recomendado](#flujo-de-trabajo-recomendado)
6. [Configuración para Nuevos Proyectos](#configuración-para-nuevos-proyectos)
7. [Resolución de Problemas](#resolución-de-problemas)

---

## Arquitectura General

```
┌─────────────────────────────────────────────────────────────────┐
│                      PIPELINE COMPLETO                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   CRAWLER ──────► CLEANER ──────► INGEST ──────► RAG API       │
│   (Crawl4AI)      (Manual/Auto)   (Embeddings)   (Search)      │
│                                                                 │
│   Parquet         Parquet         Supabase       Vector Search │
│   (pages/links)   (clean)         + Neo4j        + Graph       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Estructura de Directorios

```
data/crawl4ai_data/
├── www_ejemplo_com/              # Dominio crawleado (. → _)
│   ├── pages/                    # Páginas originales
│   │   └── crawl_date=2024-01-15/
│   │       └── pages_1705312800.parquet
│   ├── links/                    # Enlaces extraídos
│   │   └── crawl_date=2024-01-15/
│   │       └── links_1705312800.parquet
│   ├── auto_clean/               # Limpieza automática
│   │   └── all_pages.parquet
│   ├── manual_clean/             # Limpieza manual por template
│   │   ├── tpl_blog.parquet
│   │   └── tpl_root.parquet
│   └── manual_patterns.json      # Patrones de limpieza guardados
├── .crawl_status.json            # Estado del crawl actual
└── .crawl_log.jsonl              # Logs detallados
```

---

## Sistema de Crawling

### Archivos Involucrados

| Archivo | Función |
|---------|---------|
| `scripts/crawl4ai_advanced.py` | Script principal del crawler |
| `graph_rag/services/crawler_service.py` | Servicio que gestiona jobs desde la API |
| `graph_rag/api/routes.py` | Endpoints REST |

### Dependencias

```bash
pip install crawl4ai beautifulsoup4 aiohttp pandas pyarrow tqdm
```

### Características del Crawler

1. **Descubrimiento de URLs**
   - Sitemap XML (automático)
   - Robots.txt (respetado por defecto)
   - Enlaces internos durante el crawl

2. **Extracción de Contenido**
   - HTML completo (para CSS selectors en limpieza)
   - Markdown (usando Crawl4AI PruningContentFilter)
   - Metadata (title, meta description)

3. **Extracción de Enlaces con Contexto**
   - Localización: `nav`, `footer`, `sidebar`, `content`
   - Peso para PageRank: content=1.0, nav=0.5, sidebar=0.4, footer=0.3

4. **Robustez**
   - Watchdog para detectar cuelgues (120s timeout)
   - Auto-restart del browser tras crashes
   - Guardado incremental cada 10 páginas
   - Signal handlers para SIGTERM/SIGINT

### Parámetros del Crawler

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `url` | string | REQUERIDO | URL base para crawlear |
| `max_pages` | int | 0 | Máximo de páginas (0=ilimitado) |
| `delay` | float | 0.5 | Delay entre requests (segundos) |
| `use_sitemap` | bool | true | Usar sitemap para descubrir URLs |
| `content_filter` | bool | true | Filtrar páginas con <50 palabras |
| `resume` | bool | false | Continuar crawl anterior |
| `force_sitemap` | bool | false | Re-obtener sitemap en resume |
| `exclude_selectors` | array | [] | CSS selectors a excluir (cookies, modales) |
| `respect_robots` | bool | true | Respetar robots.txt |
| `skip_noindex` | bool | true | Saltar páginas con noindex |
| `sitemap_only` | bool | false | Solo URLs del sitemap (no seguir enlaces) |
| `urls_list` | array | [] | Lista específica de URLs a crawlear |
| `urls_only` | bool | false | Solo crawlear URLs de la lista |

### Uso desde Línea de Comandos

```bash
# Crawl básico
python scripts/crawl4ai_advanced.py \
  --url https://ejemplo.com \
  --max-pages 100 \
  --delay 0.5

# Solo sitemap (no sigue enlaces)
python scripts/crawl4ai_advanced.py \
  --url https://ejemplo.com \
  --sitemap-only

# Continuar crawl anterior
python scripts/crawl4ai_advanced.py \
  --url https://ejemplo.com \
  --resume \
  --force-sitemap

# Con exclusiones de cookies
python scripts/crawl4ai_advanced.py \
  --url https://ejemplo.com \
  --exclude-selectors "#cookie-banner,.popup-modal"

# Ver selectores de cookies por defecto
python scripts/crawl4ai_advanced.py --list-selectors
```

### Schema Parquet - Pages

```python
{
    "url": str,                    # URL completa
    "title": str,                  # <title> de la página
    "meta_description": str,       # <meta name="description">
    "markdown": str,               # Contenido markdown (raw)
    "html_content": str,           # HTML completo (para CSS selectors)
    "content_hash": str,           # SHA256 del markdown
    "word_count": int,             # Número de palabras
    "links_count": int,            # Número de enlaces extraídos
    "crawl_date": str,             # Fecha YYYY-MM-DD
}
```

### Schema Parquet - Links

```python
{
    "source_url": str,             # Página origen
    "target_url": str,             # Página destino
    "anchor_text": str,            # Texto del enlace
    "link_location": str,          # 'nav', 'footer', 'sidebar', 'content'
    "link_weight": float,          # Peso para PageRank
    "crawl_date": str,             # Fecha YYYY-MM-DD
}
```

### Selectores de Cookies por Defecto

El crawler incluye selectores predefinidos para banners de cookies comunes:

```python
DEFAULT_COOKIE_SELECTORS = [
    # Cookiebot
    "#CybotCookiebotDialog",
    "[id*='CookiebotDialog']",
    # OneTrust
    "#onetrust-consent-sdk",
    "#onetrust-banner-sdk",
    # Didomi
    "#didomi-host",
    # CookieYes
    "#cookie-law-info-bar",
    # Genéricos
    "[id*='cookie-consent']",
    "[id*='cookie-notice']",
    "[class*='cookie-banner']",
    "[id*='gdpr']",
    "[class*='gdpr']",
]
```

---

## Sistema de Limpieza

El sistema ofrece dos modos de limpieza que se pueden combinar:

### 1. Auto-Clean (Limpieza Automática)

Limpieza basada en heurísticas y etiquetas HTML semánticas.

#### Opciones de Auto-Clean

| Opción | Default | Descripción |
|--------|---------|-------------|
| `extract_from_first_heading` | true | Extrae contenido desde el primer H1/H2 |
| `remove_footer_content` | true | Elimina contenido de footer |
| `remove_nav_patterns` | true | Elimina patrones de navegación |
| `remove_empty_lines` | true | Elimina líneas vacías excesivas |
| `use_semantic_tags` | true | Usa `<footer>`, `<nav>`, `<aside>` para limpiar |
| `min_heading_level` | 1 | Nivel mínimo de heading (1=H1, 2=H2) |

#### Patrones de Footer Detectados

```python
FOOTER_PATTERNS_HEADINGS = [
    "## navegación", "## navigation", "## footer",
    "## compartir", "## share", "## síguenos",
    "## artículos relacionados", "## contacto",
]

FOOTER_PATTERNS_LEGAL = [
    "© ", "copyright ", "todos los derechos reservados",
]
```

### 2. Manual Cleaner (Limpieza Manual por Templates)

Sistema de limpieza controlada basado en templates (agrupación de páginas similares).

#### Flujo de Trabajo

```
1. analyze_templates()
   │ Agrupa páginas por: URL path pattern + HTML fingerprint
   ▼
2. Templates detectados: /blog/* (450 págs), /productos/* (200 págs)

3. Usuario selecciona template → get_template_sample()
   │ Muestra página de ejemplo
   ▼
4. Usuario añade patrones → add_pattern()
   │ Define qué eliminar
   ▼
5. preview_cleaning() → Muestra antes/después

6. apply_cleaning() → Guarda en manual_clean/*.parquet
```

#### Tipos de Patrones

| Tipo | Ejemplo | Descripción |
|------|---------|-------------|
| `exact` | `"Llámanos: 900 730 222"` | Elimina texto exacto donde aparezca |
| `prefix` | `"Inicio >"` | Elimina líneas que empiezan con este texto |
| `contains` | `"Síguenos en"` | Elimina líneas que contienen este texto |
| `regex` | `^\[.*\]\(https://twitter` | Patrón regex (MULTILINE) |
| `line_range` | `"Compartir\|\|\|Artículos relacionados"` | Elimina desde texto_inicio hasta texto_fin |
| `text_range` | `"Banner\|\|\|/Banner"` | Como line_range pero elimina TODAS las ocurrencias |
| `css_selector` | `.sidebar, #footer, nav` | Elimina elementos HTML, regenera markdown |

#### Ejemplo de Patrones

```json
{
  "templates": {
    "tpl_blog": {
      "patterns": [
        {
          "id": "pat_143021_0",
          "pattern_type": "line_range",
          "value": "Compartir|||Artículos relacionados",
          "description": "Eliminar bloque de compartir y relacionados"
        },
        {
          "id": "pat_143045_1",
          "pattern_type": "css_selector",
          "value": ".breadcrumb, .share-buttons, .author-bio",
          "description": "Eliminar breadcrumbs, botones compartir y bio autor"
        },
        {
          "id": "pat_143102_2",
          "pattern_type": "contains",
          "value": "Síguenos en redes",
          "description": "Eliminar CTAs de redes sociales"
        }
      ],
      "is_cleaned": true
    }
  }
}
```

### Archivos del Sistema de Limpieza

| Archivo | Función |
|---------|---------|
| `graph_rag/services/manual_cleaner_service.py` | Servicio principal de limpieza |
| `{crawl}/manual_patterns.json` | Patrones guardados por crawl |
| `{crawl}/auto_clean/all_pages.parquet` | Resultado de auto-clean |
| `{crawl}/manual_clean/{template}.parquet` | Resultado de limpieza manual |

### Prioridad de Fuentes para Ingesta

Cuando se ingesta contenido, se usa esta prioridad:

```
1. manual_clean/{template}.parquet   ← Mejor calidad
2. auto_clean/all_pages.parquet
3. pages/*.parquet                   ← Contenido original
```

---

## API Endpoints

### Crawler

```http
POST /api/v1/dashboard/crawler/start
{
  "url": "https://ejemplo.com",
  "max_pages": 100,
  "delay": 0.5,
  "use_sitemap": true,
  "content_filter": true,
  "resume": false,
  "exclude_selectors": ["#cookie-banner", ".modal"],
  "respect_robots": true,
  "skip_noindex": true,
  "sitemap_only": false
}

POST /api/v1/dashboard/crawler/stop

GET /api/v1/dashboard/crawler/status

GET /api/v1/dashboard/crawler/logs?last_n=100

GET /api/v1/dashboard/crawler/crawls

GET /api/v1/dashboard/crawler/crawls/{crawl_name}/pages?limit=100&offset=0&search=&min_words=50
```

### Manual Cleaner

```http
# Analizar templates de un crawl
GET /api/v1/dashboard/manual-cleaner/analyze/{crawl_name}

# Obtener muestra de un template
GET /api/v1/dashboard/manual-cleaner/template/{crawl_name}/{template_id}?sample_index=0

# Añadir patrón a un template
POST /api/v1/dashboard/manual-cleaner/pattern/{crawl_name}/{template_id}
{
  "pattern_type": "line_range",
  "value": "Inicio texto|||Fin texto",
  "description": "Descripción del patrón"
}

# Eliminar patrón
DELETE /api/v1/dashboard/manual-cleaner/pattern/{crawl_name}/{template_id}/{pattern_id}

# Preview de limpieza
GET /api/v1/dashboard/manual-cleaner/preview/{crawl_name}/{template_id}

# Aplicar limpieza
POST /api/v1/dashboard/manual-cleaner/apply/{crawl_name}/{template_id}
{
  "auto_clean_options": {
    "extract_from_first_heading": true,
    "remove_footer_content": true,
    "remove_nav_patterns": true,
    "remove_empty_lines": true
  }
}

# Obtener patrones de otros templates (para reutilizar)
GET /api/v1/dashboard/manual-cleaner/available-patterns/{crawl_name}/{template_id}
```

### Auto-Clean

```http
# Preview de auto-clean
GET /api/v1/dashboard/manual-cleaner/auto-clean/preview/{crawl_name}?sample_url=...

# Aplicar auto-clean a todo el crawl
POST /api/v1/dashboard/manual-cleaner/auto-clean/{crawl_name}
{
  "extract_from_first_heading": true,
  "remove_footer_content": true,
  "remove_nav_patterns": true,
  "remove_empty_lines": true
}

# Estado de limpieza
GET /api/v1/dashboard/manual-cleaner/status/{crawl_name}
```

---

## Flujo de Trabajo Recomendado

### Paso 1: Crawlear el Sitio

```bash
# Desde línea de comandos
python scripts/crawl4ai_advanced.py \
  --url https://mi-sitio.com \
  --max-pages 500 \
  --delay 1.0

# O desde la API/Dashboard
POST /api/v1/dashboard/crawler/start
{
  "url": "https://mi-sitio.com",
  "max_pages": 500,
  "delay": 1.0
}
```

### Paso 2: Aplicar Auto-Clean (Opcional pero Recomendado)

```bash
# Desde la API
POST /api/v1/dashboard/manual-cleaner/auto-clean/www_mi_sitio_com
{
  "extract_from_first_heading": true,
  "remove_footer_content": true
}
```

### Paso 3: Analizar Templates

```bash
GET /api/v1/dashboard/manual-cleaner/analyze/www_mi_sitio_com

# Respuesta:
{
  "templates": [
    {"template_id": "tpl_blog", "page_count": 450, "path_pattern": "/blog"},
    {"template_id": "tpl_productos", "page_count": 200, "path_pattern": "/productos"},
    {"template_id": "tpl_root", "page_count": 50, "path_pattern": "/"}
  ]
}
```

### Paso 4: Limpiar Templates Individualmente

Para cada template con muchas páginas:

1. **Ver muestra**:
   ```
   GET /api/v1/dashboard/manual-cleaner/template/www_mi_sitio_com/tpl_blog
   ```

2. **Identificar contenido a eliminar** en el markdown mostrado

3. **Añadir patrones**:
   ```
   POST /api/v1/dashboard/manual-cleaner/pattern/www_mi_sitio_com/tpl_blog
   {
     "pattern_type": "line_range",
     "value": "## Artículos relacionados|||## Comentarios",
     "description": "Bloque de artículos relacionados"
   }
   ```

4. **Preview**:
   ```
   GET /api/v1/dashboard/manual-cleaner/preview/www_mi_sitio_com/tpl_blog
   ```

5. **Aplicar**:
   ```
   POST /api/v1/dashboard/manual-cleaner/apply/www_mi_sitio_com/tpl_blog
   ```

### Paso 5: Ingestar a Base de Datos

Una vez limpio, ingestar para RAG:

```bash
POST /api/v1/dashboard/ingest
{
  "crawl_path": "data/crawl4ai_data/www_mi_sitio_com",
  "client_name": "Mi Sitio",
  "client_domain": "www.mi-sitio.com"
}
```

---

## Configuración para Nuevos Proyectos

### Requisitos

```bash
# Python 3.10+
pip install crawl4ai beautifulsoup4 aiohttp pandas pyarrow tqdm markdownify html2text
```

### Estructura Mínima

```
mi_proyecto/
├── scripts/
│   └── crawl4ai_advanced.py          # Copiar desde WebKnoGraph
├── graph_rag/
│   └── services/
│       ├── crawler_service.py        # Copiar desde WebKnoGraph
│       └── manual_cleaner_service.py # Copiar desde WebKnoGraph
└── data/
    └── crawl4ai_data/                # Se crea automáticamente
```

### Personalización de Selectores por Dominio

En `crawl4ai_advanced.py`, puedes añadir presets personalizados:

```python
DOMAIN_PRESETS = {
    "mi-dominio": {
        "name": "Mi Dominio Custom",
        "description": "Selectores para mi-dominio.com",
        "selectors": [
            "#mi-header",
            ".mi-footer",
            ".banner-promo",
            "[class*='popup']",
        ]
    }
}
```

### Variables de Entorno (Opcional para Ingesta)

```bash
# Solo necesarias si usas ingesta a bases de datos
OPENAI_API_KEY=...          # Para cleaners con IA
SUPABASE_URL=...            # Base de datos vectorial
SUPABASE_KEY=...
NEO4J_URI=...               # Base de datos de grafos
NEO4J_USER=...
NEO4J_PASSWORD=...
```

---

## Resolución de Problemas

### Problema: El Crawl se Cuelga

**Causa**: Usar `wait_until="networkidle"` con páginas que tienen websockets.

**Solución**: El crawler ya usa `wait_until="load"` por defecto. Si modificaste esto, revertir a `"load"`.

```python
# En crawl4ai_advanced.py
crawler_cfg = CrawlerRunConfig(
    wait_until="load",  # NO usar "networkidle"
    page_timeout=60000,
)
```

### Problema: Contenido Vacío (0-1 palabras)

**Causa**: Usar `raw_markdown` en lugar de `fit_markdown`.

**Solución**: Verificar que se usa `fit_markdown`:

```python
md_result = result.markdown
markdown = md_result.fit_markdown or md_result.raw_markdown
```

### Problema: Exclusiones CSS No Funcionan

**Causa**: Variable shadowing o selectores mal formados.

**Solución**: Verificar selectores con `--list-selectors` y asegurar que no hay conflictos de nombres.

### Problema: Resume No Funciona

**Causa**: Archivos parquet corruptos o estructura de directorios incorrecta.

**Solución**: Verificar estructura o iniciar crawl limpio (sin `--resume`).

### Problema: Timeout en Páginas

**Causa**: Sin timeout explícito.

**Solución**: El crawler ya incluye `asyncio.wait_for(..., timeout=90)`. Si sigue fallando, incrementar timeout.

### Problema: Templates No Se Detectan Correctamente

**Causa**: URLs muy diversas o HTML muy variable.

**Solución**: Los templates se agrupan por:
1. Primer segmento de la URL (`/blog/*`, `/productos/*`)
2. Fingerprint del HTML (clases de body, estructura)

Si necesitas agrupación diferente, modifica `extract_url_path_pattern()` en `manual_cleaner_service.py`.

---

## Checklist Antes de Modificar

- [ ] ¿El cambio afecta al schema de parquet? → Verificar compatibilidad con ingest
- [ ] ¿El cambio afecta a exclusiones CSS? → Probar que fit_markdown sigue limpio
- [ ] ¿El cambio afecta a templates? → Verificar detección en manual cleaner
- [ ] ¿El cambio afecta a embeddings? → Verificar dimensiones (768 por defecto)
- [ ] ¿El cambio afecta a links? → Verificar weights para PageRank

---

## Ejemplo Completo: Crawlear y Limpiar ILERNA

```bash
# 1. Crawlear
python scripts/crawl4ai_advanced.py \
  --url https://www.ilerna.es \
  --max-pages 1000 \
  --delay 0.5 \
  --preset ilerna

# 2. Ver estado
curl http://localhost:8080/api/v1/dashboard/crawler/status

# 3. Auto-clean
curl -X POST http://localhost:8080/api/v1/dashboard/manual-cleaner/auto-clean/www_ilerna_es \
  -H "Content-Type: application/json" \
  -d '{"extract_from_first_heading": true, "remove_footer_content": true}'

# 4. Analizar templates
curl http://localhost:8080/api/v1/dashboard/manual-cleaner/analyze/www_ilerna_es

# 5. Para cada template importante, añadir patrones específicos
curl -X POST http://localhost:8080/api/v1/dashboard/manual-cleaner/pattern/www_ilerna_es/tpl_blog \
  -H "Content-Type: application/json" \
  -d '{
    "pattern_type": "line_range",
    "value": "## Comparte este artículo|||## Te puede interesar",
    "description": "Eliminar bloque compartir y relacionados"
  }'

# 6. Aplicar limpieza
curl -X POST http://localhost:8080/api/v1/dashboard/manual-cleaner/apply/www_ilerna_es/tpl_blog
```

---

## Conclusión

Este sistema proporciona un pipeline completo para:

1. **Crawlear** sitios web respetando robots.txt y extrayendo contenido estructurado
2. **Limpiar** el contenido usando heurísticas automáticas y/o patrones manuales
3. **Preparar** los datos para sistemas RAG con contenido limpio y metadatos ricos

La combinación de auto-clean (rápido, bueno para la mayoría del contenido) y manual cleaner (preciso, para templates específicos) permite obtener contenido de alta calidad para cualquier sitio web.
