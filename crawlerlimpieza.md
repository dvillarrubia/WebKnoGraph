# WebKnoGraph - Sistema de Crawler y Limpieza de Contenido

Este documento describe el proceso completo de crawling y limpieza de contenido web implementado en WebKnoGraph. El sistema está diseñado para extraer contenido de sitios web, procesarlo y prepararlo para sistemas RAG (Retrieval-Augmented Generation).

---

## Tabla de Contenidos

1. [Arquitectura General](#arquitectura-general)
2. [Sistema de Crawling](#sistema-de-crawling)
3. [Sistema de Limpieza Integrada](#sistema-de-limpieza-integrada)
4. [Sistema de Limpieza Manual](#sistema-de-limpieza-manual)
5. [API Endpoints](#api-endpoints)
6. [Flujo de Trabajo Recomendado](#flujo-de-trabajo-recomendado)
7. [Configuración para Nuevos Proyectos](#configuración-para-nuevos-proyectos)
8. [Resolución de Problemas](#resolución-de-problemas)
9. [Historial de Cambios](#historial-de-cambios)

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

### Arquitectura del Crawler (Actualizada Enero 2026)

El crawler utiliza un sistema de **tres capas de limpieza** para obtener contenido limpio de forma universal:

```
┌──────────────────────────────────────────────────────────────────┐
│                    FLUJO DE EXTRACCIÓN                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. JavaScript Cleanup (antes de captura)                       │
│      └─► Elimina: cookies, nav, header, footer, overlays         │
│                                                                  │
│   2. css_selector (contenedores estándar)                        │
│      └─► Enfoca: main, article, .content, .post-content          │
│      └─► Fallback si < 200 palabras                              │
│                                                                  │
│   3. Markdown Cleanup (post-procesamiento)                       │
│      └─► Encuentra primer heading/párrafo                        │
│      └─► Elimina navegación al inicio                            │
│      └─► Elimina líneas de cookies en cualquier parte            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Características del Crawler

1. **Descubrimiento de URLs**
   - Sitemap XML (automático)
   - Robots.txt (respetado por defecto)
   - Enlaces internos durante el crawl

2. **Extracción de Contenido con Limpieza Integrada**
   - JavaScript elimina cookies/nav/footer ANTES de capturar HTML
   - css_selector enfoca en áreas de contenido (con fallback)
   - Markdown cleanup elimina navegación y cookies residuales
   - Metadata (title, meta description) - extrae de H1 si está vacío

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
| `exclude_selectors` | array | [] | CSS selectors adicionales a excluir |
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

# Con exclusiones adicionales
python scripts/crawl4ai_advanced.py \
  --url https://ejemplo.com \
  --exclude-selectors "#custom-popup,.my-modal"

# Ver selectores de cookies por defecto
python scripts/crawl4ai_advanced.py --list-selectors
```

### Schema Parquet - Pages

```python
{
    "url": str,                    # URL completa
    "title": str,                  # <title> o H1 si vacío
    "meta_description": str,       # <meta name="description">
    "markdown": str,               # Contenido markdown LIMPIO
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

---

## Sistema de Limpieza Integrada

### JavaScript Cleanup (Fase 1)

El JavaScript se ejecuta ANTES de capturar el HTML, eliminando elementos no deseados:

```javascript
// FASE 1: Cookies y banners de consentimiento
// Cookiebot, OneTrust, Didomi, genéricos
document.querySelectorAll('#CybotCookiebotDialog, [id*="Cookiebot"]').forEach(el => el.remove());
document.querySelectorAll('#onetrust-consent-sdk, [id*="onetrust"]').forEach(el => el.remove());
document.querySelectorAll('[id*="cookie-consent"], [id*="gdpr"], [id*="consent"]').forEach(el => el.remove());

// FASE 2: Navegación y layout
// Tags semánticos y roles ARIA
document.querySelectorAll('nav, header, footer, aside').forEach(el => el.remove());
document.querySelectorAll('[role="navigation"], [role="banner"], [role="contentinfo"]').forEach(el => el.remove());

// Clases comunes (case-insensitive)
document.querySelectorAll('*').forEach(el => {
    const cls = el.className.toLowerCase();
    if (cls.match(/\b(nav|navbar|menu|header|footer|sidebar|breadcrumb)\b/)) {
        el.remove();
    }
});

// FASE 3: Elementos fixed/sticky (overlays, modales)
document.querySelectorAll('*').forEach(el => {
    const style = window.getComputedStyle(el);
    if ((style.position === 'fixed' || style.position === 'sticky') && parseInt(style.zIndex) > 100) {
        el.remove();
    }
});

// FASE 4: Formularios y widgets
document.querySelectorAll('[id*="form"], [id*="chat"], [id*="widget"]').forEach(el => el.remove());
```

### css_selector con Fallback (Fase 2)

```python
# Configuración principal: con css_selector
crawler_cfg = CrawlerRunConfig(
    css_selector="main, article, .content, .post-content, .entry-content, #content, .main-content",
    # ... otras opciones
)

# Si el resultado tiene < 200 palabras, se usa fallback SIN css_selector
# Esto captura todo el body (ya limpiado por JavaScript)
crawler_cfg_fallback = CrawlerRunConfig(
    # SIN css_selector
    # ... mismas opciones de JS cleanup
)
```

**Por qué el fallback:**
- Algunos sitios no usan contenedores estándar (`main`, `article`, `.content`)
- El contenido puede estar en `div`s genéricos
- El fallback captura todo el body (ya limpiado por JS)

### Markdown Cleanup (Fase 3)

Función `clean_markdown()` que procesa el texto extraído:

```python
# Palabras clave que indican contenido de cookies
_COOKIE_KEYWORDS = [
    'cookie', 'consent', 'aceptar', 'rechazar', 'personalizar',
    'necesarias', 'funcionales', 'estadísticas', 'marketing', 'publicidad',
    'consentimiento', 'privacidad', 'almacenamiento', 'proveedor',
    'hubspot', 'google analytics', 'facebook pixel', 'gdpr',
]

# Indicadores inequívocos de cookies
_COOKIE_INDICATORS = [
    # Nombres de cookies comunes
    'test_cookie', '_ga', '_gid', '_fbp', '_gcl', '__cf_bm', '__cflb',
    '_hjid', '_hjsession', 'PHPSESSID', 'JSESSIONID',
    # Proveedores
    'cookiebot', 'onetrust', 'hubspot', 'hotjar', 'clarity',
    # Textos de consent
    'política de privacidad', 'esta cookie se utiliza',
    'distinguir entre humanos y bots', 'equilibrio de carga',
]
```

**Detección de líneas de cookies:**

```python
def _is_cookie_line(line: str) -> bool:
    # 1. Líneas que empiezan con **_variable** (ej: **_ga**, **__cf_bm**)
    if stripped.startswith('**_') or stripped.startswith('**['):
        return True

    # 2. Documentación de cookies: **VARIABLE_NAME** descripción...
    # Detecta: **PHPSESSID**, **personalization_id**, **PrestaShop-#**
    cookie_var_pattern = r'^\*\*[A-Za-z0-9_#.-]+(\s*\[.*?\])?\*\*\s+'
    if re.match(cookie_var_pattern, stripped):
        var_name = stripped.split('**')[1]
        if '_' in var_name or '-' in var_name or any(c.isdigit() for c in var_name):
            return True

    # 3. Indicadores inequívocos
    for indicator in _COOKIE_INDICATORS:
        if indicator in lower:
            return True

    # 4. Keywords (2+ coincidencias)
    matches = sum(1 for kw in _COOKIE_KEYWORDS if kw in lower)
    return matches >= 2
```

**Detección de contenido real:**

```python
def _is_content_line(line: str) -> bool:
    # Headings son contenido
    if stripped.startswith('#') and len(stripped) > 2:
        return True
    # Listas y links sueltos son navegación
    if stripped.startswith('*') or stripped.startswith('['):
        return False
    # Breadcrumbs
    if ' / ' in stripped and len(stripped) < 200:
        return False
    # Párrafos cortos pueden ser navegación
    if len(stripped) < 50:
        return False
    # Párrafos largos sin ser link son contenido
    return True
```

**Proceso de limpieza:**

```python
def clean_markdown(md_text: str) -> str:
    # FASE 1: Encontrar primer contenido real
    first_content_idx = 0
    for i, line in enumerate(lines):
        if _is_content_line(line):
            first_content_idx = i
            break

    # Descartar navegación al inicio
    lines = lines[first_content_idx:]

    # FASE 2: Eliminar líneas de cookies
    cleaned = [line for line in lines if not _is_cookie_line(line)]

    return '\n'.join(cleaned)
```

### Selectores de Cookies por Defecto

```python
DEFAULT_COOKIE_SELECTORS = [
    # Cookiebot
    "#CybotCookiebotDialog",
    "#CybotCookiebotDialogBody",
    "[id*='CookiebotDialog']",
    # OneTrust
    "#onetrust-consent-sdk",
    "#onetrust-banner-sdk",
    "[id*='onetrust']",
    # Didomi
    "#didomi-host",
    "[id*='didomi']",
    # CookieYes
    "#cookie-law-info-bar",
    "[id*='cookie-law']",
    # Complianz
    "#cmplz-cookiebanner-container",
    "[id*='cmplz']",
    # Genéricos
    "[id*='cookie-consent']",
    "[id*='cookie-notice']",
    "[id*='cookie-banner']",
    "[class*='cookie-consent']",
    "[class*='cookie-banner']",
    "[id*='gdpr']",
    "[class*='gdpr']",
]
```

---

## Sistema de Limpieza Manual

El sistema de limpieza manual permite refinar el contenido usando patrones específicos por template.

### Flujo de Trabajo

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

### Tipos de Patrones

| Tipo | Ejemplo | Descripción |
|------|---------|-------------|
| `exact` | `"Llámanos: 900 730 222"` | Elimina texto exacto donde aparezca |
| `prefix` | `"Inicio >"` | Elimina líneas que empiezan con este texto |
| `contains` | `"Síguenos en"` | Elimina líneas que contienen este texto |
| `regex` | `^\[.*\]\(https://twitter` | Patrón regex (MULTILINE) |
| `line_range` | `"Compartir\|\|\|Artículos relacionados"` | Elimina desde texto_inicio hasta texto_fin |
| `text_range` | `"Banner\|\|\|/Banner"` | Como line_range pero elimina TODAS las ocurrencias |
| `css_selector` | `.sidebar, #footer, nav` | Elimina elementos HTML, regenera markdown |

### Ejemplo de Patrones

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
        }
      ],
      "is_cleaned": true
    }
  }
}
```

### Prioridad de Fuentes para Ingesta

```
1. manual_clean/{template}.parquet   ← Mejor calidad
2. auto_clean/all_pages.parquet
3. pages/*.parquet                   ← Contenido ya limpio por crawler
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
  "exclude_selectors": ["#custom-popup", ".modal"],
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
```

---

## Flujo de Trabajo Recomendado

### Paso 1: Crawlear el Sitio

El crawler ahora produce contenido **ya limpio** gracias al sistema de 3 capas.

```bash
python scripts/crawl4ai_advanced.py \
  --url https://mi-sitio.com \
  --max-pages 500 \
  --delay 0.5
```

### Paso 2: Verificar Calidad del Contenido

```python
import pandas as pd
from glob import glob

files = glob('data/crawl4ai_data/www_mi_sitio_com/pages/**/*.parquet', recursive=True)
df = pd.read_parquet(files[-1])

for _, row in df.head(3).iterrows():
    print(f"URL: {row['url']}")
    print(f"Palabras: {row['word_count']}")
    print(f"Primeros 200 chars: {row['markdown'][:200]}")
    print()
```

### Paso 3: Limpieza Manual (Opcional)

Si hay contenido que necesita refinamiento adicional:

1. Analizar templates
2. Añadir patrones específicos
3. Aplicar limpieza

### Paso 4: Ingestar a Base de Datos

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
pip install crawl4ai beautifulsoup4 aiohttp pandas pyarrow tqdm
```

### Variables de Entorno (Opcional para Ingesta)

```bash
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

**Solución**: El crawler usa `wait_until="load"` por defecto. No cambiar.

### Problema: Contenido Vacío o Muy Corto

**Causa**: css_selector no coincide con la estructura HTML del sitio.

**Solución**: El sistema tiene fallback automático. Si una página tiene <200 palabras con css_selector, se re-crawlea sin él.

### Problema: Cookies Aparecen en el Contenido

**Causa**: Banner de cookies dinámico cargado después del JS cleanup.

**Solución**: La función `clean_markdown()` elimina cookies en post-procesamiento usando:
- Keywords (cookie, consent, aceptar, etc.)
- Indicadores (_ga, __cf_bm, cookiebot, etc.)
- Patrones de variables (**PHPSESSID**, etc.)

### Problema: Navegación al Inicio del Contenido

**Causa**: El primer heading/párrafo no fue detectado correctamente.

**Solución**: `clean_markdown()` busca el primer contenido real y descarta todo lo anterior. Ajustar `_is_content_line()` si es necesario.

### Problema: Headings o Negritas Desaparecen

**Causa**: Antes usábamos PruningContentFilter que eliminaba estos elementos.

**Solución**: Se eliminó PruningContentFilter. Ahora usamos `raw_markdown` con limpieza por JavaScript y post-procesamiento.

### Problema: Títulos Vacíos

**Causa**: css_selector solo extrae contenido, no `<head>` donde está `<title>`.

**Solución**: Si el título está vacío, se extrae del primer H1 del markdown:

```python
if not title and markdown_raw:
    h1_match = re.search(r'^#\s+(.+)$', markdown_raw, re.MULTILINE)
    if h1_match:
        title = h1_match.group(1).strip()
```

---

## Guía para Desarrolladores: Cómo Replicar Este Sistema

Esta sección documenta el proceso completo de desarrollo, los problemas encontrados y las soluciones implementadas, para que puedas replicar este sistema en otros proyectos.

### Contexto del Proyecto

**Objetivo**: Crear un sistema de web scraping robusto para alimentar sistemas RAG (Retrieval-Augmented Generation) con contenido limpio y estructurado de sitios web.

**Stack tecnológico elegido**:
- **Crawl4AI**: Librería de crawling basada en Playwright (navegador headless)
- **Parquet**: Formato de almacenamiento columnar para datos crawleados
- **FastAPI**: API REST para control del crawler
- **BeautifulSoup**: Parsing HTML para extracción de enlaces

### Evolución del Sistema: Problemas y Soluciones

#### Fase 1: Crawler Básico (Problemas Iniciales)

**Intento inicial**: Usar `crawl4ai` con configuración por defecto.

```python
# ❌ Configuración inicial problemática
result = await crawler.arun(url=url)
markdown = result.markdown  # Contenido con cookies, navegación, etc.
```

**Problemas encontrados**:
1. El contenido incluía banners de cookies
2. Navegación (menús, breadcrumbs) mezclada con contenido
3. Footers y sidebars aparecían en el markdown
4. Algunos crawls se colgaban indefinidamente

---

#### Fase 2: PruningContentFilter (Solución Fallida)

**Intento**: Usar el filtro de contenido de Crawl4AI.

```python
# ❌ Parecía buena idea pero eliminaba contenido importante
from crawl4ai import PruningContentFilter

markdown_generator = DefaultMarkdownGenerator(
    content_filter=PruningContentFilter(threshold=0.25)
)
result = await crawler.arun(url=url)
markdown = result.markdown.fit_markdown  # Contenido "filtrado"
```

**Problemas**:
- `PruningContentFilter` eliminaba **headings** (`## Título`)
- También eliminaba **negritas** (`**texto importante**`)
- El threshold era impredecible - a veces eliminaba párrafos enteros

**Lección aprendida**: Los filtros automáticos basados en heurísticas son peligrosos para contenido que necesita preservar formato.

---

#### Fase 3: JavaScript Cleanup Pre-captura (Breakthrough)

**Insight clave**: Eliminar elementos NO deseados ANTES de que Crawl4AI capture el HTML.

```python
# ✅ Ejecutar JavaScript antes de capturar
js_remove_overlays = '''
// Eliminar cookies
document.querySelectorAll('#CybotCookiebotDialog, [id*="cookie"]').forEach(el => el.remove());
// Eliminar navegación
document.querySelectorAll('nav, header, footer, aside').forEach(el => el.remove());
// Eliminar overlays fixed
document.querySelectorAll('*').forEach(el => {
    const style = window.getComputedStyle(el);
    if (style.position === 'fixed' && parseInt(style.zIndex) > 100) {
        el.remove();
    }
});
'''

crawler_cfg = CrawlerRunConfig(
    js_code=js_remove_overlays,
    wait_for="css:body",  # Esperar a que el body exista
)
```

**Ventajas**:
- Elimina elementos antes de generar markdown
- No afecta al contenido real
- Universal para cualquier sitio

**Problema nuevo**: El JS eliminaba `<div class="content-header">` porque matcheaba `header`.

**Solución**: Patrones más específicos:
```javascript
// ❌ Esto elimina content-header, post-header, article-header
if (cls.match(/\b(header)\b/)) el.remove();

// ✅ Solo eliminar headers de navegación
if (cls.match(/^(site-header|main-header|page-header)$/)) el.remove();
```

---

#### Fase 4: css_selector con Fallback (Robustez)

**Problema**: Algunos sitios no usan contenedores estándar (`main`, `article`).

**Solución**: Dos configuraciones con fallback automático.

```python
# Configuración principal: intenta extraer de contenedores estándar
crawler_cfg = CrawlerRunConfig(
    css_selector="main, article, .content, .post-content, #content",
    js_code=js_remove_overlays,
)

# Configuración fallback: sin css_selector (captura todo el body limpio)
crawler_cfg_fallback = CrawlerRunConfig(
    # SIN css_selector
    js_code=js_remove_overlays,
)

# Lógica de fallback
result = await crawler.arun(url=url, config=crawler_cfg)
word_count = len(result.markdown.raw_markdown.split())

if word_count < 200:  # Contenido muy corto = css_selector no funcionó
    result = await crawler.arun(url=url, config=crawler_cfg_fallback)
```

**Por qué funciona**:
- Si `css_selector` encuentra contenido → lo usa
- Si no → el fallback captura todo el body (ya limpio por JS)
- Umbral de 200 palabras evita páginas con solo navegación

---

#### Fase 5: Markdown Post-procesamiento (Pulido Final)

**Problema**: Algunas cookies y navegación escapaban al JS cleanup (cargadas dinámicamente).

**Solución**: Función `clean_markdown()` para post-procesamiento.

```python
def clean_markdown(md_text: str) -> str:
    lines = md_text.split('\n')

    # 1. Encontrar primer contenido real (heading o párrafo largo)
    first_content_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Headings son contenido
        if stripped.startswith('#') and len(stripped) > 2:
            first_content_idx = i
            break
        # Párrafos largos (>50 chars) son contenido
        if len(stripped) > 50 and not stripped.startswith('['):
            first_content_idx = i
            break

    # Descartar navegación al inicio
    lines = lines[first_content_idx:]

    # 2. Eliminar líneas de cookies
    cleaned = []
    for line in lines:
        if not _is_cookie_line(line):
            cleaned.append(line)

    return '\n'.join(cleaned)

def _is_cookie_line(line: str) -> bool:
    lower = line.lower()
    # Indicadores inequívocos
    indicators = ['cookiebot', '_ga', '_gid', 'hubspot', 'hotjar']
    if any(ind in lower for ind in indicators):
        return True
    # Keywords (2+ coincidencias)
    keywords = ['cookie', 'consent', 'aceptar', 'privacidad']
    if sum(1 for kw in keywords if kw in lower) >= 2:
        return True
    return False
```

---

#### Fase 6: Estabilidad del Crawler (Producción)

**Problemas de estabilidad encontrados**:

1. **Crawler colgado en `networkidle`**
   ```python
   # ❌ Se cuelga con páginas que tienen websockets
   wait_until="networkidle"

   # ✅ Solución: usar "load"
   wait_until="load"
   ```

2. **URLs de assets siendo crawleadas**
   ```python
   # ❌ El crawler intentaba procesar imágenes
   /blog/imagenes/foto.jpg  → Error: browser closed

   # ✅ Solución: filtrar antes de añadir a cola
   skip_patterns = ['/imagenes/', '.jpg', '.png', '.pdf', '.css', '.js']
   if any(pat in url.lower() for pat in skip_patterns):
       continue  # No añadir a cola
   ```

3. **Memory leaks después de muchas páginas**
   ```python
   # ❌ Browser se quedaba sin memoria después de 800+ páginas

   # ✅ Solución: reinicio preventivo cada 300 páginas
   if page_num % 300 == 0 and page_num > 0:
       log("Reinicio preventivo del navegador")
       break  # Sale del while interno, reinicia browser
   ```

4. **Resume no funcionaba correctamente**
   ```python
   # ❌ URLs externas en query string matcheaban el dominio
   if base_domain in url:  # Matchea twitter.com/...?url=https://mi-sitio.com

   # ✅ Solución: usar urlparse para verificar dominio
   from urllib.parse import urlparse
   if urlparse(url).netloc == base_domain:  # Solo dominio real
   ```

5. **Timeout en páginas lentas**
   ```python
   # ❌ Sin timeout, se quedaba esperando indefinidamente
   result = await crawler.arun(url=url)

   # ✅ Solución: timeout explícito
   result = await asyncio.wait_for(
       crawler.arun(url=url, config=crawler_cfg),
       timeout=90.0
   )
   ```

---

### Arquitectura Final

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PIPELINE DE EXTRACCIÓN                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. URL Discovery                                                    │
│     ├─ Sitemap XML                                                   │
│     ├─ Robots.txt (respetado)                                       │
│     └─ Enlaces internos durante crawl                               │
│                                                                      │
│  2. Filtrado Pre-crawl                                              │
│     ├─ Excluir assets (.jpg, .pdf, /images/, etc.)                  │
│     ├─ Excluir URLs bloqueadas por robots.txt                       │
│     └─ Excluir URLs ya visitadas (resume)                           │
│                                                                      │
│  3. Extracción (3 capas de limpieza)                                │
│     ├─ JavaScript: elimina cookies, nav, footer, overlays           │
│     ├─ css_selector: enfoca en main/article (con fallback)          │
│     └─ Markdown cleanup: elimina cookies residuales                 │
│                                                                      │
│  4. Extracción de Enlaces                                           │
│     ├─ Localización: nav, footer, sidebar, content                  │
│     └─ Peso para PageRank                                           │
│                                                                      │
│  5. Almacenamiento                                                   │
│     ├─ Parquet (pages): url, title, markdown, html, metadata        │
│     └─ Parquet (links): source, target, anchor, location, weight    │
│                                                                      │
│  6. Robustez                                                         │
│     ├─ Timeout 90s por página                                       │
│     ├─ Reinicio browser cada 300 páginas                            │
│     ├─ Auto-recovery en crashes                                      │
│     ├─ Guardado incremental cada 10 páginas                         │
│     └─ Watchdog para detectar cuelgues                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Checklist para Replicar en Nuevo Proyecto

#### 1. Dependencias

```bash
pip install crawl4ai beautifulsoup4 aiohttp pandas pyarrow tqdm
playwright install  # Instalar navegadores
```

#### 2. Configuración Base del Crawler

```python
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# Browser config
browser_cfg = BrowserConfig(headless=True, verbose=False)

# JavaScript cleanup (COPIAR COMPLETO)
js_remove_overlays = '''
// Cookies
document.querySelectorAll('[id*="cookie"], [class*="cookie"]').forEach(e => e.remove());
// Navegación
document.querySelectorAll('nav, header, footer, aside').forEach(e => e.remove());
// Fixed elements
document.querySelectorAll('*').forEach(e => {
    if (getComputedStyle(e).position === 'fixed') e.remove();
});
'''

# Crawler config principal
crawler_cfg = CrawlerRunConfig(
    markdown_generator=DefaultMarkdownGenerator(),
    css_selector="main, article, .content, .post-content, #content",
    js_code=js_remove_overlays,
    wait_until="load",  # NO usar networkidle
    page_timeout=60000,
    excluded_tags=['nav', 'header', 'footer', 'aside', 'script', 'style'],
)

# Crawler config fallback (sin css_selector)
crawler_cfg_fallback = CrawlerRunConfig(
    markdown_generator=DefaultMarkdownGenerator(),
    # SIN css_selector
    js_code=js_remove_overlays,
    wait_until="load",
    page_timeout=60000,
    excluded_tags=['nav', 'header', 'footer', 'aside', 'script', 'style'],
)
```

#### 3. Loop Principal con Robustez

```python
import asyncio
from urllib.parse import urlparse

async def crawl_site(start_url: str, max_pages: int = 0):
    base_domain = urlparse(start_url).netloc
    visited = set()
    to_visit = [start_url]
    results = []

    # Patrones de assets a saltar
    skip_patterns = ['/images/', '.jpg', '.png', '.pdf', '.css', '.js']

    browser_restarts = 0
    max_restarts = 5

    while to_visit and browser_restarts < max_restarts:
        try:
            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                page_count = 0

                while to_visit and (max_pages == 0 or len(visited) < max_pages):
                    url = to_visit.pop(0)

                    # Filtros
                    if url in visited:
                        continue
                    if urlparse(url).netloc != base_domain:
                        continue
                    if any(pat in url.lower() for pat in skip_patterns):
                        continue

                    visited.add(url)
                    page_count += 1

                    # Crawl con timeout
                    try:
                        result = await asyncio.wait_for(
                            crawler.arun(url=url, config=crawler_cfg),
                            timeout=90.0
                        )

                        if result.success:
                            markdown = result.markdown.raw_markdown
                            word_count = len(markdown.split())

                            # Fallback si contenido muy corto
                            if word_count < 200:
                                result = await asyncio.wait_for(
                                    crawler.arun(url=url, config=crawler_cfg_fallback),
                                    timeout=90.0
                                )
                                markdown = result.markdown.raw_markdown

                            # Post-procesamiento
                            markdown = clean_markdown(markdown)

                            results.append({
                                'url': url,
                                'title': result.metadata.get('title', ''),
                                'markdown': markdown,
                                'word_count': len(markdown.split()),
                            })

                            # Extraer enlaces internos
                            for link in extract_links(result.html, base_domain):
                                if link not in visited and link not in to_visit:
                                    to_visit.append(link)

                    except asyncio.TimeoutError:
                        print(f"Timeout: {url}")
                        continue

                    # Reinicio preventivo
                    if page_count % 300 == 0:
                        print("Reinicio preventivo del browser")
                        break

                    await asyncio.sleep(0.5)  # Rate limiting

        except Exception as e:
            browser_restarts += 1
            print(f"Browser crash #{browser_restarts}: {e}")
            continue

    return results
```

#### 4. Función de Limpieza de Markdown

```python
def clean_markdown(md_text: str) -> str:
    """Limpia navegación y cookies del markdown."""
    if not md_text:
        return ""

    lines = md_text.split('\n')

    # 1. Encontrar primer contenido real
    first_content_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#') and len(stripped) > 2:
            first_content_idx = i
            break
        if len(stripped) > 50 and not stripped.startswith('['):
            first_content_idx = i
            break

    lines = lines[first_content_idx:]

    # 2. Eliminar líneas de cookies
    cookie_indicators = ['cookie', 'consent', '_ga', '_gid', 'hubspot', 'gdpr']
    cleaned = []
    for line in lines:
        lower = line.lower()
        if sum(1 for ind in cookie_indicators if ind in lower) < 2:
            cleaned.append(line)

    return '\n'.join(cleaned)
```

#### 5. Extracción de Enlaces

```python
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def extract_links(html: str, base_domain: str) -> list:
    """Extrae enlaces internos del HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    links = []

    for a in soup.find_all('a', href=True):
        href = a['href']

        # Normalizar URL
        if href.startswith('/'):
            href = f"https://{base_domain}{href}"
        elif not href.startswith('http'):
            continue

        # Solo enlaces internos
        if urlparse(href).netloc == base_domain:
            links.append(href)

    return list(set(links))
```

---

### Errores Comunes y Soluciones

| Error | Causa | Solución |
|-------|-------|----------|
| Crawl se cuelga | `wait_until="networkidle"` | Usar `wait_until="load"` |
| BrowserContext closed | Memory leak | Reinicio cada 300 páginas |
| Contenido vacío | css_selector no match | Fallback sin css_selector |
| Cookies en contenido | Banner dinámico | `clean_markdown()` post-proceso |
| H1 desaparece | JS elimina `content-header` | Patrones específicos en JS |
| Resume no funciona | Domain check incorrecto | Usar `urlparse().netloc` |
| Imágenes crawleadas | Sin filtro de assets | Skip patterns antes de cola |

---

### Métricas de Éxito

Un crawler bien configurado debería lograr:

- **>95%** de páginas con contenido limpio (sin cookies/nav)
- **<5%** de errores por timeout o crashes
- **0%** de URLs de assets en la cola
- **Headings y negritas preservados** en el markdown
- **Resume funcional** sin duplicar páginas

---

## Historial de Cambios

### Enero 2026 (v2) - Estabilidad y Filtrado de Assets

**Problemas encontrados en producción:**
- Crawler se colgaba después de ~800 páginas (memory leak del browser)
- URLs de imágenes (`/imagenes/`, `.jpg`) causaban crashes
- Resume cargaba URLs externas incorrectamente (twitter, facebook en query strings)
- Directorios anidados incorrectos (`www_sitio/www_sitio/`)

**Soluciones implementadas:**

1. **Filtrado de URLs de assets** (3 puntos de control)
   ```python
   skip_patterns = [
       '/imagenes/', '/images/', '/img/', '/assets/',
       '.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg',
       '.pdf', '.doc', '.css', '.js', '.xml', '.json'
   ]
   ```
   - Al cargar URLs descubiertas (resume)
   - Al añadir nuevas URLs a la cola
   - Al procesar URLs de la cola

2. **Reinicio preventivo del browser**
   - Cada 300 páginas se reinicia el navegador
   - Previene memory leaks antes de que causen crashes
   - No cuenta como "error" - es mantenimiento preventivo

3. **Fix de domain check en resume**
   ```python
   # ❌ Antes: matcheaba URLs externas con el dominio en query string
   if base_domain in url

   # ✅ Después: verifica solo el netloc
   if urlparse(url).netloc == base_domain
   ```

4. **Fix de path de output-dir**
   - El crawler service ahora pasa el directorio base
   - El script añade el dominio una sola vez

5. **Contador de assets saltados**
   - Nueva métrica `skipped_assets` en el resumen final
   - Visible en logs y status

**Resultado:**
- Crawls de 1000+ páginas sin crashes
- Resume funciona correctamente
- No se procesan assets/imágenes
- Estructura de directorios correcta

---

### Enero 2026 (v1) - Reescritura del Sistema de Limpieza

**Problema original:**
- Las cookies aparecían en el contenido crawleado
- PruningContentFilter eliminaba headings y negritas
- css_selector perdía contenido en páginas con HTML no estándar

**Solución implementada:**

1. **Eliminado PruningContentFilter**
   - Causaba pérdida de headings (`## Título`) y negritas (`**texto**`)
   - Ahora se usa `raw_markdown` sin filtrar

2. **Añadido JavaScript cleanup** (ejecuta ANTES de capturar HTML)
   - Elimina banners de cookies (Cookiebot, OneTrust, Didomi, genéricos)
   - Elimina navegación (nav, `<header>`, `<footer>`, aside)
   - Elimina elementos fixed/sticky (overlays, modales)
   - Elimina formularios y widgets de chat
   - **NOTA**: El patrón de clases NO usa `header` genérico para no eliminar `content-header`, `post-header`, etc.

3. **Añadido css_selector con fallback**
   - Intenta extraer de `main, article, .content, .post-content`
   - Si obtiene <200 palabras, re-crawlea SIN css_selector
   - El fallback captura todo el body (ya limpiado por JS)

4. **Añadida función `clean_markdown()`**
   - Encuentra el primer contenido real (heading o párrafo largo)
   - Descarta navegación al inicio del documento
   - Elimina líneas de cookies en cualquier parte del contenido
   - Detección universal por keywords, indicadores y patrones de variables

5. **Extracción de título desde H1**
   - Si css_selector deja el título vacío, lo extrae del primer H1

6. **Fix: H1 eliminado en páginas de blog**
   - **Bug**: El regex `\b(header)\b` en el JS eliminaba elementos con clase `content-header`
   - En blogs, el H1 del artículo estaba dentro de `<div class="content-header">`
   - El JS eliminaba ese div y perdía el H1 y la metadata del artículo
   - **Fix**: Cambiar el patrón a `main-header|site-header|page-header` (específicos de navegación)
   - No eliminar headers de contenido como `content-header`, `post-header`, `article-header`

7. **CSS Selector universal**
   - Un solo selector que incluye todos los contenedores posibles de contenido:
     - Semánticos: `main`, `article`, `[role='main']`
     - Blogs: `.blog-post`, `.post-content`, `.entry-content`, `.article-content`
     - Landings: `#home`, `.home-main`
     - Genéricos: `#content`, `.content`, `.main-content`, `.page-content`
   - El mecanismo de fallback (sin selector) + filtro de contenido corto cubre los casos restantes

**Resultado:**
- Contenido limpio sin cookies ni navegación
- Headings y negritas preservados
- Funciona universalmente sin hardcodear para sitios específicos
- H1 y títulos preservados en páginas de blog y landings
- Páginas sin contenido real se saltan automáticamente

---

## Checklist Antes de Modificar

- [ ] ¿El cambio afecta a `clean_markdown()`? → Verificar que no elimina contenido real
- [ ] ¿El cambio afecta al JavaScript cleanup? → Verificar que no elimina contenido real
- [ ] ¿El cambio afecta al css_selector? → Verificar que el fallback sigue funcionando
- [ ] ¿El cambio afecta al schema de parquet? → Verificar compatibilidad con ingest
- [ ] ¿El cambio afecta a links? → Verificar weights para PageRank

---

## Conclusión

Este sistema proporciona un pipeline completo para:

1. **Crawlear** sitios web con limpieza integrada de cookies y navegación
2. **Extraer** contenido limpio preservando formato (headings, negritas, listas)
3. **Refinar** opcionalmente con patrones manuales por template
4. **Preparar** los datos para sistemas RAG con contenido de alta calidad

La arquitectura de 3 capas (JS cleanup → css_selector con fallback → markdown cleanup) garantiza contenido limpio de forma universal sin necesidad de configuración específica por sitio.
