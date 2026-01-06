"""
Manual Cleaner Service - Human-controlled markdown cleaning.

Flow:
1. Cluster pages by HTML fingerprint (template detection)
2. Show samples for each template
3. User manually defines patterns to remove (markdown or CSS selectors)
4. Apply patterns to all pages of that template
5. Preview and confirm before saving

The user controls everything - no automatic LLM cleaning.

Pattern types:
- Markdown patterns: exact, prefix, contains, regex, line_range
- HTML patterns: css_selector (removes elements from HTML, regenerates markdown)
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
from glob import glob

from bs4 import BeautifulSoup, NavigableString
import pandas as pd

# For HTML to markdown conversion
try:
    from markdownify import markdownify as md
    HAS_MARKDOWNIFY = True
except ImportError:
    HAS_MARKDOWNIFY = False


# =============================================================================
# HTML UTILITIES
# =============================================================================

def html_to_markdown(html: str) -> str:
    """Convert HTML to markdown using markdownify or fallback."""
    if not html:
        return ""

    if HAS_MARKDOWNIFY:
        try:
            return md(html, heading_style="ATX", strip=['script', 'style'])
        except Exception as e:
            print(f"[ManualCleaner] markdownify error: {e}")

    # Fallback: basic text extraction with BeautifulSoup
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script and style tags
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        return html


def parse_css_selectors(value: str) -> List[str]:
    """
    Parse a string that may contain multiple CSS selectors.
    Supports selectors separated by newlines, commas, or spaces.

    Args:
        value: String with one or more CSS selectors

    Returns:
        List of individual CSS selectors (cleaned and deduplicated)
    """
    if not value:
        return []

    selectors = []

    # First split by newlines
    lines = value.split('\n')

    for line in lines:
        line = line.strip()
        if not line or line.startswith('//') or line.startswith('#//'):
            # Skip empty lines and comments
            continue

        # Check if line contains multiple selectors separated by comma
        # But be careful: ".class1, .class2" is a valid CSS selector group
        # We want to split "selector1, selector2" but keep valid CSS
        # Simple approach: if line starts with selector char, treat as single or split by comma

        # Split by comma if it looks like multiple selectors
        if ',' in line and not line.startswith('['):
            parts = [p.strip() for p in line.split(',')]
            for part in parts:
                if part and (part.startswith('.') or part.startswith('#') or part.startswith('[') or part.isalpha() or part.startswith('*')):
                    selectors.append(part)
        else:
            selectors.append(line)

    # Remove duplicates while preserving order
    seen = set()
    unique_selectors = []
    for s in selectors:
        if s and s not in seen:
            seen.add(s)
            unique_selectors.append(s)

    return unique_selectors


def apply_css_selectors(html: str, selectors: List[str]) -> str:
    """
    Remove elements matching CSS selectors from HTML.

    Args:
        html: The HTML content
        selectors: List of CSS selectors to remove (can include multi-selector strings)

    Returns:
        Cleaned HTML with matching elements removed
    """
    if not html or not selectors:
        return html

    try:
        soup = BeautifulSoup(html, "html.parser")

        for selector_value in selectors:
            # Parse each selector value (may contain multiple selectors)
            parsed_selectors = parse_css_selectors(selector_value)

            for selector in parsed_selectors:
                try:
                    elements = soup.select(selector)
                    for el in elements:
                        el.decompose()
                except Exception as e:
                    print(f"[ManualCleaner] CSS selector error '{selector}': {e}")

        return str(soup)
    except Exception as e:
        print(f"[ManualCleaner] HTML parsing error: {e}")
        return html


# =============================================================================
# AUTO-CLEANING FUNCTIONS
# =============================================================================

@dataclass
class AutoCleanOptions:
    """Opciones de auto-limpieza activables."""
    extract_from_first_heading: bool = True  # Extraer desde primer H1/H2
    remove_footer_content: bool = True  # Eliminar contenido de footer
    remove_empty_lines: bool = True  # Eliminar líneas vacías excesivas
    remove_nav_patterns: bool = True  # Eliminar patrones de navegación comunes
    use_semantic_tags: bool = True  # Usar etiquetas HTML semánticas (<footer>, <nav>, etc.)
    min_heading_level: int = 1  # Nivel mínimo de heading (1=H1, 2=H2, etc.)

    def to_dict(self):
        return {
            "extract_from_first_heading": self.extract_from_first_heading,
            "remove_footer_content": self.remove_footer_content,
            "remove_empty_lines": self.remove_empty_lines,
            "remove_nav_patterns": self.remove_nav_patterns,
            "use_semantic_tags": self.use_semantic_tags,
            "min_heading_level": self.min_heading_level,
        }


# Patrones de navegación comunes a eliminar
NAV_PATTERNS = [
    "Inicio >",
    "Home >",
    "Breadcrumb",
    "Ir al contenido",
    "Skip to content",
    "Menú principal",
    "Main menu",
    "Buscar",
    "Search",
]

# Patrones que indican inicio de footer (DEBEN SER ESPECÍFICOS)
# Los patrones "exactos" deben coincidir con líneas que EMPIEZAN con el patrón
# Los patrones con ## son headings de footer
FOOTER_PATTERNS_HEADINGS = [
    # Headings de footer (coinciden si la línea EMPIEZA con esto)
    "## navegación",
    "## navigation",
    "## footer",
    "## pie de página",
    "## enlaces relacionados",
    "## related links",
    "## compartir",
    "## share",
    "## síguenos",
    "## follow us",
    "## redes sociales",
    "## social",
    "## artículos relacionados",
    "## related articles",
    "## te puede interesar",
    "## también te puede interesar",
    "## formación relacionada",
    "## related",
    "## contacto",
    "## contact",
]

# Patrones que solo son footer si están AL INICIO de una línea (solos o casi)
FOOTER_PATTERNS_LINE_START = [
    "compartir en:",
    "share on:",
    "compartir:",
    "---",  # Separador horizontal (debe ser toda la línea o casi)
]

# Patrones legales que son CLAROS indicadores de footer (pueden estar en medio)
FOOTER_PATTERNS_LEGAL = [
    "© ",
    "copyright ",
    "todos los derechos reservados",
    "all rights reserved",
]

# Combinación para backward compatibility
FOOTER_PATTERNS = FOOTER_PATTERNS_HEADINGS + FOOTER_PATTERNS_LINE_START + FOOTER_PATTERNS_LEGAL


def clean_html_semantic(html: str, remove_tags: list = None) -> str:
    """
    Elimina elementos HTML semánticos y regenera markdown limpio.

    Args:
        html: Contenido HTML completo
        remove_tags: Lista de tags a eliminar (default: footer, nav, aside, header)

    Returns:
        Markdown limpio generado desde el HTML sin los elementos semánticos
    """
    if not html:
        return ""

    if remove_tags is None:
        remove_tags = ["footer", "nav", "aside", "header"]

    try:
        from bs4 import BeautifulSoup
        import html2text

        soup = BeautifulSoup(html, "html.parser")

        # Eliminar todos los elementos de las tags especificadas
        for tag in remove_tags:
            for element in soup.find_all(tag):
                element.decompose()

        # También eliminar por clases comunes de footer/nav
        footer_classes = ["footer", "site-footer", "page-footer", "main-footer"]
        nav_classes = ["nav", "navbar", "navigation", "menu", "sidebar"]

        for class_name in footer_classes + nav_classes:
            for element in soup.find_all(class_=lambda x: x and class_name in x.lower() if isinstance(x, str) else False):
                element.decompose()
            for element in soup.find_all(id=lambda x: x and class_name in x.lower() if isinstance(x, str) else False):
                element.decompose()

        # Convertir a markdown
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.body_width = 0  # No wrap
        h.ignore_emphasis = False

        clean_html = str(soup)
        markdown = h.handle(clean_html)

        return markdown.strip()

    except ImportError as e:
        logger.warning(f"html2text not available, falling back to raw markdown: {e}")
        return ""
    except Exception as e:
        logger.warning(f"Error cleaning HTML: {e}")
        return ""


def auto_clean_markdown(markdown: str, options: AutoCleanOptions = None, html: str = None) -> str:
    """
    Limpieza automática del markdown con opciones configurables.

    Args:
        markdown: Contenido markdown a limpiar
        options: Opciones de limpieza (usa defaults si no se especifica)
        html: Contenido HTML opcional para usar etiquetas semánticas

    Returns:
        Markdown limpio
    """
    if not markdown:
        return ""

    if options is None:
        options = AutoCleanOptions()

    # Si tenemos HTML y use_semantic_tags está activo, limpiar usando etiquetas semánticas
    if html and options.use_semantic_tags:
        semantic_markdown = clean_html_semantic(html)
        if semantic_markdown:
            # Usar el markdown generado desde HTML limpio como base
            markdown = semantic_markdown

    lines = markdown.split('\n')
    result_lines = lines.copy()

    # 1. Extraer desde primer heading
    if options.extract_from_first_heading:
        first_heading_idx = None
        heading_prefix = '#' * options.min_heading_level

        for i, line in enumerate(result_lines):
            stripped = line.strip()
            # Buscar heading del nivel mínimo o superior
            if stripped.startswith('#'):
                # Contar cuántos # hay
                hash_count = len(stripped) - len(stripped.lstrip('#'))
                if hash_count >= options.min_heading_level and hash_count <= 6:
                    # Verificar que hay texto después de los #
                    after_hashes = stripped.lstrip('#').strip()
                    if after_hashes:
                        first_heading_idx = i
                        break

        if first_heading_idx is not None:
            result_lines = result_lines[first_heading_idx:]

    # 2. Eliminar patrones de navegación
    if options.remove_nav_patterns:
        filtered_lines = []
        for line in result_lines:
            line_lower = line.strip().lower()
            should_remove = False
            for pattern in NAV_PATTERNS:
                if pattern.lower() in line_lower:
                    should_remove = True
                    break
            if not should_remove:
                filtered_lines.append(line)
        result_lines = filtered_lines

    # 3. Eliminar contenido de footer
    if options.remove_footer_content:
        footer_start_idx = len(result_lines)
        for i, line in enumerate(result_lines):
            line_lower = line.strip().lower()
            is_footer = False

            # Headings: deben empezar con el patrón
            for pattern in FOOTER_PATTERNS_HEADINGS:
                if line_lower.startswith(pattern.lower()):
                    is_footer = True
                    break

            # Line start patterns: deben empezar con el patrón (permite espacios)
            if not is_footer:
                for pattern in FOOTER_PATTERNS_LINE_START:
                    if line_lower.startswith(pattern.lower()):
                        is_footer = True
                        break

            # Legal patterns: pueden estar en cualquier parte pero solo si
            # la línea es corta (típico de footers, no de contenido)
            if not is_footer:
                for pattern in FOOTER_PATTERNS_LEGAL:
                    if pattern.lower() in line_lower and len(line_lower) < 150:
                        is_footer = True
                        break

            if is_footer:
                footer_start_idx = i
                break
        result_lines = result_lines[:footer_start_idx]

    # 4. Limpiar líneas vacías excesivas
    result = '\n'.join(result_lines).strip()

    if options.remove_empty_lines:
        # Máximo 2 líneas vacías seguidas
        while '\n\n\n' in result:
            result = result.replace('\n\n\n', '\n\n')

    return result


# =============================================================================
# URL PATH PATTERN EXTRACTION
# =============================================================================

def extract_url_path_pattern(url: str) -> str:
    """
    Extract a simplified path pattern from a URL to group similar pages.
    Uses only the first path segment for broad grouping.

    Examples:
        /blog/some-article -> /blog
        /es/ciclo-grado-medio-xxx-123 -> /es
        /fp-madrid/curso-abc -> /fp-*
        /institucional/comunicado-2025 -> /institucional
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        if not path:
            return "/"

        parts = path.split("/")

        # Only use first segment for grouping
        first_part = parts[0]

        # Detect patterns like fp-madrid, fp-sevilla -> fp-*
        if first_part.startswith("fp-"):
            return "/fp-*"

        return "/" + first_part

    except Exception:
        return "/"


# =============================================================================
# HTML FINGERPRINTING
# =============================================================================

def extract_html_fingerprint(html: str) -> str:
    """
    Extract a fingerprint of the HTML structure to group pages by template.
    Pages with the same template will have the same fingerprint.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        fingerprint_parts = []

        # 1. Body classes (most reliable for templates)
        body = soup.find("body")
        if body:
            body_classes = sorted(body.get("class", []))[:5]
            if body_classes:
                fingerprint_parts.append(f"body:{','.join(body_classes)}")
            body_id = body.get("id", "")
            if body_id:
                fingerprint_parts.append(f"body#{body_id}")

        # 2. Main structural elements presence and classes
        for tag in ["header", "nav", "main", "article", "section", "aside", "footer"]:
            elements = soup.find_all(tag, limit=2)
            if elements:
                classes = []
                for el in elements:
                    el_classes = el.get("class", [])[:2]
                    if el_classes:
                        classes.extend(el_classes)
                if classes:
                    fingerprint_parts.append(f"{tag}:{','.join(sorted(set(classes))[:3])}")
                else:
                    fingerprint_parts.append(tag)

        # 3. Key div containers (first level inside body)
        if body:
            top_divs = body.find_all("div", recursive=False, limit=3)
            for div in top_divs:
                div_id = div.get("id", "")
                div_classes = div.get("class", [])[:2]
                if div_id:
                    fingerprint_parts.append(f"div#{div_id}")
                elif div_classes:
                    fingerprint_parts.append(f"div.{'.'.join(div_classes)}")

        # Create hash of fingerprint
        fingerprint = "|".join(fingerprint_parts)
        if fingerprint:
            return hashlib.md5(fingerprint.encode()).hexdigest()[:8]

        return "unknown"

    except Exception:
        return "unknown"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CleaningPattern:
    """A pattern to remove from content.

    Pattern types:
    - Markdown patterns (applied to markdown text):
        - "exact": Remove exact text match
        - "prefix": Remove lines starting with this text
        - "contains": Remove lines containing this text
        - "regex": Remove using regex pattern
        - "line_range": Remove lines from start to end (value = "start_text|||end_text")

    - HTML patterns (applied to HTML, then regenerate markdown):
        - "css_selector": Remove elements matching CSS selector (e.g., "nav", ".sidebar", "#footer")
    """
    id: str
    pattern_type: str  # "exact", "prefix", "contains", "regex", "line_range", "css_selector"
    value: str  # The pattern value (text or CSS selector)
    description: str = ""
    created_at: str = ""

    def to_dict(self):
        return asdict(self)

    @property
    def is_html_pattern(self) -> bool:
        """Check if this pattern operates on HTML rather than markdown."""
        return self.pattern_type in ("css_selector",)


@dataclass
class TemplateInfo:
    """Information about a detected template."""
    template_id: str
    fingerprint: str  # The HTML fingerprint (kept for compatibility)
    path_pattern: str  # URL path pattern (primary grouping)
    page_count: int
    sample_urls: List[str]
    patterns: List[CleaningPattern] = field(default_factory=list)
    is_cleaned: bool = False


# =============================================================================
# MANUAL CLEANER SERVICE
# =============================================================================

class ManualCleanerService:
    """Service for human-controlled markdown cleaning."""

    def __init__(self, data_dir: str = "./data/crawl4ai_data"):
        self.data_dir = Path(data_dir)
        self._templates_cache: Dict[str, Dict[str, TemplateInfo]] = {}  # crawl_name -> {template_id -> TemplateInfo}

    def _get_crawl_path(self, crawl_name: str) -> Path:
        return self.data_dir / crawl_name

    def _get_patterns_file(self, crawl_name: str) -> Path:
        return self._get_crawl_path(crawl_name) / "manual_patterns.json"

    def _load_pages(self, crawl_name: str) -> pd.DataFrame:
        """Load all pages from a crawl."""
        crawl_path = self._get_crawl_path(crawl_name)
        pages_dir = crawl_path / "pages"

        if not pages_dir.exists():
            raise ValueError(f"No pages found for crawl: {crawl_name}")

        dfs = []
        for pq_file in pages_dir.rglob("*.parquet"):
            try:
                df = pd.read_parquet(pq_file)
                dfs.append(df)
            except Exception as e:
                print(f"[ManualCleaner] Error reading {pq_file}: {e}")

        if not dfs:
            raise ValueError(f"No valid parquet files in crawl: {crawl_name}")

        return pd.concat(dfs, ignore_index=True).drop_duplicates(subset="url")

    def analyze_templates(self, crawl_name: str) -> Dict[str, Any]:
        """
        Analyze a crawl and detect templates using URL path pattern (primary)
        combined with HTML fingerprinting (secondary) for better grouping.

        Returns:
            Dict with template info and statistics.
        """
        df = self._load_pages(crawl_name)

        # Check if we have html_content
        if "html_content" not in df.columns:
            raise ValueError("Crawl doesn't have html_content - re-crawl needed")

        # Calculate path patterns (primary grouping) and fingerprints (secondary)
        print(f"[ManualCleaner] Analyzing {len(df)} pages...")
        df["path_pattern"] = df["url"].apply(extract_url_path_pattern)
        df["fingerprint"] = df["html_content"].apply(extract_html_fingerprint)

        # Group by path pattern (primary grouping criterion)
        templates = {}
        for path_pattern, group in df.groupby("path_pattern"):
            # Create readable template ID from path pattern
            pattern_slug = path_pattern.strip("/").replace("/", "_").replace("*", "x")
            if not pattern_slug:
                pattern_slug = "root"
            template_id = f"tpl_{pattern_slug}"

            # Get most common fingerprint in this group (for compatibility)
            most_common_fp = group["fingerprint"].mode().iloc[0] if len(group) > 0 else "unknown"

            sample_urls = group["url"].head(5).tolist()

            templates[template_id] = TemplateInfo(
                template_id=template_id,
                fingerprint=most_common_fp,
                path_pattern=path_pattern,
                page_count=len(group),
                sample_urls=sample_urls,
                patterns=[],
                is_cleaned=False,
            )

        # Load existing patterns if any
        patterns_file = self._get_patterns_file(crawl_name)
        if patterns_file.exists():
            with open(patterns_file) as f:
                saved = json.load(f)
                for tid, data in saved.get("templates", {}).items():
                    if tid in templates:
                        templates[tid].patterns = [
                            CleaningPattern(**p) for p in data.get("patterns", [])
                        ]
                        templates[tid].is_cleaned = data.get("is_cleaned", False)

        # Cache
        self._templates_cache[crawl_name] = templates

        # Return summary
        return {
            "crawl_name": crawl_name,
            "total_pages": len(df),
            "templates_count": len(templates),
            "templates": [
                {
                    "template_id": t.template_id,
                    "path_pattern": t.path_pattern,
                    "fingerprint": t.fingerprint,
                    "page_count": t.page_count,
                    "sample_urls": t.sample_urls[:3],
                    "patterns_count": len(t.patterns),
                    "is_cleaned": t.is_cleaned,
                }
                for t in sorted(templates.values(), key=lambda x: -x.page_count)
            ]
        }

    def get_template_sample(self, crawl_name: str, template_id: str, sample_index: int = 0) -> Dict[str, Any]:
        """
        Get a sample page from a template for manual inspection.

        Returns:
            Dict with URL, markdown content, HTML content, and current patterns.
        """
        # Ensure templates are loaded
        if crawl_name not in self._templates_cache:
            self.analyze_templates(crawl_name)

        templates = self._templates_cache.get(crawl_name, {})
        template = templates.get(template_id)

        if not template:
            raise ValueError(f"Template not found: {template_id}")

        if sample_index >= len(template.sample_urls):
            sample_index = 0

        url = template.sample_urls[sample_index]

        # Load the page
        df = self._load_pages(crawl_name)
        page = df[df["url"] == url]

        if page.empty:
            raise ValueError(f"Page not found: {url}")

        row = page.iloc[0]
        markdown = row.get("markdown", "")
        html_content = row.get("html_content", "")

        # Check if auto_clean exists for this URL
        auto_clean_content = None
        crawl_path = self._get_crawl_path(crawl_name)
        auto_clean_path = crawl_path / "auto_clean" / "all_pages.parquet"

        if auto_clean_path.exists():
            try:
                ac_df = pd.read_parquet(auto_clean_path)
                ac_page = ac_df[ac_df["url"] == url]
                if not ac_page.empty:
                    auto_clean_content = ac_page.iloc[0].get("markdown_clean", "")
            except Exception:
                pass

        # Check if we have HTML patterns that need HTML
        has_html_patterns = any(p.is_html_pattern for p in template.patterns)

        # Apply current patterns to show preview
        # If auto_clean exists, use that as the base for preview
        base_markdown = auto_clean_content if auto_clean_content else markdown
        clean_markdown, clean_html = self._apply_patterns(
            base_markdown,
            template.patterns,
            html=html_content if has_html_patterns else None,
        )

        return {
            "template_id": template_id,
            "sample_index": sample_index,
            "total_samples": len(template.sample_urls),
            "url": url,
            "title": row.get("title", ""),
            # Markdown content
            "markdown_original": markdown,
            "markdown_preview": clean_markdown,
            "original_length": len(markdown),
            "preview_length": len(clean_markdown),
            "reduction_pct": round((1 - len(clean_markdown) / len(markdown)) * 100, 1) if markdown else 0,
            # Auto-clean info
            "auto_clean_applied": auto_clean_content is not None,
            "auto_clean_content": auto_clean_content,
            "auto_clean_length": len(auto_clean_content) if auto_clean_content else 0,
            # HTML content (for CSS selector patterns)
            "html_original": html_content,
            "html_preview": clean_html if has_html_patterns else None,
            "html_length": len(html_content) if html_content else 0,
            "has_html": bool(html_content),
            # Patterns info
            "patterns": [p.to_dict() for p in template.patterns],
            "has_html_patterns": has_html_patterns,
            "page_count": template.page_count,
        }

    def add_pattern(
        self,
        crawl_name: str,
        template_id: str,
        pattern_type: str,
        value: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Add a cleaning pattern to a template.

        pattern_type options:
        - "exact": Remove exact text match (can be a phrase in the middle of content)
        - "prefix": Remove lines starting with this text
        - "contains": Remove lines containing this text
        - "regex": Remove using regex pattern
        - "line_range": Remove everything from start_text to end_text (inclusive). Format: "start_text|||end_text"
        - "text_range": Same as line_range but removes ALL occurrences (for repeated blocks)
        """
        if crawl_name not in self._templates_cache:
            self.analyze_templates(crawl_name)

        templates = self._templates_cache.get(crawl_name, {})
        template = templates.get(template_id)

        if not template:
            raise ValueError(f"Template not found: {template_id}")

        # Create pattern
        pattern = CleaningPattern(
            id=f"pat_{datetime.now().strftime('%H%M%S')}_{len(template.patterns)}",
            pattern_type=pattern_type,
            value=value,
            description=description,
            created_at=datetime.now().isoformat(),
        )

        template.patterns.append(pattern)

        # Save
        self._save_patterns(crawl_name)

        return {"status": "ok", "pattern": pattern.to_dict()}

    def remove_pattern(self, crawl_name: str, template_id: str, pattern_id: str) -> Dict[str, Any]:
        """Remove a pattern from a template."""
        if crawl_name not in self._templates_cache:
            self.analyze_templates(crawl_name)

        templates = self._templates_cache.get(crawl_name, {})
        template = templates.get(template_id)

        if not template:
            raise ValueError(f"Template not found: {template_id}")

        template.patterns = [p for p in template.patterns if p.id != pattern_id]
        self._save_patterns(crawl_name)

        return {"status": "ok", "remaining_patterns": len(template.patterns)}

    def get_available_patterns(self, crawl_name: str, current_template_id: str) -> Dict[str, Any]:
        """
        Get unique patterns from other templates in the same crawl.
        Useful for reusing patterns across templates.

        Returns:
            Dict with list of unique patterns from other templates.
        """
        if crawl_name not in self._templates_cache:
            self.analyze_templates(crawl_name)

        templates = self._templates_cache.get(crawl_name, {})

        # Collect patterns from other templates
        seen_values: set[str] = set()
        available_patterns: list[dict] = []

        # First, get patterns from current template to exclude them
        current_template = templates.get(current_template_id)
        current_values = set()
        if current_template:
            current_values = {p.value for p in current_template.patterns}

        # Collect from other templates
        for tid, template in templates.items():
            if tid == current_template_id:
                continue
            for pattern in template.patterns:
                # Skip if already in current template or already seen
                if pattern.value in current_values or pattern.value in seen_values:
                    continue
                # Skip CSS selectors (only markdown patterns)
                if pattern.pattern_type == "css_selector":
                    continue
                seen_values.add(pattern.value)
                # For line_range, the value contains start|||end
                value_end = None
                if pattern.pattern_type == "line_range" and "|||" in pattern.value:
                    parts = pattern.value.split("|||", 1)
                    value_end = parts[1] if len(parts) > 1 else None

                available_patterns.append({
                    "pattern_type": pattern.pattern_type,
                    "value": pattern.value,
                    "value_end": value_end,
                    "from_template": tid,
                    "preview": pattern.value[:50] + "..." if len(pattern.value) > 50 else pattern.value,
                })

        return {
            "crawl_name": crawl_name,
            "current_template": current_template_id,
            "available_count": len(available_patterns),
            "patterns": available_patterns,
        }

    def _apply_markdown_patterns(self, markdown: str, patterns: List[CleaningPattern]) -> str:
        """Apply markdown-only patterns to text."""
        result = markdown

        for pattern in patterns:
            # Skip HTML patterns
            if pattern.is_html_pattern:
                continue

            try:
                if pattern.pattern_type == "exact":
                    result = result.replace(pattern.value, "")

                elif pattern.pattern_type == "prefix":
                    lines = result.split('\n')
                    result = '\n'.join(
                        line for line in lines
                        if not line.strip().startswith(pattern.value)
                    )

                elif pattern.pattern_type == "contains":
                    lines = result.split('\n')
                    result = '\n'.join(
                        line for line in lines
                        if pattern.value not in line
                    )

                elif pattern.pattern_type == "regex":
                    result = re.sub(pattern.value, "", result, flags=re.MULTILINE)

                elif pattern.pattern_type == "line_range":
                    # Format: "start_text|||end_text"
                    # This removes everything BETWEEN start_text and end_text (inclusive)
                    parts = pattern.value.split("|||")
                    if len(parts) == 2:
                        start_text, end_text = parts
                        # Try character-based removal first (more precise)
                        start_idx = result.find(start_text)
                        if start_idx != -1:
                            end_idx = result.find(end_text, start_idx)
                            if end_idx != -1:
                                # Remove from start_text to end of end_text
                                result = result[:start_idx] + result[end_idx + len(end_text):]
                            else:
                                # end_text not found, remove from start_text to end
                                result = result[:start_idx]

                elif pattern.pattern_type == "text_range":
                    # NEW: Same as line_range but keeps searching for more occurrences
                    # Useful for removing repeated blocks
                    parts = pattern.value.split("|||")
                    if len(parts) == 2:
                        start_text, end_text = parts
                        while True:
                            start_idx = result.find(start_text)
                            if start_idx == -1:
                                break
                            end_idx = result.find(end_text, start_idx)
                            if end_idx != -1:
                                result = result[:start_idx] + result[end_idx + len(end_text):]
                            else:
                                result = result[:start_idx]
                                break

            except Exception as e:
                print(f"[ManualCleaner] Pattern error ({pattern.id}): {e}")
                continue

        # Clean up excessive whitespace
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip()

    def _apply_html_patterns(self, html: str, patterns: List[CleaningPattern]) -> str:
        """Apply HTML patterns (CSS selectors) to HTML content."""
        if not html:
            return html

        # Collect all CSS selectors
        css_selectors = [
            p.value for p in patterns
            if p.pattern_type == "css_selector" and p.value
        ]

        if not css_selectors:
            return html

        return apply_css_selectors(html, css_selectors)

    def _apply_patterns(
        self,
        markdown: str,
        patterns: List[CleaningPattern],
        html: str = None,
    ) -> Tuple[str, str]:
        """
        Apply all cleaning patterns to content.

        If there are CSS selector patterns and HTML is provided:
        1. Apply CSS selectors to HTML
        2. Regenerate markdown from cleaned HTML
        3. Apply markdown patterns to result

        Args:
            markdown: Original markdown content
            patterns: List of patterns to apply
            html: Original HTML content (optional, required for css_selector patterns)

        Returns:
            Tuple of (cleaned_markdown, cleaned_html)
        """
        # Separate pattern types
        html_patterns = [p for p in patterns if p.is_html_pattern]
        md_patterns = [p for p in patterns if not p.is_html_pattern]

        cleaned_html = html
        result_markdown = markdown

        # Apply HTML patterns if we have HTML and css_selector patterns
        if html and html_patterns:
            cleaned_html = self._apply_html_patterns(html, html_patterns)
            # Regenerate markdown from cleaned HTML
            result_markdown = html_to_markdown(cleaned_html)

        # Apply markdown patterns
        if md_patterns:
            result_markdown = self._apply_markdown_patterns(result_markdown, md_patterns)

        return result_markdown, cleaned_html

    def _apply_patterns_simple(self, markdown: str, patterns: List[CleaningPattern]) -> str:
        """Legacy method - apply patterns to markdown only (backwards compatibility)."""
        result, _ = self._apply_patterns(markdown, patterns, html=None)
        return result

    def preview_cleaning(self, crawl_name: str, template_id: str) -> Dict[str, Any]:
        """
        Preview cleaning results for all pages in a template.
        Shows statistics without actually saving.
        """
        if crawl_name not in self._templates_cache:
            self.analyze_templates(crawl_name)

        templates = self._templates_cache.get(crawl_name, {})
        template = templates.get(template_id)

        if not template:
            raise ValueError(f"Template not found: {template_id}")

        df = self._load_pages(crawl_name)
        df["path_pattern"] = df["url"].apply(extract_url_path_pattern)

        # Get pages for this template (using path pattern)
        template_pages = df[df["path_pattern"] == template.path_pattern]

        # Check if we have HTML patterns
        has_html_patterns = any(p.is_html_pattern for p in template.patterns)

        results = []
        for _, row in template_pages.iterrows():
            markdown = row.get("markdown", "")
            html_content = row.get("html_content", "") if has_html_patterns else None

            clean_markdown, _ = self._apply_patterns(
                markdown,
                template.patterns,
                html=html_content,
            )

            results.append({
                "url": row["url"],
                "original_length": len(markdown),
                "clean_length": len(clean_markdown),
                "reduction_pct": round((1 - len(clean_markdown) / len(markdown)) * 100, 1) if markdown else 0,
            })

        avg_reduction = sum(r["reduction_pct"] for r in results) / len(results) if results else 0

        return {
            "template_id": template_id,
            "page_count": len(results),
            "patterns_count": len(template.patterns),
            "has_html_patterns": has_html_patterns,
            "avg_reduction_pct": round(avg_reduction, 1),
            "pages": results[:20],  # First 20 for preview
        }

    def apply_cleaning(
        self,
        crawl_name: str,
        template_id: str,
        auto_clean_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Apply cleaning patterns to all pages in a template and save results.

        Args:
            crawl_name: Name of the crawl
            template_id: ID of the template to clean
            auto_clean_options: Optional dict with auto-cleaning options:
                - extract_from_first_heading: bool (default True)
                - remove_footer_content: bool (default True)
                - remove_empty_lines: bool (default True)
                - remove_nav_patterns: bool (default True)
                - min_heading_level: int (default 1)
        """
        if crawl_name not in self._templates_cache:
            self.analyze_templates(crawl_name)

        templates = self._templates_cache.get(crawl_name, {})
        template = templates.get(template_id)

        if not template:
            raise ValueError(f"Template not found: {template_id}")

        df = self._load_pages(crawl_name)
        df["path_pattern"] = df["url"].apply(extract_url_path_pattern)

        # Get pages for this template (using path pattern)
        template_pages = df[df["path_pattern"] == template.path_pattern].copy()

        # Check if we have HTML patterns
        has_html_patterns = any(p.is_html_pattern for p in template.patterns)

        # Parse auto-clean options
        auto_opts = None
        if auto_clean_options:
            auto_opts = AutoCleanOptions(
                extract_from_first_heading=auto_clean_options.get("extract_from_first_heading", True),
                remove_footer_content=auto_clean_options.get("remove_footer_content", True),
                remove_empty_lines=auto_clean_options.get("remove_empty_lines", True),
                remove_nav_patterns=auto_clean_options.get("remove_nav_patterns", True),
                use_semantic_tags=auto_clean_options.get("use_semantic_tags", True),
                min_heading_level=auto_clean_options.get("min_heading_level", 1),
            )

        # Apply patterns
        results = []
        for idx, row in template_pages.iterrows():
            markdown = row.get("markdown", "")
            # Always get html_content for semantic cleaning
            html_content = row.get("html_content", "")

            # First apply manual patterns
            clean_markdown, clean_html = self._apply_patterns(
                markdown,
                template.patterns,
                html=html_content if has_html_patterns else None,
            )

            # Then apply auto-cleaning if enabled
            if auto_opts:
                clean_markdown = auto_clean_markdown(clean_markdown, auto_opts, html=html_content)

            result_entry = {
                "url": row["url"],
                "title": row.get("title", ""),
                "template_id": template_id,
                "markdown_original": markdown,
                "markdown_clean": clean_markdown,
                "original_length": len(markdown),
                "clean_length": len(clean_markdown),
                "patterns_applied": len(template.patterns),
                "auto_clean_applied": auto_opts is not None,
            }

            # Include cleaned HTML if we applied HTML patterns
            if has_html_patterns and clean_html:
                result_entry["html_clean"] = clean_html

            results.append(result_entry)

        # Save results
        output_dir = self._get_crawl_path(crawl_name) / "manual_clean"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{template_id}.parquet"
        pd.DataFrame(results).to_parquet(output_file, engine="pyarrow")

        # Mark as cleaned
        template.is_cleaned = True
        self._save_patterns(crawl_name)

        return {
            "status": "ok",
            "template_id": template_id,
            "pages_cleaned": len(results),
            "output_file": str(output_file),
            "auto_clean_applied": auto_opts.to_dict() if auto_opts else None,
        }

    def auto_clean_all(
        self,
        crawl_name: str,
        auto_clean_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Apply auto-cleaning to ALL pages in a crawl without manual patterns.
        This is a quick way to clean an entire crawl with sensible defaults.

        Args:
            crawl_name: Name of the crawl
            auto_clean_options: Optional dict with options (uses defaults if not provided)

        Returns:
            Dict with cleaning results
        """
        df = self._load_pages(crawl_name)

        # Parse auto-clean options
        auto_opts = AutoCleanOptions()
        if auto_clean_options:
            auto_opts = AutoCleanOptions(
                extract_from_first_heading=auto_clean_options.get("extract_from_first_heading", True),
                remove_footer_content=auto_clean_options.get("remove_footer_content", True),
                remove_empty_lines=auto_clean_options.get("remove_empty_lines", True),
                remove_nav_patterns=auto_clean_options.get("remove_nav_patterns", True),
                use_semantic_tags=auto_clean_options.get("use_semantic_tags", True),
                min_heading_level=auto_clean_options.get("min_heading_level", 1),
            )

        # Apply auto-cleaning to all pages
        results = []
        for idx, row in df.iterrows():
            markdown = row.get("markdown", "")
            html_content = row.get("html_content", "")
            clean_markdown = auto_clean_markdown(markdown, auto_opts, html=html_content)

            results.append({
                "url": row["url"],
                "title": row.get("title", ""),
                "markdown_original": markdown,
                "markdown_clean": clean_markdown,
                "original_length": len(markdown),
                "clean_length": len(clean_markdown),
                "html_content": row.get("html_content", ""),
                "meta_description": row.get("meta_description", ""),
            })

        # Save results
        output_dir = self._get_crawl_path(crawl_name) / "auto_clean"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "all_pages.parquet"
        pd.DataFrame(results).to_parquet(output_file, engine="pyarrow")

        # Calculate stats
        total_original = sum(r["original_length"] for r in results)
        total_clean = sum(r["clean_length"] for r in results)
        reduction_pct = round((1 - total_clean / total_original) * 100, 1) if total_original else 0

        return {
            "status": "ok",
            "crawl_name": crawl_name,
            "pages_cleaned": len(results),
            "output_file": str(output_file),
            "auto_clean_options": auto_opts.to_dict(),
            "total_original_chars": total_original,
            "total_clean_chars": total_clean,
            "reduction_pct": reduction_pct,
        }

    def preview_auto_clean(
        self,
        crawl_name: str,
        sample_url: str = None,
        auto_clean_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Preview auto-cleaning on a sample page.

        Args:
            crawl_name: Name of the crawl
            sample_url: Optional specific URL to preview (uses first page if not provided)
            auto_clean_options: Optional dict with options

        Returns:
            Dict with original and cleaned content for comparison
        """
        df = self._load_pages(crawl_name)

        if sample_url:
            page = df[df["url"] == sample_url]
            if page.empty:
                raise ValueError(f"URL not found: {sample_url}")
            row = page.iloc[0]
        else:
            row = df.iloc[0]

        markdown = row.get("markdown", "")
        html_content = row.get("html_content", "")

        # Parse auto-clean options
        auto_opts = AutoCleanOptions()
        if auto_clean_options:
            auto_opts = AutoCleanOptions(
                extract_from_first_heading=auto_clean_options.get("extract_from_first_heading", True),
                remove_footer_content=auto_clean_options.get("remove_footer_content", True),
                remove_empty_lines=auto_clean_options.get("remove_empty_lines", True),
                remove_nav_patterns=auto_clean_options.get("remove_nav_patterns", True),
                use_semantic_tags=auto_clean_options.get("use_semantic_tags", True),
                min_heading_level=auto_clean_options.get("min_heading_level", 1),
            )

        clean_markdown = auto_clean_markdown(markdown, auto_opts, html=html_content)

        return {
            "url": row["url"],
            "title": row.get("title", ""),
            "markdown_original": markdown,
            "markdown_clean": clean_markdown,
            "original_length": len(markdown),
            "clean_length": len(clean_markdown),
            "reduction_pct": round((1 - len(clean_markdown) / len(markdown)) * 100, 1) if markdown else 0,
            "auto_clean_options": auto_opts.to_dict(),
        }

    def _save_patterns(self, crawl_name: str):
        """Save patterns to disk."""
        templates = self._templates_cache.get(crawl_name, {})

        data = {
            "updated_at": datetime.now().isoformat(),
            "templates": {
                tid: {
                    "patterns": [p.to_dict() for p in t.patterns],
                    "is_cleaned": t.is_cleaned,
                }
                for tid, t in templates.items()
            }
        }

        patterns_file = self._get_patterns_file(crawl_name)
        with open(patterns_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_cleaning_status(self, crawl_name: str) -> Dict[str, Any]:
        """Get overall cleaning status for a crawl."""
        if crawl_name not in self._templates_cache:
            self.analyze_templates(crawl_name)

        templates = self._templates_cache.get(crawl_name, {})

        total_pages = sum(t.page_count for t in templates.values())
        cleaned_pages = sum(t.page_count for t in templates.values() if t.is_cleaned)

        # Check if auto_clean exists
        crawl_path = self.data_dir / crawl_name
        auto_clean_path = crawl_path / "auto_clean" / "all_pages.parquet"
        auto_clean_status = None

        if auto_clean_path.exists():
            try:
                df = pd.read_parquet(auto_clean_path)
                total_orig = df['original_length'].sum() if 'original_length' in df.columns else 0
                total_clean = df['clean_length'].sum() if 'clean_length' in df.columns else 0
                reduction = round((1 - total_clean / total_orig) * 100, 1) if total_orig > 0 else 0
                auto_clean_status = {
                    "applied": True,
                    "pages": len(df),
                    "original_chars": int(total_orig),
                    "clean_chars": int(total_clean),
                    "reduction_pct": reduction,
                }
            except Exception as e:
                auto_clean_status = {"applied": True, "error": str(e)}

        return {
            "crawl_name": crawl_name,
            "total_templates": len(templates),
            "cleaned_templates": sum(1 for t in templates.values() if t.is_cleaned),
            "total_pages": total_pages,
            "cleaned_pages": cleaned_pages,
            "progress_pct": round(cleaned_pages / total_pages * 100, 1) if total_pages else 0,
            "auto_clean": auto_clean_status,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_manual_cleaner_service: Optional[ManualCleanerService] = None


def get_manual_cleaner_service() -> ManualCleanerService:
    """Get or create the manual cleaner service singleton."""
    global _manual_cleaner_service
    if _manual_cleaner_service is None:
        _manual_cleaner_service = ManualCleanerService()
    return _manual_cleaner_service
