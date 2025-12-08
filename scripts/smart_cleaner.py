#!/usr/bin/env python3
"""
Smart Cleaner - Post-procesamiento inteligente de HTML para RAG.

Estrategia:
1. Clustering: Agrupa URLs por patrón de path para identificar plantillas
2. Sampling + IA: Usa IA en 1 muestra por grupo para descubrir selectores CSS
3. Bulk Cleaning: Aplica selectores con BeautifulSoup (gratis y rápido)

Minimiza costes de tokens usando IA solo para descubrir reglas.
"""

import os
import re
import json
import asyncio
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

# Cargar variables de entorno
load_dotenv()

# Para ejecutar async en notebooks/scripts
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Selectores por defecto para contenido principal (fallback)
DEFAULT_CONTENT_SELECTORS = [
    "article",
    "main",
    "[role='main']",
    ".main-content",
    "#main-content",
    ".post-content",
    ".article-content",
    ".entry-content",
    ".content-area",
    "#content",
    ".page-content",
]

# Selectores a eliminar siempre (ruido común)
NOISE_SELECTORS = [
    "nav",
    "header",
    "footer",
    "aside",
    ".sidebar",
    ".navigation",
    ".menu",
    ".navbar",
    ".breadcrumb",
    ".social-share",
    ".comments",
    ".related-posts",
    ".advertisement",
    ".ad-container",
    "[role='navigation']",
    "[role='banner']",
    "[role='contentinfo']",
    "[id*='cookie']",
    "[class*='cookie']",
    "[id*='gdpr']",
    "[class*='gdpr']",
    "script",
    "style",
    "noscript",
    "iframe",
]


# =============================================================================
# CLUSTERING DE URLs
# =============================================================================

def extract_url_pattern(url: str) -> str:
    """
    Extrae un patrón de template basado en la estructura del path.

    Ejemplos:
    - https://web.com/blog/mi-articulo -> blog
    - https://web.com/productos/categoria/item -> productos_categoria
    - https://web.com/ -> home
    - https://web.com/es/fp-a-distancia/curso -> es_fp-a-distancia
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        if not path:
            return "home"

        # Dividir path en segmentos
        segments = path.split("/")

        # Tomar los primeros 2 segmentos significativos para el patrón
        pattern_parts = []
        for seg in segments[:2]:
            # Ignorar segmentos que parecen IDs o slugs específicos
            if seg and not re.match(r'^[\d]+$', seg):
                # Normalizar: quitar caracteres especiales
                clean_seg = re.sub(r'[^\w-]', '', seg.lower())
                if clean_seg:
                    pattern_parts.append(clean_seg)

        if pattern_parts:
            return "_".join(pattern_parts)

        return "other"

    except Exception:
        return "unknown"


def cluster_urls(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columna template_group basada en patrón de URL."""
    df = df.copy()
    df["template_group"] = df["url"].apply(extract_url_pattern)
    return df


# =============================================================================
# LIMPIEZA CON BEAUTIFULSOUP
# =============================================================================

def clean_html_with_selector(
    html: str,
    selector: Optional[str] = None,
    remove_noise: bool = True
) -> str:
    """
    Limpia HTML extrayendo contenido con selector específico.

    Args:
        html: HTML crudo
        selector: Selector CSS para contenido principal
        remove_noise: Si eliminar elementos de ruido antes de extraer

    Returns:
        Texto limpio
    """
    if not html or not isinstance(html, str):
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")

        # Eliminar ruido común
        if remove_noise:
            for noise_selector in NOISE_SELECTORS:
                for element in soup.select(noise_selector):
                    element.decompose()

        # Extraer contenido principal
        content = None

        if selector:
            # Usar selector proporcionado
            content = soup.select_one(selector)

        if not content:
            # Fallback: probar selectores por defecto
            for default_sel in DEFAULT_CONTENT_SELECTORS:
                content = soup.select_one(default_sel)
                if content:
                    break

        if not content:
            # Último fallback: body completo
            content = soup.body if soup.body else soup

        # Extraer texto limpio
        text = content.get_text(separator="\n", strip=True)

        # Limpiar espacios múltiples
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        return text.strip()

    except Exception as e:
        # En caso de error, devolver texto plano
        try:
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator="\n", strip=True)
        except:
            return ""


def get_html_structure_summary(html: str, max_depth: int = 3) -> str:
    """
    Genera un resumen de la estructura HTML para el prompt de IA.
    Reduce tokens enviando solo estructura, no contenido.
    """
    if not html:
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")

        def summarize_element(element, depth=0):
            if depth > max_depth:
                return ""

            if element.name is None:
                return ""

            # Construir descripción del elemento
            attrs = []
            if element.get("id"):
                attrs.append(f'id="{element.get("id")}"')
            if element.get("class"):
                classes = " ".join(element.get("class")[:3])  # Max 3 clases
                attrs.append(f'class="{classes}"')
            if element.get("role"):
                attrs.append(f'role="{element.get("role")}"')

            attr_str = " " + " ".join(attrs) if attrs else ""
            indent = "  " * depth

            # Contar texto directo
            direct_text = element.string or ""
            text_preview = direct_text[:50].strip() if direct_text else ""
            text_indicator = f' [{len(direct_text)} chars]' if text_preview else ""

            result = f"{indent}<{element.name}{attr_str}>{text_indicator}\n"

            # Recursión para hijos
            for child in element.children:
                if child.name:
                    result += summarize_element(child, depth + 1)

            return result

        body = soup.body if soup.body else soup
        return summarize_element(body)[:5000]  # Limitar a 5k chars

    except Exception:
        return ""


# =============================================================================
# ANÁLISIS CON IA (ScrapeGraphAI)
# =============================================================================

async def analyze_template_with_ai(
    html_sample: str,
    url_sample: str,
    group_name: str,
) -> Optional[str]:
    """
    Usa ScrapeGraphAI para analizar una muestra y descubrir el selector CSS.

    Returns:
        Selector CSS o None si falla
    """
    try:
        from scrapegraphai.graphs import SmartScraperGraph
    except ImportError:
        print("  [WARN] scrapegraphai no instalado. Usando selectores por defecto.")
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  [WARN] OPENAI_API_KEY no encontrada. Usando selectores por defecto.")
        return None

    # Configuración para SmartScraperGraph
    graph_config = {
        "llm": {
            "api_key": api_key,
            "model": "openai/gpt-4o-mini",  # Modelo eficiente en coste
        },
        "verbose": False,
        "headless": True,
    }

    # Prompt optimizado para descubrir selector
    prompt = """Analiza la estructura HTML proporcionada.

Tu tarea es identificar el Selector CSS único que contiene el CONTENIDO PRINCIPAL de la página.

Reglas:
1. El selector debe capturar el artículo/texto principal
2. Debe EXCLUIR: navegación, header, footer, sidebar, menús, breadcrumbs
3. Prefiere selectores específicos (con id o clase única) sobre genéricos
4. Ejemplos buenos: "article.post-content", "div#main-article", "main.content-area"
5. Ejemplos malos: "div" (muy genérico), "body" (incluye todo)

Devuelve SOLO un JSON válido con esta estructura exacta:
{"selector": "tu_selector_css_aquí", "confidence": "high/medium/low", "reason": "breve explicación"}

Si no puedes determinar un buen selector, devuelve:
{"selector": null, "confidence": "low", "reason": "explicación"}
"""

    try:
        # Crear y ejecutar el grafo
        scraper = SmartScraperGraph(
            prompt=prompt,
            source=html_sample[:50000],  # Limitar HTML a 50k chars
            config=graph_config,
        )

        result = scraper.run()

        # Parsear resultado
        if isinstance(result, dict):
            selector = result.get("selector")
            confidence = result.get("confidence", "low")

            if selector and confidence in ["high", "medium"]:
                print(f"    Selector encontrado: {selector} (confianza: {confidence})")
                return selector

        elif isinstance(result, str):
            # Intentar parsear JSON del string
            try:
                parsed = json.loads(result)
                selector = parsed.get("selector")
                if selector:
                    print(f"    Selector encontrado: {selector}")
                    return selector
            except json.JSONDecodeError:
                pass

        print(f"    No se encontró selector confiable")
        return None

    except Exception as e:
        print(f"    Error en análisis IA: {str(e)[:100]}")
        return None


async def discover_selectors_for_groups(
    df: pd.DataFrame,
    use_ai: bool = True,
    sample_size: int = 1,
) -> dict:
    """
    Descubre selectores CSS para cada grupo de templates.

    Args:
        df: DataFrame con columnas url, html_content, template_group
        use_ai: Si usar IA para descubrir selectores
        sample_size: Número de muestras por grupo para analizar

    Returns:
        Dict {grupo: selector}
    """
    rules = {}
    groups = df["template_group"].unique()

    print(f"\n[2/4] Analizando {len(groups)} grupos de templates...")

    for group in tqdm(groups, desc="Descubriendo selectores"):
        # Obtener muestra del grupo
        group_df = df[df["template_group"] == group]

        # Filtrar filas con HTML válido
        valid_samples = group_df[
            group_df["html_content"].notna() &
            (group_df["html_content"].str.len() > 500)
        ]

        if valid_samples.empty:
            print(f"  Grupo '{group}': Sin HTML válido, usando fallback")
            rules[group] = None
            continue

        # Tomar muestra
        sample = valid_samples.head(sample_size).iloc[0]
        html_sample = sample["html_content"]
        url_sample = sample["url"]

        print(f"\n  Grupo '{group}' ({len(group_df)} URLs)")
        print(f"    Muestra: {url_sample[:60]}...")

        if use_ai:
            # Usar IA para descubrir selector
            selector = await analyze_template_with_ai(
                html_sample=html_sample,
                url_sample=url_sample,
                group_name=group,
            )
            rules[group] = selector
        else:
            # Sin IA: intentar detectar selector automáticamente
            selector = auto_detect_selector(html_sample)
            rules[group] = selector
            if selector:
                print(f"    Auto-detectado: {selector}")

    return rules


def auto_detect_selector(html: str) -> Optional[str]:
    """
    Intenta detectar automáticamente el selector de contenido principal.
    Usa heurísticas basadas en la densidad de texto.
    """
    if not html:
        return None

    try:
        soup = BeautifulSoup(html, "html.parser")

        # Eliminar ruido
        for sel in NOISE_SELECTORS:
            for el in soup.select(sel):
                el.decompose()

        # Buscar elementos con mayor densidad de texto
        candidates = []

        for selector in DEFAULT_CONTENT_SELECTORS:
            element = soup.select_one(selector)
            if element:
                text_len = len(element.get_text(strip=True))
                if text_len > 500:  # Mínimo 500 caracteres
                    candidates.append((selector, text_len))

        if candidates:
            # Retornar el de mayor contenido
            best = max(candidates, key=lambda x: x[1])
            return best[0]

        return None

    except Exception:
        return None


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

async def smart_clean(
    input_path: str,
    output_path: Optional[str] = None,
    use_ai: bool = True,
    html_column: str = "html_content",
    url_column: str = "url",
    save_rules: bool = True,
) -> pd.DataFrame:
    """
    Pipeline completo de limpieza inteligente.

    Args:
        input_path: Ruta al parquet de entrada
        output_path: Ruta de salida (opcional)
        use_ai: Si usar IA para descubrir selectores
        html_column: Nombre de columna con HTML
        url_column: Nombre de columna con URL
        save_rules: Si guardar reglas descubiertas a archivo

    Returns:
        DataFrame limpio
    """
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║              Smart Cleaner - Post-procesamiento HTML             ║
╠══════════════════════════════════════════════════════════════════╣
║  Input:  {str(input_path):<55} ║
║  AI:     {str(use_ai):<55} ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # [1/4] Cargar datos
    print("[1/4] Cargando datos...")
    df = pd.read_parquet(input_path)
    print(f"  Filas cargadas: {len(df)}")

    # Verificar columnas requeridas
    if html_column not in df.columns:
        raise ValueError(f"Columna '{html_column}' no encontrada en el parquet")
    if url_column not in df.columns:
        raise ValueError(f"Columna '{url_column}' no encontrada en el parquet")

    # [2/4] Clustering
    print("\n[2/4] Clustering por patrón de URL...")
    df = cluster_urls(df)
    groups = df["template_group"].value_counts()
    print(f"  Grupos identificados: {len(groups)}")
    print(f"  Top 5 grupos:")
    for group, count in groups.head(5).items():
        print(f"    - {group}: {count} URLs")

    # [3/4] Descubrir selectores
    rules = await discover_selectors_for_groups(df, use_ai=use_ai)

    # Guardar reglas
    if save_rules:
        rules_path = Path(input_path).parent / "cleaning_rules.json"
        with open(rules_path, "w") as f:
            json.dump(rules, f, indent=2)
        print(f"\n  Reglas guardadas en: {rules_path}")

    # [4/4] Aplicar limpieza masiva
    print("\n[3/4] Aplicando limpieza masiva con BeautifulSoup...")

    def clean_row(row):
        html = row[html_column]
        group = row["template_group"]
        selector = rules.get(group)
        return clean_html_with_selector(html, selector)

    tqdm.pandas(desc="Limpiando HTML")
    df["clean_text"] = df.progress_apply(clean_row, axis=1)

    # Estadísticas
    df["clean_word_count"] = df["clean_text"].apply(lambda x: len(x.split()) if x else 0)

    avg_original = df["word_count"].mean() if "word_count" in df.columns else 0
    avg_clean = df["clean_word_count"].mean()

    print(f"\n[4/4] Estadísticas de limpieza:")
    print(f"  Promedio palabras original: {avg_original:.0f}")
    print(f"  Promedio palabras limpio: {avg_clean:.0f}")
    print(f"  Reducción: {((avg_original - avg_clean) / max(avg_original, 1)) * 100:.1f}%")

    # Guardar resultado
    if output_path:
        # Seleccionar columnas para output
        output_columns = [
            url_column,
            "title" if "title" in df.columns else None,
            "clean_text",
            "clean_word_count",
            "template_group",
        ]
        output_columns = [c for c in output_columns if c and c in df.columns]

        # Añadir columnas adicionales si existen
        for col in ["meta_description", "markdown", "crawl_date"]:
            if col in df.columns:
                output_columns.append(col)

        df_output = df[output_columns]
        df_output.to_parquet(output_path, engine="pyarrow", compression="snappy")
        print(f"\n  Resultado guardado en: {output_path}")

    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart Cleaner - Limpieza inteligente de HTML para RAG"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Ruta al archivo parquet de entrada"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Ruta al archivo parquet de salida (default: input_clean.parquet)"
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="No usar IA para descubrir selectores (solo heurísticas)"
    )
    parser.add_argument(
        "--html-column",
        default="html_content",
        help="Nombre de la columna con HTML (default: html_content)"
    )
    parser.add_argument(
        "--url-column",
        default="url",
        help="Nombre de la columna con URL (default: url)"
    )
    parser.add_argument(
        "--rules-file",
        default=None,
        help="Cargar reglas desde archivo JSON en lugar de descubrirlas"
    )

    args = parser.parse_args()

    # Determinar output path
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_clean.parquet")

    # Ejecutar pipeline
    asyncio.run(smart_clean(
        input_path=args.input,
        output_path=args.output,
        use_ai=not args.no_ai,
        html_column=args.html_column,
        url_column=args.url_column,
    ))


if __name__ == "__main__":
    main()
