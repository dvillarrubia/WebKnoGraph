#!/usr/bin/env python3
"""
Algoritmo de extracción de contenido principal basado en headings.

La idea: El contenido SEO importante siempre está estructurado con headings (H1-H6).
Los menús, footers y sidebars raramente tienen headings estructurados.

Algoritmo:
1. Encontrar el H1 (título principal)
2. Identificar el contenedor que tiene el H1
3. Extraer todo el contenido estructurado bajo headings
4. Filtrar bloques sin headings (navegación, footers)
"""

from bs4 import BeautifulSoup, NavigableString
from typing import List, Dict, Tuple, Optional
import re


def extract_main_content(html: str, url: str = "") -> Tuple[str, str, Dict]:
    """
    Extrae el contenido principal de una página HTML basándose en la estructura de headings.

    Returns:
        Tuple de (markdown, título, estadísticas)

    El algoritmo:
    1. Encuentra el H1 que tiene más H2/H3 estructurados después (= contenido real)
    2. Extrae el contenedor que contiene ese H1
    3. Convierte a markdown preservando la estructura de headings
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Eliminar elementos que nunca son contenido
    for tag in soup.find_all(['script', 'style', 'noscript', 'iframe', 'svg']):
        tag.decompose()

    # Eliminar elementos de cookies conocidos
    for selector in ['#CybotCookiebotDialog', '[id*="cookie"]', '[class*="cookie"]']:
        for el in soup.select(selector):
            el.decompose()

    # Encontrar todos los headings
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

    if not headings:
        # Sin headings, usar fallback
        return _fallback_extraction(soup)

    # Encontrar el H1 principal (el primero que no está en nav/header/footer)
    h1 = _find_main_h1(soup, headings)

    if not h1:
        # Sin H1 válido, usar el primer heading
        h1 = headings[0]

    # Encontrar el contenedor de contenido principal
    content_container = _find_content_container(h1, soup)

    # Extraer contenido estructurado
    markdown, stats = _extract_structured_content(content_container, h1)

    # Obtener título
    title = h1.get_text().strip() if h1 else ""

    return markdown, title, stats


def _find_main_h1(soup: BeautifulSoup, headings: List) -> Optional:
    """
    Encuentra el H1 principal del contenido.

    Estrategia universal: El H1 del contenido tiene H2/H3 DESPUÉS de él.
    El H1 de navegación/header NO tiene headings estructurados después.
    """
    h1_candidates = []

    for i, h in enumerate(headings):
        if h.name != 'h1':
            continue

        # Verificar que no esté en elementos de navegación
        parent_tags = [p.name for p in h.parents if p.name]
        if any(tag in parent_tags for tag in ['nav', 'footer']):
            continue

        # Contar cuántos H2/H3 hay DESPUÉS de este H1
        subsequent_headings = 0
        for j in range(i + 1, len(headings)):
            next_h = headings[j]
            if next_h.name == 'h1':
                # Llegamos a otro H1, parar
                break
            if next_h.name in ['h2', 'h3']:
                subsequent_headings += 1

        h1_candidates.append((h, subsequent_headings))

    # Ordenar por número de headings después (más = mejor)
    h1_candidates.sort(key=lambda x: x[1], reverse=True)

    if h1_candidates:
        # El H1 con más H2/H3 después es el del contenido
        return h1_candidates[0][0]

    # Fallback a cualquier heading
    for h in headings:
        if h.name in ['h1', 'h2']:
            return h

    return None


def _find_content_container(h1, soup: BeautifulSoup):
    """
    Encuentra el contenedor que tiene el contenido principal.
    Busca el ancestro que contiene el H1 y tiene más headings/párrafos.
    """
    if not h1:
        return soup.body or soup

    best_container = None
    best_score = 0

    # Subir por los ancestros del H1
    for parent in h1.parents:
        if parent.name in ['html', '[document]']:
            break

        # Calcular score del contenedor
        score = _calculate_content_score(parent)

        # El mejor contenedor es el que tiene buen score pero no es demasiado grande
        if score > best_score:
            # Verificar que no sea el body entero
            if parent.name != 'body':
                best_score = score
                best_container = parent

    return best_container or (soup.body or soup)


def _calculate_content_score(element) -> float:
    """
    Calcula un score de "contenido" para un elemento.
    Alto score = probablemente contenido principal.
    """
    if not element:
        return 0

    text = element.get_text()
    words = len(text.split())

    # Contar headings dentro del elemento
    headings = element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    num_headings = len(headings)

    # Contar párrafos con contenido sustancial (>20 palabras)
    paragraphs = element.find_all('p')
    substantial_paragraphs = sum(1 for p in paragraphs if len(p.get_text().split()) > 20)

    # Contar enlaces (muchos enlaces = navegación)
    links = element.find_all('a')
    num_links = len(links)

    # Calcular ratio texto/enlaces
    link_text = sum(len(a.get_text().split()) for a in links)
    non_link_text = words - link_text

    # Score basado en headings y párrafos
    # Penalizar por muchos enlaces
    score = (num_headings * 100) + (substantial_paragraphs * 50) + non_link_text
    score -= (num_links * 5)  # Penalizar enlaces

    # Bonus si tiene H1
    if element.find('h1'):
        score += 200

    return max(0, score)


def _extract_structured_content(container, main_h1) -> Tuple[str, Dict]:
    """
    Extrae contenido estructurado siguiendo los headings.
    """
    if not container:
        return "", {}

    markdown_parts = []
    stats = {
        'headings': 0,
        'paragraphs': 0,
        'lists': 0,
        'words': 0,
    }

    # Procesar elementos en orden
    for element in container.descendants:
        if isinstance(element, NavigableString):
            continue

        # Headings
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            level = int(element.name[1])
            text = element.get_text().strip()
            if text:
                markdown_parts.append(f"{'#' * level} {text}")
                stats['headings'] += 1

        # Párrafos
        elif element.name == 'p':
            text = _clean_text(element.get_text())
            if text and len(text.split()) > 5:  # Ignorar párrafos muy cortos
                markdown_parts.append(text)
                stats['paragraphs'] += 1

        # Listas
        elif element.name in ['ul', 'ol']:
            list_items = []
            for li in element.find_all('li', recursive=False):
                item_text = _clean_text(li.get_text())
                if item_text:
                    list_items.append(f"- {item_text}")
            if list_items:
                markdown_parts.append('\n'.join(list_items))
                stats['lists'] += 1

    # Unir y limpiar
    markdown = '\n\n'.join(markdown_parts)
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)

    stats['words'] = len(markdown.split())

    return markdown.strip(), stats


def _clean_text(text: str) -> str:
    """Limpia texto eliminando espacios extra."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _fallback_extraction(soup: BeautifulSoup) -> Tuple[str, str, Dict]:
    """Extracción de fallback cuando no hay headings."""
    # Buscar el bloque con más texto
    body = soup.body or soup

    # Eliminar nav, header, footer
    for tag in body.find_all(['nav', 'header', 'footer', 'aside']):
        tag.decompose()

    text = body.get_text()
    text = re.sub(r'\s+', ' ', text).strip()

    stats = {
        'headings': 0,
        'paragraphs': 1,
        'lists': 0,
        'words': len(text.split()),
        'fallback': True,
    }

    return text, "", stats


def is_content_duplicate(content1: str, content2: str, threshold: float = 0.9) -> bool:
    """
    Detecta si dos contenidos son duplicados (>90% similares).
    Útil para detectar landings que son copias de la home.
    """
    words1 = set(content1.lower().split())
    words2 = set(content2.lower().split())

    if not words1 or not words2:
        return False

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    similarity = intersection / union if union > 0 else 0
    return similarity > threshold


# === PRUEBA ===
if __name__ == "__main__":
    import asyncio
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

    async def test():
        urls = [
            ("https://www.ilerna.es/blog/fp1-fp2-equivalencias", "blog"),
            ("https://www.ilerna.es", "home"),
            ("https://www.ilerna.es/es/tecnico-superior-desarrollo-aplicaciones-web-online-1001", "landing"),
        ]

        results = {}

        async with AsyncWebCrawler(verbose=False) as crawler:
            for url, page_type in urls:
                print(f"\n{'='*70}")
                print(f"[{page_type.upper()}] {url}")
                print(f"{'='*70}")

                cfg = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    wait_until="load",
                    page_timeout=60000,
                )
                result = await crawler.arun(url=url, config=cfg)

                if result.success:
                    markdown, title, stats = extract_main_content(result.html, url)
                    results[page_type] = markdown

                    print(f"\nTítulo: {title[:60]}")
                    print(f"Stats: {stats}")
                    print(f"\nPrimeras 8 líneas:")
                    for i, line in enumerate(markdown.split('\n')[:8]):
                        print(f"  [{i}] {line[:70]}")

        # Detectar duplicados
        print(f"\n{'='*70}")
        print("DETECCIÓN DE DUPLICADOS:")
        print(f"{'='*70}")
        if 'home' in results and 'landing' in results:
            is_dup = is_content_duplicate(results['home'], results['landing'])
            print(f"  landing vs home: {'DUPLICADO ⚠️' if is_dup else 'Contenido único ✅'}")
        if 'home' in results and 'blog' in results:
            is_dup = is_content_duplicate(results['home'], results['blog'])
            print(f"  blog vs home: {'DUPLICADO ⚠️' if is_dup else 'Contenido único ✅'}")

    asyncio.run(test())
