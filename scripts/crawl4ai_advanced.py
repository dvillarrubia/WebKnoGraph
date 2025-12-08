#!/usr/bin/env python3
"""
Advanced Crawler using Crawl4AI for WebKnoGraph.
Features: Sitemap discovery, Link extraction, Resume, Content filtering, HTML capture.
"""

import asyncio
import argparse
import os
import sys
import re
import time
import json
import hashlib
import ssl
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
from glob import glob

import pandas as pd
from tqdm import tqdm
import aiohttp

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
except ImportError:
    print("Please install crawl4ai: pip install crawl4ai")
    sys.exit(1)


def update_status(status_path: Path, **kwargs):
    """Update crawl status file."""
    status = {}
    if status_path.exists():
        try:
            with open(status_path) as f:
                status = json.load(f)
        except:
            pass
    status.update(kwargs)
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)


def load_visited_urls(output_path: Path) -> set:
    """Load previously visited URLs."""
    visited = set()
    pages_files = glob(str(output_path / "pages" / "**" / "*.parquet"), recursive=True)
    for pf in pages_files:
        try:
            df = pd.read_parquet(pf, columns=["url"])
            visited.update(df["url"].tolist())
        except:
            continue
    return visited


def load_discovered_urls(output_path: Path, base_domain: str) -> set:
    """Load discovered URLs from links."""
    discovered = set()
    links_files = glob(str(output_path / "links" / "**" / "*.parquet"), recursive=True)
    for lf in links_files:
        try:
            df = pd.read_parquet(lf, columns=["target_url"])
            for url in df["target_url"].tolist():
                if base_domain in url:
                    discovered.add(url)
        except:
            continue
    return discovered


def load_urls_from_file(file_path: str) -> list:
    """Load URLs from file."""
    urls = []
    try:
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
    except:
        pass
    return urls


async def discover_urls_from_sitemap(domain: str) -> list:
    """Discover URLs from sitemap."""
    urls = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xml;q=0.9,*/*;q=0.8',
    }
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    sitemap_urls = [
        f"https://{domain}/sitemap.xml",
        f"https://{domain}/sitemap_index.xml",
        f"https://{domain}/sitemaps/sitemap-ilerna.xml",
        f"https://{domain}/blog/sitemap_index.xml",
    ]

    # Get sitemaps from robots.txt
    try:
        conn = aiohttp.TCPConnector(ssl=ssl_ctx)
        async with aiohttp.ClientSession(connector=conn, headers=headers) as session:
            async with session.get(f"https://{domain}/robots.txt", timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    for line in (await resp.text()).split('\n'):
                        if line.lower().startswith('sitemap:'):
                            sm = line.split(':', 1)[1].strip()
                            if sm not in sitemap_urls:
                                sitemap_urls.append(sm)
                                print(f"  Found in robots.txt: {sm}")
    except Exception as e:
        print(f"  Warning: robots.txt error: {e}")

    # Fetch sitemaps
    conn = aiohttp.TCPConnector(ssl=ssl_ctx)
    async with aiohttp.ClientSession(connector=conn, headers=headers) as session:
        for sm_url in sitemap_urls:
            try:
                async with session.get(sm_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        content = await resp.text()
                        if '<sitemapindex' in content:
                            children = re.findall(r'<loc>([^<]+\.xml[^<]*)</loc>', content)
                            print(f"  Sitemap index: {len(children)} children in {sm_url}")
                            for child in children:
                                try:
                                    async with session.get(child, timeout=aiohttp.ClientTimeout(total=30)) as cr:
                                        if cr.status == 200:
                                            locs = re.findall(r'<loc>([^<]+)</loc>', await cr.text())
                                            urls.extend(locs)
                                            print(f"    {len(locs)} URLs from {child}")
                                except:
                                    pass
                        else:
                            locs = re.findall(r'<loc>([^<]+)</loc>', content)
                            urls.extend(locs)
                            print(f"  {len(locs)} URLs from {sm_url}")
            except Exception as e:
                print(f"  Warning: {sm_url}: {e}")

    urls = list(set(urls))
    urls = [u for u in urls if domain in u]
    urls = [u for u in urls if not any(x in u.lower() for x in ['.jpg', '.png', '.gif', '.pdf', '.xml'])]
    return urls


# Selectores comunes para banners de cookies
# El usuario puede añadir más con --exclude-selectors
DEFAULT_COOKIE_SELECTORS = [
    # Cookiebot (como ILERNA)
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
    # Generic patterns
    "[id*='cookie-consent']",
    "[id*='cookieconsent']",
    "[id*='cookie-notice']",
    "[id*='cookie-banner']",
    "[class*='cookie-consent']",
    "[class*='cookie-banner']",
    "[class*='cookie-notice']",
    # GDPR popups
    "[id*='gdpr']",
    "[class*='gdpr']",
]


async def crawl_site_advanced(
    start_url: str,
    max_pages: int = 0,
    output_dir: str = "./data/crawl4ai_data",
    use_sitemap: bool = True,
    content_filter: bool = True,
    delay: float = 0.5,
    headless: bool = True,
    resume: bool = False,
    force_sitemap: bool = False,
    urls_file: str = None,
    urls_only: bool = False,
    exclude_selectors: list = None,  # Selectores CSS adicionales a excluir
):
    """Advanced crawl with sitemap and link extraction."""
    base_domain = urlparse(start_url).netloc
    output_path = Path(output_dir) / base_domain.replace(".", "_")
    output_path.mkdir(parents=True, exist_ok=True)
    status_path = Path(output_dir) / ".crawl_status.json"

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           Crawl4AI Advanced - WebKnoGraph RAG Crawler            ║
╠══════════════════════════════════════════════════════════════════╣
║  URL:           {start_url:<50} ║
║  Domain:        {base_domain:<50} ║
║  Max Pages:     {str(max_pages) if max_pages > 0 else 'Unlimited':<50} ║
║  Use Sitemap:   {str(use_sitemap):<50} ║
║  Resume:        {str(resume):<50} ║
║  Output:        {str(output_path):<50} ║
╚══════════════════════════════════════════════════════════════════╝
""")

    visited = set()
    pages_prev = 0
    urls_from_file = load_urls_from_file(urls_file) if urls_file else []

    if resume:
        visited = load_visited_urls(output_path)
        pages_prev = len(visited)
        print(f"Resume: {pages_prev} previously crawled")

    sitemap_urls = []
    if use_sitemap and (not resume or force_sitemap):
        print("\n[1/4] Discovering URLs from sitemap...")
        sitemap_urls = await discover_urls_from_sitemap(base_domain)
        print(f"  Total: {len(sitemap_urls)} URLs")

    discovered = load_discovered_urls(output_path, base_domain) - visited if resume else set()
    to_visit = list(dict.fromkeys(urls_from_file + list(discovered) + [start_url] + sitemap_urls))

    print(f"\n[2/4] Starting crawl... ({len(to_visit)} URLs)")

    update_status(status_path, status="running", pages_crawled=0, links_found=0, errors=0)

    browser_cfg = BrowserConfig(headless=headless, verbose=False)

    # Usar PruningContentFilter de Crawl4AI para limpieza automática
    # Este filtro usa densidad de texto y patrones para eliminar:
    # - nav, footer, header, aside, script, style, form, iframe, noscript
    # - Elementos con patrones: nav|footer|header|sidebar|ads|comment|promo|advert|social|share
    content_filter = PruningContentFilter(
        threshold=0.4,  # Umbral de densidad de texto (0-1, mayor = más estricto)
        threshold_type="dynamic",  # dynamic o fixed
        min_word_threshold=30,  # Mínimo de palabras por bloque
    )

    markdown_generator = DefaultMarkdownGenerator(
        content_filter=content_filter,
    )

    # Combinar selectores por defecto con los del usuario
    all_selectors = DEFAULT_COOKIE_SELECTORS.copy()
    if exclude_selectors:
        all_selectors.extend(exclude_selectors)
    excluded_selector = ",".join(all_selectors)

    print(f"  Excluding {len(all_selectors)} CSS selectors (cookies, overlays)")

    crawler_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_until="networkidle",
        page_timeout=30000,
        markdown_generator=markdown_generator,
        # Excluir elementos de cookies y overlays
        excluded_selector=excluded_selector,
        remove_overlay_elements=True,
    )

    results, links_graph, errors = [], [], 0

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        pbar = tqdm(total=max_pages if max_pages > 0 else len(to_visit), desc="Crawling")

        while to_visit and (max_pages == 0 or len(visited) - pages_prev < max_pages):
            url = to_visit.pop(0)
            if url in visited or urlparse(url).netloc != base_domain:
                continue

            visited.add(url)
            update_status(status_path, current_url=url, queue_size=len(to_visit))

            try:
                result = await crawler.arun(url=url, config=crawler_cfg)
                if result.success:
                    title = result.metadata.get("title", "") if result.metadata else ""
                    meta = result.metadata.get("description", "") if result.metadata else ""

                    # result.markdown es ahora MarkdownGenerationResult
                    md_result = result.markdown
                    # raw_markdown: markdown completo (sin filtros agresivos)
                    # fit_markdown: filtrado por PruningContentFilter (puede ser demasiado agresivo)
                    if hasattr(md_result, 'raw_markdown'):
                        # Usar raw_markdown como principal - el PruningContentFilter puede ser muy agresivo
                        markdown = md_result.raw_markdown or ""
                        markdown_fit = md_result.fit_markdown or ""
                    else:
                        markdown = str(md_result) if md_result else ""
                        markdown_fit = markdown

                    html = result.html or ""

                    if content_filter and len(markdown.split()) < 50:
                        continue

                    internal_links = result.links.get("internal", []) if result.links else []

                    if not urls_only:
                        for link in internal_links:
                            href = link.get("href", "")
                            if href and base_domain in href:
                                links_graph.append({
                                    "source_url": url,
                                    "target_url": href,
                                    "anchor_text": link.get("text", "")[:500],
                                    "crawl_date": datetime.now().strftime("%Y-%m-%d"),
                                })
                                if href not in visited and href not in to_visit:
                                    to_visit.append(href)

                    results.append({
                        "url": url,
                        "title": title[:500],
                        "meta_description": meta[:1000],
                        "markdown": markdown,  # raw_markdown (completo, sin cookies)
                        "markdown_fit": markdown_fit,  # fit_markdown (puede ser demasiado agresivo)
                        "html_content": html,
                        "content_hash": hashlib.sha256(markdown.encode()).hexdigest(),
                        "word_count": len(markdown.split()),
                        "links_count": len(internal_links),
                        "crawl_date": datetime.now().strftime("%Y-%m-%d"),
                    })

                    update_status(status_path, pages_crawled=len(visited)-pages_prev, links_found=len(links_graph))
                else:
                    errors += 1
                    update_status(status_path, errors=errors)
            except Exception as e:
                errors += 1
                update_status(status_path, errors=errors)

            pbar.update(1)
            pbar.set_postfix({"queue": len(to_visit), "links": len(links_graph)})

            if len(results) >= 25:
                save_results(results, links_graph, output_path)
                results, links_graph = [], []

            await asyncio.sleep(delay)
        pbar.close()

    if results or links_graph:
        save_results(results, links_graph, output_path)

    update_status(status_path, status="completed", completed_at=datetime.now().isoformat(),
                  pages_crawled=len(visited)-pages_prev, current_url=None)

    print(f"\n{'='*50}\nCOMPLETE: {len(visited)-pages_prev} pages, {errors} errors\n{'='*50}")


def save_results(results: list, links: list, output_path: Path):
    """Save to Parquet."""
    today = datetime.now().strftime("%Y-%m-%d")
    ts = int(time.time())

    if results:
        p = output_path / "pages" / f"crawl_date={today}"
        p.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_parquet(p / f"pages_{ts}.parquet", engine="pyarrow", compression="snappy")
        print(f"  Saved {len(results)} pages")

    if links:
        p = output_path / "links" / f"crawl_date={today}"
        p.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(links).to_parquet(p / f"links_{ts}.parquet", engine="pyarrow", compression="snappy")
        print(f"  Saved {len(links)} links")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Crawler using Crawl4AI with cookie/overlay exclusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl básico
  python crawl4ai_advanced.py --url https://example.com --max-pages 100

  # Con selectores adicionales para excluir
  python crawl4ai_advanced.py --url https://example.com --exclude-selectors "#custom-popup,.my-modal"

  # Ver selectores de cookies por defecto
  python crawl4ai_advanced.py --list-selectors
"""
    )
    parser.add_argument("--url", help="URL inicial para crawlear")
    parser.add_argument("--max-pages", type=int, default=0, help="Máximo de páginas (0=ilimitado)")
    parser.add_argument("--output-dir", default="./data/crawl4ai_data")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay entre requests")
    parser.add_argument("--no-headless", action="store_true", help="Mostrar navegador")
    parser.add_argument("--no-sitemap", action="store_true", help="No usar sitemap")
    parser.add_argument("--no-filter", action="store_true", help="No filtrar contenido corto")
    parser.add_argument("--resume", action="store_true", help="Continuar crawl anterior")
    parser.add_argument("--force-sitemap", action="store_true")
    parser.add_argument("--urls-file", help="Archivo con URLs a crawlear")
    parser.add_argument("--urls-only", action="store_true")
    parser.add_argument("--exclude-selectors",
                        help="Selectores CSS adicionales a excluir (separados por coma)")
    parser.add_argument("--list-selectors", action="store_true",
                        help="Mostrar selectores de cookies por defecto")
    args = parser.parse_args()

    if args.list_selectors:
        print("Selectores de cookies por defecto:")
        for selector in DEFAULT_COOKIE_SELECTORS:
            print(f"  {selector}")
        print(f"\nTotal: {len(DEFAULT_COOKIE_SELECTORS)} selectores")
        print("\nPuedes añadir más con --exclude-selectors '#mi-popup,.mi-clase'")
        return

    if not args.url:
        parser.error("--url es requerido (o usa --list-selectors)")

    # Parsear selectores adicionales
    extra_selectors = []
    if args.exclude_selectors:
        extra_selectors = [s.strip() for s in args.exclude_selectors.split(",") if s.strip()]

    asyncio.run(crawl_site_advanced(
        start_url=args.url,
        max_pages=args.max_pages,
        output_dir=args.output_dir,
        delay=args.delay,
        headless=not args.no_headless,
        use_sitemap=not args.no_sitemap,
        content_filter=not args.no_filter,
        resume=args.resume,
        force_sitemap=args.force_sitemap,
        urls_file=args.urls_file,
        urls_only=args.urls_only,
        exclude_selectors=extra_selectors if extra_selectors else None,
    ))


if __name__ == "__main__":
    main()
