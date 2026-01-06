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
import signal
import threading
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from datetime import datetime
from glob import glob

# Global para guardar datos pendientes en caso de interrupción
_pending_results = []
_pending_links = []
_output_path = None
_shutdown_requested = False
_last_activity = 0  # Timestamp de última actividad para watchdog
_watchdog_timeout = 120  # Segundos sin actividad antes de intervenir
_browser_pid = None  # PID del proceso del browser para kill forzado

import pandas as pd
from tqdm import tqdm
import aiohttp

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
except ImportError:
    print("Please install crawl4ai: pip install crawl4ai")
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Please install beautifulsoup4: pip install beautifulsoup4")
    sys.exit(1)


def _update_activity():
    """Actualiza timestamp de última actividad."""
    global _last_activity
    _last_activity = time.time()


def _kill_chrome_processes():
    """Mata procesos de Chrome/Chromium que puedan estar colgados."""
    try:
        # Buscar procesos de Chrome de Playwright
        result = subprocess.run(
            ["pgrep", "-f", "chromium|chrome.*--headless"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split('\n')
        killed = 0
        for pid in pids:
            if pid:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    killed += 1
                except:
                    pass
        return killed
    except:
        return 0


def _watchdog_thread(stop_event: threading.Event):
    """Thread watchdog que detecta cuelgues y mata el browser."""
    global _last_activity, _shutdown_requested

    while not stop_event.is_set():
        time.sleep(10)  # Check cada 10 segundos

        if _shutdown_requested:
            break

        if _last_activity > 0:
            inactive_time = time.time() - _last_activity
            if inactive_time > _watchdog_timeout:
                print(f"\n⚠️  WATCHDOG: {inactive_time:.0f}s sin actividad - matando browser...")

                # Guardar datos pendientes
                _emergency_save()

                # Matar procesos de Chrome
                killed = _kill_chrome_processes()
                print(f"  Procesos Chrome eliminados: {killed}")

                # Resetear actividad para no triggear continuamente
                _last_activity = time.time()


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


async def load_robots_txt(domain: str, user_agent: str = "*") -> RobotFileParser:
    """Load and parse robots.txt for a domain."""
    rp = RobotFileParser()
    robots_url = f"https://{domain}/robots.txt"

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    try:
        conn = aiohttp.TCPConnector(ssl=ssl_ctx)
        async with aiohttp.ClientSession(connector=conn) as session:
            async with session.get(robots_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    content = await resp.text()
                    # RobotFileParser needs to be fed line by line
                    rp.parse(content.split('\n'))
                    print(f"  Loaded robots.txt from {domain}")
                else:
                    print(f"  No robots.txt found (status {resp.status})")
    except Exception as e:
        print(f"  Warning: Could not load robots.txt: {e}")

    return rp


def is_url_allowed(rp: RobotFileParser, url: str, user_agent: str = "*") -> bool:
    """Check if URL is allowed by robots.txt."""
    if rp is None:
        return True
    try:
        return rp.can_fetch(user_agent, url)
    except:
        return True


def has_noindex(html: str) -> bool:
    """Check if HTML has noindex meta tag."""
    if not html:
        return False
    # Check for <meta name="robots" content="noindex...">
    # Also check for <meta name="googlebot" content="noindex...">
    patterns = [
        r'<meta[^>]+name=["\']robots["\'][^>]+content=["\'][^"\']*noindex[^"\']*["\']',
        r'<meta[^>]+content=["\'][^"\']*noindex[^"\']*["\'][^>]+name=["\']robots["\']',
        r'<meta[^>]+name=["\']googlebot["\'][^>]+content=["\'][^"\']*noindex[^"\']*["\']',
        r'<meta[^>]+content=["\'][^"\']*noindex[^"\']*["\'][^>]+name=["\']googlebot["\']',
    ]
    html_lower = html.lower()
    for pattern in patterns:
        if re.search(pattern, html_lower):
            return True
    return False


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


def extract_links_with_location(html: str, base_domain: str) -> list:
    """
    Extract links from HTML with their location context.

    Returns list of dicts with:
    - href: the link URL
    - text: anchor text
    - location: 'nav', 'footer', 'header', 'sidebar', 'content'
    - weight: suggested weight for PageRank (1.0 = content, 0.5 = nav, 0.3 = footer)
    """
    if not html:
        return []

    links = []
    seen_hrefs = set()

    try:
        soup = BeautifulSoup(html, "html.parser")

        # Define location selectors and their weights
        # Higher weight = more valuable for PageRank
        location_config = [
            # Footer links - lowest weight
            {
                "selectors": ["footer", "[role='contentinfo']", ".footer", "#footer",
                             "[class*='footer']", ".subfooter", ".endfooter"],
                "location": "footer",
                "weight": 0.3
            },
            # Header/Nav links - medium weight
            {
                "selectors": ["header", "nav", "[role='navigation']", "[role='banner']",
                             ".nav", ".navbar", ".menu", "#menu", ".header", "#header",
                             "[class*='menu']", "[class*='nav-']", ".breadcrumb",
                             "[class*='header']"],
                "location": "nav",
                "weight": 0.5
            },
            # Sidebar links - medium-low weight
            {
                "selectors": ["aside", "[role='complementary']", ".sidebar", "#sidebar",
                             "[class*='sidebar']", ".widget", "[class*='widget']"],
                "location": "sidebar",
                "weight": 0.4
            },
        ]

        # First pass: extract links from specific locations
        for config in location_config:
            for selector in config["selectors"]:
                try:
                    elements = soup.select(selector)
                    for element in elements:
                        for a in element.find_all("a", href=True):
                            href = a.get("href", "").strip()
                            if not href or href.startswith("#") or href.startswith("javascript:"):
                                continue

                            # Normalize relative URLs
                            if href.startswith("/"):
                                href = f"https://{base_domain}{href}"

                            # Only internal links
                            if base_domain not in href:
                                continue

                            # Avoid duplicates - first location wins
                            if href in seen_hrefs:
                                continue

                            seen_hrefs.add(href)
                            anchor_text = a.get_text(strip=True)[:500]

                            links.append({
                                "href": href,
                                "text": anchor_text,
                                "location": config["location"],
                                "weight": config["weight"]
                            })
                except Exception:
                    continue

        # Second pass: remaining links are "content" links (highest weight)
        for a in soup.find_all("a", href=True):
            href = a.get("href", "").strip()
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue

            # Normalize relative URLs
            if href.startswith("/"):
                href = f"https://{base_domain}{href}"

            # Only internal links
            if base_domain not in href:
                continue

            # Skip if already found in nav/footer/sidebar
            if href in seen_hrefs:
                continue

            seen_hrefs.add(href)
            anchor_text = a.get_text(strip=True)[:500]

            links.append({
                "href": href,
                "text": anchor_text,
                "location": "content",
                "weight": 1.0
            })

    except Exception as e:
        print(f"  Warning: Error extracting links: {e}")

    return links


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

# Presets de exclusiones por dominio
# Estos selectores se aplican SOLO al contenido/markdown, NO afectan la extracción de enlaces
DOMAIN_PRESETS = {
    "ilerna": {
        "name": "ILERNA.es",
        "description": "Selectores para limpiar contenido de ilerna.es",
        "selectors": [
            # Menú y navegación (para markdown, enlaces ya extraídos)
            "#header-mobile-tablet",
            "#header-desktop",
            "#menu-mobile-tablet",
            ".menu",
            ".main-menu",
            ".middle-menu",
            ".final-menu",
            ".sliding-menu",
            ".nav-right",
            ".blog-menu",
            ".blog-submenu",
            ".blog-supmenu",
            # Footer (para markdown, enlaces ya extraídos)
            ".footer",
            ".subfooter",
            ".endfooter",
            ".social",
            ".social-icons",
            ".logo-social",
            # Formularios
            "[id*='form']",
            ".form",
            ".bloque-form-homes",
            ".container-form",
            # Chat widgets
            "[id*='chatwith']",
            "[id*='zsiq']",
            ".chatwith",
            ".chatwithapp",
            ".chat-iframe-wrap",
            ".siqico-chat",
            # Modales/Popups
            ".modal__close",
            ".modal-response-message",
            ".modality-section",
            ".overlay",
            ".popupMode",
            ".showmodal",
            # Promociones y banners
            ".bloque-promo",
            ".cintillo-promo",
            ".promo-mobile",
            "[class*='banner']",
            ".post-conversion-banner",
            # CTAs
            "[class*='cta']",
            ".solicitar-info-float-btn",
            ".solicitar-info-mobile",
            # Blog específico
            ".breadcrumbs",
            ".block-share",
            ".share",
            ".block-author",
            ".author",
            ".block-related-courses",
            ".related-cycles",
            ".toc",
            ".navigation",
            ".date-post",
            ".data-info",
            # Reviews
            ".reviews",
            ".review",
        ]
    }
}


def get_preset_selectors(domain: str) -> list:
    """Get preset selectors for a domain if available."""
    domain_lower = domain.lower()
    for key, preset in DOMAIN_PRESETS.items():
        if key in domain_lower:
            print(f"  Using preset '{preset['name']}' with {len(preset['selectors'])} selectors")
            return preset["selectors"]
    return []


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
    respect_robots: bool = True,  # Respetar robots.txt
    skip_noindex: bool = True,  # Saltar páginas con noindex
    sitemap_only: bool = False,  # Solo crawlear URLs del sitemap (no seguir enlaces)
):
    """Advanced crawl with sitemap and link extraction."""
    global _output_path, _pending_results, _pending_links

    base_domain = urlparse(start_url).netloc
    output_path = Path(output_dir) / base_domain.replace(".", "_")
    output_path.mkdir(parents=True, exist_ok=True)
    status_path = Path(output_dir) / ".crawl_status.json"

    # Guardar referencia global para emergency save
    _output_path = output_path

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           Crawl4AI Advanced - WebKnoGraph RAG Crawler            ║
╠══════════════════════════════════════════════════════════════════╣
║  URL:           {start_url:<50} ║
║  Domain:        {base_domain:<50} ║
║  Max Pages:     {str(max_pages) if max_pages > 0 else 'Unlimited':<50} ║
║  Use Sitemap:   {str(use_sitemap):<50} ║
║  Resume:        {str(resume):<50} ║
║  Robots.txt:    {str(respect_robots):<50} ║
║  Skip noindex:  {str(skip_noindex):<50} ║
║  Sitemap only:  {str(sitemap_only):<50} ║
║  Output:        {str(output_path):<50} ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Load robots.txt if enabled
    robots_parser = None
    if respect_robots:
        print("\n[0/4] Loading robots.txt...")
        robots_parser = await load_robots_txt(base_domain)

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

    # Markdown generator SIN filtros - el contenido se guarda completo
    # La limpieza se hace en la fase de Cleaner, no aquí
    markdown_generator = DefaultMarkdownGenerator()

    # Combinar selectores: cookies por defecto + los del usuario
    # NOTA: Los presets de dominio (ilerna, etc.) ya NO se aplican automáticamente
    # El usuario debe configurar las exclusiones desde la interfaz del dashboard
    all_selectors = DEFAULT_COOKIE_SELECTORS.copy()

    # Añadir selectores del usuario (desde --exclude-selectors o desde la interfaz)
    if exclude_selectors:
        all_selectors.extend(exclude_selectors)

    excluded_selector = ",".join(all_selectors)

    custom_count = len(exclude_selectors) if exclude_selectors else 0
    print(f"  Excluding {len(all_selectors)} CSS selectors ({len(DEFAULT_COOKIE_SELECTORS)} cookies + {custom_count} custom)")

    crawler_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_until="load",  # Espera a que carguen recursos (no networkidle que se cuelga)
        page_timeout=60000,  # 60 segundos máximo
        delay_before_return_html=2.0,  # 2s extra para que JS renderice contenido
        markdown_generator=markdown_generator,
        # Excluir elementos de cookies y overlays
        excluded_selector=excluded_selector,
        remove_overlay_elements=True,
    )

    results, links_graph, errors = [], [], 0
    skipped_robots, skipped_noindex, skipped_short = 0, 0, 0
    start_time = time.time()

    # Log file for dashboard
    log_file = Path(output_dir) / ".crawl_log.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Clear previous log
    with open(log_file, "w") as f:
        pass

    def log(msg, level="INFO"):
        """Print timestamped log message and save to file."""
        ts = datetime.now().strftime("%H:%M:%S")
        prefix = {"INFO": "ℹ️", "OK": "✅", "WARN": "⚠️", "ERROR": "❌", "SKIP": "⏭️", "SAVE": "💾", "LINK": "🔗"}.get(level, "•")
        log_line = f"[{ts}] {prefix} {msg}"
        print(log_line)
        # Save to file for dashboard
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps({
                    "ts": ts,
                    "level": level,
                    "msg": msg,
                    "formatted": log_line
                }) + "\n")
        except:
            pass

    log(f"Iniciando crawl de {base_domain}", "INFO")
    log(f"URLs en cola: {len(to_visit)} | Max páginas: {max_pages if max_pages > 0 else 'ilimitado'}", "INFO")

    # Iniciar watchdog thread para detectar cuelgues
    watchdog_stop = threading.Event()
    watchdog = threading.Thread(target=_watchdog_thread, args=(watchdog_stop,), daemon=True)
    watchdog.start()
    _update_activity()  # Marcar inicio

    # Variables para auto-recovery
    max_browser_restarts = 5
    browser_restarts = 0
    page_num = 0
    consecutive_errors = 0
    max_consecutive_errors = 10  # Reiniciar browser después de 10 errores seguidos

    # Loop principal con auto-recovery
    while to_visit and (max_pages == 0 or len(visited) - pages_prev < max_pages):
        if browser_restarts >= max_browser_restarts:
            log(f"Máximo de reinicios de browser alcanzado ({max_browser_restarts}), abortando", "ERROR")
            break

        try:
            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                log(f"Browser iniciado (reinicio #{browser_restarts})" if browser_restarts > 0 else "Browser iniciado", "INFO")
                pbar = tqdm(total=max_pages if max_pages > 0 else len(to_visit), desc="Crawling", initial=page_num)

                while to_visit and (max_pages == 0 or len(visited) - pages_prev < max_pages):
                    # Check si debemos reiniciar por errores consecutivos
                    if consecutive_errors >= max_consecutive_errors:
                        log(f"Demasiados errores consecutivos ({consecutive_errors}), reiniciando browser...", "WARN")
                        consecutive_errors = 0
                        break  # Sale del while interno, reinicia browser

                    url = to_visit.pop(0)

                    # Skip already visited or external
                    if url in visited:
                        continue
                    if urlparse(url).netloc != base_domain:
                        log(f"SKIP externo: {url[:60]}...", "SKIP")
                        continue

                    # Check robots.txt before crawling
                    if respect_robots and not is_url_allowed(robots_parser, url):
                        skipped_robots += 1
                        visited.add(url)
                        log(f"SKIP robots.txt: {url[:60]}...", "SKIP")
                        continue

                    visited.add(url)
                    page_num += 1
                    update_status(status_path, current_url=url, queue_size=len(to_visit))

                    # Log de inicio de página
                    url_short = url.replace(f"https://{base_domain}", "")[:50]
                    log(f"[{page_num}] Procesando: {url_short}...", "INFO")

                    try:
                        fetch_start = time.time()
                        # Timeout explícito para evitar cuelgues indefinidos
                        try:
                            result = await asyncio.wait_for(
                                crawler.arun(url=url, config=crawler_cfg),
                                timeout=90.0  # 90 segundos máximo absoluto
                            )
                        except asyncio.TimeoutError:
                            errors += 1
                            consecutive_errors += 1
                            log(f"  TIMEOUT: página tardó más de 90s", "ERROR")
                            update_status(status_path, errors=errors)
                            continue
                        fetch_time = time.time() - fetch_start
                        _update_activity()  # Marcar actividad para watchdog
                        consecutive_errors = 0  # Reset en éxito

                        if result.success:
                            html = result.html or ""
                            html_size = len(html)

                            # Check for noindex meta tag
                            if skip_noindex and has_noindex(html):
                                skipped_noindex += 1
                                log(f"  SKIP noindex detectado", "SKIP")
                                continue

                            title = result.metadata.get("title", "") if result.metadata else ""
                            meta = result.metadata.get("description", "") if result.metadata else ""

                            # result.markdown es ahora MarkdownGenerationResult
                            md_result = result.markdown
                            if hasattr(md_result, 'raw_markdown'):
                                markdown_raw = md_result.raw_markdown or ""
                                markdown_fit = md_result.fit_markdown or ""
                                markdown = markdown_fit if markdown_fit else markdown_raw
                            else:
                                markdown = str(md_result) if md_result else ""
                                markdown_raw = markdown
                                markdown_fit = markdown

                            word_count = len(markdown.split())

                            # Skip páginas con muy poco contenido (configurable con --no-filter)
                            if content_filter and word_count < 50:
                                skipped_short += 1
                                log(f"  SKIP contenido corto ({word_count} palabras)", "SKIP")
                                continue

                            # Extract links WITH location context
                            located_links = extract_links_with_location(html, base_domain)

                            # Count links by location
                            links_by_loc = {}
                            for link in located_links:
                                loc = link.get("location", "unknown")
                                links_by_loc[loc] = links_by_loc.get(loc, 0) + 1

                            new_urls_found = 0
                            if not urls_only and not sitemap_only:
                                for link in located_links:
                                    href = link.get("href", "")
                                    if href and base_domain in href:
                                        links_graph.append({
                                            "source_url": url,
                                            "target_url": href,
                                            "anchor_text": link.get("text", "")[:500],
                                            "link_location": link.get("location", "content"),
                                            "link_weight": link.get("weight", 1.0),
                                            "crawl_date": datetime.now().strftime("%Y-%m-%d"),
                                        })
                                        if href not in visited and href not in to_visit:
                                            to_visit.append(href)
                                            new_urls_found += 1

                            # Guardar markdown SIN procesar - la limpieza se hace en el Cleaner
                            results.append({
                                "url": url,
                                "title": title[:500],
                                "meta_description": meta[:1000],
                                "markdown": markdown_raw,  # Contenido completo sin filtrar
                                "html_content": html,
                                "content_hash": hashlib.sha256(markdown_raw.encode()).hexdigest(),
                                "word_count": word_count,
                                "links_count": len(located_links),
                                "crawl_date": datetime.now().strftime("%Y-%m-%d"),
                            })

                            # Log detallado de éxito
                            links_summary = " | ".join([f"{k}:{v}" for k, v in sorted(links_by_loc.items())])
                            log(f"  OK en {fetch_time:.1f}s | HTML:{html_size//1024}KB | {word_count} palabras | Enlaces: {links_summary}", "OK")
                            if new_urls_found > 0:
                                log(f"  +{new_urls_found} URLs nuevas añadidas a cola (total: {len(to_visit)})", "LINK")

                            update_status(status_path, pages_crawled=len(visited)-pages_prev, links_found=len(links_graph))

                            # Actualizar globals para emergency save
                            _pending_results = results.copy()
                            _pending_links = links_graph.copy()
                        else:
                            errors += 1
                            consecutive_errors += 1
                            error_msg = getattr(result, 'error_message', 'Unknown error')
                            log(f"  ERROR: {error_msg}", "ERROR")
                            update_status(status_path, errors=errors)

                    except Exception as e:
                        errors += 1
                        consecutive_errors += 1
                        log(f"  EXCEPCIÓN: {str(e)[:100]}", "ERROR")
                        update_status(status_path, errors=errors)

                    pbar.update(1)
                    pbar.set_postfix({
                        "cola": len(to_visit),
                        "links": len(links_graph),
                        "ok": len(results),
                        "err": errors
                    })

                    # Save batch - más frecuente para evitar pérdida de datos
                    if len(results) >= 10:
                        log(f"Guardando batch: {len(results)} páginas, {len(links_graph)} enlaces", "SAVE")
                        save_results(results, links_graph, output_path)
                        results, links_graph = [], []
                        _pending_results, _pending_links = [], []

                    await asyncio.sleep(delay)

                pbar.close()

        except Exception as browser_error:
            # Browser crashed - save data and restart
            browser_restarts += 1
            log(f"🔄 BROWSER CRASH: {str(browser_error)[:80]}", "ERROR")
            log(f"  Guardando datos y reiniciando browser (intento {browser_restarts}/{max_browser_restarts})...", "WARN")

            # Save pending data
            if results or links_graph:
                save_results(results, links_graph, output_path)
                results, links_graph = [], []
                _pending_results, _pending_links = [], []

            # Kill any zombie Chrome processes
            _kill_chrome_processes()

            # Wait before retry
            await asyncio.sleep(5)
            continue

    # Detener watchdog
    watchdog_stop.set()
    watchdog.join(timeout=2)

    # Final save
    if results or links_graph:
        log(f"Guardando últimos resultados: {len(results)} páginas, {len(links_graph)} enlaces", "SAVE")
        save_results(results, links_graph, output_path)
        _pending_results, _pending_links = [], []  # Limpiar globals

    elapsed = time.time() - start_time
    elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s"

    update_status(status_path, status="completed", completed_at=datetime.now().isoformat(),
                  pages_crawled=len(visited)-pages_prev, current_url=None,
                  skipped_robots=skipped_robots, skipped_noindex=skipped_noindex)

    print(f"""
{'='*70}
✅ CRAWL COMPLETADO en {elapsed_str}
{'='*70}
  📄 Páginas crawleadas: {len(visited)-pages_prev}
  🔗 Enlaces extraídos:  {len(links_graph)}
  ❌ Errores:            {errors}
  ⏭️  Saltadas:
     - Por robots.txt:   {skipped_robots}
     - Por noindex:      {skipped_noindex}
     - Por contenido corto: {skipped_short}
  📁 Output: {output_path}
{'='*70}""")


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


def _emergency_save():
    """Guardar datos pendientes en caso de interrupción."""
    global _pending_results, _pending_links, _output_path
    if _output_path and (_pending_results or _pending_links):
        print(f"\n⚠️  GUARDANDO DATOS PENDIENTES: {len(_pending_results)} páginas, {len(_pending_links)} enlaces...")
        save_results(_pending_results, _pending_links, _output_path)
        _pending_results, _pending_links = [], []


def _signal_handler(signum, frame):
    """Handler para SIGTERM/SIGINT - guarda datos y sale."""
    global _shutdown_requested
    sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    print(f"\n\n🛑 Recibida señal {sig_name} - guardando datos antes de salir...")
    _shutdown_requested = True
    _emergency_save()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Crawler using Crawl4AI with robots.txt and noindex support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl básico (respeta robots.txt, salta noindex)
  python crawl4ai_advanced.py --url https://example.com --max-pages 100

  # Solo URLs del sitemap (no sigue enlaces)
  python crawl4ai_advanced.py --url https://example.com --sitemap-only

  # Ignorar robots.txt (no recomendado)
  python crawl4ai_advanced.py --url https://example.com --no-robots

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
    parser.add_argument("--list-presets", action="store_true",
                        help="Mostrar presets de exclusiones por dominio")
    parser.add_argument("--preset", type=str,
                        help="Usar preset específico (ej: ilerna)")
    # New options
    parser.add_argument("--no-robots", action="store_true",
                        help="Ignorar robots.txt (no recomendado)")
    parser.add_argument("--no-skip-noindex", action="store_true",
                        help="No saltar páginas con noindex")
    parser.add_argument("--sitemap-only", action="store_true",
                        help="Solo crawlear URLs del sitemap (no seguir enlaces)")
    args = parser.parse_args()

    if args.list_selectors:
        print("Selectores de cookies por defecto:")
        for selector in DEFAULT_COOKIE_SELECTORS:
            print(f"  {selector}")
        print(f"\nTotal: {len(DEFAULT_COOKIE_SELECTORS)} selectores")
        print("\nPuedes añadir más con --exclude-selectors '#mi-popup,.mi-clase'")
        return

    if args.list_presets:
        print("Presets de exclusiones por dominio:")
        print("=" * 60)
        for key, preset in DOMAIN_PRESETS.items():
            print(f"\n[{key}] {preset['name']}")
            print(f"  {preset['description']}")
            print(f"  Selectores ({len(preset['selectors'])}):")
            for selector in preset['selectors'][:10]:
                print(f"    {selector}")
            if len(preset['selectors']) > 10:
                print(f"    ... y {len(preset['selectors']) - 10} más")
        print("\n" + "=" * 60)
        print("Los presets se aplican automáticamente al detectar el dominio.")
        print("También puedes forzar uno con --preset <nombre>")
        return

    if not args.url:
        parser.error("--url es requerido (o usa --list-selectors)")

    # Parsear selectores adicionales
    extra_selectors = []
    if args.exclude_selectors:
        extra_selectors = [s.strip() for s in args.exclude_selectors.split(",") if s.strip()]

    # Añadir selectores de preset forzado
    if args.preset:
        preset_key = args.preset.lower()
        if preset_key in DOMAIN_PRESETS:
            print(f"Using forced preset: {DOMAIN_PRESETS[preset_key]['name']}")
            extra_selectors.extend(DOMAIN_PRESETS[preset_key]['selectors'])
        else:
            print(f"Warning: Preset '{args.preset}' not found. Available: {', '.join(DOMAIN_PRESETS.keys())}")

    # Registrar signal handlers para guardado de emergencia
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

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
        respect_robots=not args.no_robots,
        skip_noindex=not args.no_skip_noindex,
        sitemap_only=args.sitemap_only,
    ))


if __name__ == "__main__":
    main()
