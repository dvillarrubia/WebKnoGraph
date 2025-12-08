"""
Cleaning Service for Graph-RAG.
Uses ScrapeGraphAI to discover CSS selectors, then extracts FULL content.
"""

import os
import re
import json
import asyncio
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Load .env file for OPENAI_API_KEY
from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from glob import glob

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# ScrapeGraphAI for intelligent selector discovery
from scrapegraphai.graphs import SmartScraperGraph, SmartScraperMultiGraph


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CleaningJob:
    """Represents a cleaning job."""
    id: str
    crawl_name: str
    status: str  # pending, running, analyzing, cleaning, completed, failed
    use_ai: bool
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_pages: int = 0
    pages_cleaned: int = 0
    groups_found: int = 0
    groups_analyzed: int = 0
    rules: Dict[str, Optional[str]] = field(default_factory=dict)
    error_message: Optional[str] = None
    output_path: Optional[str] = None


@dataclass
class CleaningResult:
    """Result of a cleaning operation."""
    crawl_name: str
    pages_cleaned: int
    groups_found: int
    rules_discovered: Dict[str, Optional[str]]
    avg_word_reduction: float
    output_path: str


# =============================================================================
# HTML TEMPLATE FINGERPRINTING
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
# SELECTOR DISCOVERY WITH SCRAPEGRAPHAI
# =============================================================================

def discover_content_selector(html: str) -> Optional[str]:
    """
    Use ScrapeGraphAI to analyze HTML and discover the best CSS selector
    for the main content. Returns a selector string or None.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    # First, find what selectors actually exist in the HTML
    soup = BeautifulSoup(html, "html.parser")

    # Build a list of candidate selectors with their word counts
    candidates = []

    # Check common content selectors
    common_selectors = [
        "main", "article", "#content", ".content",
        "#main", ".main", "#post", ".post",
        ".entry", ".article", ".post-content",
        "[role='main']", ".blog-post", ".page-content",
        ".entry-content", ".article-content", ".post-body",
    ]

    for sel in common_selectors:
        elements = soup.select(sel)
        for el in elements:
            text = el.get_text(strip=True)
            word_count = len(text.split())
            if word_count > 100:  # Only meaningful content
                classes = el.get("class", [])
                el_id = el.get("id", "")

                # Build the actual selector
                if el_id:
                    actual_sel = f"#{el_id}"
                elif classes:
                    actual_sel = f"{el.name}.{'.'.join(classes)}"
                else:
                    actual_sel = el.name

                candidates.append({
                    "selector": actual_sel,
                    "words": word_count,
                    "tag": el.name
                })

    # Also check divs with content-related classes
    for div in soup.find_all("div", class_=True):
        classes = div.get("class", [])
        class_str = " ".join(classes).lower()

        # Skip noise patterns
        if any(x in class_str for x in ["cookie", "gdpr", "consent", "popup", "modal", "banner", "social", "share", "nav", "menu", "footer", "header", "sidebar"]):
            continue

        # Look for content patterns
        if any(x in class_str for x in ["content", "article", "post", "entry", "blog", "main", "body", "text"]):
            text = div.get_text(strip=True)
            word_count = len(text.split())
            if word_count > 200:
                actual_sel = f"div.{'.'.join(classes)}"
                candidates.append({
                    "selector": actual_sel,
                    "words": word_count,
                    "tag": "div"
                })

    if not candidates:
        return None

    # Sort by word count descending, prefer article/main tags
    def score(c):
        base = c["words"]
        if c["tag"] in ["article", "main"]:
            base *= 1.5
        if "content" in c["selector"].lower():
            base *= 1.2
        return base

    candidates.sort(key=score, reverse=True)

    # If we have a clear winner (significantly more content), use it
    if len(candidates) >= 1:
        best = candidates[0]

        # Verify it's not just wrapping everything
        total_body_words = len(soup.body.get_text(strip=True).split()) if soup.body else 0
        if total_body_words > 0 and best["words"] / total_body_words > 0.95:
            # Too much - probably selecting too broadly, try second best
            if len(candidates) > 1:
                return candidates[1]["selector"]

        return best["selector"]

    return None


def discover_selectors_with_ai(html: str) -> dict:
    """
    Use ScrapeGraphAI to analyze HTML and discover the best CSS selectors
    for extracting main content. Returns selectors that can be reused.

    This is the KEY function for the optimized approach:
    - Analyze 1 sample page per template
    - Get the CSS selectors to include/exclude
    - Apply those selectors to ALL pages with the same template

    Returns:
        dict with:
        - include_selectors: list of CSS selectors for main content
        - exclude_selectors: list of CSS selectors for noise
        - sample_content: the extracted content from the sample
        - word_count: word count of extracted content
    """
    from pydantic import BaseModel, Field

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "no_api_key"}

    try:
        class SelectorDiscovery(BaseModel):
            """Schema for CSS selector discovery."""
            main_content_selector: str = Field(
                description="The best CSS selector for the main content area (e.g., 'main', 'article', '.content', '#post')"
            )
            additional_content_selectors: list = Field(
                default=[],
                description="Additional CSS selectors that contain important content"
            )
            noise_selectors: list = Field(
                default=[],
                description="CSS selectors that should be REMOVED (navigation, forms, cookies, footer, etc.)"
            )
            extracted_title: str = Field(description="The main page title")
            extracted_content: str = Field(
                description="The main text content extracted using the selectors above"
            )

        graph_config = {
            "llm": {
                "api_key": api_key,
                "model": "openai/gpt-4o-mini",
                "temperature": 0,
            },
            "verbose": False,
            "headless": True,
        }

        prompt = """Analyze this HTML and identify the CSS selectors for content extraction.

1. Find the MAIN CONTENT selector - the container with the primary page content
2. Find any ADDITIONAL selectors for important content
3. List NOISE selectors that should be REMOVED (cookies, forms, navigation, footer, etc.)
4. Extract the title and main text content

Focus on finding REUSABLE selectors that work for similar pages with the same template."""

        scraper = SmartScraperGraph(
            prompt=prompt,
            source=html,
            config=graph_config,
            schema=SelectorDiscovery
        )

        result = scraper.run()

        if isinstance(result, dict):
            include_selectors = [result.get("main_content_selector", "main")]
            include_selectors.extend(result.get("additional_content_selectors", []))
            # Filter out empty/None selectors
            include_selectors = [s for s in include_selectors if s and s.strip()]

            exclude_selectors = result.get("noise_selectors", [])
            exclude_selectors = [s for s in exclude_selectors if s and s.strip()]

            content = result.get("extracted_content", "")
            title = result.get("extracted_title", "")

            return {
                "include_selectors": include_selectors,
                "exclude_selectors": exclude_selectors,
                "title": title,
                "content": content,
                "word_count": len(content.split()) if content else 0,
                "method": "selector_discovery"
            }

        return {"error": "unexpected_result"}

    except Exception as e:
        print(f"[Cleaner] Selector discovery error: {e}")
        return {"error": str(e)[:200]}


def extract_content_batch_with_ai(sample_htmls: List[str], template_ids: List[str], sample_urls: List[str] = None) -> Dict[str, dict]:
    """
    Extract content from multiple HTML samples using parallel ThreadPoolExecutor.

    This processes all templates in parallel using multiple threads, significantly
    reducing total processing time compared to sequential processing.

    Args:
        sample_htmls: List of HTML content strings (one per template)
        template_ids: List of template IDs corresponding to each HTML
        sample_urls: Optional list of URLs for reference

    Returns:
        Dict mapping template_id -> extraction result
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {tid: {"error": "no_api_key", "content": ""} for tid in template_ids}

    if not sample_htmls:
        return {}

    sample_urls = sample_urls or [""] * len(sample_htmls)

    print(f"[Cleaner] Processing {len(sample_htmls)} templates in PARALLEL using ThreadPoolExecutor...")

    def process_single_template(args):
        """Process a single template HTML with ScrapeGraphAI."""
        template_id, html, url = args
        try:
            result = extract_content_with_ai(html)
            if result.get("content") and result.get("word_count", 0) > 50:
                return template_id, {
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "sections": result.get("sections", []),
                    "word_count": result.get("word_count", 0),
                    "method": "scrapegraph_parallel",
                    "url": url,
                }
            else:
                return template_id, {
                    "error": result.get("error", "low_content"),
                    "content": "",
                    "url": url,
                }
        except Exception as e:
            return template_id, {
                "error": str(e)[:200],
                "content": "",
                "url": url,
            }

    # Process all templates in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(sample_htmls), 5)) as executor:
        args_list = list(zip(template_ids, sample_htmls, sample_urls))
        futures = executor.map(process_single_template, args_list)

        for template_id, result in futures:
            results[template_id] = result
            status = "OK" if result.get("word_count", 0) > 50 else "FALLBACK"
            print(f"  [{status}] Template {template_id}: {result.get('word_count', 0)} words")

    print(f"[Cleaner] Parallel batch complete. {len(results)} templates processed.")
    return results


def extract_content_with_ai(html: str, page_type: str = "generic") -> dict:
    """
    Use ScrapeGraphAI to extract clean content from HTML.

    ScrapeGraphAI handles:
    - Pre-processing of HTML (removing scripts, styles, etc.)
    - Chunking for large documents
    - Output validation with Pydantic schema
    - Automatic JSON parsing and error correction

    Args:
        html: The HTML content to clean
        page_type: Type of page for schema selection ("course", "blog", "generic")

    Returns:
        dict with extracted content fields
    """
    from pydantic import BaseModel, Field

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "no_api_key", "content": ""}

    try:
        # Define schema based on page type
        class GenericContent(BaseModel):
            """Schema for generic page content extraction."""
            title: str = Field(description="Main page title")
            main_content: str = Field(
                description="Complete main text content. Include all paragraphs, lists, and sections. Exclude navigation, forms, cookies, legal text, and footer."
            )
            sections: list = Field(
                default=[],
                description="List of section headings found"
            )

        graph_config = {
            "llm": {
                "api_key": api_key,
                "model": "openai/gpt-4o-mini",
                "temperature": 0,
            },
            "verbose": False,
            "headless": True,
        }

        prompt = """Extract the main informational content from this page.

INCLUDE everything that is actual page content:
- Main title and all headings
- All paragraphs of body text
- All list items
- Pricing information
- Product/service descriptions
- Testimonials and reviews
- FAQ questions and answers
- Course curriculum and requirements

EXCLUDE completely (do not extract):
- Navigation menus and breadcrumbs
- Contact forms and form fields
- Cookie consent dialogs
- Legal/privacy/GDPR text
- Footer content
- Social media links
- Promotional banners

Extract the COMPLETE text - do not summarize or shorten any section."""

        scraper = SmartScraperGraph(
            prompt=prompt,
            source=html,
            config=graph_config,
            schema=GenericContent
        )

        result = scraper.run()

        if isinstance(result, dict):
            # Combine all content into a single clean text
            content_parts = []
            if result.get("title"):
                content_parts.append(result["title"])
            if result.get("main_content"):
                content_parts.append(result["main_content"])

            return {
                "title": result.get("title", ""),
                "content": "\n\n".join(content_parts),
                "sections": result.get("sections", []),
                "word_count": len("\n".join(content_parts).split()),
                "method": "scrapegraph_ai"
            }

        return {"error": "unexpected_result", "content": str(result)}

    except Exception as e:
        print(f"[Cleaner] ScrapeGraphAI error: {e}")
        return {"error": str(e)[:200], "content": ""}


def extract_content_with_selectors(html: str, include_selectors: list, exclude_selectors: list) -> str:
    """
    Extract content using multiple include/exclude selectors.

    1. First removes all elements matching exclude selectors
    2. Then extracts text from elements matching include selectors
    3. If no include selectors match, falls back to body
    """
    if not html:
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")

        # Phase 1: Remove excluded elements
        for selector in exclude_selectors:
            try:
                for el in soup.select(selector):
                    el.decompose()
            except Exception:
                pass  # Invalid selector, skip

        # Phase 2: Extract from included elements
        content_parts = []

        if include_selectors:
            for selector in include_selectors:
                try:
                    for el in soup.select(selector):
                        text = el.get_text(separator="\n", strip=True)
                        if text and len(text.split()) > 20:  # Minimum content threshold
                            content_parts.append(text)
                except Exception:
                    pass  # Invalid selector, skip

        # Fallback: if no content found with include selectors, use body
        if not content_parts:
            body = soup.body if soup.body else soup
            text = body.get_text(separator="\n", strip=True)
            content_parts.append(text)

        # Combine and clean
        full_text = "\n\n".join(content_parts)
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        full_text = re.sub(r' {2,}', ' ', full_text)

        return full_text.strip()

    except Exception:
        return ""


def extract_content_with_selector(html: str, selector: str) -> str:
    """
    Extract text content from the given CSS selector.
    Basic extraction - removes scripts/styles but keeps structure.
    """
    if not html or not selector:
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")
        content = soup.select_one(selector)

        if not content:
            # Try variations
            if selector.startswith("div."):
                first_class = selector.split(".")[1].split(".")[0]
                content = soup.select_one(f".{first_class}")

        if not content:
            return ""

        # Remove only technical noise (scripts, styles)
        for el in content.select("script, style, noscript, iframe, svg"):
            el.decompose()

        # Extract text
        text = content.get_text(separator="\n", strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        return text.strip()

    except Exception:
        return ""


def clean_html_simple(html: str) -> str:
    """
    Simple HTML cleaning fallback (when AI is not used or fails).
    Removes noise elements and extracts text.
    """
    if not html:
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")

        # Remove noise elements
        noise_selectors = [
            "nav", "header", "footer", "aside", "script", "style", "noscript",
            "[id*='cookie']", "[id*='Cookie']", "[class*='cookie']",
            "[id*='gdpr']", "[class*='gdpr']", "[id*='consent']",
            "[id*='popup']", "[class*='popup']", "[id*='modal']",
            ".sidebar", ".menu", ".navbar", ".navigation",
            ".advertisement", ".social-share", ".comments",
        ]

        for sel in noise_selectors:
            try:
                for el in soup.select(sel):
                    el.decompose()
            except:
                pass

        # Try to find main content
        content = None
        for selector in ["main", "article", "[role='main']", "#content", ".content"]:
            content = soup.select_one(selector)
            if content and len(content.get_text(strip=True)) > 200:
                break

        if not content:
            content = soup.body if soup.body else soup

        # Extract text
        text = content.get_text(separator="\n", strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        return text.strip()

    except Exception:
        return ""


def clean_html_enhanced(html: str) -> str:
    """
    Enhanced HTML cleaning with aggressive noise removal.
    Used for bulk cleaning after AI has validated the template.

    This removes all common noise patterns that ScrapeGraphAI typically excludes:
    - Cookie/GDPR/consent banners
    - Navigation and menus
    - Forms and form fields
    - Social share buttons
    - Promotional banners
    - Footer content
    """
    if not html:
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")

        # AGGRESSIVE noise removal - all patterns the AI typically excludes
        noise_selectors = [
            # Technical elements
            "script", "style", "noscript", "iframe", "svg", "canvas",

            # Navigation and structure
            "nav", "header", "footer", "aside",
            "[role='navigation']", "[role='banner']", "[role='contentinfo']",

            # Cookie/GDPR/Consent (very common noise)
            "[id*='cookie']", "[id*='Cookie']", "[class*='cookie']", "[class*='Cookie']",
            "[id*='gdpr']", "[class*='gdpr']", "[id*='GDPR']", "[class*='GDPR']",
            "[id*='consent']", "[class*='consent']", "[id*='Consent']",
            "[id*='CookieBot']", "[class*='cookiebot']", "#CybotCookiebotDialog",
            "[id*='onetrust']", "[class*='onetrust']",
            "[id*='privacy']", "[class*='privacy-banner']",

            # Popups and modals
            "[id*='popup']", "[class*='popup']", "[id*='Popup']",
            "[id*='modal']", "[class*='modal']", "[id*='Modal']",
            "[class*='overlay']", "[id*='overlay']",
            "[class*='lightbox']", "[id*='lightbox']",

            # Forms (contact forms, newsletter signups)
            "form", "[id*='contact']", "[class*='contact-form']",
            "[id*='newsletter']", "[class*='newsletter']",
            "[class*='subscribe']", "[id*='subscribe']",
            "[class*='signup']", "[id*='signup']",

            # Navigation menus
            ".sidebar", ".menu", ".navbar", ".navigation",
            "[class*='nav-']", "[class*='-nav']",
            "[class*='menu-']", "[class*='-menu']",
            ".breadcrumb", "[class*='breadcrumb']",

            # Social and sharing
            ".social-share", "[class*='social']", "[class*='share-']",
            "[class*='-share']", "[id*='social']",

            # Advertising and promotions
            ".advertisement", ".ads", "[class*='banner']",
            "[id*='ad-']", "[class*='ad-']", "[id*='ads']",
            "[class*='promo']", "[id*='promo']",

            # Comments sections
            ".comments", "#comments", "[class*='comment-']",
            "[id*='disqus']", "[class*='disqus']",

            # Legal text (often in footer)
            "[class*='legal']", "[class*='terms']",
            "[class*='copyright']", "[id*='copyright']",

            # Chat widgets
            "[id*='chat']", "[class*='chat-widget']",
            "[id*='intercom']", "[class*='intercom']",
            "[id*='zendesk']", "[class*='zendesk']",
        ]

        for sel in noise_selectors:
            try:
                for el in soup.select(sel):
                    el.decompose()
            except:
                pass

        # Also remove elements with suspicious text patterns
        for el in soup.find_all(["div", "section", "p", "span"]):
            text = el.get_text(strip=True).lower()
            if len(text) < 500:  # Only check short elements
                noise_patterns = [
                    "aceptar cookies", "accept cookies", "cookie policy",
                    "política de privacidad", "privacy policy",
                    "términos y condiciones", "terms and conditions",
                    "protección de datos", "data protection",
                    "suscríbete", "subscribe to", "newsletter",
                    "síguenos en", "follow us on",
                    "compartir en", "share on",
                    "todos los derechos reservados", "all rights reserved",
                ]
                if any(pattern in text for pattern in noise_patterns):
                    el.decompose()

        # Try to find main content
        content = None
        for selector in ["main", "article", "[role='main']", "#content", ".content", ".post", ".entry"]:
            content = soup.select_one(selector)
            if content and len(content.get_text(strip=True)) > 200:
                break

        if not content:
            content = soup.body if soup.body else soup

        # Extract text with better formatting
        text = content.get_text(separator="\n", strip=True)

        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Remove any remaining noise patterns in text
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip very short lines that are likely noise
            if len(line) < 20 and any(x in line.lower() for x in ['cookie', 'accept', 'rechazar', 'dismiss']):
                continue
            clean_lines.append(line)

        return '\n'.join(clean_lines).strip()

    except Exception:
        return ""


# =============================================================================
# CLEANING SERVICE
# =============================================================================

class CleaningService:
    """Service to manage HTML cleaning jobs using selector discovery."""

    def __init__(self, base_data_dir: str = "data/crawl4ai_data"):
        self.base_data_dir = Path(base_data_dir)
        self.current_job: Optional[CleaningJob] = None
        self._status_file = self.base_data_dir / ".cleaning_status.json"
        self._discovered_selectors: Dict[str, str] = {}  # template_group -> selector

    def list_available_crawls(self) -> List[dict]:
        """List crawls that have html_content available for cleaning."""
        crawls = []

        if not self.base_data_dir.exists():
            return crawls

        for d in self.base_data_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                pages_files = list((d / "pages").glob("**/*.parquet")) if (d / "pages").exists() else []

                if pages_files:
                    has_html = False
                    total_pages = 0

                    for pf in pages_files[:1]:
                        try:
                            df = pd.read_parquet(pf)
                            has_html = "html_content" in df.columns
                            total_pages = sum(len(pd.read_parquet(f)) for f in pages_files)
                            break
                        except:
                            pass

                    clean_path = d / "pages_clean"
                    is_cleaned = clean_path.exists() and list(clean_path.glob("*.parquet"))

                    crawls.append({
                        "name": d.name,
                        "path": str(d),
                        "pages": total_pages,
                        "has_html_content": has_html,
                        "is_cleaned": bool(is_cleaned),
                        "created": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
                    })

        return sorted(crawls, key=lambda x: x["created"], reverse=True)

    async def start_cleaning(
        self,
        crawl_name: str,
        use_ai: bool = True,
    ) -> CleaningJob:
        """Start a cleaning job for a crawl."""
        if self.current_job and self.current_job.status == "running":
            raise ValueError("A cleaning job is already running")

        crawl_path = self.base_data_dir / crawl_name
        if not crawl_path.exists():
            raise ValueError(f"Crawl '{crawl_name}' not found")

        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.current_job = CleaningJob(
            id=job_id,
            crawl_name=crawl_name,
            status="running",
            use_ai=use_ai,
            started_at=datetime.now().isoformat(),
        )

        self._save_status()
        self._discovered_selectors = {}

        # Run cleaning in background
        asyncio.create_task(self._run_cleaning(crawl_path, use_ai))

        return self.current_job

    async def _run_cleaning(self, crawl_path: Path, use_ai: bool):
        """
        Run the cleaning pipeline with the optimized approach:

        1. CLUSTERING: Group pages by HTML template fingerprint (fast, no AI)
        2. SAMPLING: Take 1 sample per template group
        3. AI ANALYSIS: Use ScrapeGraphAI ONLY on samples (1 call per template)
        4. BULK CLEANING: Apply learned patterns to all pages in each group (no AI cost)

        This reduces AI calls from N_pages to N_templates (e.g., 10000 -> 100)
        """
        try:
            # Load all pages
            self.current_job.status = "loading"
            self._save_status()

            pages_pattern = str(crawl_path / "pages" / "**" / "*.parquet")
            pages_files = glob(pages_pattern, recursive=True)

            all_dfs = []
            for pf in pages_files:
                try:
                    df = pd.read_parquet(pf)
                    all_dfs.append(df)
                except:
                    pass

            if not all_dfs:
                raise ValueError("No pages found in crawl")

            df = pd.concat(all_dfs, ignore_index=True)
            df = df.drop_duplicates(subset="url")

            self.current_job.total_pages = len(df)
            self._save_status()

            if "html_content" not in df.columns:
                raise ValueError("Crawl does not have html_content column")

            # =========================================================
            # PHASE 1: CLUSTERING - Group pages by HTML template
            # =========================================================
            self.current_job.status = "clustering"
            self._save_status()

            print(f"[Cleaner] Fingerprinting {len(df)} pages by HTML template...")
            df["template_group"] = df["html_content"].apply(
                lambda h: extract_html_fingerprint(h) if h and len(h) > 500 else "unknown"
            )

            template_groups = df.groupby("template_group").size().to_dict()
            self.current_job.groups_found = len(template_groups)
            print(f"[Cleaner] Found {len(template_groups)} unique templates")
            self._save_status()

            # =========================================================
            # PHASE 2: AI ANALYSIS - BATCH processing with SmartScraperMultiGraph
            # Instead of N sequential API calls, we make ONE batch call
            # =========================================================
            template_extractors = {}  # template_id -> extraction function/pattern

            if use_ai:
                self.current_job.status = "analyzing"
                self._save_status()

                print(f"[Cleaner] Preparing {len(template_groups)} templates for PARALLEL AI analysis...")

                # Collect sample HTMLs, URLs and template IDs
                sample_htmls = []
                sample_urls = []
                template_ids = []

                for template_id in template_groups.keys():
                    # Get 1 sample page from this template group
                    sample_row = df[df["template_group"] == template_id].iloc[0]
                    sample_htmls.append(sample_row["html_content"])
                    sample_urls.append(sample_row["url"])
                    template_ids.append(template_id)

                print(f"[Cleaner] Processing {len(sample_htmls)} templates with ThreadPoolExecutor (parallel)...")

                # Process ALL templates in PARALLEL
                batch_results = extract_content_batch_with_ai(sample_htmls, template_ids, sample_urls)

                # Process results
                for template_id in template_ids:
                    extraction = batch_results.get(template_id, {})

                    if extraction.get("content") and extraction.get("word_count", 0) > 50:
                        # Store the extraction pattern AND the extracted content for this template
                        template_extractors[template_id] = {
                            "method": "scrapegraph_parallel",
                            "sample_url": extraction.get("url", ""),
                            "sample_word_count": extraction["word_count"],
                            "sections": extraction.get("sections", []),
                            # Store the AI-extracted content to use for sample page
                            "ai_content": extraction.get("content", ""),
                            "ai_title": extraction.get("title", ""),
                        }
                        # Already printed by batch function
                    else:
                        # Mark as needing simple extraction
                        template_extractors[template_id] = {
                            "method": "simple",
                            "error": extraction.get("error", "low_content"),
                        }
                        # Already printed by batch function

                    self.current_job.groups_analyzed += 1
                    self.current_job.rules[template_id] = template_extractors[template_id]["method"]

                self._save_status()
                print(f"[Cleaner] Batch analysis complete. {len(template_extractors)} templates processed.")

            # =========================================================
            # PHASE 3: BULK CLEANING - Apply patterns to all pages
            # NO AI CALLS HERE - use heuristic cleaning based on template analysis
            # =========================================================
            self.current_job.status = "cleaning"
            self._save_status()

            results = []
            print(f"[Cleaner] Extracting content from {len(df)} pages (no AI calls, using heuristics)...")

            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning"):
                try:
                    template = row["template_group"]

                    if use_ai:
                        # Check if this template was successfully analyzed
                        extractor = template_extractors.get(template, {})

                        if extractor.get("method") in ["scrapegraph_ai", "scrapegraph_multi", "scrapegraph_parallel"]:
                            # Template was successfully analyzed by AI
                            # Check if this is the sample page (use AI content directly)
                            if row["url"] == extractor.get("sample_url") and extractor.get("ai_content"):
                                # Use AI-extracted content directly for sample page
                                clean_text = extractor["ai_content"]
                                method = "ai_direct"
                                title = extractor.get("ai_title", row.get("title", ""))
                            else:
                                # Use enhanced heuristic cleaning for other pages (fast, no API call)
                                clean_text = clean_html_enhanced(row["html_content"])
                                method = "ai_heuristic"
                                title = row.get("title", "")
                        else:
                            # Use simple cleaning for this template
                            clean_text = clean_html_simple(row["html_content"])
                            method = "simple"
                            title = row.get("title", "")
                    else:
                        # Simple cleaning without AI
                        clean_text = clean_html_simple(row["html_content"])
                        method = "simple"
                        title = row.get("title", "")

                    results.append({
                        "url": row["url"],
                        "title": title[:500] if title else "",
                        "clean_text": clean_text,
                        "clean_word_count": len(clean_text.split()) if clean_text else 0,
                        "template_group": template,
                        "selector_used": method,
                    })

                    self.current_job.pages_cleaned = len(results)
                    if len(results) % 50 == 0:
                        self._save_status()

                except Exception as e:
                    print(f"[Cleaner] Error processing {row['url']}: {e}")
                    clean_text = clean_html_simple(row["html_content"])
                    results.append({
                        "url": row["url"],
                        "title": row.get("title", "")[:500] if row.get("title") else "",
                        "clean_text": clean_text,
                        "clean_word_count": len(clean_text.split()) if clean_text else 0,
                        "template_group": row["template_group"],
                        "selector_used": "error_fallback",
                    })

            # Save results
            output_path = crawl_path / "pages_clean"
            output_path.mkdir(parents=True, exist_ok=True)

            output_file = output_path / f"clean_{self.current_job.id}.parquet"
            results_df = pd.DataFrame(results)

            # Add original columns if available
            for col in ["meta_description", "crawl_date"]:
                if col in df.columns:
                    results_df[col] = df[col].values[:len(results_df)]

            results_df.to_parquet(output_file, engine="pyarrow", compression="snappy")

            self.current_job.status = "completed"
            self.current_job.completed_at = datetime.now().isoformat()
            self.current_job.output_path = str(output_file)
            self._save_status()

            print(f"[Cleaner] Completed! Cleaned {len(results)} pages -> {output_file}")

        except Exception as e:
            self.current_job.status = "failed"
            self.current_job.error_message = str(e)[:500]
            self.current_job.completed_at = datetime.now().isoformat()
            self._save_status()
            print(f"[Cleaner] Failed: {e}")

    def get_status(self) -> Optional[CleaningJob]:
        """Get current cleaning job status."""
        if self.current_job:
            return self.current_job

        if self._status_file.exists():
            try:
                with open(self._status_file) as f:
                    data = json.load(f)
                    return CleaningJob(**data)
            except:
                pass

        return None

    def _save_status(self):
        """Save current job status to file."""
        if self.current_job:
            with open(self._status_file, "w") as f:
                json.dump(asdict(self.current_job), f)

    def get_cleaning_rules(self, crawl_name: str) -> Optional[Dict]:
        """
        Get cleaning rules/selectors discovered for a crawl.
        """
        crawl_path = self.base_data_dir / crawl_name
        if not crawl_path.exists():
            return None

        # Check if we have cleaned data
        clean_path = crawl_path / "pages_clean"
        if not clean_path.exists():
            return None

        # Get selectors from cleaned data
        try:
            clean_files = list(clean_path.glob("*.parquet"))
            if not clean_files:
                return None

            # Get the most recent clean file
            clean_file = sorted(clean_files, reverse=True)[0]
            df = pd.read_parquet(clean_file)

            # Get selector info per template group
            selector_info = df.groupby("template_group").agg({
                "url": "count",
                "clean_word_count": "mean",
                "selector_used": "first"
            }).to_dict(orient="index")

            return {
                "method": "selector_discovery",
                "template_groups": selector_info,
                "total_pages": len(df),
                "avg_word_count": df["clean_word_count"].mean(),
            }
        except Exception as e:
            return {"error": str(e)}


# =============================================================================
# STANDALONE FUNCTIONS (for API endpoints)
# =============================================================================

def clean_html_with_selector(html: str, selector: str = None) -> str:
    """
    Clean HTML content using a specific selector or auto-discovery.
    """
    if not html:
        return ""

    if selector:
        # Use provided selector
        result = extract_content_with_selector(html, selector)
        if result and len(result.split()) > 50:
            return result

    # Try to discover a selector
    discovered = discover_content_selector(html)
    if discovered:
        result = extract_content_with_selector(html, discovered)
        if result and len(result.split()) > 50:
            return result

    # Fallback to simple cleaning
    return clean_html_simple(html)


def preview_cleaning(html: str, url: str = "") -> dict:
    """
    Preview the cleaning result for a page.
    Shows discovered selector and extracted content.
    """
    result = {
        "url": url,
        "original_length": len(html),
    }

    # Discover selector
    selector = discover_content_selector(html)
    if selector:
        content = extract_content_with_selector(html, selector)
        result["selector_discovered"] = selector
        result["selector_content"] = {
            "content": content[:3000] + "..." if len(content) > 3000 else content,
            "word_count": len(content.split()),
            "method": "selector_discovery",
        }

    # Also show simple cleaning for comparison
    simple_text = clean_html_simple(html)
    result["simple_cleaned"] = {
        "content": simple_text[:3000] + "..." if len(simple_text) > 3000 else simple_text,
        "word_count": len(simple_text.split()),
        "method": "simple",
    }

    return result


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_cleaning_service: Optional[CleaningService] = None


def get_cleaning_service() -> CleaningService:
    """Get or create the global cleaning service instance."""
    global _cleaning_service
    if _cleaning_service is None:
        _cleaning_service = CleaningService()
    return _cleaning_service
