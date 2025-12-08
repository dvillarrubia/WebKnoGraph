#!/usr/bin/env python3
"""
Crawler using Crawl4AI for WebKnoGraph.
Crawls a website with JavaScript rendering and saves to Parquet format.
"""

import asyncio
import argparse
import os
import sys
import time
import hashlib
from pathlib import Path
from urllib.parse import urlparse, urljoin
from datetime import datetime

import pandas as pd
from tqdm import tqdm

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
except ImportError:
    print("Please install crawl4ai: pip install crawl4ai")
    sys.exit(1)


async def crawl_site(
    start_url: str,
    max_pages: int = 1000,
    output_dir: str = "./data/crawl4ai_data",
    delay: float = 1.0,
    headless: bool = True,
):
    """Crawl a website using Crawl4AI."""

    base_domain = urlparse(start_url).netloc
    output_path = Path(output_dir) / base_domain.replace(".", "_")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                  Crawl4AI WebKnoGraph Crawler                ║
╠══════════════════════════════════════════════════════════════╣
║  URL:          {start_url:<45} ║
║  Domain:       {base_domain:<45} ║
║  Max Pages:    {max_pages:<45} ║
║  Output:       {str(output_path):<45} ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Track URLs
    visited = set()
    to_visit = [start_url]
    results = []

    # Browser config
    browser_config = BrowserConfig(
        headless=headless,
        verbose=False,
    )

    # Crawler config
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_until="networkidle",
        page_timeout=30000,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        pbar = tqdm(total=max_pages, desc="Crawling")

        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)

            if url in visited:
                continue

            # Filter to same domain
            if urlparse(url).netloc != base_domain:
                continue

            visited.add(url)

            try:
                result = await crawler.arun(url=url, config=crawler_config)

                if result.success:
                    # Extract links for crawling
                    for link in result.links.get("internal", []):
                        href = link.get("href", "")
                        if href and href not in visited and href not in to_visit:
                            # Filter out common non-content URLs
                            if not any(x in href.lower() for x in [
                                "login", "logout", "signin", "signup", "register",
                                "cart", "checkout", "account", "password",
                                ".pdf", ".jpg", ".png", ".gif", ".css", ".js",
                                "mailto:", "tel:", "javascript:", "#"
                            ]):
                                to_visit.append(href)

                    # Store result
                    content_hash = hashlib.sha256(
                        (result.markdown or "").encode()
                    ).hexdigest()

                    results.append({
                        "url": url,
                        "title": result.metadata.get("title", "") if result.metadata else "",
                        "markdown": result.markdown or "",
                        "html": result.html or "",
                        "content_hash": content_hash,
                        "status_code": 200,
                        "crawl_date": datetime.now().isoformat(),
                        "links_found": len(result.links.get("internal", [])),
                    })

                    pbar.set_postfix({"queue": len(to_visit), "url": url[:40]})
                else:
                    results.append({
                        "url": url,
                        "title": "",
                        "markdown": "",
                        "html": "",
                        "content_hash": "",
                        "status_code": result.status_code or 0,
                        "crawl_date": datetime.now().isoformat(),
                        "links_found": 0,
                    })

            except Exception as e:
                results.append({
                    "url": url,
                    "title": "",
                    "markdown": "",
                    "html": "",
                    "content_hash": "",
                    "status_code": 0,
                    "crawl_date": datetime.now().isoformat(),
                    "links_found": 0,
                    "error": str(e),
                })

            pbar.update(1)

            # Save periodically
            if len(results) % 50 == 0 and results:
                save_results(results, output_path)

            # Rate limiting
            await asyncio.sleep(delay)

        pbar.close()

    # Final save
    if results:
        save_results(results, output_path)

    # Summary
    print(f"\n{'='*60}")
    print("CRAWL COMPLETE")
    print(f"{'='*60}")
    print(f"Total pages crawled: {len(visited)}")
    print(f"Successful: {sum(1 for r in results if r['status_code'] == 200)}")
    print(f"Failed: {sum(1 for r in results if r['status_code'] != 200)}")
    print(f"URLs remaining in queue: {len(to_visit)}")
    print(f"Data saved to: {output_path}")

    return results


def save_results(results: list, output_path: Path):
    """Save results to Parquet."""
    df = pd.DataFrame(results)
    today = datetime.now().strftime("%Y-%m-%d")
    partition_path = output_path / f"crawl_date={today}"
    partition_path.mkdir(exist_ok=True)

    filename = f"{int(time.time())}.parquet"
    df.to_parquet(partition_path / filename, engine="pyarrow", compression="snappy")
    print(f"\n  Saved {len(results)} pages to {partition_path / filename}")


def main():
    parser = argparse.ArgumentParser(description="Crawl a website with Crawl4AI")
    parser.add_argument("--url", required=True, help="Start URL")
    parser.add_argument("--max-pages", type=int, default=1000, help="Maximum pages")
    parser.add_argument("--output-dir", default="./data/crawl4ai_data", help="Output directory")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests")
    parser.add_argument("--no-headless", action="store_true", help="Show browser")

    args = parser.parse_args()

    asyncio.run(crawl_site(
        start_url=args.url,
        max_pages=args.max_pages,
        output_dir=args.output_dir,
        delay=args.delay,
        headless=not args.no_headless,
    ))


if __name__ == "__main__":
    main()
