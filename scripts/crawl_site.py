#!/usr/bin/env python3
"""
CLI Crawler for WebKnoGraph.
Crawls a website and saves to Parquet format.
"""

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backend.config.crawler_config import CrawlerConfig
from src.backend.utils.strategies import (
    VisitedUrlManager,
    BFSCrawlingStrategy,
    DFSCrawlingStrategy,
)
from src.backend.data.repositories import CrawlStateRepository
from src.backend.utils.http import HttpClient
from src.backend.utils.url import UrlFilter, LinkExtractor
from src.backend.services.crawler_service import WebCrawler
from src.shared.logging_config import ConsoleAndGradioLogger
import io
import duckdb


def main():
    parser = argparse.ArgumentParser(description="Crawl a website")
    parser.add_argument("--url", required=True, help="Start URL (e.g., https://www.ilerna.es)")
    parser.add_argument("--max-pages", type=int, default=1000, help="Maximum pages to crawl")
    parser.add_argument("--output-dir", default="./data/crawled_data_parquet", help="Output directory for Parquet files")
    parser.add_argument("--state-db", default="./data/crawler_state.db", help="SQLite state database path")
    parser.add_argument("--strategy", choices=["BFS", "DFS"], default="BFS", help="Crawling strategy")
    parser.add_argument("--allowed-path", default="/", help="Allowed path segment (e.g., /blog/)")
    parser.add_argument("--delay-min", type=float, default=1.0, help="Minimum request delay (seconds)")
    parser.add_argument("--delay-max", type=float, default=3.0, help="Maximum request delay (seconds)")

    args = parser.parse_args()

    # Extract domain from URL
    base_domain = urlparse(args.url).netloc
    if not base_domain:
        print("Error: Invalid URL")
        sys.exit(1)

    # Create output directories
    output_dir = Path(args.output_dir) / base_domain.replace(".", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    state_db = Path(args.state_db).parent / f"{base_domain.replace('.', '_')}_state.db"
    state_db.parent.mkdir(parents=True, exist_ok=True)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    WebKnoGraph Crawler                       ║
╠══════════════════════════════════════════════════════════════╣
║  URL:          {args.url:<45} ║
║  Domain:       {base_domain:<45} ║
║  Max Pages:    {args.max_pages:<45} ║
║  Strategy:     {args.strategy:<45} ║
║  Output:       {str(output_dir):<45} ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Configure crawler
    config = CrawlerConfig(
        initial_start_url=args.url,
        allowed_path_segment=args.allowed_path,
        state_db_path=str(state_db),
        parquet_path=str(output_dir),
        max_pages_to_crawl=args.max_pages,
        base_domain=base_domain,
        min_request_delay=args.delay_min,
        max_request_delay=args.delay_max,
        save_interval_pages=50,  # Save every 50 pages
    )

    # Initialize logger
    log_stream = io.StringIO()
    logger = ConsoleAndGradioLogger(log_stream)

    # Initialize state repository
    state_repository = CrawlStateRepository(config.state_db_path, logger)
    visited_manager = VisitedUrlManager()

    # Rebuild visited set from existing data
    print("Checking for existing crawl data...")
    try:
        parquet_glob = str(output_dir / "**" / "*.parquet")
        visited_df = duckdb.query(f"SELECT DISTINCT URL FROM read_parquet('{parquet_glob}')").to_df()
        for url in visited_df["URL"]:
            visited_manager.add(url)
        print(f"  Found {visited_manager.size()} previously crawled URLs")
    except Exception:
        print("  No existing data found, starting fresh")

    # Initialize strategy
    strategy_class = BFSCrawlingStrategy if args.strategy == "BFS" else DFSCrawlingStrategy
    crawling_strategy = strategy_class(visited_manager, logger)

    # Load frontier from DB or start fresh
    loaded_frontier = state_repository.load_frontier()
    unvisited = [info for info in loaded_frontier if not visited_manager.contains(info[0])]

    if unvisited:
        crawling_strategy.prime_with_frontier(unvisited)
        print(f"  Resuming with {len(unvisited)} URLs in frontier")
    elif not visited_manager.contains(args.url):
        crawling_strategy.add_links([(args.url, 0)])
        print(f"  Starting from {args.url}")
    else:
        print("  Start URL already crawled, checking for new links...")

    # Initialize components
    http_client = HttpClient(config, logger)
    url_filter = UrlFilter(config.allowed_path_segment, config.base_domain)
    link_extractor = LinkExtractor(url_filter, config.allowed_query_params)

    # Create crawler
    crawler = WebCrawler(
        config,
        crawling_strategy,
        state_repository,
        http_client,
        url_filter,
        link_extractor,
        logger,
    )

    print("\nStarting crawl...\n")

    # Run crawler
    try:
        for event in crawler.crawl():
            status = event.get("status", "")
            save_event = event.get("save_event")
            if save_event:
                print(f"\n{save_event}")
    except KeyboardInterrupt:
        print("\n\nCrawl interrupted by user. Saving state...")

    # Final summary
    print("\n" + "=" * 60)
    print("CRAWL COMPLETE")
    print("=" * 60)

    try:
        parquet_glob = str(output_dir / "**" / "*.parquet")
        summary = duckdb.query(f"""
            SELECT
                CASE
                    WHEN Status_Code >= 200 AND Status_Code < 300 THEN 'Success'
                    WHEN Status_Code >= 300 AND Status_Code < 400 THEN 'Redirect'
                    ELSE 'Error'
                END AS Category,
                COUNT(*) as Total
            FROM read_parquet('{parquet_glob}')
            GROUP BY Category
        """).to_df()

        total = summary["Total"].sum()
        print(f"Total URLs crawled: {total}")
        print("\nBy status:")
        for _, row in summary.iterrows():
            print(f"  {row['Category']}: {row['Total']}")
    except Exception as e:
        print(f"Could not generate summary: {e}")

    print(f"\nData saved to: {output_dir}")
    print(f"State saved to: {state_db}")


if __name__ == "__main__":
    main()
