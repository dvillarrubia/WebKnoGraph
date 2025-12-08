"""
Crawler Service for Graph-RAG.
Manages crawl jobs and provides status tracking.
"""

import asyncio
import subprocess
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict
from glob import glob

import pandas as pd


@dataclass
class CrawlJob:
    """Represents a crawl job."""
    id: str
    url: str
    status: str  # pending, running, completed, failed, stopped
    max_pages: int
    delay: float
    use_sitemap: bool
    content_filter: bool
    resume: bool = False  # Resume from previous crawl
    force_sitemap: bool = False  # Force sitemap re-fetch in resume mode
    urls_list: list = field(default_factory=list)  # URLs to crawl from file/API
    urls_only: bool = False  # Only crawl URLs from list, don't discover new links
    exclude_selectors: list = field(default_factory=list)  # Additional CSS selectors to exclude
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    pages_crawled: int = 0
    pages_previously_crawled: int = 0  # Pages from previous crawl when resuming
    links_found: int = 0
    errors: int = 0
    current_url: Optional[str] = None
    output_dir: Optional[str] = None
    process_pid: Optional[int] = None
    error_message: Optional[str] = None


class CrawlerService:
    """Service to manage web crawling jobs."""

    def __init__(self, base_output_dir: str = "data/crawl4ai_data"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.current_job: Optional[CrawlJob] = None
        self._process: Optional[subprocess.Popen] = None
        self._status_file = self.base_output_dir / ".crawl_status.json"

    def _get_output_dir_for_url(self, url: str) -> Path:
        """Generate output directory name from URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.replace(".", "_").replace(":", "_")
        return self.base_output_dir / domain

    def _count_crawled_pages(self, output_dir: Path) -> int:
        """Count pages already crawled from parquet files."""
        pages_pattern = str(output_dir / "pages" / "**" / "*.parquet")
        pages_files = glob(pages_pattern, recursive=True)

        total_pages = 0
        for pf in pages_files:
            try:
                df = pd.read_parquet(pf)
                total_pages += len(df)
            except:
                pass

        return total_pages

    async def start_crawl(
        self,
        url: str,
        max_pages: int = 0,  # 0 = unlimited
        delay: float = 0.5,
        use_sitemap: bool = True,
        content_filter: bool = True,
        resume: bool = False,
        force_sitemap: bool = False,
        urls_list: list = None,  # List of URLs to crawl
        urls_only: bool = False,  # Only crawl URLs from list, don't discover new links
        exclude_selectors: list = None,  # Additional CSS selectors to exclude (cookies, etc.)
    ) -> CrawlJob:
        """Start a new crawl job."""
        # Check if already running
        if self.current_job and self.current_job.status == "running":
            raise ValueError("A crawl is already running")

        # Normalize URL - add https:// if no protocol specified
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"

        # Create job
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self._get_output_dir_for_url(url)

        # Count previously crawled pages if resuming
        pages_previously_crawled = 0
        if resume:
            pages_previously_crawled = self._count_crawled_pages(output_dir)

        # Normalize urls_list and exclude_selectors
        urls_list = urls_list or []
        exclude_selectors = exclude_selectors or []

        self.current_job = CrawlJob(
            id=job_id,
            url=url,
            status="running",
            max_pages=max_pages,
            delay=delay,
            use_sitemap=use_sitemap,
            content_filter=content_filter,
            resume=resume,
            force_sitemap=force_sitemap,
            urls_list=urls_list,
            urls_only=urls_only,
            exclude_selectors=exclude_selectors,
            started_at=datetime.now().isoformat(),
            output_dir=str(output_dir),
            pages_previously_crawled=pages_previously_crawled,
        )

        # Build command
        script_path = Path(__file__).parent.parent.parent / "scripts" / "crawl4ai_advanced.py"

        cmd = [
            "python3",
            str(script_path),
            "--url", url,
            "--max-pages", str(max_pages),
            "--delay", str(delay),
        ]

        # Script uses --no-sitemap and --no-filter (inverted flags)
        if not use_sitemap:
            cmd.append("--no-sitemap")
        if not content_filter:
            cmd.append("--no-filter")
        if resume:
            cmd.append("--resume")
        if force_sitemap:
            cmd.append("--force-sitemap")

        # Handle URLs list - write to temp file if provided
        urls_file_path = None
        if urls_list and len(urls_list) > 0:
            urls_file_path = output_dir / f".urls_list_{job_id}.txt"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(urls_file_path, "w") as f:
                for u in urls_list:
                    # Clean and normalize URL
                    u = u.strip()
                    if u and not u.startswith("#"):
                        f.write(f"{u}\n")
            cmd.extend(["--urls-file", str(urls_file_path)])

        if urls_only:
            cmd.append("--urls-only")

        # Add exclude selectors
        if exclude_selectors and len(exclude_selectors) > 0:
            cmd.extend(["--exclude-selectors", ",".join(exclude_selectors)])

        # Start process
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(Path(__file__).parent.parent.parent),
        )

        self.current_job.process_pid = self._process.pid
        self._save_status()

        # Start monitoring task
        asyncio.create_task(self._monitor_crawl())

        return self.current_job

    async def _monitor_crawl(self):
        """Monitor the crawl process and update status."""
        if not self._process or not self.current_job:
            return

        while self._process.poll() is None:
            # Update stats from output directory
            await self._update_stats()
            self._save_status()
            await asyncio.sleep(2)

        # Process finished
        return_code = self._process.returncode

        if return_code == 0:
            self.current_job.status = "completed"
        elif self.current_job.status == "stopped":
            pass  # Keep stopped status
        else:
            # Check if we crawled pages successfully despite exit code
            await self._update_stats()
            if self.current_job.pages_crawled > 0:
                self.current_job.status = "completed"
                self.current_job.error_message = None
            else:
                self.current_job.status = "failed"
                self.current_job.error_message = f"Process exited with code {return_code}"

        self.current_job.completed_at = datetime.now().isoformat()
        await self._update_stats()
        self._save_status()
        self._process = None

    async def _update_stats(self):
        """Update crawl statistics from output files."""
        if not self.current_job or not self.current_job.output_dir:
            return

        output_dir = Path(self.current_job.output_dir)

        # Count pages from parquet files
        pages_pattern = str(output_dir / "pages" / "**" / "*.parquet")
        pages_files = glob(pages_pattern, recursive=True)

        total_pages = 0
        for pf in pages_files:
            try:
                df = pd.read_parquet(pf)
                total_pages += len(df)
            except:
                pass

        self.current_job.pages_crawled = total_pages

        # Count links
        links_pattern = str(output_dir / "links" / "**" / "*.parquet")
        links_files = glob(links_pattern, recursive=True)

        total_links = 0
        for lf in links_files:
            try:
                df = pd.read_parquet(lf)
                total_links += len(df)
            except:
                pass

        self.current_job.links_found = total_links

    async def stop_crawl(self) -> Optional[CrawlJob]:
        """Stop the current crawl job."""
        if not self.current_job or not self._process:
            return None

        self.current_job.status = "stopped"
        self._process.terminate()

        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()

        self.current_job.completed_at = datetime.now().isoformat()
        await self._update_stats()
        self._save_status()

        return self.current_job

    def get_status(self) -> Optional[CrawlJob]:
        """Get current crawl status."""
        if self.current_job:
            return self.current_job

        # Try to load from file
        if self._status_file.exists():
            try:
                with open(self._status_file) as f:
                    data = json.load(f)
                    return CrawlJob(**data)
            except:
                pass

        return None

    def _save_status(self):
        """Save current job status to file."""
        if self.current_job:
            with open(self._status_file, "w") as f:
                json.dump(asdict(self.current_job), f)

    def list_available_crawls(self) -> list[dict]:
        """List available crawl data directories."""
        crawls = []

        for d in self.base_output_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                # Check for parquet files
                pages_files = list((d / "pages").glob("**/*.parquet")) if (d / "pages").exists() else []
                links_files = list((d / "links").glob("**/*.parquet")) if (d / "links").exists() else []

                if pages_files:
                    # Count pages
                    total_pages = 0
                    for pf in pages_files:
                        try:
                            df = pd.read_parquet(pf)
                            total_pages += len(df)
                        except:
                            pass

                    total_links = 0
                    for lf in links_files:
                        try:
                            df = pd.read_parquet(lf)
                            total_links += len(df)
                        except:
                            pass

                    crawls.append({
                        "path": str(d),
                        "name": d.name,
                        "pages": total_pages,
                        "links": total_links,
                        "created": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
                    })

        return sorted(crawls, key=lambda x: x["created"], reverse=True)


# Global instance
_crawler_service: Optional[CrawlerService] = None


def get_crawler_service() -> CrawlerService:
    """Get or create the global crawler service instance."""
    global _crawler_service
    if _crawler_service is None:
        _crawler_service = CrawlerService()
    return _crawler_service
