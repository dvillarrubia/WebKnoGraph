"""
Crawler Service for Graph-RAG.
Manages crawl jobs and provides status tracking.

Supports two modes:
- Remote: calls the crawler-service Docker container via HTTP (when CRAWLER_SERVICE_URL is set)
- Local: spawns crawl4ai_advanced.py as a subprocess (fallback)
"""

import asyncio
import subprocess
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict
from glob import glob

import pandas as pd

logger = logging.getLogger(__name__)


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
    respect_robots: bool = True  # Respect robots.txt rules
    skip_noindex: bool = True  # Skip pages with noindex meta tag
    sitemap_only: bool = False  # Only crawl URLs from sitemap, don't follow discovered links
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
    """Service to manage web crawling jobs.

    When CRAWLER_SERVICE_URL is set, delegates crawl control to the remote
    crawler-service container via HTTP. Otherwise, falls back to running
    crawl4ai_advanced.py as a local subprocess.

    Data (parquet files) is always read locally from the shared volume.
    """

    def __init__(self, base_output_dir: str = "data/crawl4ai_data"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.current_job: Optional[CrawlJob] = None
        self._process: Optional[subprocess.Popen] = None
        self._status_file = self.base_output_dir / ".crawl_status.json"

        # Check if remote crawler service is configured
        self._crawler_url = os.environ.get("CRAWLER_SERVICE_URL", "").rstrip("/")
        if self._crawler_url:
            logger.info(f"Crawler service configured at: {self._crawler_url}")
        else:
            logger.info("No CRAWLER_SERVICE_URL set, using local subprocess mode")

    @property
    def is_remote(self) -> bool:
        return bool(self._crawler_url)

    # =========================================================================
    # REMOTE MODE (HTTP calls to crawler-service container)
    # =========================================================================

    async def _http_post(self, path: str, json_data: dict = None) -> dict:
        """Make a POST request to the crawler service."""
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._crawler_url}{path}",
                json=json_data,
            )
            response.raise_for_status()
            return response.json()

    async def _http_get(self, path: str, params: dict = None) -> dict:
        """Make a GET request to the crawler service."""
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self._crawler_url}{path}",
                params=params,
            )
            response.raise_for_status()
            return response.json()

    async def _remote_start_crawl(self, **kwargs) -> CrawlJob:
        """Start crawl via remote crawler service."""
        data = await self._http_post("/crawl/start", json_data=kwargs)
        self.current_job = CrawlJob(**data)
        self._save_status()
        # Start polling the remote status
        asyncio.create_task(self._remote_monitor_crawl())
        return self.current_job

    async def _remote_monitor_crawl(self):
        """Poll the remote crawler service for status updates."""
        while True:
            await asyncio.sleep(3)
            try:
                data = await self._http_get("/crawl/status")
                if data.get("status") == "idle":
                    break
                self.current_job = CrawlJob(**data)
                self._save_status()
                if self.current_job.status in ("completed", "failed", "stopped"):
                    break
            except Exception as e:
                logger.warning(f"Error polling crawler status: {e}")
                # Don't break on transient errors, keep polling
                continue

    async def _remote_stop_crawl(self) -> Optional[CrawlJob]:
        """Stop crawl via remote crawler service."""
        data = await self._http_post("/crawl/stop")
        if data.get("status") == "no_active_crawl":
            return None
        self.current_job = CrawlJob(**data)
        self._save_status()
        return self.current_job

    async def _remote_get_status(self) -> Optional[dict]:
        """Get status from remote crawler service."""
        data = await self._http_get("/crawl/status")
        if data.get("status") == "idle":
            return None
        return data

    async def _remote_get_logs(self, last_n: int = 100) -> dict:
        """Get logs from remote crawler service."""
        return await self._http_get("/crawl/logs", params={"last_n": last_n})

    # =========================================================================
    # LOCAL MODE (subprocess - original behavior)
    # =========================================================================

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

    async def _local_start_crawl(
        self,
        url: str,
        max_pages: int = 0,
        delay: float = 0.5,
        use_sitemap: bool = True,
        content_filter: bool = True,
        resume: bool = False,
        force_sitemap: bool = False,
        urls_list: list = None,
        urls_only: bool = False,
        exclude_selectors: list = None,
        respect_robots: bool = True,
        skip_noindex: bool = True,
        sitemap_only: bool = False,
    ) -> CrawlJob:
        """Start a crawl job using local subprocess."""
        # Normalize URL
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"

        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self._get_output_dir_for_url(url)

        pages_previously_crawled = 0
        if resume:
            pages_previously_crawled = self._count_crawled_pages(output_dir)

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
            respect_robots=respect_robots,
            skip_noindex=skip_noindex,
            sitemap_only=sitemap_only,
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
            "--output-dir", str(self.base_output_dir),
            "--delay", str(delay),
        ]

        if not use_sitemap:
            cmd.append("--no-sitemap")
        if not content_filter:
            cmd.append("--no-filter")
        if resume:
            cmd.append("--resume")
        if force_sitemap:
            cmd.append("--force-sitemap")

        # Handle URLs list
        urls_file_path = None
        if urls_list and len(urls_list) > 0:
            urls_file_path = output_dir / f".urls_list_{job_id}.txt"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(urls_file_path, "w") as f:
                for u in urls_list:
                    u = u.strip()
                    if u and not u.startswith("#"):
                        f.write(f"{u}\n")
            cmd.extend(["--urls-file", str(urls_file_path)])

        if urls_only:
            cmd.append("--urls-only")

        if exclude_selectors and len(exclude_selectors) > 0:
            cmd.extend(["--exclude-selectors", ",".join(exclude_selectors)])

        if not respect_robots:
            cmd.append("--no-robots")
        if not skip_noindex:
            cmd.append("--no-skip-noindex")
        if sitemap_only:
            cmd.append("--sitemap-only")

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

        asyncio.create_task(self._local_monitor_crawl())

        return self.current_job

    async def _local_monitor_crawl(self):
        """Monitor the local crawl process and update status."""
        if not self._process or not self.current_job:
            return

        while self._process.poll() is None:
            await self._update_stats()
            self._save_status()
            await asyncio.sleep(2)

        return_code = self._process.returncode

        if return_code == 0:
            self.current_job.status = "completed"
        elif self.current_job.status == "stopped":
            pass
        else:
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

    async def _local_stop_crawl(self) -> Optional[CrawlJob]:
        """Stop the local crawl process."""
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

    # =========================================================================
    # PUBLIC API (dispatches to remote or local)
    # =========================================================================

    async def start_crawl(
        self,
        url: str,
        max_pages: int = 0,
        delay: float = 0.5,
        use_sitemap: bool = True,
        content_filter: bool = True,
        resume: bool = False,
        force_sitemap: bool = False,
        urls_list: list = None,
        urls_only: bool = False,
        exclude_selectors: list = None,
        respect_robots: bool = True,
        skip_noindex: bool = True,
        sitemap_only: bool = False,
    ) -> CrawlJob:
        """Start a new crawl job."""
        # Check if already running
        if self.current_job and self.current_job.status == "running":
            raise ValueError("A crawl is already running")

        if self.is_remote:
            return await self._remote_start_crawl(
                url=url,
                max_pages=max_pages,
                delay=delay,
                use_sitemap=use_sitemap,
                content_filter=content_filter,
                resume=resume,
                force_sitemap=force_sitemap,
                urls_list=urls_list or [],
                urls_only=urls_only,
                exclude_selectors=exclude_selectors or [],
                respect_robots=respect_robots,
                skip_noindex=skip_noindex,
                sitemap_only=sitemap_only,
            )
        else:
            return await self._local_start_crawl(
                url=url,
                max_pages=max_pages,
                delay=delay,
                use_sitemap=use_sitemap,
                content_filter=content_filter,
                resume=resume,
                force_sitemap=force_sitemap,
                urls_list=urls_list,
                urls_only=urls_only,
                exclude_selectors=exclude_selectors,
                respect_robots=respect_robots,
                skip_noindex=skip_noindex,
                sitemap_only=sitemap_only,
            )

    async def stop_crawl(self) -> Optional[CrawlJob]:
        """Stop the current crawl job."""
        if self.is_remote:
            return await self._remote_stop_crawl()
        else:
            return await self._local_stop_crawl()

    async def get_status(self) -> Optional[CrawlJob]:
        """Get current crawl status.

        In remote mode, fetches from crawler-service if no local job is cached.
        This handles the case where graph-rag-api restarts while a crawl is running.
        """
        if self.current_job:
            return self.current_job

        # In remote mode, check the crawler service directly
        if self.is_remote:
            try:
                data = await self._http_get("/crawl/status")
                if data.get("status") != "idle":
                    self.current_job = CrawlJob(**data)
                    self._save_status()
                    # If crawl is still running, resume monitoring
                    if self.current_job.status == "running":
                        asyncio.create_task(self._remote_monitor_crawl())
                    return self.current_job
            except Exception as e:
                logger.warning(f"Error fetching remote status: {e}")

        # Fallback: try to load from local status file
        if self._status_file.exists():
            try:
                with open(self._status_file) as f:
                    data = json.load(f)
                    return CrawlJob(**data)
            except Exception:
                pass

        return None

    async def get_logs(self, last_n: int = 100) -> dict:
        """Get crawler logs.

        In remote mode, fetches from crawler-service.
        In local mode, reads from .crawl_log.jsonl on disk.
        """
        if self.is_remote:
            try:
                return await self._remote_get_logs(last_n)
            except Exception as e:
                logger.warning(f"Error fetching remote logs: {e}")
                # Fall through to local file read

        # Read from local file (shared volume in Docker, or local dev)
        job = await self.get_status()
        if not job or not job.output_dir:
            return {"logs": [], "count": 0}

        log_file = Path(job.output_dir).parent / ".crawl_log.jsonl"

        if not log_file.exists():
            return {"logs": [], "count": 0}

        try:
            with open(log_file, "r") as f:
                lines = f.readlines()

            logs = []
            for line in lines[-last_n:]:
                try:
                    logs.append(json.loads(line.strip()))
                except Exception:
                    pass

            return {"logs": logs, "count": len(lines)}
        except Exception as e:
            return {"logs": [], "count": 0, "error": str(e)}

    def _save_status(self):
        """Save current job status to file."""
        if self.current_job:
            with open(self._status_file, "w") as f:
                json.dump(asdict(self.current_job), f)

    def list_available_crawls(self) -> list[dict]:
        """List available crawl data directories.

        Always reads from local filesystem (shared volume in Docker).
        """
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
