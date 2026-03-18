"""
Crawler Service - Standalone FastAPI wrapper for Crawl4AI.
Runs as a separate Docker container, exposing crawl control via HTTP.
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
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="WebKnoGraph Crawler Service", version="1.0.0")

BASE_OUTPUT_DIR = Path(os.environ.get("CRAWL_OUTPUT_DIR", "/app/data/crawl4ai_data")).resolve()
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA MODEL (same CrawlJob as the main app)
# =============================================================================

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
    resume: bool = False
    force_sitemap: bool = False
    urls_list: list = field(default_factory=list)
    urls_only: bool = False
    exclude_selectors: list = field(default_factory=list)
    respect_robots: bool = True
    skip_noindex: bool = True
    sitemap_only: bool = False
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    pages_crawled: int = 0
    pages_previously_crawled: int = 0
    links_found: int = 0
    errors: int = 0
    current_url: Optional[str] = None
    output_dir: Optional[str] = None
    process_pid: Optional[int] = None
    error_message: Optional[str] = None


# =============================================================================
# CRAWLER MANAGER (subprocess-based, same logic as original CrawlerService)
# =============================================================================

class CrawlerManager:
    """Manages the crawl4ai_advanced.py subprocess."""

    def __init__(self):
        self.current_job: Optional[CrawlJob] = None
        self._process: Optional[subprocess.Popen] = None
        self._status_file = BASE_OUTPUT_DIR / ".crawl_status.json"

    def _get_output_dir_for_url(self, url: str) -> Path:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.replace(".", "_").replace(":", "_")
        return BASE_OUTPUT_DIR / domain

    def _count_crawled_pages(self, output_dir: Path) -> int:
        pages_pattern = str(output_dir / "pages" / "**" / "*.parquet")
        pages_files = glob(pages_pattern, recursive=True)
        total_pages = 0
        for pf in pages_files:
            try:
                df = pd.read_parquet(pf)
                total_pages += len(df)
            except Exception:
                pass
        return total_pages

    async def start_crawl(self, params: dict) -> CrawlJob:
        """Start a new crawl job."""
        # Ensure stale jobs are detected before blocking
        self.get_status()
        if self.current_job and self.current_job.status == "running":
            raise ValueError("A crawl is already running")

        url = params["url"]
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"

        max_pages = params.get("max_pages", 0)
        delay = params.get("delay", 0.5)
        use_sitemap = params.get("use_sitemap", True)
        content_filter = params.get("content_filter", True)
        resume = params.get("resume", False)
        force_sitemap = params.get("force_sitemap", False)
        urls_list = params.get("urls_list", [])
        urls_only = params.get("urls_only", False)
        exclude_selectors = params.get("exclude_selectors", [])
        respect_robots = params.get("respect_robots", True)
        skip_noindex = params.get("skip_noindex", True)
        sitemap_only = params.get("sitemap_only", False)

        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self._get_output_dir_for_url(url)

        pages_previously_crawled = 0
        if resume:
            pages_previously_crawled = self._count_crawled_pages(output_dir)

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
            urls_list=urls_list or [],
            urls_only=urls_only,
            exclude_selectors=exclude_selectors or [],
            respect_robots=respect_robots,
            skip_noindex=skip_noindex,
            sitemap_only=sitemap_only,
            started_at=datetime.now().isoformat(),
            output_dir=str(output_dir),
            pages_previously_crawled=pages_previously_crawled,
        )

        # Build command - script is in the same directory
        script_path = Path(__file__).parent / "crawl4ai_advanced.py"

        cmd = [
            "python3",
            str(script_path),
            "--url", url,
            "--max-pages", str(max_pages),
            "--output-dir", str(BASE_OUTPUT_DIR),
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
            cwd=str(Path(__file__).parent),
        )

        self.current_job.process_pid = self._process.pid
        self._save_status()

        asyncio.create_task(self._monitor_crawl())

        return self.current_job

    async def _monitor_crawl(self):
        if not self._process or not self.current_job:
            return

        auto_resume_attempts = 0
        max_auto_resumes = 5  # Máximo de auto-resumes antes de rendirse

        while True:
            # Esperar a que el proceso termine
            while self._process and self._process.poll() is None:
                await self._update_stats()
                self._save_status()
                await asyncio.sleep(2)

            if not self._process:
                break

            return_code = self._process.returncode
            await self._update_stats()

            # Si fue parado manualmente, no auto-resumir
            if self.current_job.status == "stopped":
                break

            if return_code == 0:
                self.current_job.status = "completed"
                break

            # Proceso murió inesperadamente — intentar auto-resume
            if self.current_job.pages_crawled > 0 and auto_resume_attempts < max_auto_resumes:
                auto_resume_attempts += 1
                print(f"⚠️  AUTO-RESUME: Proceso murió (code={return_code}), "
                      f"pero hay {self.current_job.pages_crawled} páginas. "
                      f"Reiniciando (intento {auto_resume_attempts}/{max_auto_resumes})...")

                self._save_status()
                await asyncio.sleep(3)  # Esperar antes de relanzar

                # Relanzar con resume=True
                try:
                    self._process = self._relaunch_with_resume()
                    if self._process:
                        self.current_job.process_pid = self._process.pid
                        self._save_status()
                        continue  # Volver al while para monitorear el nuevo proceso
                except Exception as e:
                    print(f"  Error en auto-resume: {e}")

            # No se pudo resumir o sin páginas
            if self.current_job.pages_crawled > 0:
                self.current_job.status = "completed"
                self.current_job.error_message = None
            else:
                self.current_job.status = "failed"
                self.current_job.error_message = f"Process exited with code {return_code}"
            break

        self.current_job.completed_at = datetime.now().isoformat()
        await self._update_stats()
        self._save_status()
        self._process = None

    def _relaunch_with_resume(self) -> Optional[subprocess.Popen]:
        """Relaunch the crawl script with --resume flag."""
        if not self.current_job:
            return None

        script_path = Path(__file__).parent / "crawl4ai_advanced.py"
        cmd = [
            "python3", str(script_path),
            "--url", self.current_job.url,
            "--max-pages", str(self.current_job.max_pages),
            "--output-dir", str(BASE_OUTPUT_DIR),
            "--delay", str(self.current_job.delay),
            "--resume",  # Siempre resume
        ]

        if not self.current_job.use_sitemap:
            cmd.append("--no-sitemap")
        if not self.current_job.content_filter:
            cmd.append("--no-filter")
        if not self.current_job.respect_robots:
            cmd.append("--no-robots")
        if not self.current_job.skip_noindex:
            cmd.append("--no-skip-noindex")
        if self.current_job.sitemap_only:
            cmd.append("--sitemap-only")
        if self.current_job.exclude_selectors:
            cmd.extend(["--exclude-selectors", ",".join(self.current_job.exclude_selectors)])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(Path(__file__).parent),
        )

    async def _update_stats(self):
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
            except Exception:
                pass
        self.current_job.pages_crawled = total_pages

        links_pattern = str(output_dir / "links" / "**" / "*.parquet")
        links_files = glob(links_pattern, recursive=True)
        total_links = 0
        for lf in links_files:
            try:
                df = pd.read_parquet(lf)
                total_links += len(df)
            except Exception:
                pass
        self.current_job.links_found = total_links

    async def stop_crawl(self) -> Optional[CrawlJob]:
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
        if self.current_job:
            # Check if the subprocess died without updating status
            if self.current_job.status == "running" and self._process is None:
                self._mark_stale_job()
            elif self.current_job.status == "running" and self._process is not None:
                if self._process.poll() is not None:
                    self._mark_stale_job()
            return self.current_job

        if self._status_file.exists():
            try:
                with open(self._status_file) as f:
                    data = json.load(f)
                    job = CrawlJob(**data)
                # If loaded from file and says running, the process is gone (service restarted)
                if job.status == "running":
                    self.current_job = job
                    self._mark_stale_job()
                    return self.current_job
                return job
            except Exception:
                pass

        return None

    def _mark_stale_job(self):
        """Mark a stale running job as completed or failed."""
        if not self.current_job:
            return
        # Update stats one last time
        output_dir = Path(self.current_job.output_dir) if self.current_job.output_dir else None
        if output_dir:
            pages_pattern = str(output_dir / "pages" / "**" / "*.parquet")
            pages_files = glob(pages_pattern, recursive=True)
            total_pages = 0
            for pf in pages_files:
                try:
                    df = pd.read_parquet(pf)
                    total_pages += len(df)
                except Exception:
                    pass
            self.current_job.pages_crawled = total_pages

        if self.current_job.pages_crawled > 0:
            self.current_job.status = "completed"
            self.current_job.error_message = None
        else:
            self.current_job.status = "failed"
            self.current_job.error_message = "Process died unexpectedly"
        self.current_job.completed_at = self.current_job.completed_at or datetime.now().isoformat()
        self._process = None
        self._save_status()

    def _save_status(self):
        if self.current_job:
            with open(self._status_file, "w") as f:
                json.dump(asdict(self.current_job), f)

    def list_available_crawls(self) -> list[dict]:
        crawls = []
        if not BASE_OUTPUT_DIR.exists():
            return crawls

        for d in BASE_OUTPUT_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                pages_files = list((d / "pages").glob("**/*.parquet")) if (d / "pages").exists() else []
                links_files = list((d / "links").glob("**/*.parquet")) if (d / "links").exists() else []

                if pages_files:
                    total_pages = 0
                    for pf in pages_files:
                        try:
                            df = pd.read_parquet(pf)
                            total_pages += len(df)
                        except Exception:
                            pass

                    total_links = 0
                    for lf in links_files:
                        try:
                            df = pd.read_parquet(lf)
                            total_links += len(df)
                        except Exception:
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
_manager = CrawlerManager()


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "ok", "service": "crawler"}


@app.post("/crawl/start")
async def start_crawl(request: dict):
    """Start a new crawl job. Accepts the same parameters as the dashboard endpoint."""
    url = request.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    try:
        job = await _manager.start_crawl(request)
        return asdict(job)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/crawl/stop")
async def stop_crawl():
    """Stop the current crawl job."""
    job = await _manager.stop_crawl()
    if job:
        return asdict(job)
    return {"status": "no_active_crawl"}


@app.get("/crawl/status")
async def crawl_status():
    """Get current crawl status."""
    job = _manager.get_status()
    if job:
        return asdict(job)
    return {"status": "idle"}


@app.get("/crawl/logs")
async def crawl_logs(last_n: int = 100):
    """Get crawler logs."""
    job = _manager.get_status()
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


@app.get("/crawl/crawls")
async def list_crawls():
    """List available crawl data directories."""
    return {"crawls": _manager.list_available_crawls()}
