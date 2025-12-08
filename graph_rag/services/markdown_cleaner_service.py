"""
Markdown Cleaner Service - Template-based cleaning with LLM pattern generation.

Strategy:
1. Use raw_markdown from Crawl4AI (complete content, cookies already removed by CSS selectors)
2. Cluster pages by URL patterns into templates
3. For each template, use LLM to identify boilerplate text patterns
4. Apply patterns to all pages in template without additional LLM calls

This approach:
- Reduces LLM calls: 1 call per template instead of 1 per page
- Works with markdown (lighter than HTML)
- Generates reusable patterns that can be cached
"""

import os
import re
import hashlib
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, List, Any
from glob import glob
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime

# Load .env for OPENAI_API_KEY
from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

import pandas as pd
from tqdm import tqdm


@dataclass
class CleaningPattern:
    """Pattern to remove from markdown."""
    pattern_type: str  # "exact", "prefix", "contains", "regex", "section"
    pattern: str
    description: str


@dataclass
class TemplatePatterns:
    """Patterns for a specific template."""
    template_id: str
    url_pattern: str
    sample_urls: list
    patterns: list  # List of CleaningPattern
    main_content_start: str = ""  # Where main content begins
    main_content_end: str = ""  # Where main content ends
    created_at: str = ""


@dataclass
class MarkdownCleaningJob:
    """Represents a markdown cleaning job."""
    id: str
    crawl_name: str
    status: str  # pending, running, analyzing, cleaning, completed, failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_pages: int = 0
    pages_cleaned: int = 0
    templates_found: int = 0
    templates_analyzed: int = 0
    patterns_generated: Dict[str, int] = field(default_factory=dict)
    error_message: Optional[str] = None
    output_path: Optional[str] = None


class MarkdownCleanerService:
    """Service to clean markdown using template-based LLM patterns."""

    def __init__(self, data_dir: str = "./data/crawl4ai_data"):
        self.data_dir = Path(data_dir)
        self.patterns_cache: Dict[str, TemplatePatterns] = {}
        self.current_job: Optional[MarkdownCleaningJob] = None
        self._openai_client = None
        self._status_file: Optional[Path] = None

    @property
    def openai_client(self):
        """Lazy initialization of OpenAI client."""
        if self._openai_client is None:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def _save_status(self):
        """Save current job status to file."""
        if self.current_job and self._status_file:
            with open(self._status_file, "w") as f:
                json.dump(asdict(self.current_job), f, indent=2, default=str)

    def get_status(self) -> Optional[Dict]:
        """Get current job status."""
        if self.current_job:
            return asdict(self.current_job)
        return None

    def _get_url_template(self, url: str) -> str:
        """
        Extract template pattern from URL.

        Strategy: Group by first path segment (section).
        - /blog/cualquier-cosa -> /blog/*
        - /fp-madrid/curso-xyz-123 -> /fp-*/*
        - /institucional/algo -> /institucional/*
        - / -> /

        This groups pages by section, which typically share the same layout/template.
        """
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.strip('/')

        if not path:
            return '/'

        parts = path.split('/')

        # First part is the section
        section = parts[0]

        # Handle fp-{city} pattern -> normalize to fp-*
        if section.startswith('fp-'):
            section = 'fp-*'

        # If there's only one part, it's a top-level page
        if len(parts) == 1:
            # Could be /blog or /contacto
            return f'/{section}'

        # For nested paths, use section/* pattern
        return f'/{section}/*'

    def cluster_urls_by_template(self, urls: list) -> dict:
        """Group URLs by their template pattern."""
        templates = defaultdict(list)
        for url in urls:
            template = self._get_url_template(url)
            templates[template].append(url)
        return dict(templates)

    def load_crawl_data(self, crawl_name: str) -> pd.DataFrame:
        """Load all pages from a crawl."""
        crawl_path = self.data_dir / crawl_name
        if not crawl_path.exists():
            raise ValueError(f"Crawl not found: {crawl_name}")

        pages_files = glob(str(crawl_path / "pages" / "**" / "*.parquet"), recursive=True)
        if not pages_files:
            raise ValueError(f"No pages found in {crawl_name}")

        dfs = [pd.read_parquet(f) for f in pages_files]
        return pd.concat(dfs, ignore_index=True)

    def _generate_patterns_prompt(self, markdown_samples: List[str], urls: List[str], template_pattern: str) -> str:
        """Generate prompt to ask LLM for cleaning patterns based on multiple samples."""
        samples_text = ""
        for i, (md, url) in enumerate(zip(markdown_samples[:3], urls[:3])):
            # Take first 3000 and last 2000 chars (where boilerplate usually is)
            sample = md[:3000] + "\n\n[...MAIN CONTENT...]\n\n" + md[-2000:] if len(md) > 5000 else md
            samples_text += f"\n--- SAMPLE {i+1}: {url} ---\n{sample}\n"

        return f'''Analyze these markdown samples from crawled web pages of the same template type.
Identify BOILERPLATE TEXT patterns that appear across multiple samples and should be removed.

Template pattern: {template_pattern}

{samples_text}

TASK: Find TEXT PATTERNS that are:
1. Navigation menus (links lists at the start)
2. Footer content (contact info, copyright, legal links)
3. Social media sharing blocks
4. Newsletter/subscription prompts
5. Related articles suggestions (if not the main content)
6. Breadcrumb navigation
7. Any repetitive header/footer text

For each pattern found, provide:
- "type": How to match it:
  - "prefix": Remove lines starting with this text
  - "contains": Remove lines containing this text
  - "section": Remove everything from "start" to "end" markers
  - "regex": Use regex pattern (for complex patterns)
- "pattern": The text to match (or start/end for sections)
- "reason": Why this is boilerplate

IMPORTANT RULES:
- Only include patterns that appear in MULTIPLE samples (cross-template patterns)
- NEVER remove the main content - focus on headers and footers
- Be precise - prefer exact matches over broad patterns
- Spanish content is common - handle it correctly

OUTPUT FORMAT (JSON only, no markdown):
{{
    "patterns": [
        {{"type": "prefix", "pattern": "Inicio >", "reason": "Breadcrumb navigation"}},
        {{"type": "contains", "pattern": "Síguenos en", "reason": "Social media links"}},
        {{"type": "section", "pattern": {{"start": "## Artículos relacionados", "end": "## "}}, "reason": "Related articles block"}},
        {{"type": "regex", "pattern": "^\\\\[.*\\\\]\\\\(https://twitter\\\\.com.*\\\\)$", "reason": "Social media link"}}
    ],
    "main_content_markers": {{
        "likely_start": "First distinctive text of main content (e.g., first H1 or intro paragraph)",
        "likely_end": "Last distinctive text before footer boilerplate"
    }}
}}
'''

    def generate_patterns_with_llm(self, markdown_samples: List[str], urls: List[str], template_pattern: str) -> Dict[str, Any]:
        """
        Call OpenAI to generate cleaning patterns for a template.

        Args:
            markdown_samples: List of markdown content samples (3-5 recommended)
            urls: Corresponding URLs for the samples
            template_pattern: The URL template pattern these samples belong to

        Returns:
            Dict with patterns and main_content_markers
        """
        prompt = self._generate_patterns_prompt(markdown_samples, urls, template_pattern)

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective for pattern extraction
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing web content structure. Output ONLY valid JSON, no markdown formatting."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=2000,
            )

            result_text = response.choices[0].message.content.strip()

            # Clean up if wrapped in markdown
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()

            # Parse JSON
            result = json.loads(result_text)
            return result

        except json.JSONDecodeError as e:
            print(f"[MarkdownCleaner] JSON parse error: {e}")
            return {"patterns": [], "error": f"JSON parse error: {str(e)[:100]}"}
        except Exception as e:
            print(f"[MarkdownCleaner] LLM error: {e}")
            return {"patterns": [], "error": str(e)[:200]}

    def apply_patterns(self, markdown: str, patterns: list) -> str:
        """
        Apply cleaning patterns to markdown.

        Patterns can be CleaningPattern objects or dicts from LLM response.
        """
        cleaned = markdown

        for pattern in patterns:
            # Handle both CleaningPattern objects and dicts
            if isinstance(pattern, CleaningPattern):
                ptype = pattern.pattern_type
                pvalue = pattern.pattern
            elif isinstance(pattern, dict):
                ptype = pattern.get("type", "")
                pvalue = pattern.get("pattern", "")
            else:
                continue

            try:
                if ptype == "exact":
                    cleaned = cleaned.replace(pvalue, "")

                elif ptype == "prefix":
                    # Remove lines starting with this text (and following non-empty lines until blank)
                    lines = cleaned.split('\n')
                    cleaned_lines = []
                    skip_until_blank = False
                    for line in lines:
                        if line.strip().startswith(pvalue):
                            skip_until_blank = True
                            continue
                        if skip_until_blank:
                            if line.strip() == "":
                                skip_until_blank = False
                            continue
                        cleaned_lines.append(line)
                    cleaned = '\n'.join(cleaned_lines)

                elif ptype == "contains":
                    # Remove lines containing this text
                    lines = cleaned.split('\n')
                    cleaned = '\n'.join(l for l in lines if pvalue not in l)

                elif ptype == "section":
                    # Remove everything between start and end markers
                    if isinstance(pvalue, dict):
                        start_marker = pvalue.get("start", "")
                        end_marker = pvalue.get("end", "")
                        if start_marker and end_marker:
                            # Find and remove section
                            pattern_regex = re.escape(start_marker) + r'.*?' + re.escape(end_marker)
                            cleaned = re.sub(pattern_regex, end_marker, cleaned, flags=re.DOTALL)

                elif ptype == "regex":
                    if pvalue:
                        cleaned = re.sub(pvalue, "", cleaned, flags=re.MULTILINE)

            except Exception as e:
                # Skip problematic patterns silently
                print(f"[MarkdownCleaner] Pattern error ({ptype}: {str(pvalue)[:30]}): {e}")
                continue

        # Clean up excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = cleaned.strip()

        return cleaned

    def extract_between_markers(self, markdown: str, start_marker: str, end_marker: str) -> str:
        """Extract content between start and end markers."""
        if not start_marker or not end_marker:
            return markdown

        start_idx = markdown.find(start_marker)
        end_idx = markdown.rfind(end_marker)

        if start_idx == -1:
            start_idx = 0
        if end_idx == -1 or end_idx <= start_idx:
            end_idx = len(markdown)
        else:
            end_idx += len(end_marker)

        return markdown[start_idx:end_idx]

    def save_patterns(self, crawl_name: str, patterns_by_template: dict):
        """Save patterns to disk for reuse."""
        patterns_file = self.data_dir / crawl_name / "cleaning_patterns.json"

        serializable = {}
        for template_id, template_patterns in patterns_by_template.items():
            serializable[template_id] = {
                "template_id": template_patterns.template_id,
                "url_pattern": template_patterns.url_pattern,
                "sample_urls": template_patterns.sample_urls,
                "patterns": [
                    {"type": p.pattern_type, "pattern": p.pattern, "description": p.description}
                    for p in template_patterns.patterns
                ],
                "created_at": template_patterns.created_at,
            }

        with open(patterns_file, 'w') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    def load_patterns(self, crawl_name: str) -> dict:
        """Load patterns from disk."""
        patterns_file = self.data_dir / crawl_name / "cleaning_patterns.json"
        if not patterns_file.exists():
            return {}

        with open(patterns_file) as f:
            data = json.load(f)

        result = {}
        for template_id, tdata in data.items():
            patterns = [
                CleaningPattern(p["type"], p["pattern"], p["description"])
                for p in tdata["patterns"]
            ]
            result[template_id] = TemplatePatterns(
                template_id=tdata["template_id"],
                url_pattern=tdata["url_pattern"],
                sample_urls=tdata["sample_urls"],
                patterns=patterns,
                created_at=tdata.get("created_at", ""),
            )

        return result

    def list_crawls(self) -> List[Dict]:
        """List available crawls with markdown data."""
        crawls = []
        for crawl_dir in self.data_dir.iterdir():
            if crawl_dir.is_dir() and not crawl_dir.name.startswith("."):
                pages_dir = crawl_dir / "pages"
                if pages_dir.exists():
                    page_count = 0
                    has_markdown = False
                    for pq_file in pages_dir.rglob("*.parquet"):
                        try:
                            df = pd.read_parquet(pq_file)
                            page_count += len(df)
                            if "markdown" in df.columns:
                                has_markdown = True
                        except:
                            pass

                    if has_markdown:
                        crawls.append({
                            "name": crawl_dir.name,
                            "pages": page_count,
                            "has_markdown_clean": (crawl_dir / "markdown_clean").exists(),
                            "has_patterns": (crawl_dir / "cleaning_patterns.json").exists(),
                        })

        return crawls

    async def start_cleaning(self, crawl_name: str) -> Dict:
        """
        Start a markdown cleaning job using LLM-generated patterns.

        Process:
        1. Load all pages with markdown content
        2. Cluster by URL template
        3. For each template (with >1 page), generate patterns via LLM
        4. Apply patterns to all pages in template
        5. Save cleaned markdown
        """
        crawl_path = self.data_dir / crawl_name
        if not crawl_path.exists():
            raise ValueError(f"Crawl '{crawl_name}' not found")

        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_job = MarkdownCleaningJob(
            id=job_id,
            crawl_name=crawl_name,
            status="running",
            started_at=datetime.now().isoformat(),
        )
        self._status_file = crawl_path / "markdown_cleaning_status.json"
        self._save_status()

        # Run in background thread
        import threading
        thread = threading.Thread(target=self._run_cleaning_sync, args=(crawl_path,))
        thread.daemon = True
        thread.start()

        return asdict(self.current_job)

    def _run_cleaning_sync(self, crawl_path: Path):
        """Synchronous cleaning process (runs in thread)."""
        try:
            # Load all pages
            pages_dir = crawl_path / "pages"
            all_dfs = []
            for pq_file in pages_dir.rglob("*.parquet"):
                try:
                    df = pd.read_parquet(pq_file)
                    if "markdown" in df.columns:
                        all_dfs.append(df)
                except Exception as e:
                    print(f"[MarkdownCleaner] Error reading {pq_file}: {e}")

            if not all_dfs:
                raise ValueError("No pages with markdown content found")

            df = pd.concat(all_dfs, ignore_index=True)
            df = df.drop_duplicates(subset="url")
            self.current_job.total_pages = len(df)
            self._save_status()

            print(f"[MarkdownCleaner] Loaded {len(df)} pages from {crawl_path.name}")

            # Phase 1: Cluster by URL template
            print("[MarkdownCleaner] Phase 1: Clustering pages by URL template...")
            df["template"] = df["url"].apply(self._get_url_template)
            template_groups = df.groupby("template").size().to_dict()

            self.current_job.templates_found = len(template_groups)
            self._save_status()

            print(f"[MarkdownCleaner] Found {len(template_groups)} templates")
            for tid, count in sorted(template_groups.items(), key=lambda x: -x[1])[:10]:
                print(f"  - {tid}: {count} pages")

            # Phase 2: Generate patterns per template
            self.current_job.status = "analyzing"
            self._save_status()

            template_patterns: Dict[str, List[Dict]] = {}
            templates_to_analyze = [t for t, c in template_groups.items() if c >= 2]  # Need 2+ pages

            print(f"\n[MarkdownCleaner] Phase 2: Generating patterns for {len(templates_to_analyze)} templates...")

            for template in tqdm(templates_to_analyze, desc="Analyzing templates"):
                template_df = df[df["template"] == template]
                sample_rows = template_df.head(3)  # Take up to 3 samples

                markdown_samples = sample_rows["markdown"].tolist()
                urls = sample_rows["url"].tolist()

                print(f"\n[MarkdownCleaner] Template: {template} ({len(template_df)} pages)")
                print(f"  Samples: {urls[0][:60]}...")

                # Call LLM
                result = self.generate_patterns_with_llm(markdown_samples, urls, template)

                patterns = result.get("patterns", [])
                if patterns:
                    template_patterns[template] = patterns
                    self.current_job.patterns_generated[template] = len(patterns)
                    print(f"  [OK] Generated {len(patterns)} patterns")
                else:
                    print(f"  [SKIP] No patterns generated")
                    if result.get("error"):
                        print(f"    Error: {result['error']}")

                self.current_job.templates_analyzed += 1
                self._save_status()

            # Save patterns for future use
            patterns_file = crawl_path / "cleaning_patterns.json"
            with open(patterns_file, "w") as f:
                json.dump({
                    "generated_at": datetime.now().isoformat(),
                    "templates": template_patterns,
                }, f, indent=2, ensure_ascii=False)

            # Phase 3: Apply patterns to all pages
            self.current_job.status = "cleaning"
            self._save_status()

            results = []
            print(f"\n[MarkdownCleaner] Phase 3: Cleaning {len(df)} pages...")

            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning"):
                template = row["template"]
                markdown = row.get("markdown", "")
                patterns = template_patterns.get(template, [])

                if patterns:
                    clean_markdown = self.apply_patterns(markdown, patterns)
                    method = "llm_patterns"
                else:
                    clean_markdown = markdown
                    method = "passthrough"

                results.append({
                    "url": row["url"],
                    "title": row.get("title", ""),
                    "markdown_original": markdown,
                    "markdown_clean": clean_markdown,
                    "original_length": len(markdown),
                    "clean_length": len(clean_markdown),
                    "reduction_pct": round((1 - len(clean_markdown) / len(markdown)) * 100, 1) if markdown else 0,
                    "template": template,
                    "cleaning_method": method,
                    "patterns_applied": len(patterns),
                })

                self.current_job.pages_cleaned = len(results)
                if len(results) % 100 == 0:
                    self._save_status()

            # Save results
            output_path = crawl_path / "markdown_clean"
            output_path.mkdir(parents=True, exist_ok=True)

            output_file = output_path / f"clean_{self.current_job.id}.parquet"
            results_df = pd.DataFrame(results)
            results_df.to_parquet(output_file, engine="pyarrow", compression="snappy")

            self.current_job.status = "completed"
            self.current_job.completed_at = datetime.now().isoformat()
            self.current_job.output_path = str(output_file)
            self._save_status()

            print(f"\n[MarkdownCleaner] Completed! Cleaned {len(results)} pages -> {output_file}")

            # Summary stats
            avg_reduction = results_df["reduction_pct"].mean()
            print(f"\n=== Cleaning Stats ===")
            print(f"  Average reduction: {avg_reduction:.1f}%")
            print(f"  Templates with patterns: {len(template_patterns)}")
            print(f"  Total patterns generated: {sum(len(p) for p in template_patterns.values())}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.current_job.status = "failed"
            self.current_job.error_message = str(e)[:500]
            self._save_status()


# =============================================================================
# SINGLETON
# =============================================================================

_markdown_cleaner_service: Optional[MarkdownCleanerService] = None


def get_markdown_cleaner_service() -> MarkdownCleanerService:
    """Get or create the markdown cleaner service singleton."""
    global _markdown_cleaner_service
    if _markdown_cleaner_service is None:
        _markdown_cleaner_service = MarkdownCleanerService()
    return _markdown_cleaner_service
