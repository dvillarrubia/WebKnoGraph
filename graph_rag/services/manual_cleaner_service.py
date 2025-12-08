"""
Manual Cleaner Service - Human-controlled markdown cleaning.

Flow:
1. Cluster pages by HTML fingerprint (template detection)
2. Show samples for each template
3. User manually defines patterns to remove
4. Apply patterns to all pages of that template
5. Preview and confirm before saving

The user controls everything - no automatic LLM cleaning.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
from glob import glob

from bs4 import BeautifulSoup
import pandas as pd


# =============================================================================
# HTML FINGERPRINTING (from cleaning_service.py)
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
# DATA CLASSES
# =============================================================================

@dataclass
class CleaningPattern:
    """A pattern to remove from markdown."""
    id: str
    pattern_type: str  # "exact", "prefix", "contains", "regex", "line_range"
    value: str  # The pattern value
    description: str = ""
    created_at: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class TemplateInfo:
    """Information about a detected template."""
    template_id: str
    fingerprint: str  # The HTML fingerprint
    page_count: int
    sample_urls: List[str]
    patterns: List[CleaningPattern] = field(default_factory=list)
    is_cleaned: bool = False


# =============================================================================
# MANUAL CLEANER SERVICE
# =============================================================================

class ManualCleanerService:
    """Service for human-controlled markdown cleaning."""

    def __init__(self, data_dir: str = "./data/crawl4ai_data"):
        self.data_dir = Path(data_dir)
        self._templates_cache: Dict[str, Dict[str, TemplateInfo]] = {}  # crawl_name -> {template_id -> TemplateInfo}

    def _get_crawl_path(self, crawl_name: str) -> Path:
        return self.data_dir / crawl_name

    def _get_patterns_file(self, crawl_name: str) -> Path:
        return self._get_crawl_path(crawl_name) / "manual_patterns.json"

    def _load_pages(self, crawl_name: str) -> pd.DataFrame:
        """Load all pages from a crawl."""
        crawl_path = self._get_crawl_path(crawl_name)
        pages_dir = crawl_path / "pages"

        if not pages_dir.exists():
            raise ValueError(f"No pages found for crawl: {crawl_name}")

        dfs = []
        for pq_file in pages_dir.rglob("*.parquet"):
            try:
                df = pd.read_parquet(pq_file)
                dfs.append(df)
            except Exception as e:
                print(f"[ManualCleaner] Error reading {pq_file}: {e}")

        if not dfs:
            raise ValueError(f"No valid parquet files in crawl: {crawl_name}")

        return pd.concat(dfs, ignore_index=True).drop_duplicates(subset="url")

    def analyze_templates(self, crawl_name: str) -> Dict[str, Any]:
        """
        Analyze a crawl and detect templates using HTML fingerprinting.

        Returns:
            Dict with template info and statistics.
        """
        df = self._load_pages(crawl_name)

        # Check if we have html_content
        if "html_content" not in df.columns:
            raise ValueError("Crawl doesn't have html_content - re-crawl needed")

        # Calculate fingerprints
        print(f"[ManualCleaner] Analyzing {len(df)} pages...")
        df["fingerprint"] = df["html_content"].apply(extract_html_fingerprint)

        # Group by fingerprint
        templates = {}
        for fingerprint, group in df.groupby("fingerprint"):
            template_id = f"tpl_{fingerprint}"
            sample_urls = group["url"].head(5).tolist()

            templates[template_id] = TemplateInfo(
                template_id=template_id,
                fingerprint=fingerprint,
                page_count=len(group),
                sample_urls=sample_urls,
                patterns=[],
                is_cleaned=False,
            )

        # Load existing patterns if any
        patterns_file = self._get_patterns_file(crawl_name)
        if patterns_file.exists():
            with open(patterns_file) as f:
                saved = json.load(f)
                for tid, data in saved.get("templates", {}).items():
                    if tid in templates:
                        templates[tid].patterns = [
                            CleaningPattern(**p) for p in data.get("patterns", [])
                        ]
                        templates[tid].is_cleaned = data.get("is_cleaned", False)

        # Cache
        self._templates_cache[crawl_name] = templates

        # Return summary
        return {
            "crawl_name": crawl_name,
            "total_pages": len(df),
            "templates_count": len(templates),
            "templates": [
                {
                    "template_id": t.template_id,
                    "fingerprint": t.fingerprint,
                    "page_count": t.page_count,
                    "sample_urls": t.sample_urls[:3],
                    "patterns_count": len(t.patterns),
                    "is_cleaned": t.is_cleaned,
                }
                for t in sorted(templates.values(), key=lambda x: -x.page_count)
            ]
        }

    def get_template_sample(self, crawl_name: str, template_id: str, sample_index: int = 0) -> Dict[str, Any]:
        """
        Get a sample page from a template for manual inspection.

        Returns:
            Dict with URL, markdown content, and current patterns.
        """
        # Ensure templates are loaded
        if crawl_name not in self._templates_cache:
            self.analyze_templates(crawl_name)

        templates = self._templates_cache.get(crawl_name, {})
        template = templates.get(template_id)

        if not template:
            raise ValueError(f"Template not found: {template_id}")

        if sample_index >= len(template.sample_urls):
            sample_index = 0

        url = template.sample_urls[sample_index]

        # Load the page
        df = self._load_pages(crawl_name)
        page = df[df["url"] == url]

        if page.empty:
            raise ValueError(f"Page not found: {url}")

        row = page.iloc[0]
        markdown = row.get("markdown", "")

        # Apply current patterns to show preview
        clean_markdown = self._apply_patterns(markdown, template.patterns)

        return {
            "template_id": template_id,
            "sample_index": sample_index,
            "total_samples": len(template.sample_urls),
            "url": url,
            "title": row.get("title", ""),
            "markdown_original": markdown,
            "markdown_preview": clean_markdown,
            "original_length": len(markdown),
            "preview_length": len(clean_markdown),
            "reduction_pct": round((1 - len(clean_markdown) / len(markdown)) * 100, 1) if markdown else 0,
            "patterns": [p.to_dict() for p in template.patterns],
            "page_count": template.page_count,
        }

    def add_pattern(
        self,
        crawl_name: str,
        template_id: str,
        pattern_type: str,
        value: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Add a cleaning pattern to a template.

        pattern_type options:
        - "exact": Remove exact text match
        - "prefix": Remove lines starting with this text
        - "contains": Remove lines containing this text
        - "regex": Remove using regex pattern
        - "line_range": Remove lines from start to end (value = "start_text|||end_text")
        """
        if crawl_name not in self._templates_cache:
            self.analyze_templates(crawl_name)

        templates = self._templates_cache.get(crawl_name, {})
        template = templates.get(template_id)

        if not template:
            raise ValueError(f"Template not found: {template_id}")

        # Create pattern
        pattern = CleaningPattern(
            id=f"pat_{datetime.now().strftime('%H%M%S')}_{len(template.patterns)}",
            pattern_type=pattern_type,
            value=value,
            description=description,
            created_at=datetime.now().isoformat(),
        )

        template.patterns.append(pattern)

        # Save
        self._save_patterns(crawl_name)

        return {"status": "ok", "pattern": pattern.to_dict()}

    def remove_pattern(self, crawl_name: str, template_id: str, pattern_id: str) -> Dict[str, Any]:
        """Remove a pattern from a template."""
        if crawl_name not in self._templates_cache:
            self.analyze_templates(crawl_name)

        templates = self._templates_cache.get(crawl_name, {})
        template = templates.get(template_id)

        if not template:
            raise ValueError(f"Template not found: {template_id}")

        template.patterns = [p for p in template.patterns if p.id != pattern_id]
        self._save_patterns(crawl_name)

        return {"status": "ok", "remaining_patterns": len(template.patterns)}

    def _apply_patterns(self, markdown: str, patterns: List[CleaningPattern]) -> str:
        """Apply cleaning patterns to markdown."""
        result = markdown

        for pattern in patterns:
            try:
                if pattern.pattern_type == "exact":
                    result = result.replace(pattern.value, "")

                elif pattern.pattern_type == "prefix":
                    lines = result.split('\n')
                    result = '\n'.join(
                        line for line in lines
                        if not line.strip().startswith(pattern.value)
                    )

                elif pattern.pattern_type == "contains":
                    lines = result.split('\n')
                    result = '\n'.join(
                        line for line in lines
                        if pattern.value not in line
                    )

                elif pattern.pattern_type == "regex":
                    result = re.sub(pattern.value, "", result, flags=re.MULTILINE)

                elif pattern.pattern_type == "line_range":
                    # Format: "start_text|||end_text"
                    parts = pattern.value.split("|||")
                    if len(parts) == 2:
                        start_text, end_text = parts
                        lines = result.split('\n')
                        new_lines = []
                        skipping = False
                        for line in lines:
                            if start_text in line:
                                skipping = True
                                continue
                            if skipping and end_text in line:
                                skipping = False
                                continue
                            if not skipping:
                                new_lines.append(line)
                        result = '\n'.join(new_lines)

            except Exception as e:
                print(f"[ManualCleaner] Pattern error ({pattern.id}): {e}")
                continue

        # Clean up excessive whitespace
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip()

    def preview_cleaning(self, crawl_name: str, template_id: str) -> Dict[str, Any]:
        """
        Preview cleaning results for all pages in a template.
        Shows statistics without actually saving.
        """
        if crawl_name not in self._templates_cache:
            self.analyze_templates(crawl_name)

        templates = self._templates_cache.get(crawl_name, {})
        template = templates.get(template_id)

        if not template:
            raise ValueError(f"Template not found: {template_id}")

        df = self._load_pages(crawl_name)
        df["fingerprint"] = df["html_content"].apply(extract_html_fingerprint)

        # Get pages for this template
        template_pages = df[df["fingerprint"] == template.fingerprint]

        results = []
        for _, row in template_pages.iterrows():
            markdown = row.get("markdown", "")
            clean = self._apply_patterns(markdown, template.patterns)

            results.append({
                "url": row["url"],
                "original_length": len(markdown),
                "clean_length": len(clean),
                "reduction_pct": round((1 - len(clean) / len(markdown)) * 100, 1) if markdown else 0,
            })

        avg_reduction = sum(r["reduction_pct"] for r in results) / len(results) if results else 0

        return {
            "template_id": template_id,
            "page_count": len(results),
            "patterns_count": len(template.patterns),
            "avg_reduction_pct": round(avg_reduction, 1),
            "pages": results[:20],  # First 20 for preview
        }

    def apply_cleaning(self, crawl_name: str, template_id: str) -> Dict[str, Any]:
        """
        Apply cleaning patterns to all pages in a template and save results.
        """
        if crawl_name not in self._templates_cache:
            self.analyze_templates(crawl_name)

        templates = self._templates_cache.get(crawl_name, {})
        template = templates.get(template_id)

        if not template:
            raise ValueError(f"Template not found: {template_id}")

        df = self._load_pages(crawl_name)
        df["fingerprint"] = df["html_content"].apply(extract_html_fingerprint)

        # Get pages for this template
        template_pages = df[df["fingerprint"] == template.fingerprint].copy()

        # Apply patterns
        results = []
        for idx, row in template_pages.iterrows():
            markdown = row.get("markdown", "")
            clean = self._apply_patterns(markdown, template.patterns)

            results.append({
                "url": row["url"],
                "title": row.get("title", ""),
                "template_id": template_id,
                "markdown_original": markdown,
                "markdown_clean": clean,
                "original_length": len(markdown),
                "clean_length": len(clean),
                "patterns_applied": len(template.patterns),
            })

        # Save results
        output_dir = self._get_crawl_path(crawl_name) / "manual_clean"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{template_id}.parquet"
        pd.DataFrame(results).to_parquet(output_file, engine="pyarrow")

        # Mark as cleaned
        template.is_cleaned = True
        self._save_patterns(crawl_name)

        return {
            "status": "ok",
            "template_id": template_id,
            "pages_cleaned": len(results),
            "output_file": str(output_file),
        }

    def _save_patterns(self, crawl_name: str):
        """Save patterns to disk."""
        templates = self._templates_cache.get(crawl_name, {})

        data = {
            "updated_at": datetime.now().isoformat(),
            "templates": {
                tid: {
                    "patterns": [p.to_dict() for p in t.patterns],
                    "is_cleaned": t.is_cleaned,
                }
                for tid, t in templates.items()
            }
        }

        patterns_file = self._get_patterns_file(crawl_name)
        with open(patterns_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_cleaning_status(self, crawl_name: str) -> Dict[str, Any]:
        """Get overall cleaning status for a crawl."""
        if crawl_name not in self._templates_cache:
            self.analyze_templates(crawl_name)

        templates = self._templates_cache.get(crawl_name, {})

        total_pages = sum(t.page_count for t in templates.values())
        cleaned_pages = sum(t.page_count for t in templates.values() if t.is_cleaned)

        return {
            "crawl_name": crawl_name,
            "total_templates": len(templates),
            "cleaned_templates": sum(1 for t in templates.values() if t.is_cleaned),
            "total_pages": total_pages,
            "cleaned_pages": cleaned_pages,
            "progress_pct": round(cleaned_pages / total_pages * 100, 1) if total_pages else 0,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_manual_cleaner_service: Optional[ManualCleanerService] = None


def get_manual_cleaner_service() -> ManualCleanerService:
    """Get or create the manual cleaner service singleton."""
    global _manual_cleaner_service
    if _manual_cleaner_service is None:
        _manual_cleaner_service = ManualCleanerService()
    return _manual_cleaner_service
