"""
JSON-LD / Schema.org Extractor (Sprint 2).
Extracts structured data from html_content in parquet files.
"""

import json
import re
from pathlib import Path

import pandas as pd


def extract_jsonld_from_html(html: str) -> list[dict]:
    """
    Extract all JSON-LD blocks from HTML content.

    Returns:
        List of parsed JSON-LD objects.
    """
    if not html:
        return []

    results = []
    pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
    matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)

    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
        except json.JSONDecodeError:
            continue

    return results


def extract_publishing_date(jsonld_list: list[dict]) -> str | None:
    """Extract datePublished from JSON-LD data."""
    for item in jsonld_list:
        for key in ("datePublished", "dateCreated", "dateModified"):
            if key in item:
                return str(item[key])
        # Check @graph
        if "@graph" in item:
            for node in item["@graph"]:
                for key in ("datePublished", "dateCreated", "dateModified"):
                    if key in node:
                        return str(node[key])
    return None


def extract_entities(jsonld_list: list[dict]) -> dict:
    """
    Extract 'about' and 'mentions' entities from JSON-LD.

    Returns:
        {"about": [...], "mentions": [...]} with entity dicts.
    """
    about = []
    mentions = []

    for item in jsonld_list:
        nodes = [item]
        if "@graph" in item:
            nodes = item["@graph"]

        for node in nodes:
            # Extract 'about' entity
            if "about" in node:
                about_data = node["about"]
                if isinstance(about_data, list):
                    about.extend(_normalize_entities(about_data))
                else:
                    about.extend(_normalize_entities([about_data]))

            # Extract 'mentions' entities
            if "mentions" in node:
                mentions_data = node["mentions"]
                if isinstance(mentions_data, list):
                    mentions.extend(_normalize_entities(mentions_data))
                else:
                    mentions.extend(_normalize_entities([mentions_data]))

    return {"about": about, "mentions": mentions}


def _normalize_entities(entities: list) -> list[dict]:
    """Normalize entity data to consistent format."""
    result = []
    for e in entities:
        if isinstance(e, str):
            result.append({"@type": "Thing", "name": e})
        elif isinstance(e, dict):
            result.append({
                "@type": e.get("@type", "Thing"),
                "name": e.get("name", e.get("@id", "")),
                "@id": e.get("@id", ""),
                "url": e.get("url", ""),
            })
    return result


def extract_meta_title(html: str) -> str | None:
    """Extract <title> tag from HTML."""
    if not html:
        return None
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_schemas_from_parquet(pages_dir: Path) -> dict:
    """
    Extract JSON-LD from all pages in a parquet directory.

    Returns:
        {
            "pages_processed": int,
            "schemas_found": int,
            "schemas": {url: [jsonld_objects]},
            "dates": {url: datePublished},
            "entities": {url: {"about": [...], "mentions": [...]}},
            "meta_titles": {url: title_tag},
        }
    """
    if not pages_dir.exists():
        return {"pages_processed": 0, "schemas_found": 0}

    parquet_files = list(pages_dir.rglob("*.parquet"))
    if not parquet_files:
        return {"pages_processed": 0, "schemas_found": 0}

    schemas = {}
    dates = {}
    entities = {}
    meta_titles = {}
    total_schemas = 0

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        if "html_content" not in df.columns:
            continue

        for _, row in df.iterrows():
            url = str(row.get("url", ""))
            html = str(row.get("html_content", ""))

            if not url or not html:
                continue

            # Extract JSON-LD
            jsonld_list = extract_jsonld_from_html(html)
            if jsonld_list:
                schemas[url] = jsonld_list
                total_schemas += len(jsonld_list)

                # Extract date
                date = extract_publishing_date(jsonld_list)
                if date:
                    dates[url] = date

                # Extract entities
                ents = extract_entities(jsonld_list)
                if ents["about"] or ents["mentions"]:
                    entities[url] = ents

            # Extract meta title
            title = extract_meta_title(html)
            if title:
                meta_titles[url] = title

    pages_processed = sum(len(pd.read_parquet(f)) for f in parquet_files)

    return {
        "pages_processed": pages_processed,
        "schemas_found": total_schemas,
        "schemas": schemas,
        "dates": dates,
        "entities": entities,
        "meta_titles": meta_titles,
    }
