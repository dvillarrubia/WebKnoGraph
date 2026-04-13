"""
Google Search Console Service — Imports GSC data into Neo4j as :SeoQuery nodes.

Flow:
1. Read existing :SeoWebPage URLs from Neo4j
2. Download GSC data (query × page) for 3 time windows (7d, 28d, 3m)
3. Filter to only pages that exist in Neo4j
4. Create/update :SeoQuery nodes with metrics
5. Create HAS_QUERY and HAS_PRIMARY_QUERY relationships

Batched and incremental: only processes URLs already in the graph.
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from neo4j import AsyncDriver

from graph_rag.db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]

# Time windows as defined by seovoc
WINDOWS = [
    ("7d", 7),
    ("28d", 28),
    ("3m", 90),
]

# GSC API max rows per request
GSC_ROW_LIMIT = 25000


@dataclass
class GscSyncResult:
    client_id: str
    site_url: str
    queries_created: int = 0
    queries_updated: int = 0
    has_query_rels: int = 0
    primary_query_rels: int = 0
    pages_matched: int = 0
    pages_total_gsc: int = 0
    windows_synced: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def _build_gsc_service(credentials_json_path: str):
    """Build authenticated GSC API service from a service account JSON file."""
    credentials = service_account.Credentials.from_service_account_file(
        credentials_json_path, scopes=SCOPES
    )
    return build("searchconsole", "v1", credentials=credentials, cache_discovery=False)


def _fetch_gsc_window(
    service,
    site_url: str,
    days: int,
    url_filter: Optional[str] = None,
) -> list[dict]:
    """
    Fetch query×page data from GSC for a single time window.
    Paginates through all results using startRow.
    Optionally filters by URL prefix.
    """
    end_date = date.today() - timedelta(days=3)  # GSC data has ~3 day lag
    start_date = end_date - timedelta(days=days)

    all_rows = []
    start_row = 0

    while True:
        body = {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "dimensions": ["query", "page"],
            "rowLimit": GSC_ROW_LIMIT,
            "startRow": start_row,
            "dataState": "final",
        }

        if url_filter:
            body["dimensionFilterGroups"] = [
                {
                    "filters": [
                        {
                            "dimension": "page",
                            "operator": "contains",
                            "expression": url_filter,
                        }
                    ]
                }
            ]

        response = service.searchanalytics().query(
            siteUrl=site_url, body=body
        ).execute()

        rows = response.get("rows", [])
        if not rows:
            break

        all_rows.extend(rows)
        start_row += len(rows)

        # If we got less than the limit, we've exhausted the data
        if len(rows) < GSC_ROW_LIMIT:
            break

    return all_rows


def _aggregate_windows(
    service,
    site_url: str,
    url_filter: Optional[str] = None,
) -> dict:
    """
    Fetch all 3 windows and aggregate into a dict keyed by (query, page).

    Returns:
        {
            ("query_text", "page_url"): {
                "clicks7Days": int, "impressions7Days": int, "ctr7Days": float, "position7Days": float,
                "clicks28Days": int, ..., "clicks3Months": int, ...
                "clicks7DaysTo28DaysTrend": float, "clicks28DaysTo3MonthsTrend": float, ...
            }
        }
    """
    suffix_map = {"7d": "7Days", "28d": "28Days", "3m": "3Months"}
    aggregated = {}

    for label, days in WINDOWS:
        suffix = suffix_map[label]
        rows = _fetch_gsc_window(service, site_url, days, url_filter)
        logger.info(f"GSC window {label}: {len(rows)} rows fetched")

        for row in rows:
            query_text = row["keys"][0]
            page_url = row["keys"][1]
            key = (query_text, page_url)

            if key not in aggregated:
                aggregated[key] = {"query": query_text, "page": page_url}

            aggregated[key][f"clicks{suffix}"] = int(row["clicks"])
            aggregated[key][f"impressions{suffix}"] = int(row["impressions"])
            aggregated[key][f"ctr{suffix}"] = round(row["ctr"], 6)
            aggregated[key][f"position{suffix}"] = round(row["position"], 2)

    # Calculate trends for entries that have multiple windows
    for data in aggregated.values():
        for metric in ("clicks", "impressions", "ctr", "position"):
            v7 = data.get(f"{metric}7Days")
            v28 = data.get(f"{metric}28Days")
            v3m = data.get(f"{metric}3Months")

            # Trend = (recent - older) / older, or 0 if older is 0
            if v7 is not None and v28 is not None:
                data[f"{metric}7DaysTo28DaysTrend"] = (
                    round((v7 - v28) / v28, 4) if v28 != 0 else 0.0
                )
            if v28 is not None and v3m is not None:
                data[f"{metric}28DaysTo3MonthsTrend"] = (
                    round((v28 - v3m) / v3m, 4) if v3m != 0 else 0.0
                )

    return aggregated


async def _get_existing_urls(driver: AsyncDriver, client_id: str) -> set[str]:
    """Get all :SeoWebPage URLs for a client from Neo4j."""
    async with driver.session() as session:
        result = await session.run(
            "MATCH (p:SeoWebPage {client_id: $client_id}) RETURN p.url AS url",
            client_id=client_id,
        )
        records = await result.data()
        return {r["url"] for r in records}


async def _cleanup_gsc_data(driver: AsyncDriver, client_id: str) -> int:
    """Remove existing :SeoQuery nodes and HAS_QUERY/HAS_PRIMARY_QUERY rels."""
    async with driver.session() as session:
        result = await session.run(
            "MATCH (q:SeoQuery {client_id: $client_id}) DETACH DELETE q RETURN count(q) AS cnt",
            client_id=client_id,
        )
        record = await result.single()
        return record["cnt"] if record else 0


async def _ingest_queries_batch(
    driver: AsyncDriver,
    client_id: str,
    batch: list[dict],
) -> int:
    """
    MERGE :SeoQuery nodes and create HAS_QUERY relationships.
    Uses UNWIND for efficient batch processing.
    """
    async with driver.session() as session:
        result = await session.run(
            """
            UNWIND $batch AS item
            MERGE (q:SeoQuery {client_id: $client_id, queryText: item.query, page_url: item.page})
            SET q.clicks7Days = item.clicks7Days,
                q.clicks28Days = item.clicks28Days,
                q.clicks3Months = item.clicks3Months,
                q.impressions7Days = item.impressions7Days,
                q.impressions28Days = item.impressions28Days,
                q.impressions3Months = item.impressions3Months,
                q.ctr7Days = item.ctr7Days,
                q.ctr28Days = item.ctr28Days,
                q.ctr3Months = item.ctr3Months,
                q.position7Days = item.position7Days,
                q.position28Days = item.position28Days,
                q.position3Months = item.position3Months,
                q.clicks7DaysTo28DaysTrend = item.clicks7DaysTo28DaysTrend,
                q.clicks28DaysTo3MonthsTrend = item.clicks28DaysTo3MonthsTrend,
                q.impressions7DaysTo28DaysTrend = item.impressions7DaysTo28DaysTrend,
                q.impressions28DaysTo3MonthsTrend = item.impressions28DaysTo3MonthsTrend,
                q.ctr7DaysTo28DaysTrend = item.ctr7DaysTo28DaysTrend,
                q.ctr28DaysTo3MonthsTrend = item.ctr28DaysTo3MonthsTrend,
                q.position7DaysTo28DaysTrend = item.position7DaysTo28DaysTrend,
                q.position28DaysTo3MonthsTrend = item.position28DaysTo3MonthsTrend,
                q.inLanguage = item.inLanguage,
                q.dateCreated = CASE WHEN q.dateCreated IS NULL THEN datetime() ELSE q.dateCreated END,
                q.updated_at = datetime()
            WITH q, item
            MATCH (p:SeoWebPage {client_id: $client_id, url: item.page})
            MERGE (p)-[:HAS_QUERY]->(q)
            RETURN count(q) AS cnt
            """,
            client_id=client_id,
            batch=batch,
        )
        record = await result.single()
        return record["cnt"] if record else 0


async def _set_primary_queries(driver: AsyncDriver, client_id: str) -> int:
    """
    For each :SeoWebPage, find the :SeoQuery with highest clicks28Days
    and create a HAS_PRIMARY_QUERY relationship.
    """
    async with driver.session() as session:
        # Remove existing primary query rels first
        await session.run(
            """
            MATCH (p:SeoWebPage {client_id: $client_id})-[r:HAS_PRIMARY_QUERY]->()
            DELETE r
            """,
            client_id=client_id,
        )
        # Set new primary queries
        result = await session.run(
            """
            MATCH (p:SeoWebPage {client_id: $client_id})-[:HAS_QUERY]->(q:SeoQuery)
            WITH p, q ORDER BY COALESCE(q.clicks28Days, 0) DESC
            WITH p, collect(q)[0] AS primary_q
            WHERE primary_q IS NOT NULL
            MERGE (p)-[:HAS_PRIMARY_QUERY]->(primary_q)
            RETURN count(p) AS cnt
            """,
            client_id=client_id,
        )
        record = await result.single()
        return record["cnt"] if record else 0


async def _update_page_gsc_metrics(driver: AsyncDriver, client_id: str) -> int:
    """
    Aggregate GSC metrics from :SeoQuery nodes back to :SeoWebPage.
    Sets clicks, impressions, ctr, position on the page level (28d window).
    """
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (p:SeoWebPage {client_id: $client_id})-[:HAS_QUERY]->(q:SeoQuery)
            WHERE q.clicks28Days IS NOT NULL
            WITH p,
                 sum(q.clicks28Days) AS totalClicks,
                 sum(q.impressions28Days) AS totalImpressions,
                 avg(q.position28Days) AS avgPosition
            SET p.clicks = totalClicks,
                p.impressions = totalImpressions,
                p.ctr = CASE WHEN totalImpressions > 0
                         THEN toFloat(totalClicks) / totalImpressions
                         ELSE 0.0 END,
                p.position = round(avgPosition, 2)
            RETURN count(p) AS cnt
            """,
            client_id=client_id,
        )
        record = await result.single()
        return record["cnt"] if record else 0


async def run_gsc_sync(
    neo4j: Neo4jClient,
    client_id: str,
    site_url: str,
    credentials_path: str,
    url_filter: Optional[str] = None,
    refresh: bool = False,
    batch_size: int = 500,
    language: str = "es",
) -> dict:
    """
    Main GSC sync entrypoint.

    Args:
        neo4j: Neo4jClient instance
        client_id: Client UUID
        site_url: GSC property (e.g. 'sc-domain:uoc.edu')
        credentials_path: Path to service account JSON
        url_filter: Optional URL prefix filter for GSC API (e.g. 'www.uoc.edu')
        refresh: If True, delete existing :SeoQuery nodes before sync
        batch_size: Number of queries to process per Neo4j batch
        language: Default language for queries (default: 'es')
    """
    driver = neo4j._driver
    if driver is None:
        await neo4j.connect()
        driver = neo4j._driver

    result = GscSyncResult(client_id=client_id, site_url=site_url)

    try:
        # 1. Get existing URLs from Neo4j
        existing_urls = await _get_existing_urls(driver, client_id)
        logger.info(f"Found {len(existing_urls)} URLs in Neo4j")

        if not existing_urls:
            result.errors.append("No :SeoWebPage URLs found in Neo4j. Run ingest first.")
            return asdict(result)

        # 2. Cleanup if refresh
        if refresh:
            cleaned = await _cleanup_gsc_data(driver, client_id)
            logger.info(f"Cleaned {cleaned} existing SeoQuery nodes")

        # 3. Fetch GSC data (all 3 windows, aggregated)
        gsc_service = _build_gsc_service(credentials_path)
        aggregated = _aggregate_windows(gsc_service, site_url, url_filter)
        logger.info(f"GSC total: {len(aggregated)} query×page combinations")

        all_gsc_pages = {data["page"] for data in aggregated.values()}
        result.pages_total_gsc = len(all_gsc_pages)

        # 4. Filter to only pages that exist in Neo4j
        matched_data = [
            data for data in aggregated.values()
            if data["page"] in existing_urls
        ]
        matched_pages = {d["page"] for d in matched_data}
        result.pages_matched = len(matched_pages)

        logger.info(
            f"Matched {len(matched_data)} rows ({len(matched_pages)} pages) "
            f"out of {len(aggregated)} total GSC rows"
        )

        # 5. Batch ingest into Neo4j
        for i in range(0, len(matched_data), batch_size):
            batch = matched_data[i : i + batch_size]

            # Add language to each item and ensure all fields have defaults
            for item in batch:
                item["inLanguage"] = language
                for key in (
                    "clicks7Days", "clicks28Days", "clicks3Months",
                    "impressions7Days", "impressions28Days", "impressions3Months",
                    "ctr7Days", "ctr28Days", "ctr3Months",
                    "position7Days", "position28Days", "position3Months",
                    "clicks7DaysTo28DaysTrend", "clicks28DaysTo3MonthsTrend",
                    "impressions7DaysTo28DaysTrend", "impressions28DaysTo3MonthsTrend",
                    "ctr7DaysTo28DaysTrend", "ctr28DaysTo3MonthsTrend",
                    "position7DaysTo28DaysTrend", "position28DaysTo3MonthsTrend",
                ):
                    item.setdefault(key, None)

            cnt = await _ingest_queries_batch(driver, client_id, batch)
            result.queries_created += cnt
            result.has_query_rels += cnt
            logger.info(f"Batch {i // batch_size + 1}: {cnt} queries ingested")

        # 6. Set primary queries per page
        result.primary_query_rels = await _set_primary_queries(driver, client_id)
        logger.info(f"Primary queries set: {result.primary_query_rels}")

        # 7. Aggregate metrics back to pages
        pages_updated = await _update_page_gsc_metrics(driver, client_id)
        logger.info(f"Page metrics updated: {pages_updated}")

        result.windows_synced = [w[0] for w in WINDOWS]

    except Exception as e:
        logger.error(f"GSC sync error: {e}", exc_info=True)
        result.errors.append(str(e))

    return asdict(result)
