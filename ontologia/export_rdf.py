"""
RDF/TTL Export for UOC SEOntology data.
Exports fused Page:SeoWebPage nodes from Neo4j to a valid Turtle (.ttl) file.
Includes both seovoc properties AND graph_rag scores (pagerank, hub, authority).
"""

from pathlib import Path

from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, DCTERMS
from neo4j import AsyncDriver


SEOVOC = Namespace("https://w3id.org/seovoc/")
SCHEMA = Namespace("http://schema.org/")
UOC = Namespace("https://www.uoc.edu/seo/")
WKG = Namespace("https://www.uoc.edu/wkg/")  # WebKnoGraph-specific properties


def _safe_uri(url: str) -> URIRef:
    """Convert a URL string to a safe URI reference."""
    return URIRef(url.replace(" ", "%20"))


async def export_to_ttl(
    driver: AsyncDriver,
    client_id: str,
    output_path: str | Path = "uoc_ontology.ttl",
) -> dict:
    """
    Export all SEOntology nodes from Neo4j to Turtle format.

    Args:
        driver: Neo4j async driver
        client_id: Client ID to export
        output_path: Path for the output .ttl file

    Returns:
        Stats dict with counts.
    """
    g = Graph()
    g.bind("seovoc", SEOVOC)
    g.bind("schema", SCHEMA)
    g.bind("uoc", UOC)
    g.bind("wkg", WKG)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)

    stats = {"webpages": 0, "urls": 0, "chunks": 0, "links": 0, "link_groups": 0, "anchor_texts": 0}

    async with driver.session() as session:
        # Export fused Page:SeoWebPage nodes (includes both seovoc + graph_rag props)
        result = await session.run(
            """
            MATCH (wp:Page:SeoWebPage {client_id: $client_id})
            RETURN wp
            """,
            client_id=client_id,
        )
        records = await result.data()
        for rec in records:
            wp = rec["wp"]
            uri = _safe_uri(wp.get("url", ""))
            page_ref = UOC[f"page/{_slug(wp.get('url', ''))}"]

            g.add((page_ref, RDF.type, SEOVOC.WebPage))

            # seovoc properties
            if wp.get("title"):
                g.add((page_ref, SEOVOC.title, Literal(wp["title"], datatype=XSD.string)))
            if wp.get("metaDescription"):
                g.add((page_ref, SEOVOC.metaDescription, Literal(wp["metaDescription"], datatype=XSD.string)))
            if wp.get("markdownText"):
                g.add((page_ref, SEOVOC.markdownText, Literal(wp["markdownText"][:5000], datatype=XSD.string)))
            if wp.get("embeddingModel"):
                g.add((page_ref, SEOVOC.embeddingModel, Literal(wp["embeddingModel"], datatype=XSD.string)))
            if wp.get("clickDepth") is not None:
                g.add((page_ref, SEOVOC.clickDepth, Literal(wp["clickDepth"], datatype=XSD.integer)))
            if wp.get("isCrawlable") is not None:
                g.add((page_ref, SEOVOC.isCrawlable, Literal(wp["isCrawlable"], datatype=XSD.boolean)))
            if wp.get("inLanguage"):
                lang_ref = UOC[f"lang/{wp['inLanguage']}"]
                g.add((page_ref, SCHEMA.inLanguage, lang_ref))
                g.add((lang_ref, RDF.type, SCHEMA.Language))

            # graph_rag scores (from fused :Page node) — exported as wkg: namespace
            if wp.get("pagerank") is not None:
                g.add((page_ref, WKG.pagerank, Literal(wp["pagerank"], datatype=XSD.double)))
            if wp.get("hub_score") is not None:
                g.add((page_ref, WKG.hubScore, Literal(wp["hub_score"], datatype=XSD.double)))
            if wp.get("authority_score") is not None:
                g.add((page_ref, WKG.authorityScore, Literal(wp["authority_score"], datatype=XSD.double)))

            # HAS_URL relationship
            url_ref = UOC[f"url/{_slug(wp.get('url', ''))}"]
            g.add((url_ref, RDF.type, SEOVOC.URL))
            g.add((url_ref, SEOVOC.value, Literal(wp.get("url", ""), datatype=XSD.string)))
            g.add((page_ref, SEOVOC.hasURL, url_ref))
            stats["urls"] += 1

            stats["webpages"] += 1

        # Export SeoChunk nodes with relationships
        result = await session.run(
            """
            MATCH (wp:Page:SeoWebPage {client_id: $client_id})-[:HAS_CHUNK]->(c:SeoChunk)
            RETURN wp.url AS page_url, c
            """,
            client_id=client_id,
        )
        records = await result.data()
        for rec in records:
            c = rec["c"]
            page_ref = UOC[f"page/{_slug(rec['page_url'])}"]
            chunk_ref = UOC[f"chunk/{_slug(rec['page_url'])}/{c.get('chunkPosition', 0)}"]

            g.add((chunk_ref, RDF.type, SEOVOC.Chunk))
            g.add((page_ref, SEOVOC.hasChunk, chunk_ref))
            g.add((chunk_ref, SEOVOC.isChunkOf, page_ref))

            if c.get("chunkText"):
                g.add((chunk_ref, SEOVOC.chunkText, Literal(c["chunkText"][:2000], datatype=XSD.string)))
            if c.get("chunkPosition") is not None:
                g.add((chunk_ref, SEOVOC.chunkPosition, Literal(c["chunkPosition"], datatype=XSD.integer)))
            if c.get("chunkSetName"):
                g.add((chunk_ref, SEOVOC.chunkSetName, Literal(c["chunkSetName"], datatype=XSD.string)))
            if c.get("chunkStrategy"):
                g.add((chunk_ref, SEOVOC.chunkStrategy, Literal(c["chunkStrategy"], datatype=XSD.string)))
            if c.get("start") is not None:
                g.add((chunk_ref, SEOVOC.start, Literal(c["start"], datatype=XSD.integer)))
            if c.get("end") is not None:
                g.add((chunk_ref, SEOVOC.end, Literal(c["end"], datatype=XSD.integer)))

            stats["chunks"] += 1

        # Export SeoLinkGroup + SeoLink
        result = await session.run(
            """
            MATCH (wp:Page:SeoWebPage {client_id: $client_id})-[:HAS_LINK_GROUP]->(lg:SeoLinkGroup)
            OPTIONAL MATCH (lg)-[:HAS_LINK]->(l:SeoLink)
            RETURN wp.url AS page_url, lg, collect(l) AS links
            """,
            client_id=client_id,
        )
        records = await result.data()
        for rec in records:
            lg = rec["lg"]
            page_ref = UOC[f"page/{_slug(rec['page_url'])}"]
            lg_ref = UOC[f"linkgroup/{_slug(rec['page_url'])}/{_slug(lg.get('name', ''))}"]

            g.add((lg_ref, RDF.type, SEOVOC.LinkGroup))
            g.add((page_ref, SEOVOC.hasLinkGroup, lg_ref))
            if lg.get("name"):
                g.add((lg_ref, SCHEMA.name, Literal(lg["name"], datatype=XSD.string)))
            if lg.get("identifier"):
                g.add((lg_ref, SCHEMA.identifier, Literal(lg["identifier"], datatype=XSD.string)))
            stats["link_groups"] += 1

            for link in rec["links"]:
                if link is None:
                    continue
                link_ref = UOC[f"link/{_slug(link.get('link_id', ''))}"]
                g.add((link_ref, RDF.type, SEOVOC.Link))
                g.add((lg_ref, SEOVOC.hasLink, link_ref))

                if link.get("linkType"):
                    g.add((link_ref, SEOVOC.linkType, Literal(link["linkType"], datatype=XSD.string)))
                if link.get("weight") is not None:
                    g.add((link_ref, SEOVOC.weight, Literal(link["weight"], datatype=XSD.float)))
                if link.get("position") is not None:
                    g.add((link_ref, SCHEMA.position, Literal(link["position"], datatype=XSD.integer)))
                if link.get("target_url"):
                    target_ref = UOC[f"page/{_slug(link['target_url'])}"]
                    g.add((link_ref, SEOVOC.anchorResource, target_ref))

                stats["links"] += 1

    # Serialize
    output_path = Path(output_path)
    g.serialize(destination=str(output_path), format="turtle")

    stats["triples"] = len(g)
    stats["output"] = str(output_path)
    return stats


def _slug(url: str) -> str:
    """Convert URL to a safe slug for URI construction."""
    import re
    s = url.replace("https://", "").replace("http://", "")
    s = re.sub(r"[^a-zA-Z0-9_/.-]", "_", s)
    return s.strip("/").replace("/", "_")
