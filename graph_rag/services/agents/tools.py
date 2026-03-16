"""
Tool registry for RAG agents.
Central place for all tool definitions and executors.
"""

from typing import Any
from graph_rag.db.supabase_client import SupabaseClient
from graph_rag.db.neo4j_client import Neo4jClient
from graph_rag.services.embedding_service import EmbeddingService


# =============================================================================
# TOOL DEFINITIONS (OpenAI function-calling schema)
# =============================================================================

TOOL_SCHEMAS = {
    "search_content": {
        "type": "function",
        "function": {
            "name": "search_content",
            "description": "Buscar contenido relevante en la base de conocimiento por similitud semántica. Usa esto para encontrar información sobre un tema específico.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La consulta de búsqueda. Sé específico para mejores resultados."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Número de resultados (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    "get_page_content": {
        "type": "function",
        "function": {
            "name": "get_page_content",
            "description": "Obtener el contenido completo de una página específica por su URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "La URL completa de la página"
                    }
                },
                "required": ["url"]
            }
        }
    },
    "find_related_pages": {
        "type": "function",
        "function": {
            "name": "find_related_pages",
            "description": "Encontrar páginas relacionadas con una URL a través de enlaces internos del sitio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "La URL de la página origen"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Máximo de páginas relacionadas (default: 5)",
                        "default": 5
                    }
                },
                "required": ["url"]
            }
        }
    },
    "get_top_pages": {
        "type": "function",
        "function": {
            "name": "get_top_pages",
            "description": "Obtener las páginas más importantes del sitio ordenadas por PageRank.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Número de páginas (default: 10)",
                        "default": 10
                    }
                }
            }
        }
    },
    "get_link_analysis": {
        "type": "function",
        "function": {
            "name": "get_link_analysis",
            "description": "Analizar los enlaces entrantes y salientes de una URL. Muestra distribución por ubicación (contenido/nav/footer) y anchor texts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "La URL a analizar"
                    }
                },
                "required": ["url"]
            }
        }
    },
    "get_silo_structure": {
        "type": "function",
        "function": {
            "name": "get_silo_structure",
            "description": "Obtener la estructura de silos del sitio web agrupada por carpetas/secciones URL. Muestra cuántas páginas hay en cada silo y su PageRank medio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_depth": {
                        "type": "integer",
                        "description": "Profundidad máxima de carpetas a analizar (default: 3)",
                        "default": 3
                    }
                }
            }
        }
    },
    "compare_pages": {
        "type": "function",
        "function": {
            "name": "compare_pages",
            "description": "Comparar dos páginas: métricas, contenido, enlaces entrantes/salientes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url1": {
                        "type": "string",
                        "description": "Primera URL"
                    },
                    "url2": {
                        "type": "string",
                        "description": "Segunda URL"
                    }
                },
                "required": ["url1", "url2"]
            }
        }
    },
    "get_thin_pages": {
        "type": "function",
        "function": {
            "name": "get_thin_pages",
            "description": "Encontrar páginas con contenido muy corto (thin content). Útil para auditoría SEO.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_word_count": {
                        "type": "integer",
                        "description": "Umbral máximo de palabras para considerar thin (default: 200)",
                        "default": 200
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Máximo de resultados (default: 20)",
                        "default": 20
                    }
                }
            }
        }
    },
    "get_orphan_pages": {
        "type": "function",
        "function": {
            "name": "get_orphan_pages",
            "description": "Encontrar páginas huérfanas: sin enlaces entrantes desde el contenido de otras páginas. Solo tienen enlaces desde navegación.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Máximo de resultados (default: 20)",
                        "default": 20
                    }
                }
            }
        }
    },
}


class ToolRegistry:
    """Central registry for agent tools. Executes tool calls against DB clients."""

    def __init__(
        self,
        supabase: SupabaseClient,
        neo4j: Neo4jClient,
        embedding_service: EmbeddingService,
    ):
        self.supabase = supabase
        self.neo4j = neo4j
        self.embedding_service = embedding_service

    def get_tool_schemas(self, tool_names: list[str]) -> list[dict]:
        """Get OpenAI function-calling schemas for the given tool names."""
        return [TOOL_SCHEMAS[name] for name in tool_names if name in TOOL_SCHEMAS]

    async def execute(self, client_id: str, tool_name: str, args: dict) -> Any:
        """Execute a tool by name and return the result."""

        if tool_name == "search_content":
            query = args.get("query", "")
            top_k = args.get("top_k", 5)
            query_embedding = self.embedding_service.embed_query(query)
            results = await self.supabase.search_similar_chunks(
                client_id=client_id,
                query_embedding=query_embedding,
                limit=top_k,
                min_similarity=0.25,
            )
            return [
                {
                    "url": r.get("url"),
                    "title": r.get("title"),
                    "content": r.get("content", "")[:500],
                    "heading": r.get("heading_context"),
                    "similarity": round(r.get("similarity", 0), 3),
                }
                for r in results
            ]

        elif tool_name == "get_page_content":
            url = args.get("url", "")
            page = await self.supabase.get_page_by_url(client_id, url)
            if page:
                return {
                    "url": page.get("url"),
                    "title": page.get("title"),
                    "content": page.get("content", "")[:3000],
                    "pagerank": page.get("pagerank", 0),
                }
            return {"error": "Página no encontrada"}

        elif tool_name == "find_related_pages":
            url = args.get("url", "")
            limit = args.get("limit", 5)
            related = await self.neo4j.get_linked_pages(
                client_id=client_id, url=url, hops=1, limit=limit,
            )
            return [
                {"url": r.get("url"), "title": r.get("title"), "pagerank": r.get("pagerank", 0)}
                for r in related
            ]

        elif tool_name == "get_top_pages":
            limit = args.get("limit", 10)
            pages = await self.supabase.list_pages(client_id=client_id, limit=limit)
            return [
                {"url": p.get("url"), "title": p.get("title"), "pagerank": round(p.get("pagerank", 0), 4)}
                for p in pages
            ]

        elif tool_name == "get_link_analysis":
            url = args.get("url", "")
            inbound = await self.neo4j.get_incoming_links(client_id, url, limit=50)
            outbound = await self.neo4j.get_outgoing_links(client_id, url, limit=50)

            in_content = [l for l in inbound if l.get("location") == "content"]
            in_nav = [l for l in inbound if l.get("location") != "content"]
            out_content = [l for l in outbound if l.get("location") == "content"]
            out_nav = [l for l in outbound if l.get("location") != "content"]

            return {
                "url": url,
                "inbound_total": len(inbound),
                "inbound_content": len(in_content),
                "inbound_nav": len(in_nav),
                "inbound_content_pages": [
                    {"url": l["url"], "title": l.get("title"), "anchor": l.get("anchor_text")}
                    for l in in_content[:10]
                ],
                "outbound_total": len(outbound),
                "outbound_content": len(out_content),
                "outbound_nav": len(out_nav),
                "outbound_content_pages": [
                    {"url": l["url"], "title": l.get("title"), "anchor": l.get("anchor_text")}
                    for l in out_content[:10]
                ],
            }

        elif tool_name == "get_silo_structure":
            max_depth = args.get("max_depth", 3)
            silos = await self.neo4j.get_silo_structure(client_id, max_depth)
            return [
                {
                    "silo": s["silo"],
                    "pages": s["page_count"],
                    "avg_pagerank": round(s["avg_pagerank"] or 0, 4),
                    "pages_with_links": s["pages_with_links"],
                    "samples": s["sample_urls"],
                }
                for s in silos
            ]

        elif tool_name == "compare_pages":
            url1 = args.get("url1", "")
            url2 = args.get("url2", "")
            p1 = await self.supabase.get_page_by_url(client_id, url1)
            p2 = await self.supabase.get_page_by_url(client_id, url2)

            def page_summary(p, url):
                if not p:
                    return {"url": url, "error": "No encontrada"}
                return {
                    "url": p.get("url"),
                    "title": p.get("title"),
                    "content_length": len(p.get("content", "")),
                    "pagerank": p.get("pagerank", 0),
                    "content_preview": p.get("content", "")[:300],
                }

            return {"page1": page_summary(p1, url1), "page2": page_summary(p2, url2)}

        elif tool_name == "get_thin_pages":
            max_wc = args.get("max_word_count", 200)
            limit = args.get("limit", 20)
            pages = await self.supabase.get_thin_pages(client_id, max_wc, limit)
            return [
                {
                    "url": p.get("url"),
                    "title": p.get("title"),
                    "content_length": p.get("content_length", 0),
                    "pagerank": round(p.get("pagerank", 0), 4),
                }
                for p in pages
            ]

        elif tool_name == "get_orphan_pages":
            limit = args.get("limit", 20)
            pages = await self.neo4j.get_orphan_pages(client_id, limit)
            return [
                {
                    "url": p.get("url"),
                    "title": p.get("title"),
                    "pagerank": round(p.get("pagerank", 0), 4),
                }
                for p in pages
            ]

        return {"error": f"Tool desconocido: {tool_name}"}

    def summarize_result(self, tool_name: str, result: Any) -> str:
        """Create a human-readable summary of tool results."""
        if isinstance(result, list):
            n = len(result)
            labels = {
                "search_content": f"Encontrados {n} fragmentos relevantes",
                "find_related_pages": f"Encontradas {n} páginas relacionadas",
                "get_top_pages": f"Obtenidas {n} páginas principales",
                "get_silo_structure": f"Analizados {n} silos",
                "get_thin_pages": f"Encontradas {n} páginas con contenido delgado",
                "get_orphan_pages": f"Encontradas {n} páginas huérfanas",
            }
            return labels.get(tool_name, f"{n} resultados")
        elif isinstance(result, dict):
            if "error" in result:
                return f"Error: {result['error']}"
            if tool_name == "get_page_content":
                return f"Contenido obtenido: {result.get('title', '')[:50]}"
            if tool_name == "get_link_analysis":
                return f"Enlaces: {result.get('inbound_content', 0)} entrantes en contenido, {result.get('outbound_content', 0)} salientes"
            if tool_name == "compare_pages":
                return "Comparación obtenida"
            return "Resultado obtenido"
        return "Completado"
