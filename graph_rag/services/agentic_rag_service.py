"""
Agentic RAG Service for Graph-RAG.
Implements an iterative agent that uses tools to gather information before answering.
"""

import json
import logging
from typing import Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from openai import AsyncOpenAI

from graph_rag.config.settings import Settings
from graph_rag.db.supabase_client import SupabaseClient
from graph_rag.db.neo4j_client import Neo4jClient
from graph_rag.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    STATUS = "status"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    ANSWER = "answer"
    ERROR = "error"


@dataclass
class AgentStep:
    """Represents a step in the agent's execution."""
    type: StepType
    content: str
    data: Optional[dict] = None


# Tools available to the agent
AGENT_TOOLS = [
    {
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
                        "description": "Número de resultados a devolver (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_content",
            "description": "Obtener el contenido completo de una página específica por su URL. Usa esto cuando necesites más detalle de una página encontrada.",
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
    {
        "type": "function",
        "function": {
            "name": "find_related_pages",
            "description": "Encontrar páginas relacionadas con una URL dada a través de enlaces. Útil para explorar contenido relacionado.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "La URL de la página origen"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Número máximo de páginas relacionadas (default: 5)",
                        "default": 5
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_pages",
            "description": "Obtener las páginas más importantes del sitio por PageRank. Útil para entender la estructura general.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Número de páginas a devolver (default: 10)",
                        "default": 10
                    }
                }
            }
        }
    }
]

AGENT_SYSTEM_PROMPT = """Eres un asistente experto en SEO y copywriting con acceso a un grafo de conocimiento.

TIENES HERRAMIENTAS para buscar información en la base de datos del cliente. DEBES usarlas para responder preguntas.

PROCESO:
1. Analiza la pregunta del usuario
2. Usa las herramientas disponibles para buscar información relevante
3. Si necesitas más contexto, usa más herramientas
4. Cuando tengas suficiente información, sintetiza una respuesta completa

HERRAMIENTAS DISPONIBLES:
- search_content: Buscar por similitud semántica (USAR PRIMERO)
- get_page_content: Obtener contenido completo de una página
- find_related_pages: Encontrar páginas relacionadas via enlaces
- get_top_pages: Ver las páginas más importantes del sitio

REGLAS:
- SIEMPRE usa al menos una herramienta antes de responder
- Cita URLs cuando menciones información específica
- Si no encuentras información relevante, dilo claramente
- Responde en español (castellano de España)
- Sé conciso pero completo

IMPORTANTE: No inventes información. Solo usa datos de las herramientas."""


class AgenticRAGService:
    """
    Agentic RAG service that iteratively gathers information using tools.
    """

    def __init__(
        self,
        settings: Settings,
        supabase_client: SupabaseClient,
        neo4j_client: Neo4jClient,
        embedding_service: EmbeddingService,
    ):
        self.settings = settings
        self.supabase = supabase_client
        self.neo4j = neo4j_client
        self.embedding_service = embedding_service
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy load async OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        return self._client

    async def query_stream(
        self,
        client_id: str,
        question: str,
        conversation_history: Optional[list[dict]] = None,
        max_iterations: int = 5,
    ) -> AsyncGenerator[AgentStep, None]:
        """
        Execute an agentic RAG query with streaming steps.

        Yields AgentStep objects as the agent works through the problem.
        """
        yield AgentStep(type=StepType.STATUS, content="Analizando pregunta...")

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        ]

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Add current question
        messages.append({"role": "user", "content": question})

        tools_used = []

        for iteration in range(max_iterations):
            try:
                response = await self.client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=messages,
                    tools=AGENT_TOOLS,
                    tool_choice="auto" if iteration < max_iterations - 1 else "none",
                    temperature=0.7,
                )

                assistant_message = response.choices[0].message
                messages.append(assistant_message.model_dump())

                # Check if there are tool calls
                if assistant_message.tool_calls:
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)

                        yield AgentStep(
                            type=StepType.TOOL_CALL,
                            content=f"Ejecutando {tool_name}...",
                            data={"tool": tool_name, "args": tool_args}
                        )

                        # Execute the tool
                        try:
                            result = await self._execute_tool(
                                client_id, tool_name, tool_args
                            )
                            tools_used.append(tool_name)

                            # Summarize result for display
                            result_summary = self._summarize_result(tool_name, result)

                            yield AgentStep(
                                type=StepType.TOOL_RESULT,
                                content=result_summary,
                                data={"tool": tool_name, "result_count": len(result) if isinstance(result, list) else 1}
                            )

                            # Add tool result to messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result, ensure_ascii=False)[:4000]  # Limit size
                            })

                        except Exception as e:
                            logger.error(f"Tool execution error: {e}")
                            yield AgentStep(
                                type=StepType.ERROR,
                                content=f"Error en {tool_name}: {str(e)}"
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps({"error": str(e)})
                            })

                else:
                    # No more tool calls - this is the final answer
                    if assistant_message.content:
                        yield AgentStep(
                            type=StepType.ANSWER,
                            content=assistant_message.content,
                            data={"tools_used": tools_used, "iterations": iteration + 1}
                        )
                    break

            except Exception as e:
                logger.error(f"Agent error: {e}")
                yield AgentStep(
                    type=StepType.ERROR,
                    content=f"Error del agente: {str(e)}"
                )
                break

        # If we hit max iterations without an answer
        if iteration == max_iterations - 1:
            yield AgentStep(
                type=StepType.STATUS,
                content="Sintetizando respuesta final..."
            )

    async def _execute_tool(
        self,
        client_id: str,
        tool_name: str,
        args: dict
    ) -> list | dict:
        """Execute a tool and return the result."""

        if tool_name == "search_content":
            query = args.get("query", "")
            top_k = args.get("top_k", 5)

            # Generate embedding
            query_embedding = self.embedding_service.embed_query(query)

            # Search chunks
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
                    "content": page.get("content", "")[:2000],
                    "pagerank": page.get("pagerank", 0),
                }
            return {"error": "Página no encontrada"}

        elif tool_name == "find_related_pages":
            url = args.get("url", "")
            limit = args.get("limit", 5)

            related = await self.neo4j.get_linked_pages(
                client_id=client_id,
                url=url,
                hops=1,
                limit=limit,
            )

            return [
                {
                    "url": r.get("url"),
                    "title": r.get("title"),
                    "relation": r.get("relation", "linked"),
                }
                for r in related
            ]

        elif tool_name == "get_top_pages":
            limit = args.get("limit", 10)

            pages = await self.supabase.list_pages(
                client_id=client_id,
                limit=limit,
            )

            return [
                {
                    "url": p.get("url"),
                    "title": p.get("title"),
                    "pagerank": round(p.get("pagerank", 0), 4),
                }
                for p in pages
            ]

        else:
            return {"error": f"Tool desconocido: {tool_name}"}

    def _summarize_result(self, tool_name: str, result: list | dict) -> str:
        """Create a human-readable summary of tool results."""
        if isinstance(result, list):
            count = len(result)
            if tool_name == "search_content":
                return f"Encontrados {count} fragmentos relevantes"
            elif tool_name == "find_related_pages":
                return f"Encontradas {count} páginas relacionadas"
            elif tool_name == "get_top_pages":
                return f"Obtenidas {count} páginas principales"
            return f"{count} resultados"
        elif isinstance(result, dict):
            if "error" in result:
                return f"Error: {result['error']}"
            if tool_name == "get_page_content":
                title = result.get("title", "Sin título")
                return f"Contenido obtenido: {title[:50]}"
            return "Resultado obtenido"
        return "Completado"
