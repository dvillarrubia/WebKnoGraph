"""
RAG Service for Graph-RAG.
Combines vector search + graph expansion + LLM generation.
"""

from typing import Optional
from dataclasses import dataclass
from openai import OpenAI

from graph_rag.config.settings import Settings
from graph_rag.db.supabase_client import SupabaseClient
from graph_rag.db.neo4j_client import Neo4jClient
from graph_rag.services.embedding_service import EmbeddingService


@dataclass
class RAGContext:
    """Context retrieved for RAG."""
    chunks: list[dict]
    total_tokens: int
    vector_results: int
    graph_expanded: int


@dataclass
class RAGResponse:
    """Response from RAG query."""
    answer: str
    context: RAGContext
    sources: list[dict]
    tokens_used: int


class RAGService:
    """
    Multi-tenant RAG service that combines:
    1. Vector similarity search (pgvector)
    2. Graph-based context expansion (Neo4j)
    3. LLM response generation (OpenAI)
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
        self._openai_client: Optional[OpenAI] = None

    @property
    def openai_client(self) -> OpenAI:
        """Lazy load OpenAI client."""
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self.settings.openai_api_key)
        return self._openai_client

    async def query(
        self,
        client_id: str,
        question: str,
        conversation_history: Optional[list[dict]] = None,
        top_k_vectors: Optional[int] = None,
        graph_hops: Optional[int] = None,
        max_context_pages: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> RAGResponse:
        """
        Execute a RAG query.

        Flow:
        1. Generate embedding for the question
        2. Search similar pages in pgvector
        3. Expand context using graph neighborhood
        4. Build prompt with context
        5. Generate response with OpenAI

        Args:
            client_id: Client ID for multi-tenant filtering
            question: User's question
            conversation_history: Previous messages for context
            top_k_vectors: Number of vector results to retrieve
            graph_hops: Number of graph hops for expansion
            max_context_pages: Maximum pages in context
            min_similarity: Minimum similarity threshold

        Returns:
            RAGResponse with answer, context, and sources
        """
        # Use settings defaults if not provided
        top_k_vectors = top_k_vectors or self.settings.rag_top_k_vectors
        graph_hops = graph_hops or self.settings.rag_graph_hops
        max_context_pages = max_context_pages or self.settings.rag_max_context_pages
        min_similarity = min_similarity or self.settings.rag_min_similarity

        # 1. Generate query embedding
        query_embedding = self.embedding_service.embed_query(question)

        # 2. Vector similarity search on CHUNKS (not pages)
        chunk_results = await self.supabase.search_similar_chunks(
            client_id=client_id,
            query_embedding=query_embedding,
            limit=top_k_vectors,
            min_similarity=min_similarity,
        )

        # 3. Graph-based context expansion based on page URLs from chunks
        graph_expanded_chunks = []
        if chunk_results and graph_hops > 0:
            # Get unique page URLs from chunks
            source_urls = list(set(chunk["url"] for chunk in chunk_results))
            graph_expanded_pages = await self.neo4j.expand_context_from_urls(
                client_id=client_id,
                urls=source_urls,
                hops=graph_hops,
                limit_per_url=3,
            )

            # Fetch chunks for graph-expanded pages
            for page in graph_expanded_pages:
                page_data = await self.supabase.get_page_by_url(client_id, page["url"])
                if page_data:
                    page_chunks = await self.supabase.get_chunks_by_page(str(page_data["id"]))
                    for chunk in page_chunks[:2]:  # Limit chunks per graph page
                        chunk["url"] = page["url"]
                        chunk["title"] = page_data.get("title", "")
                        chunk["pagerank"] = page_data.get("pagerank", 0)
                        chunk["similarity"] = 0.0  # No direct similarity
                        chunk["from_graph"] = True
                        graph_expanded_chunks.append(chunk)
        else:
            graph_expanded_pages = []

        # Mark chunk results as not from graph
        for chunk in chunk_results:
            chunk["from_graph"] = False

        # Combine and deduplicate by chunk ID
        seen_chunk_ids = set()
        combined_chunks = []

        for chunk in chunk_results:
            chunk_id = chunk.get("id") or f"{chunk.get('page_id')}_{chunk.get('chunk_index')}"
            if chunk_id not in seen_chunk_ids:
                combined_chunks.append(chunk)
                seen_chunk_ids.add(chunk_id)

        for chunk in graph_expanded_chunks:
            chunk_id = chunk.get("id") or f"{chunk.get('page_id')}_{chunk.get('chunk_index')}"
            if chunk_id not in seen_chunk_ids:
                combined_chunks.append(chunk)
                seen_chunk_ids.add(chunk_id)

        # 4. Limit and prepare context
        context_chunks = combined_chunks[:max_context_pages]

        # Build context string from chunks
        context_text, total_tokens = self._build_context_from_chunks(context_chunks)

        # 5. Generate response with OpenAI
        answer, tokens_used = await self._generate_response(
            question=question,
            context=context_text,
            conversation_history=conversation_history,
        )

        # Prepare sources (deduplicate by URL for cleaner output)
        seen_urls = set()
        sources = []
        for chunk in context_chunks:
            if chunk["url"] not in seen_urls:
                sources.append({
                    "url": chunk["url"],
                    "title": chunk.get("title", ""),
                    "pagerank": chunk.get("pagerank", 0),
                    "similarity": chunk.get("similarity", 0),
                    "from_graph": chunk.get("from_graph", False),
                    "heading_context": chunk.get("heading_context", ""),
                })
                seen_urls.add(chunk["url"])

        return RAGResponse(
            answer=answer,
            context=RAGContext(
                chunks=context_chunks,
                total_tokens=total_tokens,
                vector_results=len(chunk_results),
                graph_expanded=len(graph_expanded_chunks),
            ),
            sources=sources,
            tokens_used=tokens_used,
        )

    def _build_context_from_chunks(self, chunks: list[dict]) -> tuple[str, int]:
        """Build context string from chunks, respecting token limits."""
        context_parts = []
        estimated_tokens = 0
        max_tokens = self.settings.rag_context_max_tokens

        for chunk in chunks:
            content = chunk.get("content", "")
            if not content:
                continue

            # Estimate tokens (rough: 4 chars per token)
            content_tokens = len(content) // 4

            if estimated_tokens + content_tokens > max_tokens:
                # Truncate this chunk's content
                remaining_tokens = max_tokens - estimated_tokens
                remaining_chars = remaining_tokens * 4
                content = content[:remaining_chars] + "..."

            # Include heading context for better understanding
            heading = chunk.get("heading_context", "")
            heading_line = f"Seccion: {heading}\n" if heading else ""

            chunk_context = f"""
---
URL: {chunk.get('url', '')}
Titulo: {chunk.get('title', 'Sin titulo')}
{heading_line}
Contenido:
{content}
---
"""
            context_parts.append(chunk_context)
            estimated_tokens += len(chunk_context) // 4

            if estimated_tokens >= max_tokens:
                break

        return "\n".join(context_parts), estimated_tokens

    def _build_context(self, pages: list[dict]) -> tuple[str, int]:
        """Build context string from pages, respecting token limits. (Legacy, kept for backwards compatibility)"""
        context_parts = []
        estimated_tokens = 0
        max_tokens = self.settings.rag_context_max_tokens

        for page in pages:
            content = page.get("content", "")
            if not content:
                continue

            # Estimate tokens (rough: 4 chars per token)
            content_tokens = len(content) // 4

            if estimated_tokens + content_tokens > max_tokens:
                # Truncate this page's content
                remaining_tokens = max_tokens - estimated_tokens
                remaining_chars = remaining_tokens * 4
                content = content[:remaining_chars] + "..."

            page_context = f"""
---
URL: {page.get('url', '')}
Titulo: {page.get('title', 'Sin titulo')}
PageRank: {page.get('pagerank', 0):.6f}

Contenido:
{content}
---
"""
            context_parts.append(page_context)
            estimated_tokens += len(page_context) // 4

            if estimated_tokens >= max_tokens:
                break

        return "\n".join(context_parts), estimated_tokens

    async def _generate_response(
        self,
        question: str,
        context: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> tuple[str, int]:
        """Generate response using OpenAI."""

        system_prompt = """Eres un asistente experto en copywriting y SEO.

IMPORTANTE - TU FUENTE DE CONOCIMIENTO:
Tienes acceso a un sistema RAG (Retrieval-Augmented Generation) conectado a un GRAFO DE CONOCIMIENTO que contiene todo el contenido indexado del cliente actualmente seleccionado. El contexto que recibes proviene de:
1. Búsqueda vectorial semántica sobre el contenido del cliente
2. Expansión mediante el grafo de enlaces entre páginas (PageRank, relaciones entre URLs)

REGLAS ABSOLUTAS:
- DEBES responder SIEMPRE y ÚNICAMENTE basándote en el contexto proporcionado
- Es tu ÚNICA fuente de verdad - no tienes acceso a información externa
- Si la información está en el contexto: úsala y CITA las URLs
- Si NO está en el contexto: responde "No tengo información sobre esto en el contenido indexado del cliente"
- NUNCA inventes información que no esté en el contexto
- NUNCA uses conocimiento general que no provenga del contexto

TU ROL:
- Ayudas al departamento de SEO y copywriting a crear y optimizar contenido
- Analizas el tono, estilo y terminología del cliente basándote en su contenido indexado
- Eres experto en posicionamiento web

CÓMO RESPONDER:
1. SIEMPRE basa tu respuesta en el contexto proporcionado
2. SIEMPRE cita las URLs de donde extraes la información
3. Responde en español (castellano de España)
4. Adapta tu respuesta al objetivo del usuario:
   - COPYS: textos listos para usar, imitando el tono del contenido existente
   - SEO: keywords, estructura de headings, meta descriptions basadas en el contenido real
   - INFORMACIÓN: sintetiza el contenido existente
   - ANÁLISIS: identifica patrones, gaps o mejoras basándote en lo indexado

PARA TAREAS DE COPYWRITING:
- Analiza el tono y estilo del contenido existente en el contexto
- Genera textos coherentes con la voz de marca que observas
- Mantén la terminología que usa el cliente en su web

PARA TAREAS SEO:
- Sugiere H1, H2 basados en el contenido existente
- Propón meta descriptions (max 155 chars) usando texto del contexto
- Identifica keywords que YA aparecen en el contenido indexado
- Señala oportunidades de enlazado interno usando las URLs del contexto
- Detecta gaps de contenido comparando con lo que el usuario pregunta

CONTEXTO DEL GRAFO DE CONOCIMIENTO (contenido del cliente seleccionado):
{context}
"""

        messages = [
            {"role": "system", "content": system_prompt.format(context=context)}
        ]

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Add current question
        messages.append({"role": "user", "content": question})

        # Call OpenAI
        response = self.openai_client.chat.completions.create(
            model=self.settings.openai_model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
        )

        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        return answer, tokens_used

    async def get_related_pages(
        self,
        client_id: str,
        url: str,
        limit: int = 10,
    ) -> list[dict]:
        """Get pages related to a given URL via links."""
        return await self.neo4j.get_linked_pages(
            client_id=client_id,
            url=url,
            hops=2,
            limit=limit,
        )

    async def find_path(
        self,
        client_id: str,
        source_url: str,
        target_url: str,
    ) -> Optional[list[str]]:
        """Find the shortest path between two pages."""
        return await self.neo4j.get_shortest_path(
            client_id=client_id,
            source_url=source_url,
            target_url=target_url,
        )
