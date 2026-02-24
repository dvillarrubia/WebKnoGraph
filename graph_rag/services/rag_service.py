"""
RAG Service for Graph-RAG.
Combines vector search + graph expansion + LLM generation.
"""

from typing import Optional
from dataclasses import dataclass
import logging
from openai import OpenAI

from graph_rag.config.settings import Settings
from graph_rag.db.supabase_client import SupabaseClient
from graph_rag.db.neo4j_client import Neo4jClient
from graph_rag.services.embedding_service import EmbeddingService
from graph_rag.services.reranking_service import RerankingService

logger = logging.getLogger(__name__)


@dataclass
class RAGContext:
    """Context retrieved for RAG."""
    chunks: list[dict]
    total_tokens: int
    vector_results: int
    graph_expanded: int
    reranked: bool = False


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
    3. LLM response generation (Gemini)
    """

    def __init__(
        self,
        settings: Settings,
        supabase_client: SupabaseClient,
        neo4j_client: Neo4jClient,
        embedding_service: EmbeddingService,
        reranking_service: Optional[RerankingService] = None,
    ):
        self.settings = settings
        self.supabase = supabase_client
        self.neo4j = neo4j_client
        self.embedding_service = embedding_service
        self.reranking_service = reranking_service
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
        use_reranking: Optional[bool] = None,
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
        use_reranking = use_reranking if use_reranking is not None else self.settings.rag_use_reranking

        # If reranking is enabled, retrieve more candidates initially
        retrieval_limit = top_k_vectors * 5 if use_reranking and self.reranking_service else top_k_vectors

        # 1. Generate query embedding
        query_embedding = self.embedding_service.embed_query(question)

        # 2. Vector similarity search on CHUNKS (not pages)
        # Retrieve more candidates if reranking is enabled
        chunk_results = await self.supabase.search_similar_chunks(
            client_id=client_id,
            query_embedding=query_embedding,
            limit=retrieval_limit,
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
                        chunk["hub_score"] = page_data.get("hub_score", 0)
                        chunk["authority_score"] = page_data.get("authority_score", 0)
                        chunk["similarity"] = 0.0  # No direct similarity
                        chunk["from_graph"] = True
                        graph_expanded_chunks.append(chunk)
        else:
            graph_expanded_pages = []

        # Mark chunk results as not from graph
        for chunk in chunk_results:
            chunk["from_graph"] = False

        # 4. Community-based context expansion
        community_chunks = []
        if chunk_results:
            community_chunks = await self._expand_by_community(
                client_id=client_id,
                source_urls=[c["url"] for c in chunk_results[:5]],  # Top 5 URLs
                exclude_urls=set(c["url"] for c in chunk_results),
                max_pages=3,
            )

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

        for chunk in community_chunks:
            chunk_id = chunk.get("id") or f"{chunk.get('page_id')}_{chunk.get('chunk_index')}"
            if chunk_id not in seen_chunk_ids:
                combined_chunks.append(chunk)
                seen_chunk_ids.add(chunk_id)

        # 4. Rerank if enabled
        reranked = False
        if use_reranking and self.reranking_service and len(combined_chunks) > max_context_pages:
            logger.info(f"Reranking {len(combined_chunks)} chunks for query: {question[:50]}...")
            combined_chunks = self.reranking_service.rerank_with_metadata(
                query=question,
                chunks=combined_chunks,
                top_k=max_context_pages,
                pagerank_weight=0.1,
            )
            reranked = True
            logger.info(f"Reranking complete, top chunk score: {combined_chunks[0].get('rerank_score', 'N/A') if combined_chunks else 'N/A'}")

        # 5. Limit and prepare context
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
                reranked=reranked,
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

            # Graph metrics for SEO context
            pagerank = chunk.get("pagerank", 0)
            hub_score = chunk.get("hub_score", 0)
            authority_score = chunk.get("authority_score", 0)
            from_graph = chunk.get("from_graph", False)

            # Build graph info line
            graph_info = ""
            if pagerank > 0 or hub_score > 0 or authority_score > 0:
                importance = "alta" if pagerank > 0.1 else "media" if pagerank > 0.01 else "normal"
                graph_info = f"Importancia: {importance} | "
                if authority_score > 0.1:
                    graph_info += "Es página de autoridad (muy citada). "
                if hub_score > 0.1:
                    graph_info += "Es hub (enlaza a muchas autoridades). "
            if from_graph:
                graph_info += "[Relacionada por enlaces]"

            graph_line = f"Grafo: {graph_info}\n" if graph_info else ""

            chunk_context = f"""
---
URL: {chunk.get('url', '')}
Titulo: {chunk.get('title', 'Sin titulo')}
{heading_line}{graph_line}
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
        """Generate response using Gemini."""

        system_prompt = """Eres un asistente experto en copywriting y SEO con amplio conocimiento del sector.

TU FUENTE DE DATOS:
Tienes acceso a un sistema RAG conectado a un GRAFO DE CONOCIMIENTO con el contenido indexado del cliente. El contexto proviene de:
1. Búsqueda vectorial semántica sobre el contenido del cliente
2. Expansión mediante grafo de enlaces entre páginas (PageRank)

REGLAS FUNDAMENTALES:
- Los DATOS del cliente (qué dice su web, qué URLs tiene, qué términos usa) SOLO provienen del contexto
- Tu EXPERTISE en SEO, copywriting, topical maps, términos LSI, etc. SÍ puedes usarlo para analizar
- SIEMPRE cita URLs cuando menciones información específica del cliente
- NUNCA inventes datos que deberían estar en la web del cliente
- SÍ puedes comparar lo que el cliente tiene vs. lo que debería tener según mejores prácticas SEO

DISTINCIÓN CLAVE:
❌ PROHIBIDO: "El cliente tiene una página sobre radioterapia" (si no está en el contexto)
✅ PERMITIDO: "Según el topical map ideal de oncología, deberían existir páginas sobre radioterapia, inmunoterapia, etc."
❌ PROHIBIDO: "El Dr. García es especialista en oncología" (inventar datos del cliente)
✅ PERMITIDO: "Para un topical map completo de oncología, se esperarían entidades como: tipos de cáncer, tratamientos, profesionales, centros..."

TU ROL:
- Experto SEO y copywriting para el departamento de marketing
- Analizas el contenido existente del cliente (contexto)
- Aplicas tu conocimiento experto para identificar gaps, oportunidades y mejoras
- Generas copys y recomendaciones basadas en el contenido real + mejores prácticas

CÓMO RESPONDER:
1. Analiza el contenido del cliente en el contexto
2. Aplica tu expertise SEO para evaluar, comparar o recomendar
3. Cita URLs cuando refieras contenido específico del cliente
4. Distingue claramente entre "lo que el cliente tiene" vs "lo que debería tener"
5. Responde en español (castellano de España)

PARA ANÁLISIS SEO (topical maps, gaps, LSI):
- Extrae las entidades y términos que YA aparecen en el contexto del cliente
- Compara con el topical map ideal según tu conocimiento SEO experto
- Identifica gaps (qué falta) y oportunidades de mejora
- Sugiere términos LSI, entidades y temas que complementarían el contenido

PARA COPYWRITING:
- Analiza tono, estilo y terminología del contenido existente
- Genera textos coherentes con la voz de marca observada
- Mantén la terminología que usa el cliente

CONTEXTO DEL GRAFO DE CONOCIMIENTO (contenido del cliente):
{context}
"""

        # Build messages for OpenAI
        messages = [
            {"role": "system", "content": system_prompt.format(context=context)}
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

    async def _expand_by_community(
        self,
        client_id: str,
        source_urls: list[str],
        exclude_urls: set,
        max_pages: int = 3,
    ) -> list[dict]:
        """
        Expand context by finding related pages from the same communities.

        Args:
            client_id: Client ID for filtering
            source_urls: URLs from vector search results
            exclude_urls: URLs to exclude (already in results)
            max_pages: Maximum pages to add from communities

        Returns:
            List of chunk dictionaries from community-related pages
        """
        community_chunks = []

        try:
            # Get community IDs for source URLs
            async with self.neo4j.get_session() as session:
                result = await session.run(
                    """
                    MATCH (p:Page)
                    WHERE p.client_id = $client_id AND p.url IN $urls
                    AND p.community_id IS NOT NULL
                    RETURN DISTINCT p.community_id AS community_id
                    """,
                    client_id=client_id,
                    urls=source_urls[:5],  # Top 5 source URLs
                )
                records = await result.data()
                community_ids = [r["community_id"] for r in records]

                if not community_ids:
                    return []

                # Find top pages from same communities (by PageRank)
                result = await session.run(
                    """
                    MATCH (p:Page)
                    WHERE p.client_id = $client_id
                    AND p.community_id IN $community_ids
                    AND NOT p.url IN $exclude_urls
                    RETURN p.url, p.title, p.pagerank, p.community_id
                    ORDER BY p.pagerank DESC
                    LIMIT $limit
                    """,
                    client_id=client_id,
                    community_ids=community_ids,
                    exclude_urls=list(exclude_urls),
                    limit=max_pages,
                )
                records = await result.data()

            # Fetch chunks for community pages
            for page in records:
                page_data = await self.supabase.get_page_by_url(client_id, page["p.url"])
                if page_data:
                    page_chunks = await self.supabase.get_chunks_by_page(str(page_data["id"]))
                    for chunk in page_chunks[:2]:  # Limit chunks per community page
                        chunk["url"] = page["p.url"]
                        chunk["title"] = page_data.get("title", "")
                        chunk["pagerank"] = page.get("p.pagerank", 0)
                        chunk["hub_score"] = page_data.get("hub_score", 0)
                        chunk["authority_score"] = page_data.get("authority_score", 0)
                        chunk["similarity"] = 0.0
                        chunk["from_graph"] = True
                        chunk["from_community"] = True
                        chunk["community_id"] = page.get("p.community_id")
                        community_chunks.append(chunk)

            logger.info(f"Community expansion: found {len(community_chunks)} chunks from {len(records)} pages")

        except Exception as e:
            logger.warning(f"Community expansion failed: {e}")

        return community_chunks

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
