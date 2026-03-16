"""
Agentic RAG Service — thin wrapper around the multi-agent orchestrator.
Maintains backwards compatibility with existing routes.
"""

from typing import Optional, AsyncGenerator

from graph_rag.config.settings import Settings
from graph_rag.db.supabase_client import SupabaseClient
from graph_rag.db.neo4j_client import Neo4jClient
from graph_rag.services.embedding_service import EmbeddingService
from graph_rag.services.agents import OrchestratorAgent, AgentStep, StepType


class AgenticRAGService:
    """
    Delegates to the multi-agent OrchestratorAgent.
    Keeps the same interface so routes.py needs minimal changes.
    """

    def __init__(
        self,
        settings: Settings,
        supabase_client: SupabaseClient,
        neo4j_client: Neo4jClient,
        embedding_service: EmbeddingService,
    ):
        self._orchestrator = OrchestratorAgent(
            settings=settings,
            supabase_client=supabase_client,
            neo4j_client=neo4j_client,
            embedding_service=embedding_service,
        )

    async def query_stream(
        self,
        client_id: str,
        question: str,
        conversation_history: Optional[list[dict]] = None,
        max_iterations: int = 5,
    ) -> AsyncGenerator[AgentStep, None]:
        """Proxy to orchestrator. Yields AgentStep objects for SSE streaming."""
        async for step in self._orchestrator.query_stream(
            client_id=client_id,
            question=question,
            conversation_history=conversation_history,
            max_iterations=max_iterations,
        ):
            yield step
