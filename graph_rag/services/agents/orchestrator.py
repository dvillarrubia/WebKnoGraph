"""
Orchestrator agent that classifies intent, delegates to sub-agents, and synthesizes results.
"""

import json
import logging
from typing import Optional, AsyncGenerator

from openai import AsyncOpenAI

from graph_rag.config.settings import Settings
from graph_rag.db.supabase_client import SupabaseClient
from graph_rag.db.neo4j_client import Neo4jClient
from graph_rag.services.embedding_service import EmbeddingService
from graph_rag.services.agents.base_agent import AgentStep, StepType
from graph_rag.services.agents.tools import ToolRegistry
from graph_rag.services.agents.sub_agents import AGENT_REGISTRY

logger = logging.getLogger(__name__)


CLASSIFICATION_PROMPT = """Eres un orquestador que clasifica preguntas y decide qué agentes especializados deben responder.

AGENTES DISPONIBLES:

1. **researcher** — Busca y recopila información del sitio web. Ideal para preguntas tipo "¿qué hay sobre X?", "¿tienen página de Y?", consultas generales sobre el contenido.

2. **seo_analyst** — Analiza estructura SEO, silos, topical maps, gaps temáticos. Ideal para "¿qué temas faltan?", "analiza la estructura", "propón un topical map".

3. **copywriter** — Genera textos SEO (meta titles, descriptions, contenido). Ideal para "escribe un texto para X", "mejora el copy de Y", "propón headings".

4. **link_analyst** — Analiza enlaces internos, detecta huérfanas, sugiere interlinking. Ideal para "¿qué enlaces tiene X?", "sugiere interlinking para Y", "páginas sin enlaces".

5. **auditor** — Auditoría técnica SEO: thin content, huérfanas, profundidad, silos rotos. Ideal para "audita el sitio", "¿hay problemas técnicos?", "páginas con poco contenido".

INSTRUCCIONES:
- Analiza la pregunta del usuario
- Selecciona 1-3 agentes (los mínimos necesarios)
- Para preguntas simples de información: solo "researcher"
- Para análisis complejos: combina agentes relevantes
- El orden importa: los primeros agentes dan contexto a los siguientes

Responde SOLO con JSON válido, sin markdown:
{"agents": ["agent1", "agent2"], "reasoning": "breve explicación"}"""


class OrchestratorAgent:
    """
    Top-level orchestrator that classifies user intent,
    delegates to specialized sub-agents, and synthesizes results.
    """

    def __init__(
        self,
        settings: Settings,
        supabase_client: SupabaseClient,
        neo4j_client: Neo4jClient,
        embedding_service: EmbeddingService,
    ):
        self.settings = settings
        self.tool_registry = ToolRegistry(
            supabase=supabase_client,
            neo4j=neo4j_client,
            embedding_service=embedding_service,
        )
        self._client: Optional[AsyncOpenAI] = None
        self.model = settings.openai_model

    @property
    def client(self) -> AsyncOpenAI:
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
        Main entry point. Classifies, delegates, synthesizes.
        Yields AgentStep objects for UI streaming.
        """
        # Step 1: Classify intent
        yield AgentStep(type=StepType.STATUS, content="Analizando consulta...")

        plan = await self._classify_intent(question, conversation_history)

        agent_names = plan.get("agents", ["researcher"])
        reasoning = plan.get("reasoning", "")

        agent_labels = {
            "researcher": "Investigador",
            "seo_analyst": "Analista SEO",
            "copywriter": "Copywriter",
            "link_analyst": "Analista de Enlaces",
            "auditor": "Auditor SEO",
        }
        names_es = [agent_labels.get(a, a) for a in agent_names]

        yield AgentStep(
            type=StepType.DELEGATION,
            content=f"Delegando a: {', '.join(names_es)}",
            data={"agents": agent_names, "reasoning": reasoning},
        )

        # Step 2: Run sub-agents sequentially
        agent_results = []
        last_answer = None

        for agent_name in agent_names:
            agent_cls = AGENT_REGISTRY.get(agent_name)
            if not agent_cls:
                yield AgentStep(
                    type=StepType.ERROR,
                    content=f"Agente desconocido: {agent_name}",
                )
                continue

            agent = agent_cls(
                tool_registry=self.tool_registry,
                openai_client=self.client,
                model=self.model,
            )

            yield AgentStep(
                type=StepType.STATUS,
                content=f"{agent_labels.get(agent_name, agent_name)} trabajando...",
                agent_name=agent_name,
            )

            # Build context from previous agents
            prev_context = None
            if agent_results:
                prev_context = "\n\n".join(
                    f"[{agent_labels.get(r['agent'], r['agent'])}]:\n{r['answer'][:2000]}"
                    for r in agent_results
                )

            # Run sub-agent and stream its steps
            async for step in agent.run(
                client_id=client_id,
                question=question,
                context=prev_context,
                conversation_history=conversation_history,
            ):
                step.agent_name = agent_name
                if step.type == StepType.ANSWER:
                    last_answer = step.content
                    agent_results.append({
                        "agent": agent_name,
                        "answer": step.content,
                    })
                    # Don't yield individual agent answers if we'll synthesize
                    if len(agent_names) == 1:
                        yield step
                else:
                    yield step

        # Step 3: Synthesize if multiple agents contributed
        if len(agent_results) > 1:
            yield AgentStep(type=StepType.STATUS, content="Sintetizando respuesta final...")

            final_answer = await self._synthesize(question, agent_results)

            yield AgentStep(
                type=StepType.ANSWER,
                content=final_answer,
                data={
                    "agents_used": [r["agent"] for r in agent_results],
                    "synthesized": True,
                },
            )
        elif len(agent_results) == 0:
            yield AgentStep(
                type=StepType.ANSWER,
                content="No pude procesar tu consulta. Intenta reformularla.",
            )

    async def _classify_intent(
        self,
        question: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> dict:
        """Classify the user's intent and select sub-agents."""
        messages = [
            {"role": "system", "content": CLASSIFICATION_PROMPT},
        ]

        # Add recent history for context
        if conversation_history:
            for msg in conversation_history[-4:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": question})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=200,
            )

            text = response.choices[0].message.content.strip()
            # Parse JSON (handle potential markdown wrapping)
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            plan = json.loads(text)

            # Validate agent names
            valid_agents = [a for a in plan.get("agents", []) if a in AGENT_REGISTRY]
            if not valid_agents:
                valid_agents = ["researcher"]

            return {"agents": valid_agents, "reasoning": plan.get("reasoning", "")}

        except Exception as e:
            logger.warning(f"Classification failed, defaulting to researcher: {e}")
            return {"agents": ["researcher"], "reasoning": "Fallback por error en clasificación"}

    async def _synthesize(self, question: str, agent_results: list[dict]) -> str:
        """Synthesize multiple agent results into a single coherent answer."""
        agent_labels = {
            "researcher": "Investigador",
            "seo_analyst": "Analista SEO",
            "copywriter": "Copywriter",
            "link_analyst": "Analista de Enlaces",
            "auditor": "Auditor SEO",
        }

        results_text = "\n\n".join(
            f"### {agent_labels.get(r['agent'], r['agent'])}:\n{r['answer'][:3000]}"
            for r in agent_results
        )

        messages = [
            {
                "role": "system",
                "content": """Eres un sintetizador experto. Combina las respuestas de varios agentes especializados en una respuesta única, coherente y bien estructurada.

REGLAS:
- Integra las perspectivas sin repetir información
- Mantén las URLs y datos específicos citados por los agentes
- Estructura la respuesta con secciones claras si es larga
- No menciones a los agentes internos al usuario
- Responde en español"""
            },
            {
                "role": "user",
                "content": f"Pregunta original: {question}\n\nRespuestas de los expertos:\n{results_text}\n\nSintetiza una respuesta completa:",
            },
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=3000,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback: concatenate results
            return "\n\n---\n\n".join(r["answer"] for r in agent_results)
