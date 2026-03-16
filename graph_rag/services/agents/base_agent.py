"""
Base agent class with reusable OpenAI function-calling loop.
"""

import json
import logging
from typing import Optional, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

from openai import AsyncOpenAI

from graph_rag.services.agents.tools import ToolRegistry

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    STATUS = "status"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    ANSWER = "answer"
    ERROR = "error"
    DELEGATION = "delegation"


@dataclass
class AgentStep:
    """Represents a step in an agent's execution."""
    type: StepType
    content: str
    data: Optional[dict] = field(default=None)
    agent_name: Optional[str] = None


class BaseAgent:
    """
    Base agent with OpenAI function-calling loop.
    Subclasses define name, system_prompt, and tool subset.
    """

    name: str = "base"
    system_prompt: str = ""
    tool_names: list[str] = []
    max_iterations: int = 3

    def __init__(
        self,
        tool_registry: ToolRegistry,
        openai_client: AsyncOpenAI,
        model: str = "gpt-4o",
    ):
        self.tools = tool_registry
        self.client = openai_client
        self.model = model

    async def run(
        self,
        client_id: str,
        question: str,
        context: Optional[str] = None,
        conversation_history: Optional[list[dict]] = None,
    ) -> AsyncGenerator[AgentStep, None]:
        """
        Run the agent loop. Yields steps as it works.
        The last step with type=ANSWER contains the final answer.
        """
        messages = [{"role": "system", "content": self._build_system_prompt(context)}]

        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": question})

        tool_schemas = self.tools.get_tool_schemas(self.tool_names)
        tools_used = []

        for iteration in range(self.max_iterations):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.5,
                }
                if tool_schemas and iteration < self.max_iterations - 1:
                    kwargs["tools"] = tool_schemas
                    kwargs["tool_choice"] = "auto"

                response = await self.client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                messages.append(msg.model_dump())

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.function.name
                        tool_args = json.loads(tc.function.arguments)

                        yield AgentStep(
                            type=StepType.TOOL_CALL,
                            content=f"{tool_name}({', '.join(f'{k}={v!r}' for k,v in tool_args.items())})",
                            data={"tool": tool_name, "args": tool_args},
                            agent_name=self.name,
                        )

                        try:
                            result = await self.tools.execute(client_id, tool_name, tool_args)
                            tools_used.append(tool_name)

                            yield AgentStep(
                                type=StepType.TOOL_RESULT,
                                content=self.tools.summarize_result(tool_name, result),
                                data={"tool": tool_name},
                                agent_name=self.name,
                            )

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(result, ensure_ascii=False, default=str)[:6000],
                            })
                        except Exception as e:
                            logger.error(f"Tool error {tool_name}: {e}")
                            yield AgentStep(
                                type=StepType.ERROR,
                                content=f"Error en {tool_name}: {e}",
                                agent_name=self.name,
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps({"error": str(e)}),
                            })
                else:
                    # Final answer
                    if msg.content:
                        yield AgentStep(
                            type=StepType.ANSWER,
                            content=msg.content,
                            data={"tools_used": tools_used, "iterations": iteration + 1},
                            agent_name=self.name,
                        )
                    break

            except Exception as e:
                logger.error(f"Agent {self.name} error: {e}")
                yield AgentStep(
                    type=StepType.ERROR,
                    content=f"Error: {e}",
                    agent_name=self.name,
                )
                break

    def _build_system_prompt(self, extra_context: Optional[str] = None) -> str:
        prompt = self.system_prompt
        if extra_context:
            prompt += f"\n\nCONTEXTO ADICIONAL de agentes previos:\n{extra_context}"
        return prompt
