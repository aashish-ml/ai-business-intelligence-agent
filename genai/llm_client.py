from __future__ import annotations

from typing import Protocol

from openai import OpenAI

from genai.mock_llm import MockLLMClient
from genai.schemas import AgentPlan
from src.tools.registry import get_tool_registry
from src.utils.config import get_settings


class LLMClient(Protocol):
    """Interface implemented by all LLM planning clients."""

    def create_plan(
        self,
        question: str,
        context: str = "",
    ) -> AgentPlan:
        """Create a structured agent execution plan."""
        ...


def build_tool_context() -> str:
    """
    Build the tool context dynamically from the canonical registry.

    This keeps the LLM's available-tool knowledge synchronized
    with the actual tools registered by the application.
    """

    registry = get_tool_registry()

    lines = [
        "Available tools:",
        "",
    ]

    for index, tool_name in enumerate(sorted(registry), start=1):
        definition = registry[tool_name]

        lines.append(
            f"{index}. {definition.name}"
        )
        lines.append(
            f"   - {definition.description}"
        )
        lines.append("")

    return "\n".join(lines).strip()


def build_system_prompt() -> str:
    """Build the complete LLM planning prompt."""

    tool_context = build_tool_context()

    return f"""
You are the planning intelligence for an autonomous Business
Intelligence and Decision Support Agent.

Your job is to convert the user's business question into a
structured execution plan.

You do NOT execute tools yourself.

You must select only tools that are available to the agent.

{tool_context}

Planning rules:

- Prefer deterministic business_analysis tools for standard
  business metrics.
- Use customer_risk_prediction when a specific customer risk
  or churn question is asked.
- Use rag_search for business policies, procedures,
  recommendations, and knowledge questions.
- Use execute_sql for custom analytical questions requiring
  database-level calculations.
- Use percentage_change for deterministic percentage calculations.
- Use group_and_aggregate for grouped analytical operations.
- Never invent database values.
- Never execute tools yourself.
- Return only a structured AgentPlan.
- Keep the plan minimal and relevant.
- For questions requiring multiple steps, return the required
  tool calls in execution order.
"""


class OpenAILLMClient:
    """
    Production LLM planner using OpenAI's Responses API.

    The model is required to return a validated AgentPlan.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

        if not self.settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured. "
                "Set it in the environment before using "
                "OpenAILLMClient."
            )

        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
        )

        self.system_prompt = build_system_prompt()

    def create_plan(
        self,
        question: str,
        context: str = "",
    ) -> AgentPlan:
        """Create a structured plan using the OpenAI model."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        user_content = (
            f"Conversation context:\n{context.strip()}\n\n"
            f"Current user question:\n{question.strip()}"
            if context and context.strip()
            else question.strip()
        )

        response = self.client.responses.parse(
            model=self.settings.llm_model,
            input=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            text_format=AgentPlan,
        )

        if response.output_parsed is None:
            raise RuntimeError(
                "The LLM did not return a structured AgentPlan."
            )

        return response.output_parsed


class FallbackLLMClient:
    """
    Development-safe LLM client.

    Uses the deterministic MockLLMClient. This keeps tests and
    local development independent of an external API.
    """

    def __init__(self) -> None:
        self.client = MockLLMClient()

    def create_plan(
        self,
        question: str,
        context: str = "",
    ) -> AgentPlan:
        """Create a deterministic plan for local development."""

        return self.client.create_plan(
            question,
            context=context,
        )


def create_llm_client() -> LLMClient:
    """
    Create the appropriate LLM client.

    Production:
        OPENAI_API_KEY configured -> OpenAILLMClient

    Development/testing:
        No API key -> deterministic FallbackLLMClient
    """

    settings = get_settings()

    if settings.openai_api_key:
        return OpenAILLMClient()

    return FallbackLLMClient()