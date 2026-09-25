"""
Structured state used by the Business Intelligence Agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """State carried through an agent execution."""

    user_question: str

    intent: str = ""

    plan: list[dict[str, Any]] = field(
        default_factory=list
    )

    tool_calls: list[dict[str, Any]] = field(
        default_factory=list
    )

    observations: list[dict[str, Any]] = field(
        default_factory=list
    )

    evidence: list[dict[str, Any]] = field(
        default_factory=list
    )

    trace: list[dict[str, Any]] = field(
        default_factory=list
    )

    final_answer: str = ""

    iteration: int = 0

    status: str = "initialized"

    error: str | None = None

    def add_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Record a tool invocation."""

        self.tool_calls.append(
            {
                "tool": tool_name,
                "arguments": arguments,
            }
        )

    def add_observation(
        self,
        tool_name: str,
        result: Any,
    ) -> None:
        """Record a tool result."""

        self.observations.append(
            {
                "tool": tool_name,
                "result": result,
            }
        )

    def add_evidence(
        self,
        source: str,
        data: Any,
    ) -> None:
        """Record evidence supporting the final answer."""

        self.evidence.append(
            {
                "source": source,
                "data": data,
            }
        )

    def add_trace(
        self,
        step: int,
        event: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Record an agent execution trace event."""

        self.trace.append(
            {
                "step": step,
                "event": event,
                "data": data or {},
            }
        )