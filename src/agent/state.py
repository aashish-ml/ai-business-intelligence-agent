"""
Structured state used by the Business Intelligence Agent.
"""

from __future__ import annotations

import time
import uuid
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

    trace_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    started_at: float = field(
        default_factory=time.perf_counter
    )

    completed_at: float | None = None

    total_duration_ms: float | None = None

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
                "trace_id": self.trace_id,
                "step": step,
                "event": event,
                "timestamp": time.time(),
                "data": data or {},
            }
        )

    def finish(self) -> None:
        """Mark execution as completed and calculate total duration."""

        self.completed_at = time.perf_counter()

        self.total_duration_ms = round(
            (self.completed_at - self.started_at) * 1000,
            2,
        )