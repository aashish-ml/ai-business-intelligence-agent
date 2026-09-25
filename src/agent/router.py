"""
Deterministic tool router for the Business Intelligence Agent.

The router provides a safe baseline before LLM-based planning is
introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any


@dataclass(frozen=True)
class ToolDefinition:
    """Metadata describing an available agent tool."""

    name: str
    description: str
    function: Callable[..., Any]


class ToolRouter:
    """Registry and dispatcher for approved agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        function: Callable[..., Any],
    ) -> None:
        """Register an approved tool."""

        if not name.strip():
            raise ValueError("Tool name cannot be empty.")

        if name in self._tools:
            raise ValueError(
                f"Tool already registered: {name}"
            )

        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            function=function,
        )

    def get_tool(
        self,
        name: str,
    ) -> ToolDefinition:
        """Return a registered tool."""

        if name not in self._tools:
            raise KeyError(
                f"Unknown or unregistered tool: {name}"
            )

        return self._tools[name]

    def list_tools(self) -> list[str]:
        """Return registered tool names."""

        return list(self._tools.keys())

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute an approved tool."""

        tool = self.get_tool(name)

        return tool.function(**arguments)