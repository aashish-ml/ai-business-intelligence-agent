"""
Tool execution router.

This router consumes the canonical tool registry and provides
controlled execution of approved agent tools.
"""

from __future__ import annotations

from typing import Any

from src.tools.registry import (
    get_tool_definition,
    get_tool_names,
)


class ToolRouter:
    """Execution router backed by the canonical tool registry."""

    def __init__(self) -> None:
        self._tools = get_tool_names()

    def available_tools(self) -> list[str]:
        """Return names of all registered tools."""

        return list(self._tools)

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute an approved tool using the canonical registry."""

        if tool_name not in self._tools:
            raise ValueError(
                f"Unknown tool: {tool_name}"
            )

        tool = get_tool_definition(tool_name)

        return tool.function(**arguments)