from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationTurn:
    """Represents one user question and agent response."""

    question: str
    answer: str
    intent: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)


class AgentMemory:
    """
    Lightweight in-process conversation memory.

    Stores recent conversation turns so the agent can use
    previous business questions and answers as context.
    """

    def __init__(self, max_turns: int = 10) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than 0.")

        self.max_turns = max_turns
        self._turns: list[ConversationTurn] = []

    def add_turn(
        self,
        question: str,
        answer: str,
        intent: str = "",
        evidence: list[dict[str, Any]] | None = None,
    ) -> None:
        """Store a completed conversation turn."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        turn = ConversationTurn(
            question=question.strip(),
            answer=answer.strip(),
            intent=intent,
            evidence=evidence or [],
        )

        self._turns.append(turn)

        # Keep only the most recent turns.
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]

    def get_turns(self) -> list[ConversationTurn]:
        """Return a copy of stored conversation turns."""

        return list(self._turns)

    def get_recent_turns(
        self,
        limit: int = 5,
    ) -> list[ConversationTurn]:
        """Return the most recent conversation turns."""

        if limit <= 0:
            return []

        return self._turns[-limit:]

    def get_context(
        self,
        limit: int = 5,
    ) -> str:
        """
        Build a compact text representation of recent
        conversation history for planner/LLM context.
        """

        turns = self.get_recent_turns(limit)

        if not turns:
            return ""

        lines = [
            "Recent conversation context:",
        ]

        for index, turn in enumerate(turns, start=1):
            lines.append(
                f"{index}. User: {turn.question}"
            )

            lines.append(
                f"   Agent: {turn.answer}"
            )

            if turn.intent:
                lines.append(
                    f"   Intent: {turn.intent}"
                )

        return "\n".join(lines)

    def last_turn(self) -> ConversationTurn | None:
        """Return the most recent turn, if available."""

        if not self._turns:
            return None

        return self._turns[-1]

    def clear(self) -> None:
        """Clear all stored conversation history."""

        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)