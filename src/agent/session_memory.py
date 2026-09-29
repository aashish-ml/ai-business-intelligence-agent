"""
Session-based memory store for the Business Intelligence Agent API.
"""

from __future__ import annotations

from threading import Lock

from src.agent.agent import BusinessIntelligenceAgent


class AgentSessionStore:
    """Maintain one agent instance per conversation session."""

    def __init__(self) -> None:
        self._sessions: dict[str, BusinessIntelligenceAgent] = {}
        self._lock = Lock()

    def get_agent(self, session_id: str) -> BusinessIntelligenceAgent:
        """
        Return the agent associated with a session.

        A new agent is created only when the session does not exist.
        """

        if not session_id.strip():
            raise ValueError("Session ID cannot be empty.")

        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = BusinessIntelligenceAgent()

            return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        """Remove a session and its conversation memory."""

        with self._lock:
            self._sessions.pop(session_id, None)

    def clear_all(self) -> None:
        """Remove all active sessions."""

        with self._lock:
            self._sessions.clear()

    def session_exists(self, session_id: str) -> bool:
        """Return whether a session currently exists."""

        with self._lock:
            return session_id in self._sessions

    def session_count(self) -> int:
        """Return the number of active sessions."""

        with self._lock:
            return len(self._sessions)


session_store = AgentSessionStore()