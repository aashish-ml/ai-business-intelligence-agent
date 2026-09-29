"""
Tests for API-level agent session memory.
"""

from src.agent.session_memory import session_store


def setup_function():
    """Start every test with a clean session store."""

    session_store.clear_all()


def teardown_function():
    """Clean up sessions after every test."""

    session_store.clear_all()


def test_session_store_starts_empty():
    """Session store should start with no active sessions."""

    assert session_store.session_count() == 0


def test_same_session_reuses_agent():
    """The same session ID should return the same agent instance."""

    agent_one = session_store.get_agent("demo-session")
    agent_two = session_store.get_agent("demo-session")

    assert agent_one is agent_two
    assert session_store.session_count() == 1


def test_different_sessions_are_isolated():
    """Different session IDs should have separate agent instances."""

    agent_one = session_store.get_agent("session-one")
    agent_two = session_store.get_agent("session-two")

    assert agent_one is not agent_two
    assert session_store.session_count() == 2


def test_session_memory_is_preserved():
    """Conversation memory should remain available inside a session."""

    agent = session_store.get_agent("memory-session")

    agent.memory.add_turn(
        question="Why did revenue change this month?",
        answer="Revenue increased by 7.37%.",
        intent="business_revenue_analysis",
    )

    same_agent = session_store.get_agent("memory-session")

    assert len(same_agent.memory) == 1
    assert (
        same_agent.memory.last_turn().question
        == "Why did revenue change this month?"
    )


def test_clear_session_removes_agent():
    """Clearing a session should remove its agent and memory."""

    agent = session_store.get_agent("clear-session")

    agent.memory.add_turn(
        question="Test question",
        answer="Test answer",
    )

    assert session_store.session_exists("clear-session")
    assert len(agent.memory) == 1

    session_store.clear_session("clear-session")

    assert not session_store.session_exists("clear-session")
    assert session_store.session_count() == 0


def test_cleared_session_gets_new_agent():
    """A cleared session should create a fresh agent instance."""

    original_agent = session_store.get_agent("recreate-session")

    session_store.clear_session("recreate-session")

    new_agent = session_store.get_agent("recreate-session")

    assert original_agent is not new_agent
    assert len(new_agent.memory) == 0
    assert session_store.session_count() == 1