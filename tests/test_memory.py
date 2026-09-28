from src.agent.memory import AgentMemory


def test_memory_starts_empty():
    """Memory should start with no conversation turns."""

    memory = AgentMemory()

    assert len(memory) == 0
    assert memory.get_turns() == []
    assert memory.last_turn() is None
    assert memory.get_context() == ""


def test_memory_stores_conversation_turn():
    """Memory should store a completed conversation turn."""

    memory = AgentMemory()

    memory.add_turn(
        question="Why did revenue change this month?",
        answer="Revenue increased by 7.37%.",
        intent="business_revenue_analysis",
    )

    assert len(memory) == 1

    turn = memory.last_turn()

    assert turn is not None
    assert turn.question == "Why did revenue change this month?"
    assert turn.answer == "Revenue increased by 7.37%."
    assert turn.intent == "business_revenue_analysis"


def test_memory_builds_context():
    """Memory should build readable context from recent turns."""

    memory = AgentMemory()

    memory.add_turn(
        question="Why did revenue change this month?",
        answer="Revenue increased by 7.37%.",
        intent="business_revenue_analysis",
    )

    memory.add_turn(
        question="Which category contributed most?",
        answer="Beauty generated the highest revenue.",
        intent="category_performance",
    )

    context = memory.get_context()

    assert "Recent conversation context:" in context
    assert "Why did revenue change this month?" in context
    assert "Revenue increased by 7.37%." in context
    assert "Which category contributed most?" in context
    assert "Beauty generated the highest revenue." in context


def test_memory_limits_recent_turns():
    """Memory should retain only the configured number of turns."""

    memory = AgentMemory(max_turns=2)

    memory.add_turn(
        "Question 1",
        "Answer 1",
    )

    memory.add_turn(
        "Question 2",
        "Answer 2",
    )

    memory.add_turn(
        "Question 3",
        "Answer 3",
    )

    assert len(memory) == 2

    turns = memory.get_turns()

    assert turns[0].question == "Question 2"
    assert turns[1].question == "Question 3"


def test_memory_clear():
    """Memory should be completely clearable."""

    memory = AgentMemory()

    memory.add_turn(
        "Question",
        "Answer",
    )

    assert len(memory) == 1

    memory.clear()

    assert len(memory) == 0
    assert memory.last_turn() is None
    assert memory.get_context() == ""


def test_memory_rejects_empty_question():
    """Memory should reject empty questions."""

    memory = AgentMemory()

    try:
        memory.add_turn(
            "",
            "Answer",
        )
        assert False, "Expected ValueError."

    except ValueError as exc:
        assert str(exc) == "Question cannot be empty."


def test_memory_rejects_empty_answer():
    """Memory should reject empty answers."""

    memory = AgentMemory()

    try:
        memory.add_turn(
            "Question",
            "",
        )
        assert False, "Expected ValueError."

    except ValueError as exc:
        assert str(exc) == "Answer cannot be empty."