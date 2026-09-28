"""
Agent-level tests for tool-result validation.
"""

from src.agent.agent import BusinessIntelligenceAgent


def test_tool_result_validation_accepts_valid_result():
    """Valid tool results should pass validation."""

    agent = BusinessIntelligenceAgent()

    is_valid, message = agent.validate_tool_result(
        "business_analysis",
        {
            "data": [
                {
                    "category": "Beauty",
                    "revenue": 1000,
                }
            ]
        },
    )

    assert is_valid is True
    assert message == "Tool result passed validation."


def test_tool_result_validation_rejects_none():
    """None results should be rejected."""

    agent = BusinessIntelligenceAgent()

    is_valid, message = agent.validate_tool_result(
        "business_analysis",
        None,
    )

    assert is_valid is False
    assert message == "Tool returned no result."


def test_tool_result_validation_rejects_failed_tool():
    """Explicit tool failures should be rejected."""

    agent = BusinessIntelligenceAgent()

    is_valid, message = agent.validate_tool_result(
        "business_analysis",
        {
            "success": False,
            "error": "Simulated tool failure.",
        },
    )

    assert is_valid is False
    assert message == "Simulated tool failure."


def test_tool_result_validation_rejects_invalid_sql_result():
    """SQL results without rows should be rejected."""

    agent = BusinessIntelligenceAgent()

    is_valid, message = agent.validate_tool_result(
        "sql_query",
        {
            "success": True,
        },
    )

    assert is_valid is False
    assert message == (
        "SQL tool result is missing the 'rows' field."
    )

def test_tool_recovery_retries_failed_tool():
    """Failed tool execution should be retried once."""

    agent = BusinessIntelligenceAgent()

    state = agent.create_state(
        "Test tool recovery"
    )

    original_execute = agent.router.execute
    call_count = {"value": 0}

    def flaky_execute(tool_name, arguments):
        call_count["value"] += 1

        if call_count["value"] == 1:
            raise RuntimeError(
                "Simulated transient tool failure."
            )

        return {
            "data": [
                {
                    "category": "Beauty",
                    "revenue": 1000,
                }
            ]
        }

    agent.router.execute = flaky_execute

    try:
        result = agent.execute_tool_with_recovery(
            state,
            "business_analysis",
            {
                "metric": "category_performance"
            },
        )
    finally:
        agent.router.execute = original_execute

    assert result["data"][0]["category"] == "Beauty"
    assert call_count["value"] == 2

    trace_events = [
        item["event"]
        for item in state.trace
    ]

    assert "TOOL_FAILED" in trace_events
    assert "TOOL_RECOVERY_TRIGGERED" in trace_events
    assert "TOOL_RECOVERY_RETRY" in trace_events
    assert "TOOL_RESULT_VALIDATED" in trace_events
    assert "TOOL_COMPLETED" in trace_events

def test_tool_recovery_exhausts_after_max_retries():
    """Recovery should stop after the configured retry limit."""

    agent = BusinessIntelligenceAgent()

    state = agent.create_state(
        "Test recovery exhaustion"
    )

    original_execute = agent.router.execute
    call_count = {"value": 0}

    def always_failing_execute(tool_name, arguments):
        call_count["value"] += 1

        raise RuntimeError(
            "Simulated persistent tool failure."
        )

    agent.router.execute = always_failing_execute

    try:
        try:
            agent.execute_tool_with_recovery(
                state,
                "business_analysis",
                {
                    "metric": "category_performance"
                },
            )

            assert False, (
                "Expected tool recovery to raise an exception."
            )

        except RuntimeError as exc:
            assert (
                "Simulated persistent tool failure."
                in str(exc)
            )

    finally:
        agent.router.execute = original_execute

    assert call_count["value"] == 2

    trace_events = [
        item["event"]
        for item in state.trace
    ]

    assert "TOOL_FAILED" in trace_events
    assert "TOOL_RECOVERY_TRIGGERED" in trace_events
    assert "TOOL_RECOVERY_RETRY" in trace_events
    assert "TOOL_RECOVERY_EXHAUSTED" in trace_events

def test_agent_stores_successful_turn_in_memory():
    """Successful agent executions should be stored in memory."""

    agent = BusinessIntelligenceAgent()

    assert len(agent.memory) == 0

    state = agent.run(
        "Why did revenue change this month?"
    )

    assert state.status == "completed"
    assert len(agent.memory) == 1

    turn = agent.memory.last_turn()

    assert turn is not None
    assert (
        turn.question
        == "Why did revenue change this month?"
    )
    assert "Revenue increased" in turn.answer
    assert turn.intent == "business_revenue_analysis"

    trace_events = [
        item["event"]
        for item in state.trace
    ]

    assert "MEMORY_UPDATED" in trace_events


def test_agent_memory_keeps_multiple_turns():
    """Agent memory should retain multiple successful turns."""

    agent = BusinessIntelligenceAgent()

    first = agent.run(
        "Why did revenue change this month?"
    )

    second = agent.run(
        "Which category contributed most?"
    )

    assert first.status == "completed"
    assert second.status == "completed"

    assert len(agent.memory) == 2

    turns = agent.memory.get_turns()

    assert (
        turns[0].question
        == "Why did revenue change this month?"
    )

    assert (
        turns[1].question
        == "Which category contributed most?"
    )


def test_agent_passes_memory_context_to_planner():
    """Previous conversation should be passed to the planner."""

    agent = BusinessIntelligenceAgent()

    captured = {}

    original_create_plan = agent.llm.create_plan

    def capture_create_plan(
        question,
        context="",
    ):
        captured["question"] = question
        captured["context"] = context

        return original_create_plan(
            question,
            context=context,
        )

    agent.llm.create_plan = capture_create_plan

    first = agent.run(
        "Why did revenue change this month?"
    )

    assert first.status == "completed"

    second = agent.run(
        "Which category contributed most?"
    )

    assert second.status == "completed"

    assert captured["question"] == (
        "Which category contributed most?"
    )

    assert (
        "Why did revenue change this month?"
        in captured["context"]
    )

    assert (
        "Revenue increased by 7.37%"
        in captured["context"]
    )

    trace_events = [
        item["event"]
        for item in second.trace
    ]

    assert "PLAN_CREATED" in trace_events

    plan_created = next(
        item
        for item in second.trace
        if item["event"] == "PLAN_CREATED"
    )

    assert plan_created["data"]["memory_context_used"] is True
    assert plan_created["data"]["memory_turns"] == 1