from __future__ import annotations

import re
import time
from typing import Any

from genai.llm_client import create_llm_client
from genai.plan_validator import validate_plan
from genai.synthesis import AnswerSynthesizer

from src.agent.planner import BusinessPlanner
from src.agent.router import ToolRouter
from src.agent.state import AgentState
from src.agent.memory import AgentMemory

from src.tools.analysis_tool import group_and_aggregate
from src.tools.business_analysis_tool import analyze_business_metric
from src.tools.calculator_tool import percentage_change
from src.tools.ml_tool import MLModelTool
from src.tools.rag_tool import RAGTool
from src.tools.sql_tool import execute_sql_query


class BusinessIntelligenceAgent:
    """Production-oriented business intelligence agent."""

    MAX_ITERATIONS = 5
    MAX_TOOL_RETRIES = 1

    def __init__(self) -> None:
        self.router = ToolRouter()
        self.planner = BusinessPlanner()
        self.llm = create_llm_client()
        self.synthesizer = AnswerSynthesizer()
        self.memory = AgentMemory(max_turns=10)

        self.ml_tool = MLModelTool()
        self.rag_tool = RAGTool()

        self.router.register(
            "ml_prediction",
            "Run predictions using a pre-trained machine learning model.",
            self.ml_tool.predict,
        )

        self.router.register(
            "rag_search",
            "Search business policies and knowledge documents using semantic retrieval.",
            self.rag_tool.search,
        )

        self.router.register(
            "customer_risk_prediction",
            "Predict customer churn/risk directly from a customer ID.",
            self.ml_tool.predict_customer,
        )

        self._register_tools()

    def _register_tools(self) -> None:
        """Register all approved agent tools."""

        # ---------------------------------------------------------
        # Business analysis
        # ---------------------------------------------------------
        self.router.register(
            "business_analysis",
            (
                "Perform deterministic business analysis such as "
                "revenue change, revenue trends, category performance, "
                "top products, segment performance, and order status."
            ),
            analyze_business_metric,
        )

        # ---------------------------------------------------------
        # Safe SQL
        # ---------------------------------------------------------
        self.router.register(
            "execute_sql",
            "Execute a safe read-only SQL query.",
            execute_sql_query,
        )

        # Backward-compatible alias for existing flows/tests.
        self.router.register(
            "sql_query",
            "Execute a safe read-only SQL query.",
            execute_sql_query,
        )

        # ---------------------------------------------------------
        # Calculation tools
        # ---------------------------------------------------------
        self.router.register(
            "percentage_change",
            "Calculate percentage change between two values.",
            percentage_change,
        )

        self.router.register(
            "group_and_aggregate",
            "Group business data and calculate an aggregation.",
            group_and_aggregate,
        )

    def create_state(self, question: str) -> AgentState:
        """Create a new agent execution state."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        return AgentState(
            user_question=question.strip()
        )

    def validate_tool_result(
        self,
        tool_name: str,
        result: Any,
    ) -> tuple[bool, str]:
        """
        Validate the structural integrity of a tool result.

        This validation does not judge whether the business answer
        is correct. It only verifies that the tool returned a
        usable result that the agent can safely process.
        """

        if result is None:
            return False, "Tool returned no result."

        if not isinstance(result, dict):
            return True, "Tool returned a non-dictionary result."

        # Explicit tool failure.
        if result.get("success") is False:
            error = result.get(
                "error",
                "Tool reported an unsuccessful execution.",
            )

            return False, str(error)

        # SQL-style results must contain rows.
        if tool_name in {
            "execute_sql",
            "sql_query",
        }:
            if "rows" not in result:
                return (
                    False,
                    "SQL tool result is missing the 'rows' field.",
                )

            if not isinstance(result["rows"], list):
                return (
                    False,
                    "SQL tool 'rows' field must be a list.",
                )

        # Business-analysis results must contain either
        # structured data or a successful metric result.
        if tool_name == "business_analysis":

            if "data" in result:
                if not isinstance(result["data"], list):
                    return (
                        False,
                        "Business analysis 'data' field "
                        "must be a list.",
                    )

            elif result.get("status") == "success":
                pass

            else:
                return (
                    False,
                    "Business analysis result does not "
                    "contain usable structured data.",
                )

        # RAG results must contain a results list when successful.
        if tool_name == "rag_search":

            if "results" not in result:
                return (
                    False,
                    "RAG result is missing the 'results' field.",
                )

            if not isinstance(result["results"], list):
                return (
                    False,
                    "RAG 'results' field must be a list.",
                )

        # Customer-risk predictions must contain the core
        # prediction fields when successful.
        if tool_name == "customer_risk_prediction":

            required_fields = {
                "customer_id",
                "risk_probability",
                "risk_level",
            }

            missing = [
                field
                for field in required_fields
                if field not in result
            ]

            if missing:
                return (
                    False,
                    "Customer risk result is missing fields: "
                    + ", ".join(sorted(missing))
                    + ".",
                )

        return True, "Tool result passed validation."

    def validate_answer(
        self,
        answer: str,
        evidence: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        """
        Validate that a generated answer is non-empty and
        numerically grounded in the available evidence.

        This is a lightweight deterministic guardrail. It does
        not attempt to judge business correctness or language
        quality. Its purpose is to prevent unsupported numerical
        claims from reaching the user.
        """

        if not answer or not answer.strip():
            return False, "Answer is empty."

        if not evidence:
            return False, "Answer has no supporting evidence."

        # ---------------------------------------------------------
        # Collect numeric values from verified evidence.
        # ---------------------------------------------------------

        evidence_numbers: set[float] = set()

        def collect_numbers(value: Any) -> None:
            if isinstance(value, bool):
                return

            if isinstance(value, (int, float)):
                evidence_numbers.add(round(float(value), 6))
                return

            if isinstance(value, dict):
                for nested_value in value.values():
                    collect_numbers(nested_value)
                return

            if isinstance(value, list):
                for nested_value in value:
                    collect_numbers(nested_value)

        for item in evidence:
            collect_numbers(item)

        if not evidence_numbers:
            return (
                False,
                "Evidence contains no numeric values "
                "to support the answer.",
            )

        # ---------------------------------------------------------
        # Extract numeric business claims.
        #
        # Ignore date components such as:
        #   2025-11
        #   2025-12
        #
        # These are temporal references, not business metrics.
        # ---------------------------------------------------------

        date_spans = re.findall(
            r"\b\d{4}-\d{1,2}(?:-\d{1,2})?\b",
            answer,
        )

        answer_without_dates = re.sub(
            r"\b\d{4}-\d{1,2}(?:-\d{1,2})?\b",
            "",
            answer,
        )

        # Product identifiers such as "Product 49" are entity
        # identifiers, not numerical business claims.
        answer_without_dates = re.sub(
            r"\bproduct\s+\d+\b",
            "product",
            answer_without_dates,
            flags=re.IGNORECASE,
        )

        numeric_tokens = re.findall(
        r"-?\d[\d,]*(?:\.\d+)?",
        answer_without_dates,
    )

        answer_numbers: list[float] = []

        for token in numeric_tokens:

            normalized = token.replace(",", "")

            try:
                answer_numbers.append(
                    round(float(normalized), 6)
                )
            except ValueError:
                continue

        # ---------------------------------------------------------
        # Every numeric claim must be present in the evidence.
        # A small tolerance handles floating-point representation.
        # ---------------------------------------------------------

        tolerance = 0.01

        for number in answer_numbers:

            supported = any(
                abs(number - evidence_number)
                <= tolerance
                or abs(
                    (number / 100) - evidence_number
                )
                <= tolerance
                for evidence_number in evidence_numbers
            )

            if not supported:
                return (
                    False,
                    f"Unsupported numeric claim: {number:g}.",
                )

        return (
            True,
            "Answer passed evidence validation.",
        )

    def execute_tool_with_recovery(
        self,
        state: AgentState,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Execute a tool with one controlled recovery retry.

        The first execution is attempted normally. If the tool
        fails or returns an invalid result, the agent retries the
        same tool once. A second failure is propagated to the
        main execution handler.
        """

        attempt = 0

        while attempt <= self.MAX_TOOL_RETRIES:

            try:
                if attempt > 0:
                    state.add_trace(
                        state.iteration + 1,
                        "TOOL_RECOVERY_RETRY",
                        {
                            "tool": tool_name,
                            "attempt": attempt + 1,
                            "max_retries": self.MAX_TOOL_RETRIES,
                        },
                    )

                return self.execute_tool(
                    state,
                    tool_name,
                    arguments,
                )

            except Exception as exc:

                if attempt >= self.MAX_TOOL_RETRIES:
                    state.add_trace(
                        state.iteration + 1,
                        "TOOL_RECOVERY_EXHAUSTED",
                        {
                            "tool": tool_name,
                            "attempts": attempt + 1,
                            "error": str(exc),
                        },
                    )

                    raise

                state.add_trace(
                    state.iteration + 1,
                    "TOOL_RECOVERY_TRIGGERED",
                    {
                        "tool": tool_name,
                        "attempt": attempt + 1,
                        "error": str(exc),
                    },
                )

                attempt += 1

        raise RuntimeError(
            f"Tool recovery failed unexpectedly for '{tool_name}'."
        )

    def execute_tool(
        self,
        state: AgentState,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute a registered tool and record observability data."""

        if state.iteration >= self.MAX_ITERATIONS:
            state.add_trace(
                state.iteration + 1,
                "MAX_ITERATIONS_REACHED",
                {
                    "tool": tool_name,
                    "max_iterations": self.MAX_ITERATIONS,
                },
            )

            raise RuntimeError(
                f"Maximum agent iterations "
                f"({self.MAX_ITERATIONS}) exceeded."
            )

        state.add_tool_call(
            tool_name,
            arguments,
        )

        tool_start = time.perf_counter()

        state.add_trace(
            state.iteration + 1,
            "TOOL_STARTED",
            {
                "tool": tool_name,
                "arguments": arguments,
            },
        )

        try:
            result = self.router.execute(
                tool_name,
                arguments,
            )
        except Exception as exc:
            duration_ms = round(
                (time.perf_counter() - tool_start) * 1000,
                2,
            )

            state.add_trace(
                state.iteration + 1,
                "TOOL_FAILED",
                {
                    "tool": tool_name,
                    "duration_ms": duration_ms,
                    "error": str(exc),
                },
            )

            raise

        duration_ms = round(
            (time.perf_counter() - tool_start) * 1000,
            2,
        )

        state.iteration += 1

        # ---------------------------------------------------------
        # Validate the tool result before recording it as usable
        # agent evidence.
        # ---------------------------------------------------------
        is_valid, validation_message = self.validate_tool_result(
            tool_name,
            result,
        )

        state.add_trace(
            state.iteration,
            "TOOL_RESULT_VALIDATED",
            {
                "tool": tool_name,
                "valid": is_valid,
                "message": validation_message,
            },
        )

        if not is_valid:
            state.add_trace(
                state.iteration,
                "TOOL_RESULT_INVALID",
                {
                    "tool": tool_name,
                    "error": validation_message,
                },
            )

            raise RuntimeError(
                f"Invalid result from tool '{tool_name}': "
                f"{validation_message}"
            )

        state.add_observation(
            tool_name,
            result,
        )

        state.add_trace(
            state.iteration,
            "TOOL_COMPLETED",
            {
                "tool": tool_name,
                "duration_ms": duration_ms,
                "success": (
                    result.get("success", True)
                    if isinstance(result, dict)
                    else True
                ),
            },
        )

        return result

    def run(self, question: str) -> AgentState:
        """
        Run the complete business intelligence agent.

        Workflow:

        Question
            ↓
        Planner
            ↓
        Tool Selection
            ↓
        Tool Execution
            ↓
        Evidence
            ↓
        Synthesis
            ↓
        Final Answer
        """

        state = self.create_state(question)

        try:
            # -----------------------------------------------------
            # 1. Build conversation memory context.
            # -----------------------------------------------------
            conversation_context = self.memory.get_context()

            state.add_trace(
                0,
                "MEMORY_CONTEXT_RETRIEVED",
                {
                    "memory_context_used": bool(conversation_context),
                    "memory_turns": len(self.memory),
                },
            )

            # -----------------------------------------------------
            # 2. Generate structured plan using memory context.
            # -----------------------------------------------------
            plan = self.llm.create_plan(
                question,
                context=conversation_context,
            )

            # Record the plan produced by the planning layer.
            state.add_trace(
                0,
                "PLAN_CREATED",
                {
                    "intent": plan.intent,
                    "memory_context_used": bool(
                        conversation_context
                    ),
                    "memory_turns": len(self.memory),
                    "plan": [
                        {
                            "tool": tool_call.tool,
                            "arguments": tool_call.arguments,
                            "purpose": tool_call.purpose,
                        }
                        for tool_call in plan.tool_calls
                    ],
                },
            )

            # Validate the structured plan before any tool execution.
            plan = validate_plan(plan)

            state.add_trace(
                0,
                "PLAN_VALIDATED",
                {
                    "intent": plan.intent,
                    "tool_count": len(plan.tool_calls),
                    "tools": [
                        tool_call.tool
                        for tool_call in plan.tool_calls
                    ],
                    "validation_status": "passed",
                },
            )

            state.intent = plan.intent

            # -----------------------------------------------------
            # 2. Execute planned tools.
            # -----------------------------------------------------
            for tool_call in plan.tool_calls:

                if state.iteration >= self.MAX_ITERATIONS:
                    state.status = "max_iterations"
                    state.error = (
                        f"Maximum iterations "
                        f"({self.MAX_ITERATIONS}) reached."
                    )

                    state.finish()

                    state.add_trace(
                        state.iteration + 1,
                        "MAX_ITERATIONS_REACHED",
                        {
                            "iteration": state.iteration,
                            "max_iterations": self.MAX_ITERATIONS,
                            "total_duration_ms": (
                                state.total_duration_ms
                            ),
                        },
                    )

                    return state

                result = self.execute_tool_with_recovery(
                    state,
                    tool_call.tool,
                    tool_call.arguments,
                )

                # -------------------------------------------------
                # Record evidence from planned tool.
                # -------------------------------------------------
                state.add_evidence(
                    tool_call.tool,
                    {
                        "purpose": tool_call.purpose,
                        "result": result,
                    },
                )

                state.add_trace(
                    state.iteration,
                    "EVIDENCE_ADDED",
                    {
                        "source": tool_call.tool,
                    },
                )

                # -------------------------------------------------
                # Multi-step revenue analysis.
                # -------------------------------------------------
                if (
                    state.intent in {
                        "business_analysis",
                        "business_revenue_analysis",
                    }
                    and tool_call.tool == "sql_query"
                ):
                    rows = result.get("rows", [])

                    if len(rows) >= 2:
                        previous = rows[-2]
                        current = rows[-1]

                        previous_revenue = float(
                            previous["revenue"]
                        )

                        current_revenue = float(
                            current["revenue"]
                        )

                        percentage_result = self.execute_tool_with_recovery(
                                            state,
                                            "percentage_change",
                                            {
                                                "old_value": previous_revenue,
                                                "new_value": current_revenue,
                                            },
                                        )

                        state.add_evidence(
                            "percentage_change",
                            {
                                "previous_month": previous["month"],
                                "current_month": current["month"],
                                "previous_revenue": previous_revenue,
                                "current_revenue": current_revenue,
                                "percentage_change": percentage_result,
                            },
                        )

                        state.add_trace(
                            state.iteration,
                            "EVIDENCE_ADDED",
                            {
                                "source": "percentage_change",
                            },
                        )

                # -------------------------------------------------
                # Multi-step customer risk → policy workflow.
                # -------------------------------------------------
                if (
                    state.intent == "customer_risk_policy"
                    and tool_call.tool == "customer_risk_prediction"
                ):
                    risk_result = result

                    if risk_result.get("success"):
                        risk_level = risk_result.get(
                            "risk_level",
                            "unknown",
                        )

                        rag_query = (
                            f"What should we do with a "
                            f"{risk_level}-risk customer?"
                        )

                        rag_result = self.execute_tool_with_recovery(
                                        state,
                                        "rag_search",
                                        {
                                            "query": rag_query,
                                            "top_k": 5,
                                        },
                                    )

                        state.add_evidence(
                            "rag_search",
                            {
                                "purpose": (
                                    "Retrieve business policy "
                                    "relevant to the predicted "
                                    "customer risk level."
                                ),
                                "result": rag_result,
                            },
                        )

                        state.add_trace(
                            state.iteration,
                            "EVIDENCE_ADDED",
                            {
                                "source": "rag_search",
                                "risk_level": risk_level,
                                "query": rag_query,
                            },
                        )

            # -----------------------------------------------------
            # 3. Validate iteration limit.
            # -----------------------------------------------------
            if state.iteration > self.MAX_ITERATIONS:
                state.status = "failed"
                state.error = (
                    f"Maximum iterations "
                    f"({self.MAX_ITERATIONS}) reached."
                )

                state.finish()

                state.add_trace(
                    state.iteration,
                    "MAX_ITERATIONS_REACHED",
                    {
                        "max_iterations": self.MAX_ITERATIONS,
                        "total_duration_ms": (
                            state.total_duration_ms
                        ),
                    },
                )

                return state

            # -----------------------------------------------------
            # 4. Generate final answer.
            # -----------------------------------------------------
            state.final_answer = self.synthesizer.synthesize(
                question=question,
                intent=state.intent,
                evidence=state.evidence,
            )

            state.add_trace(
                state.iteration,
                "FINAL_ANSWER_GENERATED",
                {
                    "answer": state.final_answer,
                },
            )

            # -----------------------------------------------------
            # 5. Validate the generated answer against evidence.
            # -----------------------------------------------------
            answer_valid, validation_message = self.validate_answer(
                state.final_answer,
                state.evidence,
            )

            state.add_trace(
                state.iteration,
                "ANSWER_VALIDATED",
                {
                    "valid": answer_valid,
                    "message": validation_message,
                },
            )

            if not answer_valid:
                state.status = "failed"
                state.error = validation_message

                state.finish()

                state.add_trace(
                    state.iteration,
                    "ANSWER_VALIDATION_FAILED",
                    {
                        "error": validation_message,
                        "total_duration_ms": (
                            state.total_duration_ms
                        ),
                    },
                )

                return state

            # -----------------------------------------------------
            # 5. Persist the completed turn in conversation memory.
            # -----------------------------------------------------
            self.memory.add_turn(
                question=question,
                answer=state.final_answer,
                intent=state.intent,
                evidence=state.evidence,
            )

            state.add_trace(
                state.iteration,
                "MEMORY_TURN_SAVED",
                {
                    "memory_turns": len(self.memory),
                    "intent": state.intent,
                },
            )

            state.add_trace(
            state.iteration,
            "MEMORY_UPDATED",
            {
                "memory_turns": len(self.memory),
                "intent": state.intent,
            },
)

            # -----------------------------------------------------
            # 6. Mark successful execution.
            # -----------------------------------------------------
            state.status = "completed"

            state.finish()

            state.add_trace(
                state.iteration,
                "EXECUTION_COMPLETED",
                {
                    "status": state.status,
                    "iterations": state.iteration,
                    "total_duration_ms": (
                        state.total_duration_ms
                    ),
                },
            )

            return state

        except Exception as exc:
            state.status = "failed"
            state.error = str(exc)

            state.finish()

            state.add_trace(
                state.iteration,
                "EXECUTION_FAILED",
                {
                    "error": str(exc),
                    "total_duration_ms": (
                        state.total_duration_ms
                    ),
                },
            )

            return state