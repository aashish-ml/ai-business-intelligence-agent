from __future__ import annotations

import time
from typing import Any

from genai.mock_llm import MockLLMClient
from genai.synthesis import AnswerSynthesizer

from src.agent.planner import BusinessPlanner
from src.agent.router import ToolRouter
from src.agent.state import AgentState

from src.tools.analysis_tool import group_and_aggregate
from src.tools.business_analysis_tool import analyze_business_metric
from src.tools.calculator_tool import percentage_change
from src.tools.ml_tool import MLModelTool
from src.tools.rag_tool import RAGTool
from src.tools.sql_tool import execute_sql_query


class BusinessIntelligenceAgent:
    """Production-oriented business intelligence agent."""

    MAX_ITERATIONS = 5

    def __init__(self) -> None:
        self.router = ToolRouter()
        self.planner = BusinessPlanner()
        self.llm = MockLLMClient()
        self.synthesizer = AnswerSynthesizer()

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
            # 1. Generate structured plan.
            # -----------------------------------------------------
            plan = self.llm.create_plan(question)

            state.intent = plan.intent

            state.plan = [
                {
                    "tool": call.tool,
                    "arguments": call.arguments,
                    "purpose": call.purpose,
                }
                for call in plan.tool_calls
            ]

            state.add_trace(
                0,
                "PLAN_CREATED",
                {
                    "intent": state.intent,
                    "plan": state.plan,
                },
            )

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

                result = self.execute_tool(
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
                    state.intent == "business_analysis"
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

                        percentage_result = self.execute_tool(
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

                        rag_result = self.execute_tool(
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
            # 5. Mark successful execution.
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