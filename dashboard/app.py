"""
Streamlit frontend for the AI Business Intelligence Agent.
"""

import requests
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Business Intelligence Agent",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

API_BASE_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------
# API helpers
# ---------------------------------------------------------

def check_api_health() -> bool:
    """Check whether the FastAPI backend is available."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def ask_agent(question: str) -> dict:
    """Send a natural-language question to the agent API."""

    response = requests.post(
        f"{API_BASE_URL}/agent/query",
        json={
            "question": question,
        },
        timeout=120,
    )

    if response.status_code != 200:
        try:
            detail = response.json().get(
                "detail",
                "Agent request failed.",
            )
        except Exception:
            detail = response.text

        raise RuntimeError(str(detail))

    return response.json()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.title("🤖 AI Business Analyst")

    st.caption(
        "AI-Powered Business Intelligence & "
        "Decision Support Agent"
    )

    st.divider()

    st.subheader("Backend")

    if check_api_health():
        st.success("API Connected")
    else:
        st.error("API Offline")

    st.caption(
        f"Backend: {API_BASE_URL}"
    )

    st.divider()

    st.subheader("Capabilities")

    st.markdown(
        """
        - 🧠 AI Agent Reasoning
        - 🗄️ SQL Analytics
        - 📊 Data Analysis
        - 🤖 ML Risk Prediction
        - 📚 RAG Knowledge Search
        - 🔎 Evidence-Grounded Answers
        - 🧭 Agent Execution Trace
        """
    )


# ---------------------------------------------------------
# Main page
# ---------------------------------------------------------

st.title("AI Business Intelligence & Decision Support Agent")

st.markdown(
    """
Ask business questions in natural language.

The AI agent can combine **SQL analytics, data analysis,
machine learning, and business-policy knowledge** to
produce evidence-grounded answers.
"""
)

st.divider()


# ---------------------------------------------------------
# Query input
# ---------------------------------------------------------

st.subheader("💬 Ask the AI Business Analyst")

question = st.text_area(
    "Business Question",
    placeholder=(
        "Example: What should we do about customer 7 "
        "based on their churn risk?"
    ),
    height=120,
)


if st.button(
    "🚀 Analyze",
    type="primary",
    use_container_width=True,
):

    if not question.strip():
        st.warning(
            "Please enter a business question."
        )

    else:

        with st.spinner(
            "AI agent is analyzing your question..."
        ):

            try:
                result = ask_agent(
                    question.strip()
                )

                st.session_state["agent_result"] = result

            except Exception as exc:

                st.error(
                    f"Agent request failed: {exc}"
                )


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------

result = st.session_state.get(
    "agent_result"
)


if result:

    st.divider()

    st.subheader("🤖 AI Analyst Answer")

    st.markdown(
        result["answer"]
    )

    st.divider()

    # -----------------------------------------------------
    # Execution metrics
    # -----------------------------------------------------

    st.subheader("⚡ Execution Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Status",
            result["status"],
        )

    with col2:
        st.metric(
            "Intent",
            result["intent"],
        )

    with col3:
        st.metric(
            "Iterations",
            result["iterations"],
        )

    with col4:
        duration = result["total_duration_ms"]

        st.metric(
            "Execution Time",
            f"{duration:.2f} ms"
            if duration is not None
            else "N/A",
        )

    # -----------------------------------------------------
    # Evidence
    # -----------------------------------------------------

    if result.get("evidence"):

        st.divider()

        st.subheader("📚 Evidence")

        for index, evidence in enumerate(
            result["evidence"],
            start=1,
        ):

            with st.expander(
                f"Evidence {index}: {evidence['source']}"
            ):

                st.json(
                    evidence["data"]
                )

    # -----------------------------------------------------
    # Trace
    # -----------------------------------------------------

    if result.get("trace"):

        st.divider()

        st.subheader("🧭 Agent Execution Trace")

        for event in result["trace"]:

            st.write(
                f"**Step {event['step']} — "
                f"{event['event']}**"
            )

            if event.get("data"):

                with st.expander(
                    "View details"
                ):

                    st.json(
                        event["data"]
                    )

    # -----------------------------------------------------
    # Trace ID
    # -----------------------------------------------------

    st.divider()

    st.caption(
        f"Trace ID: {result['trace_id']}"
    )