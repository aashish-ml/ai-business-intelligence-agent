"""
Dedicated Agent Execution Trace page.
"""

import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Agent Trace",
    page_icon="🧭",
    layout="wide",
)


# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------

st.title("🧭 Agent Execution Trace")

st.markdown(
    """
Inspect how the AI Business Intelligence Agent processed
the latest business question.

This page exposes the agent's execution lifecycle,
tool usage, evidence collection, and final answer generation.
"""
)

st.divider()


# ---------------------------------------------------------
# Retrieve latest result
# ---------------------------------------------------------

result = st.session_state.get(
    "agent_result"
)


if not result:

    st.info(
        "No agent execution is available yet."
    )

    st.markdown(
        """
        Go to **AI Business Intelligence Agent**,
        submit a business question, and return here
        to inspect the execution trace.
        """
    )

    st.stop()


# ---------------------------------------------------------
# Execution overview
# ---------------------------------------------------------

st.subheader("⚡ Execution Overview")

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

    duration = result.get(
        "total_duration_ms"
    )

    st.metric(
        "Duration",
        (
            f"{duration:.2f} ms"
            if duration is not None
            else "N/A"
        ),
    )


# ---------------------------------------------------------
# Trace ID
# ---------------------------------------------------------

st.caption(
    f"Trace ID: {result['trace_id']}"
)

st.divider()


# ---------------------------------------------------------
# Original question
# ---------------------------------------------------------

st.subheader("💬 User Question")

st.info(
    result["question"]
)

st.divider()


# ---------------------------------------------------------
# Final answer
# ---------------------------------------------------------

st.subheader("🤖 Final Answer")

st.markdown(
    result["answer"]
)

st.divider()


# ---------------------------------------------------------
# Evidence
# ---------------------------------------------------------

st.subheader("📚 Evidence Collected")

evidence = result.get(
    "evidence",
    []
)

if not evidence:

    st.info(
        "No evidence was recorded."
    )

else:

    for index, item in enumerate(
        evidence,
        start=1,
    ):

        source = item.get(
            "source",
            "Unknown",
        )

        with st.expander(
            f"Evidence {index} — {source}"
        ):

            st.json(
                item.get(
                    "data",
                    {},
                )
            )


st.divider()


# ---------------------------------------------------------
# Execution timeline
# ---------------------------------------------------------

st.subheader("⏱️ Execution Timeline")

trace = result.get(
    "trace",
    []
)

if not trace:

    st.info(
        "No trace events were recorded."
    )

else:

    for event in trace:

        event_name = event.get(
            "event",
            "UNKNOWN",
        )

        step = event.get(
            "step",
            0,
        )

        timestamp = event.get(
            "timestamp",
            0,
        )

        st.markdown(
            f"### Step {step} — `{event_name}`"
        )

        st.caption(
            f"Timestamp: {timestamp}"
        )

        data = event.get(
            "data",
            {},
        )

        if data:

            with st.expander(
                "View event details"
            ):

                st.json(data)

        st.divider()


# ---------------------------------------------------------
# Raw trace
# ---------------------------------------------------------

with st.expander(
    "🔍 View Raw Trace JSON"
):

    st.json(
        trace
    )