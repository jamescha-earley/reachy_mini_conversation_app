"""Streamlit chat UI for the Cortex Agent + Reachy Mini backend.

Displays a unified conversation showing:
  - User queries (typed in Streamlit)
  - Cortex Agent responses (with charts/SQL)
  - Robot speech (what Reachy Mini said aloud)

Listens for real-time events via SSE so robot interactions
appear automatically without refreshing.

Run:
  streamlit run agent_app/streamlit_app.py
"""

import json
import os
import threading
import time

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKEND_URL = os.getenv("AGENT_BACKEND_URL", "http://localhost:8502")


def _api(path: str) -> str:
    return f"{BACKEND_URL.rstrip('/')}{path}"


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Cortex Agent Chat",
    page_icon="🤖",
    layout="wide",
)

st.title("Cortex Agent Chat")
st.caption("Talk to a Cortex Agent — with Reachy Mini as your voice companion")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "last_event_ts" not in st.session_state:
    st.session_state.last_event_ts = 0.0


def _load_history():
    """Load conversation history from backend."""
    try:
        resp = requests.get(_api("/history"), timeout=5)
        resp.raise_for_status()
        data = resp.json()
        st.session_state.messages = data.get("messages", [])
    except Exception:
        pass  # Backend might not be running yet


# Load history on first run
if not st.session_state.messages:
    _load_history()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    # Health check
    try:
        health = requests.get(_api("/health"), timeout=2).json()
        st.success(f"Backend connected ({health.get('messages', 0)} messages)")
    except Exception:
        st.error(f"Backend not reachable at {BACKEND_URL}")
        st.info("Start it with: `python -m agent_app.backend`")

    st.divider()

    if st.button("Reset Conversation"):
        try:
            requests.post(_api("/reset"), timeout=5)
            st.session_state.messages = []
            st.session_state.thread_id = None
            st.rerun()
        except Exception as e:
            st.error(f"Reset failed: {e}")

    st.divider()
    st.markdown("**Sources**")
    st.markdown(
        "- **You** — typed queries from this UI\n"
        "- **Robot** — spoken queries from Reachy Mini\n"
        "- **Agent** — Cortex Agent responses"
    )


# ---------------------------------------------------------------------------
# Render chat messages
# ---------------------------------------------------------------------------

def _render_message(msg: dict):
    """Render a single message in the chat."""
    role = msg.get("role", "user")
    content = msg.get("content", "")
    source = msg.get("source", "")
    charts = msg.get("charts", [])
    sql = msg.get("sql", [])

    if role == "user":
        icon = "🗣️" if source == "robot" else "👤"
        label = "Robot asked" if source == "robot" else "You"
        with st.chat_message("user", avatar=icon):
            st.markdown(f"**{label}:** {content}")

    elif role == "assistant":
        with st.chat_message("assistant", avatar="❄️"):
            st.markdown(content)

            # Show charts if present
            for chart in charts:
                chart_type = chart.get("type", "")
                chart_data = chart.get("data", {})
                if chart_type and chart_data:
                    with st.expander(f"Chart: {chart_type}"):
                        st.json(chart_data)

            # Show SQL if present
            for sql_stmt in sql:
                with st.expander("SQL"):
                    st.code(sql_stmt, language="sql")

    elif role == "robot":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"*Reachy Mini said:* {content}")

    elif role == "system":
        with st.chat_message("assistant", avatar="⚠️"):
            st.warning(content)


for msg in st.session_state.messages:
    _render_message(msg)


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask the Cortex Agent something..."):
    # Show user message immediately
    user_msg = {
        "role": "user",
        "content": prompt,
        "source": "streamlit",
        "timestamp": time.time(),
        "charts": [],
        "sql": [],
    }
    st.session_state.messages.append(user_msg)

    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**You:** {prompt}")

    # Send to backend
    with st.chat_message("assistant", avatar="❄️"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    _api("/query"),
                    json={"query": prompt, "source": "streamlit"},
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()

                if "error" in data:
                    st.error(data["error"])
                    st.session_state.messages.append({
                        "role": "system",
                        "content": data["error"],
                        "source": "backend",
                        "timestamp": time.time(),
                        "charts": [],
                        "sql": [],
                    })
                else:
                    text = data.get("text", "")
                    charts = data.get("charts", [])
                    sql_stmts = data.get("sql", [])
                    thread_id = data.get("thread_id")

                    st.markdown(text)

                    for chart in charts:
                        chart_type = chart.get("type", "")
                        chart_data = chart.get("data", {})
                        if chart_type and chart_data:
                            with st.expander(f"Chart: {chart_type}"):
                                st.json(chart_data)

                    for sql_stmt in sql_stmts:
                        with st.expander("SQL"):
                            st.code(sql_stmt, language="sql")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": text,
                        "source": "cortex_agent",
                        "timestamp": time.time(),
                        "charts": charts,
                        "sql": sql_stmts,
                    })

                    if thread_id:
                        st.session_state.thread_id = thread_id

            except requests.exceptions.ConnectionError:
                st.error(
                    f"Cannot connect to backend at {BACKEND_URL}. "
                    "Start it with: `python -m agent_app.backend`"
                )
            except Exception as e:
                st.error(f"Error: {e}")


# ---------------------------------------------------------------------------
# Auto-refresh to pick up robot messages
# ---------------------------------------------------------------------------
# Streamlit doesn't natively support SSE push, so we poll the history
# endpoint periodically to catch messages from the robot.

st.markdown("---")
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 Refresh"):
        _load_history()
        st.rerun()

with col1:
    auto_refresh = st.checkbox("Auto-refresh for robot messages", value=False)

if auto_refresh:
    time.sleep(3)
    _load_history()
    st.rerun()
