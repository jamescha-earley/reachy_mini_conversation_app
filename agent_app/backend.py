"""Shared FastAPI backend that bridges Streamlit, Reachy Mini, and Cortex Agent.

Endpoints:
  POST /query        — Send a query to Cortex Agent (from Streamlit or robot)
  GET  /events       — SSE stream of all activity (queries, responses, robot speech)
  POST /robot_says   — Reachy Mini posts what it said (so Streamlit can display)
  GET  /history      — Retrieve conversation history
  POST /reset        — Clear conversation history and thread
  GET  /health       — Health check

Run:
  python -m agent_app.backend
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_app.cortex_agent_client import AgentResponse, CortexAgentClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    source: str = "streamlit"  # "streamlit" or "robot"


class RobotSaysRequest(BaseModel):
    text: str
    query: Optional[str] = None  # the original query that triggered it


class ResetRequest(BaseModel):
    pass


@dataclass
class Message:
    role: str  # "user", "assistant", "robot", "system"
    content: str
    source: str = "streamlit"  # "streamlit", "robot"
    timestamp: float = field(default_factory=time.time)
    charts: list[dict] = field(default_factory=list)
    sql: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Event bus — simple async broadcast to all connected SSE clients
# ---------------------------------------------------------------------------

class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers = [s for s in self._subscribers if s is not q]

    async def publish(self, event: dict) -> None:
        event.setdefault("timestamp", time.time())
        for q in self._subscribers:
            await q.put(event)


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

history: list[Message] = []
event_bus = EventBus()
agent_client: Optional[CortexAgentClient] = None


def _get_agent_client() -> CortexAgentClient:
    global agent_client
    if agent_client is not None:
        return agent_client

    host = os.getenv("SNOWFLAKE_AGENT_HOST", "")
    database = os.getenv("SNOWFLAKE_AGENT_DATABASE", "")
    schema = os.getenv("SNOWFLAKE_AGENT_SCHEMA", "")
    agent_name = os.getenv("SNOWFLAKE_AGENT_NAME", "")
    pat = os.getenv("SNOWFLAKE_MCP_PAT", "")

    if not all([host, database, schema, agent_name, pat]):
        raise ValueError(
            "Missing Cortex Agent config. Set SNOWFLAKE_AGENT_HOST, "
            "SNOWFLAKE_AGENT_DATABASE, SNOWFLAKE_AGENT_SCHEMA, "
            "SNOWFLAKE_AGENT_NAME, and SNOWFLAKE_MCP_PAT in .env"
        )

    agent_client = CortexAgentClient(
        host=host,
        database=database,
        schema=schema,
        agent_name=agent_name,
        pat=pat,
    )
    logger.info(f"Cortex Agent client initialized: {agent_name} at {database}.{schema}")
    return agent_client


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load .env if present
    try:
        from dotenv import find_dotenv, load_dotenv
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path, override=True)
    except ImportError:
        pass

    logger.info("Agent backend starting")
    yield
    logger.info("Agent backend shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Cortex Agent Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "healthy", "messages": len(history)}


@app.post("/query")
async def query(req: QueryRequest):
    """Send a query to the Cortex Agent and return the response."""
    # Record user message
    user_msg = Message(role="user", content=req.query, source=req.source)
    history.append(user_msg)

    await event_bus.publish({
        "type": "user_query",
        "source": req.source,
        "query": req.query,
    })

    # Query Cortex Agent (runs in thread to avoid blocking)
    try:
        client = _get_agent_client()
        loop = asyncio.get_event_loop()
        response: AgentResponse = await loop.run_in_executor(None, client.query, req.query)
    except Exception as e:
        error_text = f"Error: {e}"
        error_msg = Message(role="system", content=error_text, source="backend")
        history.append(error_msg)
        await event_bus.publish({"type": "error", "text": error_text})
        return {"error": error_text}

    # Record assistant response
    assistant_msg = Message(
        role="assistant",
        content=response.full_text,
        source="cortex_agent",
        charts=response.charts,
        sql=response.sql_statements,
    )
    history.append(assistant_msg)

    # Broadcast response
    await event_bus.publish({
        "type": "agent_response",
        "text": response.full_text,
        "summary": response.summary(),
        "charts": response.charts,
        "sql": response.sql_statements,
        "source": req.source,
        "thread_id": response.thread_id,
    })

    return {
        "text": response.full_text,
        "summary": response.summary(),
        "charts": response.charts,
        "sql": response.sql_statements,
        "thread_id": response.thread_id,
    }


@app.post("/robot_says")
async def robot_says(req: RobotSaysRequest):
    """Record what the robot said aloud (for display in Streamlit)."""
    msg = Message(role="robot", content=req.text, source="robot")
    history.append(msg)

    await event_bus.publish({
        "type": "robot_speech",
        "text": req.text,
        "query": req.query,
    })

    return {"status": "ok"}


@app.get("/events")
async def events(request: Request):
    """SSE endpoint — streams all events to connected clients."""
    queue = event_bus.subscribe()

    async def event_generator():
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
        finally:
            event_bus.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/history")
async def get_history():
    """Return conversation history."""
    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "source": msg.source,
                "timestamp": msg.timestamp,
                "charts": msg.charts,
                "sql": msg.sql,
            }
            for msg in history
        ]
    }


@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest()):
    """Clear conversation history and reset thread."""
    global agent_client
    history.clear()
    if agent_client is not None:
        agent_client.thread_id = None

    await event_bus.publish({"type": "reset"})
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    port = int(os.getenv("AGENT_BACKEND_PORT", "8502"))
    uvicorn.run(app, host="0.0.0.0", port=port)
