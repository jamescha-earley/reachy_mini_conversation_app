"""Cortex Agent REST API client with SSE streaming support.

Talks directly to the Snowflake Cortex Agent API:
  POST /api/v2/databases/{db}/schemas/{schema}/agents/{name}:run

Supports both streaming (SSE) and non-streaming modes, thread management
for multi-turn conversations, and PAT authentication.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Parsed response from a Cortex Agent run."""

    texts: list[str] = field(default_factory=list)
    charts: list[dict] = field(default_factory=list)
    sql_statements: list[str] = field(default_factory=list)
    request_id: Optional[str] = None
    thread_id: Optional[str] = None

    @property
    def full_text(self) -> str:
        """Concatenate all text fragments."""
        return "".join(self.texts)

    def summary(self, max_length: int = 500) -> str:
        """Return a shortened version suitable for the robot to read aloud."""
        text = self.full_text.strip()
        if len(text) <= max_length:
            return text
        # Cut at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind(".")
        if last_period > max_length // 2:
            return truncated[: last_period + 1]
        return truncated + "..."


def _build_url(host: str, database: str, schema: str, agent_name: str) -> str:
    """Build the agent:run endpoint URL."""
    host = host.rstrip("/")
    if not host.startswith("https://"):
        host = f"https://{host}"
    return f"{host}/api/v2/databases/{database}/schemas/{schema}/agents/{agent_name}:run"


def _build_headers(pat: str, streaming: bool = True) -> dict[str, str]:
    """Build request headers with PAT auth."""
    headers = {
        "Authorization": f"Bearer {pat}",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
        "Content-Type": "application/json",
    }
    if streaming:
        headers["Accept"] = "text/event-stream"
    return headers


def _iter_sse(response: requests.Response) -> Generator[tuple[str, str], None, None]:
    """Parse SSE events from a streaming response.

    Yields (event_type, data_string) tuples.
    """
    event = None
    buf: list[str] = []

    for raw_line in response.iter_lines():
        if raw_line is None:
            continue
        line = raw_line.decode("utf-8", errors="ignore")

        if line.startswith("event:"):
            event = line.split("event:", 1)[1].strip()
        elif line.startswith("data:"):
            buf.append(line.split("data:", 1)[1].strip())
        elif line.strip() == "":
            if event is not None:
                data = "\n".join(buf).strip()
                yield event, data
            event, buf = None, []

    # Flush any remaining event
    if event is not None:
        data = "\n".join(buf).strip()
        yield event, data


class CortexAgentClient:
    """Client for the Snowflake Cortex Agent REST API."""

    def __init__(
        self,
        host: str,
        database: str,
        schema: str,
        agent_name: str,
        pat: str,
        timeout: int = 300,
    ):
        self.host = host
        self.database = database
        self.schema = schema
        self.agent_name = agent_name
        self.pat = pat
        self.timeout = timeout
        self.url = _build_url(host, database, schema, agent_name)
        self._thread_id: Optional[str] = None

    @property
    def thread_id(self) -> Optional[str]:
        return self._thread_id

    @thread_id.setter
    def thread_id(self, value: Optional[str]) -> None:
        self._thread_id = value

    def query(self, text: str, thread_id: Optional[str] = None) -> AgentResponse:
        """Send a query and return the full parsed response (streaming internally).

        Args:
            text: The user's question.
            thread_id: Optional thread ID for multi-turn. Uses instance thread_id if not provided.

        Returns:
            AgentResponse with texts, charts, sql_statements, etc.
        """
        tid = thread_id or self._thread_id

        body: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                }
            ],
        }
        if tid is not None:
            body["thread_id"] = tid

        logger.info(f"Cortex Agent query: {text[:120]}")

        try:
            resp = requests.post(
                self.url,
                headers=_build_headers(self.pat, streaming=True),
                json=body,
                stream=True,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Cortex Agent request failed: {e}")
            result = AgentResponse()
            result.texts.append(f"Error contacting Cortex Agent: {e}")
            return result

        result = AgentResponse()
        result.request_id = resp.headers.get("X-Snowflake-Request-Id")

        for event_type, data_str in _iter_sse(resp):
            if not data_str or data_str == "[DONE]":
                continue
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            try:
                if event_type == "response.text":
                    if isinstance(data.get("text"), str):
                        result.texts.append(data["text"])

                elif event_type == "message.delta":
                    for content in (data.get("delta") or {}).get("content", []):
                        if content.get("type") == "text" and isinstance(content.get("text"), str):
                            result.texts.append(content["text"])
                        if content.get("type") == "chart":
                            spec_str = (content.get("chart") or {}).get("chart_spec")
                            if spec_str:
                                chart = json.loads(spec_str) if isinstance(spec_str, str) else spec_str
                                result.charts.append(chart)

                elif event_type == "response.chart":
                    spec_str = data.get("chart_spec") or (data.get("chart") or {}).get("chart_spec")
                    if spec_str:
                        chart = json.loads(spec_str) if isinstance(spec_str, str) else spec_str
                        result.charts.append(chart)

                elif event_type == "response.tool_result":
                    # Extract SQL from analyst tool results
                    for content in data.get("content", []):
                        json_data = content.get("json", {})
                        if isinstance(json_data, dict):
                            sql = json_data.get("sql")
                            if sql:
                                result.sql_statements.append(sql)
                            text_val = json_data.get("text")
                            if text_val and isinstance(text_val, str):
                                result.texts.append(text_val)

                elif event_type == "response":
                    # Final response object — extract thread_id if present
                    if isinstance(data, dict):
                        tid_from_resp = data.get("thread_id")
                        if tid_from_resp:
                            result.thread_id = tid_from_resp
                            self._thread_id = tid_from_resp

            except Exception as e:
                logger.debug(f"Error parsing SSE event {event_type}: {e}")
                continue

        logger.info(f"Cortex Agent response: {len(result.texts)} text fragments, {len(result.charts)} charts")
        return result

    def query_streaming(self, text: str, thread_id: Optional[str] = None) -> Generator[dict, None, None]:
        """Send a query and yield parsed events as they arrive.

        Yields dicts with keys: type ("text", "chart", "sql", "status", "done"), and relevant data.
        """
        tid = thread_id or self._thread_id

        body: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                }
            ],
        }
        if tid is not None:
            body["thread_id"] = tid

        try:
            resp = requests.post(
                self.url,
                headers=_build_headers(self.pat, streaming=True),
                json=body,
                stream=True,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            yield {"type": "error", "text": f"Error contacting Cortex Agent: {e}"}
            return

        for event_type, data_str in _iter_sse(resp):
            if not data_str or data_str == "[DONE]":
                continue
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            try:
                if event_type == "response.text":
                    if isinstance(data.get("text"), str):
                        yield {"type": "text", "text": data["text"]}

                elif event_type == "message.delta":
                    for content in (data.get("delta") or {}).get("content", []):
                        if content.get("type") == "text" and isinstance(content.get("text"), str):
                            yield {"type": "text", "text": content["text"]}
                        if content.get("type") == "chart":
                            spec_str = (content.get("chart") or {}).get("chart_spec")
                            if spec_str:
                                chart = json.loads(spec_str) if isinstance(spec_str, str) else spec_str
                                yield {"type": "chart", "chart_spec": chart}

                elif event_type == "response.chart":
                    spec_str = data.get("chart_spec") or (data.get("chart") or {}).get("chart_spec")
                    if spec_str:
                        chart = json.loads(spec_str) if isinstance(spec_str, str) else spec_str
                        yield {"type": "chart", "chart_spec": chart}

                elif event_type == "response.tool_result":
                    for content in data.get("content", []):
                        json_data = content.get("json", {})
                        if isinstance(json_data, dict):
                            sql = json_data.get("sql")
                            if sql:
                                yield {"type": "sql", "sql": sql}
                            text_val = json_data.get("text")
                            if text_val and isinstance(text_val, str):
                                yield {"type": "text", "text": text_val}

                elif event_type in ("response.status", "response.tool_use"):
                    status_msg = data.get("message") or data.get("status", "")
                    if status_msg:
                        yield {"type": "status", "message": status_msg}

                elif event_type == "response":
                    if isinstance(data, dict):
                        tid_from_resp = data.get("thread_id")
                        if tid_from_resp:
                            self._thread_id = tid_from_resp
                    yield {"type": "done"}

            except Exception:
                continue
