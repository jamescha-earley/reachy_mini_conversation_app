"""Tests for MCP client."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp

from reachy_mini_conversation_app.mcp_client import MCPClient


@pytest.fixture
def client():
    """Create an MCPClient instance for testing."""
    c = MCPClient(server_url="https://example.com/mcp", pat="test-token", timeout=5)
    yield c
    # Ensure session is closed after each test
    loop = asyncio.get_event_loop()
    if c.session is not None:
        loop.run_until_complete(c.close())


# --- query_snowflake_agent ---


@pytest.mark.asyncio
async def test_query_success(client):
    """Successful query returns parsed JSON response."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"answer": "42"})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.post = MagicMock(return_value=mock_response)

    client.session = mock_session

    result = await client.query_snowflake_agent("What is the answer?")
    assert result == {"answer": "42"}

    # Verify correct endpoint and auth header
    mock_session.post.assert_called_once()
    call_kwargs = mock_session.post.call_args
    assert call_kwargs[1]["json"]["query"] == "What is the answer?"
    assert call_kwargs[1]["headers"]["Authorization"] == "Bearer test-token"


@pytest.mark.asyncio
async def test_query_with_context(client):
    """Query with optional context includes it in the payload."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"answer": "yes"})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.post = MagicMock(return_value=mock_response)
    client.session = mock_session

    result = await client.query_snowflake_agent("follow up?", context="previous conversation")
    payload = mock_session.post.call_args[1]["json"]
    assert payload["query"] == "follow up?"
    assert payload["context"] == "previous conversation"


@pytest.mark.asyncio
async def test_query_no_pat():
    """Query without PAT omits Authorization header."""
    c = MCPClient(server_url="https://example.com/mcp", pat="")

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"data": "ok"})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.post = MagicMock(return_value=mock_response)
    c.session = mock_session

    await c.query_snowflake_agent("test")
    headers = mock_session.post.call_args[1]["headers"]
    assert "Authorization" not in headers


@pytest.mark.asyncio
async def test_query_http_error(client):
    """Non-200 response returns error dict."""
    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.text = AsyncMock(return_value="Internal Server Error")
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.post = MagicMock(return_value=mock_response)
    client.session = mock_session

    result = await client.query_snowflake_agent("test")
    assert "error" in result
    assert "500" in result["error"]


@pytest.mark.asyncio
async def test_query_timeout(client):
    """Timeout returns error dict."""
    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.post = MagicMock(side_effect=asyncio.TimeoutError())
    client.session = mock_session

    result = await client.query_snowflake_agent("test")
    assert "error" in result
    assert "Timeout" in result["error"]


@pytest.mark.asyncio
async def test_query_connection_error(client):
    """Connection error returns error dict."""
    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Connection refused"))
    client.session = mock_session

    result = await client.query_snowflake_agent("test")
    assert "error" in result
    assert "Connection error" in result["error"]


# --- list_available_tools ---


@pytest.mark.asyncio
async def test_list_tools_success(client):
    """Successful tools listing returns tool names."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"tools": ["tool_a", "tool_b"]})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.get = MagicMock(return_value=mock_response)
    client.session = mock_session

    result = await client.list_available_tools()
    assert result == ["tool_a", "tool_b"]


@pytest.mark.asyncio
async def test_list_tools_error(client):
    """Failed tools listing returns empty list."""
    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.get = MagicMock(side_effect=Exception("network error"))
    client.session = mock_session

    result = await client.list_available_tools()
    assert result == []


# --- get_tool_schema ---


@pytest.mark.asyncio
async def test_get_schema_success(client):
    """Successful schema fetch returns schema dict."""
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=schema)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.get = MagicMock(return_value=mock_response)
    client.session = mock_session

    result = await client.get_tool_schema("snowflake_agent")
    assert result == schema


@pytest.mark.asyncio
async def test_get_schema_not_found(client):
    """404 for schema returns None."""
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.get = MagicMock(return_value=mock_response)
    client.session = mock_session

    result = await client.get_tool_schema("nonexistent")
    assert result is None


# --- close ---


@pytest.mark.asyncio
async def test_close_clears_session(client):
    """Closing the client clears the session."""
    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    client.session = mock_session

    await client.close()
    assert client.session is None
    mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_noop_when_no_session(client):
    """Closing when no session is a no-op."""
    assert client.session is None
    await client.close()  # Should not raise
    assert client.session is None


# --- URL normalization ---


def test_trailing_slash_stripped():
    """Server URL trailing slash is stripped."""
    c = MCPClient(server_url="https://example.com/mcp/")
    assert c.server_url == "https://example.com/mcp"
