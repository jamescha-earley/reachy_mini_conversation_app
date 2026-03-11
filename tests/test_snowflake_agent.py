"""Tests for the SnowflakeAgent tool."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from reachy_mini_conversation_app.tools.snowflake_agent import SnowflakeAgent


@pytest.fixture
def agent():
    """Create a SnowflakeAgent instance."""
    return SnowflakeAgent()


@pytest.fixture
def mock_deps():
    """Create mock ToolDependencies."""
    return MagicMock()


# --- empty / missing query ---


@pytest.mark.asyncio
async def test_empty_query_returns_error(agent, mock_deps):
    """Empty query string returns an error."""
    result = await agent(mock_deps, query="")
    assert "error" in result
    assert "non-empty" in result["error"]


@pytest.mark.asyncio
async def test_whitespace_query_returns_error(agent, mock_deps):
    """Whitespace-only query returns an error."""
    result = await agent(mock_deps, query="   ")
    assert "error" in result


@pytest.mark.asyncio
async def test_missing_query_returns_error(agent, mock_deps):
    """No query kwarg returns an error."""
    result = await agent(mock_deps)
    assert "error" in result


# --- successful query ---


@pytest.mark.asyncio
async def test_successful_query(agent, mock_deps):
    """Successful query returns wrapped response."""
    mock_mcp = AsyncMock()
    mock_mcp.query_snowflake_agent = AsyncMock(return_value={"data": "results"})
    agent._mcp_client = mock_mcp

    result = await agent(mock_deps, query="top 5 customers")
    assert result == {"response": {"data": "results"}}
    mock_mcp.query_snowflake_agent.assert_awaited_once_with(
        query="top 5 customers",
        context=None,
    )


@pytest.mark.asyncio
async def test_query_with_context(agent, mock_deps):
    """Query with context passes it through to MCP client."""
    mock_mcp = AsyncMock()
    mock_mcp.query_snowflake_agent = AsyncMock(return_value={"data": "ok"})
    agent._mcp_client = mock_mcp

    result = await agent(mock_deps, query="follow up", context="previous chat")
    mock_mcp.query_snowflake_agent.assert_awaited_once_with(
        query="follow up",
        context="previous chat",
    )


@pytest.mark.asyncio
async def test_empty_context_sent_as_none(agent, mock_deps):
    """Empty string context is sent as None."""
    mock_mcp = AsyncMock()
    mock_mcp.query_snowflake_agent = AsyncMock(return_value={"data": "ok"})
    agent._mcp_client = mock_mcp

    await agent(mock_deps, query="test", context="")
    mock_mcp.query_snowflake_agent.assert_awaited_once_with(
        query="test",
        context=None,
    )


# --- MCP client error responses ---


@pytest.mark.asyncio
async def test_mcp_error_response_propagated(agent, mock_deps):
    """Error dict from MCP client is returned directly."""
    mock_mcp = AsyncMock()
    mock_mcp.query_snowflake_agent = AsyncMock(
        return_value={"error": "MCP server returned status 500: Internal Server Error"}
    )
    agent._mcp_client = mock_mcp

    result = await agent(mock_deps, query="bad query")
    assert "error" in result
    assert "500" in result["error"]


# --- MCP client exceptions ---


@pytest.mark.asyncio
async def test_mcp_client_exception(agent, mock_deps):
    """Exception from MCP client is caught and returned as error."""
    mock_mcp = AsyncMock()
    mock_mcp.query_snowflake_agent = AsyncMock(side_effect=Exception("connection failed"))
    agent._mcp_client = mock_mcp

    result = await agent(mock_deps, query="test")
    assert "error" in result
    assert "connection failed" in result["error"]


# --- lazy initialization ---


@pytest.mark.asyncio
async def test_lazy_init_creates_client(agent, mock_deps):
    """MCP client is lazily initialized on first call."""
    assert agent._mcp_client is None

    mock_instance = AsyncMock()
    mock_instance.query_snowflake_agent = AsyncMock(return_value={"ok": True})

    with patch("reachy_mini_conversation_app.tools.snowflake_agent.config") as mock_config:
        mock_config.SNOWFLAKE_MCP_SERVER_URL = "https://test.com/mcp"
        mock_config.SNOWFLAKE_MCP_PAT = "test-pat"
        mock_config.SNOWFLAKE_MCP_VERIFY_SSL = True
        # Ensure Cortex Agent path is skipped (empty config)
        mock_config.SNOWFLAKE_AGENT_HOST = ""
        mock_config.SNOWFLAKE_AGENT_DATABASE = ""
        mock_config.SNOWFLAKE_AGENT_SCHEMA = ""
        mock_config.SNOWFLAKE_AGENT_NAME = ""
        mock_config.AGENT_BACKEND_URL = ""

        with patch.dict(
            "sys.modules",
            {"reachy_mini_conversation_app.mcp_client": MagicMock(MCPClient=MagicMock(return_value=mock_instance))},
        ) as mocked_modules:
            MCPClientMock = mocked_modules["reachy_mini_conversation_app.mcp_client"].MCPClient

            result = await agent(mock_deps, query="test")

            MCPClientMock.assert_called_once_with(
                "https://test.com/mcp",
                pat="test-pat",
                verify_ssl=True,
            )


@pytest.mark.asyncio
async def test_lazy_init_import_error(agent, mock_deps):
    """ImportError during lazy init returns error."""
    agent._mcp_client = None

    with patch("reachy_mini_conversation_app.tools.snowflake_agent.config") as mock_config:
        mock_config.SNOWFLAKE_MCP_SERVER_URL = "https://test.com"
        mock_config.SNOWFLAKE_MCP_PAT = ""
        mock_config.SNOWFLAKE_MCP_VERIFY_SSL = True

        # Remove the module so the lazy import fails
        import sys
        saved = sys.modules.pop("reachy_mini_conversation_app.mcp_client", None)
        with patch.dict("sys.modules", {"reachy_mini_conversation_app.mcp_client": None}):
            result = await agent(mock_deps, query="test")
            assert "error" in result
        # Restore
        if saved is not None:
            sys.modules["reachy_mini_conversation_app.mcp_client"] = saved


# --- tool spec ---


def test_tool_spec(agent):
    """Tool spec has correct structure."""
    spec = agent.spec()
    assert spec["type"] == "function"
    assert spec["name"] == "snowflake_agent"
    assert "query" in spec["parameters"]["properties"]
    assert "query" in spec["parameters"]["required"]
