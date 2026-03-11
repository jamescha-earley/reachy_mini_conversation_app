import logging
from typing import Any, Dict, Optional

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)


class SnowflakeAgent(Tool):
    """Query the Snowflake Cortex Agent for data analysis, SQL queries, or other operations.

    Supports two backends (tried in order):
      1. Cortex Agent REST API — direct call via ``CortexAgentClient``
      2. MCP Server — legacy path via ``MCPClient``

    When ``AGENT_BACKEND_URL`` is configured the tool also posts the query
    to the shared FastAPI backend so Streamlit can display it.
    """

    name = "snowflake_agent"
    description = "Query the Snowflake agent for data analysis, SQL queries, or other Snowflake operations."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question or request to send to the Snowflake agent (e.g., 'What are the top 5 customers by revenue?')",
            },
            "context": {
                "type": "string",
                "description": "Optional context or previous conversation history to provide to the agent",
            },
        },
        "required": ["query"],
    }

    def __init__(self):
        """Initialize the Snowflake agent tool."""
        self._mcp_client = None
        self._cortex_client = None
        self._mcp_server_url = None
        self._use_cortex: Optional[bool] = None  # None = not yet determined

    # ------------------------------------------------------------------
    # Cortex Agent REST API (preferred)
    # ------------------------------------------------------------------

    def _get_cortex_client(self) -> Any:
        """Get or initialize the Cortex Agent REST client."""
        if self._cortex_client is not None:
            return self._cortex_client

        host = config.SNOWFLAKE_AGENT_HOST if hasattr(config, "SNOWFLAKE_AGENT_HOST") else ""
        database = config.SNOWFLAKE_AGENT_DATABASE if hasattr(config, "SNOWFLAKE_AGENT_DATABASE") else ""
        schema = config.SNOWFLAKE_AGENT_SCHEMA if hasattr(config, "SNOWFLAKE_AGENT_SCHEMA") else ""
        agent_name = config.SNOWFLAKE_AGENT_NAME if hasattr(config, "SNOWFLAKE_AGENT_NAME") else ""
        pat = config.SNOWFLAKE_MCP_PAT

        if not all([host, database, schema, agent_name, pat]):
            return None

        try:
            from agent_app.cortex_agent_client import CortexAgentClient
            self._cortex_client = CortexAgentClient(
                host=host, database=database, schema=schema,
                agent_name=agent_name, pat=pat,
            )
            logger.info(f"Initialized Cortex Agent client: {agent_name} at {database}.{schema}")
            return self._cortex_client
        except ImportError:
            logger.debug("agent_app.cortex_agent_client not available, falling back to MCP")
            return None

    # ------------------------------------------------------------------
    # MCP Server (fallback)
    # ------------------------------------------------------------------

    async def _get_mcp_client(self) -> Any:
        """Get or initialize the MCP client connection."""
        if self._mcp_client is not None:
            return self._mcp_client

        try:
            from reachy_mini_conversation_app.mcp_client import MCPClient

            mcp_url = config.SNOWFLAKE_MCP_SERVER_URL
            mcp_pat = config.SNOWFLAKE_MCP_PAT
            mcp_verify_ssl = config.SNOWFLAKE_MCP_VERIFY_SSL
            self._mcp_server_url = mcp_url
            self._mcp_client = MCPClient(mcp_url, pat=mcp_pat, verify_ssl=mcp_verify_ssl)
            logger.info(f"Initialized MCP client connecting to {mcp_url} (SSL verify: {mcp_verify_ssl})")
            return self._mcp_client
        except ImportError as e:
            logger.error(f"MCP client module not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            raise

    # ------------------------------------------------------------------
    # Backend notification (optional)
    # ------------------------------------------------------------------

    def _notify_backend(self, text: str, query: str) -> None:
        """Post the robot's speech to the shared backend (best-effort)."""
        backend_url = getattr(config, "AGENT_BACKEND_URL", "") or ""
        if not backend_url:
            return
        try:
            import requests
            requests.post(
                f"{backend_url.rstrip('/')}/robot_says",
                json={"text": text, "query": query},
                timeout=5,
            )
        except Exception as e:
            logger.debug(f"Backend notification failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Send a query to the Snowflake agent."""
        query = (kwargs.get("query") or "").strip()
        context = kwargs.get("context", "").strip()

        if not query:
            logger.warning("snowflake_agent: empty query")
            return {"error": "query must be a non-empty string"}

        logger.info(f"Tool call: snowflake_agent query={query[:120]}")

        # --- Try Cortex Agent REST API first ---
        if self._use_cortex is not False:
            cortex = self._get_cortex_client()
            if cortex is not None:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(None, cortex.query, query)

                    self._use_cortex = True
                    summary = response.summary()
                    self._notify_backend(summary, query)

                    logger.info(f"Cortex Agent response: {summary[:200]}")
                    return {
                        "response": response.full_text,
                        "summary": summary,
                        "charts": response.charts,
                        "sql": response.sql_statements,
                    }
                except Exception as e:
                    logger.warning(f"Cortex Agent failed, falling back to MCP: {e}")
                    self._use_cortex = False

        # --- Fallback to MCP ---
        try:
            mcp_client = await self._get_mcp_client()
            result = await mcp_client.query_snowflake_agent(
                query=query,
                context=context if context else None,
            )

            if isinstance(result, dict) and "error" in result:
                logger.error(f"Snowflake agent error: {result['error']}")
                return result

            response_text = str(result)
            self._notify_backend(response_text[:300], query)

            logger.info(f"Snowflake agent response: {response_text[:200]}")
            return {"response": result}

        except Exception as e:
            error_msg = f"Failed to query Snowflake agent: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
