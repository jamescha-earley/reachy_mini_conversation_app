"""MCP (Model Context Protocol) client for communicating with Snowflake agent."""

import json
import logging
import os
from typing import Any, Optional
import aiohttp
import asyncio
import ssl

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for communicating with an MCP server running a Snowflake agent."""

    def __init__(self, server_url: str, pat: str = "", timeout: int = 30, verify_ssl: bool = True):
        """Initialize MCP client.

        Args:
            server_url: The URL of the MCP server (e.g., "http://localhost:5000")
            pat: Personal Access Token for authentication
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates (default: True). Set to False for self-signed certs.
        """
        self.server_url = server_url.rstrip("/")
        self.pat = pat
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session is not None:
            await self.session.close()
            self.session = None

    async def query_snowflake_agent(
        self,
        query: str,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Send a query to the Snowflake agent through MCP server.

        Args:
            query: The question/query for the Snowflake agent
            context: Optional context or conversation history
            **kwargs: Additional parameters to pass to the agent

        Returns:
            The response from the Snowflake agent
        """
        try:
            session = await self._get_session()

            # Prepare the request payload
            payload = {
                "query": query,
                **({"context": context} if context else {}),
                **kwargs,
            }

            # Send request to MCP server's snowflake agent endpoint
            endpoint = f"{self.server_url}/tools/snowflake_agent"
            logger.debug(f"Querying MCP endpoint: {endpoint}")

            # Prepare headers with authentication
            headers = {"Content-Type": "application/json"}
            if self.pat:
                headers["Authorization"] = f"Bearer {self.pat}"

            # Prepare SSL context
            ssl_context = None
            if not self.verify_ssl:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                # Update session with the custom connector if needed
                if self.session is None:
                    timeout = aiohttp.ClientTimeout(total=self.timeout)
                    self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
                    session = self.session

            async with session.post(
                endpoint,
                json=payload,
                headers=headers,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"MCP response: {str(data)[:200]}")
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"MCP server error (status {response.status}): {error_text}")
                    return {"error": f"MCP server returned status {response.status}: {error_text}"}

        except asyncio.TimeoutError:
            error_msg = f"Timeout querying MCP server at {self.server_url}"
            logger.error(error_msg)
            return {"error": error_msg}
        except aiohttp.ClientError as e:
            error_msg = f"Connection error with MCP server: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error querying MCP server: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    async def list_available_tools(self) -> list[str]:
        """Get list of available tools from MCP server.

        Returns:
            List of tool names available on the MCP server
        """
        try:
            session = await self._get_session()
            endpoint = f"{self.server_url}/tools"

            async with session.get(endpoint) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("tools", [])
                else:
                    logger.warning(f"Failed to fetch tools list: status {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error fetching tools list: {e}")
            return []

    async def get_tool_schema(self, tool_name: str) -> Optional[dict]:
        """Get the schema for a specific tool from MCP server.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool schema or None if not found
        """
        try:
            session = await self._get_session()
            endpoint = f"{self.server_url}/tools/{tool_name}/schema"

            async with session.get(endpoint) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to fetch schema for {tool_name}: status {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Error fetching tool schema: {e}")
            return None
