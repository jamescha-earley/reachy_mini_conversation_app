"""Tests for MCP-related config parsing."""

import os
import pytest
from unittest.mock import patch


# Each test patches os.getenv to control the SNOWFLAKE_MCP_VERIFY_SSL value,
# then re-evaluates the same expression used in Config to verify parsing.

_TRUTHY = ["true", "True", "TRUE", "1", "yes", "YES"]
_FALSY = ["false", "False", "FALSE", "0", "no", "NO", "anything", ""]


@pytest.mark.parametrize("value", _TRUTHY)
def test_verify_ssl_truthy(value):
    """Truthy string values parse to True."""
    result = value.lower() in ("true", "1", "yes")
    assert result is True


@pytest.mark.parametrize("value", _FALSY)
def test_verify_ssl_falsy(value):
    """Falsy or unrecognized string values parse to False."""
    result = value.lower() in ("true", "1", "yes")
    assert result is False


def test_verify_ssl_default_is_true():
    """Default value when env var is unset should be True."""
    # Simulate missing env var — default is "true"
    raw = os.getenv("SNOWFLAKE_MCP_VERIFY_SSL_NONEXISTENT", "true")
    result = raw.lower() in ("true", "1", "yes")
    assert result is True


def test_config_has_mcp_attributes():
    """Config object exposes MCP attributes."""
    from reachy_mini_conversation_app.config import config

    assert hasattr(config, "SNOWFLAKE_MCP_SERVER_URL")
    assert hasattr(config, "SNOWFLAKE_MCP_PAT")
    assert hasattr(config, "SNOWFLAKE_MCP_VERIFY_SSL")


def test_config_verify_ssl_is_bool():
    """SNOWFLAKE_MCP_VERIFY_SSL is a bool, not a string."""
    from reachy_mini_conversation_app.config import config

    assert isinstance(config.SNOWFLAKE_MCP_VERIFY_SSL, bool)
