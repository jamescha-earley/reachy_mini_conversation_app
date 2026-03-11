import os
import logging

from dotenv import find_dotenv, load_dotenv


logger = logging.getLogger(__name__)

# Locate .env file (search upward from current working directory)
dotenv_path = find_dotenv(usecwd=True)

if dotenv_path:
    # Load .env and override environment variables
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f"Configuration loaded from {dotenv_path}")
else:
    logger.warning("No .env file found, using environment variables")


class Config:
    """Configuration class for the conversation app."""

    # Required
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # The key is downloaded in console.py if needed

    # Optional
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-realtime")
    HF_HOME = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set

    # TTS Engine configuration
    # Options: "openai" (default, uses OpenAI Realtime voices) or "qwen3" (uses Qwen3-TTS with voice cloning)
    TTS_ENGINE = os.getenv("TTS_ENGINE", "openai")
    # Path to voice sample for Qwen3-TTS voice cloning (optional, can also use profile voice_sample.wav)
    VOICE_SAMPLE_PATH = os.getenv("VOICE_SAMPLE_PATH")
    # URL of TTS service (for SPCS deployment)
    TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://localhost:8000")

    logger.debug(f"Model: {MODEL_NAME}, HF_HOME: {HF_HOME}, Vision Model: {LOCAL_VISION_MODEL}")
    logger.debug(f"TTS Engine: {TTS_ENGINE}, Voice Sample: {VOICE_SAMPLE_PATH}")

    # MCP Server configuration (for Snowflake Agent tool)
    SNOWFLAKE_MCP_SERVER_URL = os.getenv("SNOWFLAKE_MCP_SERVER_URL", "http://localhost:5000")
    SNOWFLAKE_MCP_PAT = os.getenv("SNOWFLAKE_MCP_PAT", "")
    SNOWFLAKE_MCP_VERIFY_SSL = os.getenv("SNOWFLAKE_MCP_VERIFY_SSL", "true").lower() in ("true", "1", "yes")

    # Cortex Agent REST API configuration
    SNOWFLAKE_AGENT_HOST = os.getenv("SNOWFLAKE_AGENT_HOST", "")
    SNOWFLAKE_AGENT_DATABASE = os.getenv("SNOWFLAKE_AGENT_DATABASE", "")
    SNOWFLAKE_AGENT_SCHEMA = os.getenv("SNOWFLAKE_AGENT_SCHEMA", "")
    SNOWFLAKE_AGENT_NAME = os.getenv("SNOWFLAKE_AGENT_NAME", "")

    # Shared backend URL (for Streamlit + robot coordination)
    AGENT_BACKEND_URL = os.getenv("AGENT_BACKEND_URL", "")

    REACHY_MINI_CUSTOM_PROFILE = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
    logger.debug(f"Custom Profile: {REACHY_MINI_CUSTOM_PROFILE}")


config = Config()


def set_custom_profile(profile: str | None) -> None:
    """Update the selected custom profile at runtime and expose it via env.

    This ensures modules that read `config` and code that inspects the
    environment see a consistent value.
    """
    try:
        config.REACHY_MINI_CUSTOM_PROFILE = profile
    except Exception:
        pass
    try:
        import os as _os

        if profile:
            _os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
        else:
            # Remove to reflect default
            _os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    except Exception:
        pass
