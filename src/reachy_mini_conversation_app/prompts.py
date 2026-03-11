import re
import sys
import logging
from pathlib import Path

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)


PROFILES_DIRECTORY = Path(__file__).parent / "profiles"
PROMPTS_LIBRARY_DIRECTORY = Path(__file__).parent / "prompts"
INSTRUCTIONS_FILENAME = "instructions.txt"
VOICE_FILENAME = "voice.txt"
VOICE_SAMPLE_FILENAME = "voice_sample.wav"


def _expand_prompt_includes(content: str) -> str:
    """Expand [<name>] placeholders with content from prompts library files.

    Args:
        content: The template content with [<name>] placeholders

    Returns:
        Expanded content with placeholders replaced by file contents

    """
    # Pattern to match [<name>] where name is a valid file stem (alphanumeric, underscores, hyphens)
    # pattern = re.compile(r'^\[([a-zA-Z0-9_-]+)\]$')
    # Allow slashes for subdirectories
    pattern = re.compile(r'^\[([a-zA-Z0-9/_-]+)\]$')

    lines = content.split('\n')
    expanded_lines = []

    for line in lines:
        stripped = line.strip()
        match = pattern.match(stripped)

        if match:
            # Extract the name from [<name>]
            template_name = match.group(1)
            template_file = PROMPTS_LIBRARY_DIRECTORY / f"{template_name}.txt"

            try:
                if template_file.exists():
                    template_content = template_file.read_text(encoding="utf-8").rstrip()
                    expanded_lines.append(template_content)
                    logger.debug("Expanded template: [%s]", template_name)
                else:
                    logger.warning("Template file not found: %s, keeping placeholder", template_file)
                    expanded_lines.append(line)
            except Exception as e:
                logger.warning("Failed to read template '%s': %s, keeping placeholder", template_name, e)
                expanded_lines.append(line)
        else:
            expanded_lines.append(line)

    return '\n'.join(expanded_lines)


def get_session_instructions() -> str:
    """Get session instructions, loading from REACHY_MINI_CUSTOM_PROFILE if set."""
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        logger.info(f"Loading default prompt from {PROMPTS_LIBRARY_DIRECTORY / 'default_prompt.txt'}")
        instructions_file = PROMPTS_LIBRARY_DIRECTORY / "default_prompt.txt"
    else:
        logger.info(f"Loading prompt from profile '{profile}'")
        instructions_file = PROFILES_DIRECTORY / profile / INSTRUCTIONS_FILENAME

    try:
        if instructions_file.exists():
            instructions = instructions_file.read_text(encoding="utf-8").strip()
            if instructions:
                # Expand [<name>] placeholders with content from prompts library
                expanded_instructions = _expand_prompt_includes(instructions)
                return expanded_instructions
            logger.error(f"Profile '{profile}' has empty {INSTRUCTIONS_FILENAME}")
            sys.exit(1)
        logger.error(f"Profile {profile} has no {INSTRUCTIONS_FILENAME}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load instructions from profile '{profile}': {e}")
        sys.exit(1)


def get_session_voice(default: str = "cedar") -> str:
    """Resolve the voice to use for the session.

    If a custom profile is selected and contains a voice.txt, return its
    trimmed content; otherwise return the provided default ("cedar").
    """
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        return default
    try:
        voice_file = PROFILES_DIRECTORY / profile / VOICE_FILENAME
        if voice_file.exists():
            voice = voice_file.read_text(encoding="utf-8").strip()
            return voice or default
    except Exception:
        pass
    return default


def get_voice_sample_path() -> str | None:
    """Resolve the path to a voice sample for TTS voice cloning.

    Checks in order:
    1. VOICE_SAMPLE_PATH from config/environment
    2. voice_sample.wav in the current profile directory

    Returns:
        Path to voice sample WAV file, or None if not found.

    """
    # First check config/environment override
    if config.VOICE_SAMPLE_PATH:
        sample_path = Path(config.VOICE_SAMPLE_PATH)
        if sample_path.exists():
            logger.debug(f"Using voice sample from config: {sample_path}")
            return str(sample_path)
        logger.warning(f"VOICE_SAMPLE_PATH configured but file not found: {config.VOICE_SAMPLE_PATH}")

    # Check profile directory
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if profile:
        try:
            sample_file = PROFILES_DIRECTORY / profile / VOICE_SAMPLE_FILENAME
            if sample_file.exists():
                logger.debug(f"Using voice sample from profile: {sample_file}")
                return str(sample_file)
        except Exception as e:
            logger.warning(f"Error checking profile voice sample: {e}")

    return None
