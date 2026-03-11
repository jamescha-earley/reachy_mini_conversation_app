"""CLI tool for executing shell commands.

Allows the robot to execute shell commands and return the output.
Use with caution - this gives the AI access to the system.
"""

import logging
import asyncio
import subprocess
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Commands that are blocked for safety
BLOCKED_COMMANDS = [
    "rm -rf /",
    "rm -rf ~",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",  # fork bomb
    "chmod -R 777 /",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
]

# Optional: Allowlist mode - only allow specific command prefixes
# Set to None to allow all commands (except blocked ones)
ALLOWED_COMMAND_PREFIXES = None  # e.g., ["ls", "cat", "echo", "python", "pip"]


class ExecuteCommand(Tool):
    """Execute a shell command and return the output."""

    name = "execute_command"
    description = (
        "Execute a shell command on the system and return the output. "
        "Use this to run scripts, check system info, manage files, or interact with other tools. "
        "You have access to 'cortex code' - the Snowflake Cortex Code CLI for AI-assisted coding and Snowflake operations. "
        "Example commands: 'cortex code --help', 'cortex code \"describe this project\"'. "
        "Be careful with destructive commands."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30, max: 300)",
            },
        },
        "required": ["command"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute a shell command."""
        command = (kwargs.get("command") or "").strip()
        timeout = min(kwargs.get("timeout", 30), 300)  # Max 5 minutes

        if not command:
            return {"error": "Command cannot be empty"}

        logger.info(f"Tool call: execute_command command='{command}' timeout={timeout}")

        # Safety checks
        command_lower = command.lower()

        # Check blocked commands
        for blocked in BLOCKED_COMMANDS:
            if blocked in command_lower:
                logger.warning(f"Blocked dangerous command: {command}")
                return {"error": f"Command blocked for safety: contains '{blocked}'"}

        # Check allowlist if enabled
        if ALLOWED_COMMAND_PREFIXES is not None:
            allowed = False
            for prefix in ALLOWED_COMMAND_PREFIXES:
                if command_lower.startswith(prefix.lower()):
                    allowed = True
                    break
            if not allowed:
                logger.warning(f"Command not in allowlist: {command}")
                return {"error": f"Command not allowed. Allowed prefixes: {ALLOWED_COMMAND_PREFIXES}"}

        try:
            # Run command asynchronously
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=None,  # Use current directory
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                logger.warning(f"Command timed out after {timeout}s: {command}")
                return {
                    "error": f"Command timed out after {timeout} seconds",
                    "command": command,
                }

            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()

            # Truncate very long outputs
            max_output = 4000
            if len(stdout_text) > max_output:
                stdout_text = stdout_text[:max_output] + "\n... (output truncated)"
            if len(stderr_text) > max_output:
                stderr_text = stderr_text[:max_output] + "\n... (output truncated)"

            result = {
                "command": command,
                "exit_code": process.returncode,
                "success": process.returncode == 0,
            }

            if stdout_text:
                result["stdout"] = stdout_text
            if stderr_text:
                result["stderr"] = stderr_text

            if process.returncode == 0:
                logger.info(f"Command succeeded: {command}")
            else:
                logger.warning(f"Command failed with exit code {process.returncode}: {command}")

            return result

        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            return {
                "error": str(e),
                "command": command,
            }
