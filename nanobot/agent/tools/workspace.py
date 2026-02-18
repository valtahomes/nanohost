"""Workspace tools: user-installed tools loaded from workspace/agents/*/tools/."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry


class WorkspaceTool(Tool):
    """A tool loaded from workspace/agents/<slug>/tools/<tool>/tool.json + run.py.

    Execution: run.py is invoked as a subprocess with JSON params in sys.argv[1].
    stdout is the result, stderr + non-zero exit is an error.
    """

    def __init__(self, tool_dir: Path, definition: dict[str, Any]):
        self._tool_dir = tool_dir
        self._name = definition["name"]
        self._description = definition.get("description", self._name)
        self._parameters = definition.get("parameters", {"type": "object", "properties": {}})
        self._run_script = tool_dir / "run.py"
        self._timeout = definition.get("timeout", 30)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        if not self._run_script.exists():
            return f"Error: run.py not found for tool '{self._name}'"

        params_json = json.dumps(kwargs, ensure_ascii=False)

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(self._run_script), params_json,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._tool_dir),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self._timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return f"Error: Tool '{self._name}' timed out after {self._timeout}s"

            result = stdout.decode("utf-8", errors="replace").strip()

            if process.returncode != 0:
                err = stderr.decode("utf-8", errors="replace").strip()
                return f"Error (exit {process.returncode}): {err}" if err else f"Error (exit {process.returncode})"

            if not result:
                return "(no output)"

            # Truncate very long output
            if len(result) > 10000:
                result = result[:10000] + f"\n... (truncated, {len(result) - 10000} more chars)"

            return result

        except Exception as e:
            return f"Error executing tool '{self._name}': {e}"


def _scan_tool_dirs(base: Path) -> list[Path]:
    """Return tool directories (containing tool.json) under base."""
    if not base.exists():
        return []
    return sorted(
        d for d in base.iterdir()
        if d.is_dir() and not d.name.startswith(".") and (d / "tool.json").exists()
    )


def load_workspace_tools(workspace: Path, registry: ToolRegistry) -> int:
    """Scan workspace/agents/*/tools/ for tools.

    Returns number of workspace tools registered.
    """
    tool_dirs: list[Path] = []

    # Agent tools: workspace/agents/*/tools/*/
    agents_dir = workspace / "agents"
    if agents_dir.exists():
        for agent_dir in sorted(agents_dir.iterdir()):
            if agent_dir.is_dir() and not agent_dir.name.startswith("."):
                tool_dirs.extend(_scan_tool_dirs(agent_dir / "tools"))

    count = 0
    for tool_dir in tool_dirs:
        tool_json_path = tool_dir / "tool.json"

        try:
            definition = json.loads(tool_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Workspace tools: failed to load {tool_dir.name}/tool.json: {e}")
            continue

        if "name" not in definition:
            logger.warning(f"Workspace tools: {tool_dir.name}/tool.json missing 'name' field")
            continue

        if not (tool_dir / "run.py").exists():
            logger.warning(f"Workspace tools: {tool_dir.name} has tool.json but no run.py")
            continue

        tool = WorkspaceTool(tool_dir, definition)

        if registry.has(tool.name):
            logger.warning(f"Workspace tools: '{tool.name}' conflicts with existing tool, skipping")
            continue

        registry.register(tool)
        count += 1
        logger.info(f"Workspace tools: registered '{tool.name}' from {tool_dir.name}/")

    return count
