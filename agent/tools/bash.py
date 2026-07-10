from __future__ import annotations

import subprocess
from typing import Any

from agent.tools.base import WorkspaceTool


class BashTool(WorkspaceTool):
    def run(self, action: dict[str, Any]):
        from agent.tools import ToolResult

        args = action.get("args", {})
        command = str(args.get("command") or action.get("target", ""))
        timeout = int(args.get("timeout", 30))
        if not command.strip():
            return ToolResult(False, "Empty command rejected.", {})
        completed = subprocess.run(
            command,
            cwd=self.root,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        output = (completed.stdout + completed.stderr).strip()
        return ToolResult(
            completed.returncode == 0,
            f"Command exited with code {completed.returncode}.",
            {"command": command, "output": output[:8000]},
        )
