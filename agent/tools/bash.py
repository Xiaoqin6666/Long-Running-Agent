from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any

from agent.tools.base import WorkspaceTool


class BashTool(WorkspaceTool):
    def __init__(self, root: Path, python_path: Path | None = None) -> None:
        super().__init__(root)
        self.python_path = python_path.resolve() if python_path else None

    def run(self, action: dict[str, Any]):
        from agent.tools import ToolResult

        args = action.get("args", {})
        command = str(args.get("command") or action.get("target", ""))
        timeout = int(args.get("timeout", 30))
        if not command.strip():
            return ToolResult(False, "Empty command rejected.", {})
        env = os.environ.copy()
        if self.python_path:
            current = env.get("PYTHONPATH", "")
            entries = [str(self.python_path)]
            if current:
                entries.append(current)
            env["PYTHONPATH"] = os.pathsep.join(entries)
        completed = subprocess.run(
            command,
            cwd=self.root,
            env=env,
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
            {
                "command": command,
                "output": output[:8000],
                "cwd": str(self.root),
                "python_path": str(self.python_path) if self.python_path else "",
            },
        )
