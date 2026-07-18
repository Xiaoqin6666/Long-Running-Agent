from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any

from agent.output_capture import capture_command_output
from agent.tools.base import WorkspaceTool


class BashTool(WorkspaceTool):
    def __init__(self, root: Path, python_path: Path | None = None, output_dir: Path | None = None) -> None:
        super().__init__(root)
        self.python_path = python_path.resolve() if python_path else None
        self.output_dir = output_dir or root / "state" / "tool_outputs"

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
        output_data = capture_command_output(
            root=self.root,
            output_dir=self.output_dir,
            label="bash",
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        return ToolResult(
            completed.returncode == 0,
            f"Command exited with code {completed.returncode}.",
            {
                "command": command,
                "cwd": str(self.root),
                "python_path": str(self.python_path) if self.python_path else "",
                **output_data,
            },
        )
