from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from agent.tools.base import WorkspaceTool


READ_ONLY_GIT_COMMANDS = {"status", "diff", "log", "show", "branch"}
WRITE_GIT_COMMANDS = {"add", "commit"}
ALLOWED_GIT_COMMANDS = READ_ONLY_GIT_COMMANDS | WRITE_GIT_COMMANDS


class GitTool(WorkspaceTool):
    def __init__(self, root: Path, allow_write: bool = True) -> None:
        super().__init__(root)
        self.allow_write = allow_write

    def run(self, action: dict[str, Any]):
        from agent.tools import ToolResult

        args = action.get("args", {})
        command = str(args.get("command") or action.get("target", "")).strip()
        if not command:
            command = "status --short"
        parts = command.split()
        verb = parts[0]
        if verb == "git" and len(parts) > 1:
            parts = parts[1:]
            verb = parts[0]
        if verb not in ALLOWED_GIT_COMMANDS:
            return ToolResult(
                False,
                f"Git command rejected: '{verb}' is not allowed.",
                {"command": command, "allowed": sorted(ALLOWED_GIT_COMMANDS)},
            )
        if verb in WRITE_GIT_COMMANDS and not self.allow_write:
            return ToolResult(
                False,
                "Git write rejected: benchmark runs cannot add or commit files in the host Agent repository.",
                {
                    "command": command,
                    "benchmark_git_read_only": True,
                    "allowed": sorted(READ_ONLY_GIT_COMMANDS),
                },
            )
        if any(flag in parts for flag in ["reset", "checkout", "clean", "rebase", "merge", "push", "pull"]):
            return ToolResult(False, "Git command rejected: destructive or network operation.", {"command": command})
        timeout = int(args.get("timeout", 30))
        completed = subprocess.run(
            ["git", *parts],
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        output = (completed.stdout + completed.stderr).strip()
        return ToolResult(
            completed.returncode == 0,
            f"Git exited with code {completed.returncode}.",
            {"command": "git " + " ".join(parts), "output": output[:8000], "read_only": verb in READ_ONLY_GIT_COMMANDS},
        )
