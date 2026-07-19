from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from agent.tools.base import WorkspaceTool


READ_ONLY_GIT_COMMANDS = {"status", "diff", "log", "show", "branch"}
WRITE_GIT_COMMANDS = {"add", "commit"}
ALLOWED_GIT_COMMANDS = READ_ONLY_GIT_COMMANDS | WRITE_GIT_COMMANDS


class GitTool(WorkspaceTool):
    def __init__(
        self,
        root: Path,
        allow_write: bool = True,
        auto_init: bool = False,
        scope_description: str = "workspace",
    ) -> None:
        super().__init__(root)
        self.allow_write = allow_write
        self.auto_init = auto_init
        self.scope_description = scope_description

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
                f"Git write rejected: this {self.scope_description} is read-only.",
                {
                    "command": command,
                    "git_read_only": True,
                    "allowed": sorted(READ_ONLY_GIT_COMMANDS),
                },
            )
        if any(flag in parts for flag in ["reset", "checkout", "clean", "rebase", "merge", "push", "pull"]):
            return ToolResult(False, "Git command rejected: destructive or network operation.", {"command": command})
        if not self.root.exists():
            if verb in WRITE_GIT_COMMANDS and self.auto_init:
                self.root.mkdir(parents=True, exist_ok=True)
            else:
                return ToolResult(
                    False,
                    f"Git {self.scope_description} does not exist yet.",
                    {"command": command, "git_root": str(self.root)},
                )
        isolated_repo_ready = self._is_current_git_root()
        if verb in WRITE_GIT_COMMANDS and self.auto_init:
            init_result = self._ensure_git_repository(timeout=int(args.get("timeout", 30)))
            if init_result is not None:
                return init_result
        elif self.auto_init and not isolated_repo_ready:
            return ToolResult(
                False,
                f"Git {self.scope_description} is not initialized yet.",
                {"command": command, "git_root": str(self.root)},
            )
        timeout = int(args.get("timeout", 30))
        git_command = ["git", *parts]
        if verb == "commit":
            git_command = [
                "git",
                "-c",
                "user.name=Long Agent",
                "-c",
                "user.email=long-agent@example.invalid",
                *parts,
            ]
        completed = subprocess.run(
            git_command,
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
            {
                "command": "git " + " ".join(parts),
                "output": output[:8000],
                "read_only": verb in READ_ONLY_GIT_COMMANDS,
                "git_root": str(self.root),
            },
        )

    def _ensure_git_repository(self, timeout: int):
        from agent.tools import ToolResult

        if self._is_current_git_root(timeout=timeout):
            return None
        init = subprocess.run(
            ["git", "init"],
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if init.returncode == 0:
            return None
        output = (init.stdout + init.stderr).strip()
        return ToolResult(
            False,
            f"Git init failed for {self.scope_description}.",
            {"command": "git init", "output": output[:8000], "git_root": str(self.root)},
        )

    def _is_current_git_root(self, timeout: int = 30) -> bool:
        if not self.root.exists():
            return False
        probe = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if probe.returncode != 0:
            return False
        try:
            return Path(probe.stdout.strip()).resolve() == self.root.resolve()
        except OSError:
            return False
