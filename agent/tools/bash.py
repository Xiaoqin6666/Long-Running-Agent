from __future__ import annotations

import logging
import os
from pathlib import Path
import signal
import subprocess
from typing import Any

from agent.output_capture import capture_command_output
from agent.tools.base import WorkspaceTool


LOGGER = logging.getLogger("long_agent")


class BashTool(WorkspaceTool):
    def __init__(self, root: Path, python_path: Path | None = None, output_dir: Path | None = None) -> None:
        super().__init__(root)
        self.python_path = python_path.resolve() if python_path else None
        self.output_dir = output_dir or root / "state" / "tool_outputs"
        self._background_processes: list[subprocess.Popen[Any]] = []

    def run(self, action: dict[str, Any]):
        from agent.tools import ToolResult

        self._reap_background_processes()
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
        creationflags = 0
        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["start_new_session"] = True

        LOGGER.info("Bash command starting timeout=%ss target=%s", timeout, command)
        process = subprocess.Popen(
            command,
            cwd=self.root,
            env=env,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
            **popen_kwargs,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            returncode = process.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _decode_timeout_output(exc.stdout)
            stderr = _decode_timeout_output(exc.stderr)
            LOGGER.warning(
                "Bash command timed out after %ss; terminating process tree pid=%s target=%s",
                timeout,
                process.pid,
                command,
            )
            self._terminate_process_tree(process)
            if process.returncode is None:
                process.returncode = -1
            self._background_processes.append(process)
            returncode = process.returncode if process.returncode is not None else -1
        output_data = capture_command_output(
            root=self.root,
            output_dir=self.output_dir,
            label="bash",
            stdout=stdout or "",
            stderr=stderr or "",
        )
        if timed_out:
            return ToolResult(
                False,
                f"Command timed out after {timeout} second(s).",
                {
                    "command": command,
                    "cwd": str(self.root),
                    "python_path": str(self.python_path) if self.python_path else "",
                    "returncode": returncode,
                    "timed_out": True,
                    **output_data,
                },
            )
        return ToolResult(
            returncode == 0,
            f"Command exited with code {returncode}.",
            {
                "command": command,
                "cwd": str(self.root),
                "python_path": str(self.python_path) if self.python_path else "",
                "returncode": returncode,
                "timed_out": False,
                **output_data,
            },
        )

    def _terminate_process_tree(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            try:
                killer = subprocess.Popen(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if killer.returncode is None:
                    killer.returncode = 0
                self._background_processes.append(killer)
                return
            except (OSError, subprocess.SubprocessError) as exc:
                LOGGER.warning("taskkill failed for pid=%s: %s", process.pid, exc)
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                return
            except OSError as exc:
                LOGGER.warning("killpg failed for pid=%s: %s", process.pid, exc)
        process.kill()

    def _reap_background_processes(self) -> None:
        still_running: list[subprocess.Popen[Any]] = []
        for process in self._background_processes:
            if process.poll() is None:
                still_running.append(process)
                continue
            for stream in (process.stdout, process.stderr):
                if stream:
                    stream.close()
        self._background_processes = still_running


def _decode_timeout_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
