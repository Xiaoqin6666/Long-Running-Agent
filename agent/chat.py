from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from agent.loop import AgentLoop, RunResult


UI_WIDTH = 72
TOOL_ACTIONS = {"bash", "edit", "git", "list_files", "read", "search", "write"}
HELP_TEXT = """Commands:
  /ask TEXT  Ask a question without advancing project work
  /do TEXT   Give the agent a project task to execute
  /help      Show this help
  /status    Show the current durable agent state
  /history   Show messages from this chat session
  /resume    Continue the last unfinished agent run
  /new       Start a new conversation context
  /exit      Exit the chat
"""


@dataclass
class ChatConfig:
    root: Path
    provider: str
    max_steps: int
    benchmark_id: str | None = None
    tasks_path: Path | None = None
    project_spec_path: Path | None = None
    auto_resume: bool = False
    max_sessions: int = 1
    initial_message: str | None = None


@dataclass
class ChatMessage:
    role: str
    content: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content, "created_at": self.created_at}


class InteractiveCLI:
    def __init__(
        self,
        config: ChatConfig,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        use_color: bool | None = None,
    ) -> None:
        self.config = config
        self.input = input_fn
        self.output = output_fn
        self.use_color = (
            bool(getattr(sys.stdout, "isatty", lambda: False)())
            if use_color is None and output_fn is print
            else bool(use_color)
        )
        self.state_dir = config.root / "state"
        if config.benchmark_id:
            self.state_dir = self.state_dir / "benchmarks" / config.benchmark_id
        self.state_path = self.state_dir / "current_task.json"
        self.history_path = self.state_dir / "chat_history.jsonl"
        self.messages: list[ChatMessage] = []
        self.last_task = ""
        self.last_result: RunResult | None = None

    def run(self) -> int:
        if self.use_color:
            self.output("\033[2J\033[H\033[?25h")
        self._show_header()
        if self.config.provider == "offline":
            self.output(self._paint("  Offline mode is intended for deterministic smoke tests.", "yellow"))
            self.output("")

        pending = self.config.initial_message.strip() if self.config.initial_message else ""
        if pending and not pending.startswith("/"):
            pending = f"/do {pending}"
        while True:
            try:
                message = pending or self.input(self._paint("You > ", "cyan", bold=True)).strip()
                pending = ""
            except (EOFError, KeyboardInterrupt):
                self.output("\nSession closed.")
                return 0

            if not message:
                continue
            if message.startswith("/"):
                if self._handle_command(message):
                    return 0
                continue
            self.output("Choose an explicit mode: /ask <question> or /do <task>.")

    def _handle_command(self, raw: str) -> bool:
        parts = raw.split(maxsplit=1)
        command = parts[0].lower()
        content = parts[1].strip() if len(parts) > 1 else ""
        if command in {"/exit", "/quit"}:
            self.output(self._paint("Session closed.", "dim"))
            return True
        if command == "/help":
            self.output(HELP_TEXT.rstrip())
        elif command in {"/ask", "/do"}:
            if not content:
                self.output(f"Usage: {command} <message>")
            else:
                self._run_turn(content, interaction_mode="question" if command == "/ask" else "work")
        elif command == "/status":
            self._show_status()
        elif command == "/history":
            self._show_history()
        elif command == "/resume":
            self._resume()
        elif command == "/new":
            self.messages.clear()
            self.last_task = ""
            self.last_result = None
            self._append_history(ChatMessage("system", "Conversation context reset."))
            self.output("Started a new conversation context.")
        else:
            self.output(f"Unknown command: {command}. Type /help for commands.")
        return False

    def _show_header(self) -> None:
        benchmark = self.config.benchmark_id or "repository"
        title = "LONG-RUNNING AGENT"
        self.output(self._paint("+" + "-" * (UI_WIDTH - 2) + "+", "blue", bold=True))
        self.output(self._paint("|" + title.center(UI_WIDTH - 2) + "|", "blue", bold=True))
        self.output(self._paint("+" + "-" * (UI_WIDTH - 2) + "+", "blue", bold=True))
        self.output(
            f"  Project: {self._paint(benchmark, 'cyan', bold=True)}"
            f"    Provider: {self._paint(self.config.provider, 'green')}"
        )
        self.output(f"  Workspace: {compact_text(self.config.root, UI_WIDTH - 15)}")
        self.output(self._paint("  Use /ask for questions, /do for work, /help for commands.", "dim"))
        self.output("")

    def _paint(self, text: object, color: str, bold: bool = False) -> str:
        value = str(text)
        if not self.use_color:
            return value
        codes = {
            "blue": "34",
            "cyan": "36",
            "green": "32",
            "red": "31",
            "yellow": "33",
            "gray": "90",
            "dim": "2",
        }
        prefix = codes.get(color, "0")
        if bold:
            prefix = f"1;{prefix}"
        return f"\033[{prefix}m{value}\033[0m"

    def _run_turn(self, message: str, interaction_mode: str) -> None:
        user_message = ChatMessage("user", message)
        self.messages.append(user_message)
        self._append_history(user_message)
        self.last_task = message
        self.output(self._paint("Agent > working...", "green", bold=True))
        try:
            loop = self._make_loop(
                self.last_task,
                resume=False,
                include_conversation=True,
                interaction_mode=interaction_mode,
            )
            result = loop.run()
        except Exception as exc:
            answer = f"Run failed: {exc}"
            self.output(f"{self._paint('Agent >', 'red', bold=True)} {answer}\n")
            self._record_assistant(answer)
            return
        self._finish_turn(result)

    def _resume(self) -> None:
        if not self.state_path.exists():
            self.output("There is no saved agent run to resume.")
            return
        task = self.last_task or self._saved_user_goal()
        if not task:
            self.output("The saved state has no user goal to resume.")
            return
        self.output(self._paint("Agent > resuming...", "green", bold=True))
        try:
            result = self._make_loop(task, resume=True, include_conversation=False, interaction_mode="").run()
        except Exception as exc:
            self.output(f"{self._paint('Agent >', 'red', bold=True)} Resume failed: {exc}\n")
            return
        self._finish_turn(result)

    def _make_loop(
        self,
        task: str,
        resume: bool,
        include_conversation: bool,
        interaction_mode: str,
    ) -> AgentLoop:
        return AgentLoop(
            root=self.config.root,
            task=task,
            max_steps=self.config.max_steps,
            provider=self.config.provider,
            resume=resume,
            tasks_path=self.config.tasks_path,
            project_spec_path=self.config.project_spec_path,
            benchmark_id=self.config.benchmark_id,
            auto_resume=self.config.auto_resume,
            max_sessions=self.config.max_sessions,
            event_handler=self._show_event,
            conversation_messages=(
                [{"role": message.role, "content": message.content} for message in self.messages]
                if include_conversation
                else None
            ),
            interaction_mode=interaction_mode,
        )

    def _show_event(self, event: dict[str, object]) -> None:
        event_type = str(event.get("type", "tool_result"))
        action = str(event.get("action", "unknown"))
        if action == "answer":
            return
        target = compact_text(event.get("target", ""), 56)
        detail = f" {target}" if target else ""
        if event_type == "tool_start":
            step = f"{event.get('step', '?'):>2}"
            verb = "calling" if action in TOOL_ACTIONS else "action"
            self.output(self._paint(f"  {step}  {verb} {action}{detail}", "gray"))
            return

        ok = bool(event.get("ok"))
        status = "OK" if ok else "FAILED"
        summary = compact_text(event.get("summary", ""), 100)
        self.output(self._paint(f"      result {status}", "gray"))
        if summary:
            self.output(self._paint(f"      {summary}", "gray"))

    def _finish_turn(self, result: RunResult) -> None:
        self.last_result = result
        answer = result.message
        if not result.completed:
            answer = f"{answer} Use /resume to continue this run."
        self.output(f"{self._paint('Agent >', 'green', bold=True)} {answer}\n")
        self._record_assistant(answer)

    def _record_assistant(self, content: str) -> None:
        message = ChatMessage("assistant", content)
        self.messages.append(message)
        self._append_history(message)

    def _show_status(self) -> None:
        if not self.state_path.exists():
            self.output("No durable agent state exists yet.")
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            self.output(f"Could not read agent state: {exc}")
            return
        self.output(
            "State: "
            f"task={state.get('task_id', 'unknown')}, "
            f"iterations={state.get('iterations', 0)}, "
            f"tokens={state.get('session_used_tokens', 0)}/{state.get('session_budget_tokens', '?')}, "
            f"handoff_ready={state.get('handoff_ready', False)}"
        )

    def _show_history(self) -> None:
        visible = [message for message in self.messages if message.role in {"user", "assistant"}]
        if not visible:
            self.output("No messages in the current conversation context.")
            return
        for message in visible:
            label = "You" if message.role == "user" else "Agent"
            self.output(f"{label} > {message.content}")

    def _saved_user_goal(self) -> str:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return ""
        return str(state.get("user_goal", "")).strip()

    def _append_history(self, message: ChatMessage) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")


def compact_text(value: object, limit: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def launch_chat_window(argv: list[str], cwd: Path | None = None) -> bool:
    """Launch the chat UI in a dedicated Windows console window."""
    if os.name != "nt":
        return False
    child_argv = [argument for argument in argv if argument != "--chat-child"]
    child_argv.append("--chat-child")
    command = [sys.executable, "-m", "agent.main", *child_argv]
    try:
        subprocess.Popen(
            command,
            cwd=str(cwd or Path.cwd()),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    except OSError:
        return False
    return True
