from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from agent.loop import AgentLoop, RunResult
from agent.memory import (
    MemoryDocument,
    find_semantic_duplicate,
    normalize_memory_content,
    render_memory,
    render_memory_index,
    safe_memory_id,
    validate_memory,
)
from agent.spec_builder import build_project_spec
from agent.skills import SkillDocument, parse_skill, render_skill


UI_WIDTH = 72
TOOL_ACTIONS = {"bash", "edit", "git", "list_files", "read", "search", "write"}
HELP_TEXT = """Commands:
  /agent     Switch to agent mode; collect requirements or a project spec file path before starting work
  /send      Start work from the collected /agent requirements or /adjust directions
  /clear     Clear collected /agent requirements or /adjust directions
  /adjust    Switch to adjust mode; provide changes for the existing agent run without reinitializing
  /resume    Continue the last unfinished agent run, optionally with one-time guidance
  /mode      Show the current input mode
  /skill     Add a user-authored Skill with a guided form
  /memory    Add a typed Memory with a guided form
  /status    Show the current durable agent state
  /history   Show messages from this chat session
  /new       Start a new conversation context
  /exit      Exit the chat
"""


@dataclass
class ChatConfig:
    root: Path
    provider: str
    max_steps: int | None
    benchmark_id: str | None = None
    tasks_path: Path | None = None
    project_spec_path: Path | None = None
    auto_resume: bool = False
    max_sessions: int = 1
    system_validation: bool = True
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
        self.active_mode = "idle"
        self.agent_draft: list[str] = []
        self.context_message_start = 0

    def run(self) -> int:
        if self.use_color:
            self.output("\033[2J\033[H\033[?25h")
        self._show_header()
        if self.config.provider == "offline":
            self.output(self._paint("  Offline mode is intended for deterministic smoke tests.", "yellow"))
            self.output("")

        pending = self.config.initial_message.strip() if self.config.initial_message else ""
        if pending and not pending.startswith("/"):
            pending = f"/agent {pending}"
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
            self._handle_plain_message(message)

    def _handle_command(self, raw: str) -> bool:
        parts = raw.split(maxsplit=1)
        command = parts[0].lower()
        content = parts[1].strip() if len(parts) > 1 else ""
        if command in {"/exit", "/quit"}:
            self.output(self._paint("Session closed.", "dim"))
            return True
        if command == "/help":
            self.output(HELP_TEXT.rstrip())
        elif command == "/chat":
            self.active_mode = "chat"
            if not content:
                self.output("Chat mode active. Ask questions normally; messages will not start project work. \nUse /agent to switch to autonomous project work.")
            else:
                self._run_chat_turn(content)
        elif command == "/agent":
            self.active_mode = "agent"
            if not content:
                self.output("Agent mode active. Paste or type project requirements, or say where the project spec file is. Use Shift+Enter/new lines as needed, then /send to start. Use /chat to switch to chat-only mode.")
            else:
                self._run_agent_project_flow(content)
        elif command == "/adjust":
            self.active_mode = "adjust"
            if not content:
                self.output("Adjust mode active. Describe the change you want to make to the existing agent run, then /send. This keeps the current plan/state instead of reinitializing it.")
            else:
                self._run_adjustment_flow(content)
        elif command == "/send":
            if content:
                self._append_agent_draft(content)
            if self.active_mode == "adjust":
                self._send_adjustment_draft()
            else:
                self._send_agent_draft()
        elif command == "/clear":
            self.agent_draft.clear()
            if self.active_mode == "adjust":
                self.output("Cleared collected adjustment directions.")
            else:
                self.output("Cleared collected agent requirements.")
        elif command == "/mode":
            self.output(f"Current mode: {self.active_mode}.")
        elif command in {"/ask", "/do"}:
            # Compatibility aliases for older scripts and tests.
            if not content:
                self.output(f"Usage: {command} <message>")
            elif command == "/ask":
                self.active_mode = "chat"
                self._run_chat_turn(content)
            else:
                self.active_mode = "agent"
                self._run_agent_project_flow(content)
        elif command == "/skill":
            self._run_skill_wizard()
        elif command == "/memory":
            self._run_memory_wizard()
        elif command == "/status":
            self._show_status()
        elif command == "/history":
            self._show_history()
        elif command == "/resume":
            self._resume(content)
        elif command == "/new":
            self.messages.clear()
            self.context_message_start = 0
            self.last_task = ""
            self.last_result = None
            self.active_mode = "idle"
            self.agent_draft.clear()
            self._append_history(ChatMessage("system", "Conversation context reset."))
            self.output("Started a new conversation context.")
        else:
            self.output(f"Unknown command: {command}. Type /help for commands.")
        return False

    def _run_skill_wizard(self) -> None:
        self.output(self._paint("Skill candidate", "cyan", bold=True))
        self.output(self._paint("Enter /cancel at any prompt to stop.", "dim"))
        try:
            name = self._prompt_skill_field("Name", required=True)
            if name is None:
                return
            description = self._prompt_skill_field("Description", required=True)
            if description is None:
                return
            instruction = self._prompt_skill_field("Instruction", required=True)
            if instruction is None:
                return
            example = self._prompt_skill_field("Example (optional)", required=False)
            if example is None:
                return
        except (EOFError, KeyboardInterrupt):
            self.output("\nSkill setup cancelled.")
            return

        self._save_user_skill(name, description, instruction, example)

    def _prompt_skill_field(self, label: str, required: bool) -> str | None:
        while True:
            value = self.input(self._paint(f"{label} > ", "cyan", bold=True)).strip()
            if value.lower() == "/cancel":
                self.output("Skill setup cancelled.")
                return None
            if value or not required:
                return value
            self.output(f"{label} is required.")

    def _run_memory_wizard(self) -> None:
        self.output(self._paint("Memory candidate", "cyan", bold=True))
        self.output(self._paint("Allowed types: user, feedback, project, reference. Enter /cancel at any prompt to stop.", "dim"))
        try:
            name = self._prompt_memory_field("Name", required=True)
            if name is None:
                return
            description = self._prompt_memory_field("Description", required=True)
            if description is None:
                return
            memory_type = self._prompt_memory_type()
            if memory_type is None:
                return
            content = self._prompt_memory_field("Content", required=True)
            if content is None:
                return
            why = ""
            how = ""
            if memory_type == "feedback":
                why = self._prompt_memory_field("Why", required=True) or ""
                if not why:
                    return
                how = self._prompt_memory_field("How to apply", required=True) or ""
                if not how:
                    return
        except (EOFError, KeyboardInterrupt):
            self.output("\nMemory setup cancelled.")
            return

        self._save_user_memory(name, description, memory_type, content, why, how)

    def _prompt_memory_field(self, label: str, required: bool) -> str | None:
        while True:
            value = self.input(self._paint(f"{label} > ", "cyan", bold=True)).strip()
            if value.lower() == "/cancel":
                self.output("Memory setup cancelled.")
                return None
            if value or not required:
                return value
            self.output(f"{label} is required.")

    def _prompt_memory_type(self) -> str | None:
        while True:
            value = self.input(self._paint("Type > ", "cyan", bold=True)).strip().lower()
            if value == "/cancel":
                self.output("Memory setup cancelled.")
                return None
            if value in {"user", "feedback", "project", "reference"}:
                return value
            self.output("Type must be one of: user, feedback, project, reference.")

    def _save_user_memory(
        self,
        name: str,
        description: str,
        memory_type: str,
        content: str,
        why: str = "",
        how_to_apply: str = "",
    ) -> None:
        memory_id = safe_memory_id(name)
        if not memory_id:
            self.output("Memory name must contain a letter, number, underscore, or dash.")
            return
        memory_dir = self.state_dir / "memories"
        memory_path = memory_dir / f"{memory_id}.md"
        if memory_path.exists():
            try:
                confirmation = self.input(
                    self._paint(f"Memory '{memory_id}' exists. Overwrite? [y/N] > ", "yellow", bold=True)
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.output("\nMemory setup cancelled; existing Memory was not changed.")
                return
            if confirmation not in {"y", "yes"}:
                self.output("Memory setup cancelled; existing Memory was not changed.")
                return

        rendered_content = normalize_memory_content(
            {"type": memory_type, "content": content, "why": why, "how_to_apply": how_to_apply}
        )
        memory = MemoryDocument(memory_id, description, memory_type, rendered_content)
        errors = validate_memory(memory)
        if errors:
            self.output("Memory validation failed: " + "; ".join(errors))
            return
        semantic_duplicate = find_semantic_duplicate(memory, memory_dir, exclude_name=memory_id)
        if semantic_duplicate:
            self.output(
                "Memory validation failed: semantically similar to existing Memory "
                f"'{semantic_duplicate['name']}' ({semantic_duplicate['similarity']})."
            )
            return

        memory_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = memory_path.with_suffix(".md.tmp")
        try:
            temporary_path.write_text(render_memory(memory), encoding="utf-8")
            temporary_path.replace(memory_path)
            self.state_dir.mkdir(parents=True, exist_ok=True)
            (self.state_dir / "memory.md").write_text(render_memory_index(memory_dir), encoding="utf-8")
        except OSError as exc:
            temporary_path.unlink(missing_ok=True)
            self.output(f"Could not save Memory: {exc}")
            return

        record = ChatMessage(
            "system",
            f"User added trusted Memory '{memory_id}' at {self._relative_path(memory_path)}.",
        )
        self._append_history(record)
        self.output(self._paint(f"Memory saved: {self._relative_path(memory_path)}", "green", bold=True))

    def _save_user_skill(self, name: str, description: str, instruction: str, example: str) -> None:
        skill_id = safe_skill_id(name)
        if not skill_id:
            self.output("Skill name must contain a letter, number, underscore, or dash.")
            return
        skill_dir = self.state_dir / "skills"
        skill_path = skill_dir / f"{skill_id}.md"
        if skill_path.exists():
            try:
                confirmation = self.input(
                    self._paint(f"Skill '{skill_id}' exists. Overwrite? [y/N] > ", "yellow", bold=True)
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.output("\nSkill setup cancelled; existing Skill was not changed.")
                return
            if confirmation not in {"y", "yes"}:
                self.output("Skill setup cancelled; existing Skill was not changed.")
                return

        skill = SkillDocument(skill_id, description, instruction, example)
        rendered = render_skill(skill)
        parsed = parse_skill(rendered, fallback_name=skill_id)
        if parsed != skill:
            self.output("Skill validation failed; no file was written.")
            return

        skill_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = skill_path.with_suffix(".md.tmp")
        try:
            temporary_path.write_text(rendered, encoding="utf-8")
            temporary_path.replace(skill_path)
        except OSError as exc:
            temporary_path.unlink(missing_ok=True)
            self.output(f"Could not save Skill: {exc}")
            return

        record = ChatMessage(
            "system",
            f"User added trusted Skill '{skill_id}' at {self._relative_path(skill_path)}.",
        )
        self._append_history(record)
        self.output(self._paint(f"Skill saved: {self._relative_path(skill_path)}", "green", bold=True))

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.config.root)).replace("\\", "/")
        except ValueError:
            return str(path)

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
        self.output(self._paint("  Use /agent for new project work, /adjust for changes, /resume for continue in next session, /help for commands.", "dim"))
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

    def _handle_plain_message(self, message: str) -> None:
        if self.active_mode in {"agent", "adjust"}:
            self._append_agent_draft(message)
            return
        if self.active_mode != "chat":
            self.output("Choose /chat for conversation, /agent for new project work, or /adjust to modify an existing run.")
            return
        self._run_chat_turn(message)

    def _run_chat_turn(self, message: str) -> None:
        self._run_turn(message, interaction_mode="question")

    def _append_agent_draft(self, message: str) -> None:
        self.agent_draft.append(message)
        line_count = sum(max(1, len(item.splitlines())) for item in self.agent_draft)
        if self.active_mode == "adjust":
            self.output(f"Added to adjustment directions draft ({line_count} line(s)). Use /send to start or /clear to reset.")
        else:
            self.output(f"Added to agent requirements draft ({line_count} line(s)). Use /send to start or /clear to reset.")

    def _send_agent_draft(self) -> None:
        if self.active_mode != "agent":
            self.output("Switch to /agent before sending project requirements.")
            return
        requirement = "\n".join(item.strip() for item in self.agent_draft if item.strip()).strip()
        if not requirement:
            self.output("No agent requirements collected yet. Paste or type requirements first.")
            return
        self.agent_draft.clear()
        self._run_agent_project_flow(requirement)

    def _send_adjustment_draft(self) -> None:
        if self.active_mode != "adjust":
            self.output("Switch to /adjust before sending adjustment directions.")
            return
        adjustment = "\n".join(item.strip() for item in self.agent_draft if item.strip()).strip()
        if not adjustment:
            self.output("No adjustment directions collected yet. Paste or type the requested changes first.")
            return
        self.agent_draft.clear()
        self._run_adjustment_flow(adjustment)

    def _run_adjustment_flow(self, message: str) -> None:
        if not self.state_path.exists():
            answer = "No existing agent run is available to adjust. Use /agent to start a project first."
            self.output(f"{self._paint('Agent >', 'red', bold=True)} {answer}\n")
            self._record_assistant(answer)
            return

        user_message = ChatMessage("user", message)
        self.messages.append(user_message)
        self._append_history(user_message)
        self.last_task = message

        self.output(self._paint("Agent > adjusting existing run...", "green", bold=True))
        try:
            loop = self._make_loop(
                f"Adjust existing project based on user guidance: {message}",
                resume=True,
                include_conversation=True,
                interaction_mode="adjust",
                project_spec_path=self._existing_project_spec_path(),
                materialize_project_spec=False,
            )
            result = loop.run()
        except Exception as exc:
            answer = f"Run failed: {exc}"
            self.output(f"{self._paint('Agent >', 'red', bold=True)} {answer}\n")
            self._record_assistant(answer)
            return
        self._finish_turn(result)

    def _run_agent_project_flow(self, message: str) -> None:
        source_path = self._extract_project_spec_path(message)
        if source_path:
            benchmark_id = self._infer_benchmark_id_from_path(source_path)
            if benchmark_id:
                self._select_benchmark(benchmark_id)

        user_message = ChatMessage("user", message)
        self.messages.append(user_message)
        self._append_history(user_message)
        self.last_task = message

        self.output(self._paint("Agent > preparing project spec...", "green", bold=True))
        materialize_project_spec = True
        if source_path:
            try:
                project_spec = source_path.read_text(encoding="utf-8")
            except OSError as exc:
                answer = f"Could not read project spec file {self._relative_path(source_path)}: {exc}"
                self.output(f"{self._paint('Agent >', 'red', bold=True)} {answer}\n")
                self._record_assistant(answer)
                return
            spec_path = source_path
            materialize_project_spec = False
            self.output(self._paint(f"Agent > project spec source: {self._relative_path(source_path)}", "green", bold=True))
        else:
            try:
                project_spec = build_project_spec(
                    self.config.provider,
                    [{"role": item.role, "content": item.content} for item in self.messages],
                )
            except Exception as exc:
                answer = f"Could not build project spec: {exc}"
                self.output(f"{self._paint('Agent >', 'red', bold=True)} {answer}\n")
                self._record_assistant(answer)
                return
            spec_path = self._write_project_spec(project_spec)

        self._reset_generated_project_state()
        if materialize_project_spec:
            self.output(self._paint(f"Agent > project spec saved: {self._relative_path(spec_path)}", "green", bold=True))
        self.output(self._paint("Agent > working...", "green", bold=True))

        try:
            loop = self._make_loop(
                project_spec,
                resume=False,
                include_conversation=True,
                interaction_mode="work",
                project_spec_path=spec_path,
                materialize_project_spec=materialize_project_spec,
                use_config_tasks_path=False,
            )
            result = loop.run()
        except Exception as exc:
            answer = f"Run failed: {exc}"
            self.output(f"{self._paint('Agent >', 'red', bold=True)} {answer}\n")
            self._record_assistant(answer)
            return
        self._finish_turn(result)

    def _extract_project_spec_path(self, message: str) -> Path | None:
        candidates: list[str] = []
        markers = [
            "规格文件在",
            "规格文件:",
            "规格文件：",
            "规格说明在",
            "规格说明存放在",
            "项目规格在",
            "项目规格文件在",
            "spec file is",
            "spec file:",
            "project spec is",
            "project spec:",
            "requirements are in",
            "requirements are at",
            "requirements are described in",
            "requirements are described by",
        ]
        for line in message.splitlines():
            lowered = line.lower()
            for marker in markers:
                index = lowered.find(marker)
                if index == -1:
                    continue
                tail = line[index + len(marker) :].strip(" \t:：")
                candidates.extend(self._path_candidates(tail))

        stripped = message.strip()
        if "\n" not in stripped:
            candidates.extend(self._path_candidates(stripped, include_whole=True))

        seen: set[str] = set()
        for candidate in candidates:
            cleaned = self._clean_user_path(candidate)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            path = Path(cleaned).expanduser()
            if not path.is_absolute():
                path = self.config.root / path
            try:
                if path.exists() and path.is_file():
                    return path.resolve()
            except OSError:
                continue
        return None

    def _path_candidates(self, text: str, include_whole: bool = False) -> list[str]:
        candidates: list[str] = []
        if include_whole:
            candidates.append(text)
            return candidates
        quoted = re.findall(r"[`\"']([^`\"']+\.(?:md|markdown|txt))[`\"']", text, flags=re.IGNORECASE)
        candidates.extend(quoted)
        unquoted = re.findall(
            r"((?:[A-Za-z]:[\\/]|\.{1,2}[\\/]|[A-Za-z0-9_.-]+[\\/]).*?\.(?:md|markdown|txt))",
            text,
            flags=re.IGNORECASE,
        )
        candidates.extend(unquoted)
        return candidates

    @staticmethod
    def _clean_user_path(raw: str) -> str:
        return raw.strip().strip("`\"'“”‘’<>").rstrip(" \t,.;，。；、")

    def _infer_benchmark_id_from_path(self, path: Path) -> str | None:
        try:
            relative = path.resolve().relative_to((self.config.root / "eval" / "benchmarks").resolve())
        except ValueError:
            return None
        return relative.parts[0] if relative.parts else None

    def _select_benchmark(self, benchmark_id: str) -> None:
        self.config.benchmark_id = benchmark_id
        self.state_dir = self.config.root / "state" / "benchmarks" / benchmark_id
        self.state_path = self.state_dir / "current_task.json"
        self.history_path = self.state_dir / "chat_history.jsonl"

    def _write_project_spec(self, project_spec: str) -> Path:
        spec_path = self.state_dir / "project_spec.md"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(project_spec, encoding="utf-8")
        return spec_path

    def _existing_project_spec_path(self) -> Path | None:
        state_spec = self.state_dir / "project_spec.md"
        if state_spec.exists():
            return state_spec
        return self.config.project_spec_path

    def _reset_generated_project_state(self) -> None:
        for path in [
            self.state_dir / "generated_tasks.json",
            self.state_dir / "init.sh",
            self.state_dir / "current_task.json",
            self.state_dir / "handoff.md",
            self.state_dir / "handoff_payload.json",
        ]:
            try:
                if path.exists() and path.is_file():
                    path.unlink()
            except OSError:
                pass

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

    def _resume(self, instruction: str = "") -> None:
        if not self.state_path.exists():
            self.output("There is no saved agent run to resume.")
            return
        task = self.last_task or self._saved_user_goal()
        if not task:
            self.output("The saved state has no user goal to resume.")
            return
        instruction = instruction.strip()
        self.context_message_start = len(self.messages)
        if instruction:
            user_message = ChatMessage("user", instruction)
            self.messages.append(user_message)
            self._append_history(user_message)
        self.output(self._paint("Agent > resuming...", "green", bold=True))
        try:
            result = self._make_loop(
                task,
                resume=True,
                include_conversation=bool(instruction),
                interaction_mode="adjust" if instruction else "",
            ).run()
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
        project_spec_path: Path | None = None,
        materialize_project_spec: bool = True,
        use_config_tasks_path: bool = True,
    ) -> AgentLoop:
        return AgentLoop(
            root=self.config.root,
            task=task,
            max_steps=self.config.max_steps,
            provider=self.config.provider,
            resume=resume,
            tasks_path=self.config.tasks_path if use_config_tasks_path else None,
            project_spec_path=project_spec_path or self.config.project_spec_path,
            materialize_project_spec=materialize_project_spec,
            benchmark_id=self.config.benchmark_id,
            auto_resume=self.config.auto_resume,
            max_sessions=self.config.max_sessions,
            system_validation=self.config.system_validation,
            event_handler=self._show_event,
            conversation_messages=(
                [
                    {"role": message.role, "content": message.content}
                    for message in self.messages[self.context_message_start :]
                ]
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
        if event_type != "tool_start":
            return

        thought_summary = str(event.get("thought_summary", ""))
        if thought_summary:
            self.output(f"      {thought_summary}")
        if action == "verify":
            return

        target = str(event.get("target", ""))
        detail = f" {target}" if target else ""
        verb = "calling" if action in TOOL_ACTIONS else "action"
        self.output(self._paint(f"      {verb} {action}{detail}", "gray"))

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
        token_totals = {}
        token_usage = state.get("token_usage", {})
        if isinstance(token_usage, dict) and isinstance(token_usage.get("totals"), dict):
            token_totals = token_usage["totals"]
        self.output(
            "State: "
            f"task={state.get('task_id', 'unknown')}, "
            f"iterations={state.get('iterations', 0)}, "
            f"turn_tokens={state.get('session_used_tokens', 0)}/{state.get('session_budget_tokens', '?')}, "
            f"llm_input_tokens={token_totals.get('input_tokens', 0)}, "
            f"llm_output_tokens={token_totals.get('output_tokens', 0)}, "
            f"llm_cost={format_costs_by_currency(token_totals)}, "
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


def safe_skill_id(raw: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in raw.strip().lower())
    return cleaned.strip("-_")


def format_costs_by_currency(totals: dict[str, object]) -> str:
    costs = totals.get("costs_by_currency", {}) if isinstance(totals, dict) else {}
    if not isinstance(costs, dict) or not costs:
        unpriced = int(totals.get("unpriced_turn_count", 0) or 0) if isinstance(totals, dict) else 0
        return f"not_configured(unpriced_turns={unpriced})"
    parts: list[str] = []
    for currency, values in sorted(costs.items()):
        if not isinstance(values, dict):
            continue
        parts.append(f"{currency}:{float(values.get('total_cost', 0.0) or 0.0):.6f}")
    unpriced = int(totals.get("unpriced_turn_count", 0) or 0) if isinstance(totals, dict) else 0
    if unpriced:
        parts.append(f"unpriced_turns:{unpriced}")
    return "|".join(parts) if parts else "not_configured"


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
