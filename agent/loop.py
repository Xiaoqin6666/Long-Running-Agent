from __future__ import annotations

from itertools import count
import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent.context import ContextBuilder
from agent.llm import create_decision_maker
from agent.memory import (
    MemoryDocument,
    memory_catalog,
    normalize_memory_content,
    parse_memory,
    render_memory,
    render_memory_index,
    safe_memory_id,
    validate_memory,
)
from agent.memory_retrieval import MemoryRetriever, render_relevant_memories
from agent.orchestrator import Orchestrator
from agent.planner import (
    TaskState,
    create_initial_state,
    create_initializer_state,
    validate_generated_task_graph,
    validate_initializer_script,
)
from agent.prompts import MAIN_AGENT_SYSTEM_PROMPT
from agent.skills import (
    SkillDocument,
    normalize_examples,
    normalize_instruction,
    parse_skill,
    render_skill,
    skill_catalog,
)
from agent.termination import ProjectTerminator
from agent.tools import BashTool, EditTool, GitTool, ListFilesTool, ReadTool, SearchTool, ToolResult, WriteTool
from agent.verifier import Verifier


LOGGER = logging.getLogger("long_agent")
SKILL_REFLECTION_SESSION_THRESHOLD = 5
SKILL_REFLECTION_ERROR_THRESHOLD = 3


@dataclass
class RunResult:
    completed: bool
    steps: int
    trace_path: Path
    state_path: Path
    message: str
    sessions: int = 1

    def to_human_summary(self) -> str:
        status = "completed" if self.completed else "stopped"
        return (
            f"Agent {status} after {self.steps} step(s) across {self.sessions} session(s).\n"
            f"State: {self.state_path}\n"
            f"Trace: {self.trace_path}\n"
            f"{self.message}"
        )


class AgentLoop:
    def __init__(
        self,
        root: Path,
        task: str,
        max_steps: int | None,
        provider: str = "offline",
        resume: bool = False,
        tasks_path: Path | None = None,
        project_spec_path: Path | None = None,
        benchmark_id: str | None = None,
        auto_resume: bool = False,
        max_sessions: int = 1,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
        conversation_messages: list[dict[str, str]] | None = None,
        interaction_mode: str = "",
    ) -> None:
        self.root = root
        self.task = task
        self.max_steps = max_steps if max_steps and max_steps > 0 else None
        self.provider = provider
        self.resume = resume
        self.auto_resume = auto_resume
        self.max_sessions = max(1, max_sessions)
        self.event_handler = event_handler
        self.conversation_messages = self._normalize_conversation_messages(conversation_messages or [])
        self.interaction_mode = interaction_mode if interaction_mode in {"question", "work"} else ""
        self.project_spec_path = project_spec_path
        self.source_tasks_path = tasks_path
        self.benchmark_id = self._safe_benchmark_id(benchmark_id) if benchmark_id else None
        self.state_dir = self._benchmark_state_dir(self.benchmark_id) if self.benchmark_id else root / "state"
        self.trace_dir = self.state_dir / "traces"
        self.debug_context_dir = self.state_dir / "debug_contexts"
        self.project_spec_materialized_path = self.state_dir / "project_spec.md" if project_spec_path else None
        self.generated_tasks_path = self.state_dir / "generated_tasks.json" if project_spec_path and not tasks_path else None
        self.runtime_tasks_path = self.state_dir / "runtime_tasks.json" if tasks_path else None
        self.tasks_path = self.runtime_tasks_path or self.generated_tasks_path or tasks_path
        self.state_path = self.state_dir / "current_task.json"
        self.memory_path = self.state_dir / "memory.md"
        self.handoff_path = self.state_dir / "handoff.md"
        self.handoff_payload_path = self.state_dir / "handoff_payload.json"
        self.initializer_candidate_path = self.state_dir / "rejected_candidates" / "generated_tasks.json"
        self.trace_path = self.trace_dir / self._trace_name()
        expected_workspace = self._expected_initializer_workspace_root()
        benchmark_python_path = (root / expected_workspace).resolve() if expected_workspace else None
        benchmark_git_root = benchmark_python_path if self.benchmark_id else None
        self.context_builder = ContextBuilder(root, state_dir=self.state_dir, git_root=benchmark_git_root)
        self.memory_retriever = MemoryRetriever.from_env(self.state_dir)
        self.orchestrator = Orchestrator(root, tasks_path=self.tasks_path, state_dir=self.state_dir)
        self.terminator = ProjectTerminator(root, tasks_path=self.tasks_path, benchmark_id=self.benchmark_id)
        self.decision_maker = create_decision_maker(provider)
        self.verifier = Verifier(root, state_dir=self.state_dir)
        self.tools = {
            "bash": BashTool(root, python_path=benchmark_python_path),
            "edit": EditTool(root),
            "git": GitTool(
                benchmark_git_root or root,
                allow_write=True,
                auto_init=bool(benchmark_git_root),
                scope_description=(
                    f"benchmark workspace {expected_workspace}" if benchmark_git_root else "workspace"
                ),
            ),
            "list_files": ListFilesTool(root),
            "read": ReadTool(root),
            "search": SearchTool(root),
            "write": WriteTool(root),
        }
        self._last_memory_selection: dict[str, Any] = {}
        self._current_context_snapshot: dict[str, Any] = {}

    def _benchmark_state_dir(self, benchmark_id: str) -> Path:
        return self.root / "state" / "benchmarks" / benchmark_id

    def _safe_benchmark_id(self, raw: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in raw.strip())
        cleaned = cleaned.strip("-_")
        if not cleaned:
            raise ValueError("benchmark_id must contain a letter, number, underscore, or dash")
        return cleaned

    def run(self) -> RunResult:
        LOGGER.info("Preparing state directory=%s", self.state_dir)
        self._ensure_state_files()
        self._prepare_runtime_task_graph()
        state = self._load_or_create_state()
        total_steps = 0
        completed = False
        message = "Reached max steps before completion."
        sessions = 1

        while sessions <= self.max_sessions:
            LOGGER.info(
                "Session %s/%s started task_id=%s trace=%s",
                sessions,
                self.max_sessions,
                state.task_id,
                self.trace_path,
            )
            session = self._run_one_session(state)
            total_steps += session.steps
            state = session.state
            completed = session.completed
            message = session.message
            if completed:
                break
            if not session.handoff_ready:
                break
            if not self.auto_resume or sessions >= self.max_sessions:
                break
            sessions += 1
            LOGGER.info("Auto-resuming from handoff into session %s/%s", sessions, self.max_sessions)
            state = self._prepare_auto_resume_session()

        if not completed:
            self._write_handoff(state)

        LOGGER.info(
            "Loop stopped completed=%s total_steps=%s sessions=%s handoff_ready=%s message=%s",
            completed,
            total_steps,
            sessions,
            state.handoff_ready,
            message,
        )

        return RunResult(
            completed=completed,
            steps=total_steps,
            trace_path=self.trace_path,
            state_path=self.state_path,
            message=message,
            sessions=sessions,
        )

    @dataclass
    class _SessionResult:
        completed: bool
        handoff_ready: bool
        steps: int
        state: TaskState
        message: str

    def _run_one_session(self, state: TaskState) -> _SessionResult:
        steps = 0
        completed = False
        message = "Reached max steps before completion."
        self._record_task_session(state)

        step_iter = count(1) if self.max_steps is None else range(1, self.max_steps + 1)
        for step in step_iter:
            steps = step
            self._current_trace_step = step
            self._record_task_session(state)
            memory_context = self._relevant_memory_context(state)
            self.context_builder.current_trace_path = self.trace_path
            context = self.context_builder.build(state, relevant_memories=memory_context, include_handoff=(step == 1))
            context_snapshot = self._record_context_snapshot(step, state, context)
            try:
                action = self.decision_maker.next_action(context, state)
                self._emit_event(
                    {
                        "type": "tool_start",
                        "step": step,
                        "action": str(action.get("action", "unknown")),
                        "target": str(action.get("target", "")),
                    }
                )
                observation = self._execute_action(action, state)
            except Exception as exc:
                LOGGER.exception("Step %s model/tool protocol error", step)
                action = {
                    "thought_summary": "Harness caught a model or tool protocol error.",
                    "action": "protocol_error",
                    "target": "decision_maker",
                    "args": {},
                    "expected_observation": "The error is recorded so the loop can continue.",
                    "risk": "low",
                }
                observation = ToolResult(False, f"Protocol error: {exc}", {"error_type": type(exc).__name__})
            self._record_budget_usage(state, context, action, observation)
            self._update_state(state, action, observation)
            self._append_trace(step, action, observation, state, context_snapshot)
            self._write_state(state)
            LOGGER.info(
                "Step %s action=%s target=%s ok=%s task_id=%s turn_tokens=%s handoff_ready=%s observation=%s",
                step,
                action.get("action", "unknown"),
                action.get("target", ""),
                observation.ok,
                state.task_id,
                state.session_used_tokens,
                state.handoff_ready,
                self._log_text(observation.summary),
            )
            self._emit_event(
                {
                    "type": "tool_result",
                    "step": step,
                    "session": len(state.task_session_ids.get(self._active_task_id(state), [])) or 1,
                    "action": str(action.get("action", "unknown")),
                    "target": str(action.get("target", "")),
                    "ok": observation.ok,
                    "summary": observation.summary,
                }
            )

            if (
                action["action"] in {"answer", "finish"}
                and observation.ok
                and (not self._is_initializer_task(state) or state.interaction_mode == "question")
            ):
                completed = True
                message = observation.data.get("answer", observation.summary)
                break
            if action["action"] == "verify" and observation.ok:
                self._apply_orchestrator_selection(state)
                self._write_state(state)
            if state.handoff_ready:
                self._write_handoff(state)
                message = "Session handoff threshold reached. Handoff written."
                break

        return self._SessionResult(
            completed=completed,
            handoff_ready=state.handoff_ready,
            steps=steps,
            message=message,
            state=state,
        )

    def _emit_event(self, event: dict[str, Any]) -> None:
        if self.event_handler is None:
            return
        try:
            self.event_handler(event)
        except Exception:
            LOGGER.exception("Agent event handler failed")

    def _prepare_auto_resume_session(self) -> TaskState:
        self.resume = True
        self.trace_path = self.trace_dir / self._trace_name()
        state = self._load_or_create_state()
        self._write_state(state)
        return state

    @staticmethod
    def _log_text(value: object, limit: int = 500) -> str:
        text = " ".join(str(value).split())
        return text if len(text) <= limit else f"{text[:limit]}..."

    def _load_or_create_state(self) -> TaskState:
        if self.resume and self.state_path.exists():
            state = TaskState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8-sig")))
            state.session_used_tokens = 0
            state.handoff_ready = False
        elif self._initializer_needed():
            state = create_initializer_state(
                self.task,
                project_spec_artifact=self._rel(self.project_spec_materialized_path or self.root / "project_spec.md"),
                generated_tasks_artifact=self._rel(self.generated_tasks_path or self.state_dir / "generated_tasks.json"),
                init_artifact=self._rel(self.state_dir / "init.sh"),
            )
        else:
            state = create_initial_state(self.task)
        if not self.resume and self.conversation_messages:
            state.conversation_messages = list(self.conversation_messages)
        if not self.resume and self.interaction_mode:
            state.interaction_mode = self.interaction_mode
        self._dedupe_contracts(state)
        if self._is_initializer_task(state) and not state.initializer_repair:
            self._recover_initializer_repair_from_state(state)
        if not self._is_initializer_task(state):
            self._apply_orchestrator_selection(state)
        if self.resume and not state.pending_repair:
            self._recover_pending_repair_from_recent_trace(state)
        return state

    @staticmethod
    def _normalize_conversation_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            normalized.append({"role": role, "content": content})
        return normalized

    def _apply_orchestrator_selection(self, state: TaskState) -> None:
        selection = self.orchestrator.choose_current_task()
        state.orchestrator_decision = selection.to_dict()
        if not selection.task:
            return
        task = selection.task
        task_id = str(task.get("id", state.task_id))
        state.task_id = task_id
        if not state.user_goal.startswith(f"{task_id}:"):
            state.user_goal = f"{task_id}: {task.get('title', state.user_goal)}"
        acceptance = task.get("acceptance_criteria")
        if isinstance(acceptance, list) and acceptance:
            state.acceptance_criteria = [str(item) for item in acceptance]
        state.nodes = [
            {
                "id": task_id,
                "title": str(task.get("title", task_id)),
                "status": str(task.get("status", "pending")),
                "evidence": task.get("evidence", []),
                "acceptance_criteria": task.get("acceptance_criteria", []),
                "depends_on": task.get("depends_on", []),
                "priority": task.get("priority", 1000),
                "expected_artifacts": task.get("expected_artifacts", []),
                "implementation_artifacts": task.get("implementation_artifacts", []),
                "worker_test_artifacts": task.get("worker_test_artifacts", []),
                "acceptance_artifacts": task.get("acceptance_artifacts", []),
                "frozen_acceptance_artifacts": task.get("frozen_acceptance_artifacts", []),
                "test_policy": task.get("test_policy", {}),
                "verification_commands": task.get("verification_commands", []),
                "criterion_command_map": task.get("criterion_command_map", {}),
                "contract_managed": True,
            }
        ]
        if str(task.get("status", "pending")) == "pending":
            updated = self.orchestrator.transition_task(task_id, "in_progress", "scheduled by orchestrator")
            if updated:
                state.nodes[0]["status"] = "in_progress"
        self._ensure_frozen_acceptance_contract(state, task)

    def _record_task_session(self, state: TaskState, task_id: str | None = None) -> None:
        task_id = task_id or self._active_task_id(state)
        if not task_id:
            return
        session_id = self.trace_path.stem
        sessions = state.task_session_ids.setdefault(task_id, [])
        if session_id not in sessions:
            sessions.append(session_id)

    def _ensure_frozen_acceptance_contract(self, state: TaskState, task: dict[str, Any]) -> None:
        task_id = str(task.get("id", state.task_id))
        criteria = [str(item) for item in task.get("acceptance_criteria", [])]
        commands = [str(item) for item in task.get("verification_commands", [])]
        raw_mapping = task.get("criterion_command_map")
        if isinstance(raw_mapping, dict):
            mapping = {
                str(criterion): [str(command) for command in mapped]
                for criterion, mapped in raw_mapping.items()
                if isinstance(mapped, list)
            }
        else:
            mapping = {}
        if not mapping and criteria and commands:
            # Durable pre-mapping task graphs remain resumable; newly generated graphs are validated strictly.
            mapping = {criterion: list(commands) for criterion in criteria}
        contract = {
            "task_id": task_id,
            "summary": f"Frozen task-graph acceptance contract for {task_id}: {task.get('title', task_id)}",
            "scope": [str(item) for item in task.get("expected_artifacts", [])],
            "frozen_requirements": criteria,
            "verification_procedure": {"commands": commands},
            "checks": commands,
            "criterion_command_map": mapping,
            "required_evidence": criteria,
            "forbidden_shortcuts": [
                "Do not weaken frozen requirements after task activation.",
                "Verification procedure may be corrected only to prove the same frozen requirements.",
            ],
            "source": "task_graph",
            "frozen": True,
            "status": "proposed",
        }
        existing = next(
            (
                item
                for item in reversed(state.acceptance_contracts)
                if item.get("task_id") == task_id
                and item.get("source") == "task_graph"
                and item.get("frozen") is True
                and item.get("frozen_requirements", item.get("required_evidence", [])) == criteria
            ),
            None,
        )
        if existing and existing.get("status") == "agreed":
            existing.setdefault("frozen_requirements", list(criteria))
            existing.setdefault("verification_procedure", {"commands": list(existing.get("checks", commands))})
            return
        result = self.verifier.validate_contract(contract, task)
        contract["status"] = "agreed" if result.ok else "rejected"
        contract["validation"] = result.data.get("checks", {})
        state.acceptance_contracts = [
            item
            for item in state.acceptance_contracts
            if not (item.get("task_id") == task_id and item.get("source") == "task_graph")
        ]
        state.acceptance_contracts.append(contract)

    def _record_context_snapshot(self, step: int, state: TaskState, context: str) -> dict[str, Any]:
        session_dir = self.debug_context_dir / self.trace_path.stem
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / f"step_{step:04d}.md"
        content = (
            "# Full Model Context\n\n"
            f"- trace: {self._rel(self.trace_path)}\n"
            f"- step: {step}\n"
            f"- task_id: {self._active_task_id(state)}\n"
            f"- written_at: {utc_now()}\n\n"
            "## System Message\n\n"
            f"{MAIN_AGENT_SYSTEM_PROMPT}\n\n"
            "## User Context\n\n"
            f"{context}\n"
        )
        path.write_text(content, encoding="utf-8")
        snapshot = {
            "path": self._rel(path),
            "trace": self._rel(self.trace_path),
            "step": step,
            "chars": len(content),
            "system_chars": len(MAIN_AGENT_SYSTEM_PROMPT),
            "user_context_chars": len(context),
        }
        self._current_context_snapshot = snapshot
        return snapshot

    def _handle_debug_context_action(self, action: dict[str, Any]) -> ToolResult:
        args = action.get("args", {})
        include_content = bool(args.get("include_content")) if isinstance(args, dict) else False
        target = str(action.get("target", "") or "current").strip().lower()
        snapshot = self._current_context_snapshot
        if target and target not in {"current", "latest"}:
            try:
                step = int(target)
            except ValueError:
                return ToolResult(False, f"Debug context target must be 'current' or a step number, got: {target}", {})
            candidate = self.debug_context_dir / self.trace_path.stem / f"step_{step:04d}.md"
            if not candidate.exists():
                return ToolResult(False, f"No debug context snapshot exists for step {step}.", {"step": step})
            snapshot = {
                "path": self._rel(candidate),
                "trace": self._rel(self.trace_path),
                "step": step,
                "chars": candidate.stat().st_size,
            }
        if not snapshot:
            return ToolResult(False, "No debug context snapshot has been recorded yet.", {})
        data = dict(snapshot)
        if include_content:
            path = self.root / str(snapshot.get("path", ""))
            data["content"] = path.read_text(encoding="utf-8") if path.exists() else ""
        return ToolResult(True, f"Debug context snapshot is available at {snapshot.get('path')}.", data)

    def _execute_action(self, action: dict[str, Any], state: TaskState) -> ToolResult:
        name = action.get("action")
        if state.interaction_mode == "question" and name in {
            "bash",
            "contract",
            "dismiss_skill",
            "edit",
            "finish",
            "git",
            "load_skill",
            "save_memory",
            "save_skill",
            "skill",
            "update_plan",
            "verify",
            "write",
        }:
            return ToolResult(
                False,
                (
                    f"Interactive question cannot use project-progress action '{name}'. "
                    "Answer the latest user question; use only bounded read-only inspection if more evidence is needed."
                ),
                {"interactive_question": True, "required_action": "answer_or_read_only_inspection"},
            )
        if state.pending_skill_review and name not in {"debug_context", "save_skill", "skill", "dismiss_skill"}:
            return ToolResult(
                False,
                "Pending Skill Reflection must be resolved with save_skill or dismiss_skill before ordinary work continues.",
                {
                    "required_action": "save_skill_or_dismiss_skill",
                    "pending_skill_review": state.pending_skill_review,
                    "counts_as_progress": False,
                },
            )
        if name == "debug_context":
            return self._handle_debug_context_action(action)
        if name == "dismiss_skill":
            return self._handle_dismiss_skill_action(action, state)
        if name == "answer":
            if self._is_initializer_task(state) and state.interaction_mode != "question":
                outputs = self._validate_initializer_outputs()
                data = dict(outputs.data)
                data.update({"initializer_requires_verification": True, "counts_as_progress": False})
                return ToolResult(
                    False,
                    outputs.summary if not outputs.ok else (
                        "INIT answer rejected: Initializer must run its verification command and receive "
                        "Verifier PASS before handoff to the first Worker task."
                    ),
                    data,
                )
            evidence_ok, evidence_message = self._check_answer_evidence(state)
            if not evidence_ok:
                return ToolResult(False, evidence_message, {"missing_evidence": True})
            answer = str(action.get("args", {}).get("answer") or action.get("target") or "")
            if not answer.strip():
                return ToolResult(False, "Answer rejected because it was empty.", {})
            return ToolResult(True, "Final answer produced.", {"answer": answer})
        if name == "contract":
            if self._is_initializer_task(state):
                return ToolResult(
                    False,
                    "INIT does not create an acceptance contract; write only initializer artifacts.",
                    {"initializer_restricted": True, "allowed_targets": sorted(self._initializer_allowed_targets(state))},
                )
            return self._validate_contract_action(action, state)
        if name == "load_skill":
            return self._handle_load_skill_action(action, state)
        if name == "save_memory":
            if self._is_initializer_task(state):
                return ToolResult(False, "Memory saving is disabled during INIT.", {"initializer_restricted": True})
            return self._handle_save_memory_action(action, state)
        if name in {"save_skill", "skill"}:
            if self._is_initializer_task(state):
                return ToolResult(False, "Skill promotion is disabled during INIT.", {"initializer_restricted": True})
            return self._handle_save_skill_action(action, state)
        if name == "bash":
            repeated_bash = self._reject_repeated_failed_bash(action, state)
            if repeated_bash is not None:
                return repeated_bash
        if name in {"edit", "write"} and state.handoff_ready:
            return ToolResult(
                False,
                "Write rejected: session is past handoff threshold; generate handoff instead of starting new edits.",
                {"handoff_ready": True},
            )
        if name in {"edit", "write"} and self._is_initializer_task(state):
            initializer_check = self._validate_initializer_write_action(action, state)
            if initializer_check is not None:
                return initializer_check
        elif name in {"edit", "write"} and not self._active_node(state):
            return ToolResult(
                False,
                "Write rejected: no active worker task is writable. Run finish to let the harness schedule a repair task, or verify if a task was just completed.",
                {"no_active_task": True, "required_action": "finish_or_verify"},
            )
        elif name in {"edit", "write"} and not self._has_contract_for_active_task(state):
            return ToolResult(
                False,
                "Write rejected: create an acceptance contract with the verifier before generating code.",
                {"missing_contract": True},
            )
        if name == "bash" and self._is_initializer_task(state):
            allowed_commands = set(self._active_verification_commands(state))
            if str(action.get("target", "")) not in allowed_commands:
                return ToolResult(
                    False,
                    "INIT may run only its deterministic initializer verification command.",
                    {"initializer_restricted": True, "allowed_commands": sorted(allowed_commands)},
                )
        if name == "update_plan":
            return ToolResult(True, "Plan updated by harness.", {"target": action.get("target")})
        if name == "verify":
            task_id = self._active_task_id(state)
            if self._is_initializer_task(state):
                initializer_result = self._validate_initializer_outputs()
                if not initializer_result.ok:
                    initializer_result.data["task_id"] = task_id
                    self.verifier.record_result(initializer_result)
                    return initializer_result
            self.orchestrator.mark_awaiting_verification(task_id, "worker submitted candidate for verification")
            result = self.verifier.run(action.get("target", "default"), state)
            result.data["task_id"] = task_id
            if result.ok:
                archived = self._archive_verifier_success(task_id, result)
                result.data.update(archived)
            pending_skills = [item for item in state.loaded_skills if item.get("status") == "loaded"]
            if pending_skills:
                validation_status = "verified_pass" if result.ok else "verified_fail"
                result.data["skill_validation"] = [
                    {
                        "name": item.get("name"),
                        "status": validation_status,
                        "tool_calls_since_load": max(
                            0, state.iterations - int(item.get("loaded_iteration", state.iterations)) - 1
                        ),
                    }
                    for item in pending_skills
                ]
                for item in pending_skills:
                    item["status"] = validation_status
                    item["verified_at"] = utc_now()
            self.orchestrator.mark_verified(task_id, result.ok, result.summary)
            return result
        if name == "finish":
            if self._is_initializer_task(state):
                return ToolResult(
                    False,
                    "INIT finish rejected: only Verifier PASS may complete initialization.",
                    {"initializer_requires_verification": True, "counts_as_progress": False},
                )
            termination = self.terminator.evaluate()
            if termination.status == "completed":
                return ToolResult(True, "Project completed.", termination.to_dict())
            return ToolResult(False, f"Finish rejected: {termination.status}.", termination.to_dict())
        tool = self.tools.get(str(name))
        if not tool:
            return ToolResult(False, f"Unknown action: {name}", {})
        result = tool.run(action)
        if result.ok and name in {"edit", "write"} and self._is_initializer_task(state):
            post_write_check = self._validate_initializer_artifact_after_write(action, state)
            if post_write_check is not None:
                return post_write_check
        return result

    def _reject_repeated_failed_bash(self, action: dict[str, Any], state: TaskState) -> ToolResult | None:
        last_action = state.last_action if isinstance(state.last_action, dict) else {}
        last_observation = state.last_observation if isinstance(state.last_observation, dict) else {}
        if last_action.get("action") != "bash" or last_observation.get("ok") is not False:
            return None
        command = self._bash_command_from_action(action)
        last_command = self._bash_command_from_action(last_action)
        if not command or self._normalize_command(command) != self._normalize_command(last_command):
            return None
        return ToolResult(
            False,
            (
                "Repeated bash rejected: the same command just failed. "
                "Use the existing failure output to repair the implementation or update the verification procedure; "
                "do not rerun it unchanged to look for a fuller traceback."
            ),
            {
                "command": command,
                "required_action": "repair_or_update_verification",
                "counts_as_progress": False,
            },
        )

    def _bash_command_from_action(self, action: dict[str, Any]) -> str:
        args = action.get("args", {})
        if isinstance(args, dict) and args.get("command"):
            return str(args.get("command", ""))
        return str(action.get("target", ""))

    def _pending_repair_command_failure_type(self, state: TaskState) -> str | None:
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        recorded = str(repair.get("command_failure_type", ""))
        if recorded:
            return recorded
        return self._command_failure_type(
            str(repair.get("output", "")),
            command=str(repair.get("command", "")),
            state=state,
        )

    def _command_failure_type(
        self,
        output: str,
        command: str = "",
        state: TaskState | None = None,
    ) -> str | None:
        if "SyntaxError:" in output and "invalid syntax" in output:
            return "command_syntax_error"
        if self._looks_like_cwd_environment_failure(output, command):
            return "command_environment_error"
        if state and self._command_has_unconfigured_workspace_module(command, state):
            return "command_environment_error"
        return None

    def _looks_like_cwd_environment_failure(self, output: str, command: str) -> bool:
        if not command:
            return False
        combined = f"{output}\n{command}".replace("\\", "/")
        if not any(marker in combined for marker in ("NotADirectoryError", "WinError 267")):
            return False
        return "cwd=" in combined or "os.chdir(" in combined or "subprocess.run" in combined

    def _suggest_corrected_command(self, command: str, failure_type: str, state: TaskState) -> str:
        if failure_type == "command_environment_error":
            workspace = self._workspace_root_from_artifacts(state)
            if not workspace:
                return ""
            return f"python -c \"import sys; sys.path.insert(0,{workspace!r}); print('workspace import path configured')\""
        if failure_type != "command_syntax_error":
            return ""
        if "tempfile.NamedTemporaryFile" not in command or "python -c" not in command:
            return ""
        module = self._module_invoked_by_command(command)
        if not module:
            return ""
        workspace = self._workspace_root_from_command(command) or self._workspace_root_from_artifacts(state)
        sys_path = f"sys.path.insert(0,{workspace!r}); " if workspace else ""
        pretty = "--pretty" in command
        args = f"[sys.executable,'-m',{module!r},str(p){",'--pretty'" if pretty else ''}]"
        pretty_assert = "; assert '\\n  ' in r.stdout" if pretty else ""
        return (
            "python -c \"import sys,json,tempfile,subprocess,pathlib; "
            f"{sys_path}"
            "p=pathlib.Path(tempfile.gettempdir())/'long_agent_todos.txt'; "
            "p.write_text('[ ] task1\\n[x] task2\\n', encoding='utf-8'); "
            f"r=subprocess.run({args},capture_output=True,text=True); "
            "p.unlink(missing_ok=True); "
            "assert r.returncode==0, r.stderr; "
            "assert json.loads(r.stdout)=={'total':2,'done':1,'open':1}"
            f"{pretty_assert}; "
            f"print({'Pretty JSON OK' if pretty else 'Basic JSON OK'!r})\""
        )

    def _module_invoked_by_command(self, command: str) -> str | None:
        patterns = [
            r"\[\s*sys\.executable\s*,\s*['\"]-m['\"]\s*,\s*['\"]([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)['\"]",
            r"\bpython\s+-m\s+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, command)
            if match:
                return match.group(1)
        return None

    def _command_has_unconfigured_workspace_module(self, command: str, state: TaskState) -> bool:
        if not command or self._workspace_root_from_command(command):
            return False
        workspace_root = self._workspace_root_from_artifacts(state)
        if not workspace_root:
            return False
        modules: list[str] = []
        for pattern in (
            r"\bfrom\s+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\s+import\b",
            r"\bpython\s+-m\s+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\b",
            r"['\"]-m['\"]\s*,\s*['\"]([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)['\"]",
        ):
            modules.extend(match.group(1) for match in re.finditer(pattern, command))
        for module in dict.fromkeys(modules):
            module_path = self.root / workspace_root / Path(*module.split("."))
            if module_path.with_suffix(".py").exists() or (module_path / "__init__.py").exists():
                return True
        return False

    def _pending_repair_targets(self, state: TaskState) -> list[str]:
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        targets = repair.get("targets", [])
        if not isinstance(targets, list):
            return []
        active = {self._normalize_target(target) for target in self._active_task_expected_artifacts(state)}
        command = str(repair.get("command", ""))
        output = str(repair.get("output", ""))
        if command or output:
            targets = list(targets) + self._module_repair_targets_from_failure(command, output, state)
        result: list[str] = []
        for target in targets:
            normalized = self._normalize_target(target)
            if (normalized in active or "/workspace/" in normalized) and normalized not in result:
                result.append(normalized)
        return result

    def _pending_repair_write_targets(self, state: TaskState) -> list[str]:
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        command = str(repair.get("command", ""))
        output = str(repair.get("output", ""))
        dynamic_targets = (
            self._module_repair_targets_from_failure(command, output, state)
            if command or output
            else []
        )
        explicit = repair.get("repair_targets", [])
        if isinstance(explicit, list) and (explicit or dynamic_targets):
            active = {self._normalize_target(item) for item in self._active_task_expected_artifacts(state)}
            combined = list(dynamic_targets) + list(explicit)
            seen: set[str] = set()
            return [
                target
                for target in (self._normalize_target(item) for item in combined)
                if (target in active or "/workspace/" in target)
                and (not self._looks_like_test_artifact(target) or self._is_test_repair_allowed(target, state))
                and not (target in seen or seen.add(target))
            ]
        targets = self._pending_repair_targets(state)
        implementation_targets = [
            target
            for target in self._active_task_implementation_artifacts(state)
            if target in {self._normalize_target(item) for item in targets}
        ]
        if implementation_targets:
            return implementation_targets
        mutable_test_targets = [target for target in targets if self._is_test_repair_allowed(target, state)]
        return mutable_test_targets or targets

    def _artifact_referenced_by_command(self, command: str, state: TaskState) -> str | None:
        normalized_command = command.replace("\\", "/")
        for target in self._active_task_expected_artifacts(state):
            normalized_target = target.replace("\\", "/")
            if normalized_target in normalized_command:
                return target
        return None

    def _repair_targets_from_failed_output(self, command: str, output: str, state: TaskState) -> list[str]:
        artifacts = [self._normalize_target(target) for target in self._active_task_expected_artifacts(state)]
        if not artifacts:
            return []
        combined = f"{command}\n{output}".replace("\\", "/")
        targets: list[str] = []
        for target in self._module_repair_targets_from_failure(command, output, state):
            if target not in targets:
                targets.append(target)

        for pattern in [
            r"from ['\"]([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)['\"]",
            r"No module named ['\"]([^'\"]+)['\"]",
            r"ModuleNotFoundError:.*named ['\"]([^'\"]+)['\"]",
        ]:
            for match in re.finditer(pattern, output):
                module_path = match.group(1).replace(".", "/") + ".py"
                for artifact in artifacts:
                    if artifact.endswith(module_path) and artifact not in targets:
                        targets.append(artifact)

        for artifact in artifacts:
            if artifact in combined and artifact not in targets:
                targets.append(artifact)

        basenames = {Path(artifact).name: artifact for artifact in artifacts}
        for basename, artifact in basenames.items():
            if basename in combined and artifact not in targets:
                targets.append(artifact)

        referenced = self._artifact_referenced_by_command(command, state)
        if referenced:
            normalized = self._normalize_target(referenced)
            if normalized not in targets:
                targets.append(normalized)

        if targets:
            if all(self._looks_like_test_artifact(target) for target in targets):
                for artifact in artifacts:
                    if (
                        not self._looks_like_test_artifact(artifact)
                        and Path(artifact).suffix.lower() == ".py"
                        and Path(artifact).name != "__init__.py"
                        and artifact not in targets
                    ):
                        targets.append(artifact)
            return targets
        implementation_targets = [
            artifact
            for artifact in artifacts
            if Path(artifact).suffix.lower() == ".py" and Path(artifact).name != "__init__.py"
            and not self._looks_like_test_artifact(artifact)
        ]
        test_targets = [
            artifact
            for artifact in artifacts
            if Path(artifact).suffix.lower() == ".py"
            and Path(artifact).name != "__init__.py"
            and self._looks_like_test_artifact(artifact)
        ]
        return implementation_targets + test_targets or artifacts

    def _module_repair_targets_from_failure(self, command: str, output: str, state: TaskState) -> list[str]:
        combined = f"{command}\n{output}".replace("\\", "/")
        if "ModuleNotFoundError" not in combined and "No module named" not in combined:
            return []
        workspace_root = self._workspace_root_from_command(command) or self._workspace_root_from_artifacts(state)
        if not workspace_root:
            return []
        missing_roots = {
            match.group(1).split(".", 1)[0]
            for match in re.finditer(r"No module named ['\"]([^'\"]+)['\"]", combined)
        }
        modules: list[str] = []
        for pattern in (
            r"\bfrom\s+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\s+import\b",
            r"\bpython\s+-m\s+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\b",
        ):
            for match in re.finditer(pattern, command.replace("\\", "/")):
                module = match.group(1)
                if missing_roots and module.split(".", 1)[0] not in missing_roots:
                    continue
                if module not in modules:
                    modules.append(module)
        targets: list[str] = []
        for module in modules:
            parts = module.split(".")
            module_target = f"{workspace_root}/{'/'.join(parts)}.py"
            if module_target not in targets:
                targets.append(module_target)
            package_init = f"{workspace_root}/{parts[0]}/__init__.py"
            if package_init not in targets:
                targets.append(package_init)
        return targets

    def _workspace_root_from_command(self, command: str) -> str | None:
        normalized = command.replace("\\", "/")
        for pattern in (
            r"sys\.path\.insert\(\s*0\s*,\s*['\"]([^'\"]*workspace[^'\"]*)['\"]\s*\)",
            r"PYTHONPATH['\"]?\s*:\s*['\"]([^'\"]*workspace[^'\"]*)['\"]",
            r"PYTHONPATH=([^'\"\s]*workspace[^'\"\s]*)",
            r"cwd\s*=\s*['\"]([^'\"]*workspace[^'\"]*)['\"]",
            r"\bcd(?:\s+/d)?\s+['\"]?([^'\";&|]*workspace)",
        ):
            for match in re.finditer(pattern, normalized):
                root = self._normalize_target(match.group(1))
                if root:
                    return root
        return None

    def _workspace_root_from_artifacts(self, state: TaskState) -> str | None:
        for artifact in self._active_task_expected_artifacts(state):
            normalized = self._normalize_target(artifact)
            marker = "/workspace/"
            if marker in normalized:
                return normalized.split(marker, 1)[0] + "/workspace"
        return None

    def _default_repair_write_targets(self, targets: list[str], state: TaskState | None = None) -> list[str]:
        normalized_targets = [self._normalize_target(target) for target in targets]
        implementation_targets = [
            target
            for target in normalized_targets
            if not self._looks_like_test_artifact(target)
            and Path(target).suffix.lower() == ".py"
            and Path(target).name != "__init__.py"
        ]
        if implementation_targets:
            return list(dict.fromkeys(implementation_targets))
        if state is not None:
            mutable_tests = [target for target in normalized_targets if self._is_test_repair_allowed(target, state)]
            if mutable_tests:
                return list(dict.fromkeys(mutable_tests))
        return list(dict.fromkeys(normalized_targets))

    def _repair_read_targets_from_failed_output(self, command: str, output: str, state: TaskState) -> list[str]:
        artifacts = [self._normalize_target(target) for target in self._active_task_expected_artifacts(state)]
        if not artifacts:
            return []
        combined = f"{command}\n{output}".replace("\\", "/")
        referenced = [artifact for artifact in artifacts if artifact in combined]
        referenced.sort(key=lambda target: (not self._looks_like_test_artifact(target), target))
        result: list[str] = []
        active_artifacts = {self._normalize_target(target) for target in self._active_task_expected_artifacts(state)}
        for target in referenced:
            if target not in result:
                result.append(target)
        for target in self._repair_targets_from_failed_output(command, output, state):
            if target not in active_artifacts and not (self.root / target).exists():
                continue
            if target not in result:
                result.append(target)
        if result:
            return result
        tests = [artifact for artifact in artifacts if self._looks_like_test_artifact(artifact)]
        implementation = [
            artifact
            for artifact in artifacts
            if Path(artifact).suffix.lower() == ".py"
            and Path(artifact).name != "__init__.py"
            and artifact not in tests
        ]
        return tests + implementation

    def _looks_like_test_artifact(self, target: str) -> bool:
        normalized = target.replace("\\", "/")
        return "/tests/" in normalized or Path(normalized).name.startswith("test_")

    def _active_contract(self, state: TaskState) -> dict[str, Any] | None:
        active = self._active_task_id(state)
        contracts = [
            item
            for item in state.acceptance_contracts
            if item.get("task_id") in {active, "current"} and item.get("status") == "agreed"
        ]
        if not contracts:
            return None
        return contracts[-1]

    def _contract_commands(self, state: TaskState) -> list[str]:
        latest = self._active_contract(state)
        checks = self._contract_procedure_commands(latest) if latest else []
        if not checks and latest:
            checks = list(latest.get("checks", []))
        if not latest or latest.get("source") != "task_graph":
            checks.extend(self._active_task_verification_commands(state))
        commands: list[str] = []
        for check in checks:
            text = str(check).strip()
            smoke_match = re.search(r"smoke\s+test:\s*(.+)$", text, re.IGNORECASE)
            command = smoke_match.group(1).strip() if smoke_match else text
            if command.startswith("python "):
                commands.append(command)
        return list(dict.fromkeys(commands))

    def _contract_procedure_commands(self, contract: dict[str, Any] | None) -> list[str]:
        if not contract:
            return []
        procedure = contract.get("verification_procedure")
        if not isinstance(procedure, dict):
            return []
        commands = procedure.get("commands")
        if isinstance(commands, list):
            return [str(command) for command in commands if str(command).strip()]
        command = str(procedure.get("command", "")).strip()
        return [command] if command else []

    def _normalize_verification_procedure(self, raw: object) -> dict[str, Any]:
        if isinstance(raw, dict):
            procedure: dict[str, Any] = {}
            working_directory = str(raw.get("working_directory", "")).strip()
            if working_directory:
                procedure["working_directory"] = working_directory
            commands = raw.get("commands")
            if isinstance(commands, list):
                procedure["commands"] = [str(command) for command in commands if str(command).strip()]
                return procedure
            command = str(raw.get("command", "")).strip()
            if command:
                procedure["command"] = command
            return procedure
        if isinstance(raw, list):
            return {"commands": [str(command) for command in raw if str(command).strip()]}
        return {}

    def _is_contract_command(self, command: object, state: TaskState) -> bool:
        normalized = self._normalize_command(command)
        return bool(normalized) and normalized in {
            self._normalize_command(contract_command) for contract_command in self._contract_commands(state)
        }

    def _canonical_contract_command(self, command: object, state: TaskState) -> str:
        normalized = self._normalize_command(command)
        for contract_command in self._contract_commands(state):
            if self._normalize_command(contract_command) == normalized:
                return contract_command
        return str(command).strip()

    def _normalize_command(self, command: object) -> str:
        text = str(command or "").strip()
        text = re.sub(r"\s+2>\s*&\s*1\s*$", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _dedupe_contracts(self, state: TaskState) -> None:
        deduped: dict[tuple[object, object], dict[str, Any]] = {}
        for contract in state.acceptance_contracts:
            key = (contract.get("task_id"), contract.get("summary"))
            deduped[key] = contract
        state.acceptance_contracts = list(deduped.values())

    def _active_task_expected_artifacts(self, state: TaskState) -> list[str]:
        task = self._active_task_metadata(state)
        artifacts = task.get("expected_artifacts", [])
        if not isinstance(artifacts, list):
            return []
        targets: list[str] = []
        for item in artifacts:
            if isinstance(item, str):
                targets.append(item)
            elif isinstance(item, dict) and item.get("path"):
                targets.append(str(item["path"]))
        return targets

    def _active_task_implementation_artifacts(self, state: TaskState) -> list[str]:
        if self._active_task_has_key(state, "implementation_artifacts"):
            explicit = self._active_task_artifacts_by_key(state, "implementation_artifacts")
            if explicit:
                return explicit
        return [
            target
            for target in self._active_task_expected_artifacts(state)
            if Path(target).suffix.lower() == ".py"
            and Path(target).name != "__init__.py"
            and not self._looks_like_test_artifact(target)
        ]

    def _active_task_worker_test_artifacts(self, state: TaskState) -> list[str]:
        if self._active_task_has_key(state, "worker_test_artifacts"):
            explicit = self._active_task_artifacts_by_key(state, "worker_test_artifacts")
            if explicit:
                return explicit
        return [
            target
            for target in self._active_task_expected_artifacts(state)
            if self._looks_like_test_artifact(target)
        ]

    def _active_task_acceptance_artifacts(self, state: TaskState) -> list[str]:
        if self._active_task_has_key(state, "acceptance_artifacts"):
            return self._active_task_artifacts_by_key(state, "acceptance_artifacts")
        return []

    def _active_task_frozen_acceptance_artifacts(self, state: TaskState) -> list[str]:
        if self._active_task_has_key(state, "frozen_acceptance_artifacts"):
            return self._active_task_artifacts_by_key(state, "frozen_acceptance_artifacts")
        return []

    def _active_task_has_key(self, state: TaskState, key: str) -> bool:
        return key in self._active_task_metadata(state)

    def _active_task_artifacts_by_key(self, state: TaskState, key: str) -> list[str]:
        task = self._active_task_metadata(state)
        return self._format_artifacts(task.get(key, []))

    def _active_task_test_policy(self, state: TaskState) -> dict[str, Any]:
        task = self._active_task_metadata(state)
        policy = task.get("test_policy", {})
        if not isinstance(policy, dict):
            policy = {}
        defaults = {
            "acceptance_tests_mutable_by_worker": False,
            "acceptance_test_repair_requires_verifier_approval": True,
        }
        defaults.update(policy)
        return defaults

    def _is_frozen_acceptance_artifact(self, target: str, state: TaskState) -> bool:
        normalized = self._normalize_target(target)
        return normalized in {
            self._normalize_target(item)
            for item in self._active_task_frozen_acceptance_artifacts(state)
        }

    def _test_repair_explicitly_allowed(self, target: str, state: TaskState) -> bool:
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        allowed = repair.get("allow_test_repair")
        if allowed is True:
            return True
        if isinstance(allowed, list):
            normalized = self._normalize_target(target)
            return normalized in {self._normalize_target(item) for item in allowed}
        return False

    def _is_test_repair_allowed(self, target: str, state: TaskState) -> bool:
        if not self._looks_like_test_artifact(target):
            return True
        if self._test_repair_explicitly_allowed(target, state):
            return True
        if self._is_frozen_acceptance_artifact(target, state):
            return False
        normalized = self._normalize_target(target)
        worker_tests = {self._normalize_target(item) for item in self._active_task_worker_test_artifacts(state)}
        return normalized in worker_tests

    def _active_task_verification_commands(self, state: TaskState) -> list[str]:
        task = self._active_task_metadata(state)
        commands = task.get("verification_commands", [])
        if not isinstance(commands, list):
            return []
        return [str(item) for item in commands]

    def _active_task_metadata(self, state: TaskState) -> dict[str, Any]:
        active = self._active_task_id(state)
        for node in state.nodes:
            if str(node.get("id", "")) == active:
                return node
        for task in self.orchestrator.load_tasks():
            if str(task.get("id", "")) == active:
                return task
        return {}

    def _task_graph_task_ids(self) -> set[str]:
        return {str(task.get("id")) for task in self.orchestrator.load_tasks() if str(task.get("id", "")).strip()}

    def _update_state(self, state: TaskState, action: dict[str, Any], observation: ToolResult) -> None:
        state.iterations += 1
        state.updated_at = utc_now()
        state.last_action = action
        state.last_observation = observation.to_dict()

        name = action.get("action")
        active_task_id = self._active_task_id(state)
        if name == "debug_context":
            state.last_observation["counts_as_progress"] = False
        elif name == "contract" and observation.ok:
            contract = dict(observation.data["contract"])
            contract.setdefault("status", "agreed")
            state.acceptance_contracts = [
                item
                for item in state.acceptance_contracts
                if not (
                    item.get("task_id") == contract.get("task_id")
                    and item.get("summary") == contract.get("summary")
                )
            ]
            state.acceptance_contracts.append(contract)
        elif name in {"load_skill", "save_skill", "skill", "save_memory"} and observation.ok:
            state.evidence_sources.append(
                {
                    "action": name,
                    "target": observation.data.get("name", observation.data.get("path", "")),
                    "summary": observation.summary,
                }
            )
            if name in {"save_skill", "skill"} and state.pending_skill_review:
                observation.data["skill_review_decision"] = "saved"
                state.skill_review_history.append(
                    {
                        **state.pending_skill_review,
                        "decision": "saved",
                        "skill_name": observation.data.get("name", ""),
                        "decided_at": utc_now(),
                    }
                )
                state.pending_skill_review = {}
        elif name == "dismiss_skill" and observation.ok:
            observation.data["skill_review_decision"] = "dismissed"
            state.skill_review_history.append(
                {
                    **state.pending_skill_review,
                    "decision": "dismissed",
                    "reason": observation.data.get("reason", ""),
                    "decided_at": utc_now(),
                }
            )
            state.pending_skill_review = {}
        elif name == "answer" and observation.ok and not self._is_initializer_task(state):
            if any(node.get("contract_managed") is True for node in state.nodes):
                state.evidence_sources.append(
                    {
                        "action": "answer",
                        "target": active_task_id,
                        "summary": observation.summary,
                        "task_id": active_task_id,
                        "evidence_type": "user_response",
                    }
                )
            else:
                for node in state.nodes:
                    if node["status"] != "done":
                        node["status"] = "done"
                        node["evidence"].append(observation.summary)
        elif name == "update_plan" and state.nodes and not self._is_initializer_task(state):
            state.nodes[0]["status"] = "done"
            state.nodes[0]["evidence"].append("initialized plan")
        elif (
            name in {"list_files", "read", "search", "bash", "git", "edit", "write"}
            and observation.ok
            and self._is_initializer_task(state)
        ):
            state.evidence_sources.append(
                {
                    "action": name,
                    "target": str(action.get("target", "")),
                    "summary": observation.summary,
                    "task_id": active_task_id,
                    "evidence_type": (
                        "initializer_command_passed"
                        if name == "bash" and self._is_initializer_verification_command(action, state)
                        else "initializer_artifact_observation"
                    ),
                    "ok": True,
                }
            )
            if state.nodes:
                state.nodes[0]["evidence"].append(observation.summary)
            if name in {"write", "edit"}:
                state.initializer_command_passed = False
            elif name == "bash" and self._is_initializer_verification_command(action, state):
                state.initializer_command_passed = True
        elif (
            name == "bash"
            and observation.ok
            and self._is_contract_command(
                str(observation.data.get("command") or action.get("target", "")),
                state,
            )
        ):
            command = str(observation.data.get("command") or action.get("target", ""))
            self._record_success_evidence(
                state,
                task_id=active_task_id,
                action="bash",
                target=self._canonical_contract_command(command, state),
                summary=observation.summary,
                evidence_type="acceptance_command_passed",
            )
        elif name in {"list_files", "read", "search", "bash", "git", "edit", "write"} and observation.ok and len(state.nodes) > 1:
            state.evidence_sources.append(
                {
                    "action": name,
                    "target": str(action.get("target", "")),
                    "summary": observation.summary,
                }
            )
            state.nodes[1]["status"] = "done"
            state.nodes[1]["evidence"].append(observation.summary)
        elif name in {
            "list_files",
            "read",
            "search",
            "bash",
            "git",
            "edit",
            "write",
            "protocol_error",
        } and not observation.ok:
            state.last_observation["counts_as_progress"] = False
        elif name == "verify" and observation.ok and len(state.nodes) > 2:
            state.nodes[2]["status"] = "done"
            state.nodes[2]["evidence"].append(observation.summary)
            state.last_verified_at = utc_now()
        elif name == "verify":
            if state.nodes:
                state.nodes[0]["status"] = "completed" if observation.ok else "in_progress"
                state.nodes[0]["evidence"].append(observation.summary)
            if observation.ok:
                state.last_verified_at = utc_now()
        elif name == "finish" and observation.ok:
            for node in state.nodes:
                if node["status"] != "done":
                    node["status"] = "done"
                    node["evidence"].append("finish verifier passed")
        if name == "verify" and observation.ok:
            self._record_success_evidence(
                state,
                task_id=active_task_id,
                action="verify",
                target=active_task_id,
                summary=observation.summary,
                evidence_type="verifier_passed",
            )
            self._maybe_create_pending_skill_review(state, active_task_id, observation)
        self._record_failure_pattern(state, active_task_id, action, observation)
        self._update_pending_repair(state, action, observation)

    def _record_failure_pattern(
        self,
        state: TaskState,
        task_id: str,
        action: dict[str, Any],
        observation: ToolResult,
    ) -> None:
        if observation.ok:
            return
        name = str(action.get("action", ""))
        command = ""
        output = ""
        if name == "bash":
            command = str(observation.data.get("command") or action.get("target", ""))
            output = str(observation.data.get("output", observation.summary))
        elif name == "verify":
            verification = observation.data.get("verification", {})
            results = verification.get("commands", []) if isinstance(verification, dict) else []
            failed = next((item for item in results if isinstance(item, dict) and item.get("ok") is False), None)
            if failed:
                command = str(failed.get("command", ""))
                output = str(failed.get("output", failed.get("summary", observation.summary)))
        if not output:
            return
        failure_type = self._command_failure_type(output, command=command, state=state) or "execution_error"
        fingerprint = self._error_fingerprint(output, failure_type)
        record = state.error_patterns.setdefault(
            fingerprint,
            {"count": 0, "failure_type": failure_type, "first_seen_at": utc_now(), "task_ids": []},
        )
        record["count"] = int(record.get("count", 0)) + 1
        record["last_seen_at"] = utc_now()
        record["last_summary"] = str(observation.summary)[:500]
        task_ids = record.setdefault("task_ids", [])
        if task_id not in task_ids:
            task_ids.append(task_id)
        task_patterns = state.task_error_fingerprints.setdefault(task_id, [])
        if fingerprint not in task_patterns:
            task_patterns.append(fingerprint)

    def _error_fingerprint(self, output: str, failure_type: str) -> str:
        exception_matches = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\s*:", output)
        if exception_matches:
            return f"{failure_type}:{exception_matches[-1]}"
        normalized = output.lower().replace("\\", "/")
        normalized = re.sub(r"[a-z]:/[\w./-]+", "<path>", normalized)
        normalized = re.sub(r"/(?:[\w.-]+/)+[\w.-]+", "<path>", normalized)
        normalized = re.sub(r"\b\d+\b", "<n>", normalized)
        words = re.findall(r"[a-z_]+|<path>|<n>", normalized)[:12]
        suffix = "-".join(words) if words else "unknown"
        return f"{failure_type}:{suffix[:160]}"

    def _maybe_create_pending_skill_review(
        self, state: TaskState, task_id: str, observation: ToolResult
    ) -> None:
        if state.pending_skill_review:
            return
        session_ids = state.task_session_ids.get(task_id, [])
        repeated = []
        for fingerprint in state.task_error_fingerprints.get(task_id, []):
            pattern = state.error_patterns.get(fingerprint, {})
            if int(pattern.get("count", 0)) >= SKILL_REFLECTION_ERROR_THRESHOLD:
                repeated.append(
                    {
                        "fingerprint": fingerprint,
                        "count": int(pattern.get("count", 0)),
                        "failure_type": pattern.get("failure_type", ""),
                    }
                )
        reasons: list[dict[str, Any]] = []
        if len(session_ids) > SKILL_REFLECTION_SESSION_THRESHOLD:
            reasons.append(
                {
                    "type": "high_cost_success",
                    "session_count": len(session_ids),
                    "threshold": SKILL_REFLECTION_SESSION_THRESHOLD,
                }
            )
        if repeated:
            reasons.append(
                {
                    "type": "repeated_error_resolved",
                    "patterns": repeated,
                    "threshold": SKILL_REFLECTION_ERROR_THRESHOLD,
                }
            )
        if not reasons:
            return
        report_id = str(observation.data.get("report_id", ""))
        report_path = str(observation.data.get("archived_verifier_report", ""))
        state.pending_skill_review = {
            "task_id": task_id,
            "report_id": report_id,
            "report_path": report_path,
            "trace_ref": {
                "type": "trace",
                "path": self._rel(self.trace_path),
                "step": int(getattr(self, "_current_trace_step", 0) or 0),
                "task_id": task_id,
            },
            "trigger_reasons": reasons,
            "relevant_trace": self._skill_review_trace_window(task_id, action_name="verify", observation=observation),
            "created_at": utc_now(),
        }

    def _skill_review_trace_window(
        self, task_id: str, *, action_name: str, observation: ToolResult
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for path in sorted(self.trace_dir.glob("run_*.jsonl"), key=lambda item: item.stat().st_mtime):
            for event in self._load_trace_events(path):
                if str(event.get("task_id", "")) != task_id:
                    continue
                action = event.get("action", {}) if isinstance(event.get("action"), dict) else {}
                result = event.get("observation", {}) if isinstance(event.get("observation"), dict) else {}
                events.append(
                    {
                        "action": action.get("action", ""),
                        "target": str(action.get("target", ""))[:240],
                        "ok": result.get("ok"),
                        "summary": str(result.get("summary", ""))[:300],
                    }
                )
        events.append(
            {"action": action_name, "target": "default", "ok": observation.ok, "summary": observation.summary}
        )
        return events[-8:]

    def _record_success_evidence(
        self,
        state: TaskState,
        *,
        task_id: str,
        action: str,
        target: str,
        summary: str,
        evidence_type: str,
    ) -> None:
        record = {
            "action": action,
            "target": target,
            "summary": summary,
            "task_id": task_id,
            "evidence_type": evidence_type,
            "ok": True,
        }
        duplicate = any(
            isinstance(item, dict)
            and item.get("task_id") == task_id
            and item.get("evidence_type") == evidence_type
            and item.get("target") == target
            for item in state.evidence_sources
        )
        if not duplicate:
            state.evidence_sources.append(record)
        node_summary = (
            "Acceptance command passed."
            if evidence_type == "acceptance_command_passed"
            else "Verifier passed."
        )
        for node in state.nodes:
            if str(node.get("id", "")) != task_id:
                continue
            evidence = node.setdefault("evidence", [])
            if node_summary not in evidence:
                evidence.append(node_summary)
            break

    def _update_pending_repair(self, state: TaskState, action: dict[str, Any], observation: ToolResult) -> None:
        name = action.get("action")
        if name == "verify":
            verification = observation.data.get("verification", {})
            results = verification.get("commands", []) if isinstance(verification, dict) else []
            failed = next(
                (item for item in results if isinstance(item, dict) and item.get("ok") is False),
                None,
            )
            if observation.ok:
                state.pending_repair = {}
                return
            if not failed:
                return
            command = str(failed.get("command", ""))
            output = str(failed.get("output", ""))
            failure_type = self._command_failure_type(output, command=command, state=state)
            if failure_type in {"command_syntax_error", "command_environment_error"}:
                targets: list[str] = []
                required_reads: list[str] = []
            else:
                targets = self._repair_targets_from_failed_output(command, output, state)
                required_reads = self._repair_read_targets_from_failed_output(command, output, state)
            state.pending_repair = {
                "reason": "failed_verification_command",
                "command": command,
                "summary": str(failed.get("summary", observation.summary)),
                "output": output[:4000],
                "targets": targets,
                "repair_targets": self._default_repair_write_targets(targets, state),
                "required_reads": required_reads,
                "read_targets": self._preserve_pending_repair_reads(state, required_reads),
                "repaired_targets": [],
            }
            if failure_type:
                state.pending_repair["command_failure_type"] = failure_type
            return
        if name == "bash":
            command = str(observation.data.get("command") or action.get("target", ""))
            if observation.ok and self._pending_repair_command_failure_type(state) in {
                "command_syntax_error",
                "command_environment_error",
            }:
                state.pending_repair = {}
                return
            if not self._is_contract_command(command, state):
                return
            canonical_command = self._canonical_contract_command(command, state)
            if observation.ok:
                state.pending_repair = {}
                return
            output = str(observation.data.get("output", ""))
            failure_type = self._command_failure_type(output, command=command, state=state)
            if failure_type in {"command_syntax_error", "command_environment_error"}:
                targets: list[str] = []
                required_reads: list[str] = []
            else:
                targets = self._repair_targets_from_failed_output(command, output, state)
                required_reads = self._repair_read_targets_from_failed_output(command, output, state)
            read_targets = self._preserve_pending_repair_reads(state, required_reads)
            state.pending_repair = {
                "reason": "failed_acceptance_command",
                "command": canonical_command,
                "summary": observation.summary,
                "output": output[:4000],
                "targets": targets,
                "repair_targets": self._default_repair_write_targets(targets, state),
                "required_reads": required_reads,
                "read_targets": read_targets,
                "repaired_targets": [],
            }
            if failure_type:
                state.pending_repair["command_failure_type"] = failure_type
            return
        if name == "read" and observation.ok and state.pending_repair:
            target = self._normalize_target(action.get("target", ""))
            required_reads = state.pending_repair.get("required_reads", [])
            if isinstance(required_reads, list) and target in {self._normalize_target(item) for item in required_reads}:
                read_targets = state.pending_repair.setdefault("read_targets", [])
                if isinstance(read_targets, list) and target not in {self._normalize_target(item) for item in read_targets}:
                    read_targets.append(target)
            return
        if name in {"write", "edit"} and observation.ok:
            targets = set(self._pending_repair_write_targets(state))
            target = self._normalize_target(action.get("target", ""))
            if target in targets:
                repaired_targets = state.pending_repair.setdefault("repaired_targets", [])
                if isinstance(repaired_targets, list) and target not in {
                    self._normalize_target(item) for item in repaired_targets
                }:
                    repaired_targets.append(target)

    def _preserve_pending_repair_reads(self, state: TaskState, required_reads: list[str]) -> list[str]:
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        previous_reads = repair.get("read_targets", [])
        if not isinstance(previous_reads, list):
            previous_reads = []
        required = {self._normalize_target(item) for item in required_reads}
        preserved: list[str] = []
        for target in previous_reads:
            normalized = self._normalize_target(target)
            if normalized in required and normalized not in preserved:
                preserved.append(normalized)
        return preserved

    def _recover_pending_repair_from_recent_trace(self, state: TaskState) -> None:
        if not self.trace_dir.exists():
            return
        expected = {self._normalize_target(target) for target in self._active_task_expected_artifacts(state)}
        if not expected:
            return
        commands = set(self._contract_commands(state))
        if not commands:
            return
        trace_paths = sorted(self.trace_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)[:5]
        read_after_failure: list[str] = []
        repaired_after_failure: list[str] = []
        for trace_path in trace_paths:
            for event in reversed(self._load_trace_events(trace_path)):
                action = event.get("action", {}) if isinstance(event, dict) else {}
                observation = event.get("observation", {}) if isinstance(event, dict) else {}
                if not isinstance(action, dict) or not isinstance(observation, dict):
                    continue
                name = action.get("action")
                if name == "read" and observation.get("ok") is True:
                    target = self._normalize_target(action.get("target", ""))
                    if target in expected and target not in read_after_failure:
                        read_after_failure.append(target)
                if name in {"write", "edit"} and observation.get("ok") is True:
                    target = self._normalize_target(action.get("target", ""))
                    if target in expected:
                        repaired_after_failure.append(target)
                if name != "bash":
                    continue
                data = observation.get("data", {})
                if not isinstance(data, dict):
                    continue
                command = str(data.get("command") or action.get("target", ""))
                if not self._is_contract_command(command, state):
                    continue
                if observation.get("ok") is True:
                    return
                output = str(data.get("output", ""))
                failure_type = self._command_failure_type(output, command=command, state=state)
                if failure_type in {"command_syntax_error", "command_environment_error"}:
                    targets = []
                    required_reads = []
                else:
                    targets = self._repair_targets_from_failed_output(command, output, state)
                    required_reads = self._repair_read_targets_from_failed_output(command, output, state)
                read_targets = [
                    target
                    for target in reversed(read_after_failure)
                    if target in {self._normalize_target(item) for item in required_reads}
                ]
                if targets or failure_type:
                    state.pending_repair = {
                        "reason": "failed_acceptance_command",
                        "command": self._canonical_contract_command(command, state),
                        "summary": observation.get("summary", ""),
                        "output": output[:4000],
                        "targets": targets,
                        "repair_targets": self._default_repair_write_targets(targets, state),
                        "required_reads": required_reads,
                        "read_targets": list(dict.fromkeys(read_targets)),
                        "repaired_targets": list(dict.fromkeys(repaired_after_failure)),
                        "recovered_from_trace": str(trace_path.relative_to(self.root)).replace("\\", "/"),
                    }
                    if failure_type:
                        state.pending_repair["command_failure_type"] = failure_type
                    return

    def _load_trace_events(self, trace_path: Path) -> list[dict[str, Any]]:
        try:
            text = trace_path.read_text(encoding="utf-8")
        except OSError:
            return []
        decoder = json.JSONDecoder()
        events: list[dict[str, Any]] = []
        index = 0
        while index < len(text):
            while index < len(text) and text[index].isspace():
                index += 1
            if index >= len(text):
                break
            try:
                event, index = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                break
            if isinstance(event, dict):
                events.append(event)
        return events

    def _check_answer_evidence(self, state: TaskState) -> tuple[bool, str]:
        if not self._is_answer_task(state):
            return True, "Evidence gate skipped for non-answer task."
        targets = {self._normalize_target(item.get("target", "")) for item in state.evidence_sources}
        required = ["README.md", "agent/loop.py", "agent/tools"]
        missing = [target for target in required if target not in targets]
        if missing:
            return (
                False,
                "Answer rejected: collect more repository evidence first. Missing: "
                + ", ".join(missing),
            )
        return True, "Answer evidence gate passed."

    def _is_answer_task(self, state: TaskState) -> bool:
        return any("final answer" in item.lower() for item in state.acceptance_criteria)

    def _normalize_target(self, target: object) -> str:
        text = str(target).replace("\\", "/").strip()
        while text.startswith("./"):
            text = text[2:]
        return text.rstrip("/")

    def _format_artifacts(self, items: object) -> list[str]:
        if not isinstance(items, list):
            return []
        formatted: list[str] = []
        for item in items:
            if isinstance(item, str):
                formatted.append(item)
            elif isinstance(item, dict) and item.get("path"):
                formatted.append(str(item["path"]))
        return formatted

    def _is_initializer_task(self, state: TaskState) -> bool:
        if state.task_id == "INIT":
            return True
        return any(node.get("id") == "INIT" for node in state.nodes)

    def _initializer_verification_command(self, state: TaskState) -> str:
        commands = self._active_verification_commands(state)
        return commands[0] if commands else ""

    def _is_initializer_verification_command(self, action: dict[str, Any], state: TaskState) -> bool:
        command = self._initializer_verification_command(state)
        return bool(command) and action.get("action") == "bash" and str(action.get("target", "")) == command

    def _initializer_allowed_targets(self, state: TaskState | None = None) -> set[str]:
        targets = {
            self._normalize_target(
                self._rel(self.project_spec_materialized_path or self.state_dir / "project_spec.md")
            ),
            self._normalize_target(self._rel(self.generated_tasks_path or self.state_dir / "generated_tasks.json")),
            self._normalize_target(self._rel(self.state_dir / "init.sh")),
        }
        if state and state.initializer_repair:
            targets.add(self._normalize_target(self._rel(self.initializer_candidate_path)))
        return targets

    def _validate_initializer_write_action(
        self,
        action: dict[str, Any],
        state: TaskState,
    ) -> ToolResult | None:
        target = self._normalize_target(action.get("target", ""))
        allowed_targets = self._initializer_allowed_targets(state)
        if target not in allowed_targets:
            return ToolResult(
                False,
                "INIT write rejected: Initializer may write only project_spec.md, generated_tasks.json, and init.sh in its state directory.",
                {
                    "initializer_restricted": True,
                    "target": target,
                    "allowed_targets": sorted(allowed_targets),
                    "counts_as_progress": False,
                },
            )
        if action.get("action") != "write":
            return None
        args = action.get("args", {})
        content = args.get("content") if isinstance(args, dict) else None
        if content is None or not str(content).strip():
            return ToolResult(
                False,
                f"INIT write rejected: {target} must not be empty.",
                {"initializer_validation_errors": [f"{target} must not be empty."], "counts_as_progress": False},
            )
        generated_target = self._normalize_target(
            self._rel(self.generated_tasks_path or self.state_dir / "generated_tasks.json")
        )
        init_target = self._normalize_target(self._rel(self.state_dir / "init.sh"))
        if target == init_target:
            errors = self._initializer_script_errors(content)
            if errors:
                return ToolResult(
                    False,
                    "INIT write rejected: init.sh failed deterministic validation.",
                    {
                        "initializer_validation_errors": errors,
                        "init_script_path": init_target,
                        "counts_as_progress": False,
                    },
                )
            return None
        if target != generated_target:
            return None
        try:
            data = json.loads(str(content))
        except json.JSONDecodeError as exc:
            repair = self._record_initializer_candidate(state, str(content), [str(exc)])
            return ToolResult(
                False,
                f"INIT write rejected: generated_tasks.json is invalid JSON: {exc}.",
                {
                    "initializer_validation_errors": [str(exc)],
                    **repair,
                    "counts_as_progress": False,
                },
            )
        errors = self._initializer_graph_errors(data)
        if errors:
            repair = self._record_initializer_candidate(state, str(content), errors)
            return ToolResult(
                False,
                "INIT write rejected: generated task graph failed deterministic validation.",
                {
                    "initializer_validation_errors": errors,
                    "expected_workspace_root": self._expected_initializer_workspace_root(),
                    **repair,
                    "counts_as_progress": False,
                },
            )
        return None

    def _validate_initializer_artifact_after_write(
        self,
        action: dict[str, Any],
        state: TaskState,
    ) -> ToolResult | None:
        target = self._normalize_target(action.get("target", ""))
        generated_target = self._normalize_target(
            self._rel(self.generated_tasks_path or self.state_dir / "generated_tasks.json")
        )
        candidate_target = self._normalize_target(self._rel(self.initializer_candidate_path))
        init_target = self._normalize_target(self._rel(self.state_dir / "init.sh"))
        if target == init_target:
            try:
                content = (self.state_dir / "init.sh").read_text(encoding="utf-8")
            except OSError as exc:
                return ToolResult(
                    False,
                    f"Initializer artifact validation failed: {exc}.",
                    {"initializer_validation_errors": [str(exc)], "counts_as_progress": False},
                )
            errors = self._initializer_script_errors(content)
            if not errors:
                return None
            return ToolResult(
                False,
                "Initializer init.sh validation failed after edit.",
                {"initializer_validation_errors": errors, "counts_as_progress": False},
            )
        if target not in {generated_target, candidate_target}:
            return None
        source_path = (
            self.initializer_candidate_path
            if target == candidate_target
            else self.generated_tasks_path or self.state_dir / "generated_tasks.json"
        )
        try:
            content = source_path.read_text(encoding="utf-8")
            data = json.loads(content)
        except (OSError, json.JSONDecodeError) as exc:
            repair = self._record_initializer_candidate(state, content if "content" in locals() else "", [str(exc)])
            return ToolResult(
                False,
                f"Initializer artifact validation failed: {exc}.",
                {
                    "initializer_validation_errors": [str(exc)],
                    **repair,
                    "counts_as_progress": False,
                },
            )
        errors = self._initializer_graph_errors(data)
        if not errors:
            if target == candidate_target:
                destination = self.generated_tasks_path or self.state_dir / "generated_tasks.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, destination)
                state.initializer_repair = {}
                return ToolResult(
                    True,
                    "Initializer candidate repaired and promoted to generated_tasks.json.",
                    {
                        "candidate_path": candidate_target,
                        "promoted_path": self._normalize_target(self._rel(destination)),
                    },
                )
            state.initializer_repair = {}
            return None
        repair = self._record_initializer_candidate(state, content, errors)
        return ToolResult(
            False,
            "Initializer artifact validation failed after edit.",
            {
                "initializer_validation_errors": errors,
                "expected_workspace_root": self._expected_initializer_workspace_root(),
                **repair,
                "counts_as_progress": False,
            },
        )

    def _validate_initializer_outputs(self) -> ToolResult:
        missing = self._missing_initializer_artifacts()
        if missing:
            return ToolResult(
                False,
                "Initializer verification failed: required artifacts are missing or empty.",
                {"missing_initializer_artifacts": missing},
            )
        generated_path = self.generated_tasks_path or self.state_dir / "generated_tasks.json"
        try:
            data = json.loads(generated_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ToolResult(
                False,
                f"Initializer verification failed: generated_tasks.json is invalid: {exc}.",
                {"initializer_validation_errors": [str(exc)]},
            )
        expected_workspace = self._expected_initializer_workspace_root()
        errors = self._initializer_graph_errors(data)
        init_path = self.state_dir / "init.sh"
        try:
            init_content = init_path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(str(exc))
        else:
            errors.extend(self._initializer_script_errors(init_content))
        if errors:
            return ToolResult(
                False,
                "Initializer verification failed: generated task graph or init.sh is invalid.",
                {"initializer_validation_errors": errors, "expected_workspace_root": expected_workspace},
            )
        return ToolResult(
            True,
            "Initializer artifacts passed deterministic validation.",
            {"task_count": len(data["tasks"]), "expected_workspace_root": expected_workspace},
        )

    def _record_initializer_candidate(
        self,
        state: TaskState,
        content: str,
        errors: list[str],
    ) -> dict[str, Any]:
        self.initializer_candidate_path.parent.mkdir(parents=True, exist_ok=True)
        if content:
            self.initializer_candidate_path.write_text(content, encoding="utf-8")
        signature = self._initializer_error_signature(errors)
        previous = state.initializer_repair if isinstance(state.initializer_repair, dict) else {}
        repeat_count = int(previous.get("repeat_count", 0)) + 1 if previous.get("error_signature") == signature else 1
        candidate = self._normalize_target(self._rel(self.initializer_candidate_path))
        state.initializer_repair = {
            "candidate_path": candidate,
            "validation_errors": list(errors),
            "error_signature": signature,
            "repeat_count": repeat_count,
            "updated_at": utc_now(),
        }
        return {
            "candidate_path": candidate,
            "initializer_error_signature": signature,
            "initializer_error_repeat_count": repeat_count,
        }

    def _recover_initializer_repair_from_state(self, state: TaskState) -> None:
        if state.last_action.get("action") != "write":
            return
        generated_target = self._normalize_target(
            self._rel(self.generated_tasks_path or self.state_dir / "generated_tasks.json")
        )
        if self._normalize_target(state.last_action.get("target", "")) != generated_target:
            return
        errors = state.last_observation.get("data", {}).get("initializer_validation_errors", [])
        args = state.last_action.get("args", {})
        content = args.get("content") if isinstance(args, dict) else None
        if isinstance(errors, list) and errors and isinstance(content, str) and content:
            normalized_errors = [str(error) for error in errors]
            self._record_initializer_candidate(state, content, normalized_errors)
            signature = self._initializer_error_signature(normalized_errors)
            recovered_count = self._recent_initializer_error_repeat_count(signature)
            state.initializer_repair["repeat_count"] = max(
                int(state.initializer_repair.get("repeat_count", 1)),
                recovered_count,
            )

    def _recent_initializer_error_repeat_count(self, signature: str) -> int:
        traces = sorted(self.trace_dir.glob("run_*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
        for trace_path in traces:
            events = self._load_trace_events(trace_path)
            if not events:
                continue
            count = 0
            for event in reversed(events):
                observation = event.get("observation", {})
                data = observation.get("data", {}) if isinstance(observation, dict) else {}
                errors = data.get("initializer_validation_errors", []) if isinstance(data, dict) else []
                if not isinstance(errors, list) or not errors:
                    if count:
                        break
                    continue
                event_signature = self._initializer_error_signature([str(error) for error in errors])
                if event_signature != signature:
                    if count:
                        break
                    continue
                count += 1
            if count:
                return count
        return 1

    def _initializer_error_signature(self, errors: list[str]) -> str:
        normalized = {
            re.sub(r"tasks\[\d+\]", "tasks[*]", str(error)).strip()
            for error in errors
            if str(error).strip()
        }
        return " | ".join(sorted(normalized))

    def _missing_initializer_artifacts(self) -> list[str]:
        paths = [
            self.project_spec_materialized_path or self.state_dir / "project_spec.md",
            self.generated_tasks_path or self.state_dir / "generated_tasks.json",
            self.state_dir / "init.sh",
        ]
        missing: list[str] = []
        for path in paths:
            try:
                exists_with_content = path.is_file() and bool(path.read_text(encoding="utf-8").strip())
            except OSError:
                exists_with_content = False
            if not exists_with_content:
                missing.append(self._normalize_target(self._rel(path)))
        return missing

    def _expected_initializer_workspace_root(self) -> str | None:
        spec_path = (
            self.project_spec_path
            if self.project_spec_path and self.project_spec_path.exists()
            else self.project_spec_materialized_path or self.state_dir / "project_spec.md"
        )
        try:
            spec = spec_path.read_text(encoding="utf-8")
        except OSError:
            spec = ""
        for candidate in re.findall(r"`([^`\r\n]+)`", spec):
            normalized = self._normalize_target(candidate)
            if normalized.lower().endswith("/workspace") and not any(char.isspace() for char in normalized):
                return normalized
        if self.benchmark_id:
            return f"eval/benchmarks/{self.benchmark_id}/workspace"
        return None

    def _project_requires_standard_library(self) -> bool:
        spec_path = (
            self.project_spec_path
            if self.project_spec_path and self.project_spec_path.exists()
            else self.project_spec_materialized_path or self.state_dir / "project_spec.md"
        )
        try:
            spec = spec_path.read_text(encoding="utf-8").lower()
        except OSError:
            return False
        return "use only the python standard library" in spec or "standard library only" in spec

    def _initializer_graph_errors(self, data: object) -> list[str]:
        return validate_generated_task_graph(
            data,
            self._expected_initializer_workspace_root(),
            standard_library_only=self._project_requires_standard_library(),
        )

    def _initializer_script_errors(self, content: object) -> list[str]:
        return validate_initializer_script(
            content,
            self._expected_initializer_workspace_root(),
            standard_library_only=self._project_requires_standard_library(),
        )

    def _active_verification_commands(self, state: TaskState) -> list[str]:
        active = self._active_node(state)
        if not active and self._is_initializer_task(state):
            active = next((node for node in state.nodes if node.get("id") == "INIT"), None)
        if not active:
            return []
        commands = active.get("verification_commands", [])
        return [str(command) for command in commands] if isinstance(commands, list) else []

    def _validate_contract_action(self, action: dict[str, Any], state: TaskState) -> ToolResult:
        args = action.get("args", {})
        if not isinstance(args, dict):
            return ToolResult(False, "Contract rejected: args must be an object.", {})
        task_id = str(args.get("task_id") or action.get("target") or "current")
        if task_id == "current" and self._active_node(state):
            task_id = self._active_task_id(state)
        graph_task_ids = self._task_graph_task_ids()
        if graph_task_ids and task_id not in graph_task_ids:
            return ToolResult(
                False,
                "Contract rejected: task_id must refer to an active task graph node; repair tasks are created by the harness.",
                {"task_id": task_id, "known_task_ids": sorted(graph_task_ids)},
            )
        managed = next(
            (
                item
                for item in reversed(state.acceptance_contracts)
                if item.get("task_id") == task_id and item.get("source") == "task_graph"
            ),
            None,
        )
        if managed:
            frozen_requirements = [
                str(item)
                for item in managed.get("frozen_requirements", managed.get("required_evidence", []))
                if str(item).strip()
            ]
            raw_requested_requirements = args.get(
                "frozen_requirements", args.get("required_evidence", frozen_requirements)
            )
            requested_requirements = [
                str(item)
                for item in raw_requested_requirements
                if str(item).strip()
            ] if isinstance(raw_requested_requirements, list) else []
            if requested_requirements != frozen_requirements:
                return ToolResult(
                    False,
                    "Contract rejected: frozen_requirements are semantic requirements and cannot be weakened or replaced.",
                    {
                        "task_id": task_id,
                        "frozen_requirements": frozen_requirements,
                        "requested_requirements": requested_requirements,
                    },
                )
            procedure = self._normalize_verification_procedure(
                args.get("verification_procedure", args.get("checks", []))
            )
            procedure_commands = self._contract_procedure_commands({"verification_procedure": procedure})
            if not procedure_commands:
                return ToolResult(
                    False,
                    "Contract rejected: generated-task updates may only provide a non-empty verification_procedure.",
                    {"task_id": task_id, "managed_contract": managed},
                )
            updated = dict(managed)
            updated["summary"] = str(args.get("summary") or managed.get("summary") or "").strip()
            updated["frozen_requirements"] = frozen_requirements
            updated["required_evidence"] = frozen_requirements
            updated["verification_procedure"] = procedure
            updated["checks"] = procedure_commands
            raw_mapping = args.get("criterion_command_map")
            if isinstance(raw_mapping, dict):
                updated["criterion_command_map"] = {
                    str(criterion): [str(command) for command in commands]
                    for criterion, commands in raw_mapping.items()
                    if isinstance(commands, list)
                }
            else:
                updated["criterion_command_map"] = {
                    criterion: list(procedure_commands) for criterion in frozen_requirements
                }
            updated["status"] = "agreed"
            updated["frozen"] = True
            updated["source"] = "task_graph"
            task = self._active_task_metadata(state)
            result = self.verifier.validate_contract(updated, task if task.get("id") == task_id else None)
            if not result.ok:
                return result
            state.acceptance_contracts = [
                item
                for item in state.acceptance_contracts
                if not (item.get("task_id") == task_id and item.get("source") == "task_graph")
            ]
            state.acceptance_contracts.append(updated)
            return ToolResult(
                True,
                f"Verification procedure updated for frozen requirements on {task_id}.",
                {"contract": updated},
            )
        summary = str(args.get("summary", "")).strip()
        checks = args.get("checks", [])
        if not summary:
            return ToolResult(False, "Contract rejected: summary is required.", {})
        if not isinstance(checks, list) or not checks:
            return ToolResult(False, "Contract rejected: checks must be a non-empty list.", {})
        procedure = self._normalize_verification_procedure(args.get("verification_procedure", checks))
        frozen_requirements = args.get("frozen_requirements", args.get("required_evidence", checks))
        contract = {
            "task_id": task_id,
            "summary": summary,
            "scope": args.get("scope", []),
            "frozen_requirements": frozen_requirements,
            "verification_procedure": procedure,
            "checks": checks,
            "required_evidence": frozen_requirements,
            "forbidden_shortcuts": args.get("forbidden_shortcuts", []),
            "status": "agreed",
        }
        result = self.verifier.validate_contract(contract)
        if not result.ok:
            return result
        return ToolResult(True, f"Acceptance contract agreed for {task_id}.", {"contract": contract})

    def _handle_load_skill_action(self, action: dict[str, Any], state: TaskState) -> ToolResult:
        requested = self._safe_skill_id(str(action.get("target", "")))
        if not requested:
            return ToolResult(False, "Skill load rejected: target name is required.", {})
        skill_dir = self.state_dir / "skills"
        matches: list[tuple[Path, SkillDocument]] = []
        for path in sorted(skill_dir.glob("*.md")):
            skill = parse_skill(path.read_text(encoding="utf-8"), fallback_name=path.stem)
            if self._safe_skill_id(skill.name) == requested or path.stem == requested:
                matches.append((path, skill))
        if not matches:
            return ToolResult(
                False,
                f"Skill not found: {requested}.",
                {"name": requested, "available": [item["name"] for item in skill_catalog(skill_dir)]},
            )
        if len(matches) > 1:
            return ToolResult(False, f"Skill load rejected: duplicate metadata name {requested}.", {})
        path, skill = matches[0]
        if not skill.instruction.strip():
            return ToolResult(False, f"Skill load rejected: {requested} has no instructions.", {})
        existing = next(
            (
                item
                for item in state.loaded_skills
                if item.get("name") == skill.name and item.get("content_hash") == skill.content_hash
            ),
            None,
        )
        if existing:
            return ToolResult(
                True,
                f"Skill already loaded: {skill.name}.",
                {
                    "name": skill.name,
                    "description": skill.description,
                    "content_hash": skill.content_hash,
                    "path": self._rel(path),
                    "already_loaded": True,
                },
            )
        record = {
            "name": skill.name,
            "content_hash": skill.content_hash,
            "status": "loaded",
            "loaded_at": utc_now(),
            "loaded_iteration": state.iterations,
        }
        state.loaded_skills = [item for item in state.loaded_skills if item.get("name") != skill.name]
        state.loaded_skills.append(record)
        return ToolResult(
            True,
            f"Skill loaded: {skill.name}.",
            {
                "name": skill.name,
                "description": skill.description,
                "content": skill.content,
                "content_hash": skill.content_hash,
                "path": self._rel(path),
            },
        )

    def _handle_save_memory_action(self, action: dict[str, Any], state: TaskState) -> ToolResult:
        args = action.get("args", {})
        if not isinstance(args, dict):
            return ToolResult(False, "Memory rejected: args must be an object.", {})
        memory_id = safe_memory_id(str(args.get("name") or args.get("memory_id") or action.get("target") or ""))
        description = str(args.get("description") or args.get("title") or "").strip()
        memory_type = str(args.get("type") or "").strip()
        content = normalize_memory_content(args)
        memory = MemoryDocument(memory_id, description, memory_type, content)
        errors = validate_memory(memory)
        if errors:
            return ToolResult(False, "Memory rejected: " + "; ".join(errors) + ".", {"errors": errors})

        memory_dir = self.state_dir / "memories"
        catalog = memory_catalog(memory_dir)
        normalized_description = " ".join(description.lower().split())
        duplicate = next(
            (
                item
                for item in catalog
                if safe_memory_id(item["name"]) == memory_id
                or " ".join(item["description"].lower().split()) == normalized_description
            ),
            None,
        )
        if duplicate:
            return ToolResult(False, f"Memory rejected: duplicate of existing memory {duplicate['name']}.", {"duplicate": duplicate})

        memory_path = memory_dir / f"{memory_id}.md"
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = memory_path.with_suffix(".md.tmp")
        temporary_path.write_text(render_memory(memory), encoding="utf-8")
        parsed = validate_memory(parse_memory(temporary_path.read_text(encoding="utf-8"), fallback_name=memory_id))
        if parsed:
            temporary_path.unlink(missing_ok=True)
            return ToolResult(False, "Memory rejected: rendered Memory failed validation.", {"errors": parsed})
        temporary_path.replace(memory_path)
        self.memory_path.write_text(render_memory_index(memory_dir), encoding="utf-8")
        return ToolResult(
            True,
            f"Memory saved: {memory_id}.",
            {
                "name": memory_id,
                "description": description,
                "type": memory_type,
                "path": self._rel(memory_path),
                "content_hash": memory.content_hash,
            },
        )

    def _handle_dismiss_skill_action(self, action: dict[str, Any], state: TaskState) -> ToolResult:
        review = state.pending_skill_review if isinstance(state.pending_skill_review, dict) else {}
        if not review:
            return ToolResult(False, "No Pending Skill Reflection exists.", {})
        args = action.get("args", {})
        reason = str(args.get("reason", "")).strip() if isinstance(args, dict) else ""
        if not reason:
            return ToolResult(False, "dismiss_skill requires args.reason.", {})
        return ToolResult(
            True,
            "Pending Skill Reflection dismissed.",
            {
                "decision": "dismissed",
                "reason": reason,
                "report_id": review.get("report_id"),
                "task_id": review.get("task_id"),
            },
        )

    def _handle_save_skill_action(self, action: dict[str, Any], state: TaskState) -> ToolResult:
        args = action.get("args", {})
        if not isinstance(args, dict):
            return ToolResult(False, "Skill rejected: args must be an object.", {})
        skill_id = self._safe_skill_id(str(args.get("name") or args.get("skill_id") or action.get("target") or ""))
        description = str(args.get("description") or args.get("title") or "").strip()
        instruction = normalize_instruction(args.get("instruction", args.get("body", "")))
        examples = normalize_examples(args.get("examples"))
        evidence_type = str(args.get("evidence_type", "")).strip()
        evidence_refs = args.get("evidence_refs", [])
        if not skill_id or not description or not instruction:
            return ToolResult(False, "Skill rejected: name, description, and instruction are required.", {})
        if evidence_type not in {"verified_success", "evidence_confirmed_failure"}:
            return ToolResult(
                False,
                "Skill rejected: evidence_type must be verified_success or evidence_confirmed_failure.",
                {},
            )
        if not isinstance(evidence_refs, list) or not evidence_refs:
            return ToolResult(False, "Skill rejected: evidence_refs list is required.", {})
        candidate = self._create_skill_candidate(
            skill_id=skill_id,
            description=description,
            instruction=instruction,
            examples=examples,
            evidence_type=evidence_type,
            evidence_refs=evidence_refs,
            state=state,
        )

        candidate_path = self.state_dir / "skill_candidates" / f"{candidate['candidate_id']}.json"
        self._write_json_atomic(candidate_path, candidate)
        skill_dir = self.state_dir / "skills"
        catalog = skill_catalog(skill_dir)
        normalized_description = " ".join(description.lower().split())
        duplicate = next(
            (
                item
                for item in catalog
                if self._safe_skill_id(item["name"]) == skill_id
                or " ".join(item["description"].lower().split()) == normalized_description
            ),
            None,
        )
        if duplicate:
            self._transition_skill_candidate(
                candidate, "rejected_duplicate", {"duplicate": duplicate}
            )
            self._write_json_atomic(candidate_path, candidate)
            return ToolResult(
                False,
                f"Skill rejected: duplicate of existing skill {duplicate['name']}.",
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_path": self._rel(candidate_path),
                    "candidate_status": candidate["status"],
                    "duplicate": duplicate,
                },
            )
        result = self.verifier.validate_skill_promotion(
            {
                "name": skill_id,
                "description": description,
                "instruction": instruction,
                "evidence_type": evidence_type,
                "evidence_refs": evidence_refs,
            },
            state,
        )
        if not result.ok:
            status = (
                "rejected_missing_evidence"
                if not result.data.get("checks", {}).get("all_evidence_refs_resolved", False)
                else "rejected_invalid"
            )
            self._transition_skill_candidate(candidate, status, result.data)
            self._write_json_atomic(candidate_path, candidate)
            data = dict(result.data)
            data.update(
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_path": self._rel(candidate_path),
                    "candidate_status": candidate["status"],
                }
            )
            return ToolResult(False, result.summary, data)
        self._transition_skill_candidate(candidate, "evidence_validated", result.data)
        self._transition_skill_candidate(candidate, "content_validated", result.data.get("checks", {}))
        self._transition_skill_candidate(candidate, "approved", {"decision": "create"})
        self._write_json_atomic(candidate_path, candidate)
        skill_path = skill_dir / f"{skill_id}.md"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill = SkillDocument(skill_id, description, instruction, examples)
        temporary_skill_path = skill_path.with_suffix(".md.tmp")
        temporary_skill_path.write_text(render_skill(skill), encoding="utf-8")
        parsed = parse_skill(temporary_skill_path.read_text(encoding="utf-8"), fallback_name=skill_id)
        if parsed.name != skill_id or not parsed.description or not parsed.instruction:
            temporary_skill_path.unlink(missing_ok=True)
            self._transition_skill_candidate(candidate, "rejected_invalid", {"atomic_validation": False})
            self._write_json_atomic(candidate_path, candidate)
            return ToolResult(
                False,
                "Skill promotion rejected: rendered Skill failed validation.",
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_path": self._rel(candidate_path),
                    "candidate_status": candidate["status"],
                },
            )
        temporary_skill_path.replace(skill_path)
        self._transition_skill_candidate(
            candidate,
            "promoted",
            {"path": self._rel(skill_path), "content_hash": skill.content_hash},
        )
        self._write_json_atomic(candidate_path, candidate)
        return ToolResult(
            True,
            f"Skill saved: {skill_id}.",
            {
                "name": skill_id,
                "path": self._rel(skill_path),
                "content_hash": skill.content_hash,
                "evidence_type": evidence_type,
                "evidence_refs": evidence_refs,
                "candidate_id": candidate["candidate_id"],
                "candidate_path": self._rel(candidate_path),
                "candidate_status": candidate["status"],
            },
        )

    def _create_skill_candidate(
        self,
        *,
        skill_id: str,
        description: str,
        instruction: str,
        examples: str,
        evidence_type: str,
        evidence_refs: list[Any],
        state: TaskState,
    ) -> dict[str, Any]:
        candidate_id = self._next_skill_candidate_id()
        now = utc_now()
        return {
            "candidate_id": candidate_id,
            "status": "proposed",
            "proposed_skill": {
                "name": skill_id,
                "description": description,
                "instruction": instruction,
                "examples": examples,
            },
            "source": {"task_id": self._active_task_id(state), "evidence_type": evidence_type},
            "evidence_refs": evidence_refs,
            "validation": {},
            "status_history": [{"status": "proposed", "time": now}],
            "created_at": now,
            "updated_at": now,
        }

    def _next_skill_candidate_id(self) -> str:
        candidate_dir = self.state_dir / "skill_candidates"
        highest = 0
        if candidate_dir.exists():
            for path in candidate_dir.glob("SC-*.json"):
                match = re.fullmatch(r"SC-(\d+)", path.stem)
                if match:
                    highest = max(highest, int(match.group(1)))
        return f"SC-{highest + 1:04d}"

    def _transition_skill_candidate(
        self, candidate: dict[str, Any], status: str, validation: dict[str, Any]
    ) -> None:
        candidate["status"] = status
        candidate["updated_at"] = utc_now()
        candidate["validation"] = validation
        candidate.setdefault("status_history", []).append({"status": status, "time": candidate["updated_at"]})

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(path)

    def _safe_skill_id(self, raw: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in raw.strip().lower())
        return cleaned.strip("-_")

    def _archive_verifier_success(self, task_id: str, result: ToolResult) -> dict[str, Any]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        safe_task = self._safe_skill_id(task_id) or "task"
        report_id = f"VR-{safe_task}-{stamp}"
        path = self.state_dir / "verifier_reports" / f"{report_id}.json"
        payload = {
            "report_id": report_id,
            "task_id": task_id,
            "ok": True,
            "summary": result.summary,
            "checks": result.data.get("checks", {}),
            "verification": result.data.get("verification", {}),
            "trace_ref": {
                "path": self._rel(self.trace_path),
                "step": int(getattr(self, "_current_trace_step", 0) or 0),
                "task_id": task_id,
            },
            "created_at": utc_now(),
        }
        self._write_json_atomic(path, payload)
        return {"report_id": report_id, "archived_verifier_report": self._rel(path)}

    def _has_contract_for_active_task(self, state: TaskState) -> bool:
        active = self._active_task_id(state)
        return any(
            item.get("task_id") in {active, "current"} and item.get("status") == "agreed"
            for item in state.acceptance_contracts
        )

    def _active_task_id(self, state: TaskState) -> str:
        if state.task_id == "INIT":
            return "INIT"
        for node in state.nodes:
            if node.get("status") in {"in_progress", "pending"}:
                return str(node.get("id", "current"))
        return "current"

    def _write_state(self, state: TaskState) -> None:
        self.state_path.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _relevant_memory_context(self, state: TaskState) -> str:
        query = self._memory_retrieval_query(state)
        try:
            retrieved = self.memory_retriever.retrieve(query)
        except Exception as exc:
            LOGGER.warning("Memory retrieval failed: %s", exc)
            self._last_memory_selection = {
                "source": "error",
                "selected": [],
                "error_type": type(exc).__name__,
            }
            return render_relevant_memories([], source="error")
        self._last_memory_selection = {
            "source": retrieved.source,
            "selected": retrieved.selected_filenames,
        }
        return render_relevant_memories(retrieved.memories, source=retrieved.source)

    def _memory_retrieval_query(self, state: TaskState) -> str:
        messages = state.conversation_messages if isinstance(state.conversation_messages, list) else []
        latest_user = ""
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).strip().lower() == "user":
                latest_user = str(message.get("content", "")).strip()
                break
        parts = [
            f"active_task_id: {self._active_task_id(state)}",
            f"user_goal: {state.user_goal}",
            f"interaction_mode: {state.interaction_mode or 'non-interactive'}",
        ]
        if latest_user:
            parts.append(f"latest_user_message: {latest_user}")
        if state.last_action:
            parts.append(f"last_action: {state.last_action.get('action')} {state.last_action.get('target', '')}")
        if isinstance(state.last_observation, dict) and state.last_observation.get("summary"):
            parts.append(f"last_observation: {state.last_observation.get('summary')}")
        return "\n".join(parts)

    def _append_trace(
        self,
        step: int,
        action: dict[str, Any],
        observation: ToolResult,
        state: TaskState,
        context_snapshot: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "step": step,
            "time": utc_now(),
            "context_ref": context_snapshot or {},
            "action": action,
            "observation": observation.to_dict(),
            "tool_return": observation.to_dict(),
            "state_summary": state.summary(),
            "task_id": state.task_id,
            "session_used_tokens": state.session_used_tokens,
            "handoff_ready": state.handoff_ready,
            "orchestrator_decision": state.orchestrator_decision,
            "nodes": state.nodes,
            "skill_catalog_size": len(skill_catalog(self.state_dir / "skills")),
            "memory_catalog_size": len(memory_catalog(self.state_dir / "memories")),
            "memory_selection": self._last_memory_selection,
            "loaded_skill_names": [
                str(item.get("name")) for item in state.loaded_skills if isinstance(item, dict) and item.get("name")
            ],
            "pending_skill_review": state.pending_skill_review,
        }
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, indent=2) + "\n")

    def _write_handoff(self, state: TaskState) -> None:
        active_node = self._active_node(state)
        completed = [node for node in state.nodes if node.get("status") == "done"]
        pending = [node for node in state.nodes if node.get("status") != "done"]
        contracts = self._active_handoff_contracts(state)
        evidence = state.evidence_sources[-20:]
        payload = self._write_handoff_payload(state, active_node, completed, pending, contracts, evidence)
        lines = [
            "# Worker Session Handoff",
            "",
            "## Critical Context",
            "### User Goal",
            state.user_goal,
            "",
            "### Session Budget",
            f"- budget_tokens: {state.session_budget_tokens}",
            f"- threshold_ratio: {state.handoff_threshold}",
            f"- threshold_tokens: {int(state.session_budget_tokens * state.handoff_threshold)}",
            f"- estimated_turn_tokens: {state.session_used_tokens}",
            f"- handoff_ready: {state.handoff_ready}",
            "",
            "### Active Task",
            self._format_node(active_node) if active_node else "No active task.",
            "",
            "### Last Step Summary",
            f"- last_action: {state.last_action.get('action')} {state.last_action.get('target', '')}",
            f"- last_observation_ok: {state.last_observation.get('ok')}",
            f"- last_observation_summary: {state.last_observation.get('summary')}",
            "",
            "### Pending Repair",
            *self._format_pending_repair(state),
            "",
            "### Initializer Repair",
            *self._format_initializer_repair(state),
            "",
            "## Working Context",
            "### Orchestrator Decision",
            f"- selected_task_id: {state.orchestrator_decision.get('selected_task_id')}",
            f"- reason: {state.orchestrator_decision.get('reason')}",
            f"- ready_task_ids: {state.orchestrator_decision.get('ready_task_ids')}",
            "",
            "### Active Acceptance Contract",
            *([self._format_contract(contract) for contract in contracts] or ["- none"]),
            "",
            "### Active Verification Commands",
            *self._active_verification_commands_for_handoff(state),
            "",
            "### Evidence Sources",
            *[f"- {item.get('action')}: {item.get('target')} -- {item.get('summary')}" for item in evidence],
            "",
            "### Task Progress",
            "#### Completed Tasks",
            *[self._format_node(node) for node in completed],
            "",
            "#### Pending Or Blocked Tasks",
            *[self._format_node(node) for node in pending],
            "",
            "### Pending Skill Reflection",
            json.dumps(state.pending_skill_review, ensure_ascii=False) if state.pending_skill_review else "none",
            "",
            "### Verification Status",
            f"- last_verified_at: {state.last_verified_at}",
            "- deterministic verifier: run `python -m unittest discover -s tests` and `python -m compileall agent eval tests`.",
            "",
            "## Reference Context",
            "### Handoff Data References",
            f"- structured_payload: {payload['payload_path']}",
            f"- current_state: {self._rel(self.state_path)}",
            f"- task_graph_runtime: {self._rel(self.runtime_tasks_path) if self.runtime_tasks_path else 'not used'}",
            f"- task_graph_generated: {self._rel(self.generated_tasks_path) if self.generated_tasks_path else 'not used'}",
            f"- rejected_initializer_candidate: {self._rel(self.initializer_candidate_path)}",
            "- task_graph_source: the original `--tasks-json` file is benchmark input and should remain read-only",
            f"- immutable_verifier_reports: {self._rel(self.state_dir / 'verifier_reports')}/",
            f"- memory_index: {self._rel(self.memory_path)}",
            f"- memories: {self._rel(self.state_dir / 'memories')}/",
            f"- traces: {self._rel(self.trace_dir)}/",
            "",
            "### Current State Summary",
            state.summary(),
            "",
            "## Resume Guidance",
            "### Known Risks And Failed Attempts",
            "- Review trace for failed observations and protocol errors before resuming.",
            "- Do not repeat failed actions unchanged.",
            "- Do not start new large edits until the next worker session has rebuilt context from this handoff.",
            "",
            "### Resume Instructions",
            "1. Read this handoff first.",
            f"2. Load `{self._rel(self.state_path)}`, the active task graph, and relevant source files.",
            "3. Continue the active task only after checking the acceptance contract.",
            "4. If no contract exists for a coding task, create one before writing code.",
            "5. Prefer verification or small repair actions before new feature work.",
            f"6. Inspect `{payload['payload_path']}` only when the summary above is insufficient.",
            "",
            "### Suggested Next Action",
            self._suggest_next_action(state),
            "",
        ]
        self.handoff_path.write_text("\n".join(lines), encoding="utf-8")

    def _write_handoff_payload(
        self,
        state: TaskState,
        active_node: dict[str, Any] | None,
        completed: list[dict[str, Any]],
        pending: list[dict[str, Any]],
        contracts: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> dict[str, str]:
        payload_path = str(self.handoff_payload_path.relative_to(self.root)).replace("\\", "/")
        payload = {
            "schema": "long-agent.handoff-payload.v1",
            "written_at": utc_now(),
            "payload_path": payload_path,
            "user_goal": state.user_goal,
            "session_budget": {
                "budget_tokens": state.session_budget_tokens,
                "threshold_ratio": state.handoff_threshold,
                "threshold_tokens": int(state.session_budget_tokens * state.handoff_threshold),
                "estimated_used_tokens": state.session_used_tokens,
                "estimated_turn_tokens": state.session_used_tokens,
                "handoff_ready": state.handoff_ready,
            },
            "orchestrator_decision": state.orchestrator_decision,
            "active_task": active_node,
            "completed_tasks": completed,
            "pending_or_blocked_tasks": pending,
            "acceptance_contracts": contracts,
            "evidence_sources": evidence,
            "pending_repair": state.pending_repair,
            "initializer_repair": state.initializer_repair,
            "pending_skill_review": state.pending_skill_review,
            "skill_review_history": state.skill_review_history[-20:],
            "task_session_ids": state.task_session_ids,
            "error_patterns": state.error_patterns,
            "last_action": state.last_action,
            "last_observation": state.last_observation,
            "last_verified_at": state.last_verified_at,
            "state_summary": state.summary(),
            "resume_hint": self._suggest_next_action(state),
        }
        self.handoff_payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"payload_path": payload_path}

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root)).replace("\\", "/")
        except ValueError:
            return str(path).replace("\\", "/")

    def _read_state_file(self, name: str, max_chars: int = 3000) -> str:
        path = self.state_dir / name
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")[:max_chars]

    def _record_budget_usage(
        self,
        state: TaskState,
        context: str,
        action: dict[str, Any],
        observation: ToolResult,
    ) -> None:
        payload = context + json.dumps(action, ensure_ascii=False) + json.dumps(observation.to_dict(), ensure_ascii=False)
        state.session_used_tokens = max(1, len(payload) // 4)
        threshold_tokens = int(state.session_budget_tokens * state.handoff_threshold)
        if state.session_used_tokens >= threshold_tokens:
            state.handoff_ready = True

    def _active_node(self, state: TaskState) -> dict[str, Any] | None:
        for node in state.nodes:
            if node.get("status") in {"in_progress", "pending"}:
                return node
        return None

    def _active_handoff_contracts(self, state: TaskState) -> list[dict[str, Any]]:
        active = self._active_task_id(state)
        active_ids = {active, "current"}
        if active == "INIT":
            active_ids = {"INIT"}
        for contract in reversed(state.acceptance_contracts):
            if contract.get("task_id") in active_ids:
                return [contract]
        return []

    def _active_verification_commands_for_handoff(self, state: TaskState) -> list[str]:
        commands: list[str] = []
        for node in state.nodes:
            if node.get("status") not in {"in_progress", "pending"}:
                continue
            raw_commands = node.get("verification_commands", [])
            if isinstance(raw_commands, list):
                commands.extend(str(command) for command in raw_commands if str(command).strip())
        return [f"- {command}" for command in commands] or ["- none"]

    def _format_node(self, node: dict[str, Any]) -> str:
        evidence = node.get("evidence", [])
        evidence_text = "; ".join(str(item) for item in evidence[-3:]) if evidence else "no evidence yet"
        return f"- {node.get('id')}: [{node.get('status')}] {node.get('title')} | evidence: {evidence_text}"

    def _format_contract(self, contract: dict[str, Any]) -> str:
        requirements = contract.get("frozen_requirements", contract.get("required_evidence", []))
        procedure_commands = self._contract_procedure_commands(contract) or list(contract.get("checks", []))
        requirement_text = "; ".join(str(item) for item in requirements)
        procedure_text = "; ".join(str(item) for item in procedure_commands)
        return (
            f"- {contract.get('task_id')}: {contract.get('summary')} | "
            f"frozen_requirements: {requirement_text} | verification_procedure: {procedure_text}"
        )

    def _format_pending_repair(self, state: TaskState) -> list[str]:
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        targets = self._pending_repair_targets(state)
        if not repair:
            return ["- none"]
        failure_type = str(repair.get("command_failure_type", ""))
        suggested = self._suggest_corrected_command(str(repair.get("command", "")), failure_type, state)
        lines = [
            f"- reason: {repair.get('reason', 'failed_acceptance_command')}",
            f"- command: {repair.get('command', '')}",
            f"- targets: {', '.join(targets)}",
            f"- repair_targets: {', '.join(str(item) for item in repair.get('repair_targets', []))}",
            f"- required_reads: {', '.join(str(item) for item in repair.get('required_reads', []))}",
            f"- read_targets: {', '.join(str(item) for item in repair.get('read_targets', []))}",
            f"- repaired_targets: {', '.join(str(item) for item in repair.get('repaired_targets', []))}",
            f"- command_failure_type: {failure_type}",
            f"- summary: {repair.get('summary', '')}",
        ]
        if suggested:
            lines.append(f"- suggested_command: {suggested}")
        return lines

    def _format_initializer_repair(self, state: TaskState) -> list[str]:
        repair = state.initializer_repair if isinstance(state.initializer_repair, dict) else {}
        if not repair:
            return ["- none"]
        errors = repair.get("validation_errors", [])
        return [
            f"- candidate_path: {repair.get('candidate_path', '')}",
            f"- repeat_count: {repair.get('repeat_count', 0)}",
            f"- error_signature: {repair.get('error_signature', '')}",
            *[f"- validation_error: {error}" for error in errors if isinstance(errors, list)],
        ]

    def _suggest_next_action(self, state: TaskState) -> str:
        active = self._active_node(state)
        if not active:
            return "Run verifier and finish if all acceptance criteria pass."
        if self._is_initializer_task(state):
            if state.initializer_repair:
                candidate = state.initializer_repair.get("candidate_path", "")
                return f"Repair the saved INIT candidate at {candidate}; do not regenerate the full graph."
            missing = self._missing_initializer_artifacts()
            if missing:
                return f"Resume INIT by writing {missing[0]}; INIT does not require an acceptance contract."
            return "Submit INIT to the verifier; it executes the deterministic command and only PASS schedules the first Worker task."
        repair_targets = self._pending_repair_write_targets(state)
        if repair_targets:
            return (
                f"Repair failed acceptance command for {active.get('id')}: "
                f"use write or edit on {repair_targets[0]} before more read/list/test actions."
            )
        if state.last_observation.get("data", {}).get("missing_path") and self._has_contract_for_active_task(state):
            return (
                f"Use write to create the first required file for {active.get('id')}; "
                "do not repeat list_files on the missing directory."
            )
        if not self._has_contract_for_active_task(state):
            if active.get("contract_managed"):
                return (
                    f"The frozen task-graph contract for {active.get('id')} is missing or rejected. "
                    "Do not create a manual contract; repair the generated task graph from INIT and let activation freeze it."
                )
            return f"Create an acceptance contract for {active.get('id')} before writing code."
        return f"Resume {active.get('id')} with a small evidence-backed action, then verify."

    def _ensure_state_files(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.debug_context_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "skills").mkdir(exist_ok=True)
        (self.state_dir / "memories").mkdir(exist_ok=True)
        (self.state_dir / "skill_candidates").mkdir(exist_ok=True)
        (self.state_dir / "verifier_reports").mkdir(exist_ok=True)
        if not self.memory_path.exists():
            self.memory_path.write_text(render_memory_index(self.state_dir / "memories"), encoding="utf-8")
        for legacy_memory in (self.state_dir / "hard_memory.md", self.state_dir / "soft_memory.md"):
            legacy_memory.unlink(missing_ok=True)
        if not (self.state_dir / "skills" / "coding.md").exists():
            (self.state_dir / "skills" / "coding.md").write_text(
                "---\n"
                "name: coding\n"
                "description: Apply the standard evidence-driven workflow for coding tasks.\n"
                "---\n\n"
                "# Instructions\n\n"
                "1. Inspect files before editing.\n"
                "2. Prefer small verifiable steps.\n"
                "3. Run syntax checks before finishing.\n",
                encoding="utf-8",
            )

    def _prepare_runtime_task_graph(self) -> None:
        self._materialize_project_spec()
        if not self.source_tasks_path or not self.runtime_tasks_path:
            return
        if not self.source_tasks_path.exists():
            return
        if self.resume and self.runtime_tasks_path.exists():
            return
        self.runtime_tasks_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.source_tasks_path, self.runtime_tasks_path)

    def _materialize_project_spec(self) -> None:
        if not self.project_spec_path:
            return
        if not self.project_spec_path.exists():
            return
        target = self.project_spec_materialized_path or self.root / "project_spec.md"
        try:
            same_file = target.exists() and target.resolve() == self.project_spec_path.resolve()
        except OSError:
            same_file = False
        if same_file:
            return
        if self.resume and target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.project_spec_path.read_text(encoding="utf-8"), encoding="utf-8")

    def _initializer_needed(self) -> bool:
        if not self.project_spec_path or not self.generated_tasks_path:
            return False
        init_path = self.state_dir / "init.sh"
        if not self.generated_tasks_path.exists() or not init_path.exists():
            return True
        try:
            data = json.loads(self.generated_tasks_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        try:
            init_content = init_path.read_text(encoding="utf-8")
        except OSError:
            return True
        return bool(self._initializer_graph_errors(data) or self._initializer_script_errors(init_content))

    @staticmethod
    def _trace_name() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        return f"run_{stamp}.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
