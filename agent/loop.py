from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.context import ContextBuilder
from agent.llm import create_decision_maker
from agent.orchestrator import Orchestrator
from agent.planner import (
    TaskState,
    create_initial_state,
    create_initializer_state,
    validate_generated_task_graph,
    validate_initializer_script,
)
from agent.termination import ProjectTerminator
from agent.tools import BashTool, EditTool, GitTool, ListFilesTool, ReadTool, SearchTool, ToolResult, WriteTool
from agent.verifier import Verifier


@dataclass
class RunResult:
    completed: bool
    steps: int
    trace_path: Path
    state_path: Path
    message: str

    def to_human_summary(self) -> str:
        status = "completed" if self.completed else "stopped"
        return (
            f"Agent {status} after {self.steps} step(s).\n"
            f"State: {self.state_path}\n"
            f"Trace: {self.trace_path}\n"
            f"{self.message}"
        )


class AgentLoop:
    def __init__(
        self,
        root: Path,
        task: str,
        max_steps: int,
        provider: str = "offline",
        resume: bool = False,
        tasks_path: Path | None = None,
        project_spec_path: Path | None = None,
        benchmark_id: str | None = None,
    ) -> None:
        self.root = root
        self.task = task
        self.max_steps = max_steps
        self.provider = provider
        self.resume = resume
        self.project_spec_path = project_spec_path
        self.source_tasks_path = tasks_path
        self.benchmark_id = self._safe_benchmark_id(benchmark_id) if benchmark_id else None
        self.state_dir = self._benchmark_state_dir(self.benchmark_id) if self.benchmark_id else root / "state"
        self.trace_dir = self.state_dir / "traces"
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
        self.context_builder = ContextBuilder(root, state_dir=self.state_dir)
        self.orchestrator = Orchestrator(root, tasks_path=self.tasks_path, state_dir=self.state_dir)
        self.terminator = ProjectTerminator(root, tasks_path=self.tasks_path, benchmark_id=self.benchmark_id)
        self.decision_maker = create_decision_maker(provider)
        self.verifier = Verifier(root, state_dir=self.state_dir)
        self.tools = {
            "bash": BashTool(root),
            "edit": EditTool(root),
            "git": GitTool(root, allow_write=self.benchmark_id is None),
            "list_files": ListFilesTool(root),
            "read": ReadTool(root),
            "search": SearchTool(root),
            "write": WriteTool(root),
        }

    def _benchmark_state_dir(self, benchmark_id: str) -> Path:
        return self.root / "state" / "benchmarks" / benchmark_id

    def _safe_benchmark_id(self, raw: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in raw.strip())
        cleaned = cleaned.strip("-_")
        if not cleaned:
            raise ValueError("benchmark_id must contain a letter, number, underscore, or dash")
        return cleaned

    def run(self) -> RunResult:
        self._ensure_state_files()
        self._prepare_runtime_task_graph()
        state = self._load_or_create_state()
        steps = 0
        completed = False
        message = "Reached max steps before completion."

        for step in range(1, self.max_steps + 1):
            steps = step
            context = self.context_builder.build(state)
            model_action: dict[str, Any] | None = None
            try:
                model_action = self.decision_maker.next_action(context, state)
                action = self._guard_action(model_action, state)
                observation = self._execute_action(action, state)
            except Exception as exc:
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
            self._append_trace(step, action, observation, state, model_action=model_action)
            self._write_state(state)

            if (
                action["action"] in {"answer", "finish"}
                and observation.ok
                and not self._is_initializer_task(state)
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

        if not completed:
            self._write_handoff(state)

        return RunResult(
            completed=completed,
            steps=steps,
            trace_path=self.trace_path,
            state_path=self.state_path,
            message=message,
        )

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
        self._dedupe_contracts(state)
        if self._is_initializer_task(state) and not state.initializer_repair:
            self._recover_initializer_repair_from_state(state)
        if not self._is_initializer_task(state):
            self._apply_orchestrator_selection(state)
        if self.resume and not state.pending_repair:
            self._recover_pending_repair_from_recent_trace(state)
        return state

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
                "depends_on": task.get("depends_on", []),
                "priority": task.get("priority", 1000),
                "expected_artifacts": task.get("expected_artifacts", []),
                "implementation_artifacts": task.get("implementation_artifacts", []),
                "worker_test_artifacts": task.get("worker_test_artifacts", []),
                "acceptance_artifacts": task.get("acceptance_artifacts", []),
                "frozen_acceptance_artifacts": task.get("frozen_acceptance_artifacts", []),
                "hidden_acceptance": task.get("hidden_acceptance", []),
                "test_policy": task.get("test_policy", {}),
                "verification_commands": task.get("verification_commands", []),
            }
        ]
        if str(task.get("status", "pending")) == "pending":
            updated = self.orchestrator.transition_task(task_id, "in_progress", "scheduled by orchestrator")
            if updated:
                state.nodes[0]["status"] = "in_progress"

    def _execute_action(self, action: dict[str, Any], state: TaskState) -> ToolResult:
        name = action.get("action")
        if name == "required_write":
            target = str(action.get("target", ""))
            return ToolResult(
                False,
                f"Action rejected: expected code artifact is incomplete. Write an implementation to {target} with mode='overwrite' before further inspection or tests.",
                {"required_action": "write", "target": target, "mode": "overwrite", "counts_as_progress": False},
            )
        if name == "required_repair":
            args = action.get("args", {})
            targets = args.get("targets", []) if isinstance(args, dict) else []
            primary = str(action.get("target", ""))
            return ToolResult(
                False,
                (
                    "Action rejected: the last acceptance command failed. "
                    f"Repair {primary} with write or edit before further inspection or tests."
                ),
                {
                    "required_action": "write_or_edit",
                    "target": primary,
                    "targets": targets,
                    "counts_as_progress": False,
                },
            )
        if name == "required_command_repair":
            args = action.get("args", {}) if isinstance(action.get("args"), dict) else {}
            return ToolResult(
                False,
                "Action rejected: the acceptance command has invalid syntax. Run a corrected equivalent bash command before verification or implementation repair.",
                {
                    "required_action": "bash_corrected_acceptance_command",
                    "bad_command": args.get("bad_command", action.get("target", "")),
                    "failure_summary": args.get("failure_summary", ""),
                    "counts_as_progress": False,
                },
            )
        if name == "required_initializer_repair":
            args = action.get("args", {}) if isinstance(action.get("args"), dict) else {}
            candidate = str(action.get("target", ""))
            errors = args.get("errors", [])
            return ToolResult(
                False,
                "INIT repair required: edit or overwrite the saved candidate instead of regenerating the full task graph.",
                {
                    "required_action": "edit_or_write_candidate",
                    "candidate_path": candidate,
                    "initializer_validation_errors": errors if isinstance(errors, list) else [],
                    "repeat_count": args.get("repeat_count", 2),
                    "counts_as_progress": False,
                },
            )
        if name == "blocked_repeat":
            return ToolResult(
                False,
                "Action rejected: do not repeat a failed or no-progress action unchanged; use the handoff suggested next action or explain a blocker.",
                {
                    "blocked_repeat": True,
                    "required_next_action": self._suggest_next_action(state),
                    "counts_as_progress": False,
                },
            )
        if name == "answer":
            if self._is_initializer_task(state):
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
            return self._validate_contract_action(action)
        if name == "skill":
            if self._is_initializer_task(state):
                return ToolResult(False, "Skill promotion is disabled during INIT.", {"initializer_restricted": True})
            return self._handle_skill_action(action, state)
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
                if not state.initializer_command_passed:
                    result = ToolResult(
                        False,
                        "Initializer verification rejected: run the deterministic INIT verification command successfully first.",
                        {
                            "task_id": task_id,
                            "initializer_command_passed": False,
                            "required_command": self._initializer_verification_command(state),
                        },
                    )
                    self.verifier.record_result(result)
                    return result
                initializer_result = self._validate_initializer_outputs()
                if not initializer_result.ok:
                    initializer_result.data["task_id"] = task_id
                    self.verifier.record_result(initializer_result)
                    return initializer_result
            self.orchestrator.mark_awaiting_verification(task_id, "worker submitted candidate for verification")
            result = self.verifier.run(action.get("target", "default"), state)
            result.data["task_id"] = task_id
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

    def _guard_action(self, action: dict[str, Any], state: TaskState) -> dict[str, Any]:
        initializer_action = self._guard_initializer_progress(action, state)
        if initializer_action:
            return initializer_action
        pending_repair_action = self._guard_pending_repair(action, state)
        if pending_repair_action:
            return pending_repair_action
        frozen_test_action = self._guard_frozen_acceptance_test_edit(action, state)
        if frozen_test_action:
            return frozen_test_action
        repair_action = self._repair_action_after_failed_contract(state)
        if repair_action and action.get("action") != "verify":
            return repair_action
        required_write_target = self._required_write_target(state)
        if required_write_target and not self._is_write_for_target(action, required_write_target):
            return {
                "thought_summary": (
                    "Guard override: an expected code artifact is empty/incomplete, "
                    "so further inspection or tests are blocked until the worker writes it."
                ),
                "action": "required_write",
                "target": required_write_target,
                "args": {"mode": "overwrite"},
                "expected_observation": "Harness rejects no-progress actions until the artifact is implemented.",
                "risk": "low",
                "guard_override": "block_until_required_write",
            }
        if action.get("action") != "verify" and self._last_contract_check_passed(state):
            return {
                "thought_summary": (
                    "Guard override: an acceptance contract check already passed, "
                    "so submit the current task to the verifier instead of collecting more context."
                ),
                "action": "verify",
                "target": "default",
                "args": {},
                "expected_observation": "Verifier checks the candidate and updates task status.",
                "risk": "low",
                "guard_override": "smoke_passed_to_verify",
            }
        resume_action = self._guard_resume_suggested_action(action, state)
        if resume_action:
            return resume_action
        if action.get("action") == "write":
            return self._guard_duplicate_create(action, state)
        repeated_failed_action = self._guard_repeated_failed_action(action, state)
        if repeated_failed_action and action.get("action") != "list_files":
            return repeated_failed_action
        if action.get("action") != "list_files":
            return action
        if self._is_initializer_task(state):
            return self._guard_repeated_list_files(action, state) or repeated_failed_action or action
        if not self._has_contract_for_active_task(state):
            if self._should_create_contract_after_inspection(action, state):
                return self._synthesize_contract_action(state)
            return self._guard_repeated_list_files(action, state) or repeated_failed_action or action
        target = self._next_contract_file_target(state)
        if not target and self._should_rewrite_list_to_write(action, state):
            target = self._next_implementation_file_target(state)
        if not target and self._all_contract_file_targets_exist(state):
            smoke_command = self._contract_smoke_command(state)
            if smoke_command:
                return {
                    "thought_summary": (
                        "Guard override: required contract files already exist, "
                        "so run the contract smoke test instead of listing again."
                    ),
                    "action": "bash",
                    "target": smoke_command,
                    "args": {},
                    "expected_observation": "Smoke test exits with code 0.",
                    "risk": "low",
                    "guard_override": "contract_files_exist_to_smoke",
                }
        if not target and self._implementation_files_exist(state) and self._should_run_contract_command(action, state):
            test_command = self._contract_test_command(state)
            if test_command:
                return {
                    "thought_summary": (
                        "Guard override: implementation files exist, so run the acceptance test "
                        "instead of listing directories again."
                    ),
                    "action": "bash",
                    "target": test_command,
                    "args": {},
                    "expected_observation": "Acceptance test exits with code 0.",
                    "risk": "low",
                    "guard_override": "implementation_files_exist_to_test",
                }
        if not target:
            return self._guard_repeated_list_files(action, state) or repeated_failed_action or action
        if not self._should_rewrite_list_to_write(action, state):
            return self._guard_repeated_list_files(action, state) or repeated_failed_action or action
        content = self._initial_content_for_target(target, state)
        if content is None:
            existing = (self.root / target).resolve()
            if existing.exists():
                if self._last_read_empty_artifact(state, target):
                    return action
                return {
                    "thought_summary": (
                        "Guard override: a required code artifact exists but has no safe generic content, "
                        "so read it and let the worker implement it instead of writing an empty placeholder."
                    ),
                    "action": "read",
                    "target": target,
                    "args": {},
                    "expected_observation": f"Read required code artifact {target}.",
                    "risk": "low",
                    "guard_override": "incomplete_code_artifact_to_read",
                }
            return action
        guarded = dict(action)
        guarded["thought_summary"] = (
            "Guard override: an acceptance contract exists and required files are still missing, "
            "so create the next required file instead of repeating list_files."
        )
        guarded["action"] = "write"
        guarded["target"] = target
        guarded["args"] = {"mode": "create", "content": content}
        guarded["expected_observation"] = f"Create required file {target}."
        guarded["risk"] = "low"
        guarded["guard_override"] = "missing_path_list_files_to_write"
        return guarded

    def _guard_resume_suggested_action(self, action: dict[str, Any], state: TaskState) -> dict[str, Any] | None:
        if not self.resume:
            return None
        if self._is_initializer_task(state):
            return None
        if self._has_contract_for_active_task(state):
            return None
        if action.get("action") == "contract":
            return None
        if not state.acceptance_criteria:
            return None
        active = self._active_node(state)
        if not active:
            return None
        return self._synthesize_contract_action(state)

    def _guard_repeated_failed_action(self, action: dict[str, Any], state: TaskState) -> dict[str, Any] | None:
        if state.last_observation.get("ok") is not False:
            return None
        if not self._same_action(action, state.last_action):
            return None
        return {
            "thought_summary": (
                "Guard override: the previous action failed or made no progress. "
                "Do not repeat it unchanged; follow the handoff suggested next action or choose a repair."
            ),
            "action": "blocked_repeat",
            "target": str(action.get("target", "")),
            "args": {
                "previous_action": state.last_action,
                "previous_observation_summary": state.last_observation.get("summary", ""),
            },
            "expected_observation": "Harness rejects unchanged failed actions.",
            "risk": "low",
            "guard_override": "failed_action_repeat_blocked",
        }

    def _guard_repeated_list_files(self, action: dict[str, Any], state: TaskState) -> dict[str, Any] | None:
        if action.get("action") != "list_files":
            return None
        target = self._normalize_target(action.get("target", ""))
        if not target:
            return None
        if self._recent_action_target_count(state, "list_files", target) < 2:
            return None
        return {
            "thought_summary": (
                "Guard override: list_files has already inspected this target enough times. "
                "Use the collected evidence or the handoff suggested next action instead of listing again."
            ),
            "action": "blocked_repeat",
            "target": target,
            "args": {
                "repeat_limit": 2,
                "suggested_next_action": self._suggest_next_action(state),
            },
            "expected_observation": "Harness rejects repeated directory listings of the same target.",
            "risk": "low",
            "guard_override": "list_files_repeat_limit",
        }

    def _same_action(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        if left.get("action") != right.get("action"):
            return False
        if self._normalize_target(left.get("target", "")) != self._normalize_target(right.get("target", "")):
            return False
        return self._canonical_args(left.get("args", {})) == self._canonical_args(right.get("args", {}))

    def _canonical_args(self, value: object) -> str:
        try:
            return json.dumps(value if isinstance(value, dict) else {}, sort_keys=True, ensure_ascii=False)
        except TypeError:
            return str(value)

    def _recent_action_target_count(self, state: TaskState, action_name: str, target: str) -> int:
        normalized_target = self._normalize_target(target)
        count = 0
        if (
            state.last_action.get("action") == action_name
            and self._normalize_target(state.last_action.get("target", "")) == normalized_target
        ):
            count += 1
        for item in reversed(state.evidence_sources):
            if item.get("action") != action_name:
                continue
            if self._normalize_target(item.get("target", "")) != normalized_target:
                continue
            count += 1
            if count >= 2:
                break
        return count

    def _required_write_target(self, state: TaskState) -> str | None:
        if not self._has_contract_for_active_task(state):
            return None
        for target in self._active_task_expected_artifacts(state):
            if self._is_incomplete_expected_artifact(target):
                return target
        return None

    def _is_write_for_target(self, action: dict[str, Any], target: str) -> bool:
        if action.get("action") != "write":
            return False
        return self._normalize_target(action.get("target", "")) == self._normalize_target(target)

    def _guard_frozen_acceptance_test_edit(self, action: dict[str, Any], state: TaskState) -> dict[str, Any] | None:
        if action.get("action") not in {"write", "edit"}:
            return None
        target = self._normalize_target(action.get("target", ""))
        if not target or not self._is_frozen_acceptance_artifact(target, state):
            return None
        if self._test_repair_explicitly_allowed(target, state):
            return None
        repair_targets = self._active_task_implementation_artifacts(state)
        primary = repair_targets[0] if repair_targets else target
        return {
            "thought_summary": (
                "Guard override: this test artifact is frozen acceptance evidence for the agreed contract. "
                "Repair implementation code instead of rewriting the acceptance test."
            ),
            "action": "required_repair",
            "target": primary,
            "args": {
                "targets": repair_targets or [primary],
                "blocked_test_artifact": target,
                "test_policy": self._active_task_test_policy(state),
            },
            "expected_observation": "Harness rejects edits to frozen acceptance tests unless the verifier explicitly allows test repair.",
            "risk": "low",
            "guard_override": "frozen_acceptance_test_write_blocked",
        }

    def _guard_pending_repair(self, action: dict[str, Any], state: TaskState) -> dict[str, Any] | None:
        if self._pending_repair_is_command_syntax_error(state):
            if action.get("action") == "bash":
                return action
            return {
                "thought_summary": (
                    "Guard override: the acceptance command itself has invalid Python syntax, "
                    "so run a corrected equivalent bash command before verification or implementation repair."
                ),
                "action": "required_command_repair",
                "target": str(state.pending_repair.get("command", "")),
                "args": {
                    "bad_command": state.pending_repair.get("command", ""),
                    "failure_summary": state.pending_repair.get("summary", ""),
                },
                "expected_observation": "Harness requires a corrected equivalent acceptance command.",
                "risk": "low",
                "guard_override": "failed_contract_command_syntax_requires_corrected_bash",
            }
        targets = self._pending_repair_targets(state)
        if not targets:
            return None
        repair_targets = self._pending_repair_write_targets(state)
        primary = repair_targets[0] if repair_targets else targets[0]
        missing_read = self._next_pending_repair_read(state)
        if missing_read:
            if action.get("action") == "read" and self._normalize_target(action.get("target", "")) == missing_read:
                return action
            return {
                "thought_summary": (
                    "Guard override: the last acceptance command failed. "
                    "Read the failing test/source artifact before attempting another repair."
                ),
                "action": "read",
                "target": missing_read,
                "args": {},
                "expected_observation": f"Read repair evidence artifact {missing_read}.",
                "risk": "low",
                "guard_override": "failed_contract_requires_read_before_repair",
            }
        if self._pending_repair_has_attempt(state):
            if action.get("action") == "bash" and self._is_contract_command(action.get("target", ""), state):
                return action
            command = str(state.pending_repair.get("command", "")) or self._contract_test_command(state) or ""
            return {
                "thought_summary": (
                    "Guard override: a repair was attempted for the failed acceptance command, "
                    "so rerun the same acceptance command before further inspection or edits."
                ),
                "action": "bash",
                "target": command,
                "args": {},
                "expected_observation": "Acceptance command exits with code 0 after the repair.",
                "risk": "low",
                "guard_override": "failed_contract_repair_to_retest",
            }
        if self._is_repair_for_targets(action, repair_targets):
            return self._normalize_repair_write(action)
        suffix = Path(primary).suffix.lower()
        if suffix in {".md", ".txt"}:
            content = self._initial_content_for_target(primary, state)
            if content is not None:
                return {
                    "thought_summary": (
                        "Guard override: the last acceptance command failed and referenced a text artifact, "
                        "so repair that artifact before repeating verification."
                    ),
                    "action": "write",
                    "target": primary,
                    "args": {"mode": "overwrite", "content": content},
                    "expected_observation": f"Repair acceptance artifact {primary}.",
                    "risk": "low",
                    "guard_override": "failed_contract_to_artifact_repair",
                }
        return {
            "thought_summary": (
                "Guard override: the last acceptance command failed. "
                "The worker must repair an implementation artifact before more read/list/test actions."
            ),
            "action": "required_repair",
            "target": primary,
            "args": {
                "targets": repair_targets,
                "diagnostic_targets": targets,
                "command": state.pending_repair.get("command", ""),
                "failure_summary": state.pending_repair.get("summary", ""),
            },
            "expected_observation": "Harness rejects no-progress actions until the failed acceptance command is repaired in implementation code.",
            "risk": "low",
            "guard_override": "failed_contract_requires_repair",
        }

    def _next_pending_repair_read(self, state: TaskState) -> str | None:
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        required_reads = repair.get("required_reads", [])
        read_targets = repair.get("read_targets", [])
        if not isinstance(required_reads, list):
            return None
        if not isinstance(read_targets, list):
            read_targets = []
        already_read = {self._normalize_target(target) for target in read_targets}
        for target in required_reads:
            normalized = self._normalize_target(target)
            if normalized and normalized not in already_read:
                return normalized
        return None

    def _pending_repair_has_attempt(self, state: TaskState) -> bool:
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        repaired_targets = repair.get("repaired_targets", [])
        return isinstance(repaired_targets, list) and bool(repaired_targets)

    def _pending_repair_is_command_syntax_error(self, state: TaskState) -> bool:
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        if repair.get("command_failure_type") == "command_syntax_error":
            return True
        return self._command_failure_type(str(repair.get("output", ""))) == "command_syntax_error"

    def _command_failure_type(self, output: str) -> str | None:
        if "SyntaxError:" in output and "invalid syntax" in output:
            return "command_syntax_error"
        return None

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

    def _is_repair_for_targets(self, action: dict[str, Any], targets: list[str]) -> bool:
        if action.get("action") not in {"write", "edit"}:
            return False
        target = self._normalize_target(action.get("target", ""))
        return target in {self._normalize_target(item) for item in targets}

    def _normalize_repair_write(self, action: dict[str, Any]) -> dict[str, Any]:
        if action.get("action") != "write":
            return action
        args = action.get("args", {})
        if not isinstance(args, dict) or args.get("mode", "create") != "create":
            return action
        target = str(action.get("target", ""))
        try:
            exists = bool(target) and (self.root / target).resolve().exists()
        except OSError:
            return action
        if not exists:
            return action
        guarded = dict(action)
        guarded_args = dict(args)
        guarded_args["mode"] = "overwrite"
        guarded["args"] = guarded_args
        guarded["thought_summary"] = (
            "Guard override: this is a repair for a failed acceptance command, "
            "so overwrite the existing artifact instead of using create mode."
        )
        guarded["guard_override"] = "failed_contract_create_to_overwrite"
        return guarded

    def _repair_action_after_failed_contract(self, state: TaskState) -> dict[str, Any] | None:
        if state.last_action.get("action") != "bash" or state.last_observation.get("ok") is not False:
            return None
        command = str(state.last_observation.get("data", {}).get("command") or state.last_action.get("target", ""))
        if not self._is_contract_command(command, state):
            return None
        output = str(state.last_observation.get("data", {}).get("output", ""))
        if self._command_failure_type(output) == "command_syntax_error":
            return None
        target = self._artifact_referenced_by_command(command, state)
        if not target:
            return None
        suffix = Path(target).suffix.lower()
        if suffix not in {".md", ".txt"}:
            return None
        return {
            "thought_summary": (
                "Guard override: the last acceptance command failed and referenced a task artifact, "
                "so repair that artifact before repeating verification."
            ),
            "action": "write",
            "target": target,
            "args": {"mode": "overwrite", "content": self._initial_content_for_target(target, state)},
            "expected_observation": f"Repair acceptance artifact {target}.",
            "risk": "low",
            "guard_override": "failed_contract_to_artifact_repair",
        }

    def _last_read_empty_artifact(self, state: TaskState, target: str) -> bool:
        if state.last_action.get("action") != "read":
            return False
        if self._normalize_target(state.last_action.get("target", "")) != self._normalize_target(target):
            return False
        content = state.last_observation.get("data", {}).get("content")
        return isinstance(content, str) and not content.strip()

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

    def _should_run_contract_command(self, action: dict[str, Any], state: TaskState) -> bool:
        if self._should_rewrite_list_to_write(action, state):
            return True
        return action.get("action") == "list_files" and state.last_action.get("action") == "bash"

    def _guard_duplicate_create(self, action: dict[str, Any], state: TaskState) -> dict[str, Any]:
        args = action.get("args", {})
        if not isinstance(args, dict) or args.get("mode", "create") != "create":
            return action
        target = str(action.get("target", ""))
        try:
            target_exists = bool(target) and (self.root / target).resolve().exists()
        except OSError:
            return action
        if not target_exists or not self._has_contract_for_active_task(state):
            return action
        if self._is_incomplete_expected_artifact(target):
            guarded = dict(action)
            guarded_args = dict(args)
            guarded_args["mode"] = "overwrite"
            guarded["args"] = guarded_args
            guarded["thought_summary"] = (
                "Guard override: the requested code artifact already exists but is incomplete, "
                "so overwrite it with the worker-provided implementation."
            )
            guarded["guard_override"] = "incomplete_create_to_overwrite"
            return guarded
        next_target = self._next_contract_file_target(state) or self._next_implementation_file_target(state)
        if not next_target:
            test_command = self._contract_test_command(state) if self._implementation_files_exist(state) else None
            if not test_command:
                return action
            return {
                "thought_summary": (
                    "Guard override: the requested create target already exists and required files exist, "
                    "so run the active task verification command instead."
                ),
                "action": "bash",
                "target": test_command,
                "args": {},
                "expected_observation": "Verification command exits with code 0.",
                "risk": "low",
                "guard_override": "duplicate_create_to_verification",
            }
        if next_target == target:
            return action
        guarded = dict(action)
        guarded["thought_summary"] = (
            "Guard override: the requested create target already exists, so create the next "
            "missing file from the active task requirements."
        )
        guarded["target"] = next_target
        content = self._initial_content_for_target(next_target, state)
        if content is None:
            return action
        guarded["args"] = {"mode": "create", "content": content}
        guarded["expected_observation"] = f"Create required file {next_target}."
        guarded["risk"] = "low"
        guarded["guard_override"] = "duplicate_create_to_next_required_file"
        return guarded

    def _should_rewrite_list_to_write(self, action: dict[str, Any], state: TaskState) -> bool:
        last_data = state.last_observation.get("data", {})
        if last_data.get("missing_path"):
            return True
        if state.last_action.get("action") != "list_files":
            return False
        current_target = self._normalize_target(action.get("target", ""))
        previous_target = self._normalize_target(state.last_action.get("target", ""))
        if current_target == previous_target:
            return True
        previous_entries = last_data.get("entries", [])
        if isinstance(previous_entries, list) and previous_entries:
            return current_target == self._normalize_target(last_data.get("target", ""))
        return False

    def _should_create_contract_after_inspection(self, action: dict[str, Any], state: TaskState) -> bool:
        if action.get("action") != "list_files":
            return False
        if state.last_action.get("action") == "protocol_error":
            return bool(state.acceptance_criteria)
        if state.last_action.get("action") != "list_files":
            return False
        if not state.acceptance_criteria:
            return False
        if state.last_observation.get("ok") is not True:
            return False
        current_target = self._normalize_target(action.get("target", ""))
        previous_target = self._normalize_target(state.last_action.get("target", ""))
        if current_target == previous_target:
            return True
        previous_entries = state.last_observation.get("data", {}).get("entries", [])
        return isinstance(previous_entries, list) and bool(previous_entries)

    def _synthesize_contract_action(self, state: TaskState) -> dict[str, Any]:
        task_id = self._active_task_id(state)
        checks = [str(item) for item in state.acceptance_criteria]
        for command in self._active_task_verification_commands(state):
            if command not in checks:
                checks.append(command)
        return {
            "thought_summary": (
                "Guard override: enough structure has been inspected and no acceptance contract exists, "
                "so create the contract before further coding actions."
            ),
            "action": "contract",
            "target": task_id,
            "args": {
                "task_id": task_id,
                "summary": f"Complete {state.user_goal}.",
                "checks": checks,
                "required_evidence": checks,
                "forbidden_shortcuts": [],
            },
            "expected_observation": "Verifier agrees or rejects the acceptance contract.",
            "risk": "low",
            "guard_override": "repeated_inspection_to_contract",
        }

    def _next_contract_file_target(self, state: TaskState) -> str | None:
        latest = self._active_contract(state)
        if not latest:
            return None
        text_items = list(latest.get("checks", [])) + list(latest.get("required_evidence", []))
        for item in text_items:
            for target in self._extract_file_targets(str(item)):
                path = (self.root / target).resolve()
                if path.suffix and not path.exists():
                    return target
        return None

    def _next_implementation_file_target(self, state: TaskState) -> str | None:
        for target in self._active_task_expected_artifacts(state):
            if not self._expected_artifact_satisfied(target):
                return target
        return None

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

    def _all_contract_file_targets_exist(self, state: TaskState) -> bool:
        latest = self._active_contract(state)
        if not latest:
            return False
        text_items = list(latest.get("checks", [])) + list(latest.get("required_evidence", []))
        targets: list[str] = []
        for item in text_items:
            targets.extend(self._extract_file_targets(str(item)))
        unique_targets = list(dict.fromkeys(targets))
        if not unique_targets:
            return False
        return all((self.root / target).resolve().exists() for target in unique_targets)

    def _contract_smoke_command(self, state: TaskState) -> str | None:
        latest = self._active_contract(state)
        if not latest:
            return None
        for check in latest.get("checks", []):
            text = str(check)
            match = re.search(r"smoke\s+test:\s*(.+)$", text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _contract_test_command(self, state: TaskState) -> str | None:
        commands = self._contract_commands(state)
        if not commands:
            return None
        last_command = str(state.last_observation.get("data", {}).get("command") or state.last_action.get("target", ""))
        normalized_last = self._normalize_command(last_command)
        normalized_commands = [self._normalize_command(command) for command in commands]
        if (
            state.last_action.get("action") == "bash"
            and state.last_observation.get("ok") is True
            and normalized_last in normalized_commands
        ):
            index = normalized_commands.index(normalized_last)
            if index + 1 < len(commands):
                return commands[index + 1]
        return commands[0]

    def _contract_commands(self, state: TaskState) -> list[str]:
        latest = self._active_contract(state)
        checks = list(latest.get("checks", [])) if latest else []
        checks.extend(self._active_task_verification_commands(state))
        commands: list[str] = []
        for check in checks:
            text = str(check).strip()
            smoke_match = re.search(r"smoke\s+test:\s*(.+)$", text, re.IGNORECASE)
            command = smoke_match.group(1).strip() if smoke_match else text
            if command.startswith("python "):
                commands.append(command)
        return list(dict.fromkeys(commands))

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

    def _extract_file_targets(self, text: str) -> list[str]:
        targets: list[str] = []
        pattern = r"\bFile\s+([A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+)"
        for match in re.finditer(pattern, text):
            target = match.group(1).strip("`'\".,;)")
            normalized = target.replace("\\", "/")
            try:
                path = (self.root / normalized).resolve()
            except OSError:
                continue
            if self.root in path.parents or path == self.root:
                targets.append(normalized)
        return list(dict.fromkeys(targets))

    def _dedupe_contracts(self, state: TaskState) -> None:
        deduped: dict[tuple[object, object], dict[str, Any]] = {}
        for contract in state.acceptance_contracts:
            key = (contract.get("task_id"), contract.get("summary"))
            deduped[key] = contract
        state.acceptance_contracts = list(deduped.values())

    def _last_contract_check_passed(self, state: TaskState) -> bool:
        if state.last_action.get("action") != "bash":
            return False
        if state.last_observation.get("ok") is not True:
            return False
        command = str(state.last_observation.get("data", {}).get("command") or state.last_action.get("target", ""))
        if not command:
            return False
        commands = self._contract_commands(state)
        return bool(commands) and self._normalize_command(command) == self._normalize_command(commands[-1])

    def _implementation_files_exist(self, state: TaskState) -> bool:
        targets = self._active_task_expected_artifacts(state)
        if not targets:
            return False
        return all(self._expected_artifact_satisfied(target) for target in targets)

    def _expected_artifact_satisfied(self, target: str) -> bool:
        path = (self.root / target).resolve()
        if not path.exists():
            return False
        return not self._is_incomplete_expected_artifact(target)

    def _is_incomplete_expected_artifact(self, target: str) -> bool:
        path = (self.root / target).resolve()
        if not path.exists() or not path.is_file():
            return False
        if Path(target).name == "__init__.py":
            return False
        if path.suffix.lower() != ".py":
            return False
        try:
            return not path.read_text(encoding="utf-8").strip()
        except OSError:
            return False

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
        if self._has_contract_for_active_task(state):
            return self._active_task_worker_test_artifacts(state)
        return []

    def _active_task_frozen_acceptance_artifacts(self, state: TaskState) -> list[str]:
        if self._active_task_has_key(state, "frozen_acceptance_artifacts"):
            return self._active_task_artifacts_by_key(state, "frozen_acceptance_artifacts")
        policy = self._active_task_test_policy(state)
        if policy.get("acceptance_tests_mutable_by_worker") is False and self._has_contract_for_active_task(state):
            return self._active_task_acceptance_artifacts(state)
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
            "worker_tests_mutable_until_contract_freeze": True,
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
        if normalized in worker_tests:
            policy = self._active_task_test_policy(state)
            return bool(policy.get("worker_tests_mutable_until_contract_freeze", True)) and not self._has_contract_for_active_task(state)
        return False

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

    def _initial_content_for_target(self, target: str, state: TaskState | None = None) -> str | None:
        name = Path(target).name.lower()
        if name == "init.sh" and self._normalize_target(target) == self._normalize_target(self._rel(self.state_dir / "init.sh")):
            workspace = self._expected_initializer_workspace_root() or "workspace"
            return (
                "#!/usr/bin/env sh\n"
                "set -eu\n"
                f"python -c \"import pathlib; pathlib.Path({workspace!r}).mkdir(parents=True, exist_ok=True)\"\n"
            )
        if name == "readme.md":
            title = state.user_goal if state else "Project"
            criteria = state.acceptance_criteria if state else []
            lines = [
                f"# {title}",
                "",
                "## Commands",
                "",
                "This project will provide CLI command support for create, list, show, update, and delete workflows.",
            ]
            if criteria:
                lines.extend(["", "## Acceptance Criteria", ""])
                lines.extend(f"- {item}" for item in criteria)
            return "\n".join(lines) + "\n"
        if name == "__init__.py":
            return '"""Package marker."""\n'
        if target.endswith(".py"):
            return None
        return ""

    def _update_state(self, state: TaskState, action: dict[str, Any], observation: ToolResult) -> None:
        state.iterations += 1
        state.updated_at = utc_now()
        state.last_action = action
        state.last_observation = observation.to_dict()

        name = action.get("action")
        active_task_id = self._active_task_id(state)
        if name == "contract" and observation.ok:
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
        elif name == "skill" and observation.ok:
            state.evidence_sources.append(
                {
                    "action": "skill",
                    "target": observation.data.get("path", ""),
                    "summary": observation.summary,
                }
            )
        elif name == "answer" and observation.ok and not self._is_initializer_task(state):
            for node in state.nodes:
                if node["status"] != "done":
                    node["status"] = "done"
                    node["evidence"].append(observation.summary)
        elif name == "update_plan" and state.nodes:
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
            "required_write",
            "required_repair",
            "required_command_repair",
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
        self._update_pending_repair(state, action, observation)

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
        if name == "bash":
            command = str(observation.data.get("command") or action.get("target", ""))
            if observation.ok and self._pending_repair_is_command_syntax_error(state):
                state.pending_repair = {}
                return
            if not self._is_contract_command(command, state):
                return
            canonical_command = self._canonical_contract_command(command, state)
            if observation.ok:
                state.pending_repair = {}
                return
            output = str(observation.data.get("output", ""))
            failure_type = self._command_failure_type(output)
            if failure_type == "command_syntax_error":
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
                failure_type = self._command_failure_type(output)
                if failure_type == "command_syntax_error":
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

    def _guard_initializer_progress(
        self,
        action: dict[str, Any],
        state: TaskState,
    ) -> dict[str, Any] | None:
        if not self._is_initializer_task(state):
            return None
        outputs = self._validate_initializer_outputs()
        if not outputs.ok:
            repair_action = self._guard_initializer_repair(action, state)
            if repair_action:
                return repair_action
            repair = state.initializer_repair if isinstance(state.initializer_repair, dict) else {}
            candidate = self._normalize_target(repair.get("candidate_path", ""))
            if candidate:
                if action.get("action") in {"read", "edit", "write"} and self._normalize_target(action.get("target", "")) == candidate:
                    return None
                generated_target = self._normalize_target(
                    self._rel(self.generated_tasks_path or self.state_dir / "generated_tasks.json")
                )
                args = action.get("args", {}) if isinstance(action.get("args"), dict) else {}
                content = args.get("content")
                if (
                    action.get("action") == "write"
                    and self._normalize_target(action.get("target", "")) == generated_target
                    and isinstance(content, str)
                    and content.strip()
                ):
                    return None
                return {
                    "thought_summary": (
                        "Guard override: INIT has a saved rejected candidate. "
                        "Repair that candidate before synthesizing missing initializer artifacts."
                    ),
                    "action": "read",
                    "target": candidate,
                    "args": {},
                    "expected_observation": f"Read saved INIT candidate {candidate}.",
                    "risk": "low",
                    "guard_override": "initializer_candidate_repair_before_missing_artifact",
                }
            if int(repair.get("repeat_count", 0)) >= 2:
                return None
            missing = outputs.data.get("missing_initializer_artifacts", [])
            if isinstance(missing, list) and missing:
                missing_target = self._normalize_target(missing[0])
                if action.get("action") in {"write", "edit"} and self._normalize_target(action.get("target", "")) == missing_target:
                    return None
                content = self._initial_content_for_target(missing_target, state)
                if content:
                    return {
                        "thought_summary": (
                            "Guard override: the INIT handoff requires the next missing initializer artifact. "
                            f"Write {missing_target} before further directory listings or verification."
                        ),
                        "action": "write",
                        "target": missing_target,
                        "args": {"mode": "create", "content": content},
                        "expected_observation": f"Create missing initializer artifact {missing_target}.",
                        "risk": "low",
                        "guard_override": "initializer_missing_artifact_to_write",
                    }
            return None
        command = self._initializer_verification_command(state)
        if not state.initializer_command_passed:
            if action.get("action") == "bash" and str(action.get("target", "")) == command:
                return None
            return {
                "thought_summary": (
                    "Guard override: all INIT artifacts are present and valid, so run the deterministic "
                    "initializer verification command before any answer, finish, or further inspection."
                ),
                "action": "bash",
                "target": command,
                "args": {},
                "expected_observation": "Initializer verification command exits with code 0.",
                "risk": "low",
                "guard_override": "initializer_artifacts_ready_to_command",
            }
        if action.get("action") == "verify":
            return None
        return {
            "thought_summary": (
                "Guard override: the INIT verification command passed, so submit INIT to the Verifier "
                "instead of answering, finishing, or collecting more context."
            ),
            "action": "verify",
            "target": "default",
            "args": {},
            "expected_observation": "Verifier independently validates INIT and permits Orchestrator scheduling.",
            "risk": "low",
            "guard_override": "initializer_command_passed_to_verify",
        }

    def _guard_initializer_repair(
        self,
        action: dict[str, Any],
        state: TaskState,
    ) -> dict[str, Any] | None:
        repair = state.initializer_repair if isinstance(state.initializer_repair, dict) else {}
        if int(repair.get("repeat_count", 0)) < 2:
            return None
        candidate = self._normalize_target(repair.get("candidate_path", ""))
        if not candidate:
            return None
        if action.get("action") in {"edit", "write"} and self._normalize_target(action.get("target", "")) == candidate:
            return None
        candidate_already_read = candidate in {
            self._normalize_target(item.get("target", ""))
            for item in state.evidence_sources
            if item.get("action") == "read"
        }
        if (
            action.get("action") == "read"
            and self._normalize_target(action.get("target", "")) == candidate
            and not candidate_already_read
        ):
            return None
        errors = repair.get("validation_errors", [])
        return {
            "thought_summary": (
                "Guard override: the same INIT validation error repeated. Repair the saved candidate "
                "in place instead of regenerating the whole task graph."
            ),
            "action": "required_initializer_repair",
            "target": candidate,
            "args": {
                "errors": errors if isinstance(errors, list) else [],
                "repeat_count": repair.get("repeat_count", 2),
            },
            "expected_observation": "Harness requires an edit or overwrite of the saved candidate.",
            "risk": "low",
            "guard_override": "repeated_initializer_error_to_candidate_repair",
        }

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

    def _validate_contract_action(self, action: dict[str, Any]) -> ToolResult:
        args = action.get("args", {})
        if not isinstance(args, dict):
            return ToolResult(False, "Contract rejected: args must be an object.", {})
        task_id = str(args.get("task_id") or action.get("target") or "current")
        summary = str(args.get("summary", "")).strip()
        checks = args.get("checks", [])
        if not summary:
            return ToolResult(False, "Contract rejected: summary is required.", {})
        if not isinstance(checks, list) or not checks:
            return ToolResult(False, "Contract rejected: checks must be a non-empty list.", {})
        contract = {
            "task_id": task_id,
            "summary": summary,
            "scope": args.get("scope", []),
            "checks": checks,
            "required_evidence": args.get("required_evidence", checks),
            "forbidden_shortcuts": args.get("forbidden_shortcuts", []),
            "status": "agreed",
        }
        result = self.verifier.validate_contract(contract)
        if not result.ok:
            return result
        return ToolResult(True, f"Acceptance contract agreed for {task_id}.", {"contract": contract})

    def _handle_skill_action(self, action: dict[str, Any], state: TaskState) -> ToolResult:
        args = action.get("args", {})
        if not isinstance(args, dict):
            return ToolResult(False, "Skill rejected: args must be an object.", {})
        skill_id = self._safe_skill_id(str(args.get("skill_id") or action.get("target") or ""))
        title = str(args.get("title", "")).strip()
        body = str(args.get("body", "")).strip()
        evidence_type = str(args.get("evidence_type", "")).strip()
        evidence = args.get("evidence", [])
        if not skill_id or not title or not body:
            return ToolResult(False, "Skill rejected: skill_id, title, and body are required.", {})
        if evidence_type not in {"verified_success", "evidence_confirmed_failure"}:
            return ToolResult(
                False,
                "Skill rejected: evidence_type must be verified_success or evidence_confirmed_failure.",
                {},
            )
        if not isinstance(evidence, list) or not evidence:
            return ToolResult(False, "Skill rejected: evidence list is required.", {})
        result = self.verifier.validate_skill_promotion(
            {
                "skill_id": skill_id,
                "title": title,
                "body": body,
                "evidence_type": evidence_type,
                "evidence": evidence,
            },
            state,
        )
        if not result.ok:
            return result
        skill_path = self.state_dir / "skills" / f"{skill_id}.md"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(
            f"# {title}\n\n"
            f"Evidence type: {evidence_type}\n\n"
            "## Evidence\n\n"
            + "\n".join(f"- {item}" for item in evidence)
            + "\n\n## Procedure\n\n"
            + body
            + "\n",
            encoding="utf-8",
        )
        return ToolResult(True, f"Skill promoted: {skill_id}.", {"path": str(skill_path.relative_to(self.root))})

    def _safe_skill_id(self, raw: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in raw.strip().lower())
        return cleaned.strip("-_")

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

    def _append_trace(
        self,
        step: int,
        action: dict[str, Any],
        observation: ToolResult,
        state: TaskState,
        model_action: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "step": step,
            "time": utc_now(),
            "model_action": model_action,
            "action": action,
            "observation": observation.to_dict(),
            "state_summary": state.summary(),
            "task_id": state.task_id,
            "session_used_tokens": state.session_used_tokens,
            "handoff_ready": state.handoff_ready,
            "orchestrator_decision": state.orchestrator_decision,
            "nodes": state.nodes,
        }
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, indent=2) + "\n")

    def _write_handoff(self, state: TaskState) -> None:
        active_node = self._active_node(state)
        completed = [node for node in state.nodes if node.get("status") == "done"]
        pending = [node for node in state.nodes if node.get("status") != "done"]
        contracts = state.acceptance_contracts[-5:]
        evidence = state.evidence_sources[-20:]
        payload = self._write_handoff_payload(state, active_node, completed, pending, contracts, evidence)
        lines = [
            "# Worker Session Handoff",
            "",
            "## 1. User Goal",
            state.user_goal,
            "",
            "## 2. Session Budget",
            f"- budget_tokens: {state.session_budget_tokens}",
            f"- threshold_ratio: {state.handoff_threshold}",
            f"- threshold_tokens: {int(state.session_budget_tokens * state.handoff_threshold)}",
            f"- estimated_used_tokens: {state.session_used_tokens}",
            f"- handoff_ready: {state.handoff_ready}",
            "",
            "## 3. Active Task",
            self._format_node(active_node) if active_node else "No active task.",
            "",
            "## 4. Handoff Data References",
            f"- structured_payload: {payload['payload_path']}",
            f"- current_state: {self._rel(self.state_path)}",
            f"- task_graph_runtime: {self._rel(self.runtime_tasks_path) if self.runtime_tasks_path else 'not used'}",
            f"- task_graph_generated: {self._rel(self.generated_tasks_path) if self.generated_tasks_path else 'not used'}",
            f"- rejected_initializer_candidate: {self._rel(self.initializer_candidate_path)}",
            "- task_graph_source: the original `--tasks-json` file is benchmark input and should remain read-only",
            f"- latest_verifier_report: {self._rel(self.state_dir / 'verifier_report.md')}",
            f"- hard_memory: {self._rel(self.state_dir / 'hard_memory.md')}",
            f"- soft_memory: {self._rel(self.state_dir / 'soft_memory.md')}",
            f"- traces: {self._rel(self.trace_dir)}/",
            "",
            "## 5. Orchestrator Decision",
            f"- selected_task_id: {state.orchestrator_decision.get('selected_task_id')}",
            f"- reason: {state.orchestrator_decision.get('reason')}",
            f"- ready_task_ids: {state.orchestrator_decision.get('ready_task_ids')}",
            "",
            "## 6. Completed Tasks",
            *[self._format_node(node) for node in completed],
            "",
            "## 7. Pending Or Blocked Tasks",
            *[self._format_node(node) for node in pending],
            "",
            "## 8. Acceptance Contracts",
            *[self._format_contract(contract) for contract in contracts],
            "",
            "## 9. Evidence Sources",
            *[f"- {item.get('action')}: {item.get('target')} -- {item.get('summary')}" for item in evidence],
            "",
            "## 10. Last Step Summary",
            f"- last_action: {state.last_action.get('action')} {state.last_action.get('target', '')}",
            f"- last_observation_ok: {state.last_observation.get('ok')}",
            f"- last_observation_summary: {state.last_observation.get('summary')}",
            "",
            "## 10a. Pending Repair",
            *self._format_pending_repair(state),
            "",
            "## 10b. Initializer Repair",
            *self._format_initializer_repair(state),
            "",
            "## 11. Verification Status",
            f"- last_verified_at: {state.last_verified_at}",
            "- deterministic verifier: run `python -m unittest discover -s tests` and `python -m compileall agent eval tests`.",
            "",
            "## 12. Known Risks And Failed Attempts",
            "- Review trace for failed observations and protocol errors before resuming.",
            "- Do not repeat failed actions unchanged.",
            "- Do not start new large edits until the next worker session has rebuilt context from this handoff.",
            "- Treat Soft Memory as hypotheses, not facts.",
            "",
            "## 13. Current State Summary",
            state.summary(),
            "",
            "## 14. Resume Instructions",
            "1. Read this handoff first.",
            f"2. Load `{self._rel(self.state_path)}`, the active task graph, and relevant source files.",
            "3. Continue the active task only after checking the acceptance contract.",
            "4. If no contract exists for a coding task, create one before writing code.",
            "5. Prefer verification or small repair actions before new feature work.",
            "6. Promote Soft Memory to Hard Memory only after verification.",
            f"7. Inspect `{payload['payload_path']}` only when the summary above is insufficient.",
            "",
            "## 15. Suggested Next Action",
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
        state.session_used_tokens += max(1, len(payload) // 4)
        threshold_tokens = int(state.session_budget_tokens * state.handoff_threshold)
        if state.session_used_tokens >= threshold_tokens:
            state.handoff_ready = True

    def _active_node(self, state: TaskState) -> dict[str, Any] | None:
        for node in state.nodes:
            if node.get("status") in {"in_progress", "pending"}:
                return node
        return None

    def _format_node(self, node: dict[str, Any]) -> str:
        evidence = node.get("evidence", [])
        evidence_text = "; ".join(str(item) for item in evidence[-3:]) if evidence else "no evidence yet"
        return f"- {node.get('id')}: [{node.get('status')}] {node.get('title')} | evidence: {evidence_text}"

    def _format_contract(self, contract: dict[str, Any]) -> str:
        checks = "; ".join(str(item) for item in contract.get("checks", []))
        return f"- {contract.get('task_id')}: {contract.get('summary')} | checks: {checks}"

    def _format_pending_repair(self, state: TaskState) -> list[str]:
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        targets = self._pending_repair_targets(state)
        if not repair or not targets:
            return ["- none"]
        return [
            f"- reason: {repair.get('reason', 'failed_acceptance_command')}",
            f"- command: {repair.get('command', '')}",
            f"- targets: {', '.join(targets)}",
            f"- repair_targets: {', '.join(str(item) for item in repair.get('repair_targets', []))}",
            f"- required_reads: {', '.join(str(item) for item in repair.get('required_reads', []))}",
            f"- read_targets: {', '.join(str(item) for item in repair.get('read_targets', []))}",
            f"- repaired_targets: {', '.join(str(item) for item in repair.get('repaired_targets', []))}",
            f"- summary: {repair.get('summary', '')}",
        ]

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
            if not state.initializer_command_passed:
                return "Run the deterministic INIT verification command; answer and finish are not allowed."
            return "Submit INIT to the verifier; only Verifier PASS may schedule the first Worker task."
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
            return f"Create an acceptance contract for {active.get('id')} before writing code."
        return f"Resume {active.get('id')} with a small evidence-backed action, then verify."

    def _ensure_state_files(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "skills").mkdir(exist_ok=True)
        if not self.memory_path.exists():
            self.memory_path.write_text("# Memory Index\n\nSee hard_memory.md and soft_memory.md.\n", encoding="utf-8")
        hard_memory = self.state_dir / "hard_memory.md"
        if not hard_memory.exists():
            hard_memory.write_text("# Hard Memory\n\n## Entries\n\n", encoding="utf-8")
        soft_memory = self.state_dir / "soft_memory.md"
        if not soft_memory.exists():
            soft_memory.write_text("# Soft Memory\n\n## Entries\n\n", encoding="utf-8")
        if not (self.state_dir / "skills" / "coding.md").exists():
            (self.state_dir / "skills" / "coding.md").write_text(
                "# Coding Skill\n\n- Inspect files before editing.\n- Prefer small verifiable steps.\n- Run syntax checks before finishing.\n",
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
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"run_{stamp}.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
