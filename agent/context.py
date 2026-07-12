from __future__ import annotations

import subprocess
from pathlib import Path

from agent.planner import TaskState


class ContextBuilder:
    def __init__(self, root: Path, max_chars: int = 12000, state_dir: Path | None = None) -> None:
        self.root = root
        self.max_chars = max_chars
        self.state_dir = state_dir or root / "state"

    def build(self, state: TaskState) -> str:
        sections = [
            self._always_on_context(state),
            self._startup_context(),
            self._just_in_time_context(state),
            self._persistent_context(state),
        ]
        text = "\n\n".join(section for section in sections if section.strip())
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars] + "\n\n[context truncated by harness]"

    def _always_on_context(self, state: TaskState) -> str:
        lines = [
            "# Always-on Context",
            "You are the decision model inside a long-running coding agent harness.",
            "Return one schema-valid action. The harness owns verification and state transitions.",
            f"Current task id: {self._active_task_id(state)}",
            f"Orchestrator decision: {state.orchestrator_decision}",
            f"Required next action: {self._required_next_action(state)}",
            "Runtime environment: Windows PowerShell. Prefer portable Python commands or PowerShell commands.",
            "Tool conventions: use list_files for directories; use read for file ranges; use bash target as the command string.",
            "Use action='answer' when enough evidence has been collected for an inspection, explanation, recommendation, or next-step request.",
            "For action='answer', put the final response in args.answer and cite evidence from observations.",
            "For action='contract', args must include task_id, summary, and checks; checks must be a non-empty list with behavior-level test or smoke evidence.",
            "If list_files reports a missing target directory for a task whose goal is to create that directory, do not repeat list_files; use write to create the first required file. Write creates parent directories.",
            "If an agreed acceptance contract already exists for the current task, do not submit another contract unless the verifier rejected the existing one.",
            "Do not modify acceptance criteria. Use update_plan only to propose state changes.",
            "Completion requires verifier evidence; do not self-certify completion.",
            "Worker cannot mark tasks completed. Only Verifier PASS followed by Orchestrator state transition may complete a task.",
            "Avoid Unix-only commands such as head, grep, sed, and find unless you know they exist.",
        ]
        return "\n".join(lines)

    def _startup_context(self) -> str:
        state_label = self._rel(self.state_dir)
        lines = [
            "# Startup Context",
            f"## {state_label}/project_spec.md",
            self._read_optional(self.state_dir / "project_spec.md", max_chars=2500)
            or self._read_optional(self.root / "project_spec.md", max_chars=2500),
            "## tasks.json",
            self._read_optional(self.root / "tasks.json", max_chars=2500),
            f"## {state_label}/generated_tasks.json",
            self._read_optional(self.state_dir / "generated_tasks.json", max_chars=2500),
            f"## {state_label}/runtime_tasks.json",
            self._read_optional(self.state_dir / "runtime_tasks.json", max_chars=2500),
            f"## {state_label}/handoff.md",
            self._read_optional(self.state_dir / "handoff.md", max_chars=2500),
            f"## {state_label}/verifier_report.md",
            self._read_optional(self.state_dir / "verifier_report.md", max_chars=2500),
            "## git log",
            self._run_git(["log", "--oneline", "-5"]),
            "## git status",
            self._run_git(["status", "--short", "--branch"]),
        ]
        return "\n".join(lines)

    def _just_in_time_context(self, state: TaskState) -> str:
        lines = [
            "# Just-in-Time Context",
            "Do not preload the whole repository. Read only what is needed for the active task.",
            "Recommended discovery flow:",
            "1. list a small directory with read target='.' or read target='<dir>';",
            "2. search relevant symbols or filenames;",
            "3. read the smallest relevant source file ranges;",
            "4. read corresponding tests;",
            "5. use errors or verifier output to guide the next search.",
            "PowerShell/Python examples:",
            "- list_files target='agent'",
            "- search target='create_issue' args={'path': 'agent'}",
            "- read target='agent/loop.py' args={'start': 1, 'end': 220}",
            "",
            "## Evidence Sources Read So Far",
            *[f"- {item.get('action')}: {item.get('target')} -- {item.get('summary')}" for item in state.evidence_sources[-12:]],
        ]
        return "\n".join(lines)

    def _persistent_context(self, state: TaskState) -> str:
        memory_index = self._read_optional(self.state_dir / "memory.md")
        hard_memory = self._read_optional(self.state_dir / "hard_memory.md")
        soft_memory = self._read_optional(self.state_dir / "soft_memory.md")
        skills = self._read_skills()
        lines = [
            "# Persistent Context",
            "Persist cross-session information in files rather than relying on chat history.",
            "Persistent files include task status, verified facts, architecture decisions, failed attempts, verifier reports, git commits, and next actions.",
            "Hard Memory is evidence-grade. Soft Memory is not evidence; treat it only as a hypothesis or suggestion.",
            "",
            "# User Goal",
            state.user_goal,
            "",
            "# Acceptance Criteria",
            *[f"- {item}" for item in state.acceptance_criteria],
            "",
            "# Session Budget",
            f"- budget_tokens: {state.session_budget_tokens}",
            f"- handoff_threshold: {state.handoff_threshold}",
            f"- estimated_used_tokens: {state.session_used_tokens}",
            f"- handoff_ready: {state.handoff_ready}",
            "When handoff_ready is true, do not start new large edits. Prefer verify, summarize, or finish.",
            "",
            "# Plan",
            *[f"- [{n['status']}] {n['id']}: {n['title']}" for n in state.nodes],
            "",
            "# Active Task Expected Artifacts",
            *[
                f"- {artifact}"
                for node in state.nodes
                if node.get("status") in {"in_progress", "pending"}
                for artifact in self._format_artifacts(node.get("expected_artifacts", []))
            ],
            "",
            *self._initializer_instruction_lines(state),
            "",
            "# Active Task Artifact Policy",
            *self._artifact_policy_lines(state),
            "",
            "# Active Task Verification Commands",
            *[
                f"- {command}"
                for node in state.nodes
                if node.get("status") in {"in_progress", "pending"}
                for command in self._format_artifacts(node.get("verification_commands", []))
            ],
            "",
            "# Evidence Sources",
            *[f"- {item.get('action')}: {item.get('target')}" for item in state.evidence_sources[-12:]],
            "",
            "# Acceptance Contracts",
            *[
                f"- {item.get('task_id')}: {item.get('status', 'proposed')} - {item.get('summary', '')}"
                for item in state.acceptance_contracts[-5:]
            ],
            "",
            "# Last Action",
            str(state.last_action),
            "",
            "# Last Observation",
            str(state.last_observation),
            "",
            "# Required Next Action",
            self._required_next_action(state),
            "",
            "# Memory Index",
            memory_index,
            "",
            "# Hard Memory",
            hard_memory,
            "",
            "# Soft Memory",
            soft_memory,
            "",
            "# Skills",
            skills,
        ]
        return "\n".join(lines)

    def _initializer_instruction_lines(self, state: TaskState) -> list[str]:
        if self._active_task_id(state) != "INIT":
            return []
        return [
            "# Initializer Requirements",
            "This is the one-time Initializer / Planner stage.",
            f"Read {self._rel(self.state_dir / 'project_spec.md')} and transform it into a structured task graph.",
            "Required outputs:",
            f"- {self._rel(self.state_dir / 'project_spec.md')} must exist as the durable project specification.",
            f"- {self._rel(self.state_dir / 'generated_tasks.json')} must contain a JSON object with a non-empty tasks list.",
            f"- {self._rel(self.state_dir / 'init.sh')} must contain repeatable setup or smoke-test commands.",
            "Each generated task should include: id, title, priority, depends_on, status, acceptance_criteria, expected_artifacts, implementation_artifacts when applicable, worker_test_artifacts when applicable, acceptance_artifacts when applicable, frozen_acceptance_artifacts when applicable, test_policy when tests are involved, and verification_commands.",
            "Do not implement the application during INIT beyond tiny skeleton files only if the spec requires them. The main output is the task graph.",
            "After writing initializer artifacts, run the initializer verification command from the active task.",
        ]

    def _required_next_action(self, state: TaskState) -> str:
        target = self._incomplete_expected_code_artifact(state)
        if target:
            return (
                f"The expected code artifact {target} is empty/incomplete. "
                f"Next action must be write target='{target}' with args.mode='overwrite' and complete implementation content. "
                "Do not list directories, rerun tests, or reread files before writing it."
            )
        if state.last_observation.get("data", {}).get("required_action") == "write":
            target = str(state.last_observation.get("data", {}).get("target", ""))
            return (
                f"The harness rejected the previous action because {target} is incomplete. "
                f"Next action must be write target='{target}' with args.mode='overwrite'."
            )
        if state.last_observation.get("data", {}).get("required_action") == "write_or_edit":
            targets = self._pending_repair_write_targets(state)
            target_text = ", ".join(targets) if targets else str(state.last_observation.get("data", {}).get("target", ""))
            return (
                "The harness rejected the previous action because the last acceptance command failed. "
                f"Next action must be write or edit one of these implementation artifacts: {target_text}. "
                "Do not list directories, reread files, or rerun tests before making a repair. "
                "Frozen acceptance tests are read-only unless allow_test_repair is explicitly set."
            )
        diagnostic_targets = self._pending_repair_targets(state)
        if diagnostic_targets:
            repair_targets = self._pending_repair_write_targets(state)
            missing_read = self._next_pending_repair_read(state)
            output = str(state.pending_repair.get("output", "")).strip().splitlines()
            excerpt = " | ".join(output[:4])[:700] if output else state.pending_repair.get("summary", "")
            if missing_read:
                return (
                    "The last acceptance or verification command failed. "
                    f"Next action must be read target='{missing_read}' before any write/edit repair. "
                    "Use the failing test/source artifact to derive the required interface instead of guessing. "
                    f"Failure excerpt: {excerpt}"
                )
            if self._pending_repair_has_attempt(state):
                command = str(state.pending_repair.get("command", ""))
                return (
                    "A repair was attempted for the failed acceptance command. "
                    f"Next action must be bash target='{command}' to rerun the same acceptance command. "
                    "Do not list directories or continue editing until this command is rerun."
                )
            return (
                "The last acceptance or verification command failed. "
                f"Next action must be write or edit one of these implementation artifacts: {', '.join(repair_targets)}. "
                f"Start with {repair_targets[0]}. "
                "Worker-owned tests may be edited before contract freeze, but frozen acceptance tests must not be modified unless the harness explicitly marks allow_test_repair=true. "
                "Do not list directories, reread files, or rerun tests before making a repair. "
                f"Failure excerpt: {excerpt}"
            )
        if state.last_action.get("action") != "read":
            return "No forced next action."
        target = str(state.last_action.get("target", ""))
        content = state.last_observation.get("data", {}).get("content")
        if not isinstance(content, str) or content.strip():
            return "No forced next action."
        active_artifacts = {
            artifact
            for node in state.nodes
            if node.get("status") in {"in_progress", "pending"}
            for artifact in self._format_artifacts(node.get("expected_artifacts", []))
        }
        if target not in active_artifacts or not target.endswith(".py"):
            return "No forced next action."
        return (
            f"The expected code artifact {target} was read and is empty. "
            "Next action should be write with args.mode='overwrite' and a complete implementation for that file. "
            "Do not list directories, rerun tests, or reread the same empty file before writing it."
        )

    def _incomplete_expected_code_artifact(self, state: TaskState) -> str | None:
        for node in state.nodes:
            if node.get("status") not in {"in_progress", "pending"}:
                continue
            for artifact in self._format_artifacts(node.get("expected_artifacts", [])):
                path = self.root / artifact
                if path.name == "__init__.py" or path.suffix.lower() != ".py":
                    continue
                if not path.exists() or not path.is_file():
                    continue
                try:
                    if not path.read_text(encoding="utf-8").strip():
                        return artifact
                except OSError:
                    continue
        return None

    def _next_pending_repair_read(self, state: TaskState) -> str | None:
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        required_reads = repair.get("required_reads", [])
        read_targets = repair.get("read_targets", [])
        if not isinstance(required_reads, list):
            return None
        if not isinstance(read_targets, list):
            read_targets = []
        already_read = {str(target).replace("\\", "/").strip().rstrip("/") for target in read_targets}
        for target in required_reads:
            normalized = str(target).replace("\\", "/").strip().rstrip("/")
            if normalized and normalized not in already_read:
                return normalized
        return None

    def _pending_repair_has_attempt(self, state: TaskState) -> bool:
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        repaired_targets = repair.get("repaired_targets", [])
        return isinstance(repaired_targets, list) and bool(repaired_targets)

    def _pending_repair_targets(self, state: TaskState) -> list[str]:
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        targets = repair.get("targets", [])
        if not isinstance(targets, list):
            return []
        active_artifacts = {
            artifact
            for node in state.nodes
            if node.get("status") in {"in_progress", "pending"}
            for artifact in self._format_artifacts(node.get("expected_artifacts", []))
        }
        result: list[str] = []
        for target in targets:
            normalized = str(target).replace("\\", "/").strip().rstrip("/")
            if normalized in active_artifacts and normalized not in result:
                result.append(normalized)
        return result

    def _pending_repair_write_targets(self, state: TaskState) -> list[str]:
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        explicit = repair.get("repair_targets", [])
        if isinstance(explicit, list) and explicit:
            active = {
                artifact
                for node in state.nodes
                if node.get("status") in {"in_progress", "pending"}
                for artifact in self._format_artifacts(node.get("expected_artifacts", []))
            }
            return [
                target
                for target in (str(item).replace("\\", "/").strip().rstrip("/") for item in explicit)
                if target in active and (not self._looks_like_test_artifact(target) or self._is_test_repair_allowed(target, state))
            ]
        targets = self._pending_repair_targets(state)
        implementation_targets = [
            target
            for target in self._active_task_implementation_artifacts(state)
            if target in {str(item).replace("\\", "/").strip().rstrip("/") for item in targets}
        ]
        return implementation_targets or targets

    def _artifact_policy_lines(self, state: TaskState) -> list[str]:
        implementation = self._active_task_implementation_artifacts(state)
        worker_tests = self._active_task_worker_test_artifacts(state)
        acceptance = self._active_task_acceptance_artifacts(state)
        frozen = self._active_task_frozen_acceptance_artifacts(state)
        policy = self._active_task_test_policy(state)
        return [
            f"- implementation_artifacts: {', '.join(implementation) if implementation else 'none'}",
            f"- worker_test_artifacts: {', '.join(worker_tests) if worker_tests else 'none'}",
            f"- acceptance_artifacts: {', '.join(acceptance) if acceptance else 'none'}",
            f"- frozen_acceptance_artifacts: {', '.join(frozen) if frozen else 'none'}",
            f"- test_policy: {policy}",
            "- Rule: implementation artifacts are normal repair targets; worker tests are mutable before contract freeze; frozen acceptance tests are read-only for Worker.",
        ]

    def _active_task_implementation_artifacts(self, state: TaskState) -> list[str]:
        if self._active_task_has_key(state, "implementation_artifacts"):
            explicit = self._active_task_artifacts_by_key(state, "implementation_artifacts")
            if explicit:
                return explicit
        return [
            artifact
            for node in state.nodes
            if node.get("status") in {"in_progress", "pending"}
            for artifact in self._format_artifacts(node.get("expected_artifacts", []))
            if Path(artifact).suffix.lower() == ".py"
            and Path(artifact).name != "__init__.py"
            and not self._looks_like_test_artifact(artifact)
        ]

    def _active_task_worker_test_artifacts(self, state: TaskState) -> list[str]:
        if self._active_task_has_key(state, "worker_test_artifacts"):
            explicit = self._active_task_artifacts_by_key(state, "worker_test_artifacts")
            if explicit:
                return explicit
        return [
            artifact
            for node in state.nodes
            if node.get("status") in {"in_progress", "pending"}
            for artifact in self._format_artifacts(node.get("expected_artifacts", []))
            if self._looks_like_test_artifact(artifact)
        ]

    def _active_task_acceptance_artifacts(self, state: TaskState) -> list[str]:
        if self._active_task_has_key(state, "acceptance_artifacts"):
            return self._active_task_artifacts_by_key(state, "acceptance_artifacts")
        if any(item.get("status") == "agreed" for item in state.acceptance_contracts):
            return self._active_task_worker_test_artifacts(state)
        return []

    def _active_task_frozen_acceptance_artifacts(self, state: TaskState) -> list[str]:
        if self._active_task_has_key(state, "frozen_acceptance_artifacts"):
            return self._active_task_artifacts_by_key(state, "frozen_acceptance_artifacts")
        policy = self._active_task_test_policy(state)
        if policy.get("acceptance_tests_mutable_by_worker") is False and any(
            item.get("status") == "agreed" for item in state.acceptance_contracts
        ):
            return self._active_task_acceptance_artifacts(state)
        return []

    def _active_task_has_key(self, state: TaskState, key: str) -> bool:
        return any(node.get("status") in {"in_progress", "pending"} and key in node for node in state.nodes)

    def _active_task_artifacts_by_key(self, state: TaskState, key: str) -> list[str]:
        artifacts: list[str] = []
        for node in state.nodes:
            if node.get("status") in {"in_progress", "pending"}:
                artifacts.extend(self._format_artifacts(node.get(key, [])))
        return artifacts

    def _active_task_test_policy(self, state: TaskState) -> dict[str, object]:
        for node in state.nodes:
            if node.get("status") in {"in_progress", "pending"}:
                policy = node.get("test_policy", {})
                if isinstance(policy, dict):
                    merged: dict[str, object] = {
                        "worker_tests_mutable_until_contract_freeze": True,
                        "acceptance_tests_mutable_by_worker": False,
                        "acceptance_test_repair_requires_verifier_approval": True,
                    }
                    merged.update(policy)
                    return merged
        return {
            "worker_tests_mutable_until_contract_freeze": True,
            "acceptance_tests_mutable_by_worker": False,
            "acceptance_test_repair_requires_verifier_approval": True,
        }

    def _is_frozen_acceptance_artifact(self, target: str, state: TaskState) -> bool:
        normalized = target.replace("\\", "/").strip().rstrip("/")
        return normalized in {
            item.replace("\\", "/").strip().rstrip("/")
            for item in self._active_task_frozen_acceptance_artifacts(state)
        }

    def _is_test_repair_allowed(self, target: str, state: TaskState) -> bool:
        if not self._looks_like_test_artifact(target):
            return True
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        if repair.get("allow_test_repair") is True:
            return True
        if self._is_frozen_acceptance_artifact(target, state):
            return False
        normalized = target.replace("\\", "/").strip().rstrip("/")
        worker_tests = {
            item.replace("\\", "/").strip().rstrip("/")
            for item in self._active_task_worker_test_artifacts(state)
        }
        return normalized in worker_tests and not any(item.get("status") == "agreed" for item in state.acceptance_contracts)

    def _looks_like_test_artifact(self, target: str) -> bool:
        normalized = target.replace("\\", "/")
        return "/tests/" in normalized or Path(normalized).name.startswith("test_")

    def _format_artifacts(self, items: object) -> list[str]:
        if not isinstance(items, list):
            return []
        formatted = []
        for item in items:
            if isinstance(item, str):
                formatted.append(item)
            elif isinstance(item, dict) and item.get("path"):
                formatted.append(str(item["path"]))
        return formatted

    def _read_optional(self, path: Path, max_chars: int = 4000) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")[:max_chars]

    def _read_skills(self) -> str:
        skill_dir = self.state_dir / "skills"
        if not skill_dir.exists():
            return ""
        chunks = []
        for path in sorted(skill_dir.glob("*.md"))[:5]:
            chunks.append(f"## {path.name}\n{path.read_text(encoding='utf-8')[:2000]}")
        return "\n\n".join(chunks)

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root)).replace("\\", "/")
        except ValueError:
            return str(path).replace("\\", "/")

    def _run_git(self, args: list[str]) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=self.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
        except Exception as exc:
            return f"git command failed: {exc}"
        output = (completed.stdout + completed.stderr).strip()
        return output[:2500]

    def _active_task_id(self, state: TaskState) -> str:
        for node in state.nodes:
            if node.get("status") in {"in_progress", "pending"}:
                return str(node.get("id", "current"))
        return "current"
