from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TERMINAL_DONE_STATUSES = {"completed", "done"}
WORKER_READY_STATUSES = {"pending", "in_progress"}
VALID_TASK_STATUSES = {"pending", "in_progress", "awaiting_verification", "completed", "done", "blocked"}


@dataclass(frozen=True)
class TaskSelection:
    task: dict[str, Any] | None
    reason: str
    ready_task_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_task_id": self.task.get("id") if self.task else None,
            "reason": self.reason,
            "ready_task_ids": self.ready_task_ids,
        }


class Orchestrator:
    """Selects the single task a Worker session should focus on.

    The Orchestrator does not implement code and does not verify completion.
    It only chooses the next Worker-ready task from durable task state.
    """

    def __init__(self, root: Path, tasks_path: Path | None = None, state_dir: Path | None = None) -> None:
        self.root = root
        self.tasks_path = tasks_path or root / "tasks.json"
        self.state_dir = state_dir or root / "state"
        self.verifier_report_path = self.state_dir / "verifier_report.md"

    def choose_current_task(self) -> TaskSelection:
        tasks = self.load_tasks()
        failed_task_id = self.latest_failed_task_id()
        return select_current_task(tasks, failed_task_id=failed_task_id)

    def mark_in_progress(self, task_id: str, evidence: str | None = None) -> None:
        self.transition_task(task_id, "in_progress", evidence=evidence)

    def mark_awaiting_verification(self, task_id: str, evidence: str | None = None) -> None:
        self.transition_task(task_id, "awaiting_verification", evidence=evidence)

    def mark_verified(self, task_id: str, passed: bool, evidence: str | None = None) -> None:
        self.transition_task(task_id, "completed" if passed else "in_progress", evidence=evidence)

    def ensure_repair_task(
        self,
        *,
        source: str,
        title: str,
        acceptance_criteria: list[str],
        expected_artifacts: list[str],
        verification_commands: list[str],
        evidence: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        data = self.load_task_file()
        tasks = data.get("tasks", [])
        if not isinstance(tasks, list):
            return None
        for task in tasks:
            if not isinstance(task, dict):
                continue
            if task.get("repair_source") == source and normalize_status(task.get("status")) in WORKER_READY_STATUSES:
                return task

        repair_id = self.next_repair_task_id(tasks)
        dependencies = [
            str(task.get("id"))
            for task in tasks
            if isinstance(task, dict)
            and str(task.get("id", "")).strip()
            and normalize_status(task.get("status")) in TERMINAL_DONE_STATUSES
            and not str(task.get("id", "")).startswith("R")
        ]
        commands = [str(command) for command in verification_commands if str(command).strip()]
        criteria = [str(item) for item in acceptance_criteria if str(item).strip()]
        mapping = {criterion: list(commands) for criterion in criteria}
        artifacts = [str(item) for item in expected_artifacts if str(item).strip()]
        repair_task: dict[str, Any] = {
            "id": repair_id,
            "title": title,
            "priority": 0,
            "depends_on": dependencies,
            "status": "pending",
            "acceptance_criteria": criteria,
            "criterion_command_map": mapping,
            "expected_artifacts": artifacts,
            "implementation_artifacts": list(artifacts),
            "worker_test_artifacts": [],
            "acceptance_artifacts": [],
            "frozen_acceptance_artifacts": [],
            "test_policy": {
                "acceptance_tests_mutable_by_worker": False,
                "acceptance_test_repair_requires_verifier_approval": True,
            },
            "verification_commands": commands,
            "repair_source": source,
            "repair_metadata": metadata or {},
            "evidence": [evidence] if evidence else [],
        }
        tasks.append(repair_task)
        self.save_task_file(data)
        return repair_task

    def next_repair_task_id(self, tasks: list[dict[str, Any]]) -> str:
        highest = 0
        for task in tasks:
            match = re.fullmatch(r"R(\d+)", str(task.get("id", "")))
            if match:
                highest = max(highest, int(match.group(1)))
        return f"R{highest + 1}"

    def transition_task(self, task_id: str, status: str, evidence: str | None = None) -> dict[str, Any] | None:
        status = normalize_status(status)
        if status not in VALID_TASK_STATUSES:
            raise ValueError(f"Unsupported task status: {status}")
        data = self.load_task_file()
        tasks = data.get("tasks", [])
        if not isinstance(tasks, list):
            return None
        for task in tasks:
            if not isinstance(task, dict) or str(task.get("id", "")) != task_id:
                continue
            task["status"] = status
            if evidence:
                task.setdefault("evidence", [])
                if isinstance(task["evidence"], list):
                    task["evidence"].append(evidence)
            self.save_task_file(data)
            return task
        return None

    def load_tasks(self) -> list[dict[str, Any]]:
        data = self.load_task_file()
        tasks = data.get("tasks", [])
        if not isinstance(tasks, list):
            return []
        return [task for task in tasks if isinstance(task, dict)]

    def load_task_file(self) -> dict[str, Any]:
        if not self.tasks_path.exists():
            return {"tasks": []}
        data = json.loads(self.tasks_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"tasks": []}

    def save_task_file(self, data: dict[str, Any]) -> None:
        self.tasks_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def latest_failed_task_id(self) -> str | None:
        if not self.verifier_report_path.exists():
            return None
        text = self.verifier_report_path.read_text(encoding="utf-8")
        start = text.find("```json")
        if start == -1:
            return None
        start = text.find("\n", start)
        end = text.find("```", start + 1)
        if start == -1 or end == -1:
            return None
        try:
            report = json.loads(text[start:end].strip())
        except json.JSONDecodeError:
            return None
        if report.get("ok") is not False:
            return None
        data = report.get("data", {})
        if not isinstance(data, dict):
            return None
        task_id = data.get("task_id") or data.get("selected_task_id")
        if task_id:
            return str(task_id)
        contract = data.get("contract")
        if isinstance(contract, dict) and contract.get("task_id"):
            return str(contract["task_id"])
        return None


def select_current_task(tasks: list[dict[str, Any]], failed_task_id: str | None = None) -> TaskSelection:
    ready_tasks = [task for task in tasks if is_worker_ready(task, tasks, failed_task_id)]
    ready_tasks.sort(key=lambda task: task_sort_key(task, tasks, failed_task_id))
    if not ready_tasks:
        return TaskSelection(None, "No worker-ready task found.", [])
    selected = ready_tasks[0]
    return TaskSelection(
        selected,
        selection_reason(selected, tasks, failed_task_id),
        [str(task.get("id", "")) for task in ready_tasks],
    )


def is_worker_ready(task: dict[str, Any], tasks: list[dict[str, Any]], failed_task_id: str | None = None) -> bool:
    task_id = str(task.get("id", ""))
    status = normalize_status(task.get("status", "pending"))
    if task_id == failed_task_id:
        return dependencies_completed(task, tasks)
    if status not in WORKER_READY_STATUSES:
        return False
    return dependencies_completed(task, tasks)


def dependencies_completed(task: dict[str, Any], tasks: list[dict[str, Any]]) -> bool:
    by_id = {str(item.get("id", "")): item for item in tasks}
    for dependency in task.get("depends_on", []):
        dependency_task = by_id.get(str(dependency))
        if not dependency_task:
            return False
        if normalize_status(dependency_task.get("status")) not in TERMINAL_DONE_STATUSES:
            return False
    return True


def task_sort_key(
    task: dict[str, Any],
    tasks: list[dict[str, Any]],
    failed_task_id: str | None = None,
) -> tuple[bool, bool, int, int, str]:
    task_id = str(task.get("id", ""))
    status = normalize_status(task.get("status", "pending"))
    return (
        task_id != failed_task_id,
        status != "in_progress",
        -count_unlocked_tasks(task, tasks),
        task_priority(task),
        task_id,
    )


def count_unlocked_tasks(task: dict[str, Any], tasks: list[dict[str, Any]]) -> int:
    task_id = str(task.get("id", ""))
    completed_ids = {
        str(item.get("id", ""))
        for item in tasks
        if normalize_status(item.get("status")) in TERMINAL_DONE_STATUSES or item is task
    }
    unlocked = 0
    for candidate in tasks:
        candidate_id = str(candidate.get("id", ""))
        if candidate_id == task_id:
            continue
        if normalize_status(candidate.get("status")) not in WORKER_READY_STATUSES:
            continue
        dependencies = {str(item) for item in candidate.get("depends_on", [])}
        if task_id in dependencies and dependencies.issubset(completed_ids):
            unlocked += 1
    return unlocked


def task_priority(task: dict[str, Any]) -> int:
    try:
        return int(task.get("priority", 1000))
    except (TypeError, ValueError):
        return 1000


def normalize_status(status: object) -> str:
    text = str(status or "pending").strip().lower()
    if text == "done":
        return "done"
    return text


def selection_reason(task: dict[str, Any], tasks: list[dict[str, Any]], failed_task_id: str | None = None) -> str:
    task_id = str(task.get("id", ""))
    if task_id == failed_task_id:
        return f"Selected {task_id}: latest verifier failure must be repaired before other work."
    if normalize_status(task.get("status")) == "in_progress":
        return f"Selected {task_id}: existing in-progress task continues before starting new work."
    unlocked = count_unlocked_tasks(task, tasks)
    return (
        f"Selected {task_id}: ready task with priority={task_priority(task)}, "
        f"unlocks={unlocked}, stable_id={task_id}."
    )
