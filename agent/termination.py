from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


COMPLETED_STATUSES = {"completed", "done"}
ACTIVE_STATUSES = {"pending", "in_progress", "awaiting_verification"}
HUMAN_INTERVENTION_REASONS = {
    "external_api_key_required",
    "unresolved_requirement_conflict",
    "user_decision_required",
    "missing_uninstallable_dependency",
}


@dataclass
class TerminationResult:
    status: str
    summary: str
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "reasons": self.reasons,
            "checks": self.checks,
        }


class ProjectTerminator:
    def __init__(
        self,
        root: Path,
        tasks_path: Path | None = None,
        benchmark_id: str | None = None,
    ) -> None:
        self.root = root
        self.tasks_path = tasks_path or root / "tasks.json"
        self.benchmark_id = benchmark_id

    def evaluate(self, signals: dict[str, Any] | None = None) -> TerminationResult:
        signals = signals or {}
        tasks = self._load_tasks()
        checks = {
            "tasks": evaluate_task_graph(tasks),
            "regression": self._run_regression(),
            "git_clean": self._git_clean(),
            "budget": evaluate_budget(signals),
            "failure_limits": evaluate_failure_limits(signals),
            "human_intervention": evaluate_human_intervention(signals),
        }
        return decide_termination(checks)

    def _load_tasks(self) -> list[dict[str, Any]]:
        path = self.tasks_path
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        tasks = data.get("tasks", [])
        if not isinstance(tasks, list):
            return []
        return [task for task in tasks if isinstance(task, dict)]

    def _run_regression(self) -> dict[str, Any]:
        if self.benchmark_id:
            return {
                "ok": True,
                "skipped": True,
                "summary": "Host Agent regression is outside benchmark scope; task verification provides benchmark evidence.",
            }
        compile_result = run_command(["python", "-m", "compileall", "agent", "eval", "tests"], self.root)
        test_result = run_command(["python", "-m", "unittest", "discover", "-s", "tests"], self.root)
        ok = compile_result["ok"] and test_result["ok"]
        return {"ok": ok, "compile": compile_result, "tests": test_result}

    def _git_clean(self) -> dict[str, Any]:
        if self.benchmark_id:
            return {
                "ok": True,
                "skipped": True,
                "summary": "Host Agent repository cleanliness is outside benchmark scope.",
            }
        result = run_command(["git", "status", "--short"], self.root)
        if not result["ok"]:
            return result
        output = result.get("output", "")
        result["ok"] = output.strip() == ""
        result["summary"] = "Git worktree is clean." if result["ok"] else "Git worktree has uncommitted changes."
        return result


def evaluate_task_graph(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    required = [task for task in tasks if task.get("optional") is not True]
    completed = [task for task in required if normalize_status(task.get("status")) in COMPLETED_STATUSES]
    blocked = [task for task in required if normalize_status(task.get("status")) == "blocked"]
    remaining = [task for task in required if normalize_status(task.get("status")) not in COMPLETED_STATUSES]
    all_remaining_blocked = bool(remaining) and all(normalize_status(task.get("status")) == "blocked" for task in remaining)
    return {
        "ok": bool(required) and len(completed) == len(required) and not blocked,
        "required_count": len(required),
        "completed_count": len(completed),
        "blocked_task_ids": [str(task.get("id", "")) for task in blocked],
        "remaining_task_ids": [str(task.get("id", "")) for task in remaining],
        "all_remaining_blocked": all_remaining_blocked,
    }


def evaluate_budget(signals: dict[str, Any]) -> dict[str, Any]:
    used = int(signals.get("total_tokens_used", 0))
    budget = int(signals.get("total_token_budget", 0))
    exhausted = bool(budget) and used >= budget
    return {"ok": not exhausted, "used": used, "budget": budget, "exhausted": exhausted}


def evaluate_failure_limits(signals: dict[str, Any]) -> dict[str, Any]:
    critical_failures = int(signals.get("critical_task_consecutive_failures", 0))
    max_critical_failures = int(signals.get("max_critical_task_failures", 3))
    environment_failures = int(signals.get("environment_start_failures", 0))
    max_environment_failures = int(signals.get("max_environment_start_failures", 3))
    no_progress_sessions = int(signals.get("no_progress_sessions", 0))
    max_no_progress_sessions = int(signals.get("max_no_progress_sessions", 3))
    exceeded = {
        "critical_task_failures": critical_failures > max_critical_failures,
        "environment_start_failures": environment_failures > max_environment_failures,
        "no_progress_sessions": no_progress_sessions >= max_no_progress_sessions,
    }
    return {
        "ok": not any(exceeded.values()),
        "values": {
            "critical_task_consecutive_failures": critical_failures,
            "environment_start_failures": environment_failures,
            "no_progress_sessions": no_progress_sessions,
        },
        "limits": {
            "max_critical_task_failures": max_critical_failures,
            "max_environment_start_failures": max_environment_failures,
            "max_no_progress_sessions": max_no_progress_sessions,
        },
        "exceeded": exceeded,
    }


def evaluate_human_intervention(signals: dict[str, Any]) -> dict[str, Any]:
    reasons = [reason for reason in HUMAN_INTERVENTION_REASONS if signals.get(reason)]
    return {"ok": not reasons, "reasons": sorted(reasons)}


def decide_termination(checks: dict[str, Any]) -> TerminationResult:
    human_reasons = checks["human_intervention"]["reasons"]
    if human_reasons:
        return TerminationResult(
            "requires_human_intervention",
            "Project paused because autonomous progress requires human input.",
            human_reasons,
            checks,
        )

    failure_reasons = []
    if checks["budget"]["exhausted"]:
        failure_reasons.append("total_token_budget_exhausted")
    if not checks["failure_limits"]["ok"]:
        failure_reasons.extend(
            key for key, value in checks["failure_limits"]["exceeded"].items() if value
        )
    if checks["tasks"]["all_remaining_blocked"]:
        failure_reasons.append("all_remaining_tasks_blocked")
    if failure_reasons:
        return TerminationResult(
            "stopped_with_failure",
            "Project stopped because autonomous failure limits were reached.",
            failure_reasons,
            checks,
        )

    success_checks = [
        checks["tasks"]["ok"],
        checks["regression"]["ok"],
        checks["git_clean"]["ok"],
    ]
    if all(success_checks):
        return TerminationResult("completed", "All project-level completion checks passed.", [], checks)

    reasons = []
    if not checks["tasks"]["ok"]:
        reasons.append("required_tasks_not_completed")
    if not checks["regression"]["ok"]:
        reasons.append("regression_not_passing")
    if not checks["git_clean"]["ok"]:
        reasons.append("worktree_not_clean")
    return TerminationResult("continue_running", "Project is not ready to terminate.", reasons, checks)


def normalize_status(status: object) -> str:
    return str(status or "pending").strip().lower()


def run_command(command: list[str], cwd: Path, timeout: int = 120) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except Exception as exc:
        return {"ok": False, "command": command, "output": str(exc), "returncode": None}
    output = (completed.stdout + completed.stderr).strip()
    return {
        "ok": completed.returncode == 0,
        "command": command,
        "output": output[-8000:],
        "returncode": completed.returncode,
    }
