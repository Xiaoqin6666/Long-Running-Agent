from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


DEFAULT_SESSION_BUDGET_TOKENS = 64000
DEFAULT_HANDOFF_THRESHOLD = 0.75


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskState:
    task_id: str
    user_goal: str
    acceptance_criteria: list[str]
    nodes: list[dict[str, Any]]
    iterations: int = 0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    last_action: dict[str, Any] = field(default_factory=dict)
    last_observation: dict[str, Any] = field(default_factory=dict)
    last_verified_at: str | None = None
    evidence_sources: list[dict[str, Any]] = field(default_factory=list)
    acceptance_contracts: list[dict[str, Any]] = field(default_factory=list)
    session_budget_tokens: int = DEFAULT_SESSION_BUDGET_TOKENS
    handoff_threshold: float = DEFAULT_HANDOFF_THRESHOLD
    session_used_tokens: int = 0
    handoff_ready: bool = False
    orchestrator_decision: dict[str, Any] = field(default_factory=dict)
    pending_repair: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskState":
        data.setdefault("evidence_sources", [])
        data.setdefault("acceptance_contracts", [])
        data.setdefault("session_budget_tokens", DEFAULT_SESSION_BUDGET_TOKENS)
        data.setdefault("handoff_threshold", DEFAULT_HANDOFF_THRESHOLD)
        data.setdefault("session_used_tokens", 0)
        data.setdefault("handoff_ready", False)
        data.setdefault("orchestrator_decision", {})
        data.setdefault("pending_repair", {})
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "user_goal": self.user_goal,
            "acceptance_criteria": self.acceptance_criteria,
            "nodes": self.nodes,
            "iterations": self.iterations,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_action": self.last_action,
            "last_observation": self.last_observation,
            "last_verified_at": self.last_verified_at,
            "evidence_sources": self.evidence_sources,
            "acceptance_contracts": self.acceptance_contracts,
            "session_budget_tokens": self.session_budget_tokens,
            "handoff_threshold": self.handoff_threshold,
            "session_used_tokens": self.session_used_tokens,
            "handoff_ready": self.handoff_ready,
            "orchestrator_decision": self.orchestrator_decision,
            "pending_repair": self.pending_repair,
        }

    def summary(self) -> str:
        counts = {}
        for node in self.nodes:
            counts[node["status"]] = counts.get(node["status"], 0) + 1
        parts = [f"{key}={value}" for key, value in sorted(counts.items())]
        return f"iterations={self.iterations}; " + ", ".join(parts)


def create_initial_state(task: str) -> TaskState:
    lower_task = task.lower()
    is_answer_task = any(
        keyword in lower_task
        for keyword in ["inspect", "suggest", "explain", "recommend", "summarize", "分析", "建议", "解释", "总结"]
    )
    if is_answer_task:
        acceptance_criteria = [
            "The agent collects enough repository evidence to support the response.",
            "The final answer names a concrete next implementation step.",
            "State and trace files are written to disk.",
        ]
        nodes = [
            {"id": "T1", "title": "Inspect relevant repository context", "status": "pending", "evidence": []},
            {"id": "T2", "title": "Produce an evidence-based answer", "status": "pending", "evidence": []},
        ]
    else:
        acceptance_criteria = [
            "The agent loop can run from the command line.",
            "State and trace files are written to disk.",
            "A verifier decides whether finish is allowed.",
        ]
        nodes = [
            {"id": "T1", "title": "Initialize explicit task plan", "status": "pending", "evidence": []},
            {"id": "T2", "title": "Collect or produce one useful observation", "status": "pending", "evidence": []},
            {"id": "T3", "title": "Run independent verification", "status": "pending", "evidence": []},
        ]
    return TaskState(
        task_id="current",
        user_goal=task,
        acceptance_criteria=acceptance_criteria,
        nodes=nodes,
    )


def create_initializer_state(
    project_spec: str,
    project_spec_artifact: str = "project_spec.md",
    generated_tasks_artifact: str = "state/generated_tasks.json",
    init_artifact: str = "init.sh",
) -> TaskState:
    acceptance_criteria = [
        f"The project specification is materialized as {project_spec_artifact}.",
        f"A structured task graph is generated at {generated_tasks_artifact}.",
        "The task graph contains executable tasks with ids, dependencies, priorities, statuses, acceptance criteria, expected artifacts, and verification commands.",
        f"An init script is generated at {init_artifact} with repeatable setup or smoke-test commands.",
    ]
    verification_command = (
        "python -c \"import json, pathlib; "
        f"data=json.loads(pathlib.Path('{generated_tasks_artifact}').read_text(encoding='utf-8')); "
        "assert isinstance(data.get('tasks'), list) and data['tasks']; "
        f"assert pathlib.Path('{project_spec_artifact}').is_file(); "
        f"assert pathlib.Path('{init_artifact}').is_file()\""
    )
    nodes = [
        {
            "id": "INIT",
            "title": "Initialize project plan from project specification",
            "status": "in_progress",
            "evidence": [],
            "depends_on": [],
            "priority": 0,
            "expected_artifacts": [
                project_spec_artifact,
                generated_tasks_artifact,
                init_artifact,
            ],
            "implementation_artifacts": [
                generated_tasks_artifact,
                init_artifact,
            ],
            "worker_test_artifacts": [],
            "acceptance_artifacts": [],
            "frozen_acceptance_artifacts": [],
            "test_policy": {
                "worker_tests_mutable_until_contract_freeze": True,
                "acceptance_tests_mutable_by_worker": False,
                "acceptance_test_repair_requires_verifier_approval": True,
            },
            "verification_commands": [
                verification_command
            ],
        }
    ]
    return TaskState(
        task_id="INIT",
        user_goal="INIT: Generate project plan from project_spec.md",
        acceptance_criteria=acceptance_criteria,
        nodes=nodes,
    )
