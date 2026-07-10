from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


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
    session_budget_tokens: int = 16000
    handoff_threshold: float = 0.7
    session_used_tokens: int = 0
    handoff_ready: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskState":
        data.setdefault("evidence_sources", [])
        data.setdefault("acceptance_contracts", [])
        data.setdefault("session_budget_tokens", 16000)
        data.setdefault("handoff_threshold", 0.7)
        data.setdefault("session_used_tokens", 0)
        data.setdefault("handoff_ready", False)
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
