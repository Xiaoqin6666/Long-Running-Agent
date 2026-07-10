from __future__ import annotations

from pathlib import Path

from agent.planner import TaskState


class ContextBuilder:
    def __init__(self, root: Path, max_chars: int = 12000) -> None:
        self.root = root
        self.max_chars = max_chars

    def build(self, state: TaskState) -> str:
        memory = self._read_optional(self.root / "state" / "memory.md")
        handoff = self._read_optional(self.root / "state" / "handoff.md")
        skills = self._read_skills()
        lines = [
            "You are the decision model inside a long-running coding agent harness.",
            "Return one schema-valid action. The harness owns verification and state transitions.",
            "Runtime environment: Windows PowerShell. Prefer portable Python commands or PowerShell commands.",
            "Tool conventions: use read with target='.' to list a directory; use bash target as the command string.",
            "Use action='answer' when enough evidence has been collected for an inspection, explanation, recommendation, or next-step request.",
            "For action='answer', put the final response in args.answer and cite evidence from observations.",
            "Avoid Unix-only commands such as head, grep, sed, and find unless you know they exist.",
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
            "# Memory",
            memory,
            "",
            "# Handoff",
            handoff,
            "",
            "# Skills",
            skills,
        ]
        text = "\n".join(lines)
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars] + "\n\n[context truncated by harness]"

    def _read_optional(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")[:4000]

    def _read_skills(self) -> str:
        skill_dir = self.root / "state" / "skills"
        if not skill_dir.exists():
            return ""
        chunks = []
        for path in sorted(skill_dir.glob("*.md"))[:5]:
            chunks.append(f"## {path.name}\n{path.read_text(encoding='utf-8')[:2000]}")
        return "\n\n".join(chunks)
