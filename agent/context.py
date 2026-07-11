from __future__ import annotations

import subprocess
from pathlib import Path

from agent.planner import TaskState


class ContextBuilder:
    def __init__(self, root: Path, max_chars: int = 12000) -> None:
        self.root = root
        self.max_chars = max_chars

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
            "Runtime environment: Windows PowerShell. Prefer portable Python commands or PowerShell commands.",
            "Tool conventions: use list_files for directories; use read for file ranges; use bash target as the command string.",
            "Use action='answer' when enough evidence has been collected for an inspection, explanation, recommendation, or next-step request.",
            "For action='answer', put the final response in args.answer and cite evidence from observations.",
            "Do not modify acceptance criteria. Use update_plan only to propose state changes.",
            "Completion requires verifier evidence; do not self-certify completion.",
            "Worker cannot mark tasks completed. Only Verifier PASS followed by Orchestrator state transition may complete a task.",
            "Avoid Unix-only commands such as head, grep, sed, and find unless you know they exist.",
        ]
        return "\n".join(lines)

    def _startup_context(self) -> str:
        lines = [
            "# Startup Context",
            "## project_spec.md",
            self._read_optional(self.root / "project_spec.md", max_chars=2500),
            "## tasks.json",
            self._read_optional(self.root / "tasks.json", max_chars=2500),
            "## state/handoff.md",
            self._read_optional(self.root / "state" / "handoff.md", max_chars=2500),
            "## state/verifier_report.md",
            self._read_optional(self.root / "state" / "verifier_report.md", max_chars=2500),
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
        memory_index = self._read_optional(self.root / "state" / "memory.md")
        hard_memory = self._read_optional(self.root / "state" / "hard_memory.md")
        soft_memory = self._read_optional(self.root / "state" / "soft_memory.md")
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

    def _read_optional(self, path: Path, max_chars: int = 4000) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")[:max_chars]

    def _read_skills(self) -> str:
        skill_dir = self.root / "state" / "skills"
        if not skill_dir.exists():
            return ""
        chunks = []
        for path in sorted(skill_dir.glob("*.md"))[:5]:
            chunks.append(f"## {path.name}\n{path.read_text(encoding='utf-8')[:2000]}")
        return "\n\n".join(chunks)

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
