from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.context import ContextBuilder
from agent.llm import create_decision_maker
from agent.planner import TaskState, create_initial_state
from agent.tools import BashTool, ReadTool, SearchTool, ToolResult, WriteTool
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
    ) -> None:
        self.root = root
        self.task = task
        self.max_steps = max_steps
        self.provider = provider
        self.resume = resume
        self.state_dir = root / "state"
        self.trace_dir = self.state_dir / "traces"
        self.state_path = self.state_dir / "current_task.json"
        self.memory_path = self.state_dir / "memory.md"
        self.handoff_path = self.state_dir / "handoff.md"
        self.trace_path = self.trace_dir / self._trace_name()
        self.context_builder = ContextBuilder(root)
        self.decision_maker = create_decision_maker(provider)
        self.verifier = Verifier(root)
        self.tools = {
            "bash": BashTool(root),
            "read": ReadTool(root),
            "search": SearchTool(root),
            "write": WriteTool(root),
        }

    def run(self) -> RunResult:
        self._ensure_state_files()
        state = self._load_or_create_state()
        steps = 0
        completed = False
        message = "Reached max steps before completion."

        for step in range(1, self.max_steps + 1):
            steps = step
            context = self.context_builder.build(state)
            try:
                action = self.decision_maker.next_action(context, state)
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
            self._append_trace(step, action, observation, state)
            self._write_state(state)

            if action["action"] in {"answer", "finish"} and observation.ok:
                completed = True
                message = observation.data.get("answer", observation.summary)
                break
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
            return TaskState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))
        return create_initial_state(self.task)

    def _execute_action(self, action: dict[str, Any], state: TaskState) -> ToolResult:
        name = action.get("action")
        if name == "answer":
            evidence_ok, evidence_message = self._check_answer_evidence(state)
            if not evidence_ok:
                return ToolResult(False, evidence_message, {"missing_evidence": True})
            answer = str(action.get("args", {}).get("answer") or action.get("target") or "")
            if not answer.strip():
                return ToolResult(False, "Answer rejected because it was empty.", {})
            return ToolResult(True, "Final answer produced.", {"answer": answer})
        if name == "contract":
            return self._validate_contract_action(action)
        if name == "write" and state.handoff_ready:
            return ToolResult(
                False,
                "Write rejected: session is past handoff threshold; generate handoff instead of starting new edits.",
                {"handoff_ready": True},
            )
        if name == "write" and not self._has_contract_for_active_task(state):
            return ToolResult(
                False,
                "Write rejected: create an acceptance contract with the verifier before generating code.",
                {"missing_contract": True},
            )
        if name == "update_plan":
            return ToolResult(True, "Plan updated by harness.", {"target": action.get("target")})
        if name == "verify":
            return self.verifier.run(action.get("target", "default"), state)
        if name == "finish":
            verification = self.verifier.run("finish", state)
            if verification.ok:
                return ToolResult(True, "All acceptance checks passed. Task can finish.", verification.data)
            return ToolResult(False, "Finish rejected because verification failed.", verification.data)
        tool = self.tools.get(str(name))
        if not tool:
            return ToolResult(False, f"Unknown action: {name}", {})
        return tool.run(action)

    def _update_state(self, state: TaskState, action: dict[str, Any], observation: ToolResult) -> None:
        state.iterations += 1
        state.updated_at = utc_now()
        state.last_action = action
        state.last_observation = observation.to_dict()

        name = action.get("action")
        if name == "contract" and observation.ok:
            contract = dict(observation.data["contract"])
            contract.setdefault("status", "agreed")
            state.acceptance_contracts.append(contract)
        elif name == "answer" and observation.ok:
            for node in state.nodes:
                if node["status"] != "done":
                    node["status"] = "done"
                    node["evidence"].append(observation.summary)
        elif name == "update_plan" and state.nodes:
            state.nodes[0]["status"] = "done"
            state.nodes[0]["evidence"].append("initialized plan")
        elif name in {"read", "search", "bash", "write"} and observation.ok and len(state.nodes) > 1:
            state.evidence_sources.append(
                {
                    "action": name,
                    "target": str(action.get("target", "")),
                    "summary": observation.summary,
                }
            )
            state.nodes[1]["status"] = "done"
            state.nodes[1]["evidence"].append(observation.summary)
        elif name in {"read", "search", "bash", "write", "protocol_error"} and not observation.ok:
            state.last_observation["counts_as_progress"] = False
        elif name == "verify" and observation.ok and len(state.nodes) > 2:
            state.nodes[2]["status"] = "done"
            state.nodes[2]["evidence"].append(observation.summary)
            state.last_verified_at = utc_now()
        elif name == "finish" and observation.ok:
            for node in state.nodes:
                if node["status"] != "done":
                    node["status"] = "done"
                    node["evidence"].append("finish verifier passed")

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

    def _has_contract_for_active_task(self, state: TaskState) -> bool:
        active = self._active_task_id(state)
        return any(
            item.get("task_id") in {active, "current"} and item.get("status") == "agreed"
            for item in state.acceptance_contracts
        )

    def _active_task_id(self, state: TaskState) -> str:
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
    ) -> None:
        event = {
            "step": step,
            "time": utc_now(),
            "action": action,
            "observation": observation.to_dict(),
            "state_summary": state.summary(),
        }
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _write_handoff(self, state: TaskState) -> None:
        active_node = self._active_node(state)
        completed = [node for node in state.nodes if node.get("status") == "done"]
        pending = [node for node in state.nodes if node.get("status") != "done"]
        contracts = state.acceptance_contracts[-5:]
        evidence = state.evidence_sources[-20:]
        hard_memory = self._read_state_file("hard_memory.md", max_chars=3000)
        soft_memory = self._read_state_file("soft_memory.md", max_chars=3000)
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
            "## 4. Completed Tasks",
            *[self._format_node(node) for node in completed],
            "",
            "## 5. Pending Or Blocked Tasks",
            *[self._format_node(node) for node in pending],
            "",
            "## 6. Acceptance Contracts",
            *[self._format_contract(contract) for contract in contracts],
            "",
            "## 7. Evidence Sources",
            *[f"- {item.get('action')}: {item.get('target')} -- {item.get('summary')}" for item in evidence],
            "",
            "## 8. Hard Memory",
            hard_memory,
            "",
            "## 9. Soft Memory",
            soft_memory,
            "",
            "## 10. Last Action",
            json.dumps(state.last_action, ensure_ascii=False, indent=2),
            "",
            "## 11. Last Observation",
            json.dumps(state.last_observation, ensure_ascii=False, indent=2),
            "",
            "## 12. Verification Status",
            f"- last_verified_at: {state.last_verified_at}",
            "- deterministic verifier: run `python -m unittest discover -s tests` and `python -m compileall agent eval tests`.",
            "",
            "## 13. Known Risks And Failed Attempts",
            "- Review trace for failed observations and protocol errors before resuming.",
            "- Do not repeat failed actions unchanged.",
            "- Do not start new large edits until the next worker session has rebuilt context from this handoff.",
            "- Treat Soft Memory as hypotheses, not facts.",
            "",
            "## 14. Current State Summary",
            state.summary(),
            "",
            "## 15. Resume Instructions",
            "1. Read this handoff first.",
            "2. Load `state/current_task.json`, `tasks.json`, and relevant source files.",
            "3. Continue the active task only after checking the acceptance contract.",
            "4. If no contract exists for a coding task, create one before writing code.",
            "5. Prefer verification or small repair actions before new feature work.",
            "6. Promote Soft Memory to Hard Memory only after verification.",
            "",
            "## 16. Suggested Next Action",
            self._suggest_next_action(state),
            "",
        ]
        self.handoff_path.write_text("\n".join(lines), encoding="utf-8")

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

    def _suggest_next_action(self, state: TaskState) -> str:
        active = self._active_node(state)
        if not active:
            return "Run verifier and finish if all acceptance criteria pass."
        if not self._has_contract_for_active_task(state):
            return f"Create an acceptance contract for {active.get('id')} before writing code."
        return f"Resume {active.get('id')} with a small evidence-backed action, then verify."

    def _ensure_state_files(self) -> None:
        self.state_dir.mkdir(exist_ok=True)
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

    @staticmethod
    def _trace_name() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"run_{stamp}.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
