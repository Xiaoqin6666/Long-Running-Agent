from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from agent.planner import TaskState


RECENT_TOOL_OBSERVATION_LIMIT = 3
RECENT_TOOL_OBSERVATION_MAX_CHARS = 8000
RECENT_TOOL_OBSERVATIONS_MAX_CHARS = 18000


class ContextBuilder:
    def __init__(self, root: Path, max_chars: int | None = None, state_dir: Path | None = None) -> None:
        self.root = root
        del max_chars
        self.state_dir = state_dir or root / "state"

    def build(self, state: TaskState) -> str:
        critical = self._critical_context(state)
        working = self._working_context(state)
        reference = self._reference_context(state)
        tail_guard = self._tail_guard_context(state)
        return self._pack_context(critical, working, reference, tail_guard)

    def _pack_context(self, critical: str, working: str, reference: str, tail_guard: str) -> str:
        sections = [section for section in [critical, working, reference, tail_guard] if section.strip()]
        return "\n\n".join(sections)

    def _critical_context(self, state: TaskState) -> str:
        lines = [
            "# Critical Context",
            "This section contains immediate operational state and the benchmark safety boundary.",
            f"- current_task_id: {self._active_task_id(state)}",
            "",
            "## Last Step Summary",
            self._last_step_summary(state),
            "",
            "## Repair Summary",
            self._repair_summary(state),
        ]
        if self.state_dir != self.root / "state":
            lines.extend(
                [
                    "",
                    "## Safety Boundary",
                    "- Benchmark isolation: Git is read-only. Never run git add or git commit, and never try to clean the host Agent repository.",
                ]
            )
        return "\n".join(lines)

    def _last_step_summary(self, state: TaskState) -> str:
        action = self._compact_action(state.last_action)
        observation = state.last_observation if isinstance(state.last_observation, dict) else {}
        ok = observation.get("ok", "unknown")
        summary = str(observation.get("summary", "")).replace("\n", " ").strip()
        data_text = self._compact_observation_data(observation.get("data", {}))
        parts = [f"action={action}", f"ok={ok}"]
        if summary:
            parts.append(f"summary={summary[:280]}")
        if data_text:
            parts.append(f"data={data_text}")
        return "; ".join(parts)

    def _repair_summary(self, state: TaskState) -> str:
        lines: list[str] = []
        pending = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        initializer = state.initializer_repair if isinstance(state.initializer_repair, dict) else {}
        if pending:
            reason = str(pending.get("reason", "pending_repair")).replace("\n", " ")[:80]
            command = str(pending.get("command", "")).replace("\n", " ")[:180]
            targets = ", ".join(self._pending_repair_write_targets(state) or self._pending_repair_targets(state))
            failure = str(pending.get("output", pending.get("summary", ""))).replace("\n", " ")[:180]
            lines.append(
                f"- pending_repair: reason={reason}; command={command or 'none'}; "
                f"targets={targets or 'none'}; failure={failure or 'none'}"
            )
        if initializer:
            candidate = str(initializer.get("candidate_path", ""))
            errors = initializer.get("validation_errors", [])
            error_text = " | ".join(str(error) for error in errors) if isinstance(errors, list) else str(errors)
            lines.append(f"- initializer_repair: candidate={candidate}; first_error={error_text[:180]}")
        return "\n".join(lines) if lines else "No pending repair."

    def _tail_guard_context(self, state: TaskState) -> str:
        observation = state.last_observation if isinstance(state.last_observation, dict) else {}
        pending = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        initializer = state.initializer_repair if isinstance(state.initializer_repair, dict) else {}
        lines = ["# Tail Guard", "Immediate forced action block. Follow this before earlier context."]
        if initializer:
            candidate = str(initializer.get("candidate_path", ""))
            lines.append(f"- Repair INIT candidate: {candidate}. Read once if needed, then edit/write it.")
        elif pending:
            lines.append(f"- {self._required_next_action(state)}")
        else:
            required = self._required_next_action(state)
            if observation.get("ok") is False and required == "No forced next action.":
                lines.append("- Last step failed. Do not repeat it unchanged; act on the failure summary/data.")
            else:
                lines.append(f"- {required}")
        text = "\n".join(lines)
        if len(text) <= 500:
            return text
        return text[:497].rstrip() + "..."

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
            "On resume, handoff.md is authoritative operational context. Read its Resume Instructions, Known Risks And Failed Attempts, and Suggested Next Action before choosing an action.",
            "If handoff.md has a Suggested Next Action and it is a low-risk write, read, bash, contract, or verify step, execute that action first. If you do not execute it, thought_summary must name the concrete blocker.",
            "Do not repeat a failed action unchanged from handoff.md, current state, or the latest observation.",
            "Do not list the same target repeatedly. After one useful listing, act on the evidence; after a missing_path listing, write the required artifact instead of listing again.",
            "If list_files reports a missing target directory for a task whose goal is to create that directory, do not repeat list_files; use write to create the first required file. Write creates parent directories.",
            "If an agreed acceptance contract already exists for the current task, do not submit another contract unless the verifier rejected the existing one.",
            "Do not modify acceptance criteria. Use update_plan only to propose state changes.",
            "Completion requires verifier evidence; do not self-certify completion.",
            "Worker cannot mark tasks completed. Only Verifier PASS followed by Orchestrator state transition may complete a task.",
            "Avoid Unix-only commands such as head, grep, sed, and find unless you know they exist.",
        ]
        if self.state_dir != self.root / "state":
            lines.append(
                "Benchmark isolation: Git is read-only; host Agent repository cleanliness and commits are outside this benchmark."
            )
        return "\n".join(lines)

    def _startup_context(self, state: TaskState | None = None) -> str:
        state_label = self._rel(self.state_dir)
        project_spec = self._read_optional(self.state_dir / "project_spec.md", max_chars=2500)
        root_tasks = ""
        if self.state_dir == self.root / "state":
            project_spec = project_spec or self._read_optional(self.root / "project_spec.md", max_chars=2500)
            root_tasks = self._read_optional(self.root / "tasks.json", max_chars=2500)
        candidate = ""
        if state and isinstance(state.initializer_repair, dict) and state.initializer_repair.get("candidate_path"):
            candidate = self._read_optional(self.state_dir / "rejected_candidates" / "generated_tasks.json", max_chars=5000)
        lines = [
            "# Startup Context",
            f"## {state_label}/project_spec.md",
            project_spec,
            "## repository tasks.json (non-benchmark runs only)",
            root_tasks,
            f"## {state_label}/generated_tasks.json",
            self._read_optional(self.state_dir / "generated_tasks.json", max_chars=2500),
            f"## {state_label}/rejected_candidates/generated_tasks.json (only when initializer repair is active)",
            candidate,
            f"## {state_label}/runtime_tasks.json",
            self._read_optional(self.state_dir / "runtime_tasks.json", max_chars=2500),
            f"## {state_label}/handoff.md focus",
            self._handoff_focus_context(),
            f"## {state_label}/verifier_report.md",
            self._read_optional(self.state_dir / "verifier_report.md", max_chars=2500),
            "## git log",
            self._run_git(["log", "--oneline", "-5"]),
            "## git status",
            self._run_git(["status", "--short", "--branch"]),
        ]
        return "\n".join(lines)

    def _working_context(self, state: TaskState) -> str:
        repair_details = self._pending_repair_context(state)
        recent_tool_observations = self._recent_tool_observations_context(state)
        lines = [
            "# Working Context",
            "Use this to choose the next task-local action.",
            "",
            "# User Goal",
            state.user_goal,
            "",
            "# Acceptance Criteria",
            *[f"- {item}" for item in state.acceptance_criteria],
            "",
            "# Plan",
            *[f"- [{n['status']}] {n['id']}: {n['title']}" for n in state.nodes],
            "",
            self._tool_use_reference_context(),
            "",
            *self._initializer_instruction_lines(state),
            "",
            "# Active Task Artifact Policy",
            *self._artifact_policy_lines(state),
            "",
            "# Evidence Sources",
            *[f"- {item.get('action')}: {item.get('target')} -- {item.get('summary')}" for item in state.evidence_sources[-8:]],
            "",
            "# Acceptance Contracts",
            *[
                f"- {item.get('task_id')}: {item.get('status', 'proposed')} - {item.get('summary', '')}"
                for item in state.acceptance_contracts[-3:]
            ],
            "",
            "# Active Verification Commands",
            *[
                f"- {command}"
                for node in state.nodes
                if node.get("status") in {"in_progress", "pending"}
                for command in self._format_artifacts(node.get("verification_commands", []))
            ],
            "",
            *(["# Repair Details", repair_details, ""] if repair_details else []),
            *(["# Recent Tool Observations", recent_tool_observations, ""] if recent_tool_observations else []),
            "# Just-in-Time Discovery",
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
            "## Recent Step Trace",
            self._recent_step_trace_context(),
        ]
        return "\n".join(lines)

    def _tool_use_reference_context(self) -> str:
        lines = [
            "# Available Tools And Calling Format",
            "Return exactly one JSON object using this schema:",
            '{"thought_summary":"brief non-hidden reasoning","action":"<one action>","target":"<path|command|query|task|empty>","args":{},"expected_observation":"expected result","risk":"low|medium|high"}',
            "Callable actions:",
            "- contract: define verifier agreement before coding; target='<task_id>'; args.task_id, args.summary, args.checks=[...].",
            "- list_files: inspect a directory or file entry; target='<path>'; args.recursive=false, args.limit=200.",
            "- read: bounded file or directory read; target='<path>'; args.start=1, args.end=200.",
            "- search: literal text search; target='<pattern>'; args.path='.'.",
            "- write: create/overwrite/append file; target='<path>'; args.content='<text>', args.mode='create|overwrite|append'.",
            "- edit: exact text replacement; target='<path>'; args.old='<text>', args.new='<text>', args.count=1, args.allow_multiple=false.",
            "- bash: run a needed command from repository root; target='<command>'; args.timeout=30.",
            "- git: status/diff/log/show/branch/add/commit only; target='<git args or git command>'; args.timeout=30.",
            "- verify: ask harness verifier to evaluate current task; target='default'; args={}.",
            "- update_plan: request harness plan update; target='current_task'; args={}.",
            "- answer: final evidence-based response for inspection/explanation tasks; target='' and args.answer='<response>'.",
            "- skill: promote reusable learned procedure only after verifier-confirmed success or evidence-confirmed failure; args.skill_id, args.title, args.body, args.evidence_type, args.evidence.",
            "- finish: project-level termination only after verifier/project completion evidence; target='current_task'; args={}.",
        ]
        return "\n".join(lines)

    def _reference_context(self, state: TaskState) -> str:
        lines = [
            self._startup_context(state),
            self._memory_context(),
        ]
        return "\n\n".join(section for section in lines if section.strip())

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
        del state
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

    def _memory_context(self) -> str:
        memory_index = self._read_optional(self.state_dir / "memory.md")
        hard_memory = self._read_optional(self.state_dir / "hard_memory.md")
        soft_memory = self._read_optional(self.state_dir / "soft_memory.md")
        skills = self._read_skills()
        lines = [
            "# Persistent Context",
            "Persist cross-session information in files rather than relying on chat history.",
            "Hard Memory is evidence-grade. Soft Memory is not evidence; treat it only as a hypothesis or suggestion.",
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

    def _handoff_focus_context(self) -> str:
        text = self._read_optional(self.state_dir / "handoff.md", max_chars=12000)
        if not text.strip():
            return "No handoff.md available."
        wanted = {
            "## 10. Last Step Summary",
            "## 10a. Pending Repair",
            "## 10b. Initializer Repair",
            "## 12. Known Risks And Failed Attempts",
            "## 14. Resume Instructions",
            "## 15. Suggested Next Action",
        }
        sections: list[str] = []
        current: list[str] = []
        keep = False
        for line in text.splitlines():
            if line.startswith("## "):
                if keep and current:
                    sections.append("\n".join(current).strip())
                current = [line]
                keep = line.strip() in wanted
                continue
            if keep:
                current.append(line)
        if keep and current:
            sections.append("\n".join(current).strip())
        if not sections:
            return text[:1500]
        return "\n\n".join(sections)[:3000]

    def _pending_repair_context(self, state: TaskState) -> str:
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        initializer = state.initializer_repair if isinstance(state.initializer_repair, dict) else {}
        lines: list[str] = []
        if repair:
            lines.extend(
                [
                    f"- reason: {repair.get('reason', 'pending_repair')}",
                    f"- command: {str(repair.get('command', '')).replace(chr(10), ' ')[:1000] or 'none'}",
                    f"- summary: {str(repair.get('summary', ''))[:1000] or 'none'}",
                    f"- failure_output: {str(repair.get('output', ''))[:2000] or 'none'}",
                    f"- diagnostic_targets: {repair.get('targets', [])}",
                    f"- repair_targets: {self._pending_repair_write_targets(state)}",
                    f"- required_reads: {repair.get('required_reads', [])}",
                    f"- completed_reads: {repair.get('read_targets', [])}",
                    f"- repaired_targets: {repair.get('repaired_targets', [])}",
                ]
            )
            inferred = self._inferred_pending_repair_targets(state)
            if inferred:
                lines.append(f"- inferred_import_targets: {inferred}")
        if initializer:
            errors = initializer.get("validation_errors", [])
            lines.extend(
                [
                    f"- initializer_candidate: {initializer.get('candidate_path', '')}",
                    f"- initializer_validation_errors: {errors if isinstance(errors, list) else [str(errors)]}",
                ]
            )
        return "\n".join(lines)

    def _recent_tool_observations_context(self, state: TaskState) -> str:
        records: list[tuple[dict[str, object], dict[str, object]]] = []
        seen_targets: set[str] = set()

        def consider(action: object, observation: object) -> None:
            if len(records) >= RECENT_TOOL_OBSERVATION_LIMIT:
                return
            if not isinstance(action, dict) or not isinstance(observation, dict):
                return
            action_name = str(action.get("action", ""))
            target = str(action.get("target", "")).replace("\\", "/").strip().rstrip("/")
            if not target:
                return
            if action_name in {"write", "edit"} and observation.get("ok") is True:
                seen_targets.add(target)
                return
            if action_name not in {"read", "list_files", "search"} or observation.get("ok") is not True:
                return
            if target in seen_targets:
                return
            seen_targets.add(target)
            records.append((action, observation))

        consider(state.last_action, state.last_observation)
        trace_dir = self.state_dir / "traces"
        if trace_dir.exists() and len(records) < RECENT_TOOL_OBSERVATION_LIMIT:
            trace_paths = sorted(
                trace_dir.glob("run_*.jsonl"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for trace_path in trace_paths[:5]:
                for event in reversed(self._load_trace_events(trace_path)):
                    consider(event.get("action"), event.get("observation"))
                    if len(records) >= RECENT_TOOL_OBSERVATION_LIMIT:
                        break
                if len(records) >= RECENT_TOOL_OBSERVATION_LIMIT:
                    break

        chunks: list[str] = []
        used = 0
        for index, (action, observation) in enumerate(records, start=1):
            heading = f"## Observation {index}{' (most recent)' if index == 1 else ''}\n"
            separator = "\n\n" if chunks else ""
            remaining = RECENT_TOOL_OBSERVATIONS_MAX_CHARS - used - len(separator) - len(heading)
            if remaining < 200:
                break
            block = self._format_tool_observation(
                action,
                observation,
                max_chars=min(RECENT_TOOL_OBSERVATION_MAX_CHARS, remaining),
            )
            chunks.append(heading + block)
            used += len(separator) + len(heading) + len(block)
        return "\n\n".join(chunks)

    def _format_tool_observation(
        self,
        action: dict[str, object],
        observation: dict[str, object],
        *,
        max_chars: int,
    ) -> str:
        action_name = str(action.get("action", ""))
        data = observation.get("data", {})
        if not isinstance(data, dict):
            return ""

        target = str(action.get("target", ""))
        metadata = [f"- action: {action_name}", f"- target: {target}", f"- ok: {observation.get('ok')}"]
        if action_name == "read":
            metadata.append(f"- range: {data.get('start', '?')}-{data.get('end', '?')}")
            payload = str(data.get("content", ""))
        elif action_name == "list_files":
            payload = json.dumps(data.get("entries", []), ensure_ascii=False, indent=2)
        else:
            payload = json.dumps(data.get("matches", []), ensure_ascii=False, indent=2)

        prefix = "\n".join([*metadata, "--- BEGIN TOOL OUTPUT ---", ""])
        suffix = "\n--- END TOOL OUTPUT ---"
        truncation = "\n[tool output truncated]"
        available = max(0, max_chars - len(prefix) - len(suffix))
        if len(payload) > available:
            payload = payload[: max(0, available - len(truncation))].rstrip() + truncation
        return prefix + payload + suffix

    def _inferred_pending_repair_targets(self, state: TaskState) -> list[str]:
        repair = state.pending_repair if isinstance(state.pending_repair, dict) else {}
        command = str(repair.get("command", ""))
        output = str(repair.get("output", ""))
        combined = f"{command}\n{output}".replace("\\", "/")
        if "ModuleNotFoundError" not in combined and "No module named" not in combined:
            return []
        workspace_root = self._workspace_root_from_repair(command, state)
        if not workspace_root:
            return []
        missing_roots = {
            match.group(1).split(".", 1)[0]
            for match in re.finditer(r"No module named ['\"]([^'\"]+)['\"]", combined)
        }
        targets: list[str] = []
        for match in re.finditer(r"\bfrom\s+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\s+import\b", command):
            module = match.group(1)
            parts = module.split(".")
            if missing_roots and parts[0] not in missing_roots:
                continue
            for target in (
                f"{workspace_root}/{'/'.join(parts)}.py",
                f"{workspace_root}/{parts[0]}/__init__.py",
            ):
                if target not in targets:
                    targets.append(target)
        return targets

    def _workspace_root_from_repair(self, command: str, state: TaskState) -> str | None:
        normalized = command.replace("\\", "/")
        for pattern in (
            r"sys\.path\.insert\(0,\s*['\"]([^'\"]*workspace)['\"]\)",
            r"PYTHONPATH\s*=\s*['\"]?([^'\"\s;]*workspace)",
        ):
            match = re.search(pattern, normalized)
            if match:
                return match.group(1).rstrip("/")
        for node in state.nodes:
            if node.get("status") not in {"in_progress", "pending"}:
                continue
            for artifact in self._format_artifacts(node.get("expected_artifacts", [])):
                normalized_artifact = artifact.replace("\\", "/").rstrip("/")
                marker = "/workspace/"
                if marker in normalized_artifact:
                    return normalized_artifact[: normalized_artifact.index(marker) + len("/workspace")]
                if normalized_artifact.startswith("workspace/"):
                    return "workspace"
        return None

    def _recent_step_trace_context(self, limit: int = 8) -> str:
        trace_dir = self.state_dir / "traces"
        if not trace_dir.exists():
            return "No trace events available."
        trace_paths = sorted(trace_dir.glob("run_*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
        events: list[dict[str, object]] = []
        for trace_path in trace_paths[:3]:
            events = self._load_trace_events(trace_path)
            if events:
                break
        if not events:
            return "No trace events available."
        lines = []
        for event in events[-limit:]:
            step = event.get("step", "?")
            action = event.get("action", {})
            observation = event.get("observation", {})
            if not isinstance(action, dict):
                action = {}
            if not isinstance(observation, dict):
                observation = {}
            action_text = self._compact_action(action)
            ok = observation.get("ok")
            summary = str(observation.get("summary", "")).replace("\n", " ")[:240]
            data_text = self._compact_observation_data(observation.get("data", {}))
            if data_text:
                data_text = f"; data={data_text}"
            lines.append(
                f"- step {step}: action={action_text}; ok={ok}; summary={summary}{data_text}"
            )
        return "\n".join(lines)

    def _load_trace_events(self, trace_path: Path) -> list[dict[str, object]]:
        try:
            text = trace_path.read_text(encoding="utf-8")
        except OSError:
            return []
        decoder = json.JSONDecoder()
        index = 0
        events: list[dict[str, object]] = []
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

    def _compact_action(self, action: object) -> str:
        if not isinstance(action, dict):
            return "unknown"
        name = str(action.get("action", ""))
        target = str(action.get("target", "")).replace("\n", " ")[:180]
        return f"{name} {target}".strip()

    def _compact_observation_data(self, data: object) -> str:
        if not isinstance(data, dict):
            return ""
        keys = [
            "missing_path",
            "required_action",
            "target",
            "candidate_path",
            "initializer_validation_errors",
            "initializer_error_repeat_count",
            "command",
            "output",
            "suggested_command",
            "repair_hint",
        ]
        compact: list[str] = []
        for key in keys:
            if key not in data:
                continue
            value = data[key]
            if key == "output":
                value = str(value).replace("\n", " | ")[:220]
            elif isinstance(value, list):
                value = [str(item)[:120] for item in value[:3]]
            else:
                value = str(value).replace("\n", " ")[:160]
            compact.append(f"{key}={value}")
        return "; ".join(compact)

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
            f"- {self._rel(self.state_dir / 'init.sh')} is the run-local initializer entrypoint. It must be a POSIX shell script beginning with '#!/usr/bin/env sh' and 'set -eu'; it may invoke Python commands but must not contain Python source code.",
            "Each generated task should include: id, title, priority, depends_on, status, acceptance_criteria, expected_artifacts, implementation_artifacts when applicable, worker_test_artifacts when applicable, acceptance_artifacts when applicable, frozen_acceptance_artifacts when applicable, test_policy when tests are involved, and verification_commands.",
            "Verification commands run from the repository root. Commands that import or invoke project modules under the workspace must explicitly configure sys.path, PYTHONPATH, or subprocess cwd, including nested subprocess calls.",
            "priority MUST be an integer. Lower numbers are higher priority; use 1, 2, 3, ... and never strings such as 'high' or 'medium'.",
            "Minimal complete task example:",
            '{"id":"T1","title":"Implement feature","priority":1,"depends_on":[],"status":"pending","acceptance_criteria":["Behavior is verified."],"expected_artifacts":["<workspace>/pkg/feature.py"],"implementation_artifacts":["<workspace>/pkg/feature.py"],"worker_test_artifacts":[],"acceptance_artifacts":[],"frozen_acceptance_artifacts":[],"test_policy":{"acceptance_tests_mutable_by_worker":false,"acceptance_test_repair_requires_verifier_approval":true},"verification_commands":["python -m unittest discover -s <workspace>/tests"]}',
            "Implementation tasks must declare non-empty implementation_artifacts, and every owned implementation/test/acceptance artifact must also appear in expected_artifacts.",
            "Respect dependency constraints from project_spec.md (for example, standard-library-only means no pytest or package installation). Verification commands must be substantive and must not be placeholders such as bare echo, TODO, or 'not implemented'.",
            "INIT does not require an acceptance contract.",
            "During INIT, write or edit only the three initializer artifacts listed above. Do not create application code, tests, skeleton files, or workspace files.",
            "The repository-root init.sh belongs to the Long-Running Agent harness and must not be modified by a benchmark INIT.",
            "Any application artifact in the generated task graph must be under the workspace path required by project_spec.md.",
            "Do not use answer or finish during INIT.",
            "After writing initializer artifacts, run the initializer verification command from the active task, then use verify. Only Verifier PASS completes INIT and allows Orchestrator to schedule the first Worker task.",
        ]

    def _required_next_action(self, state: TaskState) -> str:
        initializer_repair = state.initializer_repair if isinstance(state.initializer_repair, dict) else {}
        if initializer_repair:
            candidate = str(initializer_repair.get("candidate_path", ""))
            errors = initializer_repair.get("validation_errors", [])
            error_text = " | ".join(str(error) for error in errors) if isinstance(errors, list) else str(errors)
            return (
                f"Repair the saved INIT candidate at '{candidate}' instead of regenerating the task graph. "
                "Use read once if its exact content is needed, then use edit or write on that candidate. "
                f"Validation errors: {error_text[:1200]}"
            )
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
            if not repair_targets:
                command = str(state.pending_repair.get("command", ""))
                return (
                    "The last acceptance or verification command failed, but no mutable repair target is available. "
                    "Do not attempt to edit a frozen or contract-owned test artifact. "
                    f"Repair or replace the acceptance command, then rerun bash target='{command}'. "
                    f"Failure excerpt: {excerpt}"
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
            "- Rule: implementation artifacts are normal repair targets; worker tests remain mutable unless explicitly listed in frozen_acceptance_artifacts.",
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
        return []

    def _active_task_frozen_acceptance_artifacts(self, state: TaskState) -> list[str]:
        if self._active_task_has_key(state, "frozen_acceptance_artifacts"):
            return self._active_task_artifacts_by_key(state, "frozen_acceptance_artifacts")
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
                        "acceptance_tests_mutable_by_worker": False,
                        "acceptance_test_repair_requires_verifier_approval": True,
                    }
                    merged.update(policy)
                    return merged
        return {
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
        return normalized in worker_tests

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
        if state.task_id == "INIT":
            return "INIT"
        for node in state.nodes:
            if node.get("status") in {"in_progress", "pending"}:
                return str(node.get("id", "current"))
        return "current"
