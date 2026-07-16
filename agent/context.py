from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from agent.planner import TaskState
from agent.skills import parse_skill, skill_catalog


RECENT_TOOL_OBSERVATION_LIMIT = 5
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
        return self._pack_context(critical, working, reference)

    def _pack_context(self, critical: str, working: str, reference: str) -> str:
        sections = [section for section in [critical, working, reference] if section.strip()]
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

    def _always_on_context(self, state: TaskState) -> str:
        lines = [
            "# Always-on Context",
            "You are the decision model inside a long-running coding agent harness.",
            "Return one schema-valid action. The harness owns verification and state transitions.",
            f"Current task id: {self._active_task_id(state)}",
            f"Orchestrator decision: {state.orchestrator_decision}",
            f"Required next action: {self._required_next_action(state)}",
            "Runtime environment: Windows PowerShell. Prefer portable Python commands or PowerShell commands.",
            "Tool conventions: use list_files for directories; use search as grep before reading when you know an id, symbol, error text, or filename; then use read with args.query to fetch matching code. If read returns has_more=true, continue only when the needed content is clearly beyond the returned window; use bash target as the command string.",
            "Use action='answer' when enough evidence has been collected for an inspection, explanation, recommendation, or next-step request.",
            "For action='answer', put the final response in args.answer and cite evidence from observations.",
            "Generated-task acceptance contracts freeze semantic frozen_requirements; do not weaken them. If the execution command is wrong, action='contract' may update only verification_procedure for the same frozen_requirements.",
            "For ad-hoc tasks without a generated task graph, action='contract' args must include task_id, summary, frozen_requirements, and verification_procedure or checks.",
            "On resume, handoff.md is authoritative operational context. Read its Resume Instructions, Known Risks And Failed Attempts, and Suggested Next Action before choosing an action.",
            "If handoff.md has a Suggested Next Action and it is a low-risk write, read, bash, contract, or verify step, execute that action first. If you do not execute it, thought_summary must name the concrete blocker.",
            "Do not repeat a failed action unchanged from handoff.md, current state, or the latest observation.",
            "Do not list the same target repeatedly. After one useful listing, act on the evidence; after a missing_path listing, write the required artifact instead of listing again.",
            "If list_files reports a missing target directory for a task whose goal is to create that directory, do not repeat list_files; use write to create the first required file. Write creates parent directories.",
            "If an agreed acceptance contract already exists for the current task, do not submit another contract unless the verifier rejected it or the verification_procedure itself needs correction.",
            "Do not modify acceptance criteria. Use update_plan only to propose state changes.",
            "Completion requires verifier evidence; do not self-certify completion.",
            "Worker cannot mark tasks completed. Only Verifier PASS followed by Orchestrator state transition may complete a task.",
            "Skill selection: compare the current task or error with Available Skills descriptions. Load a Skill only for a clear match, only once per unchanged version, and only when no forced next action has priority.",
            "Do not load a Skill for keyword overlap alone, when its description excludes the current case, or when the next low-cost action is already clear.",
            "Avoid Unix-only commands such as head, grep, sed, and find unless you know they exist.",
        ]
        if self.state_dir != self.root / "state":
            lines.append(
                "Benchmark isolation: Git is read-only; host Agent repository cleanliness and commits are outside this benchmark."
            )
        return "\n".join(lines)

    def _session_startup_context(self, state: TaskState | None = None) -> str:
        state_label = self._rel(self.state_dir)
        project_spec = self._read_optional(self.state_dir / "project_spec.md", max_chars=2500)
        root_tasks_overview = ""
        if self.state_dir == self.root / "state":
            project_spec = project_spec or self._read_optional(self.root / "project_spec.md", max_chars=2500)
            root_tasks_overview = self._task_graph_overview(self.root / "tasks.json", state)
        candidate_path = ""
        if state and isinstance(state.initializer_repair, dict) and state.initializer_repair.get("candidate_path"):
            candidate_path = self._rel(self.state_dir / "rejected_candidates" / "generated_tasks.json")
        lines = [
            "# Session Startup Context",
            "Included only on the first model call of a Worker session.",
            f"## {state_label}/project_spec.md",
            project_spec,
            "## Task Graph Overview",
            *(["### repository tasks.json (non-benchmark runs only)", root_tasks_overview] if root_tasks_overview else []),
            f"### {state_label}/generated_tasks.json",
            self._task_graph_overview(self.state_dir / "generated_tasks.json", state),
            f"### {state_label}/runtime_tasks.json",
            self._task_graph_overview(self.state_dir / "runtime_tasks.json", state),
            f"### {state_label}/rejected_candidates/generated_tasks.json",
            candidate_path or "No initializer repair candidate.",
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

    def _incremental_reference_context(self, state: TaskState) -> str:
        state_label = self._rel(self.state_dir)
        verifier = ""
        if state.last_verified_at or state.last_action.get("action") == "verify":
            verifier = self._read_optional(self.state_dir / "verifier_report.md", max_chars=1200)
        handoff = ""
        if state.handoff_ready or self._read_optional(self.state_dir / "handoff.md", max_chars=1):
            handoff = self._handoff_focus_context()
        generated_summary = self._task_graph_summary(self.state_dir / "generated_tasks.json")
        runtime_summary = self._task_graph_summary(self.state_dir / "runtime_tasks.json")
        git_status = self._run_git(["status", "--short", "--branch"])
        lines = [
            "# Incremental Reference Context",
            "Subsequent-call reference state. Full startup files are omitted; this block carries compact task, verifier, handoff, and git signals.",
            f"- current_task_id: {self._active_task_id(state)}",
            f"- state_iterations: {state.iterations}",
            f"- last_verified_at: {state.last_verified_at or 'never'}",
            f"- handoff_ready: {state.handoff_ready}",
            f"- orchestrator_decision: {state.orchestrator_decision}",
            f"- generated_tasks_summary: {generated_summary}",
            f"- runtime_tasks_summary: {runtime_summary}",
            f"## {state_label}/handoff.md focus",
            handoff or "No new handoff state.",
            f"## {state_label}/verifier_report.md",
            verifier or "No new verifier result.",
            "## git status",
            git_status,
        ]
        return "\n".join(lines)

    def _working_context(self, state: TaskState) -> str:
        repair_details = self._pending_repair_context(state)
        recent_tool_observations = self._recent_tool_observations_context(state)
        lines = [
            "# Working Context",
            "Use this to choose the next task-local action.",
            "",
            "# Active Task",
            state.user_goal,
            "",
            "# User Conversation",
            self._conversation_context(state),
            "",
            f"# Interaction Mode\n{state.interaction_mode or 'non-interactive'}",
            "",
            "# Acceptance Criteria",
            *[f"- {item}" for item in state.acceptance_criteria],
            "",
            self._skill_reflection_context(state),
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
            *[self._format_contract(item) for item in state.acceptance_contracts[-3:]],
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
            "2. search/grep relevant ids, symbols, filenames, or error strings before reading; for example search target='T7' or search target='initializer_validation_errors' args={'path': '<candidate-or-dir>'};",
            "3. read matching code with args.query before falling back to explicit ranges; for example read target='<file>' args={'query': '\"id\": \"T7\"'};",
            "4. read corresponding tests;",
            "5. use errors or verifier output to guide the next search.",
            "PowerShell/Python examples:",
            "- list_files target='agent'",
            "- search target='create_issue' args={'path': 'agent'}",
            "- search target='initializer_validation_errors' args={'path': 'state/benchmarks/issue_tracker'}",
            "- read target='agent/loop.py' args={'query': 'def _execute_action'}",
            "- if read returns has_more=true, continue with data.next_read args only when the needed content was not found; otherwise act on the returned evidence.",
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
            "- contract: ad-hoc tasks create an agreement with args.task_id, args.summary, args.frozen_requirements=[...], args.verification_procedure={command:'...' or commands:[...]}; generated tasks may only update verification_procedure while preserving frozen_requirements exactly.",
            "- list_files: inspect a directory or file entry; target='<path>'; args.recursive=false, args.limit=200.",
            "- search: grep-style literal text search; target='<known id|symbol|error text|filename>'; args.path='.'. Use this before read when locating T7, validation errors, functions, classes, or filenames.",
            "- read: targeted file read; target='<path>'; prefer args.query='<literal symbol/text>' after search/grep to return matching code. If has_more=true, continue with returned data.next_read args only when the needed content is beyond the returned window. Explicit args.start/args.end are allowed only for known line ranges.",
            "- write: create/overwrite/append file; target='<path>'; args.content='<text>', args.mode='create|overwrite|append'.",
            "- edit: exact text replacement; target='<path>'; args.old='<text>', args.new='<text>', args.count=1, args.allow_multiple=false.",
            "- bash: run a needed command from repository root; target='<command>'; args.timeout=30.",
            "- git: status/diff/log/show/branch/add/commit only; target='<git args or git command>'; args.timeout=30.",
            "- verify: ask harness verifier to evaluate current task; target='default'; args={}.",
            "- update_plan: request harness plan update; target='current_task'; args={}.",
            "- answer: final evidence-based response for inspection/explanation tasks; target='' and args.answer='<response>'.",
            "- load_skill: load one relevant Skill by metadata name before applying it.",
            "- save_skill: submit a reusable procedure candidate; args.name, description, instruction, optional examples, evidence_type, evidence_refs=[{type:'verifier_report',report_id:'VR-...',task_id:'...'} or {type:'trace',path:'state/traces/...',step:N,task_id:'...'}]. Prefer immutable report_id references. Free-text evidence is rejected.",
            "- dismiss_skill: decline the current Pending Skill Reflection; target='<report_id>'; args.reason='<why this is not reusable>'.",
            "- finish: project-level termination only after verifier/project completion evidence; target='current_task'; args={}.",
        ]
        return "\n".join(lines)

    def _conversation_context(self, state: TaskState) -> str:
        messages = state.conversation_messages if isinstance(state.conversation_messages, list) else []
        rendered: list[str] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            label = "User" if role == "user" else "Agent"
            if role == "user" and not any(
                isinstance(later, dict) and str(later.get("role", "")).strip().lower() == "user"
                for later in messages[index + 1 :]
            ):
                label = "Latest User Message"
            rendered.append(f"{label}: {content}")
        text = "\n\n".join(rendered)
        if len(text) > 8000:
            text = text[-8000:]
        return text or "No interactive user conversation is attached to this run."

    def _format_contract(self, contract: dict[str, object]) -> str:
        requirements = contract.get("frozen_requirements", contract.get("required_evidence", []))
        requirement_text = "; ".join(str(item) for item in requirements) if isinstance(requirements, list) else str(requirements)
        procedure = contract.get("verification_procedure")
        procedure_commands: list[str] = []
        if isinstance(procedure, dict):
            commands = procedure.get("commands")
            if isinstance(commands, list):
                procedure_commands = [str(command) for command in commands]
            elif procedure.get("command"):
                procedure_commands = [str(procedure.get("command"))]
        if not procedure_commands and isinstance(contract.get("checks"), list):
            procedure_commands = [str(command) for command in contract.get("checks", [])]
        procedure_text = "; ".join(procedure_commands)
        return (
            f"- {contract.get('task_id')}: {contract.get('status', 'proposed')} - {contract.get('summary', '')} | "
            f"frozen_requirements: {requirement_text or 'none'} | verification_procedure: {procedure_text or 'none'}"
        )

    def _reference_context(self, state: TaskState) -> str:
        reference = self._session_startup_context(state) if self._is_session_start(state) else self._incremental_reference_context(state)
        lines = [
            reference,
            self._memory_context(),
            self._loaded_skills_context(state),
        ]
        return "\n\n".join(section for section in lines if section.strip())

    def _is_session_start(self, state: TaskState) -> bool:
        return int(getattr(state, "session_used_tokens", 0) or 0) == 0

    def _task_graph_overview(self, path: Path, state: TaskState | None = None) -> str:
        rel_path = self._rel(path)
        if not path.exists():
            return f"Task graph: {rel_path}\nStatus: missing"
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            return f"Task graph: {rel_path}\nStatus: unreadable: {exc}"
        tasks = data.get("tasks") if isinstance(data, dict) else None
        if not isinstance(tasks, list):
            return f"Task graph: {rel_path}\nStatus: no tasks list"

        counts: dict[str, int] = {}
        completed_ids: set[str] = set()
        in_progress_ids: list[str] = []
        pending_tasks: list[dict[str, object]] = []
        explicit_blocked = 0
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("id", "")).strip()
            status = str(task.get("status", "unknown")).strip() or "unknown"
            counts[status] = counts.get(status, 0) + 1
            if status in {"completed", "done"} and task_id:
                completed_ids.add(task_id)
            elif status == "in_progress" and task_id:
                in_progress_ids.append(task_id)
            elif status == "pending":
                pending_tasks.append(task)
            elif status == "blocked":
                explicit_blocked += 1

        ready_now: list[str] = []
        ready_after_current: list[str] = []
        blocked_pending = 0
        in_progress_set = set(in_progress_ids)
        for task in pending_tasks:
            task_id = str(task.get("id", "")).strip()
            depends_on = task.get("depends_on", [])
            deps = [str(item).strip() for item in depends_on] if isinstance(depends_on, list) else []
            if all(dep in completed_ids for dep in deps):
                if task_id:
                    ready_now.append(task_id)
                continue
            if all(dep in completed_ids or dep in in_progress_set for dep in deps):
                if task_id:
                    ready_after_current.append(task_id)
                continue
            blocked_pending += 1

        current_task = self._active_task_id(state) if state else (in_progress_ids[0] if in_progress_ids else "none")
        count_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "empty"
        return "\n".join(
            [
                f"Task graph: {rel_path}",
                f"Total: {len(tasks)}",
                f"Done: {len(completed_ids)}",
                f"Current task: {current_task}",
                f"In progress: {self._format_id_list(in_progress_ids)}",
                f"Ready now: {self._format_id_list(ready_now)}",
                f"Ready after current completion: {self._format_id_list(ready_after_current)}",
                f"Blocked: {blocked_pending + explicit_blocked}",
                f"Status counts: {count_text}",
            ]
        )

    def _task_graph_summary(self, path: Path) -> str:
        if not path.exists():
            return "missing"
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            return f"unreadable: {exc}"
        tasks = data.get("tasks") if isinstance(data, dict) else None
        if not isinstance(tasks, list):
            return "no tasks list"
        counts: dict[str, int] = {}
        current: list[str] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            status = str(task.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
            if status in {"in_progress", "pending"} and len(current) < 3:
                current.append(str(task.get("id", "unknown")))
        count_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "empty"
        current_text = ", ".join(current) if current else "none"
        return f"{len(tasks)} task(s); {count_text}; next={current_text}"

    def _format_id_list(self, items: list[str], limit: int = 12) -> str:
        if not items:
            return "none"
        visible = items[:limit]
        suffix = f", ... +{len(items) - limit}" if len(items) > limit else ""
        return ", ".join(visible) + suffix

    def _just_in_time_context(self, state: TaskState) -> str:
        lines = [
            "# Just-in-Time Context",
            "Do not preload the whole repository. Read only what is needed for the active task.",
            "Recommended discovery flow:",
            "1. list a small directory with read target='.' or read target='<dir>';",
            "2. search/grep relevant ids, symbols, filenames, or error strings before reading; for example search target='T7' or search target='initializer_validation_errors' args={'path': '<candidate-or-dir>'};",
            "3. read matching code with args.query before falling back to explicit ranges; for example read target='<file>' args={'query': '\"id\": \"T7\"'};",
            "4. read corresponding tests;",
            "5. use errors or verifier output to guide the next search.",
            "PowerShell/Python examples:",
            "- list_files target='agent'",
            "- search target='create_issue' args={'path': 'agent'}",
            "- search target='initializer_validation_errors' args={'path': 'state/benchmarks/issue_tracker'}",
            "- read target='agent/loop.py' args={'query': 'def _execute_action'}",
            "- if read returns has_more=true, continue with data.next_read args only when the needed content was not found; otherwise act on the returned evidence.",
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
            "",
            self._loaded_skills_context(state),
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

    def _loaded_skills_context(self, state: TaskState) -> str:
        records = state.loaded_skills if isinstance(state.loaded_skills, list) else []
        if not records:
            return "# Loaded Skills\n\nNo Skill is currently loaded."
        skill_dir = self.state_dir / "skills"
        chunks: list[str] = []
        seen: set[str] = set()
        invalidated: list[str] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            requested = str(record.get("name", "")).strip()
            if not requested or requested in seen:
                continue
            seen.add(requested)
            match = None
            for path in sorted(skill_dir.glob("*.md")):
                skill = parse_skill(path.read_text(encoding="utf-8"), fallback_name=path.stem)
                if skill.name == requested:
                    match = skill
                    break
            if match is None or match.content_hash != str(record.get("content_hash", "")):
                invalidated.append(requested)
                continue
            chunks.append(match.content.rstrip())
        lines = ["# Loaded Skills", "Loaded Skill contents are workflow guidance, not completion evidence."]
        if chunks:
            lines.extend(["", "\n\n".join(chunks)])
        else:
            lines.extend(["", "No valid Skill is currently loaded."])
        if invalidated:
            lines.extend(["", "Invalidated Skills (reload before use): " + ", ".join(invalidated)])
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
            data = observation.get("data", {})
            command = data.get("command", "") if isinstance(data, dict) else ""
            target = str(command if action_name == "bash" and command else action.get("target", ""))
            target = target.replace("\\", "/").strip().rstrip("/")
            if not target:
                return
            if action_name in {"write", "edit"} and observation.get("ok") is True:
                seen_targets.add(target)
                return
            if action_name not in {"read", "list_files", "search", "bash"} or observation.get("ok") is not True:
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
        elif action_name == "search":
            payload = json.dumps(data.get("matches", []), ensure_ascii=False, indent=2)
        elif action_name == "bash":
            if data.get("command"):
                metadata.append(f"- command: {data.get('command')}")
            if data.get("cwd"):
                metadata.append(f"- cwd: {data.get('cwd')}")
            payload = str(data.get("output", ""))
        else:
            payload = json.dumps(data, ensure_ascii=False, indent=2)

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
            "report_id",
            "archived_verifier_report",
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
            "Each generated task should include: id, title, priority, depends_on, status, acceptance_criteria, criterion_command_map, expected_artifacts, implementation_artifacts when applicable, worker_test_artifacts when applicable, acceptance_artifacts when applicable, frozen_acceptance_artifacts when applicable, test_policy when tests are involved, and verification_commands.",
            "criterion_command_map must map every exact acceptance criterion string to one or more exact entries from verification_commands, and every verification command must be mapped.",
            "Verification commands run from the repository root and must be direct, portable Python commands without Unix-only shell setup. Commands that import or invoke project modules under the workspace must explicitly configure sys.path, PYTHONPATH, or subprocess cwd, including nested subprocess calls. Do not mix os.chdir('<workspace>') with subprocess cwd='<workspace>'; for contract repairs prefer verification_procedure.working_directory.",
            "priority MUST be an integer. Lower numbers are higher priority; use 1, 2, 3, ... and never strings such as 'high' or 'medium'.",
            "Minimal complete task example:",
            '{"id":"T1","title":"Implement feature","priority":1,"depends_on":[],"status":"pending","acceptance_criteria":["Behavior is verified."],"criterion_command_map":{"Behavior is verified.":["python -m unittest discover -s <workspace>/tests"]},"expected_artifacts":["<workspace>/pkg/feature.py"],"implementation_artifacts":["<workspace>/pkg/feature.py"],"worker_test_artifacts":[],"acceptance_artifacts":[],"frozen_acceptance_artifacts":[],"test_policy":{"acceptance_tests_mutable_by_worker":false,"acceptance_test_repair_requires_verifier_approval":true},"verification_commands":["python -m unittest discover -s <workspace>/tests"]}',
            "Implementation tasks must declare non-empty implementation_artifacts, and every owned implementation/test/acceptance artifact must also appear in expected_artifacts.",
            "Respect dependency constraints from project_spec.md (for example, standard-library-only means no pytest or package installation). Verification commands must be substantive and must not be placeholders such as bare echo, TODO, or 'not implemented'.",
            "INIT does not require an acceptance contract.",
            "During INIT, write or edit only the three initializer artifacts listed above. Do not create application code, tests, skeleton files, or workspace files.",
            "The repository-root init.sh belongs to the Long-Running Agent harness and must not be modified by a benchmark INIT.",
            "Any application artifact in the generated task graph must be under the workspace path required by project_spec.md.",
            "Do not use answer or finish during INIT.",
            "After writing initializer artifacts, use verify. The verifier executes the INIT verification command itself; only Verifier PASS completes INIT and allows Orchestrator to schedule the first Worker task.",
        ]

    def _required_next_action(self, state: TaskState) -> str:
        if state.pending_skill_review:
            report_id = str(state.pending_skill_review.get("report_id", ""))
            return (
                f"A hard Skill Reflection trigger fired for report {report_id}. "
                "Next action must be save_skill with the archived verifier report evidence, or dismiss_skill with a concrete reason. "
                "Do not continue ordinary task work until this review is resolved."
            )
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
                if state.pending_repair.get("reason") == "failed_verification_command":
                    return (
                        "A repair was attempted for the failed verification procedure. "
                        "Next action must be verify to rerun the agreed procedure. "
                        "Do not list directories or continue editing until verification is rerun."
                    )
                return (
                    "A repair was attempted for the failed acceptance command. "
                    f"Next action must be bash target='{command}' to rerun the same acceptance command. "
                    "Do not list directories or continue editing until this command is rerun."
                )
            if not repair_targets:
                command = str(state.pending_repair.get("command", ""))
                if state.pending_repair.get("reason") == "failed_verification_command":
                    return (
                        "The verification procedure failed, but no mutable implementation target is available. "
                        "Do not edit frozen acceptance artifacts. "
                        "If the procedure path/cwd is wrong, update only verification_procedure with action='contract'; otherwise use verify only after the implementation/environment issue is addressed. "
                        f"Failure excerpt: {excerpt}"
                    )
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

    def _skill_reflection_context(self, state: TaskState) -> str:
        review = state.pending_skill_review if isinstance(state.pending_skill_review, dict) else {}
        if not review:
            return "# Pending Skill Reflection\n\nNo hard trigger threshold was met. Do not save a Skill merely because a task completed."
        lines = [
            "# Pending Skill Reflection",
            "A hard trigger threshold was met after independent verification. Choose save_skill or dismiss_skill.",
            f"- task_id: {review.get('task_id', '')}",
            f"- report_id: {review.get('report_id', '')}",
            f"- report_path: {review.get('report_path', '')}",
            f"- trace_ref: {json.dumps(review.get('trace_ref', {}), ensure_ascii=False)}",
            f"- trigger_reasons: {json.dumps(review.get('trigger_reasons', []), ensure_ascii=False)}",
            "- Rule: save only a generalizable, repeatable, actionable procedure; otherwise dismiss with a concrete reason.",
            "",
            "## Relevant Trace Window",
        ]
        for item in review.get("relevant_trace", []):
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('action')} {item.get('target', '')}: ok={item.get('ok')} — {item.get('summary', '')}"
                )
        return "\n".join(lines)

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
        missing_owned = self._missing_active_owned_artifacts(state)
        policy = self._active_task_test_policy(state)
        return [
            f"- implementation_artifacts: {', '.join(implementation) if implementation else 'none'}",
            f"- worker_test_artifacts: {', '.join(worker_tests) if worker_tests else 'none'}",
            f"- acceptance_artifacts: {', '.join(acceptance) if acceptance else 'none'}",
            f"- frozen_acceptance_artifacts: {', '.join(frozen) if frozen else 'none'}",
            f"- missing_owned_artifacts: {', '.join(missing_owned) if missing_owned else 'none'}",
            f"- test_policy: {policy}",
            "- Rule: implementation artifacts are normal repair targets; worker tests remain mutable unless explicitly listed in frozen_acceptance_artifacts.",
        ]

    def _missing_active_owned_artifacts(self, state: TaskState) -> list[str]:
        owned: list[str] = []
        for artifact in [
            *self._active_task_implementation_artifacts(state),
            *self._active_task_worker_test_artifacts(state),
        ]:
            if artifact not in owned:
                owned.append(artifact)
        missing: list[str] = []
        for artifact in owned:
            if not (self.root / artifact).exists():
                missing.append(artifact)
        return missing

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
        catalog = skill_catalog(skill_dir)
        if not catalog:
            return "No skills available."
        return "# Available Skills\n\n" + "\n".join(
            f"- {item['name']}: {item['description']}" for item in catalog
        )

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
