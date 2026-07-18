from __future__ import annotations

import json
import os
import py_compile
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.output_capture import capture_command_output
from agent.planner import TaskState, verification_command_portability_error
from agent.tools import ToolResult


class Verifier:
    def __init__(self, root: Path, state_dir: Path | None = None) -> None:
        self.root = root
        self.state_dir = state_dir or root / "state"

    def run(self, profile: str, state: TaskState) -> ToolResult:
        del profile
        checks = []
        checks.append(("state_dir_exists", self.state_dir.exists()))
        checks.append(("trace_dir_exists", (self.state_dir / "traces").exists()))
        checks.append(("has_plan_nodes", bool(state.nodes)))
        active_node = self._active_node(state)
        contract = self._active_contract(state)
        procedures: list[dict[str, Any]] = []
        contract_validation: dict[str, bool] | None = None
        if state.task_id == "INIT":
            procedures = self._procedures_from_commands(self._node_verification_commands(active_node))
        elif active_node.get("contract_managed"):
            contract_checks = self._contract_validation_checks(contract or {}, active_node)
            contract_validation = dict(contract_checks)
            contract_ok = bool(contract) and all(value for _, value in contract_checks)
            checks.append(("contract_frozen", contract_ok))
            if contract_ok and contract:
                procedures = self._contract_verification_procedures(contract)
        elif contract and contract.get("status") == "agreed":
            procedures = self._contract_verification_procedures(contract)
        else:
            procedures = self._procedures_from_commands(self._node_verification_commands(active_node))

        verification_ok, command_results = self._run_verification_commands(procedures)
        if procedures or active_node.get("contract_managed") or state.task_id == "INIT":
            checks.append(("verification_commands", verification_ok and bool(procedures)))
        self._record_verification_evidence(state, command_results)
        compile_ok, compile_error = self._compile_agent()
        checks.append(("python_compile", compile_ok))
        tests_ok, tests_output = self._run_tests()
        checks.append(("unit_tests", tests_ok))
        ok = all(value for _, value in checks)
        data = {"checks": dict(checks), "task_id": state.task_id}
        if contract_validation is not None:
            data["contract_validation"] = contract_validation
        if procedures or active_node.get("contract_managed") or state.task_id == "INIT":
            data["verification"] = {"commands": command_results}
        if compile_error:
            data["compile_error"] = compile_error
        if tests_output:
            data["test_output"] = tests_output
        summary = "Verifier passed." if ok else "Verifier failed."
        result = ToolResult(ok, summary, data)
        self._write_report(result)
        return result

    def _active_node(self, state: TaskState) -> dict[str, Any]:
        for node in state.nodes:
            if str(node.get("id", "")) == state.task_id:
                return node
        return state.nodes[0] if state.nodes else {}

    def _active_contract(self, state: TaskState) -> dict[str, Any] | None:
        matches = [
            contract
            for contract in state.acceptance_contracts
            if isinstance(contract, dict)
            and contract.get("task_id") in {state.task_id, "current"}
            and contract.get("status") == "agreed"
        ]
        return matches[-1] if matches else None

    @staticmethod
    def _node_verification_commands(node: dict[str, Any]) -> list[str]:
        commands = node.get("verification_commands", [])
        return [str(item) for item in commands] if isinstance(commands, list) else []

    def _contract_verification_procedures(self, contract: dict[str, Any]) -> list[dict[str, Any]]:
        procedure = contract.get("verification_procedure")
        if isinstance(procedure, dict):
            working_directory = str(procedure.get("working_directory", "")).strip()
            commands = procedure.get("commands")
            if isinstance(commands, list):
                return [
                    self._normalize_procedure({"command": str(command), "working_directory": working_directory})
                    for command in commands
                    if str(command).strip()
                ]
            command = str(procedure.get("command", "")).strip()
            if command:
                return [self._normalize_procedure({"command": command, "working_directory": working_directory})]
        return self._procedures_from_commands(contract.get("checks", []))

    def _procedures_from_commands(self, commands: object) -> list[dict[str, Any]]:
        if not isinstance(commands, list):
            return []
        return [self._normalize_procedure({"command": str(command)}) for command in commands if str(command).strip()]

    def _normalize_procedure(self, procedure: dict[str, Any]) -> dict[str, Any]:
        normalized = {"command": str(procedure.get("command", "")).strip()}
        working_directory = str(procedure.get("working_directory", "")).strip()
        if working_directory:
            normalized["working_directory"] = working_directory
        return normalized

    def _run_verification_commands(self, procedures: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
        results: list[dict[str, Any]] = []
        env = os.environ.copy()
        benchmark_id = self._benchmark_id()
        if benchmark_id:
            workspace = (self.root / "eval" / "benchmarks" / benchmark_id / "workspace").resolve()
            current = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join([str(workspace), current] if current else [str(workspace)])
        for procedure in procedures:
            command = str(procedure.get("command", "")).strip()
            if not command:
                continue
            cwd = self.root
            working_directory = str(procedure.get("working_directory", "")).strip()
            if working_directory:
                try:
                    cwd = (self.root / working_directory).resolve()
                    cwd.relative_to(self.root.resolve())
                except (OSError, ValueError):
                    results.append(
                        {
                            "command": command,
                            "working_directory": working_directory,
                            "ok": False,
                            "returncode": None,
                            "summary": "Verification working directory is outside the workspace.",
                            "output": working_directory,
                        }
                    )
                    continue
            try:
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    env=env,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=180,
                )
                output_data = capture_command_output(
                    root=self.root,
                    output_dir=self.state_dir / "tool_outputs" / "verifier",
                    label="verify",
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
                results.append(
                    {
                        "command": command,
                        "working_directory": working_directory or ".",
                        "ok": completed.returncode == 0,
                        "returncode": completed.returncode,
                        "summary": f"Verification command exited with code {completed.returncode}.",
                        **output_data,
                    }
                )
            except (OSError, subprocess.SubprocessError) as exc:
                results.append(
                    {
                        "command": command,
                        "working_directory": working_directory or ".",
                        "ok": False,
                        "returncode": None,
                        "summary": "Verification command could not be executed.",
                        "output": str(exc),
                    }
                )
        return bool(results) and all(item["ok"] for item in results), results

    def _record_verification_evidence(self, state: TaskState, results: list[dict[str, Any]]) -> None:
        for result in results:
            if not result.get("ok"):
                continue
            record = {
                "action": "verify",
                "target": result["command"],
                "summary": result["summary"],
                "task_id": state.task_id,
                "evidence_type": "verification_command_passed",
                "ok": True,
            }
            duplicate = any(
                isinstance(item, dict)
                and item.get("task_id") == state.task_id
                and item.get("evidence_type") == "verification_command_passed"
                and item.get("target") == result["command"]
                for item in state.evidence_sources
            )
            if not duplicate:
                state.evidence_sources.append(record)
            node = self._active_node(state)
            evidence = node.setdefault("evidence", [])
            marker = f"Verification command passed: {result['command']}"
            if marker not in evidence:
                evidence.append(marker)

    def _benchmark_id(self) -> str | None:
        benchmark_root = (self.root / "state" / "benchmarks").resolve()
        try:
            relative = self.state_dir.resolve().relative_to(benchmark_root)
        except ValueError:
            return None
        if len(relative.parts) != 1:
            return None
        benchmark_id = relative.parts[0]
        if not benchmark_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in benchmark_id):
            return None
        return benchmark_id

    def validate_contract(
        self,
        contract: dict[str, Any],
        task: dict[str, Any] | None = None,
    ) -> ToolResult:
        checks = self._contract_validation_checks(contract, task)
        ok = all(value for _, value in checks)
        summary = "Acceptance contract agreed." if ok else "Acceptance contract rejected."
        result = ToolResult(ok, summary, {"checks": dict(checks), "contract": contract})
        self._write_report(result)
        return result

    def _contract_validation_checks(
        self,
        contract: dict[str, Any],
        task: dict[str, Any] | None = None,
    ) -> list[tuple[str, bool]]:
        checks: list[tuple[str, bool]] = []
        procedures = self._contract_verification_procedures(contract)
        commands = [item["command"] for item in procedures]
        frozen_requirements = contract.get("frozen_requirements", contract.get("required_evidence", []))
        checks.append(("has_task_id", bool(str(contract.get("task_id", "")).strip())))
        checks.append(("has_summary", bool(str(contract.get("summary", "")).strip())))
        checks.append(("has_frozen_requirements", isinstance(frozen_requirements, list) and bool(frozen_requirements)))
        checks.append(("has_verification_procedure", bool(procedures)))
        checks.append(
            (
                "portable_executable_checks",
                bool(commands)
                and all(verification_command_portability_error(str(command)) is None for command in commands),
            )
        )
        if task is None:
            return checks

        criteria = task.get("acceptance_criteria", [])
        mapping = contract.get("criterion_command_map")
        expected_artifacts = task.get("expected_artifacts", [])
        checks.append(("matches_active_task", str(contract.get("task_id")) == str(task.get("id"))))
        checks.append(("is_frozen", contract.get("frozen") is True))
        checks.append(
            (
                "requirements_match_task_graph",
                isinstance(criteria, list)
                and isinstance(frozen_requirements, list)
                and [str(item) for item in frozen_requirements] == [str(item) for item in criteria],
            )
        )
        checks.append(
            (
                "scope_matches_artifacts",
                isinstance(contract.get("scope"), list)
                and isinstance(expected_artifacts, list)
                and [str(item) for item in contract.get("scope", [])]
                == [str(item) for item in expected_artifacts],
            )
        )
        mapping_ok = isinstance(mapping, dict) and isinstance(criteria, list)
        if mapping_ok:
            criterion_texts = [str(item) for item in criteria]
            executable = set(commands)
            mapping_ok = set(str(item) for item in mapping) == set(criterion_texts)
            mapped_commands: set[str] = set()
            for criterion in criterion_texts:
                mapped = mapping.get(criterion, [])
                if not isinstance(mapped, list) or not mapped:
                    mapping_ok = False
                    continue
                normalized = {str(item) for item in mapped}
                if not normalized.issubset(executable):
                    mapping_ok = False
                mapped_commands.update(normalized)
            mapping_ok = mapping_ok and mapped_commands == executable
        checks.append(("criteria_fully_mapped", mapping_ok))
        return checks

    def validate_skill_promotion(self, proposal: dict[str, Any], state: TaskState) -> ToolResult:
        evidence_refs = proposal.get("evidence_refs", [])
        evidence_type = proposal.get("evidence_type")
        checks = []
        checks.append(("has_name", bool(str(proposal.get("name", proposal.get("skill_id", ""))).strip())))
        checks.append(("has_description", bool(str(proposal.get("description", proposal.get("title", ""))).strip())))
        checks.append(("has_instruction", bool(str(proposal.get("instruction", proposal.get("body", ""))).strip())))
        checks.append(("has_evidence_refs", bool(evidence_refs) and isinstance(evidence_refs, list)))
        evidence_checks, resolved_evidence = self._validate_skill_evidence_refs(
            evidence_refs if isinstance(evidence_refs, list) else [], evidence_type, state
        )
        checks.extend(evidence_checks)
        checks.append(("valid_evidence_type", evidence_type in {"verified_success", "evidence_confirmed_failure"}))
        ok = all(value for _, value in checks)
        summary = "Skill promotion accepted." if ok else "Skill promotion rejected."
        result = ToolResult(
            ok,
            summary,
            {"checks": dict(checks), "proposal": proposal, "resolved_evidence": resolved_evidence},
        )
        self._write_report(result)
        return result

    def _validate_skill_evidence_refs(
        self,
        refs: list[Any],
        evidence_type: object,
        state: TaskState,
    ) -> tuple[list[tuple[str, bool]], list[dict[str, Any]]]:
        resolved: list[dict[str, Any]] = []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            ref_type = str(ref.get("type", ""))
            evidence = None
            if ref_type == "verifier_report":
                evidence = self._resolve_verifier_report_ref(ref, state)
            elif ref_type == "trace":
                evidence = self._resolve_trace_ref(ref, state)
            if evidence:
                resolved.append(evidence)
        if evidence_type == "verified_success":
            matching = any(
                item.get("ok") is True
                and (item.get("type") == "verifier_report" or item.get("action") == "verify")
                for item in resolved
            )
        else:
            matching = any(item.get("ok") is False for item in resolved)
        return [
            ("all_evidence_refs_resolved", bool(refs) and len(resolved) == len(refs)),
            ("evidence_matches_type", matching),
        ], resolved

    def _resolve_verifier_report_ref(self, ref: dict[str, Any], state: TaskState) -> dict[str, Any] | None:
        report_id = str(ref.get("report_id", "")).strip()
        if report_id:
            if not all(ch.isalnum() or ch in {"-", "_"} for ch in report_id):
                return None
            path = self.state_dir / "verifier_reports" / f"{report_id}.json"
            if not path.is_file():
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
            if not isinstance(payload, dict) or payload.get("report_id") != report_id:
                return None
            actual_task = str(payload.get("task_id", ""))
            expected_task = str(ref.get("task_id", actual_task))
            if not actual_task or expected_task != actual_task:
                return None
            return {
                "type": "verifier_report",
                "report_id": report_id,
                "path": str(path.relative_to(self.root)).replace("\\", "/"),
                "task_id": actual_task,
                "ok": payload.get("ok") is True,
            }
        path = self.state_dir / "verifier_report.md"
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8")
        start = text.find("```json")
        if start == -1:
            return None
        start = text.find("\n", start)
        end = text.find("```", start + 1)
        if start == -1 or end == -1:
            return None
        try:
            payload = json.loads(text[start:end].strip())
        except json.JSONDecodeError:
            return None
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        actual_task = str(data.get("task_id", state.task_id)) if isinstance(data, dict) else state.task_id
        expected_task = str(ref.get("task_id", actual_task))
        if expected_task != actual_task:
            return None
        return {
            "type": "verifier_report",
            "path": str(path.relative_to(self.root)).replace("\\", "/"),
            "task_id": actual_task,
            "ok": payload.get("ok") is True,
        }

    def _resolve_trace_ref(self, ref: dict[str, Any], state: TaskState) -> dict[str, Any] | None:
        raw_path = str(ref.get("path", "")).strip()
        if not raw_path:
            return None
        path = (self.root / raw_path).resolve()
        trace_dir = (self.state_dir / "traces").resolve()
        if trace_dir not in path.parents or not path.is_file():
            return None
        step = ref.get("step", ref.get("trace_step"))
        try:
            expected_step = int(step)
        except (TypeError, ValueError):
            return None
        for event in self._load_json_stream(path):
            if int(event.get("step", -1)) != expected_step:
                continue
            event_task = str(event.get("task_id", state.task_id))
            expected_task = str(ref.get("task_id", event_task))
            if expected_task != event_task:
                return None
            observation = event.get("observation", {})
            if not isinstance(observation, dict):
                return None
            return {
                "type": "trace",
                "path": str(path.relative_to(self.root)).replace("\\", "/"),
                "step": expected_step,
                "task_id": event_task,
                "action": str(event.get("action", {}).get("action", "")) if isinstance(event.get("action"), dict) else "",
                "ok": observation.get("ok") is True,
            }
        return None

    def _load_json_stream(self, path: Path) -> list[dict[str, Any]]:
        text = path.read_text(encoding="utf-8").strip()
        events: list[dict[str, Any]] = []
        decoder = json.JSONDecoder()
        index = 0
        while index < len(text):
            while index < len(text) and text[index].isspace():
                index += 1
            if index >= len(text):
                break
            try:
                event, index = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                return []
            if isinstance(event, dict):
                events.append(event)
        return events

    def record_result(self, result: ToolResult) -> None:
        self._write_report(result)

    def _compile_agent(self) -> tuple[bool, str | None]:
        try:
            benchmark_id = self._benchmark_id()
            if benchmark_id:
                workspace = self.root / "eval" / "benchmarks" / benchmark_id / "workspace"
                if not workspace.is_dir():
                    return True, None
                for path in workspace.rglob("*.py"):
                    py_compile.compile(str(path), doraise=True)
                return True, None
            for path in (self.root / "agent").glob("*.py"):
                py_compile.compile(str(path), doraise=True)
            for path in (self.root / "agent" / "tools").glob("*.py"):
                py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            return False, str(exc)
        return True, None

    def _run_tests(self) -> tuple[bool, str]:
        if self._benchmark_id():
            return True, "Host-agent unit tests are skipped for benchmark tasks; frozen requirement procedures run separately."
        tests_dir = self.root / "tests"
        if not tests_dir.exists():
            return True, "No tests directory found."
        completed = subprocess.run(
            ["python", "-m", "unittest", "discover", "-s", "tests"],
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        output_data = capture_command_output(
            root=self.root,
            output_dir=self.state_dir / "tool_outputs" / "verifier",
            label="unit-tests",
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        return completed.returncode == 0, str(output_data.get("output", ""))

    def _write_report(self, result: ToolResult) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "time": datetime.now(timezone.utc).isoformat(),
            "ok": result.ok,
            "summary": result.summary,
            "data": result.data,
        }
        (self.state_dir / "verifier_report.md").write_text(
            "# Latest Verifier Report\n\n"
            + "```json\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n```\n",
            encoding="utf-8",
        )
