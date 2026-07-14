from __future__ import annotations

import json
import os
import py_compile
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
        hidden_result: dict[str, Any] | None = None
        if self._requires_hidden_acceptance(state):
            hidden_result = self._run_benchmark_hidden_acceptance()
            checks.append(("hidden_acceptance", hidden_result["ok"]))

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
        if hidden_result is not None:
            data["hidden_acceptance"] = hidden_result
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
                            "output": working_directory[-4000:],
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
                output = (completed.stdout + completed.stderr).strip()
                results.append(
                    {
                        "command": command,
                        "working_directory": working_directory or ".",
                        "ok": completed.returncode == 0,
                        "returncode": completed.returncode,
                        "summary": f"Verification command exited with code {completed.returncode}.",
                        "output": output[-4000:],
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
                        "output": str(exc)[-4000:],
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

    def _requires_hidden_acceptance(self, state: TaskState) -> bool:
        return any("hidden acceptance" in str(item).lower() for item in state.acceptance_criteria)

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

    def _run_benchmark_hidden_acceptance(self) -> dict[str, Any]:
        benchmark_id = self._benchmark_id()
        script = self.root / "eval" / "benchmarks" / str(benchmark_id) / "hidden_acceptance.py"
        if not benchmark_id or not script.is_file():
            return {
                "ok": False,
                "configured": False,
                "returncode": None,
                "summary": "Benchmark hidden acceptance is not configured.",
            }
        try:
            completed = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
            )
        except Exception:
            return {
                "ok": False,
                "configured": True,
                "returncode": None,
                "summary": "Benchmark hidden acceptance could not be executed.",
            }
        ok = completed.returncode == 0
        return {
            "ok": ok,
            "configured": True,
            "returncode": completed.returncode,
            "summary": "Benchmark hidden acceptance passed." if ok else "Benchmark hidden acceptance failed.",
        }

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
        checks.append(
            (
                "hidden_acceptance_is_private",
                all("hidden_acceptance" not in str(command).lower() for command in commands),
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
        evidence = proposal.get("evidence", [])
        evidence_text = "\n".join(str(item).lower() for item in evidence)
        evidence_type = proposal.get("evidence_type")
        checks = []
        checks.append(("has_skill_id", bool(str(proposal.get("skill_id", "")).strip())))
        checks.append(("has_title", bool(str(proposal.get("title", "")).strip())))
        checks.append(("has_body", bool(str(proposal.get("body", "")).strip())))
        checks.append(("has_skill_evidence", bool(evidence) and isinstance(evidence, list)))
        if evidence_type == "verified_success":
            checks.append(
                (
                    "verifier_confirmed_success",
                    state.last_observation.get("ok") is True
                    and "verifier passed" in str(state.last_observation.get("summary", "")).lower(),
                )
            )
        elif evidence_type == "evidence_confirmed_failure":
            checks.append(
                (
                    "evidence_confirmed_failure",
                    any(marker in evidence_text for marker in ["failed", "failure", "error", "rejected", "trace:"]),
                )
            )
        else:
            checks.append(("valid_evidence_type", False))
        ok = all(value for _, value in checks)
        summary = "Skill promotion accepted." if ok else "Skill promotion rejected."
        result = ToolResult(ok, summary, {"checks": dict(checks), "proposal": proposal})
        self._write_report(result)
        return result

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
        output = (completed.stdout + completed.stderr).strip()
        return completed.returncode == 0, output[-4000:]

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
