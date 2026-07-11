from __future__ import annotations

import json
import py_compile
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.planner import TaskState
from agent.tools import ToolResult


class Verifier:
    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, profile: str, state: TaskState) -> ToolResult:
        del profile
        checks = []
        checks.append(("state_dir_exists", (self.root / "state").exists()))
        checks.append(("trace_dir_exists", (self.root / "state" / "traces").exists()))
        checks.append(("has_plan_nodes", bool(state.nodes)))
        checks.append(("has_evidence", any(node["evidence"] for node in state.nodes)))
        compile_ok, compile_error = self._compile_agent()
        checks.append(("python_compile", compile_ok))
        tests_ok, tests_output = self._run_tests()
        checks.append(("unit_tests", tests_ok))

        ok = all(value for _, value in checks)
        data = {"checks": dict(checks)}
        if compile_error:
            data["compile_error"] = compile_error
        if tests_output:
            data["test_output"] = tests_output
        summary = "Verifier passed." if ok else "Verifier failed."
        result = ToolResult(ok, summary, data)
        self._write_report(result)
        return result

    def validate_contract(self, contract: dict[str, Any]) -> ToolResult:
        checks = []
        checks.append(("has_task_id", bool(str(contract.get("task_id", "")).strip())))
        checks.append(("has_summary", bool(str(contract.get("summary", "")).strip())))
        checks.append(("has_checks", bool(contract.get("checks")) and isinstance(contract.get("checks"), list)))
        checks.append(
            (
                "behavior_level_checks",
                any("test" in str(item).lower() or "smoke" in str(item).lower() for item in contract.get("checks", [])),
            )
        )
        ok = all(value for _, value in checks)
        summary = "Acceptance contract agreed." if ok else "Acceptance contract rejected."
        result = ToolResult(ok, summary, {"checks": dict(checks), "contract": contract})
        self._write_report(result)
        return result

    def _compile_agent(self) -> tuple[bool, str | None]:
        try:
            for path in (self.root / "agent").glob("*.py"):
                py_compile.compile(str(path), doraise=True)
            for path in (self.root / "agent" / "tools").glob("*.py"):
                py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            return False, str(exc)
        return True, None

    def _run_tests(self) -> tuple[bool, str]:
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
        state_dir = self.root / "state"
        state_dir.mkdir(exist_ok=True)
        payload = {
            "time": datetime.now(timezone.utc).isoformat(),
            "ok": result.ok,
            "summary": result.summary,
            "data": result.data,
        }
        (state_dir / "verifier_report.md").write_text(
            "# Latest Verifier Report\n\n"
            + "```json\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n```\n",
            encoding="utf-8",
        )
