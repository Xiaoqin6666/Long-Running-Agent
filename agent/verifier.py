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
    def __init__(self, root: Path, state_dir: Path | None = None) -> None:
        self.root = root
        self.state_dir = state_dir or root / "state"

    def run(self, profile: str, state: TaskState) -> ToolResult:
        del profile
        checks = []
        checks.append(("state_dir_exists", self.state_dir.exists()))
        checks.append(("trace_dir_exists", (self.state_dir / "traces").exists()))
        checks.append(("has_plan_nodes", bool(state.nodes)))
        checks.append(("has_evidence", any(node["evidence"] for node in state.nodes)))
        compile_ok, compile_error = self._compile_agent()
        checks.append(("python_compile", compile_ok))
        tests_ok, tests_output = self._run_tests()
        checks.append(("unit_tests", tests_ok))

        ok = all(value for _, value in checks)
        data = {"checks": dict(checks), "task_id": state.task_id}
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
        # 校验1：存在非空task_id
        checks.append(("has_task_id", bool(str(contract.get("task_id", "")).strip())))
        # 校验2：存在非空summary
        checks.append(("has_summary", bool(str(contract.get("summary", "")).strip())))
        # 校验3：checks字段为非空list
        checks.append(("has_checks", bool(contract.get("checks")) and isinstance(contract.get("checks"), list)))

        # 新增工具函数：判断单条check是否是可执行校验命令
        def is_executable_command(line: str) -> bool:
            line = str(line).strip().lower()
            # 匹配系统中所有合法验收执行命令前缀
            exec_prefix = (
                "python -c",
                "python -m unittest",
                "bash",
                "./",
                "pwsh",
                "python3 -c"
            )
            return any(line.startswith(prefix) for prefix in exec_prefix)

        # 校验4：至少包含一条可执行校验脚本（替换原test/smoke关键词匹配）
        check_items = contract.get("checks", [])
        has_exec_check = any(is_executable_command(item) for item in check_items)
        checks.append(("behavior_level_checks", has_exec_check))

        # 汇总所有校验结果
        ok = all(value for _, value in checks)
        summary = "Acceptance contract agreed." if ok else "Acceptance contract rejected."
        result = ToolResult(ok, summary, {"checks": dict(checks), "contract": contract})
        self._write_report(result)
        return result

    def validate_skill_promotion(self, proposal: dict[str, Any], state: TaskState) -> ToolResult:
        evidence = proposal.get("evidence", [])
        evidence_text = "\n".join(str(item).lower() for item in evidence)
        evidence_type = proposal.get("evidence_type")
        checks = []
        checks.append(("has_skill_id", bool(str(proposal.get("skill_id", "")).strip())))
        checks.append(("has_title", bool(str(proposal.get("title", "")).strip())))
        checks.append(("has_body", bool(str(proposal.get("body", "")).strip())))
        checks.append(("has_evidence", bool(evidence) and isinstance(evidence, list)))
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
