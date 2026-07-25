from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INTEGRATION_CONTRACT_KIND = "system_owned_integration_contract"
INTEGRATION_RESULTS_KIND = "integration_check_results"
INTEGRATION_CONTRACT_VERSION = 1

INTEGRATION_RESULT_JUDGE_PROMPT = """You are the verifier for project-level integration results.

You judge whether declared integration items are actually satisfied by the implemented project.
Return JSON only. Do not include Markdown fences or commentary.

Required response shape:
{
  "passed": true,
  "summary": "short project integration summary",
  "checks": [
    {
      "id": "static-1 | dynamic-1 | edge-1 | scenario-1",
      "type": "static | dynamic | edge | scenario",
      "passed": true,
      "reason": "specific explanation tied to evidence",
      "evidence": "short summary of the evidence you relied on"
    }
  ]
}

Rules:
- Return one verdict for every provided check item.
- Judge the real integration behavior, not whether a superficial string is present.
- A static text check should fail when the only evidence is an unused import or a placeholder that does not connect modules.
- A dynamic command should pass only when its output actually demonstrates the intended integrated behavior; "0 tests ran" is a failure even if a command exits successfully.
- A module edge should pass only when its covered checks passed and the evidence shows the source module/component really calls, imports and uses, configures, or otherwise connects to the target.
- A required scenario should pass only when implemented entry points and checks prove the scenario end-to-end.
- Use only the supplied local evidence. Do not ask for tools.
- Keep verdicts conservative. If evidence is missing or ambiguous, return passed=false.
"""

INTEGRATION_CONTRACT_GENERATOR_PROMPT = """You generate verifier-owned Integration Contracts for project-level final acceptance.

Return JSON only. Do not include Markdown fences or commentary.

Required top-level shape:
{
  "kind": "system_owned_integration_contract",
  "version": 1,
  "generated_by": "llm",
  "entry_points": ["concrete project entry points or workflows that must be exercised"],
  "module_edges": [
    {
      "id": "edge-1",
      "from": "source module/component/artifact",
      "to": "target module/component/artifact",
      "covered_by": ["static-1", "dynamic-1"],
      "reason": "why this connection matters for the whole system"
    }
  ],
  "required_scenarios": [
    {
      "id": "scenario-1",
      "name": "user-visible or system-level integration scenario",
      "entry_point": "matching entry point",
      "steps": ["observable setup/action/check steps"],
      "expected_observable": "observable result proving modules are connected"
    }
  ],
  "static_checks": [
    {
      "id": "static-1",
      "kind": "static_artifact_exists | static_text_contains",
      "path": "repo-relative implemented file path",
      "contains": "required text only for static_text_contains"
    }
  ],
  "dynamic_checks": [
    {
      "id": "dynamic-1",
      "kind": "dynamic_command",
      "command": "direct command that exits 0 only when the integrated system works",
      "working_directory": "repo-relative directory when needed",
      "covers_edges": ["edge-1"]
    }
  ]
}

Rules:
- Use the task graph, project specification, requirements, and current file inventory together.
- Focus on whether modules are connected end-to-end, not whether each isolated task exists.
- Prefer existing test commands, CLI entry points, importable service APIs, and non-interactive test modes already implied by files.
- Do not invent unimplemented file paths. Static checks must reference paths in the file inventory or task artifacts.
- Dynamic checks must be automated and bounded; do not require GUI mainloop(), manual clicking, network access, or package installation.
- Use direct Python commands or project test commands that should work from the generated workspace.
- If a project has too little executable surface, still describe entry_points/module_edges/scenarios and use static checks for concrete integration artifacts.
- Every module edge with covered_by should reference check ids that exist in static_checks or dynamic_checks.
- Keep the contract concise: usually 3-8 module_edges and 1-5 dynamic_checks are enough.
"""


def build_integration_contract(
    tasks: list[dict[str, Any]],
    *,
    project_spec: str = "",
    requirements: list[dict[str, Any]] | None = None,
    file_inventory: list[dict[str, Any]] | None = None,
    provider: str = "offline",
) -> dict[str, Any]:
    if provider == "offline":
        return _build_offline_contract(tasks)
    if provider == "openai-compatible":
        contract = OpenAICompatibleIntegrationContractBuilder.from_env().build(
            tasks=tasks,
            project_spec=project_spec,
            requirements=requirements or [],
            file_inventory=file_inventory or [],
        )
        errors = validate_integration_contract(contract)
        if errors:
            raise RuntimeError("LLM-generated Integration Contract failed schema validation: " + "; ".join(errors))
        return contract
    raise ValueError(f"Unsupported provider: {provider}")


def _build_offline_contract(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a harness-owned project integration contract from task metadata."""
    ordinary_tasks = [
        task
        for task in tasks
        if isinstance(task, dict)
        and task.get("optional") is not True
        and str(task.get("id", "")).strip() != "FINAL_ACCEPTANCE"
    ]
    return {
        "kind": INTEGRATION_CONTRACT_KIND,
        "version": INTEGRATION_CONTRACT_VERSION,
        "generated_by": "harness",
        "entry_points": _unique_strings(
            entry
            for task in ordinary_tasks
            for entry in _string_list(task.get("entry_points"))
        ),
        "module_edges": _unique_objects(
            edge
            for task in ordinary_tasks
            for edge in _object_list(task.get("integration_edges", task.get("module_edges", [])))
        ),
        "required_scenarios": _unique_objects(
            scenario
            for task in ordinary_tasks
            for scenario in _object_list(task.get("required_scenarios", task.get("integration_scenarios", [])))
        ),
        "static_checks": _unique_objects(
            check
            for task in ordinary_tasks
            for check in [
                *_object_list(task.get("static_checks", [])),
                *[
                    item
                    for item in _object_list(task.get("integration_checks", []))
                    if _check_kind(item).startswith("static")
                ],
            ]
        ),
        "dynamic_checks": _unique_objects(
            check
            for task in ordinary_tasks
            for check in [
                *_object_list(task.get("dynamic_checks", [])),
                *[
                    item
                    for item in _object_list(task.get("integration_checks", []))
                    if not _check_kind(item).startswith("static")
                ],
            ]
        ),
    }


def write_integration_contract(
    path: Path,
    tasks: list[dict[str, Any]],
    *,
    project_spec: str = "",
    requirements: list[dict[str, Any]] | None = None,
    file_inventory: list[dict[str, Any]] | None = None,
    provider: str = "offline",
) -> dict[str, Any]:
    payload = build_integration_contract(
        tasks,
        project_spec=project_spec,
        requirements=requirements,
        file_inventory=file_inventory,
        provider=provider,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


@dataclass
class OpenAICompatibleIntegrationContractBuilder:
    api_key: str
    base_url: str
    model: str
    timeout: int = 60
    temperature: float = 0.1

    @classmethod
    def from_env(cls) -> "OpenAICompatibleIntegrationContractBuilder":
        api_key = os.environ.get("LONG_AGENT_API_KEY")
        if not api_key:
            raise RuntimeError("Missing LONG_AGENT_API_KEY.")
        return cls(
            api_key=api_key,
            base_url=os.environ.get("LONG_AGENT_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            model=os.environ.get(
                "LONG_AGENT_INTEGRATION_CONTRACT_MODEL",
                os.environ.get("LONG_AGENT_MODEL", "gpt-4.1-mini"),
            ),
            timeout=int(
                os.environ.get("LONG_AGENT_INTEGRATION_CONTRACT_TIMEOUT", os.environ.get("LONG_AGENT_TIMEOUT", "60"))
            ),
            temperature=float(
                os.environ.get(
                    "LONG_AGENT_INTEGRATION_CONTRACT_TEMPERATURE",
                    os.environ.get("LONG_AGENT_TEMPERATURE", "0.1"),
                )
            ),
        )

    def build(
        self,
        *,
        tasks: list[dict[str, Any]],
        project_spec: str,
        requirements: list[dict[str, Any]],
        file_inventory: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": INTEGRATION_CONTRACT_GENERATOR_PROMPT},
                {
                    "role": "user",
                    "content": self._render_user_content(
                        tasks=tasks,
                        project_spec=project_spec,
                        requirements=requirements,
                        file_inventory=file_inventory,
                    ),
                },
            ],
        }
        response = self._post_chat_completions(payload)
        content = str(response["choices"][0]["message"]["content"]).strip()
        parsed = _loads_json_object(_strip_markdown_fence(content))
        if not isinstance(parsed, dict):
            extracted = _extract_first_json_object(content)
            parsed = _loads_json_object(extracted) if extracted else None
        if not isinstance(parsed, dict):
            raise RuntimeError(f"LLM did not return a JSON object for Integration Contract: {content[:500]}")
        return _normalize_contract_payload(parsed, generated_by="llm")

    def _render_user_content(
        self,
        *,
        tasks: list[dict[str, Any]],
        project_spec: str,
        requirements: list[dict[str, Any]],
        file_inventory: list[dict[str, Any]],
    ) -> str:
        return json.dumps(
            {
                "project_spec": project_spec,
                "requirements": requirements,
                "task_graph": {"tasks": tasks},
                "file_inventory": file_inventory,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc


@dataclass
class OpenAICompatibleIntegrationResultJudge:
    api_key: str
    base_url: str
    model: str
    timeout: int = 60
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "OpenAICompatibleIntegrationResultJudge":
        api_key = os.environ.get("LONG_AGENT_API_KEY")
        if not api_key:
            raise RuntimeError("Missing LONG_AGENT_API_KEY.")
        return cls(
            api_key=api_key,
            base_url=os.environ.get("LONG_AGENT_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            model=os.environ.get(
                "LONG_AGENT_INTEGRATION_RESULTS_MODEL",
                os.environ.get("LONG_AGENT_INTEGRATION_CONTRACT_MODEL", os.environ.get("LONG_AGENT_MODEL", "gpt-4.1-mini")),
            ),
            timeout=int(
                os.environ.get("LONG_AGENT_INTEGRATION_RESULTS_TIMEOUT", os.environ.get("LONG_AGENT_TIMEOUT", "60"))
            ),
            temperature=float(
                os.environ.get(
                    "LONG_AGENT_INTEGRATION_RESULTS_TEMPERATURE",
                    os.environ.get("LONG_AGENT_TEMPERATURE", "0.0"),
                )
            ),
        )

    def judge_all(
        self,
        *,
        contract_data: dict[str, Any],
        tasks: list[dict[str, Any]],
        static_results: list[dict[str, Any]],
        dynamic_results: list[dict[str, Any]],
        edge_results: list[dict[str, Any]],
        scenario_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": INTEGRATION_RESULT_JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "contract_context": _contract_context_for_judge(contract_data),
                            "task_context": _task_context_for_judge(tasks),
                            "local_evidence": {
                                "checks": _flatten_integration_checks(
                                    static_results=static_results,
                                    dynamic_results=dynamic_results,
                                    edge_results=edge_results,
                                    scenario_results=scenario_results,
                                ),
                                "static_checks": static_results,
                                "dynamic_checks": dynamic_results,
                                "module_edges": edge_results,
                                "required_scenarios": scenario_results,
                            },
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
        }
        response = self._post_chat_completions(payload)
        content = str(response["choices"][0]["message"].get("content") or "").strip()
        parsed = _loads_json_object(_strip_markdown_fence(content))
        if not isinstance(parsed, dict):
            extracted = _extract_first_json_object(content)
            parsed = _loads_json_object(extracted) if extracted else None
        if not isinstance(parsed, dict):
            raise RuntimeError(f"LLM integration results judge did not return valid JSON: {content[:500]}")
        return _normalize_batch_judgement(parsed)

    def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc


def evaluate_integration_contract(
    *,
    root: Path,
    state_dir: Path,
    tasks: list[dict[str, Any]],
    contract_data: dict[str, Any],
    benchmark_id: str = "",
    timeout: int = 180,
    result_judge: "OpenAICompatibleIntegrationResultJudge | None" = None,
) -> dict[str, Any]:
    validation_errors = validate_integration_contract(contract_data)
    result_judge = result_judge or _integration_result_judge_from_env()
    static_results = [
        _evaluate_static_check(root=root, check=check)
        for check in _object_list(contract_data.get("static_checks", []))
    ]
    dynamic_results = [
        _evaluate_dynamic_check(
            root=root,
            state_dir=state_dir,
            check=check,
            benchmark_id=benchmark_id,
            timeout=timeout,
        )
        for check in _object_list(contract_data.get("dynamic_checks", []))
    ]
    edge_results = _evaluate_module_edges(
        edges=_object_list(contract_data.get("module_edges", [])),
        checks=[*static_results, *dynamic_results],
    )
    scenario_results = [
        _evaluate_required_scenario(scenario, [*static_results, *dynamic_results], edge_results)
        for scenario in _object_list(contract_data.get("required_scenarios", []))
    ]
    if result_judge is not None:
        judged = _judge_integration_results_once(
            judge=result_judge,
            contract_data=contract_data,
            tasks=tasks,
            static_results=static_results,
            dynamic_results=dynamic_results,
            edge_results=edge_results,
            scenario_results=scenario_results,
        )
        static_results = judged["static_checks"]
        dynamic_results = judged["dynamic_checks"]
        edge_results = judged["module_edges"]
        scenario_results = judged["required_scenarios"]
    passed = (
        not validation_errors
        and all(item.get("passed") is True for item in static_results)
        and all(item.get("passed") is True for item in dynamic_results)
        and all(item.get("passed") is True for item in edge_results)
        and all(item.get("passed") is True for item in scenario_results)
    )
    checks = _flatten_integration_checks(
        static_results=static_results,
        dynamic_results=dynamic_results,
        edge_results=edge_results,
        scenario_results=scenario_results,
    )
    return {
        "kind": INTEGRATION_RESULTS_KIND,
        "version": 2,
        "source": "system_validation",
        "judged_by": "llm" if result_judge is not None else "harness",
        "judge_mode": "batch" if result_judge is not None else "local",
        "passed": passed,
        "validation_errors": validation_errors,
        "entry_points": _string_list(contract_data.get("entry_points")),
        "checks": checks,
        "summary": _integration_summary(passed, validation_errors, checks),
    }


def write_integration_results(state_dir: Path, results: dict[str, Any]) -> None:
    path = state_dir / "system_validation" / "integration_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def integration_result_errors(results: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for error in results.get("validation_errors", []):
        errors.append(f"Integration Contract validation failed: {error}")
    checks = results.get("checks", [])
    if isinstance(checks, list):
        for item in checks:
            if isinstance(item, dict) and item.get("passed") is not True:
                label = str(item.get("id", "check")).strip() or "check"
                check_type = str(item.get("type", "integration")).strip() or "integration"
                reason = str(item.get("reason", item.get("summary", "failed"))).strip()
                errors.append(f"Integration {check_type} check failed for {label}: {reason}")
    else:
        for section in ("module_edges", "static_checks", "dynamic_checks", "required_scenarios"):
            for item in results.get(section, []):
                if isinstance(item, dict) and item.get("passed") is not True:
                    label = str(item.get("id", item.get("edge_id", section))).strip() or section
                    reason = str(item.get("reason", item.get("summary", "failed"))).strip()
                    errors.append(f"Integration {section[:-1]} failed for {label}: {reason}")
    if results.get("passed") is not True:
        errors.append("Final integration validation requires every declared module edge and integration check to pass.")
    return errors


def _integration_result_judge_from_env() -> OpenAICompatibleIntegrationResultJudge | None:
    provider = os.environ.get("LONG_AGENT_INTEGRATION_RESULTS_PROVIDER", "").strip().lower()
    if provider in {"offline", "harness", "disabled", "false", "0"}:
        return None
    if provider == "openai-compatible" or (not provider and os.environ.get("LONG_AGENT_API_KEY")):
        return OpenAICompatibleIntegrationResultJudge.from_env()
    if provider:
        raise RuntimeError(f"Unsupported integration results provider: {provider}")
    return None


def _judge_integration_results_once(
    *,
    judge: OpenAICompatibleIntegrationResultJudge,
    contract_data: dict[str, Any],
    tasks: list[dict[str, Any]],
    static_results: list[dict[str, Any]],
    dynamic_results: list[dict[str, Any]],
    edge_results: list[dict[str, Any]],
    scenario_results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    try:
        judgement = judge.judge_all(
            contract_data=contract_data,
            tasks=tasks,
            static_results=static_results,
            dynamic_results=dynamic_results,
            edge_results=edge_results,
            scenario_results=scenario_results,
        )
    except Exception as exc:  # pragma: no cover - network/provider failures are environment dependent
        checks = _flatten_integration_checks(
            static_results=static_results,
            dynamic_results=dynamic_results,
            edge_results=edge_results,
            scenario_results=scenario_results,
        )
        judgement = {
            "checks": [_failure_verdict(item.get("id"), exc, item_type=item.get("type")) for item in checks],
        }
    verdicts = judgement.get("checks", [])
    return {
        "static_checks": _apply_batch_judgements(static_results, verdicts, id_key="id"),
        "dynamic_checks": _apply_batch_judgements(dynamic_results, verdicts, id_key="id"),
        "module_edges": _apply_batch_judgements(edge_results, verdicts, id_key="edge_id"),
        "required_scenarios": _apply_batch_judgements(
            scenario_results,
            verdicts,
            id_key="id",
        ),
    }


def _failure_verdict(identifier: object, exc: Exception, *, item_type: object = "") -> dict[str, Any]:
    return {
        "id": str(identifier or "").strip(),
        "type": str(item_type or "").strip(),
        "passed": False,
        "reason": f"LLM integration results judge failed: {exc}",
        "evidence": "",
    }


def _apply_batch_judgements(
    preliminary_results: list[dict[str, Any]],
    verdicts: object,
    *,
    id_key: str,
) -> list[dict[str, Any]]:
    verdict_by_id = {
        str(item.get(id_key, item.get("id", item.get("edge_id", "")))).strip(): item
        for item in _object_list(verdicts)
        if str(item.get(id_key, item.get("id", item.get("edge_id", "")))).strip()
    }
    results: list[dict[str, Any]] = []
    for preliminary in preliminary_results:
        identifier = str(preliminary.get(id_key, preliminary.get("id", preliminary.get("edge_id", "")))).strip()
        verdict = verdict_by_id.get(identifier)
        result = dict(preliminary)
        result["preliminary_passed"] = preliminary.get("passed")
        result["judged_by"] = "llm"
        if verdict is None:
            result["passed"] = False
            result["confidence"] = "low"
            result["reason"] = f"LLM batch judge omitted verdict for {identifier or id_key}."
            result["evidence_summary"] = ""
        else:
            normalized = _normalize_llm_verdict(verdict)
            result["passed"] = normalized["passed"]
            result["confidence"] = normalized["confidence"]
            result["reason"] = normalized["reason"] or "LLM judge returned no reason."
            result["evidence_summary"] = normalized["evidence_summary"]
        results.append(result)
    return results


def _normalize_batch_judgement(payload: dict[str, Any]) -> dict[str, Any]:
    checks = _object_list(payload.get("checks", []))
    if not checks:
        checks = [
            *[
                {**item, "type": "static"}
                for item in _object_list(payload.get("static_checks", []))
            ],
            *[
                {**item, "type": "dynamic"}
                for item in _object_list(payload.get("dynamic_checks", []))
            ],
            *[
                {**item, "id": str(item.get("edge_id", item.get("id", ""))).strip(), "type": "edge"}
                for item in _object_list(payload.get("module_edges", []))
            ],
            *[
                {**item, "type": "scenario"}
                for item in _object_list(payload.get("required_scenarios", []))
            ],
        ]
    return {
        "passed": payload.get("passed") is True,
        "summary": str(payload.get("summary", "")).strip(),
        "checks": checks,
    }


def _flatten_integration_checks(
    *,
    static_results: list[dict[str, Any]],
    dynamic_results: list[dict[str, Any]],
    edge_results: list[dict[str, Any]],
    scenario_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for item in static_results:
        checks.append(
            {
                "id": str(item.get("id", "")).strip(),
                "type": "static",
                "subject": str(item.get("target", "")).strip(),
                "passed": item.get("passed") is True,
                "reason": str(item.get("reason", "")).strip(),
                "evidence": _simple_static_evidence(item),
            }
        )
    for item in dynamic_results:
        checks.append(
            {
                "id": str(item.get("id", "")).strip(),
                "type": "dynamic",
                "subject": str(item.get("command", "")).strip(),
                "passed": item.get("passed") is True,
                "reason": str(item.get("reason", "")).strip(),
                "evidence": _simple_dynamic_evidence(item),
            }
        )
    for item in edge_results:
        checks.append(
            {
                "id": str(item.get("edge_id", "")).strip(),
                "type": "edge",
                "subject": f"{str(item.get('from', '')).strip()} -> {str(item.get('to', '')).strip()}",
                "passed": item.get("passed") is True,
                "reason": str(item.get("reason", "")).strip(),
                "evidence": _simple_edge_evidence(item),
            }
        )
    for item in scenario_results:
        checks.append(
            {
                "id": str(item.get("id", "")).strip(),
                "type": "scenario",
                "subject": str(item.get("name", item.get("entry_point", ""))).strip(),
                "passed": item.get("passed") is True,
                "reason": str(item.get("reason", "")).strip(),
                "evidence": str(item.get("evidence_summary", item.get("expected_observable", ""))).strip(),
            }
        )
    return checks


def _simple_static_evidence(item: dict[str, Any]) -> str:
    evidence = item.get("evidence", {})
    if isinstance(evidence, dict):
        excerpt = str(evidence.get("excerpt", "")).strip()
        if excerpt:
            return excerpt[:1000]
    contains = str(item.get("contains", "")).strip()
    return f"contains={contains}" if contains else ""


def _simple_dynamic_evidence(item: dict[str, Any]) -> str:
    output = str(item.get("output", "")).strip()
    if output:
        return output[-1000:]
    if "returncode" in item:
        return f"returncode={item.get('returncode')}"
    return ""


def _simple_edge_evidence(item: dict[str, Any]) -> str:
    covered_by = _string_list(item.get("covered_by", []))
    return "covered_by=" + ", ".join(covered_by) if covered_by else ""


def _normalize_llm_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    confidence = str(payload.get("confidence", "low")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return {
        "passed": payload.get("passed") is True,
        "confidence": confidence,
        "reason": str(payload.get("reason", "")).strip(),
        "evidence_summary": str(payload.get("evidence", payload.get("evidence_summary", ""))).strip(),
    }

def _contract_context_for_judge(contract_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "entry_points": _string_list(contract_data.get("entry_points")),
        "module_edges": _object_list(contract_data.get("module_edges", [])),
        "required_scenarios": _object_list(contract_data.get("required_scenarios", [])),
        "static_checks": _contract_check_summaries(contract_data.get("static_checks", [])),
        "dynamic_checks": _contract_check_summaries(contract_data.get("dynamic_checks", [])),
    }


def _contract_check_summaries(value: object) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for check in _object_list(value):
        summaries.append(
            {
                "id": str(check.get("id", "")).strip(),
                "kind": _check_kind(check),
                "path": str(check.get("path", check.get("target", ""))).strip(),
                "command": str(check.get("command", "")).strip(),
                "covers_edges": _string_list(check.get("covers_edges", [])),
            }
        )
    return summaries


def _task_context_for_judge(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for task in tasks[:80]:
        if not isinstance(task, dict):
            continue
        result.append(
            {
                "id": str(task.get("id", "")).strip(),
                "title": str(task.get("title", "")).strip(),
                "status": str(task.get("status", "")).strip(),
                "depends_on": _string_list(task.get("depends_on", [])),
                "requirement_ids": _string_list(task.get("requirement_ids", [])),
                "expected_artifacts": _string_list(task.get("expected_artifacts", []))[:20],
                "verification_commands": _string_list(task.get("verification_commands", []))[:10],
            }
        )
    return result

def validate_integration_contract(contract_data: dict[str, Any]) -> list[str]:
    if not isinstance(contract_data, dict) or not contract_data:
        return ["Integration Contract is missing for final system validation."]
    errors: list[str] = []
    if contract_data.get("kind") != INTEGRATION_CONTRACT_KIND:
        errors.append(f"Integration Contract kind must be '{INTEGRATION_CONTRACT_KIND}'.")
    if contract_data.get("version") != INTEGRATION_CONTRACT_VERSION:
        errors.append(f"Integration Contract version must be {INTEGRATION_CONTRACT_VERSION}.")
    for field in ("entry_points", "module_edges", "required_scenarios", "static_checks", "dynamic_checks"):
        if not isinstance(contract_data.get(field), list):
            errors.append(f"Integration Contract {field} must be a list.")
    return errors


def _normalize_contract_payload(payload: dict[str, Any], *, generated_by: str) -> dict[str, Any]:
    return {
        "kind": INTEGRATION_CONTRACT_KIND,
        "version": INTEGRATION_CONTRACT_VERSION,
        "generated_by": generated_by,
        "entry_points": _coerce_string_list(payload.get("entry_points")),
        "module_edges": [
            _normalize_module_edge(edge, index)
            for index, edge in enumerate(_object_list(payload.get("module_edges", [])), start=1)
        ],
        "required_scenarios": [
            _normalize_required_scenario(scenario, index)
            for index, scenario in enumerate(_object_list(payload.get("required_scenarios", [])), start=1)
        ],
        "static_checks": [
            _normalize_static_check(check, index)
            for index, check in enumerate(_object_list(payload.get("static_checks", [])), start=1)
        ],
        "dynamic_checks": [
            _normalize_dynamic_check(check, index)
            for index, check in enumerate(_object_list(payload.get("dynamic_checks", [])), start=1)
        ],
    }


def _normalize_module_edge(edge: dict[str, Any], index: int) -> dict[str, Any]:
    normalized = dict(edge)
    normalized["id"] = str(edge.get("id", f"edge-{index}")).strip() or f"edge-{index}"
    normalized["from"] = str(edge.get("from", edge.get("source", ""))).strip()
    normalized["to"] = str(edge.get("to", edge.get("target", ""))).strip()
    normalized["covered_by"] = _coerce_string_list(edge.get("covered_by", edge.get("covers", [])))
    if "reason" in edge:
        normalized["reason"] = str(edge.get("reason", "")).strip()
    return normalized


def _normalize_required_scenario(scenario: dict[str, Any], index: int) -> dict[str, Any]:
    normalized = dict(scenario)
    normalized["id"] = str(scenario.get("id", f"scenario-{index}")).strip() or f"scenario-{index}"
    normalized["name"] = str(scenario.get("name", scenario.get("title", normalized["id"]))).strip()
    normalized["entry_point"] = str(scenario.get("entry_point", scenario.get("entry", ""))).strip()
    normalized["steps"] = _coerce_string_list(scenario.get("steps"))
    normalized["expected_observable"] = str(
        scenario.get("expected_observable", scenario.get("expected", ""))
    ).strip()
    return normalized


def _normalize_static_check(check: dict[str, Any], index: int) -> dict[str, Any]:
    kind = _check_kind(check)
    if not kind.startswith("static"):
        kind = "static_artifact_exists"
    normalized = dict(check)
    normalized["id"] = str(check.get("id", check.get("name", f"static-{index}"))).strip() or f"static-{index}"
    normalized["kind"] = kind
    target = str(check.get("path", check.get("target", ""))).strip()
    if target:
        normalized["path"] = target
    contains = str(check.get("contains", check.get("text", ""))).strip()
    if contains:
        normalized["contains"] = contains
    if "covers_edges" in check:
        normalized["covers_edges"] = _coerce_string_list(check.get("covers_edges"))
    return normalized


def _normalize_dynamic_check(check: dict[str, Any], index: int) -> dict[str, Any]:
    normalized = dict(check)
    normalized["id"] = str(check.get("id", check.get("name", f"dynamic-{index}"))).strip() or f"dynamic-{index}"
    normalized["kind"] = "dynamic_command"
    normalized["command"] = str(check.get("command", "")).strip()
    working_directory = str(check.get("working_directory", check.get("cwd", ""))).strip()
    if working_directory:
        normalized["working_directory"] = working_directory
    if "covers_edges" in check:
        normalized["covers_edges"] = _coerce_string_list(check.get("covers_edges"))
    return normalized


def _evaluate_module_edges(edges: list[dict[str, Any]], checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks_by_id = {
        str(check.get("id", "")).strip(): check
        for check in checks
        if str(check.get("id", "")).strip()
    }
    results: list[dict[str, Any]] = []
    for index, edge in enumerate(edges, start=1):
        edge_id = str(edge.get("id", f"edge-{index}")).strip()
        source = str(edge.get("from", "")).strip()
        target = str(edge.get("to", "")).strip()
        covered_by = _string_list(edge.get("covered_by", edge.get("covers", [])))
        missing_checks = [check_id for check_id in covered_by if check_id not in checks_by_id]
        failed_checks = [
            check_id
            for check_id in covered_by
            if check_id in checks_by_id and checks_by_id[check_id].get("passed") is not True
        ]
        passed = bool(source and target) and not missing_checks and not failed_checks
        if not source or not target:
            reason = "module edge must include non-empty from and to."
        elif missing_checks:
            reason = "module edge references missing integration checks: " + ", ".join(missing_checks) + "."
        elif failed_checks:
            reason = "module edge is covered only by failed integration checks: " + ", ".join(failed_checks) + "."
        elif not covered_by:
            reason = "module edge is declared without explicit covering checks; accepted as a traceability edge."
        else:
            reason = "module edge has passing covering integration checks."
        results.append(
            {
                "edge_id": edge_id,
                "from": source,
                "to": target,
                "covered_by": covered_by,
                "passed": passed,
                "reason": reason,
            }
        )
    return results


def _evaluate_static_check(*, root: Path, check: dict[str, Any]) -> dict[str, Any]:
    check_id = str(check.get("id", check.get("name", "static-check"))).strip() or "static-check"
    kind = _check_kind(check)
    if kind in {"static_artifact_exists", "artifact_exists"}:
        target = str(check.get("path", check.get("target", ""))).strip()
        path = _inside_root(root, target)
        passed = bool(path and path.is_file())
        return {
            "id": check_id,
            "kind": kind,
            "target": target,
            "passed": passed,
            "reason": "artifact exists" if passed else "artifact does not exist",
        }
    if kind in {"static_text_contains", "text_contains"}:
        target = str(check.get("path", check.get("target", ""))).strip()
        needle = str(check.get("contains", check.get("text", ""))).strip()
        path = _inside_root(root, target)
        text = path.read_text(encoding="utf-8", errors="replace") if path and path.is_file() else ""
        passed = bool(needle and needle in text)
        excerpt = _static_evidence_excerpt(text, needle)
        return {
            "id": check_id,
            "kind": kind,
            "target": target,
            "contains": needle,
            "passed": passed,
            "reason": "text is present" if passed else "text is missing",
            "evidence": {
                "file_exists": bool(path and path.is_file()),
                "contains_found": passed,
                "excerpt": excerpt,
            },
        }
    return {
        "id": check_id,
        "kind": kind,
        "passed": False,
        "reason": f"unsupported static integration check kind: {kind}",
    }


def _evaluate_dynamic_check(
    *,
    root: Path,
    state_dir: Path,
    check: dict[str, Any],
    benchmark_id: str,
    timeout: int,
) -> dict[str, Any]:
    check_id = str(check.get("id", check.get("name", "dynamic-check"))).strip() or "dynamic-check"
    command = str(check.get("command", "")).strip()
    if not command:
        return {"id": check_id, "kind": _check_kind(check), "passed": False, "reason": "dynamic check has no command."}
    cwd = _dynamic_check_cwd(root, check, benchmark_id)
    env = os.environ.copy()
    temp_dir = state_dir / "system_validation" / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    env["TMPDIR"] = str(temp_dir)
    if benchmark_id:
        workspace = root / "eval" / "benchmarks" / benchmark_id / "workspace"
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join([str(workspace), current] if current else [str(workspace)])
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
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "id": check_id,
            "kind": _check_kind(check),
            "command": command,
            "passed": False,
            "reason": f"dynamic check timed out: {exc}",
        }
    except OSError as exc:
        return {
            "id": check_id,
            "kind": _check_kind(check),
            "command": command,
            "passed": False,
            "reason": f"dynamic check could not execute: {exc}",
        }
    output = (completed.stdout + completed.stderr).strip()
    if len(output) > 4000:
        output = output[-4000:]
    return {
        "id": check_id,
        "kind": _check_kind(check),
        "command": command,
        "working_directory": _display_path(root, cwd),
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "reason": f"dynamic check exited with code {completed.returncode}",
        "output": output,
    }


def _evaluate_required_scenario(
    scenario: dict[str, Any],
    check_results: list[dict[str, Any]],
    edge_results: list[dict[str, Any]],
) -> dict[str, Any]:
    scenario_id = str(scenario.get("id", scenario.get("name", "scenario"))).strip() or "scenario"
    if not check_results and not edge_results:
        return {
            "id": scenario_id,
            "name": str(scenario.get("name", scenario_id)).strip(),
            "passed": False,
            "reason": "required scenario has no integration checks or module edges to prove it.",
        }
    failed = [item for item in [*check_results, *edge_results] if item.get("passed") is not True]
    return {
        "id": scenario_id,
        "name": str(scenario.get("name", scenario_id)).strip(),
        "entry_point": str(scenario.get("entry_point", "")).strip(),
        "expected_observable": str(scenario.get("expected_observable", "")).strip(),
        "passed": not failed,
        "reason": "all related integration checks passed" if not failed else "related integration checks failed",
    }


def _static_evidence_excerpt(text: str, needle: str, max_chars: int = 4000) -> str:
    if not text:
        return ""
    if needle:
        index = text.find(needle)
        if index >= 0:
            start = max(0, index - max_chars // 2)
            end = min(len(text), start + max_chars)
            return text[start:end]
    return text[:max_chars]


def _dynamic_check_cwd(root: Path, check: dict[str, Any], benchmark_id: str) -> Path:
    working_directory = str(check.get("working_directory", check.get("cwd", ""))).strip()
    if benchmark_id:
        working_directory = _rewrite_benchmark_workspace(working_directory, benchmark_id)
    if working_directory:
        path = _inside_root(root, working_directory)
        if path:
            return path
    return root


def _integration_summary(
    passed: bool,
    validation_errors: list[str],
    checks: list[dict[str, Any]],
) -> str:
    if passed and not checks:
        return "Integration Contract is present; no explicit module edges or integration checks were declared."
    if passed:
        return "All declared integration checks passed."
    failed_count = len(validation_errors) + sum(1 for item in checks if item.get("passed") is not True)
    return f"{failed_count} integration validation issue(s) found."


def _check_kind(check: dict[str, Any]) -> str:
    return str(check.get("kind", check.get("type", "dynamic_command"))).strip().lower()


def _object_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _unique_strings(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _unique_objects(values: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(value))
    return result


def _inside_root(root: Path, target: str) -> Path | None:
    if not target:
        return None
    path = Path(target)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return resolved


def _rewrite_benchmark_workspace(path: str, benchmark_id: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if not normalized or not benchmark_id:
        return normalized
    parts = normalized.split("/")
    for index in range(len(parts) - 3):
        if parts[index] == "eval" and parts[index + 1] == "benchmarks" and parts[index + 3] == "workspace":
            parts[index + 2] = benchmark_id
            return "/".join(parts)
    return normalized


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
        if match:
            return match.group(1).strip()
    return stripped


def _loads_json_object(text: str | None) -> Any | None:
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return text[start : start + end]
