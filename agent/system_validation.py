from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from agent.requirement_verifier import project_requirement_evidence_errors
from agent.requirement_verifier import load_task_requirement_evidence
from agent.ui_contract import UI_CONTRACT_APPLICABILITY_VALUES, UI_CONTRACT_REQUIRED_FIELDS, validate_ui_contract


FINAL_ACCEPTANCE_TASK_ID = "FINAL_ACCEPTANCE"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run harness-owned final project validation.")
    parser.add_argument("--root", default="")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--tasks-path", required=True)
    parser.add_argument("--requirements-path", required=True)
    parser.add_argument("--benchmark-id", default="")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else Path.cwd().resolve()
    state_dir = _inside_root(root, args.state_dir)
    tasks_path = _inside_root(root, args.tasks_path)
    requirements_path = _inside_root(root, args.requirements_path)

    errors: list[str] = []
    tasks_data = _load_json(tasks_path, errors, "tasks")
    requirements_data = _load_optional_json(requirements_path, errors, "requirements")
    state_data = _load_optional_json(state_dir / "current_task.json", errors, "current task state")
    manifest_data = _load_optional_json(
        state_dir / "system_validation" / "final_acceptance_manifest.json",
        errors,
        "final acceptance manifest",
    )
    ui_contract_data = _load_optional_json(state_dir / "system_validation" / "ui_contract.json", errors, "UI Contract")
    tasks = tasks_data.get("tasks", []) if isinstance(tasks_data, dict) else []
    if not isinstance(tasks, list):
        errors.append("tasks.tasks must be a list.")
        tasks = []
    contracts = _agreed_contracts_by_task_id(state_data)
    errors.extend(
        _validate_manifest(
            manifest_data=manifest_data,
            state_dir=state_dir,
            tasks_path=tasks_path,
            requirements_path=requirements_path,
        )
    )

    required_tasks = [
        task
        for task in tasks
        if isinstance(task, dict)
        and task.get("optional") is not True
        and str(task.get("id", "")).strip() != FINAL_ACCEPTANCE_TASK_ID
    ]
    for task in required_tasks:
        task_id = str(task.get("id", "")).strip() or "<unknown>"
        status = str(task.get("status", "pending")).strip().lower()
        if status not in {"completed", "done"}:
            errors.append(f"{task_id} is not completed before final validation.")

    command_failures = _run_task_verification_commands(
        root=root,
        state_dir=state_dir,
        tasks=required_tasks,
        contracts=contracts,
        benchmark_id=args.benchmark_id,
        timeout=max(1, args.timeout),
    )
    errors.extend(command_failures)

    if isinstance(requirements_data, dict) and requirements_data:
        if isinstance(ui_contract_data, dict) and ui_contract_data:
            ui_contract_requirements = _final_acceptance_requirements(tasks) or requirements_data.get("requirements", [])
            errors.extend(validate_ui_contract({"requirements": ui_contract_requirements}, ui_contract_data))
            ui_check_results = _evaluate_ui_contract_checks(
                root=root,
                requirements=ui_contract_requirements,
                contract_data=ui_contract_data,
                tasks=tasks,
                state_dir=state_dir,
                benchmark_id=args.benchmark_id,
            )
            _write_ui_check_results(state_dir, ui_check_results)
            errors.extend(_ui_check_errors(ui_check_results))
        else:
            errors.append("UI Contract is missing for final system validation.")
        errors.extend(
            project_requirement_evidence_errors(
                requirements_data=requirements_data,
                tasks=[task for task in tasks if isinstance(task, dict)],
                state_dir=state_dir,
            )
        )

    if errors:
        _print_line("SYSTEM_VALIDATION_FAIL")
        for error in errors:
            _print_line(f"- {error}")
        return 1
    _print_line("SYSTEM_VALIDATION_PASS")
    return 0


def _final_acceptance_requirements(tasks: list[Any]) -> list[dict[str, Any]]:
    for task in tasks:
        if not isinstance(task, dict) or str(task.get("id", "")).strip() != FINAL_ACCEPTANCE_TASK_ID:
            continue
        requirements = task.get("requirements", [])
        if isinstance(requirements, list):
            return [dict(item) for item in requirements if isinstance(item, dict)]
        return []
    return []


def _validate_manifest(
    *,
    manifest_data: dict[str, Any],
    state_dir: Path,
    tasks_path: Path,
    requirements_path: Path,
) -> list[str]:
    if not isinstance(manifest_data, dict) or not manifest_data:
        return ["Final acceptance manifest is missing for final system validation."]
    errors: list[str] = []
    if manifest_data.get("kind") != "system_owned_final_acceptance":
        errors.append("Final acceptance manifest kind must be 'system_owned_final_acceptance'.")
    if manifest_data.get("validator") != "agent.system_validation":
        errors.append("Final acceptance manifest validator must be agent.system_validation.")
    expected_refs = {
        "tasks_path": tasks_path,
        "requirements_path": requirements_path,
        "ui_contract_path": state_dir / "system_validation" / "ui_contract.json",
    }
    for key, expected_path in expected_refs.items():
        raw = str(manifest_data.get(key, "")).replace("\\", "/").strip()
        if not raw:
            errors.append(f"Final acceptance manifest {key} is missing.")
            continue
        if Path(raw).name != expected_path.name:
            errors.append(f"Final acceptance manifest {key} should reference {expected_path.name}.")
    return errors


def _evaluate_ui_contract_checks(
    *,
    root: Path,
    requirements: list[Any],
    contract_data: dict[str, Any],
    tasks: list[dict[str, Any]],
    state_dir: Path,
    benchmark_id: str = "",
) -> dict[str, Any]:
    verified_ids = _verified_requirement_ids(tasks=tasks, state_dir=state_dir)
    source_by_requirement = _ui_source_by_requirement(root=root, tasks=tasks, benchmark_id=benchmark_id)
    contract_by_id = {
        str(item.get("requirement_id", "")).strip(): item
        for item in contract_data.get("contracts", [])
        if isinstance(item, dict) and str(item.get("requirement_id", "")).strip()
    }
    results: list[dict[str, Any]] = []
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        requirement_id = str(requirement.get("id", "")).strip()
        if not requirement_id:
            continue
        contract = contract_by_id.get(requirement_id, {})
        ui_contract = contract.get("ui_contract", {}) if isinstance(contract, dict) else {}
        source_info = source_by_requirement.get(requirement_id, {})
        source_text = str(source_info.get("text", "")) if isinstance(source_info, dict) else ""
        source_files = source_info.get("files", []) if isinstance(source_info, dict) else []
        if not isinstance(source_files, list):
            source_files = []
        source_files = [str(item) for item in source_files if str(item).strip()]
        requirement_type = str(requirement.get("type", "")).strip().lower()
        ui_applicability = _contract_ui_applicability(contract)
        ui_surface = _contract_ui_surface(contract)
        ui_applicable = ui_applicability == "required"
        if ui_applicable:
            check_details = {
                field: _evaluate_ui_field(
                    field=field,
                    value=ui_contract.get(field) if isinstance(ui_contract, dict) else None,
                    source_text=source_text,
                    has_verified_evidence=requirement_id in verified_ids,
                )
                for field in UI_CONTRACT_REQUIRED_FIELDS
            }
        else:
            check_details = {
                field: {
                    "passed": True,
                    "reason": (
                        f"UI widget check is {ui_applicability} for "
                        f"surface {ui_surface}; full UI field checks apply only when ui_applicability is required"
                    ),
                }
                for field in UI_CONTRACT_REQUIRED_FIELDS
            }
        checks = {field: detail["passed"] for field, detail in check_details.items()}
        results.append(
            {
                "requirement_id": requirement_id,
                "requirement_type": requirement_type,
                "ui_applicability": ui_applicability,
                "ui_surface": ui_surface,
                "ui_check_applicable": ui_applicable,
                "source_files": source_files,
                "repair_targets": source_files if ui_applicable and not all(checks.values()) else [],
                "required_action": (
                    "inspect_and_repair_generated_code" if ui_applicable and not all(checks.values()) else ""
                ),
                "checks": checks,
                "details": check_details,
                "passed": all(checks.values()),
            }
        )
    return {
        "kind": "ui_check_results",
        "version": 1,
        "source": "system_validation",
        "results": results,
        "passed": all(item.get("passed") is True for item in results) if results else False,
    }


def _contract_ui_applicability(contract: object) -> str:
    if not isinstance(contract, dict):
        return "required"
    value = str(contract.get("ui_applicability", "")).strip().lower()
    if value in UI_CONTRACT_APPLICABILITY_VALUES:
        return value
    return "required"


def _contract_ui_surface(contract: object) -> str:
    if not isinstance(contract, dict):
        return "widget"
    value = str(contract.get("ui_surface", "")).strip().lower()
    return value or "widget"


def _verified_requirement_ids(*, tasks: list[dict[str, Any]], state_dir: Path) -> set[str]:
    verified: set[str] = set()
    for task in tasks:
        task_id = str(task.get("id", "")).strip()
        if not task_id or task_id == FINAL_ACCEPTANCE_TASK_ID:
            continue
        evidence = load_task_requirement_evidence(state_dir, task_id)
        if not isinstance(evidence, dict):
            continue
        for item in evidence.get("requirements", []):
            if isinstance(item, dict) and item.get("status") == "verified":
                requirement_id = str(item.get("id", "")).strip()
                if requirement_id:
                    verified.add(requirement_id)
    return verified


def _ui_source_by_requirement(*, root: Path, tasks: list[dict[str, Any]], benchmark_id: str) -> dict[str, dict[str, Any]]:
    files_by_requirement: dict[str, list[Path]] = {}
    targets_by_requirement: dict[str, list[str]] = {}
    fallback_files = _workspace_python_files(root=root, tasks=tasks, benchmark_id=benchmark_id)
    fallback_targets = [_display_path(root, path) for path in fallback_files]
    for task in tasks:
        if not isinstance(task, dict) or task.get("final_acceptance") is True:
            continue
        requirement_ids = _string_list(task.get("requirement_ids"))
        if not requirement_ids:
            continue
        targets = _task_ui_source_targets(root=root, task=task, benchmark_id=benchmark_id)
        for requirement_id in requirement_ids:
            path_bucket = files_by_requirement.setdefault(requirement_id, [])
            target_bucket = targets_by_requirement.setdefault(requirement_id, [])
            for path, target in targets:
                if target not in target_bucket:
                    target_bucket.append(target)
                if path.is_file() and path not in path_bucket:
                    path_bucket.append(path)
    result: dict[str, dict[str, Any]] = {}
    all_requirement_ids = {
        requirement_id
        for task in tasks
        if isinstance(task, dict)
        for requirement_id in _string_list(task.get("requirement_ids"))
    }
    for requirement_id in all_requirement_ids:
        files = files_by_requirement.get(requirement_id) or fallback_files
        targets = targets_by_requirement.get(requirement_id) or fallback_targets
        result[requirement_id] = {
            "files": targets,
            "text": "\n".join(_read_source_file(path) for path in files),
        }
    return result


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _task_ui_source_files(*, root: Path, task: dict[str, Any], benchmark_id: str) -> list[Path]:
    return [path for path, _target in _task_ui_source_targets(root=root, task=task, benchmark_id=benchmark_id) if path.is_file()]


def _task_ui_source_targets(*, root: Path, task: dict[str, Any], benchmark_id: str) -> list[tuple[Path, str]]:
    paths: list[Path] = []
    targets: list[tuple[Path, str]] = []
    for key in ("implementation_artifacts", "expected_artifacts"):
        for artifact in _string_list(task.get(key)):
            normalized = _rewrite_benchmark_workspace(artifact, benchmark_id)
            path = (root / normalized).resolve()
            try:
                path.relative_to(root.resolve())
            except (OSError, ValueError):
                continue
            if path.suffix.lower() != ".py":
                continue
            comparable = normalized.replace("\\", "/").lower()
            if "/tests/" in comparable or Path(comparable).name.startswith("test_"):
                continue
            if path.suffix.lower() != ".py":
                continue
            target = _display_path(root, path)
            if path not in paths:
                paths.append(path)
                targets.append((path, target))
    return targets


def _workspace_python_files(*, root: Path, tasks: list[dict[str, Any]], benchmark_id: str) -> list[Path]:
    workspace = ""
    if benchmark_id:
        workspace = f"eval/benchmarks/{benchmark_id}/workspace"
    if not workspace:
        for task in tasks:
            if isinstance(task, dict):
                workspace = _rewrite_benchmark_workspace(_workspace_root_from_task(task), benchmark_id)
                if workspace:
                    break
    workspace_path = (root / workspace).resolve() if workspace else root.resolve()
    try:
        workspace_path.relative_to(root.resolve())
    except (OSError, ValueError):
        return []
    if not workspace_path.is_dir():
        return []
    return [
        path
        for path in workspace_path.rglob("*.py")
        if "/tests/" not in str(path.relative_to(workspace_path)).replace("\\", "/").lower()
        and not path.name.startswith("test_")
    ]


def _read_source_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _evaluate_ui_field(
    *,
    field: str,
    value: object,
    source_text: str,
    has_verified_evidence: bool,
) -> dict[str, Any]:
    if not has_verified_evidence:
        return {"passed": False, "reason": "requirement has no verified task evidence"}
    if not _ui_contract_field_has_value(value):
        return {"passed": False, "reason": "ui_contract field is empty"}
    if not source_text.strip():
        return {"passed": False, "reason": "no implementation UI source was found for this requirement"}

    text = source_text.lower()
    contract_text = _field_text(value).lower()
    if _field_claims_no_dedicated_ui(contract_text):
        return {"passed": True, "reason": "ui_contract explicitly states no dedicated UI element is required"}

    if field == "entry_points":
        if _mentions_action_control(contract_text):
            return _source_check(_has_action_control(text), "source contains an action control", "source has no button/menu command for the entry point")
        return _source_check(_has_container_or_navigation(text), "source contains a UI container/navigation element", "source has no visible container/navigation element")
    if field == "buttons":
        return _source_check(_has_action_control(text), "source contains a button or menu command", "source has no button or menu command")
    if field == "inputs":
        return _source_check(_has_input_control(text), "source contains an input control", "source has no Entry/Listbox/Combobox/Scale/Spinbox/Text input control")
    if field == "dialogs":
        return _source_check(_has_dialog_or_feedback(text), "source contains dialog or inline feedback code", "source has no messagebox/Toplevel/status feedback")
    if field == "data_display":
        return _source_check(_has_data_display(text), "source contains a data display widget", "source has no Treeview/Listbox/Canvas/Text/Label display widget")
    if field == "empty_state":
        return _source_check(_has_empty_state(text), "source contains empty-state UI text or state handling", "source has no empty-state UI evidence")
    if field == "success_refresh":
        return _source_check(_has_success_refresh(text), "source contains visible refresh/update behavior", "source has no visible refresh/update behavior")
    return {"passed": False, "reason": f"unknown UI field {field}"}


def _source_check(passed: bool, ok_reason: str, fail_reason: str) -> dict[str, Any]:
    return {"passed": bool(passed), "reason": ok_reason if passed else fail_reason}


def _ui_contract_field_has_value(value: object) -> bool:
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    return bool(str(value or "").strip())


def _field_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value or "")


def _field_claims_no_dedicated_ui(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "no dedicated",
            "no modal required",
            "no input required",
            "no button required",
            "not required",
            "无 dedicated",
            "无;",
            "无；",
            "无需",
            "不需要",
            "无输入",
        )
    )


def _mentions_action_control(text: str) -> bool:
    return any(marker in text for marker in ("button", "menu", "click", "按钮", "按鈕", "菜单", "點擊", "点击", "每行"))


def _has_action_control(text: str) -> bool:
    return any(marker in text for marker in ("tk.button", "ttk.button", ".add_command(", "menubutton"))


def _has_container_or_navigation(text: str) -> bool:
    return any(marker in text for marker in ("tk.frame", "ttk.frame", "tk.toplevel", "ttk.notebook", ".add(", "menu("))


def _has_input_control(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "tk.entry",
            "ttk.entry",
            "listbox",
            "combobox",
            "tk.scale",
            "ttk.scale",
            "spinbox",
            "tk.text",
            "ttk.checkbutton",
            "tk.checkbutton",
            "radiobutton",
        )
    )


def _has_dialog_or_feedback(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "messagebox.",
            "tk.toplevel",
            "status_label",
            "status_var",
            ".config(text=",
            ".configure(text=",
        )
    )


def _has_data_display(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "treeview",
            "listbox",
            "canvas",
            "tk.text",
            "ttk.label",
            "tk.label",
            ".insert(",
            ".create_text(",
            ".create_rectangle(",
        )
    )


def _has_empty_state(text: str) -> bool:
    return _has_data_display(text) and (
        any(marker in text for marker in ("empty", "no ", "none", "暂无", "无", "沒有", "没有"))
        or bool(re.search(r"if\s+not\s+\w+", text))
    )


def _has_success_refresh(text: str) -> bool:
    return _has_data_display(text) and any(
        marker in text
        for marker in (
            "refresh",
            "_refresh",
            "update",
            "_update",
            "_load_from_store",
            ".delete(",
            ".insert(",
            ".config(",
            ".configure(",
            "save_data(",
        )
    )


def _write_ui_check_results(state_dir: Path, results: dict[str, Any]) -> None:
    path = state_dir / "system_validation" / "ui_check_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ui_check_errors(results: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for item in results.get("results", []):
        if not isinstance(item, dict):
            continue
        if item.get("passed") is True:
            continue
        requirement_id = str(item.get("requirement_id", "")).strip() or "<unknown>"
        checks = item.get("checks", {})
        failed_fields = [
            field
            for field, passed in checks.items()
            if passed is not True
        ] if isinstance(checks, dict) else list(UI_CONTRACT_REQUIRED_FIELDS)
        repair_targets = [
            str(target)
            for target in item.get("repair_targets", [])
            if str(target).strip()
        ] if isinstance(item.get("repair_targets", []), list) else []
        target_text = " repair_targets: " + ", ".join(repair_targets) if repair_targets else ""
        errors.append(
            f"UI check failed for {requirement_id}: "
            + ", ".join(failed_fields)
            + ". Inspect and repair generated code files."
            + target_text
        )
    if results.get("passed") is not True:
        errors.append(
            "Final UI validation requires every UI Contract check for every requirement to pass; "
            "when a field is false, inspect and repair the listed generated workspace code before retrying."
        )
    return errors


def _print_line(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe)


def _inside_root(root: Path, target: str) -> Path:
    raw = Path(target)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    path.relative_to(root)
    return path


def _load_json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label} could not be loaded from {path}: {exc}")
        return {}
    if not isinstance(loaded, dict):
        errors.append(f"{label} must be a JSON object.")
        return {}
    return loaded


def _load_optional_json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load_json(path, errors, label)


def _run_task_verification_commands(
    *,
    root: Path,
    state_dir: Path,
    tasks: list[dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
    benchmark_id: str,
    timeout: int,
) -> list[str]:
    errors: list[str] = []
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
    for task in tasks:
        task_id = str(task.get("id", "")).strip() or "<unknown>"
        procedures = _verification_procedures(task, contracts.get(task_id))
        if not procedures:
            errors.append(f"{task_id} has no frozen verification_commands for final validation.")
            continue
        for procedure in procedures:
            command = str(procedure.get("command", "")).strip()
            cwd = _procedure_cwd(root, procedure, task, benchmark_id)
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
                errors.append(f"{task_id} verification timed out: {command}\n{exc}")
                continue
            except OSError as exc:
                errors.append(f"{task_id} verification could not execute: {command}\n{exc}")
                continue
            if completed.returncode != 0:
                output = (completed.stdout + completed.stderr).strip()
                if len(output) > 4000:
                    output = output[:4000] + "\n...<truncated>"
                errors.append(
                    f"{task_id} verification failed with code {completed.returncode}: {command} (cwd={_display_path(root, cwd)})\n{output}"
                )
    return errors


def _verification_procedures(
    task: dict[str, Any],
    contract: dict[str, Any] | None,
) -> list[dict[str, str]]:
    procedures = _contract_verification_procedures(contract)
    if procedures:
        return procedures
    return [{"command": command} for command in _verification_commands(task)]


def _contract_verification_procedures(contract: dict[str, Any] | None) -> list[dict[str, str]]:
    if not isinstance(contract, dict):
        return []
    procedure = contract.get("verification_procedure")
    if isinstance(procedure, dict):
        working_directory = str(procedure.get("working_directory", "")).strip()
        commands = procedure.get("commands")
        if isinstance(commands, list):
            return [
                {"command": str(command.get("command", "") if isinstance(command, dict) else command).strip(), "working_directory": working_directory}
                for command in commands
                if str(command.get("command", "") if isinstance(command, dict) else command).strip()
            ]
        command = str(procedure.get("command", "")).strip()
        if command:
            return [{"command": command, "working_directory": working_directory}]
    return [{"command": command} for command in _verification_commands(contract)]


def _verification_commands(item: dict[str, Any]) -> list[str]:
    raw = item.get("verification_commands", item.get("checks", []))
    if not isinstance(raw, list):
        return []
    commands: list[str] = []
    for item in raw:
        command = str(item.get("command", "") if isinstance(item, dict) else item).strip()
        if command:
            commands.append(command)
    return commands


def _agreed_contracts_by_task_id(state_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = state_data.get("acceptance_contracts", [])
    if not isinstance(raw, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict) or item.get("status") != "agreed":
            continue
        task_id = str(item.get("task_id", "")).strip()
        if task_id:
            result[task_id] = item
    return result


def _procedure_cwd(
    root: Path,
    procedure: dict[str, str],
    task: dict[str, Any],
    benchmark_id: str,
) -> Path:
    working_directory = _rewrite_benchmark_workspace(
        str(procedure.get("working_directory", "")).strip(),
        benchmark_id,
    )
    if working_directory:
        return _inside_root(root, working_directory)
    inferred = _rewrite_benchmark_workspace(_workspace_root_from_task(task), benchmark_id)
    if inferred and (root / inferred).is_dir():
        return _inside_root(root, inferred)
    return root


def _rewrite_benchmark_workspace(path: str, benchmark_id: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if not normalized or not benchmark_id:
        return normalized
    marker = "/workspace"
    parts = normalized.split("/")
    for index in range(len(parts) - 2):
        if parts[index] == "eval" and parts[index + 1] == "benchmarks" and parts[index + 3 : index + 4] == ["workspace"]:
            parts[index + 2] = benchmark_id
            return "/".join(parts)
    if normalized.startswith("eval/benchmarks/") and marker in normalized:
        return f"eval/benchmarks/{benchmark_id}/workspace"
    return normalized


def _workspace_root_from_task(task: dict[str, Any]) -> str:
    for key in ("expected_artifacts", "implementation_artifacts", "worker_test_artifacts"):
        raw = task.get(key, [])
        if not isinstance(raw, list):
            continue
        for item in raw:
            normalized = str(item).replace("\\", "/")
            marker = "/workspace/"
            if marker in normalized:
                return normalized.split(marker, 1)[0] + "/workspace"
    return ""


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
