from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from agent.integration_contract import (
    evaluate_integration_contract,
    integration_result_errors,
    write_integration_results,
)
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
    ui_contract_group = parser.add_mutually_exclusive_group()
    ui_contract_group.add_argument(
        "--ui-contract",
        dest="ui_contract_validation",
        action="store_true",
        default=True,
        help="Require and evaluate the system-owned UI Contract. Enabled by default.",
    )
    ui_contract_group.add_argument(
        "--no-ui-contract",
        dest="ui_contract_validation",
        action="store_false",
        help="Skip UI Contract loading and UI validation checks.",
    )
    integration_contract_group = parser.add_mutually_exclusive_group()
    integration_contract_group.add_argument(
        "--integration-contract",
        dest="integration_contract_validation",
        action="store_true",
        default=False,
        help="Require and evaluate the system-owned Integration Contract. Disabled by default.",
    )
    integration_contract_group.add_argument(
        "--no-integration-contract",
        dest="integration_contract_validation",
        action="store_false",
        help="Skip Integration Contract loading and integration validation checks.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else Path.cwd().resolve()
    state_dir = _inside_root(root, args.state_dir)
    tasks_path = _inside_root(root, args.tasks_path)
    requirements_path = _inside_root(root, args.requirements_path)

    errors: list[str] = []
    tasks_data = _load_json(tasks_path, errors, "tasks")
    requirements_data = _load_optional_json(requirements_path, errors, "requirements")
    manifest_data = _load_optional_json(
        state_dir / "system_validation" / "final_acceptance_manifest.json",
        errors,
        "final acceptance manifest",
    )
    ui_contract_data = (
        _load_optional_json(state_dir / "system_validation" / "ui_contract.json", errors, "UI Contract")
        if args.ui_contract_validation
        else {}
    )
    integration_contract_data = (
        _load_optional_json(
            state_dir / "system_validation" / "integration_contract.json",
            errors,
            "Integration Contract",
        )
        if args.integration_contract_validation
        else {}
    )
    tasks = tasks_data.get("tasks", []) if isinstance(tasks_data, dict) else []
    if not isinstance(tasks, list):
        errors.append("tasks.tasks must be a list.")
        tasks = []
    errors.extend(
        _validate_manifest(
            manifest_data=manifest_data,
            state_dir=state_dir,
            tasks_path=tasks_path,
            requirements_path=requirements_path,
            ui_contract_validation=args.ui_contract_validation,
            integration_contract_validation=args.integration_contract_validation,
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

    if isinstance(requirements_data, dict) and requirements_data:
        if args.ui_contract_validation:
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

    if args.integration_contract_validation:
        integration_results = evaluate_integration_contract(
            root=root,
            state_dir=state_dir,
            tasks=[task for task in tasks if isinstance(task, dict)],
            contract_data=integration_contract_data,
            benchmark_id=args.benchmark_id,
            timeout=max(1, args.timeout),
        )
        write_integration_results(state_dir, integration_results)
        errors.extend(integration_result_errors(integration_results))

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
    ui_contract_validation: bool = True,
    integration_contract_validation: bool = False,
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
    }
    if ui_contract_validation:
        expected_refs["ui_contract_path"] = state_dir / "system_validation" / "ui_contract.json"
    if integration_contract_validation:
        expected_refs["integration_contract_path"] = state_dir / "system_validation" / "integration_contract.json"
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
    contract_by_id = {
        str(item.get("requirement_id", "")).strip(): item
        for item in contract_data.get("contracts", [])
        if isinstance(item, dict) and str(item.get("requirement_id", "")).strip()
    }
    ui_requirement_ids = {
        str(requirement.get("id", "")).strip()
        for requirement in requirements
        if isinstance(requirement, dict)
        and str(requirement.get("type", "")).strip().lower() in {"gui_workflow", "report"}
        and str(requirement.get("id", "")).strip()
    }
    verified_ids = _verified_requirement_ids(tasks=tasks, state_dir=state_dir)
    source_by_requirement = _ui_source_by_requirement(
        root=root,
        tasks=tasks,
        benchmark_id=benchmark_id,
        ui_requirement_ids=ui_requirement_ids,
    )
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


def _ui_source_by_requirement(
    *,
    root: Path,
    tasks: list[dict[str, Any]],
    benchmark_id: str,
    ui_requirement_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    files_by_requirement: dict[str, list[Path]] = {}
    targets_by_requirement: dict[str, list[str]] = {}
    fallback_files = _workspace_python_files(root=root, tasks=tasks, benchmark_id=benchmark_id)
    fallback_targets = [_display_path(root, path) for path in fallback_files]
    support_files = _workspace_ui_support_files(root=root, tasks=tasks, benchmark_id=benchmark_id)
    support_targets = [_display_path(root, path) for path in support_files]
    ui_requirement_ids = ui_requirement_ids or set()
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
        if requirement_id in ui_requirement_ids:
            files = _dedupe_paths([*files, *support_files])
            targets = _dedupe_strings([*targets, *support_targets])
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


def _workspace_ui_support_files(*, root: Path, tasks: list[dict[str, Any]], benchmark_id: str) -> list[Path]:
    workspace_path = _workspace_path(root=root, tasks=tasks, benchmark_id=benchmark_id)
    if workspace_path is None:
        return []
    support_files: list[Path] = []
    for path in _workspace_python_files(root=root, tasks=tasks, benchmark_id=benchmark_id):
        try:
            relative = path.relative_to(workspace_path).as_posix().lower()
        except ValueError:
            continue
        if relative == "main.py" or relative.endswith("/main.py") or "/ui/" in f"/{relative}":
            support_files.append(path)
    return support_files


def _workspace_python_files(*, root: Path, tasks: list[dict[str, Any]], benchmark_id: str) -> list[Path]:
    workspace_path = _workspace_path(root=root, tasks=tasks, benchmark_id=benchmark_id)
    if workspace_path is None:
        return []
    return [
        path
        for path in workspace_path.rglob("*.py")
        if "/tests/" not in str(path.relative_to(workspace_path)).replace("\\", "/").lower()
        and not path.name.startswith("test_")
    ]


def _workspace_path(*, root: Path, tasks: list[dict[str, Any]], benchmark_id: str) -> Path | None:
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
        return None
    if not workspace_path.is_dir():
        return None
    return workspace_path


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        if path not in result:
            result.append(path)
    return result


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


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
    if _field_claims_no_dedicated_ui(contract_text, field=field):
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


def _field_claims_no_dedicated_ui(text: str, *, field: str) -> bool:
    normalized = re.sub(r"[\s:：,，。；;（）()【】\[\]\"'`]+", "", text.lower())
    if not normalized:
        return False
    no_value_markers = {"n/a", "na", "none", "notapplicable", "不适用"}
    if normalized in no_value_markers:
        return field in {"buttons", "inputs", "dialogs"}
    if normalized in {"无", "无需", "无须", "不需要"}:
        return field in {"buttons", "inputs", "dialogs"}

    generic_markers = (
        "no dedicated",
        "not required",
        "无需专用",
        "无专用",
        "不需要专用",
        "不需要额外",
    )
    field_markers = {
        "buttons": (
            "no button required",
            "no buttons required",
            "no dedicated button",
            "无按钮",
            "无需按钮",
            "无须按钮",
            "不需要按钮",
        ),
        "inputs": (
            "no input required",
            "no dedicated input",
            "no dedicated inputs",
            "无输入",
            "无需输入",
            "无须输入",
            "无需用户输入",
            "不需要输入",
            "不需要用户输入",
        ),
        "dialogs": (
            "no modal required",
            "no dialog required",
            "no dedicated dialog",
            "inline feedback is acceptable",
            "inline ui feedback is acceptable",
            "inline form feedback is acceptable",
            "无弹窗",
            "无需弹窗",
            "无须弹窗",
            "无模态弹窗",
            "无需模态弹窗",
            "不需要弹窗",
            "不需要模态弹窗",
            "内联反馈",
            "行内反馈",
        ),
    }
    if any(marker in text for marker in generic_markers):
        return field in {"buttons", "inputs", "dialogs"}
    return any(marker in text for marker in field_markers.get(field, ()))


def _mentions_action_control(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "button",
            "menu",
            "click",
            "select",
            "dropdown",
            "按钮",
            "按鈕",
            "菜单",
            "點擊",
            "点击",
            "选择",
            "每行",
        )
    )


def _has_action_control(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "tk.button",
            "ttk.button",
            ".add_command(",
            "menubutton",
            "tk.optionmenu",
            "ttk.combobox",
            ".bind(",
            "command=",
            "<<listboxselect>>",
        )
    )


def _has_container_or_navigation(text: str) -> bool:
    return any(marker in text for marker in ("tk.frame", "ttk.frame", "tk.toplevel", "ttk.notebook", ".add(", "menu(", "tk.optionmenu"))


def _has_input_control(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "tk.entry",
            "ttk.entry",
            "listbox",
            "combobox",
            "tk.optionmenu",
            "optionmenu",
            "tk.stringvar",
            "ttk.combobox",
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
            "status_var.set(",
            ".set(",
            ".config(text=",
            ".configure(text=",
            "raise valueerror",
        )
    )


def _has_data_display(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "treeview",
            "listbox",
            "canvas",
            "tk.canvas",
            "tk.text",
            "ttk.label",
            "tk.label",
            ".insert(",
            "create_arc",
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
