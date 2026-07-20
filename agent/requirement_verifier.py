from __future__ import annotations

import json
from pathlib import Path
from typing import Any


WEAK_TEST_MARKERS = (
    "pytest.skip(",
    "@pytest.mark.skip",
    "@pytest.mark.xfail",
    "unittest.skip(",
    "@unittest.skip",
    "assert true",
    "self.asserttrue(true",
    "hasattr(",
    "assert app is not none",
    "assert callable(",
    "assert isinstance(result, dict)",
    "assert isinstance(report, dict)",
    "assert isinstance(conflicts, list)",
)
GUI_FINAL_ACCEPTANCE_MARKERS = (
    ".invoke(",
    "event_generate(",
    "winfo_children(",
    "tabs()",
    "nametowidget(",
    "assertgreater(",
    "assert greater",
    "assert len(",
)


def task_evidence_path(state_dir: Path, task_id: str) -> Path:
    return state_dir / "task_evidence" / f"{_safe_id(task_id)}.json"


def write_task_requirement_evidence(
    *,
    root: Path,
    state_dir: Path,
    task: dict[str, Any],
    command_results: list[dict[str, Any]],
    contract: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    task_id = str(task.get("id", "")).strip()
    requirement_ids = _normalized_list(task.get("requirement_ids"))
    if not task_id or not requirement_ids:
        return None
    passed_commands = {str(item.get("command", "")).strip() for item in command_results if item.get("ok") is True}
    if not passed_commands:
        return None
    assets = _verification_assets(task)
    commands = _verification_command_records(task.get("verification_commands", []))
    contract_entries = _contract_evidence_entries_by_requirement(
        task=task,
        contract=contract,
        passed_commands=passed_commands,
    )
    requirements: list[dict[str, Any]] = []
    for requirement_id in requirement_ids:
        entries: list[dict[str, Any]] = []
        for command in commands:
            command_text = str(command.get("command", "")).strip()
            if command_text not in passed_commands:
                continue
            covers = set(_normalized_list(command.get("covers")))
            if requirement_id not in covers:
                continue
            linked_assets = [
                asset
                for asset in assets
                if str(asset.get("id", "")).strip() in set(_normalized_list(command.get("asset_ids")))
                and requirement_id in set(_normalized_list(asset.get("covers")))
            ]
            assertion_targets = _assertion_targets_for_requirement(linked_assets, requirement_id)
            entries.append(
                {
                    "type": "automated_test",
                    "command": command_text,
                    "test_files": [str(asset.get("path", "")).strip() for asset in linked_assets],
                    "result": "passed",
                    "assertion_targets": assertion_targets,
                }
            )
        for entry in contract_entries.get(requirement_id, []):
            if not any(
                existing.get("command") == entry.get("command")
                and existing.get("assertion_targets") == entry.get("assertion_targets")
                for existing in entries
            ):
                entries.append(entry)
        requirements.append(
            {
                "id": requirement_id,
                "status": "verified" if entries else "unverified",
                "evidence": entries,
            }
        )
    payload = {
        "task_id": task_id,
        "requirements": requirements,
    }
    path = task_evidence_path(state_dir, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def validate_task_requirement_closeout(
    *,
    root: Path,
    task: dict[str, Any],
    evidence: dict[str, Any] | None,
    command_results: list[dict[str, Any]],
) -> list[str]:
    task_id = str(task.get("id", "")).strip()
    requirement_ids = _normalized_list(task.get("requirement_ids"))
    if not requirement_ids:
        return []
    errors: list[str] = []
    if not isinstance(evidence, dict):
        return [f"{task_id} has no structured requirement evidence."]
    if str(evidence.get("task_id", "")).strip() != task_id:
        errors.append(f"{task_id} evidence task_id does not match active task.")
    passed_commands = {str(item.get("command", "")).strip() for item in command_results if item.get("ok") is True}
    requirements = evidence.get("requirements", [])
    if not isinstance(requirements, list):
        return [f"{task_id} evidence.requirements must be a list."]
    evidence_by_id = {
        str(item.get("id", "")).strip(): item
        for item in requirements
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }
    frozen_targets = _frozen_targets_by_requirement(task)
    for requirement_id in requirement_ids:
        item = evidence_by_id.get(requirement_id)
        if not item:
            errors.append(f"{task_id} evidence is missing requirement {requirement_id}.")
            continue
        if item.get("status") != "verified":
            errors.append(f"{task_id} requirement {requirement_id} is not verified.")
        entries = item.get("evidence", [])
        if not isinstance(entries, list) or not entries:
            errors.append(f"{task_id} requirement {requirement_id} has no evidence entries.")
            continue
        target_union: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                errors.append(f"{task_id} requirement {requirement_id} has invalid evidence entry.")
                continue
            command = str(entry.get("command", "")).strip()
            if command not in passed_commands:
                errors.append(f"{task_id} requirement {requirement_id} evidence command did not pass this verifier run.")
            test_files = _normalized_list(entry.get("test_files"))
            if _requirement_needs_test_asset(task, requirement_id) and not test_files:
                errors.append(f"{task_id} requirement {requirement_id} needs test-file evidence.")
            for test_file in test_files:
                path = (root / test_file).resolve()
                try:
                    path.relative_to(root.resolve())
                except (OSError, ValueError):
                    errors.append(f"{task_id} evidence test file is outside workspace: {test_file}.")
                    continue
                if not path.is_file():
                    errors.append(f"{task_id} evidence test file does not exist: {test_file}.")
            target_union.update(str(target).strip() for target in entry.get("assertion_targets", []) if str(target).strip())
        missing_targets = [target for target in frozen_targets.get(requirement_id, []) if target not in target_union]
        if missing_targets:
            errors.append(
                f"{task_id} requirement {requirement_id} evidence does not cover frozen assertion targets: "
                + ", ".join(missing_targets)
                + "."
            )
    errors.extend(validate_verification_asset_files(root=root, task=task))
    if task.get("final_acceptance") is True:
        errors.extend(validate_final_acceptance_test_files(root=root, task=task))
    return errors


def validate_verification_asset_files(*, root: Path, task: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for asset in _verification_assets(task):
        path_text = str(asset.get("path", "")).strip()
        if not path_text:
            continue
        path = (root / path_text).resolve()
        try:
            path.relative_to(root.resolve())
        except (OSError, ValueError):
            errors.append(f"Verification asset is outside workspace: {path_text}.")
            continue
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for marker in WEAK_TEST_MARKERS:
            if marker in text:
                errors.append(f"Verification asset {path_text} contains forbidden weak test marker: {marker}.")
    return errors


def validate_final_acceptance_test_files(*, root: Path, task: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    test_files = _final_acceptance_artifacts(task)
    if not test_files:
        return ["Final acceptance task must declare a project-level frozen or acceptance validation artifact."]
    gui_like = _task_is_gui_like(task)
    for test_file in test_files:
        path = (root / test_file).resolve()
        try:
            path.relative_to(root.resolve())
        except (OSError, ValueError):
            errors.append(f"Final acceptance test file is outside workspace: {test_file}.")
            continue
        if not path.is_file():
            errors.append(f"Final acceptance test file does not exist: {test_file}.")
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for marker in WEAK_TEST_MARKERS:
            if marker in text:
                errors.append(f"Final acceptance test {test_file} contains forbidden weak test marker: {marker}.")
        if "assert" not in text:
            errors.append(f"Final acceptance test {test_file} must contain concrete assertions.")
        if gui_like and not any(marker in text for marker in GUI_FINAL_ACCEPTANCE_MARKERS):
            errors.append(
                f"Final acceptance GUI test {test_file} must exercise visible widgets, GUI events, or observable widget state."
            )
    return errors


def _final_acceptance_artifacts(task: dict[str, Any]) -> list[str]:
    artifacts: list[str] = []
    for key in ("frozen_acceptance_artifacts", "acceptance_artifacts", "worker_test_artifacts"):
        for item in _normalized_list(task.get(key)):
            if item not in artifacts:
                artifacts.append(item)
    return artifacts


def load_task_requirement_evidence(state_dir: Path, task_id: str) -> dict[str, Any] | None:
    path = task_evidence_path(state_dir, task_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def project_requirement_evidence_errors(
    *,
    requirements_data: dict[str, Any] | None,
    tasks: list[dict[str, Any]],
    state_dir: Path,
) -> list[str]:
    if not isinstance(requirements_data, dict):
        return ["requirements.json is missing or invalid."]
    raw_requirements = requirements_data.get("requirements", [])
    if not isinstance(raw_requirements, list):
        return ["requirements.json requirements must be a list."]
    must_ids = {
        str(item.get("id", "")).strip()
        for item in raw_requirements
        if isinstance(item, dict) and str(item.get("priority", "")).strip().lower() == "must"
    }
    verified: set[str] = set()
    for task in tasks:
        task_id = str(task.get("id", "")).strip()
        if not task_id:
            continue
        evidence = load_task_requirement_evidence(state_dir, task_id)
        if not evidence:
            continue
        for item in evidence.get("requirements", []):
            if isinstance(item, dict) and item.get("status") == "verified":
                verified.add(str(item.get("id", "")).strip())
    missing = sorted(must_ids - verified)
    if missing:
        return ["Project evidence does not cover must requirements: " + ", ".join(missing) + "."]
    return []


def _verification_assets(task: dict[str, Any]) -> list[dict[str, Any]]:
    assets = task.get("verification_assets", [])
    return [item for item in assets if isinstance(item, dict)] if isinstance(assets, list) else []


def _verification_command_records(commands: object) -> list[dict[str, Any]]:
    if not isinstance(commands, list):
        return []
    records: list[dict[str, Any]] = []
    for item in commands:
        if isinstance(item, dict):
            records.append(item)
        else:
            records.append({"command": str(item)})
    return records


def _assertion_targets_for_requirement(assets: list[dict[str, Any]], requirement_id: str) -> list[str]:
    targets: list[str] = []
    for asset in assets:
        raw_targets = asset.get("assertion_targets", {})
        if not isinstance(raw_targets, dict):
            continue
        for target in raw_targets.get(requirement_id, []):
            text = str(target).strip()
            if text and text not in targets:
                targets.append(text)
    return targets


def _contract_evidence_entries_by_requirement(
    *,
    task: dict[str, Any],
    contract: dict[str, Any] | None,
    passed_commands: set[str],
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(contract, dict):
        return {}
    mapping = contract.get("criterion_command_map")
    if not isinstance(mapping, dict):
        return {}
    requirement_ids = _normalized_list(task.get("requirement_ids"))
    frozen_targets = _frozen_targets_by_requirement(task)
    test_files = _final_acceptance_artifacts(task)
    if not test_files:
        test_files = [
            artifact
            for artifact in _normalized_list(task.get("expected_artifacts"))
            if "/tests/" in artifact.replace("\\", "/") or Path(artifact.replace("\\", "/")).name.startswith("test_")
        ]
    entries_by_requirement: dict[str, list[dict[str, Any]]] = {requirement_id: [] for requirement_id in requirement_ids}
    for criterion, raw_commands in mapping.items():
        commands = _normalized_list(raw_commands)
        matched_requirement_ids = _requirement_ids_for_criterion(str(criterion), requirement_ids, frozen_targets)
        for command in commands:
            if command not in passed_commands:
                continue
            for requirement_id in matched_requirement_ids:
                targets = _assertion_targets_for_criterion(str(criterion), requirement_id, frozen_targets)
                entries_by_requirement.setdefault(requirement_id, []).append(
                    {
                        "type": "automated_test",
                        "command": command,
                        "test_files": test_files,
                        "result": "passed",
                        "assertion_targets": targets,
                    }
                )
    return entries_by_requirement


def _requirement_ids_for_criterion(
    criterion: str,
    requirement_ids: list[str],
    frozen_targets: dict[str, list[str]],
) -> list[str]:
    explicit = [requirement_id for requirement_id in requirement_ids if criterion.startswith(requirement_id + ":")]
    if explicit:
        return explicit
    by_target = [
        requirement_id
        for requirement_id, targets in frozen_targets.items()
        if criterion in targets or any(target and target in criterion for target in targets)
    ]
    if by_target:
        return by_target
    return list(requirement_ids) if len(requirement_ids) == 1 else []


def _assertion_targets_for_criterion(
    criterion: str,
    requirement_id: str,
    frozen_targets: dict[str, list[str]],
) -> list[str]:
    prefix = requirement_id + ":"
    if criterion.startswith(prefix):
        criterion = criterion[len(prefix):].strip()
    targets = frozen_targets.get(requirement_id, [])
    if criterion in targets:
        return [criterion]
    contained = [target for target in targets if target and target in criterion]
    return contained or targets


def _frozen_targets_by_requirement(task: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    snapshots = task.get("requirements", [])
    if not isinstance(snapshots, list):
        return result
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        requirement_id = str(snapshot.get("id", "")).strip()
        frozen = snapshot.get("frozen_acceptance", {})
        if not requirement_id or not isinstance(frozen, dict):
            continue
        result[requirement_id] = [
            str(target).strip()
            for target in frozen.get("assertion_targets", [])
            if str(target).strip()
        ]
    return result


def _requirement_needs_test_asset(task: dict[str, Any], requirement_id: str) -> bool:
    snapshots = task.get("requirements", [])
    if not isinstance(snapshots, list):
        return False
    for snapshot in snapshots:
        if not isinstance(snapshot, dict) or str(snapshot.get("id", "")).strip() != requirement_id:
            continue
        requirement_type = str(snapshot.get("type", "")).strip().lower()
        return requirement_type in {"gui_workflow", "workflow", "persistence", "report", "scheduling"}
    return False


def _task_is_gui_like(task: dict[str, Any]) -> bool:
    parts: list[str] = [
        str(task.get("title", "")),
        " ".join(str(item) for item in _normalized_list(task.get("acceptance_criteria"))),
    ]
    snapshots = task.get("requirements", [])
    if isinstance(snapshots, list):
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                continue
            parts.extend(
                str(snapshot.get(key, ""))
                for key in ("id", "text", "type", "acceptance_intent")
            )
    text = " ".join(parts).lower()
    return any(marker in text for marker in ("gui", "ui", "tkinter", "window", "panel", "tab", "button", "图形界面"))


def _normalized_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-_") or "current"
