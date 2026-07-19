from __future__ import annotations

import ast
import re

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


DEFAULT_SESSION_BUDGET_TOKENS = 100000
DEFAULT_HANDOFF_THRESHOLD = 0.75

GENERATED_TASK_REQUIRED_FIELDS = {
    "id",
    "title",
    "priority",
    "depends_on",
    "status",
    "acceptance_criteria",
    "criterion_command_map",
    "expected_artifacts",
    "verification_commands",
}
GENERATED_TASK_ARTIFACT_FIELDS = {
    "expected_artifacts",
    "implementation_artifacts",
    "worker_test_artifacts",
    "acceptance_artifacts",
    "frozen_acceptance_artifacts",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskState:
    task_id: str
    user_goal: str
    acceptance_criteria: list[str]
    nodes: list[dict[str, Any]]
    iterations: int = 0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    last_action: dict[str, Any] = field(default_factory=dict)
    last_observation: dict[str, Any] = field(default_factory=dict)
    last_verified_at: str | None = None
    evidence_sources: list[dict[str, Any]] = field(default_factory=list)
    acceptance_contracts: list[dict[str, Any]] = field(default_factory=list)
    session_budget_tokens: int = DEFAULT_SESSION_BUDGET_TOKENS
    handoff_threshold: float = DEFAULT_HANDOFF_THRESHOLD
    session_used_tokens: int = 0
    handoff_ready: bool = False
    initializer_command_passed: bool = False
    initializer_repair: dict[str, Any] = field(default_factory=dict)
    orchestrator_decision: dict[str, Any] = field(default_factory=dict)
    pending_repair: dict[str, Any] = field(default_factory=dict)
    loaded_skills: list[dict[str, Any]] = field(default_factory=list)
    task_session_ids: dict[str, list[str]] = field(default_factory=dict)
    error_patterns: dict[str, dict[str, Any]] = field(default_factory=dict)
    task_error_fingerprints: dict[str, list[str]] = field(default_factory=dict)
    pending_skill_review: dict[str, Any] = field(default_factory=dict)
    skill_review_history: list[dict[str, Any]] = field(default_factory=list)
    conversation_messages: list[dict[str, str]] = field(default_factory=list)
    interaction_mode: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskState":
        data.setdefault("evidence_sources", [])
        data.setdefault("acceptance_contracts", [])
        data.setdefault("session_budget_tokens", DEFAULT_SESSION_BUDGET_TOKENS)
        data.setdefault("handoff_threshold", DEFAULT_HANDOFF_THRESHOLD)
        data.setdefault("session_used_tokens", 0)
        data.setdefault("handoff_ready", False)
        data.setdefault("initializer_command_passed", False)
        data.setdefault("initializer_repair", {})
        data.setdefault("orchestrator_decision", {})
        data.setdefault("pending_repair", {})
        data.setdefault("loaded_skills", [])
        data.setdefault("task_session_ids", {})
        data.setdefault("error_patterns", {})
        data.setdefault("task_error_fingerprints", {})
        data.setdefault("pending_skill_review", {})
        data.setdefault("skill_review_history", [])
        data.setdefault("conversation_messages", [])
        data.setdefault("interaction_mode", "")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "user_goal": self.user_goal,
            "acceptance_criteria": self.acceptance_criteria,
            "nodes": self.nodes,
            "iterations": self.iterations,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_action": self.last_action,
            "last_observation": self.last_observation,
            "last_verified_at": self.last_verified_at,
            "evidence_sources": self.evidence_sources,
            "acceptance_contracts": self.acceptance_contracts,
            "session_budget_tokens": self.session_budget_tokens,
            "handoff_threshold": self.handoff_threshold,
            "session_used_tokens": self.session_used_tokens,
            "handoff_ready": self.handoff_ready,
            "initializer_command_passed": self.initializer_command_passed,
            "initializer_repair": self.initializer_repair,
            "orchestrator_decision": self.orchestrator_decision,
            "pending_repair": self.pending_repair,
            "loaded_skills": self.loaded_skills,
            "task_session_ids": self.task_session_ids,
            "error_patterns": self.error_patterns,
            "task_error_fingerprints": self.task_error_fingerprints,
            "pending_skill_review": self.pending_skill_review,
            "skill_review_history": self.skill_review_history,
            "conversation_messages": self.conversation_messages,
            "interaction_mode": self.interaction_mode,
        }

    def summary(self) -> str:
        counts = {}
        for node in self.nodes:
            counts[node["status"]] = counts.get(node["status"], 0) + 1
        parts = [f"{key}={value}" for key, value in sorted(counts.items())]
        return f"iterations={self.iterations}; " + ", ".join(parts)


def create_initial_state(task: str) -> TaskState:
    lower_task = task.lower()
    is_answer_task = any(
        keyword in lower_task
        for keyword in ["inspect", "suggest", "explain", "recommend", "summarize", "分析", "建议", "解释", "总结"]
    )
    if is_answer_task:
        acceptance_criteria = [
            "The agent collects enough repository evidence to support the response.",
            "The final answer names a concrete next implementation step.",
            "State and trace files are written to disk.",
        ]
        nodes = [
            {"id": "T1", "title": "Inspect relevant repository context", "status": "pending", "evidence": []},
            {"id": "T2", "title": "Produce an evidence-based answer", "status": "pending", "evidence": []},
        ]
    else:
        acceptance_criteria = [
            "The agent loop can run from the command line.",
            "State and trace files are written to disk.",
            "A verifier decides whether finish is allowed.",
        ]
        nodes = [
            {"id": "T1", "title": "Initialize explicit task plan", "status": "pending", "evidence": []},
            {"id": "T2", "title": "Collect or produce one useful observation", "status": "pending", "evidence": []},
            {"id": "T3", "title": "Run independent verification", "status": "pending", "evidence": []},
        ]
    return TaskState(
        task_id="current",
        user_goal=task,
        acceptance_criteria=acceptance_criteria,
        nodes=nodes,
    )


def create_initializer_state(
    project_spec: str,
    project_spec_artifact: str = "state/project_spec.md",
    requirements_artifact: str = "state/requirements.json",
    generated_tasks_artifact: str = "state/generated_tasks.json",
    init_artifact: str = "state/init.sh",
) -> TaskState:
    acceptance_criteria = [
        f"The project specification is available at {project_spec_artifact}.",
        f"A Requirement Coverage Matrix is generated at {requirements_artifact}.",
        f"A structured task graph is generated at {generated_tasks_artifact}.",
        "The task graph contains a requirements coverage matrix and every must requirement is assigned to one or more tasks.",
        "The task graph contains executable tasks with ids, dependencies, priorities, statuses, acceptance criteria, expected artifacts, and verification commands.",
        "The generated task graph passes deterministic schema, dependency, hidden-test, and project workspace-boundary validation.",
        f"A run-local POSIX shell init script is generated at {init_artifact} with repeatable setup or smoke-test commands.",
    ]
    verification_command = (
        "python -c \"import json, pathlib; "
        f"requirements=json.loads(pathlib.Path('{requirements_artifact}').read_text(encoding='utf-8')); "
        f"tasks=json.loads(pathlib.Path('{generated_tasks_artifact}').read_text(encoding='utf-8')); "
        "assert isinstance(requirements.get('requirements'), list) and requirements['requirements']; "
        "assert isinstance(tasks.get('tasks'), list) and tasks['tasks']; "
        f"assert pathlib.Path('{project_spec_artifact}').is_file(); "
        f"script=pathlib.Path('{init_artifact}').read_text(encoding='utf-8'); "
        "assert script.startswith('#!/usr/bin/env sh\\n'); assert 'set -eu' in script.splitlines()\""
    )
    nodes = [
        {
            "id": "INIT",
            "title": "Initialize project plan from project specification",
            "status": "in_progress",
            "evidence": [],
            "depends_on": [],
            "priority": 0,
            "expected_artifacts": [
                project_spec_artifact,
                requirements_artifact,
                generated_tasks_artifact,
                init_artifact,
            ],
            "implementation_artifacts": [
                requirements_artifact,
                generated_tasks_artifact,
                init_artifact,
            ],
            "worker_test_artifacts": [],
            "acceptance_artifacts": [],
            "frozen_acceptance_artifacts": [],
            "test_policy": {
                "acceptance_tests_mutable_by_worker": False,
                "acceptance_test_repair_requires_verifier_approval": True,
            },
            "verification_commands": [
                verification_command
            ],
        }
    ]
    return TaskState(
        task_id="INIT",
        user_goal="INIT: Generate project plan from project_spec.md",
        acceptance_criteria=acceptance_criteria,
        nodes=nodes,
    )


def validate_generated_task_graph(
    data: object,
    expected_workspace_root: str | None = None,
    standard_library_only: bool = False,
    require_requirement_coverage: bool = False,
    requirements_data: object | None = None,
) -> list[str]:
    """Return deterministic validation errors for an Initializer-produced task graph."""
    if not isinstance(data, dict):
        return ["The generated task graph must be a JSON object."]
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return ["The generated task graph must contain a non-empty tasks list."]

    errors: list[str] = []
    requirements_source = requirements_data if requirements_data is not None else data.get("requirements")
    requirement_ids: set[str] = set()
    must_requirement_ids: set[str] = set()
    requirement_types: dict[str, str] = {}
    requirements_by_id: dict[str, dict[str, Any]] = {}
    if require_requirement_coverage or requirements_source is not None:
        (
            requirement_errors,
            requirement_ids,
            must_requirement_ids,
            requirement_types,
            requirements_by_id,
        ) = _requirement_matrix_errors(requirements_source)
        errors.extend(requirement_errors)
    task_ids: list[str] = []
    normalized_workspace = _normalize_artifact_path(expected_workspace_root) if expected_workspace_root else None
    covered_requirement_ids: set[str] = set()
    for index, task in enumerate(tasks):
        label = f"tasks[{index}]"
        if not isinstance(task, dict):
            errors.append(f"{label} must be an object.")
            continue
        missing = sorted(GENERATED_TASK_REQUIRED_FIELDS - set(task))
        if missing:
            errors.append(f"{label} is missing required fields: {', '.join(missing)}.")
        task_id = str(task.get("id", "")).strip()
        if not task_id:
            errors.append(f"{label}.id must be non-empty.")
        elif task_id in task_ids:
            errors.append(f"Duplicate task id: {task_id}.")
        else:
            task_ids.append(task_id)
        if not str(task.get("title", "")).strip():
            errors.append(f"{label}.title must be non-empty.")
        priority = task.get("priority")
        if not isinstance(priority, int) or isinstance(priority, bool):
            errors.append(f"{label}.priority must be an integer.")
        if task.get("status") != "pending":
            errors.append(f"{label}.status must initially be pending.")
        for field in ("depends_on", "acceptance_criteria", "expected_artifacts", "verification_commands"):
            if not isinstance(task.get(field), list):
                errors.append(f"{label}.{field} must be a list.")
        if isinstance(task.get("acceptance_criteria"), list) and not task["acceptance_criteria"]:
            errors.append(f"{label}.acceptance_criteria must not be empty.")
        if isinstance(task.get("verification_commands"), list) and not task["verification_commands"]:
            errors.append(f"{label}.verification_commands must not be empty.")
        task_requirement_ids = task.get("requirement_ids", [])
        normalized_task_requirement_ids: list[str] = []
        if require_requirement_coverage:
            if not isinstance(task_requirement_ids, list) or not task_requirement_ids:
                errors.append(f"{label}.requirement_ids must be a non-empty list.")
            else:
                normalized_task_requirement_ids = [str(item).strip() for item in task_requirement_ids]
                covered_requirement_ids.update(normalized_task_requirement_ids)
                unknown = [item for item in normalized_task_requirement_ids if item not in requirement_ids]
                if unknown:
                    errors.append(f"{label}.requirement_ids references unknown requirements: " + ", ".join(unknown) + ".")
                errors.extend(
                    _task_requirement_snapshot_errors(
                        task,
                        label,
                        normalized_task_requirement_ids,
                        requirements_by_id,
                    )
                )
                errors.extend(
                    _task_verification_coverage_errors(
                        task,
                        label,
                        normalized_task_requirement_ids,
                        requirements_by_id,
                    )
                )

        criteria = task.get("acceptance_criteria", [])
        criterion_command_map = task.get("criterion_command_map")
        commands = task.get("verification_commands", [])
        if not isinstance(criterion_command_map, dict):
            errors.append(f"{label}.criterion_command_map must be an object.")
        elif isinstance(criteria, list) and isinstance(commands, list):
            criterion_texts = [str(item) for item in criteria]
            missing_criteria = [criterion for criterion in criterion_texts if criterion not in criterion_command_map]
            extra_criteria = [str(criterion) for criterion in criterion_command_map if str(criterion) not in criterion_texts]
            if missing_criteria:
                errors.append(
                    f"{label}.criterion_command_map is missing acceptance criteria: "
                    + ", ".join(missing_criteria)
                    + "."
                )
            if extra_criteria:
                errors.append(
                    f"{label}.criterion_command_map contains unknown acceptance criteria: "
                    + ", ".join(extra_criteria)
                    + "."
                )
            declared_commands = set(_verification_command_texts(commands))
            mapped_commands: set[str] = set()
            for criterion in criterion_texts:
                mapped = criterion_command_map.get(criterion, [])
                if not isinstance(mapped, list) or not mapped:
                    errors.append(f"{label}.criterion_command_map['{criterion}'] must be a non-empty list.")
                    continue
                unknown_commands = [str(command) for command in mapped if str(command) not in declared_commands]
                if unknown_commands:
                    errors.append(
                        f"{label}.criterion_command_map['{criterion}'] references undeclared verification commands: "
                        + ", ".join(unknown_commands)
                        + "."
                    )
                mapped_commands.update(str(command) for command in mapped)
            unmapped_commands = [command for command in _verification_command_texts(commands) if command not in mapped_commands]
            if unmapped_commands:
                errors.append(
                    f"{label}.criterion_command_map does not assign verification commands: "
                    + ", ".join(unmapped_commands)
                    + "."
                )

        for field in GENERATED_TASK_ARTIFACT_FIELDS:
            artifacts = task.get(field, [])
            if not isinstance(artifacts, list):
                errors.append(f"{label}.{field} must be a list when present.")
                continue
            for artifact in artifacts:
                path = _normalize_artifact_path(artifact)
                if not path:
                    errors.append(f"{label}.{field} contains an empty artifact path.")
                    continue
                if normalized_workspace and not _is_under_path(path, normalized_workspace):
                    errors.append(
                        f"{label}.{field} artifact '{path}' must be under '{normalized_workspace}/'."
                    )

        implementation_artifacts = task.get("implementation_artifacts", [])
        worker_test_artifacts = task.get("worker_test_artifacts", [])
        acceptance_artifacts = task.get("acceptance_artifacts", [])
        expected_artifacts = task.get("expected_artifacts", [])
        normalized_expected_artifacts = {
            _normalize_artifact_path(artifact)
            for artifact in expected_artifacts
            if _normalize_artifact_path(artifact)
        } if isinstance(expected_artifacts, list) else set()
        if _looks_like_implementation_task(str(task.get("title", ""))) and not implementation_artifacts:
            errors.append(f"{label} is an implementation task but implementation_artifacts is empty.")
        for field, artifacts in (
            ("implementation_artifacts", implementation_artifacts),
            ("worker_test_artifacts", worker_test_artifacts),
            ("acceptance_artifacts", acceptance_artifacts),
        ):
            if isinstance(artifacts, list) and isinstance(expected_artifacts, list):
                missing_from_expected = [artifact for artifact in artifacts if artifact not in expected_artifacts]
                if missing_from_expected:
                    errors.append(
                        f"{label}.{field} must also be declared in expected_artifacts: "
                        + ", ".join(str(item) for item in missing_from_expected)
                        + "."
                    )

        commands = task.get("verification_commands", [])
        if isinstance(commands, list):
            for command in _verification_command_texts(commands):
                command_text = command.replace("\\", "/")
                if require_requirement_coverage:
                    weak_reason = _weak_verification_command_reason(
                        command_text,
                        task,
                        requirement_types=requirement_types,
                    )
                    if weak_reason:
                        errors.append(f"{label}.verification_commands is too weak: {weak_reason}.")
                if normalized_workspace and "workspace" in command_text.lower() and normalized_workspace not in command_text:
                    errors.append(
                        f"{label}.verification_commands references a workspace outside '{normalized_workspace}/'."
                    )
                if _is_placeholder_verification_command(command_text):
                    errors.append(f"{label}.verification_commands contains a placeholder/no-op command: {command_text}.")
                portability_error = verification_command_portability_error(command_text)
                if portability_error:
                    errors.append(f"{label}.verification_commands is not cross-platform: {portability_error}.")
                errors.extend(_python_c_syntax_errors(label, command_text))
                if standard_library_only and _uses_external_python_tool(command_text):
                    errors.append(
                        f"{label}.verification_commands violates the standard-library-only constraint: {command_text}."
                    )
                if normalized_workspace:
                    errors.extend(
                        _imported_module_artifact_errors(
                            label,
                            command_text,
                            normalized_workspace,
                            normalized_expected_artifacts,
                        )
                    )
                    errors.extend(
                        _workspace_import_path_errors(
                            label,
                            command_text,
                            normalized_workspace,
                            normalized_expected_artifacts,
                        )
                    )
        if standard_library_only:
            criteria_text = "\n".join(str(item) for item in task.get("acceptance_criteria", []))
            if _uses_external_python_tool(criteria_text):
                errors.append(f"{label}.acceptance_criteria violates the standard-library-only constraint.")

    if require_requirement_coverage:
        uncovered_must = sorted(must_requirement_ids - covered_requirement_ids)
        if uncovered_must:
            errors.append("Task graph does not cover must requirements: " + ", ".join(uncovered_must) + ".")

    known_ids = set(task_ids)
    dependencies: dict[str, list[str]] = {}
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("id", "")).strip()
        depends_on = task.get("depends_on", [])
        if not task_id or not isinstance(depends_on, list):
            continue
        normalized_dependencies = [str(item).strip() for item in depends_on]
        dependencies[task_id] = normalized_dependencies
        for dependency in normalized_dependencies:
            if dependency == task_id:
                errors.append(f"Task {task_id} cannot depend on itself.")
            elif dependency not in known_ids:
                errors.append(f"Task {task_id} depends on unknown task {dependency}.")
    if _has_dependency_cycle(dependencies):
        errors.append("The generated task graph contains a dependency cycle.")
    return errors


def validate_requirements_matrix(data: object) -> list[str]:
    """Return validation errors for a standalone requirements.json file."""
    errors, _, _, _, _ = _requirement_matrix_errors(data)
    return errors


def _requirement_matrix_errors(requirements: object) -> tuple[
    list[str],
    set[str],
    set[str],
    dict[str, str],
    dict[str, dict[str, Any]],
]:
    errors: list[str] = []
    requirement_ids: set[str] = set()
    must_requirement_ids: set[str] = set()
    requirement_types: dict[str, str] = {}
    requirements_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(requirements, dict):
        requirements = requirements.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        return (
            ["requirements must be a non-empty list."],
            requirement_ids,
            must_requirement_ids,
            requirement_types,
            requirements_by_id,
        )
    for index, requirement in enumerate(requirements):
        label = f"requirements[{index}]"
        if not isinstance(requirement, dict):
            errors.append(f"{label} must be an object.")
            continue
        missing = [
            field
            for field in ("id", "source", "text", "type", "priority")
            if not str(requirement.get(field, "")).strip()
        ]
        if missing:
            errors.append(f"{label} is missing required fields: " + ", ".join(missing) + ".")
        requirement_id = str(requirement.get("id", "")).strip()
        if not requirement_id:
            continue
        if requirement_id in requirement_ids:
            errors.append(f"Duplicate requirement id: {requirement_id}.")
            continue
        requirement_ids.add(requirement_id)
        priority = str(requirement.get("priority", "")).strip().lower()
        if priority not in {"must", "should", "could", "won't", "wont"}:
            errors.append(f"{label}.priority must be one of: must, should, could, won't.")
        if priority == "must":
            must_requirement_ids.add(requirement_id)
        requirement_types[requirement_id] = str(requirement.get("type", "")).strip().lower()
        frozen_acceptance = requirement.get("frozen_acceptance")
        errors.extend(_frozen_acceptance_errors(frozen_acceptance, f"{label}.frozen_acceptance", requirement))
        requirements_by_id[requirement_id] = {
            "id": requirement_id,
            "source": str(requirement.get("source", "")).strip(),
            "text": str(requirement.get("text", "")).strip(),
            "type": str(requirement.get("type", "")).strip(),
            "priority": str(requirement.get("priority", "")).strip(),
            "acceptance_intent": str(requirement.get("acceptance_intent", "")).strip(),
            "frozen_acceptance": frozen_acceptance if isinstance(frozen_acceptance, dict) else {},
        }
    if not must_requirement_ids:
        errors.append("requirements must include at least one priority='must' requirement.")
    return errors, requirement_ids, must_requirement_ids, requirement_types, requirements_by_id


def _task_requirement_snapshot_errors(
    task: dict[str, Any],
    label: str,
    requirement_ids: list[str],
    requirements_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    snapshots = task.get("requirements")
    if not isinstance(snapshots, list) or not snapshots:
        return [f"{label}.requirements must contain snapshots for every requirement_id."]
    snapshot_ids = [str(item.get("id", "")).strip() if isinstance(item, dict) else "" for item in snapshots]
    if set(snapshot_ids) != set(requirement_ids):
        errors.append(f"{label}.requirements ids must exactly match requirement_ids.")
    for snapshot_index, snapshot in enumerate(snapshots):
        snapshot_label = f"{label}.requirements[{snapshot_index}]"
        if not isinstance(snapshot, dict):
            errors.append(f"{snapshot_label} must be an object.")
            continue
        requirement_id = str(snapshot.get("id", "")).strip()
        source = requirements_by_id.get(requirement_id)
        if not source:
            continue
        for field in ("id", "source", "text", "type", "priority", "acceptance_intent"):
            expected = source.get(field, "")
            actual = str(snapshot.get(field, "")).strip()
            if expected or actual:
                if actual != expected:
                    errors.append(f"{snapshot_label}.{field} must match requirements.json for {requirement_id}.")
        if snapshot.get("frozen_acceptance") != source.get("frozen_acceptance"):
            errors.append(f"{snapshot_label}.frozen_acceptance must match requirements.json for {requirement_id}.")
    return errors


def _frozen_acceptance_errors(
    frozen_acceptance: object,
    label: str,
    requirement: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(frozen_acceptance, dict):
        return [f"{label} must be an object with intent and assertion_targets."]
    intent = str(frozen_acceptance.get("intent", "")).strip()
    if not intent:
        errors.append(f"{label}.intent must be non-empty.")
    targets = frozen_acceptance.get("assertion_targets")
    if not isinstance(targets, list) or not [str(item).strip() for item in targets if str(item).strip()]:
        errors.append(f"{label}.assertion_targets must be a non-empty list.")
    else:
        requirement_type = str(requirement.get("type", "")).strip().lower()
        target_count = len([item for item in targets if str(item).strip()])
        if _requirement_type_needs_test_asset(requirement_type) and target_count < 2:
            errors.append(f"{label}.assertion_targets must include at least two observable targets for {requirement_type}.")
    forbidden = frozen_acceptance.get("forbidden_weak_assertions", [])
    if forbidden is not None and not isinstance(forbidden, list):
        errors.append(f"{label}.forbidden_weak_assertions must be a list when present.")
    return errors


def _task_verification_coverage_errors(
    task: dict[str, Any],
    label: str,
    requirement_ids: list[str],
    requirements_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    requirement_id_set = set(requirement_ids)
    assets = task.get("verification_assets")
    commands = task.get("verification_commands", [])
    if not isinstance(assets, list) or not assets:
        return [f"{label}.verification_assets must be a non-empty list covering every requirement_id."]

    asset_ids: set[str] = set()
    asset_covered: set[str] = set()
    asset_coverage_by_id: dict[str, set[str]] = {}
    for asset_index, asset in enumerate(assets):
        asset_label = f"{label}.verification_assets[{asset_index}]"
        if not isinstance(asset, dict):
            errors.append(f"{asset_label} must be an object.")
            continue
        asset_id = str(asset.get("id", "")).strip()
        if not asset_id:
            errors.append(f"{asset_label}.id must be non-empty.")
        elif asset_id in asset_ids:
            errors.append(f"Duplicate verification asset id: {asset_id}.")
        else:
            asset_ids.add(asset_id)
        path = _normalize_artifact_path(asset.get("path", ""))
        if not path:
            errors.append(f"{asset_label}.path must be non-empty.")
        runner = str(asset.get("runner", "")).strip().lower()
        if runner not in {"pytest", "unittest", "python"}:
            errors.append(f"{asset_label}.runner must be pytest, unittest, or python.")
        repair_policy = str(asset.get("repair_policy", "")).strip().lower()
        if repair_policy and repair_policy != "infra_only":
            errors.append(f"{asset_label}.repair_policy must be infra_only.")
        covers = _normalized_id_list(asset.get("covers"))
        if not covers:
            errors.append(f"{asset_label}.covers must be a non-empty list.")
        unknown = sorted(set(covers) - requirement_id_set)
        if unknown:
            errors.append(f"{asset_label}.covers references requirements outside this task: " + ", ".join(unknown) + ".")
        asset_covered.update(covers)
        if asset_id:
            asset_coverage_by_id[asset_id] = set(covers)
        errors.extend(_asset_assertion_target_errors(asset, asset_label, covers, requirements_by_id))

    missing_asset_coverage = sorted(requirement_id_set - asset_covered)
    if missing_asset_coverage:
        errors.append(
            f"{label}.verification_assets do not cover task requirements: "
            + ", ".join(missing_asset_coverage)
            + "."
        )

    command_records = _verification_command_records(commands)
    command_covered: set[str] = set()
    for command_index, command in enumerate(command_records):
        command_label = f"{label}.verification_commands[{command_index}]"
        command_text = str(command.get("command", "")).strip()
        if not command_text:
            errors.append(f"{command_label}.command must be non-empty.")
        covers = _normalized_id_list(command.get("covers"))
        if not covers:
            errors.append(f"{command_label}.covers must be a non-empty list.")
        unknown = sorted(set(covers) - requirement_id_set)
        if unknown:
            errors.append(f"{command_label}.covers references requirements outside this task: " + ", ".join(unknown) + ".")
        asset_refs = _normalized_id_list(command.get("asset_ids"))
        if not asset_refs:
            errors.append(f"{command_label}.asset_ids must reference one or more verification_assets.")
        missing_assets = sorted(set(asset_refs) - asset_ids)
        if missing_assets:
            errors.append(f"{command_label}.asset_ids references unknown assets: " + ", ".join(missing_assets) + ".")
        linked_coverage: set[str] = set()
        for asset_id in asset_refs:
            linked_coverage.update(asset_coverage_by_id.get(asset_id, set()))
        not_backed_by_assets = sorted(set(covers) - linked_coverage)
        if not_backed_by_assets:
            errors.append(
                f"{command_label}.covers includes requirements not covered by linked assets: "
                + ", ".join(not_backed_by_assets)
                + "."
            )
        command_covered.update(covers)
        if any(_requirement_type_needs_test_asset(str(requirements_by_id.get(req_id, {}).get("type", "")).lower()) for req_id in covers):
            if not _is_test_runner_command(command_text):
                errors.append(f"{command_label}.command must run a test file for GUI/workflow/persistence/report requirements.")

    missing_command_coverage = sorted(requirement_id_set - command_covered)
    if missing_command_coverage:
        errors.append(
            f"{label}.verification_commands do not cover task requirements: "
            + ", ".join(missing_command_coverage)
            + "."
        )
    return errors


def _asset_assertion_target_errors(
    asset: dict[str, Any],
    label: str,
    covers: list[str],
    requirements_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    raw_targets = asset.get("assertion_targets")
    if not isinstance(raw_targets, dict):
        return [f"{label}.assertion_targets must be an object keyed by requirement id."]
    for requirement_id in covers:
        targets = raw_targets.get(requirement_id)
        if not isinstance(targets, list) or not [str(item).strip() for item in targets if str(item).strip()]:
            errors.append(f"{label}.assertion_targets['{requirement_id}'] must be a non-empty list.")
            continue
        frozen = requirements_by_id.get(requirement_id, {}).get("frozen_acceptance", {})
        frozen_targets = []
        if isinstance(frozen, dict):
            frozen_targets = [str(item).strip() for item in frozen.get("assertion_targets", []) if str(item).strip()]
        missing = [item for item in frozen_targets if item not in [str(target).strip() for target in targets]]
        if missing:
            errors.append(
                f"{label}.assertion_targets['{requirement_id}'] must cover frozen targets: "
                + ", ".join(missing)
                + "."
            )
    return errors


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


def _verification_command_texts(commands: object) -> list[str]:
    return [str(item.get("command", "")).strip() for item in _verification_command_records(commands) if str(item.get("command", "")).strip()]


def _normalized_id_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _requirement_type_needs_test_asset(requirement_type: str) -> bool:
    return requirement_type in {"gui_workflow", "workflow", "persistence", "report", "scheduling"}


def _is_test_runner_command(command: str) -> bool:
    lower = command.lower()
    return "python -m pytest" in lower or "pytest" in lower or "python -m unittest" in lower or "unittest discover" in lower


def _weak_verification_command_reason(
    command_text: str,
    task: dict[str, Any],
    *,
    requirement_types: dict[str, str],
) -> str:
    lower = command_text.lower()
    criteria = " ".join(str(item).lower() for item in task.get("acceptance_criteria", []))
    title = str(task.get("title", "")).lower()
    requirement_ids = task.get("requirement_ids", [])
    task_requirement_types = {
        requirement_types.get(str(item).strip(), "")
        for item in requirement_ids
        if isinstance(requirement_ids, list)
    }
    task_text = f"{title} {criteria} {' '.join(task_requirement_types)}"
    gui_like = any(marker in task_text for marker in ("gui", "ui", "button", "workflow", "panel", "window", "menu"))
    if gui_like and ("lambda: none" in lower or "command=lambda" in lower):
        return "GUI verification must reject placeholder button handlers"
    weak_patterns = (
        "assert app is not none",
        "assert ep is not none",
        "assert pp is not none",
        "assert ap is not none",
        "assert callable(",
        "assert isinstance(result, dict)",
        "assert isinstance(report, dict)",
        "assert isinstance(conflicts, list)",
    )
    if any(pattern in lower for pattern in weak_patterns):
        return "command only checks imports, instantiation, callability, or container type"
    if "python -m unittest" in lower or "unittest discover" in lower or "pytest" in lower:
        return ""
    if gui_like and not any(marker in lower for marker in ("invoke(", ".invoke", "command", "event_generate", "assert len(", "assert getattr(")):
        return "GUI workflow requirements need a command that exercises a handler or observable state change"
    if "assert " not in lower:
        return "command has no assertion"
    return ""


def validate_initializer_script(
    content: object,
    expected_workspace_root: str | None = None,
    standard_library_only: bool = False,
) -> list[str]:
    """Validate the run-local POSIX shell bootstrap without executing it."""
    text = str(content or "")
    lines = [line.rstrip() for line in text.splitlines()]
    nonempty = [line.strip() for line in lines if line.strip()]
    if not nonempty:
        return ["init.sh must not be empty."]

    errors: list[str] = []
    if nonempty[0] != "#!/usr/bin/env sh":
        errors.append("init.sh must start with the POSIX shell shebang '#!/usr/bin/env sh'.")
    if "set -eu" not in nonempty:
        errors.append("init.sh must enable deterministic failure handling with 'set -eu'.")
    source_markers = ("import ", "from ", "def ", "class ")
    if any(line.startswith(source_markers) for line in nonempty):
        errors.append("init.sh contains Python source code; it must be a shell script that may invoke Python commands.")
    if any("os.makedirs(" in line or "pathlib.Path(" in line and not line.startswith("python ") for line in nonempty):
        errors.append("init.sh contains embedded Python statements outside a Python command.")

    normalized_workspace = _normalize_artifact_path(expected_workspace_root) if expected_workspace_root else None
    normalized_text = text.replace("\\", "/")
    if re.search(r"state/benchmarks/[^\s'\"]+/workspace", normalized_text, re.IGNORECASE):
        errors.append("init.sh must not create or reference an application workspace under state/benchmarks/.")
    if normalized_workspace:
        workspace_references: list[str] = []
        for match in re.findall(r"['\"]([^'\"]*workspace[^'\"]*)['\"]", normalized_text, re.IGNORECASE):
            reference = _normalize_artifact_path(match)
            if reference and not any(char.isspace() for char in reference):
                workspace_references.append(reference)
        for match in re.findall(
            r"\b(?:workspace|workspace_root)\s*=\s*['\"]?([^'\"\s]+)",
            normalized_text,
            re.IGNORECASE,
        ):
            reference = _normalize_artifact_path(match)
            if "(" not in reference:
                workspace_references.append(reference)
        for reference in dict.fromkeys(workspace_references):
            if not _is_under_path(reference, normalized_workspace):
                errors.append(
                    f"init.sh workspace references must use '{normalized_workspace}/': {reference}."
                )
    if standard_library_only and _uses_external_python_tool(normalized_text):
        errors.append("init.sh violates the standard-library-only constraint.")
    if not any(re.search(r"(^|\s)python(?:\.exe)?(\s|$)", line, re.IGNORECASE) for line in nonempty):
        errors.append("init.sh must contain at least one Python environment or smoke-check command.")
    if any(marker in normalized_text.lower() for marker in ("not implemented", "placeholder", "todo:")):
        errors.append("init.sh must not contain placeholder or no-op commands.")
    return list(dict.fromkeys(errors))


def _normalize_artifact_path(value: object) -> str:
    text = str(value or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/")


def _is_under_path(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent + "/")


def _has_dependency_cycle(dependencies: dict[str, list[str]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> bool:
        if task_id in visiting:
            return True
        if task_id in visited:
            return False
        visiting.add(task_id)
        for dependency in dependencies.get(task_id, []):
            if dependency in dependencies and visit(dependency):
                return True
        visiting.remove(task_id)
        visited.add(task_id)
        return False

    return any(visit(task_id) for task_id in dependencies)


def _looks_like_implementation_task(title: str) -> bool:
    normalized = title.strip().lower()
    return normalized.startswith(("implement ", "create ", "build ", "add ", "fix ", "update ", "refactor "))


def _uses_external_python_tool(text: str) -> bool:
    return bool(re.search(r"\b(pytest|pip(?:3)?\s+install|poetry|pipenv|tox|nose2)\b", text, re.IGNORECASE))


def _is_placeholder_verification_command(command: str) -> bool:
    normalized = command.strip().lower()
    if any(marker in normalized for marker in ("not implemented", "placeholder", "todo:")):
        return True
    if re.match(r"^(echo|write-output)\b", normalized):
        has_followup_command = bool(re.search(r"(?:&&|\|\||;|\|)\s*\S+", normalized))
        return not has_followup_command
    return bool(re.fullmatch(r"python(?:\.exe)?\s+-c\s+[\"']assert\s+true;?[\"']", normalized))


def verification_command_portability_error(command: str) -> str | None:
    """Return why a public verification command is unsuitable for frozen cross-platform use."""
    normalized = command.strip()
    lower = normalized.lower().replace("\\", "/")
    if not re.match(r"^(python(?:\.exe|3)?|py)\s+(?:-c|-m)\b", lower):
        return "use a direct Python command instead of shell-specific setup or pipelines"
    nested_cwd_error = _nested_python_cwd_error(normalized)
    if nested_cwd_error:
        return nested_cwd_error
    unix_markers = (
        "/tmp/",
        "echo -e ",
        "mkdir -p ",
        "printf ",
        " grep ",
        " rm ",
    )
    marker = next((item for item in unix_markers if item in f" {lower} "), None)
    if marker:
        return f"contains Unix-specific construct {marker.strip()!r}"
    return None


def _nested_python_cwd_error(command: str) -> str | None:
    for code in _python_c_snippets(command):
        chdir_paths = _literal_call_paths(code, "os", "chdir")
        cwd_paths = _literal_keyword_paths(code, "cwd")
        for chdir_path in chdir_paths:
            for cwd_path in cwd_paths:
                if _paths_conflict_after_chdir(chdir_path, cwd_path):
                    return (
                        "mixes os.chdir() with a relative subprocess cwd for the same workspace; "
                        "use verification_procedure.working_directory or one cwd mechanism"
                    )
    return None


def _literal_call_paths(code: str, module: str, function: str) -> list[str]:
    paths: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return paths
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == function
            and isinstance(func.value, ast.Name)
            and func.value.id == module
        ):
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            paths.append(node.args[0].value)
    return paths


def _literal_keyword_paths(code: str, keyword_name: str) -> list[str]:
    paths: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return paths
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != keyword_name:
                continue
            value = keyword.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                paths.append(value.value)
    return paths


def _paths_conflict_after_chdir(chdir_path: str, cwd_path: str) -> bool:
    chdir_normalized = _relative_path_for_cwd_check(chdir_path)
    cwd_normalized = _relative_path_for_cwd_check(cwd_path)
    if not chdir_normalized or not cwd_normalized:
        return False
    if chdir_normalized == cwd_normalized:
        return True
    return cwd_normalized.startswith(f"{chdir_normalized}/")


def _relative_path_for_cwd_check(path: str) -> str:
    normalized = path.strip().replace("\\", "/").strip("/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized or re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith("/"):
        return ""
    return normalized


def _python_c_syntax_errors(label: str, command: str) -> list[str]:
    errors: list[str] = []
    for code in _python_c_snippets(command):
        try:
            ast.parse(code)
        except SyntaxError as exc:
            message = exc.msg or "invalid syntax"
            errors.append(f"{label}.verification_commands contains invalid python -c syntax: {message}.")
    return errors


def _python_c_snippets(command: str) -> list[str]:
    snippets: list[str] = []
    pattern = r"\bpython(?:\.exe)?\s+-c\s+(?P<quote>['\"])(?P<code>.*?)(?P=quote)"
    for match in re.finditer(pattern, command, re.IGNORECASE):
        snippets.append(match.group("code"))
    return snippets


def _imported_module_artifact_errors(
    label: str,
    command: str,
    workspace_root: str,
    expected_artifacts: set[str],
) -> list[str]:
    errors: list[str] = []
    for module in _imported_project_modules(command):
        module_artifact = f"{workspace_root}/{module.replace('.', '/')}.py"
        if module_artifact not in expected_artifacts:
            errors.append(
                f"{label}.verification_commands imports '{module}', but expected_artifacts does not include '{module_artifact}'."
            )
    return errors


def _workspace_import_path_errors(
    label: str,
    command: str,
    workspace_root: str,
    expected_artifacts: set[str],
) -> list[str]:
    project_modules = [
        module
        for module in _imported_project_modules(command)
        if f"{workspace_root}/{module.replace('.', '/')}.py" in expected_artifacts
    ]
    if not project_modules or _command_configures_workspace_imports(command, workspace_root):
        return []
    return [
        f"{label}.verification_commands invokes workspace module(s) {', '.join(project_modules)} "
        f"without configuring '{workspace_root}' via sys.path, PYTHONPATH, or cwd; commands run from the repository root."
    ]


def _command_configures_workspace_imports(command: str, workspace_root: str) -> bool:
    normalized = command.replace("\\", "/")
    escaped = re.escape(workspace_root)
    patterns = (
        rf"sys\.path\.insert\([^\r\n]*{escaped}",
        rf"PYTHONPATH[^\r\n]*{escaped}",
        rf"cwd\s*=\s*['\"][^'\"]*{escaped}",
        rf"\bcd(?:\s+/d)?\s+['\"]?{escaped}(?:\s|['\"]|$)",
    )
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns)


def _imported_project_modules(command: str) -> list[str]:
    ignored_roots = {
        "argparse",
        "collections",
        "contextlib",
        "csv",
        "dataclasses",
        "datetime",
        "functools",
        "io",
        "itertools",
        "json",
        "math",
        "os",
        "pathlib",
        "re",
        "shutil",
        "subprocess",
        "sys",
        "tempfile",
        "typing",
        "unittest",
    }
    modules: list[str] = []
    for pattern in (
        r"\bfrom\s+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\s+import\b",
        r"\bpython\s+-m\s+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\b",
        r"['\"]-m['\"]\s*,\s*['\"]([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)['\"]",
    ):
        for match in re.finditer(pattern, command):
            module = match.group(1)
            root = module.split(".", 1)[0]
            if root in ignored_roots:
                continue
            if module not in modules:
                modules.append(module)
    return modules
