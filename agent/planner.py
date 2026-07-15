from __future__ import annotations

import ast
import re

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


DEFAULT_SESSION_BUDGET_TOKENS = 64000
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
    generated_tasks_artifact: str = "state/generated_tasks.json",
    init_artifact: str = "state/init.sh",
) -> TaskState:
    acceptance_criteria = [
        f"The project specification is materialized as {project_spec_artifact}.",
        f"A structured task graph is generated at {generated_tasks_artifact}.",
        "The task graph contains executable tasks with ids, dependencies, priorities, statuses, acceptance criteria, expected artifacts, and verification commands.",
        "The generated task graph passes deterministic schema, dependency, hidden-test, and project workspace-boundary validation.",
        f"A run-local POSIX shell init script is generated at {init_artifact} with repeatable setup or smoke-test commands.",
    ]
    verification_command = (
        "python -c \"import json, pathlib; "
        f"data=json.loads(pathlib.Path('{generated_tasks_artifact}').read_text(encoding='utf-8')); "
        "assert isinstance(data.get('tasks'), list) and data['tasks']; "
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
                generated_tasks_artifact,
                init_artifact,
            ],
            "implementation_artifacts": [
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
) -> list[str]:
    """Return deterministic validation errors for an Initializer-produced task graph."""
    if not isinstance(data, dict):
        return ["The generated task graph must be a JSON object."]
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return ["The generated task graph must contain a non-empty tasks list."]

    errors: list[str] = []
    task_ids: list[str] = []
    normalized_workspace = _normalize_artifact_path(expected_workspace_root) if expected_workspace_root else None
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
            declared_commands = {str(command) for command in commands}
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
            unmapped_commands = [str(command) for command in commands if str(command) not in mapped_commands]
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
                if "hidden_acceptance" in path.lower():
                    errors.append(f"{label}.{field} must not expose hidden acceptance artifacts: {path}.")
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
            for command in commands:
                command_text = str(command).replace("\\", "/")
                if "hidden_acceptance" in command_text.lower():
                    errors.append(f"{label}.verification_commands must not invoke hidden acceptance tests.")
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
    if "hidden_acceptance" in lower:
        return "must not invoke hidden acceptance"
    return None


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
