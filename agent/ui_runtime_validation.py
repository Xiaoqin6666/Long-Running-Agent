from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


RUNTIME_UI_RESULTS_FILE = "runtime_ui_results.json"


def evaluate_runtime_ui_contract(
    *,
    root: Path,
    state_dir: Path,
    tasks: list[dict[str, Any]],
    contract_data: dict[str, Any],
    benchmark_id: str = "",
) -> dict[str, Any]:
    """Evaluate runtime-relevant UI wiring from the real workspace UI sources.

    This probe intentionally starts with checks that do not need a live Tk mainloop:
    it finds Tkinter buttons in application UI files and fails when their command
    target resolves to a no-op handler such as ``pass`` or ``lambda self: None``.
    """

    workspace_path = _workspace_path(root=root, tasks=tasks, benchmark_id=benchmark_id)
    source_files = _workspace_ui_source_files(workspace_path) if workspace_path else []
    required_contracts = _required_ui_contracts(contract_data)
    button_issues = _find_noop_button_command_issues(root=root, source_files=source_files)
    source_targets = [_display_path(root, path) for path in source_files]
    issue_targets = _dedupe_strings(
        [
            target
            for issue in button_issues
            for target in issue.get("repair_targets", [])
            if str(target).strip()
        ]
    )
    checks = [
        {
            "id": "workspace_ui_sources",
            "passed": bool(source_files) or not required_contracts,
            "source_files": source_targets,
            "reason": (
                "workspace UI source files were found"
                if source_files
                else "no required UI contracts need runtime probing"
                if not required_contracts
                else "no workspace UI source files were found"
            ),
        },
        {
            "id": "button_command_wiring",
            "passed": not button_issues,
            "issues": button_issues,
            "repair_targets": issue_targets,
            "reason": (
                "all discovered button commands resolve to non-empty handlers"
                if not button_issues
                else "one or more discovered button commands resolve to no-op handlers"
            ),
        },
    ]
    per_requirement_results = [
        _runtime_result_for_contract(contract=contract, issues=button_issues, fallback_targets=issue_targets)
        for contract in required_contracts
    ]
    passed = all(check.get("passed") is True for check in checks) and all(
        item.get("passed") is True for item in per_requirement_results
    )
    return {
        "kind": "runtime_ui_results",
        "version": 1,
        "source": "system_validation",
        "framework": "tkinter_static_wiring",
        "workspace": _display_path(root, workspace_path) if workspace_path else "",
        "source_files": source_targets,
        "checks": checks,
        "results": per_requirement_results,
        "passed": passed,
    }


def write_runtime_ui_results(state_dir: Path, results: dict[str, Any]) -> None:
    path = state_dir / "system_validation" / RUNTIME_UI_RESULTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def runtime_ui_result_errors(results: dict[str, Any]) -> list[str]:
    if results.get("passed") is True:
        return []
    errors: list[str] = []
    for check in results.get("checks", []):
        if not isinstance(check, dict) or check.get("passed") is True:
            continue
        if check.get("id") == "button_command_wiring":
            issues = check.get("issues", [])
            if isinstance(issues, list):
                for issue in issues:
                    if not isinstance(issue, dict):
                        continue
                    target_text = ", ".join(str(item) for item in issue.get("repair_targets", []) if str(item).strip())
                    suffix = f" repair_targets: {target_text}" if target_text else ""
                    errors.append(
                        "Runtime UI check failed: "
                        f"button {issue.get('button_text', '<unknown>')} in {issue.get('file', '<unknown>')} "
                        f"uses no-op handler {issue.get('handler', '<unknown>')}."
                        + suffix
                    )
                continue
        errors.append(
            "Runtime UI check failed: "
            + str(check.get("reason", "runtime UI validation did not pass"))
        )
    if not errors:
        errors.append("Runtime UI validation did not pass.")
    return errors


def _required_ui_contracts(contract_data: dict[str, Any]) -> list[dict[str, Any]]:
    contracts = contract_data.get("contracts", []) if isinstance(contract_data, dict) else []
    if not isinstance(contracts, list):
        return []
    result: list[dict[str, Any]] = []
    for contract in contracts:
        if not isinstance(contract, dict):
            continue
        applicability = str(contract.get("ui_applicability", "")).strip().lower()
        requirement_type = str(contract.get("type", "")).strip().lower()
        surface = str(contract.get("ui_surface", "")).strip().lower()
        if applicability != "required":
            continue
        if requirement_type not in {"gui_workflow", "report"} and surface not in {"window", "dialog", "widget", "panel"}:
            continue
        result.append(contract)
    return result


def _runtime_result_for_contract(
    *,
    contract: dict[str, Any],
    issues: list[dict[str, Any]],
    fallback_targets: list[str],
) -> dict[str, Any]:
    requirement_id = str(contract.get("requirement_id", "")).strip()
    labels = _contract_action_labels(contract)
    matching_issues = [
        issue
        for issue in issues
        if _issue_matches_labels(issue, labels)
    ]
    if not labels and issues:
        matching_issues = issues
    repair_targets = _dedupe_strings(
        [
            target
            for issue in matching_issues
            for target in issue.get("repair_targets", [])
            if str(target).strip()
        ]
    )
    if not repair_targets and matching_issues:
        repair_targets = fallback_targets
    return {
        "requirement_id": requirement_id,
        "ui_surface": str(contract.get("ui_surface", "")).strip(),
        "action_labels": labels,
        "checks": {"button_command_wiring": not matching_issues},
        "issues": matching_issues,
        "repair_targets": repair_targets,
        "required_action": "inspect_and_repair_generated_code" if matching_issues else "",
        "passed": not matching_issues,
    }


def _contract_action_labels(contract: dict[str, Any]) -> list[str]:
    ui_contract = contract.get("ui_contract", {})
    if not isinstance(ui_contract, dict):
        return []
    labels: list[str] = []
    for key in ("buttons", "entry_points"):
        value = ui_contract.get(key)
        if isinstance(value, list):
            labels.extend(str(item).strip() for item in value if str(item).strip())
        elif str(value or "").strip():
            labels.append(str(value).strip())
    return _dedupe_strings(labels)


def _issue_matches_labels(issue: dict[str, Any], labels: list[str]) -> bool:
    if not labels:
        return False
    button_text = _normalize_label(str(issue.get("button_text", "")))
    handler = _normalize_label(str(issue.get("handler", "")))
    for label in labels:
        normalized = _normalize_label(label)
        if normalized and (normalized in button_text or button_text in normalized):
            return True
        if normalized and normalized in handler:
            return True
    return False


def _normalize_label(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _find_noop_button_command_issues(*, root: Path, source_files: list[Path]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for path in source_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        visitor = _ButtonCommandVisitor(root=root, path=path)
        visitor.visit(tree)
        issues.extend(visitor.issues)
    return issues


class _ButtonCommandVisitor(ast.NodeVisitor):
    def __init__(self, *, root: Path, path: Path) -> None:
        self.root = root
        self.path = path
        self.class_stack: list[str] = []
        self.noop_handlers_by_class = _noop_handlers_by_class(path)
        self.issues: list[dict[str, Any]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_Call(self, node: ast.Call) -> Any:
        if _is_button_constructor(node):
            issue = self._button_issue(node)
            if issue:
                self.issues.append(issue)
        self.generic_visit(node)

    def _button_issue(self, node: ast.Call) -> dict[str, Any] | None:
        command = _keyword(node, "command")
        if command is None:
            return None
        button_text = _constant_keyword_text(node, "text")
        current_class = self.class_stack[-1] if self.class_stack else ""
        handler = _self_handler_name(command)
        if _is_direct_noop_command(command):
            return self._issue(
                button_text=button_text,
                handler="<inline no-op>",
                line=node.lineno,
                reason="button command is None or an inline no-op lambda",
            )
        if not handler or not current_class:
            return None
        noop_handlers = self.noop_handlers_by_class.get(current_class, set())
        if handler not in noop_handlers:
            return None
        return self._issue(
            button_text=button_text,
            handler=handler,
            line=node.lineno,
            reason="button command resolves to a no-op handler on the real widget class",
        )

    def _issue(self, *, button_text: str, handler: str, line: int, reason: str) -> dict[str, Any]:
        target = _display_path(self.root, self.path)
        return {
            "file": target,
            "line": line,
            "button_text": button_text or "<unknown>",
            "handler": handler,
            "reason": reason,
            "repair_targets": [target],
        }


def _noop_handlers_by_class(path: Path) -> dict[str, set[str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except (OSError, SyntaxError):
        return {}
    result: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        handlers: set[str] = set()
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and _function_is_noop(item):
                handlers.add(item.name)
            elif isinstance(item, ast.Assign) and _value_is_noop_lambda(item.value):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        handlers.add(target.id)
            elif isinstance(item, ast.AnnAssign) and item.value is not None and _value_is_noop_lambda(item.value):
                if isinstance(item.target, ast.Name):
                    handlers.add(item.target.id)
        result[node.name] = handlers
    return result


def _function_is_noop(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = _strip_docstring(list(node.body))
    if not body:
        return True
    return all(_stmt_is_noop(stmt) for stmt in body)


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        return body[1:]
    return body


def _stmt_is_noop(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
        return True
    if isinstance(stmt, ast.Return):
        return stmt.value is None or (
            isinstance(stmt.value, ast.Constant) and stmt.value.value is None
        )
    return False


def _value_is_noop_lambda(value: ast.AST) -> bool:
    if not isinstance(value, ast.Lambda):
        return False
    return _expr_is_none_or_ellipsis(value.body)


def _is_direct_noop_command(value: ast.AST) -> bool:
    if _expr_is_none_or_ellipsis(value):
        return True
    return _value_is_noop_lambda(value)


def _expr_is_none_or_ellipsis(value: ast.AST) -> bool:
    return isinstance(value, ast.Constant) and value.value in {None, Ellipsis}


def _self_handler_name(value: ast.AST) -> str:
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id == "self":
        return value.attr
    if isinstance(value, ast.Lambda):
        body = value.body
        if isinstance(body, ast.Call) and isinstance(body.func, ast.Attribute):
            if isinstance(body.func.value, ast.Name) and body.func.value.id == "self":
                return body.func.attr
    return ""


def _is_button_constructor(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == "Button"
    if isinstance(func, ast.Name):
        return func.id == "Button"
    return False


def _keyword(node: ast.Call, name: str) -> ast.AST | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _constant_keyword_text(node: ast.Call, name: str) -> str:
    value = _keyword(node, name)
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return ""


def _workspace_ui_source_files(workspace_path: Path) -> list[Path]:
    files = [
        path
        for path in workspace_path.rglob("*.py")
        if not _is_test_path(workspace_path, path)
    ]
    ui_files: list[Path] = []
    for path in files:
        try:
            relative = path.relative_to(workspace_path).as_posix().lower()
        except ValueError:
            continue
        if relative == "main.py" or relative.endswith("/main.py") or "/ui/" in f"/{relative}":
            ui_files.append(path)
    return ui_files or files


def _is_test_path(workspace_path: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(workspace_path).as_posix().lower()
    except ValueError:
        return False
    return "/tests/" in f"/{relative}" or path.name.startswith("test_")


def _workspace_path(*, root: Path, tasks: list[dict[str, Any]], benchmark_id: str) -> Path | None:
    workspace = f"eval/benchmarks/{benchmark_id}/workspace" if benchmark_id else ""
    if not workspace:
        for task in tasks:
            workspace = _workspace_root_from_task(task)
            if workspace:
                break
    workspace_path = (root / workspace).resolve() if workspace else root.resolve()
    try:
        workspace_path.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return workspace_path if workspace_path.is_dir() else None


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


def _display_path(root: Path, path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
