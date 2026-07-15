from __future__ import annotations

from typing import Any

from agent.tools.base import WorkspaceTool


class ListFilesTool(WorkspaceTool):
    def run(self, action: dict[str, Any]):
        from agent.tools import ToolResult

        target = str(action.get("target", ".") or ".")
        args = action.get("args", {})
        recursive = bool(args.get("recursive", False))
        limit = max(1, min(int(args.get("limit", 200)), 1000))
        try:
            path = self.resolve_path(target)
            if not path.exists():
                return ToolResult(
                    False,
                    "List failed: path does not exist. If this task is to create the path and an acceptance contract exists, use write to create the first required file instead of listing again.",
                    {"target": target, "missing_path": True, "recommended_action": "write"},
                )
            if path.is_file():
                rel = str(path.relative_to(self.root))
                return ToolResult(True, f"Listed file '{target}'.", {"target": target, "entries": [entry_dict(rel, "file")]})
            children = path.rglob("*") if recursive else path.iterdir()
            entries = []
            for child in sorted(children, key=lambda item: str(item.relative_to(self.root)).lower()):
                if self._skip(child):
                    continue
                kind = "dir" if child.is_dir() else "file"
                entries.append(entry_dict(str(child.relative_to(self.root)), kind))
                if len(entries) >= limit:
                    break
        except Exception as exc:
            return ToolResult(False, f"List failed: {exc}", {"target": target})
        return ToolResult(
            True,
            f"Listed {len(entries)} item(s) from '{target}'.",
            {"target": target, "recursive": recursive, "limit": limit, "entries": entries},
        )

    def _skip(self, path) -> bool:
        parts = set(path.parts)
        return ".git" in parts or "__pycache__" in parts


def entry_dict(path: str, kind: str) -> dict[str, str]:
    return {"path": path.replace("\\", "/"), "type": kind}
