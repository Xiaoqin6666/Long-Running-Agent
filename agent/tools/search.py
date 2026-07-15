from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.tools.base import WorkspaceTool


class SearchTool(WorkspaceTool):
    def run(self, action: dict[str, Any], *, excluded_path_fragments: tuple[str, ...] = ()):
        from agent.tools import ToolResult

        pattern = str(action.get("target", ""))
        args = action.get("args", {})
        scope = str(args.get("path", "."))
        if not pattern:
            return ToolResult(False, "Empty search pattern rejected.", {})
        try:
            root = self.resolve_path(scope)
        except Exception as exc:
            return ToolResult(False, f"Search path rejected: {exc}", {"path": scope})

        matches = []
        files = [root] if root.is_file() else root.rglob("*")
        for path in files:
            if not path.is_file() or self._skip(path, excluded_path_fragments):
                continue
            try:
                for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if pattern in line:
                        matches.append({"path": str(path.relative_to(self.root)), "line": idx, "text": line})
                        if len(matches) >= 50:
                            raise StopIteration
            except UnicodeDecodeError:
                continue
            except StopIteration:
                break
        return ToolResult(True, f"Found {len(matches)} match(es).", {"matches": matches})

    def _skip(self, path: Path, excluded_path_fragments: tuple[str, ...] = ()) -> bool:
        parts = set(path.parts)
        normalized = str(path.relative_to(self.root)).replace("\\", "/").lower()
        return (
            ".git" in parts
            or "__pycache__" in parts
            or any(fragment.lower() in normalized for fragment in excluded_path_fragments)
        )

