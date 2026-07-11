from __future__ import annotations

from typing import Any

from agent.tools.base import WorkspaceTool


class EditTool(WorkspaceTool):
    def run(self, action: dict[str, Any]):
        from agent.tools import ToolResult

        target = str(action.get("target", ""))
        args = action.get("args", {})
        old = args.get("old")
        new = args.get("new")
        if old is None or new is None:
            return ToolResult(False, "Edit requires args.old and args.new.", {"target": target})
        count = int(args.get("count", 1))
        if count < 1:
            return ToolResult(False, "Edit count must be positive.", {"target": target})
        try:
            path = self.resolve_path(target)
            original = path.read_text(encoding="utf-8")
            occurrences = original.count(str(old))
            if occurrences == 0:
                return ToolResult(False, "Edit failed: old text not found.", {"target": target, "occurrences": 0})
            if occurrences > count and not args.get("allow_multiple", False):
                return ToolResult(
                    False,
                    "Edit refused: old text appears multiple times.",
                    {"target": target, "occurrences": occurrences},
                )
            updated = original.replace(str(old), str(new), count)
            path.write_text(updated, encoding="utf-8")
        except Exception as exc:
            return ToolResult(False, f"Edit failed: {exc}", {"target": target})
        return ToolResult(
            True,
            f"Edited {target}.",
            {"target": target, "occurrences": occurrences, "replacements": min(occurrences, count)},
        )
