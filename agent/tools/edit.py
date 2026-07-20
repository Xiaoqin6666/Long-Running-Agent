from __future__ import annotations

from typing import Any

from agent.tools.base import WorkspaceTool


class EditTool(WorkspaceTool):
    def run(self, action: dict[str, Any]):
        from agent.tools import ToolResult

        target = str(action.get("target", ""))
        args = action.get("args", {})
        if "start" in args or "end" in args:
            return self._edit_line_range(target, args)
        old = args.get("old")
        new = args.get("new")
        if old is None or new is None:
            return ToolResult(
                False,
                "Edit requires args.old and args.new, or args.start/args.end with args.content.",
                {"target": target},
            )
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

    def _edit_line_range(self, target: str, args: dict[str, Any]):
        from agent.tools import ToolResult

        if "content" not in args:
            return ToolResult(False, "Line edit requires args.content.", {"target": target})
        try:
            start = int(args.get("start", 0))
            end = int(args.get("end", start))
        except (TypeError, ValueError):
            return ToolResult(False, "Line edit start/end must be integers.", {"target": target})
        if start < 1 or end < start:
            return ToolResult(False, "Line edit requires 1-based start <= end.", {"target": target})
        try:
            path = self.resolve_path(target)
            original = path.read_text(encoding="utf-8")
            trailing_newline = original.endswith("\n")
            lines = original.splitlines()
            if end > len(lines):
                return ToolResult(
                    False,
                    "Line edit range is outside the file.",
                    {"target": target, "line_count": len(lines), "start": start, "end": end},
                )
            replacement = str(args.get("content", "")).splitlines()
            updated_lines = lines[: start - 1] + replacement + lines[end:]
            updated = "\n".join(updated_lines)
            if trailing_newline or args.get("trailing_newline", True):
                updated += "\n"
            path.write_text(updated, encoding="utf-8")
        except Exception as exc:
            return ToolResult(False, f"Line edit failed: {exc}", {"target": target})
        return ToolResult(
            True,
            f"Edited {target} lines {start}-{end}.",
            {
                "target": target,
                "start": start,
                "end": end,
                "replacement_lines": len(replacement),
            },
        )
