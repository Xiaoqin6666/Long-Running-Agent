from __future__ import annotations

from typing import Any

from agent.tools.base import WorkspaceTool


class ReadTool(WorkspaceTool):
    DEFAULT_RANGE_LINES = 200
    DEFAULT_MATCH_CONTEXT = 20
    DEFAULT_MATCH_LINES = 120
    MAX_LINES = 400

    def run(self, action: dict[str, Any]):
        from agent.tools import ToolResult

        target = str(action.get("target", ""))
        args = action.get("args", {})
        try:
            path = self.resolve_path(target)
            if path.is_dir():
                start = max(int(args.get("start", 1)), 1)
                end = max(int(args.get("end", start + self.DEFAULT_RANGE_LINES - 1)), start)
                entries = []
                for child in sorted(path.iterdir(), key=lambda item: item.name.lower()):
                    kind = "dir" if child.is_dir() else "file"
                    entries.append(f"{kind}\t{child.name}")
                selected = entries[start - 1 : end]
                return ToolResult(
                    True,
                    f"Listed {len(selected)} entries from '{target}'.",
                    {"target": target, "start": start, "end": end, "content": "\n".join(selected)},
                )
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            return ToolResult(False, f"Read failed: {exc}", {"target": target})

        query = self._query(args)
        if query:
            return self._read_match(target, lines, query, args)

        start = max(int(args.get("start", 1)), 1)
        end = max(int(args.get("end", start + self.DEFAULT_RANGE_LINES - 1)), start)
        end = min(end, start + self.MAX_LINES - 1)
        selected = lines[start - 1 : end]
        has_more = end < len(lines)
        data = {
            "target": target,
            "start": start,
            "end": end,
            "content": "\n".join(selected),
            "has_more": has_more,
        }
        if has_more:
            data["next_read"] = {"target": target, "args": {"start": end + 1, "end": end + self.DEFAULT_RANGE_LINES}}
        return ToolResult(
            True,
            self._summary("Read", len(selected), target, start, end, has_more),
            data,
        )

    def _read_match(self, target: str, lines: list[str], query: str, args: dict[str, Any]):
        from agent.tools import ToolResult

        context = max(int(args.get("context", self.DEFAULT_MATCH_CONTEXT)), 0)
        max_lines = max(1, min(int(args.get("max_lines", self.DEFAULT_MATCH_LINES)), self.MAX_LINES))
        continue_from = int(args.get("continue_from", 0) or 0)
        search_from = max(int(args.get("after", continue_from or 1)), 1)

        match_line = None
        if continue_from:
            start = min(max(continue_from, 1), len(lines) + 1)
        else:
            for idx, line in enumerate(lines[search_from - 1 :], search_from):
                if query in line:
                    match_line = idx
                    break
            if match_line is None:
                return ToolResult(
                    False,
                    f"Read found no match for {query!r} in {target}.",
                    {"target": target, "query": query, "matches": 0},
                )
            start = max(match_line - context, 1)

        end = min(start + max_lines - 1, len(lines))
        selected = lines[start - 1 : end]
        has_more = end < len(lines)
        data: dict[str, Any] = {
            "target": target,
            "query": query,
            "match_line": match_line,
            "start": start,
            "end": end,
            "content": "\n".join(selected),
            "has_more": has_more,
            "truncated": has_more,
        }
        if has_more:
            data["next_read"] = {
                "target": target,
                "args": {"query": query, "continue_from": end + 1, "max_lines": max_lines},
            }
        summary = self._summary(f"Read match for {query!r}", len(selected), target, start, end, has_more)
        return ToolResult(True, summary, data)

    def _query(self, args: dict[str, Any]) -> str:
        for key in ("query", "pattern", "grep", "match"):
            value = str(args.get(key, "")).strip()
            if value:
                return value
        return ""

    def _summary(self, prefix: str, count: int, target: str, start: int, end: int, has_more: bool) -> str:
        summary = f"{prefix} {count} line(s) from {target} lines {start}-{end}."
        if has_more:
            summary += (
                " More lines exist after this window. Continue with data.next_read.args only if the needed "
                "content is beyond these lines; otherwise use search/read args.query for a known id, symbol, "
                "filename, or error text."
            )
        return summary
