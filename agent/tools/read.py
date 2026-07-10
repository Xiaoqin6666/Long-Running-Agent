from __future__ import annotations

from typing import Any

from agent.tools.base import WorkspaceTool


class ReadTool(WorkspaceTool):
    def run(self, action: dict[str, Any]):
        from agent.tools import ToolResult

        target = str(action.get("target", ""))
        args = action.get("args", {})
        start = max(int(args.get("start", 1)), 1)
        end = max(int(args.get("end", start + 200)), start)
        try:
            path = self.resolve_path(target)
            if path.is_dir():
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
        selected = lines[start - 1 : end]
        return ToolResult(
            True,
            f"Read {len(selected)} line(s) from {target}.",
            {"target": target, "start": start, "end": end, "content": "\n".join(selected)},
        )
