from __future__ import annotations

from typing import Any

from agent.tools.base import WorkspaceTool


class WriteTool(WorkspaceTool):
    def run(self, action: dict[str, Any]):
        from agent.tools import ToolResult

        target = str(action.get("target", ""))
        args = action.get("args", {})
        content = args.get("content")
        mode = args.get("mode", "create")
        if content is None:
            return ToolResult(False, "Write requires args.content.", {"target": target})
        try:
            path = self.resolve_path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            if mode == "append":
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(str(content))
            elif mode in {"create", "overwrite"}:
                if mode == "create" and path.exists():
                    return ToolResult(False, "Create refused because file already exists.", {"target": target})
                path.write_text(str(content), encoding="utf-8")
            else:
                return ToolResult(False, f"Unsupported write mode: {mode}", {"target": target})
        except Exception as exc:
            return ToolResult(False, f"Write failed: {exc}", {"target": target})
        return ToolResult(True, f"Wrote {target}.", {"target": target, "mode": mode})

