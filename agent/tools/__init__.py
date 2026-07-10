from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.tools.bash import BashTool
from agent.tools.read import ReadTool
from agent.tools.search import SearchTool
from agent.tools.write import WriteTool


@dataclass
class ToolResult:
    ok: bool
    summary: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "summary": self.summary, "data": self.data}


__all__ = ["BashTool", "ReadTool", "SearchTool", "ToolResult", "WriteTool"]

