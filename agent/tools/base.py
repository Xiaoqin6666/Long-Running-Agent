from __future__ import annotations

from pathlib import Path


class WorkspaceTool:
    def __init__(self, root: Path) -> None:
        self.root = root

    def resolve_path(self, target: str) -> Path:
        path = (self.root / target).resolve()
        if self.root not in path.parents and path != self.root:
            raise ValueError(f"Path escapes workspace: {target}")
        return path

