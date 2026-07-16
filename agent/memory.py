from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MEMORY_TYPES = ("user", "feedback", "project", "reference")


@dataclass(frozen=True)
class MemoryDocument:
    name: str
    description: str
    type: str
    content: str

    @property
    def rendered(self) -> str:
        return render_memory(self)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.rendered.encode("utf-8")).hexdigest()


def parse_memory(text: str, fallback_name: str = "") -> MemoryDocument:
    metadata: dict[str, str] = {}
    body = text
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            for line in text[4:end].splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip() in {"name", "description", "type"}:
                    metadata[key.strip()] = _parse_scalar(value.strip())
            body = text[end + 4 :].lstrip("\r\n")

    name = metadata.get("name") or fallback_name
    description = metadata.get("description", "")
    memory_type = metadata.get("type", "")
    return MemoryDocument(name=name, description=description, type=memory_type, content=body.strip())


def render_memory(memory: MemoryDocument) -> str:
    return (
        "---\n"
        f"name: {json.dumps(memory.name, ensure_ascii=False)}\n"
        f"description: {json.dumps(memory.description, ensure_ascii=False)}\n"
        f"type: {memory.type}\n"
        "---\n\n"
        f"{memory.content.strip()}\n"
    )


def memory_catalog(memory_dir: Path) -> list[dict[str, str]]:
    catalog: list[dict[str, str]] = []
    if not memory_dir.exists():
        return catalog
    for path in sorted(memory_dir.glob("*.md")):
        memory = parse_memory(path.read_text(encoding="utf-8"), fallback_name=path.stem)
        if memory.name and memory.description and memory.type in MEMORY_TYPES:
            catalog.append(
                {
                    "name": memory.name,
                    "description": memory.description,
                    "type": memory.type,
                    "path": str(path.name),
                }
            )
    return catalog


def render_memory_index(memory_dir: Path) -> str:
    items = memory_catalog(memory_dir)
    lines = [
        "# Memory Index",
        "",
        "Typed memories live in `memories/` as one Markdown file per memory.",
        "Allowed types: user, feedback, project, reference.",
        "",
        "## Entries",
    ]
    if not items:
        lines.append("")
        lines.append("No typed memories available.")
        return "\n".join(lines) + "\n"
    for item in items:
        lines.append(f"- [{item['type']}] {item['name']}: {item['description']} (`memories/{item['path']}`)")
    return "\n".join(lines) + "\n"


def normalize_memory_content(args: dict[str, Any]) -> str:
    content = str(args.get("content") or args.get("body") or "").strip()
    memory_type = str(args.get("type", "")).strip()
    if memory_type == "feedback":
        why = str(args.get("why") or "").strip()
        how = str(args.get("how_to_apply") or args.get("how") or "").strip()
        if why or how:
            parts = [content]
            if why:
                parts.extend(["", f"**Why:** {why}"])
            if how:
                parts.extend(["", f"**How to apply:** {how}"])
            content = "\n".join(part for part in parts if part is not None).strip()
    return content


def validate_memory(memory: MemoryDocument) -> list[str]:
    errors: list[str] = []
    if not safe_memory_id(memory.name):
        errors.append("name must contain a letter, number, underscore, or dash")
    if not memory.description.strip():
        errors.append("description is required")
    if memory.type not in MEMORY_TYPES:
        errors.append("type must be one of: " + ", ".join(MEMORY_TYPES))
    if not memory.content.strip():
        errors.append("content is required")
    if memory.type == "feedback":
        lower = memory.content.lower()
        if "**why:**" not in lower and "why:" not in lower:
            errors.append("feedback memory must include Why")
        if "**how to apply:**" not in lower and "how to apply:" not in lower:
            errors.append("feedback memory must include How to apply")
    if memory.type == "project" and contains_relative_date(memory.description + "\n" + memory.content):
        errors.append("project memory must convert relative dates to absolute dates such as YYYY-MM-DD")
    return errors


def contains_relative_date(text: str) -> bool:
    patterns = [
        r"\b(today|tomorrow|yesterday|tonight|this\s+(week|month|quarter|year)|next\s+(week|month|quarter|year))\b",
        r"\b(before|by|until)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(next|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"(今天|明天|昨天|今晚|本周|这周|下周|本月|下月|今年|明年|周[一二三四五六日天]|星期[一二三四五六日天])",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def safe_memory_id(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in raw.strip().lower())
    return cleaned.strip("-_")


def _parse_scalar(value: str) -> str:
    if value.startswith(('"', "'")):
        try:
            parsed = json.loads(value)
            return str(parsed)
        except json.JSONDecodeError:
            return value.strip("\"'")
    return value
