from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SkillDocument:
    name: str
    description: str
    instruction: str
    examples: str = ""

    @property
    def content(self) -> str:
        return render_skill(self)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


def parse_skill(text: str, fallback_name: str = "") -> SkillDocument:
    metadata: dict[str, str] = {}
    body = text
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            for line in text[4:end].splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip() in {"name", "description"}:
                    metadata[key.strip()] = _parse_scalar(value.strip())
            body = text[end + 4 :].lstrip("\r\n")

    instruction, examples = _split_sections(body)
    name = metadata.get("name") or fallback_name
    description = metadata.get("description", "")
    if not description:
        description = _legacy_description(body)
    return SkillDocument(name=name, description=description, instruction=instruction, examples=examples)


def render_skill(skill: SkillDocument) -> str:
    text = (
        "---\n"
        f"name: {json.dumps(skill.name, ensure_ascii=False)}\n"
        f"description: {json.dumps(skill.description, ensure_ascii=False)}\n"
        "---\n\n"
        "# Instructions\n\n"
        f"{skill.instruction.strip()}\n"
    )
    if skill.examples.strip():
        text += f"\n# Examples\n\n{skill.examples.strip()}\n"
    return text


def skill_catalog(skill_dir: Path) -> list[dict[str, str]]:
    catalog: list[dict[str, str]] = []
    if not skill_dir.exists():
        return catalog
    for path in sorted(skill_dir.glob("*.md")):
        skill = parse_skill(path.read_text(encoding="utf-8"), fallback_name=path.stem)
        if skill.name and skill.description:
            catalog.append({"name": skill.name, "description": skill.description})
    return catalog


def normalize_instruction(value: Any) -> str:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return "\n".join(f"{index}. {item}" for index, item in enumerate(items, 1))
    return str(value or "").strip()


def normalize_examples(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        blocks = []
        for index, item in enumerate(value, 1):
            if isinstance(item, dict):
                lines = [f"## Example {index}"]
                for key, content in item.items():
                    lines.extend(["", f"**{str(key).replace('_', ' ').title()}**", "", str(content).strip()])
                blocks.append("\n".join(lines))
            else:
                blocks.append(f"## Example {index}\n\n{str(item).strip()}")
        return "\n\n".join(blocks)
    return str(value).strip()


def _split_sections(body: str) -> tuple[str, str]:
    match = re.search(r"(?im)^#\s+examples\s*$", body)
    before = body[: match.start()] if match else body
    examples = body[match.end() :].strip() if match else ""
    before = re.sub(r"(?im)^#\s+instructions\s*$", "", before, count=1).strip()
    return before, examples


def _legacy_description(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip().lstrip("- ")
        if stripped and not stripped.startswith("#") and not stripped.lower().startswith("evidence type:"):
            return stripped[:300]
    return "Legacy skill; load it to inspect its instructions."


def _parse_scalar(value: str) -> str:
    if value.startswith(('"', "'")):
        try:
            parsed = json.loads(value)
            return str(parsed)
        except json.JSONDecodeError:
            return value.strip("\"'")
    return value
