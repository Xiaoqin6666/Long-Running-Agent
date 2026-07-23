from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SkillDocument:
    name: str
    description: str
    content: str
    source_text: str | None = field(default=None, repr=False, compare=False)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(render_skill(self).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SkillEntry:
    path: Path
    document: SkillDocument

    @property
    def content_hash(self) -> str:
        if self.path.name != "SKILL.md":
            return self.document.content_hash
        digest = hashlib.sha256()
        for resource_path in sorted(
            (path for path in self.path.parent.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(self.path.parent).as_posix(),
        ):
            relative_path = resource_path.relative_to(self.path.parent).as_posix()
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(resource_path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()


def parse_skill(text: str) -> SkillDocument:
    metadata: dict[str, str] = {}
    body = text
    frontmatter = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)", text, flags=re.DOTALL)
    if frontmatter:
        metadata = _parse_frontmatter_metadata(frontmatter.group(1))
        body = text[frontmatter.end() :]

    name = metadata.get("name", "")
    description = " ".join(metadata.get("description", "").split())
    description = description or _content_description(body, name)
    return SkillDocument(name=name, description=description, content=body, source_text=text)


def render_skill(skill: SkillDocument) -> str:
    if skill.source_text is not None:
        return _with_final_newline(skill.source_text)
    frontmatter = [
        "---",
        f"name: {json.dumps(skill.name, ensure_ascii=False)}",
    ]
    if skill.description:
        frontmatter.append(f"description: {json.dumps(skill.description, ensure_ascii=False)}")
    frontmatter.extend(["---", ""])
    text = "\n".join(frontmatter)
    if skill.content:
        text += "\n" + skill.content
    return _with_final_newline(text)


def build_skill(name: str, content: str, description: str = "") -> SkillDocument:
    parsed = parse_skill(content)
    if parsed.name:
        return parsed
    return SkillDocument(
        name=name,
        description=description or _content_description(content, name),
        content=content,
    )


def discover_skill_entries(skill_dir: Path) -> list[SkillEntry]:
    entries: list[SkillEntry] = []
    if not skill_dir.exists():
        return entries
    paths = [*skill_dir.glob("*.md"), *skill_dir.glob("*/SKILL.md")]
    for path in sorted(set(paths), key=lambda item: item.as_posix()):
        skill = parse_skill(path.read_text(encoding="utf-8"))
        if skill.name:
            entries.append(SkillEntry(path=path, document=skill))
    return entries


def skill_catalog(skill_dirs: Path | Iterable[Path]) -> list[dict[str, str]]:
    catalog: list[dict[str, str]] = []
    seen: set[str] = set()
    directories = [skill_dirs] if isinstance(skill_dirs, Path) else list(skill_dirs)
    for skill_dir in directories:
        for entry in discover_skill_entries(skill_dir):
            skill = entry.document
            if skill.name in seen:
                continue
            seen.add(skill.name)
            catalog.append({"name": skill.name, "description": skill.description})
    return catalog


def normalize_skill_content(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def _content_description(body: str, fallback_name: str) -> str:
    for paragraph in re.split(r"\r?\n\s*\r?\n", body):
        text = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        text = re.sub(r"^#{1,6}\s+", "", text).strip()
        if text:
            return text[:300]
    label = fallback_name or "unnamed"
    return f"Skill {label}; load it to inspect its content."


def _with_final_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def _parse_scalar(value: str) -> str:
    if value.startswith(('"', "'")):
        try:
            parsed = json.loads(value)
            return str(parsed)
        except json.JSONDecodeError:
            return value.strip("\"'")
    return value


def _parse_frontmatter_metadata(frontmatter: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    lines = frontmatter.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        key, separator, value = line.partition(":")
        normalized_key = key.strip()
        if not separator or normalized_key not in {"name", "description"}:
            index += 1
            continue
        scalar = value.strip()
        if scalar in {"|", "|-", "|+", ">", ">-", ">+"}:
            block_lines: list[str] = []
            index += 1
            while index < len(lines):
                block_line = lines[index]
                if block_line and not block_line[0].isspace():
                    break
                block_lines.append(block_line.strip())
                index += 1
            joiner = "\n" if scalar.startswith("|") else " "
            metadata[normalized_key] = joiner.join(block_lines).strip()
            continue
        metadata[normalized_key] = _parse_scalar(scalar)
        index += 1
    return metadata
