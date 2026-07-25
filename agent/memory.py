from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


MEMORY_TYPES = ("user", "feedback", "project", "reference")
SEMANTIC_DUPLICATE_THRESHOLD = 0.82


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


class MemoryStoreError(RuntimeError):
    def __init__(self, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.data = data or {}


class MemoryStore:
    """Canonical typed-Memory persistence shared by the Agent and user CLI."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.memory_dir = state_dir / "memories"
        self.index_path = state_dir / "memory.md"
        self.legacy_dir = state_dir / "legacy_memories"

    def ensure_layout(self) -> list[Path]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        archived: list[Path] = []
        for legacy_path in (
            self.state_dir / "hard_memory.md",
            self.state_dir / "soft_memory.md",
        ):
            if legacy_path.exists():
                archived.append(self._archive_legacy_file(legacy_path))
        if self.index_path.exists():
            current = self.index_path.read_text(encoding="utf-8")
            if not current.startswith("# Memory Index\n\nTyped memories live in `memories/`"):
                archived.append(self._archive_legacy_file(self.index_path))
        self.rebuild_index()
        return archived

    def save(self, memory: MemoryDocument, *, overwrite: bool = False) -> dict[str, str]:
        self.ensure_layout()
        errors = validate_memory(memory)
        if errors:
            raise MemoryStoreError(
                "Memory rejected: " + "; ".join(errors) + ".",
                {"errors": errors},
            )

        memory_id = safe_memory_id(memory.name)
        memory = MemoryDocument(memory_id, memory.description.strip(), memory.type, memory.content.strip())
        target = self.memory_dir / f"{memory_id}.md"
        catalog = memory_catalog(self.memory_dir)
        normalized_description = " ".join(memory.description.lower().split())
        duplicate = next(
            (
                item
                for item in catalog
                if (
                    safe_memory_id(item["name"]) == memory_id
                    and (not overwrite or target.name != item["path"])
                )
                or (
                    " ".join(item["description"].lower().split()) == normalized_description
                    and target.name != item["path"]
                )
            ),
            None,
        )
        if duplicate:
            raise MemoryStoreError(
                f"Memory rejected: duplicate of existing memory {duplicate['name']}.",
                {"duplicate": duplicate},
            )
        semantic_duplicate = find_semantic_duplicate(
            memory,
            self.memory_dir,
            exclude_name=memory_id if overwrite else "",
        )
        if semantic_duplicate:
            raise MemoryStoreError(
                f"Memory rejected: semantically similar to existing memory {semantic_duplicate['name']}.",
                {
                    "duplicate": semantic_duplicate,
                    "duplicate_reason": "semantic_similarity",
                },
            )
        if target.exists() and not overwrite:
            raise MemoryStoreError(
                f"Memory rejected: duplicate of existing memory {memory_id}.",
                {"duplicate": {"name": memory_id, "path": target.name}},
            )

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = target.with_suffix(".md.tmp")
        temporary_path.write_text(render_memory(memory), encoding="utf-8")
        rendered_errors = validate_memory(
            parse_memory(temporary_path.read_text(encoding="utf-8"), fallback_name=memory_id)
        )
        if rendered_errors:
            temporary_path.unlink(missing_ok=True)
            raise MemoryStoreError(
                "Memory rejected: rendered Memory failed validation.",
                {"errors": rendered_errors},
            )
        temporary_path.replace(target)
        self.rebuild_index()
        return {
            "name": memory_id,
            "description": memory.description,
            "type": memory.type,
            "path": target.as_posix(),
            "content_hash": memory.content_hash,
        }

    def rebuild_index(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = self.index_path.with_suffix(".md.tmp")
        temporary_path.write_text(render_memory_index(self.memory_dir), encoding="utf-8")
        temporary_path.replace(self.index_path)

    def _archive_legacy_file(self, source: Path) -> Path:
        self.legacy_dir.mkdir(parents=True, exist_ok=True)
        target = self.legacy_dir / source.name
        suffix = 1
        while target.exists():
            target = self.legacy_dir / f"{source.stem}-{suffix}{source.suffix}"
            suffix += 1
        source.replace(target)
        return target


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


def find_semantic_duplicate(
    memory: MemoryDocument,
    memory_dir: Path,
    *,
    exclude_name: str = "",
    threshold: float = SEMANTIC_DUPLICATE_THRESHOLD,
) -> dict[str, str] | None:
    """Return an existing memory that appears to capture the same durable fact."""
    if not memory_dir.exists():
        return None
    excluded = safe_memory_id(exclude_name)
    for path in sorted(memory_dir.glob("*.md")):
        existing = parse_memory(path.read_text(encoding="utf-8"), fallback_name=path.stem)
        if safe_memory_id(existing.name) == excluded:
            continue
        if validate_memory(existing):
            continue
        score = semantic_memory_similarity(memory, existing)
        if score >= threshold:
            return {
                "name": existing.name,
                "description": existing.description,
                "type": existing.type,
                "path": path.name,
                "similarity": f"{score:.3f}",
            }
    return None


def semantic_memory_similarity(left: MemoryDocument, right: MemoryDocument) -> float:
    left_text = _memory_similarity_text(left)
    right_text = _memory_similarity_text(right)
    left_tokens = _memory_tokens(left_text)
    right_tokens = _memory_tokens(right_text)
    token_score = _token_similarity(left_tokens, right_tokens)
    sequence_score = 0.0
    if len(set(left_tokens) & set(right_tokens)) >= 4:
        sequence_score = SequenceMatcher(None, _normalize_text(left_text), _normalize_text(right_text)).ratio()
    return max(token_score, sequence_score)


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


def _memory_similarity_text(memory: MemoryDocument) -> str:
    return f"{memory.description}\n{memory.content}"


def _normalize_text(text: str) -> str:
    return " ".join(_memory_tokens(text))


def _memory_tokens(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())
    tokens: list[str] = []
    for token in raw_tokens:
        canonical = _canonical_token(token)
        if canonical:
            tokens.append(canonical)
    return tokens


def _canonical_token(token: str) -> str:
    if token in _STOPWORDS:
        return ""
    token = _stem_token(token)
    return _SYNONYMS.get(token, token)


def _stem_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _token_similarity(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    left_counts = Counter(left)
    right_counts = Counter(right)
    shared_terms = set(left_counts) & set(right_counts)
    if len(shared_terms) < 4:
        return 0.0
    overlap = sum(min(left_counts[token], right_counts[token]) for token in shared_terms)
    containment = overlap / min(sum(left_counts.values()), sum(right_counts.values()))
    dot = sum(left_counts[token] * right_counts[token] for token in shared_terms)
    left_norm = sum(value * value for value in left_counts.values()) ** 0.5
    right_norm = sum(value * value for value in right_counts.values()) ** 0.5
    cosine = dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
    return max(containment, cosine)


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "this",
    "to",
    "use",
    "when",
    "with",
}


_SYNONYMS = {
    "actual": "real",
    "db": "database",
    "mocked": "mock",
    "mocking": "mock",
    "mocker": "mock",
    "preference": "prefer",
}


def _parse_scalar(value: str) -> str:
    if value.startswith(('"', "'")):
        try:
            parsed = json.loads(value)
            return str(parsed)
        except json.JSONDecodeError:
            return value.strip("\"'")
    return value
