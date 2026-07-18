from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any
from uuid import uuid4


DEFAULT_INLINE_OUTPUT_LIMIT = 8000
DEFAULT_OUTPUT_HEAD_CHARS = 2000
DEFAULT_OUTPUT_TAIL_CHARS = 6000


def capture_command_output(
    *,
    root: Path,
    output_dir: Path,
    label: str,
    stdout: str,
    stderr: str,
    inline_limit: int = DEFAULT_INLINE_OUTPUT_LIMIT,
    head_chars: int = DEFAULT_OUTPUT_HEAD_CHARS,
    tail_chars: int = DEFAULT_OUTPUT_TAIL_CHARS,
) -> dict[str, Any]:
    output = (stdout + stderr).strip()
    stdout = stdout.strip()
    stderr = stderr.strip()
    truncated = len(output) > inline_limit
    data: dict[str, Any] = {
        "stdout": stdout if not truncated else _preview(stdout, inline_limit, head_chars, tail_chars),
        "stderr": stderr if not truncated else _preview(stderr, inline_limit, head_chars, tail_chars),
        "output": output if not truncated else _preview(output, inline_limit, head_chars, tail_chars),
        "stdout_chars": len(stdout),
        "stderr_chars": len(stderr),
        "output_chars": len(output),
        "output_truncated": truncated,
    }
    if not truncated:
        return data

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(label)
    data.update(
        {
            "stdout_path": _write_output_file(root, output_dir, f"{stem}.stdout.txt", stdout),
            "stderr_path": _write_output_file(root, output_dir, f"{stem}.stderr.txt", stderr),
            "output_path": _write_output_file(root, output_dir, f"{stem}.combined.txt", output),
        }
    )
    return data


def _preview(text: str, inline_limit: int, head_chars: int, tail_chars: int) -> str:
    if len(text) <= inline_limit:
        return text
    if not text:
        return ""
    head = text[:head_chars].rstrip()
    tail = text[-tail_chars:].lstrip()
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n...[{omitted} characters omitted; full output saved to output_path]...\n{tail}"


def _write_output_file(root: Path, output_dir: Path, filename: str, content: str) -> str:
    path = output_dir / filename
    path.write_text(content, encoding="utf-8")
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_stem(label: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", label.strip())[:48].strip("-_.")
    if not cleaned:
        cleaned = "command"
    return f"{timestamp}_{cleaned}_{uuid4().hex[:8]}"
