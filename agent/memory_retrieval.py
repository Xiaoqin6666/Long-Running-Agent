from __future__ import annotations

from dataclasses import dataclass


MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000


@dataclass(frozen=True)
class EntrypointTruncation:
    content: str
    was_line_truncated: bool
    was_byte_truncated: bool
    line_count: int
    byte_count: int


def truncate_entrypoint_content(raw: str) -> EntrypointTruncation:
    line_count = len(raw.splitlines())
    byte_count = len(raw.encode("utf-8"))
    was_line_truncated = line_count > MAX_ENTRYPOINT_LINES
    was_byte_truncated = byte_count > MAX_ENTRYPOINT_BYTES
    if not was_line_truncated and not was_byte_truncated:
        return EntrypointTruncation(raw, False, False, line_count, byte_count)

    lines = raw.splitlines()[:MAX_ENTRYPOINT_LINES]
    content = "\n".join(lines)
    while len(content.encode("utf-8")) > MAX_ENTRYPOINT_BYTES:
        content = content[:-256] if len(content) > 256 else ""
    warning = (
        "\n\n> WARNING: memory.md was truncated because it exceeded "
        f"{MAX_ENTRYPOINT_LINES} lines or {MAX_ENTRYPOINT_BYTES} bytes."
    )
    return EntrypointTruncation(content.rstrip() + warning, was_line_truncated, was_byte_truncated, line_count, byte_count)
