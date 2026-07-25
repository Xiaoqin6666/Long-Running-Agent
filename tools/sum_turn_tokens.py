"""Sum turn_tokens values from agent log files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TURN_TOKENS_RE = re.compile(r"\bturn_tokens=(\d+)\b")


def sum_turn_tokens(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for match in TURN_TOKENS_RE.finditer(line):
                total += int(match.group(1))
                count += 1
    return total, count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sum all turn_tokens=<number> values in one or more agent log files."
    )
    parser.add_argument("logs", nargs="+", type=Path, help="Path(s) to .log file(s).")
    args = parser.parse_args()

    grand_total = 0
    grand_count = 0
    for log_path in args.logs:
        total, count = sum_turn_tokens(log_path)
        grand_total += total
        grand_count += count
        print(f"{log_path}: count={count} total_turn_tokens={total}")

    if len(args.logs) > 1:
        print(f"ALL: count={grand_count} total_turn_tokens={grand_total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
