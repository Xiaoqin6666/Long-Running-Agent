from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    events = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def summarize(path: Path) -> dict:
    events = load_events(path)
    actions: dict[str, int] = {}
    failed = 0
    premature_finish = 0
    for event in events:
        name = event["action"]["action"]
        actions[name] = actions.get(name, 0) + 1
        ok = event["observation"]["ok"]
        if not ok:
            failed += 1
        if name == "finish" and not ok:
            premature_finish += 1
    return {
        "trace": str(path),
        "steps": len(events),
        "actions": actions,
        "failed_observations": failed,
        "premature_finish_attempts": premature_finish,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize an agent JSONL trace.")
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.trace), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

