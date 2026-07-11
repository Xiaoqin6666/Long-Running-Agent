from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_events(path: Path) -> list[dict]:
    events = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


PROGRESS_ACTIONS = {"contract", "edit", "write", "verify", "finish", "skill"}


def summarize(path: Path, tasks_path: Path | None = None) -> dict:
    events = load_events(path)
    actions: dict[str, int] = {}
    failed = 0
    premature_finish = 0
    handoffs = 0
    contract_rejections = 0
    verifier_failures = 0
    skill_promotions = 0
    skill_rejections = 0
    no_progress_session = 1
    max_session_tokens = 0
    final_task_status: dict[str, str] = {}
    for event in events:
        name = event["action"]["action"]
        actions[name] = actions.get(name, 0) + 1
        observation = event["observation"]
        ok = observation["ok"]
        summary = str(observation.get("summary", ""))
        data = observation.get("data", {})
        if not ok:
            failed += 1
        if name == "finish" and not ok:
            premature_finish += 1
        if event.get("handoff_ready") or data.get("handoff_ready") or "handoff" in summary.lower():
            handoffs += 1
        if name == "contract" and not ok:
            contract_rejections += 1
        if name == "verify" and not ok:
            verifier_failures += 1
        if name == "skill" and ok:
            skill_promotions += 1
        if name == "skill" and not ok:
            skill_rejections += 1
        if name in PROGRESS_ACTIONS and ok:
            no_progress_session = 0
        max_session_tokens = max(max_session_tokens, int(event.get("session_used_tokens", 0)))
        for node in event.get("nodes", []):
            if isinstance(node, dict) and node.get("id"):
                final_task_status[str(node["id"])] = str(node.get("status", "unknown"))
    task_counts = summarize_tasks(tasks_path) if tasks_path else summarize_nodes(final_task_status)
    return {
        "trace": str(path),
        "steps": len(events),
        "actions": actions,
        "failed_observations": failed,
        "premature_finish_attempts": premature_finish,
        "handoff_count": handoffs,
        "contract_rejections": contract_rejections,
        "verifier_failures": verifier_failures,
        "skill_promotions": skill_promotions,
        "skill_rejections": skill_rejections,
        "no_progress_sessions": no_progress_session,
        "max_session_used_tokens": max_session_tokens,
        "completed_tasks": task_counts["completed_tasks"],
        "blocked_tasks": task_counts["blocked_tasks"],
        "task_final_status": task_counts["task_final_status"],
    }


def summarize_nodes(status_by_id: dict[str, str]) -> dict[str, Any]:
    completed = sum(1 for status in status_by_id.values() if status in {"completed", "done"})
    blocked = sum(1 for status in status_by_id.values() if status == "blocked")
    return {"completed_tasks": completed, "blocked_tasks": blocked, "task_final_status": status_by_id}


def summarize_tasks(path: Path) -> dict[str, Any]:
    if not path.exists():
        return summarize_nodes({})
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    status_by_id = {
        str(task.get("id", "")): str(task.get("status", "unknown"))
        for task in tasks
        if isinstance(task, dict) and task.get("id")
    }
    return summarize_nodes(status_by_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize an agent JSONL trace.")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--tasks", type=Path, help="Optional tasks.json path for final task status metrics.")
    args = parser.parse_args()
    print(json.dumps(summarize(args.trace, args.tasks), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
