from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    return {
        "command": command,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "output": (completed.stdout + completed.stderr).strip()[-4000:],
    }


def task_graph_check() -> dict:
    tasks = json.loads((ROOT / "tasks.json").read_text(encoding="utf-8")).get("tasks", [])
    required = [task for task in tasks if task.get("optional") is not True]
    ok = bool(required) and all(task.get("status") in {"completed", "done"} for task in required)
    return {
        "command": ["task_graph"],
        "ok": ok,
        "returncode": 0 if ok else 1,
        "output": f"{sum(1 for task in required if task.get('status') in {'completed', 'done'})}/{len(required)} required tasks completed.",
    }


def main() -> int:
    checks = [
        run(["python", "-m", "compileall", "agent", "eval", "tests"]),
        run(["python", "-m", "unittest", "discover", "-s", "tests"]),
        run(["python", "-m", "agent.main", "--help"]),
        task_graph_check(),
    ]
    payload = {"ok": all(item["ok"] for item in checks), "checks": checks}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
