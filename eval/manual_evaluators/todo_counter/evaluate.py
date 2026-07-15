from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "eval" / "benchmarks" / "todo_counter"
APP = BENCHMARK / "workspace"


def run(command: list[str], cwd: Path | None = None) -> dict:
    completed = subprocess.run(
        command,
        cwd=cwd or APP,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    return {
        "command": command,
        "cwd": str(cwd or APP),
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "output": (completed.stdout + completed.stderr).strip()[-4000:],
    }


def check_core_api() -> dict:
    sys.path.insert(0, str(APP))
    try:
        from todo_counter.core import parse_todos, summarize_todos

        text = "\n[ ] buy milk\n[x] write report\nnot a todo\n[ ] call Alice\n"
        items = parse_todos(text)
        summary = summarize_todos(items)
        ok = (
            items
            == [
                {"done": False, "text": "buy milk"},
                {"done": True, "text": "write report"},
                {"done": False, "text": "call Alice"},
            ]
            and summary == {"total": 3, "done": 1, "open": 2}
            and summarize_todos(parse_todos("")) == {"total": 0, "done": 0, "open": 0}
        )
        return {"command": ["core_api"], "cwd": str(APP), "ok": ok, "returncode": 0 if ok else 1, "output": ""}
    except Exception as exc:
        return {"command": ["core_api"], "cwd": str(APP), "ok": False, "returncode": 1, "output": repr(exc)}


def check_cli_json() -> dict:
    with tempfile.TemporaryDirectory(prefix="todo-counter-acceptance-") as tmp:
        todo_file = Path(tmp) / "todos.txt"
        todo_file.write_text("[ ] one\n[x] two\n\nignored\n", encoding="utf-8")
        result = run(["python", "-m", "todo_counter.cli", str(todo_file)], cwd=APP)
        if not result["ok"]:
            return result
        try:
            payload = json.loads(result["output"])
        except json.JSONDecodeError as exc:
            result["ok"] = False
            result["returncode"] = 1
            result["output"] = f"stdout was not valid JSON: {exc}; output={result['output']}"
            return result
        result["ok"] = payload == {"total": 2, "done": 1, "open": 1}
        result["returncode"] = 0 if result["ok"] else 1
        return result


def check_missing_file() -> dict:
    result = run(["python", "-m", "todo_counter.cli", "__missing_todos__.txt"], cwd=APP)
    result["ok"] = result["returncode"] != 0
    return result


def main() -> int:
    checks = [
        {"command": ["exists", str(APP)], "cwd": str(BENCHMARK), "ok": APP.exists(), "returncode": 0 if APP.exists() else 1, "output": ""},
    ]
    if APP.exists():
        checks.append(run(["python", "-m", "unittest", "discover", "-s", "tests"], cwd=APP))
        checks.append(check_core_api())
        checks.append(check_cli_json())
        checks.append(check_missing_file())
    result = {"ok": all(check["ok"] for check in checks), "checks": checks}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
