from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "eval" / "benchmarks" / "issue_tracker" / "workspace"


def run(command: list[str], cwd: Path | None = None) -> dict:
    completed = subprocess.run(
        command,
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    return {
        "command": command,
        "cwd": str(cwd or ROOT),
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "output": (completed.stdout + completed.stderr).strip()[-4000:],
    }


def main() -> int:
    checks = []
    checks.append({"command": ["exists", str(APP)], "ok": APP.exists(), "returncode": 0 if APP.exists() else 1, "output": ""})
    if not APP.exists():
        print(json.dumps({"ok": False, "checks": checks}, ensure_ascii=False, indent=2))
        return 1

    checks.append(run(["python", "-m", "unittest", "discover", "-s", "tests"], cwd=APP))

    temp_dir = Path(tempfile.mkdtemp(prefix="issue-tracker-acceptance-"))
    data_file = temp_dir / "issues.json"
    try:
        commands = [
            ["python", "-m", "issue_tracker.cli", "--data-file", str(data_file), "create", "--title", "Alpha", "--description", "first", "--priority", "high"],
            ["python", "-m", "issue_tracker.cli", "--data-file", str(data_file), "list"],
            ["python", "-m", "issue_tracker.cli", "--data-file", str(data_file), "show", "1"],
            ["python", "-m", "issue_tracker.cli", "--data-file", str(data_file), "update", "1", "--status", "closed"],
            ["python", "-m", "issue_tracker.cli", "--data-file", str(data_file), "delete", "1"],
        ]
        checks.extend(run(command, cwd=APP) for command in commands)
        if data_file.exists():
            payload = json.loads(data_file.read_text(encoding="utf-8"))
            checks.append(
                {
                    "command": ["json_file_shape", str(data_file)],
                    "ok": isinstance(payload, dict) and "issues" in payload,
                    "returncode": 0 if isinstance(payload, dict) and "issues" in payload else 1,
                    "output": "",
                }
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    result = {"ok": all(check["ok"] for check in checks), "checks": checks}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
