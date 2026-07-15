from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = Path(__file__).resolve().parent
WORKSPACE = BENCHMARK / "workspace"
STATE = ROOT / "state" / "benchmarks" / "skill_mechanism"


def load_json_stream(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    decoder = json.JSONDecoder()
    events: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        event, index = decoder.raw_decode(text, index)
        if isinstance(event, dict):
            events.append(event)
    return events


def load_function(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_path


def check_behavior() -> dict[str, Any]:
    try:
        first = load_function(WORKSPACE / "first_path.py")
        second = load_function(WORKSPACE / "second_path.py")
        cases = {
            r"C:\alpha//beta\file.txt": "C:/alpha/beta/file.txt",
            r"/srv//app\data": "/srv/app/data",
            r"relative\\folder///item": "relative/folder/item",
            "": "",
        }
        ok = all(first(value) == expected and second(value) == expected for value, expected in cases.items())
        return {"name": "behavior", "ok": ok}
    except Exception as exc:
        return {"name": "behavior", "ok": False, "error": repr(exc)}


def check_skill_artifacts() -> dict[str, Any]:
    try:
        skill = STATE / "skills" / "normalize-portable-path.md"
        candidates = sorted((STATE / "skill_candidates").glob("SC-*.json"))
        if not skill.is_file() or not candidates:
            return {"name": "skill_artifacts", "ok": False, "error": "Skill or candidate missing"}
        candidate = json.loads(candidates[-1].read_text(encoding="utf-8"))
        statuses = [item.get("status") for item in candidate.get("status_history", [])]
        expected = ["proposed", "evidence_validated", "content_validated", "approved", "promoted"]
        refs = candidate.get("evidence_refs", [])
        ok = (
            candidate.get("status") == "promoted"
            and statuses == expected
            and refs
            and refs[0].get("type") == "verifier_report"
            and refs[0].get("task_id") == "T1"
            and refs[0].get("report_id")
            and (STATE / "verifier_reports" / f"{refs[0]['report_id']}.json").is_file()
            and "evidence_refs" not in skill.read_text(encoding="utf-8")
        )
        return {"name": "skill_artifacts", "ok": ok}
    except Exception as exc:
        return {"name": "skill_artifacts", "ok": False, "error": repr(exc)}


def check_trace_protocol() -> dict[str, Any]:
    try:
        events = []
        for path in sorted((STATE / "traces").glob("run_*.jsonl")):
            events.extend(load_json_stream(path))
        save_indexes = [
            index
            for index, event in enumerate(events)
            if event.get("action", {}).get("action") in {"save_skill", "skill"}
            and event.get("observation", {}).get("ok") is True
        ]
        load_indexes = [
            index
            for index, event in enumerate(events)
            if event.get("action", {}).get("action") == "load_skill"
            and event.get("action", {}).get("target") == "normalize-portable-path"
            and event.get("observation", {}).get("ok") is True
        ]
        second_write_indexes = [
            index
            for index, event in enumerate(events)
            if event.get("action", {}).get("action") in {"write", "edit"}
            and str(event.get("action", {}).get("target", "")).replace("\\", "/").endswith("workspace/second_path.py")
        ]
        loaded_content_ok = any(
            "# Instructions" in str(events[index].get("observation", {}).get("data", {}).get("content", ""))
            for index in load_indexes
        )
        ordered = bool(
            save_indexes
            and load_indexes
            and second_write_indexes
            and save_indexes[0] < load_indexes[0] < second_write_indexes[0]
        )
        return {
            "name": "trace_protocol",
            "ok": ordered and loaded_content_ok,
            "save_count": len(save_indexes),
            "load_count": len(load_indexes),
        }
    except Exception as exc:
        return {"name": "trace_protocol", "ok": False, "error": repr(exc)}


def main() -> int:
    checks = [check_behavior(), check_skill_artifacts(), check_trace_protocol()]
    result = {"ok": all(item["ok"] for item in checks), "checks": checks}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
