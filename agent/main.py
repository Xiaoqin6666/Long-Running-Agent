from __future__ import annotations

import argparse
from pathlib import Path
import re

from agent.loop import AgentLoop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="long-agent",
        description="Run a minimal long-running coding agent loop.",
    )
    parser.add_argument("task", nargs="?", help="Coding task to run.")
    parser.add_argument(
        "--task-file",
        type=Path,
        help="Read the task description from a file.",
    )
    parser.add_argument(
        "--project-spec",
        type=Path,
        help="Read a project specification and let the initializer generate the task graph.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum agent loop iterations.",
    )
    parser.add_argument(
        "--provider",
        choices=["offline", "openai-compatible"],
        default="offline",
        help="LLM provider. Use openai-compatible for real API calls.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the active state directory's current_task.json if it exists.",
    )
    parser.add_argument(
        "--benchmark",
        help="Benchmark id for isolated state under state/benchmarks/<id>. Inferred from eval/benchmarks/<id>/ paths when omitted.",
    )
    parser.add_argument(
        "--tasks-json",
        type=Path,
        help="Use a task graph JSON file other than ./tasks.json.",
    )
    return parser


def resolve_task(args: argparse.Namespace) -> str:
    if args.project_spec:
        return args.project_spec.read_text(encoding="utf-8").strip()
    if args.task_file:
        return args.task_file.read_text(encoding="utf-8").strip()
    if args.task:
        return args.task.strip()
    raise SystemExit("Provide a task string or --task-file.")


def infer_benchmark_id(args: argparse.Namespace) -> str | None:
    if args.benchmark:
        return sanitize_benchmark_id(args.benchmark)
    for candidate in (args.project_spec, args.task_file, args.tasks_json):
        if not candidate:
            continue
        parts = [part.lower() for part in Path(candidate).parts]
        for index in range(len(parts) - 2):
            if parts[index] == "eval" and parts[index + 1] == "benchmarks":
                return sanitize_benchmark_id(Path(candidate).parts[index + 2])
    return None


def sanitize_benchmark_id(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", raw.strip())
    cleaned = cleaned.strip("-_")
    if not cleaned:
        raise SystemExit("Benchmark id must contain a letter, number, underscore, or dash.")
    return cleaned


def main() -> int:
    args = build_parser().parse_args()
    task = resolve_task(args)
    benchmark_id = infer_benchmark_id(args)
    loop = AgentLoop(
        root=args.root.resolve(),
        task=task,
        max_steps=args.max_steps,
        provider=args.provider,
        resume=args.resume,
        tasks_path=args.tasks_json.resolve() if args.tasks_json else None,
        project_spec_path=args.project_spec.resolve() if args.project_spec else None,
        benchmark_id=benchmark_id,
    )
    result = loop.run()
    print(result.to_human_summary())
    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
