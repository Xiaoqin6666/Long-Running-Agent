from __future__ import annotations

import argparse
from pathlib import Path

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
        help="Resume from state/current_task.json if it exists.",
    )
    return parser


def resolve_task(args: argparse.Namespace) -> str:
    if args.task_file:
        return args.task_file.read_text(encoding="utf-8").strip()
    if args.task:
        return args.task.strip()
    raise SystemExit("Provide a task string or --task-file.")


def main() -> int:
    args = build_parser().parse_args()
    task = resolve_task(args)
    loop = AgentLoop(
        root=args.root.resolve(),
        task=task,
        max_steps=args.max_steps,
        provider=args.provider,
        resume=args.resume,
    )
    result = loop.run()
    print(result.to_human_summary())
    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
