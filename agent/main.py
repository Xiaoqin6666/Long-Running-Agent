from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
import sys

from agent.loop import AgentLoop
from agent.chat import ChatConfig, InteractiveCLI, launch_chat_window


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
        default=None,
        help="Maximum agent loop iterations per session. Defaults to unlimited.",
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
        "--auto-resume",
        action="store_true",
        help="Automatically start a fresh resumed session after writing a handoff.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=1,
        help="Maximum sessions to run when --auto-resume is enabled.",
    )
    validation_group = parser.add_mutually_exclusive_group()
    validation_group.add_argument(
        "--system-validation",
        dest="system_validation",
        action="store_true",
        default=True,
        help="Run the project-level final system validation task before finish. Enabled by default.",
    )
    validation_group.add_argument(
        "--no-system-validation",
        dest="system_validation",
        action="store_false",
        help="Disable the project-level final system validation task before finish.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Write runtime diagnostics to this file. Defaults to the active state directory's logs folder.",
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
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive terminal conversation with the agent.",
    )
    parser.add_argument(
        "--chat-inline",
        action="store_true",
        help="Run the chat UI in the current terminal instead of opening a new window.",
    )
    parser.add_argument("--chat-child", action="store_true", help=argparse.SUPPRESS)
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


def resolve_log_path(args: argparse.Namespace, benchmark_id: str | None) -> Path:
    if args.log_file:
        path = args.log_file
        return path if path.is_absolute() else (args.root / path).resolve()
    state_dir = args.root.resolve() / "state"
    if benchmark_id:
        state_dir = state_dir / "benchmarks" / benchmark_id
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return state_dir / "logs" / f"run_{stamp}.log"


def configure_run_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("long_agent")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def main() -> int:
    args = build_parser().parse_args()
    if args.chat and not args.chat_inline and not args.chat_child:
        if launch_chat_window(sys.argv[1:], cwd=Path.cwd()):
            print("Long Agent chat opened in a new terminal window.")
            return 0
        print("Could not open a new terminal window; continuing in the current terminal.", file=sys.stderr)
    benchmark_id = infer_benchmark_id(args)
    log_path = resolve_log_path(args, benchmark_id)
    logger = configure_run_logger(log_path)
    print(f"Log: {log_path}", flush=True)
    logger.info(
        "Starting provider=%s benchmark=%s max_steps=%s auto_resume=%s max_sessions=%s system_validation=%s "
        "api_key_configured=%s base_url=%s model=%s",
        args.provider,
        benchmark_id or "none",
        args.max_steps,
        args.auto_resume,
        args.max_sessions,
        args.system_validation,
        bool(os.environ.get("LONG_AGENT_API_KEY")),
        os.environ.get("LONG_AGENT_BASE_URL", "https://api.openai.com/v1"),
        os.environ.get("LONG_AGENT_MODEL", "gpt-4.1-mini"),
    )
    try:
        if args.chat:
            initial_message = resolve_optional_task(args)
            return InteractiveCLI(
                ChatConfig(
                    root=args.root.resolve(),
                    provider=args.provider,
                    max_steps=args.max_steps,
                    benchmark_id=benchmark_id,
                    tasks_path=args.tasks_json.resolve() if args.tasks_json else None,
                    project_spec_path=args.project_spec.resolve() if args.project_spec else None,
                    auto_resume=args.auto_resume,
                    max_sessions=args.max_sessions,
                    system_validation=args.system_validation,
                    initial_message=initial_message,
                )
            ).run()
        task = resolve_task(args)
        loop = AgentLoop(
            root=args.root.resolve(),
            task=task,
            max_steps=args.max_steps,
            provider=args.provider,
            resume=args.resume,
            tasks_path=args.tasks_json.resolve() if args.tasks_json else None,
            project_spec_path=args.project_spec.resolve() if args.project_spec else None,
            benchmark_id=benchmark_id,
            auto_resume=args.auto_resume,
            max_sessions=args.max_sessions,
            system_validation=args.system_validation,
        )
        result = loop.run()
        summary = result.to_human_summary()
        print(summary)
        logger.info(
            "Finished completed=%s steps=%s sessions=%s message=%s",
            result.completed,
            result.steps,
            result.sessions,
            result.message,
        )
        return 0 if result.completed else 1
    except Exception as exc:
        logger.exception("Run failed during startup or execution")
        print(f"Agent failed: {exc}\nLog: {log_path}", file=sys.stderr)
        return 1


def resolve_optional_task(args: argparse.Namespace) -> str | None:
    if args.task_file:
        return args.task_file.read_text(encoding="utf-8").strip()
    if args.task:
        return args.task.strip()
    return None


if __name__ == "__main__":
    raise SystemExit(main())
