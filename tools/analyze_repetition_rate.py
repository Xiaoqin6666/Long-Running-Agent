"""Estimate repeated-work rate from an agent run log.

The script treats completed tool actions as effective actions by default and
flags conservative, explainable repeated work:

- read: the same target/query range was already covered within the recent
  context window and without a later mutation
- search: the same search target was already executed without a later mutation
- write/edit: a previous write/edit attempt against the same target failed
- bash/git: the same command was already executed without a later mutation

Protocol errors are excluded from the denominator by default because they are
not valid tool actions.

When the log step number resets or decreases, the script treats that as a new
session and resets repeat history. Cross-session actions are not counted as
repeats.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ACTION_START_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) "
    r"INFO Step (?P<step>\d+) action=(?P<action>\S+) target=(?P<rest>.*)$"
)
OK_RE = re.compile(r"\bok=(?P<ok>True|False) task_id=(?P<task_id>\S+)\b")
OBSERVATION_MARKER = " observation="
STARTING_MARKER = " starting task_id="

READ_RANGE_RE = re.compile(
    r"Read(?: match for '(?P<query>.*?)')? (?P<count>\d+) line\(s\) "
    r"from (?P<path>.*?) lines (?P<start>\d+)-(?P<end>\d+)",
    re.DOTALL,
)
READ_NO_MATCH_RE = re.compile(
    r"Read found no match for '(?P<query>.*?)' in (?P<path>.*?)\.",
    re.DOTALL,
)
SEARCH_FOUND_RE = re.compile(r"Found (?P<count>\d+) match\(es\)")

READ_ACTIONS = {"read"}
SEARCH_ACTIONS = {"search", "list_files"}
MUTATION_ACTIONS = {"write", "edit"}
COMMAND_ACTIONS = {"bash", "git"}


@dataclass(frozen=True)
class ActionEvent:
    timestamp: str
    step: int
    action: str
    target: str
    ok: bool
    task_id: str
    observation: str


@dataclass(frozen=True)
class RepeatFinding:
    event: ActionEvent
    category: str
    reason: str
    previous_ref: str | None = None


@dataclass
class ReadState:
    ranges: list[tuple[int, int, str, int]] = field(default_factory=list)
    no_match_queries: dict[str, tuple[str, int]] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    log_path: Path
    total_completed_actions: int
    effective_actions: int
    repeated_actions: int
    repeated_rate: float
    excluded_protocol_errors: int
    actions_by_type: Counter[str]
    repeats_by_category: Counter[str]
    findings: list[RepeatFinding]
    read_repeat_window: int
    session_count: int


def normalize_target(value: str) -> str:
    value = value.strip().strip('"').strip("'")
    value = value.replace("\\", "/")
    value = re.sub(r"\s+", " ", value)
    return value.lower()


def normalize_command(value: str) -> str:
    value = value.strip()
    lines = [line.rstrip() for line in value.splitlines()]
    return "\n".join(lines)


def event_ref(event: ActionEvent) -> str:
    return f"{event.task_id}#{event.step}"


def parse_events(log_path: Path) -> list[ActionEvent]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    events: list[ActionEvent] = []
    i = 0
    while i < len(lines):
        match = ACTION_START_RE.match(lines[i])
        if not match:
            i += 1
            continue

        rest = match.group("rest")
        if STARTING_MARKER in rest:
            i += 1
            continue

        record_lines = [lines[i]]
        ok_match = OK_RE.search(lines[i])
        j = i + 1
        while ok_match is None and j < len(lines):
            next_start = ACTION_START_RE.match(lines[j])
            if next_start and STARTING_MARKER in next_start.group("rest"):
                break
            record_lines.append(lines[j])
            ok_match = OK_RE.search("\n".join(record_lines))
            j += 1

        record = "\n".join(record_lines)
        ok_match = OK_RE.search(record)
        if ok_match is None:
            i += 1
            continue

        target_text = record.split(" target=", 1)[1][: ok_match.start() - record.find(" target=") - 8]
        target_text = target_text.rstrip()
        observation = ""
        obs_index = record.find(OBSERVATION_MARKER, ok_match.end())
        if obs_index >= 0:
            observation = record[obs_index + len(OBSERVATION_MARKER) :].strip()

        events.append(
            ActionEvent(
                timestamp=match.group("timestamp"),
                step=int(match.group("step")),
                action=match.group("action"),
                target=target_text,
                ok=ok_match.group("ok") == "True",
                task_id=ok_match.group("task_id"),
                observation=observation,
            )
        )
        i = max(j, i + 1)
    return events


def parse_read_identity(event: ActionEvent) -> tuple[str, str | None, int | None, int | None, bool]:
    range_match = READ_RANGE_RE.search(event.observation)
    if range_match:
        target = normalize_target(range_match.group("path"))
        query = range_match.group("query")
        return (
            target,
            query.strip().lower() if query else None,
            int(range_match.group("start")),
            int(range_match.group("end")),
            False,
        )

    no_match = READ_NO_MATCH_RE.search(event.observation)
    if no_match:
        return (
            normalize_target(no_match.group("path")),
            no_match.group("query").strip().lower(),
            None,
            None,
            True,
        )

    return (normalize_target(event.target), None, None, None, False)


def is_range_covered(
    start: int,
    end: int,
    ranges: list[tuple[int, int, str, int]],
    current_index: int,
    read_repeat_window: int,
) -> str | None:
    cursor = start
    previous_ref: str | None = None
    recent_ranges = [
        (prev_start, prev_end, prev_ref, prev_index)
        for prev_start, prev_end, prev_ref, prev_index in ranges
        if current_index - prev_index <= read_repeat_window
    ]
    for prev_start, prev_end, prev_ref, _prev_index in sorted(recent_ranges):
        if prev_end < cursor:
            continue
        if prev_start > cursor:
            return None
        previous_ref = prev_ref
        cursor = max(cursor, prev_end + 1)
        if cursor > end:
            return previous_ref
    return None


def analyze_events(
    events: list[ActionEvent],
    log_path: Path,
    include_protocol_errors: bool,
    read_repeat_window: int,
) -> AnalysisResult:
    completed_actions = len(events)
    effective_events = [
        event
        for event in events
        if include_protocol_errors or event.action != "protocol_error"
    ]

    actions_by_type: Counter[str] = Counter(event.action for event in effective_events)
    findings: list[RepeatFinding] = []
    repeats_by_category: Counter[str] = Counter()

    version_by_target: defaultdict[str, int] = defaultdict(int)
    global_version = 0
    read_history: dict[tuple[str, str | None, int], ReadState] = {}
    search_history: dict[tuple[str, int], str] = {}
    command_history: dict[tuple[str, str, int], str] = {}
    failed_mutations: dict[tuple[str, str], str] = {}
    previous_step: int | None = None
    session_count = 1 if effective_events else 0

    for event_index, event in enumerate(effective_events, start=1):
        if previous_step is not None and event.step <= previous_step:
            session_count += 1
            version_by_target = defaultdict(int)
            global_version = 0
            read_history = {}
            search_history = {}
            command_history = {}
            failed_mutations = {}
        previous_step = event.step

        action = event.action
        normalized_target = normalize_target(event.target)

        finding: RepeatFinding | None = None
        if action in READ_ACTIONS:
            read_target, query, start, end, no_match = parse_read_identity(event)
            version = version_by_target[read_target]
            key = (read_target, query, version)
            state = read_history.setdefault(key, ReadState())
            if no_match and query is not None:
                previous = state.no_match_queries.get(query)
                if previous is not None and event_index - previous[1] <= read_repeat_window:
                    finding = RepeatFinding(
                        event,
                        "read",
                        (
                            "same no-match read query for target at version "
                            f"{version} within {read_repeat_window} effective actions"
                        ),
                        previous[0],
                    )
                state.no_match_queries[query] = (event_ref(event), event_index)
            elif start is not None and end is not None:
                previous = is_range_covered(
                    start,
                    end,
                    state.ranges,
                    event_index,
                    read_repeat_window,
                )
                if previous is not None:
                    finding = RepeatFinding(
                        event,
                        "read",
                        (
                            f"read lines {start}-{end}, already covered for target at "
                            f"version {version} within {read_repeat_window} effective actions"
                        ),
                        previous,
                    )
                state.ranges.append((start, end, event_ref(event), event_index))
            else:
                marker = state.no_match_queries.get("__whole_target__")
                if marker is not None and event_index - marker[1] <= read_repeat_window:
                    finding = RepeatFinding(
                        event,
                        "read",
                        (
                            "same read target/query at version "
                            f"{version} within {read_repeat_window} effective actions"
                        ),
                        marker[0],
                    )
                state.no_match_queries["__whole_target__"] = (event_ref(event), event_index)

        elif action in SEARCH_ACTIONS:
            key = (normalized_target, global_version)
            previous = search_history.get(key)
            if previous is not None:
                finding = RepeatFinding(
                    event,
                    "search",
                    f"same {action} target repeated without an intervening mutation",
                    previous,
                )
            search_history.setdefault(key, event_ref(event))

        elif action in MUTATION_ACTIONS:
            mutation_key = (action, normalized_target)
            previous_failed_step = failed_mutations.get(mutation_key)
            if previous_failed_step is not None:
                finding = RepeatFinding(
                    event,
                    "modification",
                    f"another {action} attempt after a failed attempt on the same target",
                    previous_failed_step,
                )

            if event.ok:
                version_by_target[normalized_target] += 1
                global_version += 1
                failed_mutations.pop(mutation_key, None)
            else:
                failed_mutations.setdefault(mutation_key, event_ref(event))

        elif action in COMMAND_ACTIONS:
            normalized_command = normalize_command(event.target)
            key = (action, normalized_command, global_version)
            previous = command_history.get(key)
            if previous is not None:
                finding = RepeatFinding(
                    event,
                    "command",
                    f"same {action} command repeated without an intervening mutation",
                    previous,
                )
            command_history.setdefault(key, event_ref(event))
            if event.ok and action == "bash" and looks_like_mutating_command(event.target):
                global_version += 1

        if finding is not None:
            findings.append(finding)
            repeats_by_category[finding.category] += 1

    effective_count = len(effective_events)
    repeated_count = len(findings)
    rate = repeated_count / effective_count if effective_count else 0.0
    return AnalysisResult(
        log_path=log_path,
        total_completed_actions=completed_actions,
        effective_actions=effective_count,
        repeated_actions=repeated_count,
        repeated_rate=rate,
        excluded_protocol_errors=completed_actions - effective_count,
        actions_by_type=actions_by_type,
        repeats_by_category=repeats_by_category,
        findings=findings,
        read_repeat_window=read_repeat_window,
        session_count=session_count,
    )


def looks_like_mutating_command(command: str) -> bool:
    lowered = command.lower()
    mutation_markers = (
        " > ",
        ">>",
        ".write(",
        "write_text(",
        "open(",
        "sed -i",
        "mv ",
        "cp ",
        "touch ",
        "mkdir ",
        "python <<",
        "python -c @",
    )
    return any(marker in lowered for marker in mutation_markers)


def truncate(value: str, limit: int = 120) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def result_to_json(result: AnalysisResult) -> dict[str, Any]:
    return {
        "log_path": str(result.log_path),
        "total_completed_actions": result.total_completed_actions,
        "effective_actions": result.effective_actions,
        "repeated_actions": result.repeated_actions,
        "repeated_rate": result.repeated_rate,
        "excluded_protocol_errors": result.excluded_protocol_errors,
        "actions_by_type": dict(result.actions_by_type),
        "repeats_by_category": dict(result.repeats_by_category),
        "read_repeat_window": result.read_repeat_window,
        "session_count": result.session_count,
        "findings": [
            {
                "step": finding.event.step,
                "timestamp": finding.event.timestamp,
                "task_id": finding.event.task_id,
                "action": finding.event.action,
                "target": finding.event.target,
                "ok": finding.event.ok,
                "category": finding.category,
                "reason": finding.reason,
                "previous": finding.previous_ref,
            }
            for finding in result.findings
        ],
    }


def print_text_report(result: AnalysisResult, details_limit: int) -> None:
    print(f"log={result.log_path}")
    print(f"completed_actions={result.total_completed_actions}")
    print(f"effective_actions={result.effective_actions}")
    print(f"repeated_actions={result.repeated_actions}")
    print(f"repeated_rate={result.repeated_rate:.2%}")
    print(f"excluded_protocol_errors={result.excluded_protocol_errors}")
    print(f"read_repeat_window={result.read_repeat_window}")
    print(f"session_count={result.session_count}")

    print("\nactions_by_type:")
    for action, count in sorted(result.actions_by_type.items()):
        print(f"  {action}: {count}")

    print("\nrepeats_by_category:")
    if result.repeats_by_category:
        for category, count in sorted(result.repeats_by_category.items()):
            print(f"  {category}: {count}")
    else:
        print("  none: 0")

    if details_limit == 0:
        return

    print("\nrepeated_action_details:")
    details = result.findings[:details_limit]
    if not details:
        print("  none")
        return
    for finding in details:
        previous = f" previous={finding.previous_ref}" if finding.previous_ref else ""
        print(
            "  "
            f"step={finding.event.step} task_id={finding.event.task_id} "
            f"action={finding.event.action} ok={finding.event.ok}{previous} "
            f"category={finding.category} target={truncate(finding.event.target)} "
            f"reason={finding.reason}"
        )
    remaining = len(result.findings) - len(details)
    if remaining > 0:
        print(f"  ... {remaining} more; rerun with --details-limit {len(result.findings)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate repeated-work count and rate from one agent .log file."
    )
    parser.add_argument("log", type=Path, help="Path to the agent .log file.")
    parser.add_argument(
        "--include-protocol-errors",
        action="store_true",
        help="Include protocol_error events in the denominator.",
    )
    parser.add_argument(
        "--details-limit",
        type=int,
        default=50,
        help="Maximum repeated-action details to print in text mode; use 0 to hide details.",
    )
    parser.add_argument(
        "--read-repeat-window",
        type=int,
        default=20,
        help=(
            "Only count repeated reads when the prior read is within this many "
            "effective actions; older reads are assumed to have fallen out of context."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    if not args.log.exists():
        parser.error(f"log file does not exist: {args.log}")
    if not args.log.is_file():
        parser.error(f"log path is not a file: {args.log}")
    if args.details_limit < 0:
        parser.error("--details-limit must be >= 0")
    if args.read_repeat_window < 0:
        parser.error("--read-repeat-window must be >= 0")

    events = parse_events(args.log)
    result = analyze_events(
        events,
        args.log,
        args.include_protocol_errors,
        args.read_repeat_window,
    )

    if args.json:
        print(json.dumps(result_to_json(result), ensure_ascii=False, indent=2))
    else:
        print_text_report(result, args.details_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
