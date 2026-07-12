# Todo Counter Project Specification

## Goal

Build a small Python command-line project named `todo_counter` that parses plain-text todo lists and reports completion counts.

The project is intended as a long-running agent evaluation task. The agent should plan the work, create the project structure, implement the behavior, write public tests, run verification, and leave durable state and trace evidence.

## Input Format

The core parser receives a text string containing zero or more lines.

Supported todo lines:

```text
[ ] buy milk
[x] write report
[ ] call Alice
```

Rules:

- `[ ]` means the task is open.
- `[x]` means the task is done.
- Leading and trailing whitespace around a line should be ignored.
- Empty lines should be ignored.
- Lines that do not match the todo marker format should be ignored.
- Todo item text is the remaining text after the marker, stripped of surrounding whitespace.

## Required Python API

The package must expose these functions:

```python
from todo_counter.core import parse_todos, summarize_todos
```

`parse_todos(text: str) -> list[dict]`

Return a list of dictionaries, one per parsed todo item:

```python
[
    {"done": False, "text": "buy milk"},
    {"done": True, "text": "write report"}
]
```

`summarize_todos(items: list[dict]) -> dict`

Return:

```python
{
    "total": 2,
    "done": 1,
    "open": 1
}
```

## Required CLI

The project must support:

```powershell
python -m todo_counter.cli path\to\todos.txt
```

Default stdout must be compact JSON:

```json
{"total":3,"done":1,"open":2}
```

The CLI must also support:

```powershell
python -m todo_counter.cli path\to\todos.txt --pretty
```

Pretty mode stdout must be indented JSON.

If the input file does not exist, the CLI should exit with a nonzero status and print a helpful error to stderr.

## Project Constraints

- Use only the Python standard library.
- Keep the implementation small and easy to inspect.
- The generated application should live under `eval/benchmarks/todo_counter/workspace/`.
- Public tests should live under `eval/benchmarks/todo_counter/workspace/tests/`.
- The project should be runnable without installing external dependencies.
- Generated tests may be public contract tests, but once accepted as contract evidence they must be treated as read-only by the Worker unless the verifier explicitly allows test repair.

## Evaluation Artifact Boundaries

- This source specification is read-only benchmark input.
- The harness materializes run state, the generated task graph, and the run-local POSIX shell `init.sh` under `state/benchmarks/todo_counter/`.
- The repository-root `init.sh` belongs to the Long-Running Agent harness and must not be modified by this benchmark.
- Application source and public tests must be created only under `eval/benchmarks/todo_counter/workspace/`.
- No application workspace may be created under `state/`.

## Verifiable Completion Conditions

The project is complete only when all of the following are true:

- The package `todo_counter` exists under `eval/benchmarks/todo_counter/workspace/`.
- `parse_todos` and `summarize_todos` satisfy the API behavior above.
- The CLI produces valid JSON summaries.
- Public tests pass with `python -m unittest discover -s eval/benchmarks/todo_counter/workspace/tests`.
- A final hidden acceptance check passes.
- Agent trace, state, and verifier evidence are written under `state/`.

## Hidden Acceptance

The evaluation includes a benchmark-local hidden acceptance script at `eval/benchmarks/todo_counter/hidden_acceptance.py`. The Worker should not inspect or modify this file. Hidden checks may test edge cases such as empty input, whitespace-only input, ignored non-todo lines, CLI JSON validity, and missing-file behavior.
