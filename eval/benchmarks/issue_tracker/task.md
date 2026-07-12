# Long-Running Evaluation Task: Issue Tracker CLI

Build a small issue-tracker application inside `eval/benchmarks/issue_tracker/workspace`.

The goal is intentionally larger than a single short coding session. The agent should use task state, verifier feedback, handoff, and metrics rather than relying on chat history.

## Product Goal

Create a Python command-line issue tracker with JSON-file persistence.

The application should support:

- creating an issue with title, description, priority, and status;
- listing issues;
- showing one issue by id;
- updating title, description, priority, or status;
- deleting an issue;
- persistence in a JSON file;
- deterministic tests for storage and CLI behavior;
- a README explaining how to run the app and tests.

## Required Location

All implementation files must live under:

```text
eval/benchmarks/issue_tracker/workspace/
```

Suggested structure:

```text
eval/benchmarks/issue_tracker/workspace/
  issue_tracker/
    __init__.py
    cli.py
    store.py
  tests/
    test_store.py
    test_cli.py
  README.md
```

## Constraints

- Use only the Python standard library.
- Do not install dependencies.
- Keep all generated app state under `eval/benchmarks/issue_tracker/workspace/.data/` or a temporary path in tests.
- The app must run on Windows PowerShell.
- Use `python -m issue_tracker.cli ...` from inside `eval/benchmarks/issue_tracker/workspace`.
- Do not mark the evaluation complete until the task-specific hidden acceptance script passes.

## Acceptance Checks

The final candidate must pass:

```powershell
python -m unittest discover -s eval\benchmarks\issue_tracker\workspace\tests
python eval\benchmarks\issue_tracker\hidden_acceptance.py
```

The agent trace should show:

- at least one acceptance contract before editing code;
- verifier feedback before claiming completion;
- task status transitions in `eval/benchmarks/issue_tracker/tasks.json`;
- at least one handoff when using the artificial 16K/70% session budget, or a documented reason if the run finishes before threshold.

## Evaluation Notes

Use this task for:

- main system run;
- no-handoff ablation;
- no-verifier ablation;
- no-memory/skill ablation;
- no-explicit-task-state ablation.

Compare runs using `eval/metrics.py`.
