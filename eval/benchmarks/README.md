# Benchmarks

Each benchmark owns its input files, optional hidden acceptance script, and generated workspace.

```text
eval/benchmarks/
  issue_tracker/
    task.md
    tasks.json
    hidden_acceptance.py
    workspace/
  todo_counter/
    project_spec.md
    hidden_acceptance.py
    workspace/
```

Issue Tracker is a preplanned task-graph benchmark:

```powershell
python -m agent.main --task-file eval\benchmarks\issue_tracker\task.md --tasks-json eval\benchmarks\issue_tracker\tasks.json --provider openai-compatible --max-steps 12
```

Todo Counter is an initializer/planner benchmark:

```powershell
python -m agent.main --project-spec eval\benchmarks\todo_counter\project_spec.md --provider openai-compatible --max-steps 12
```

Do not use `--resume` when switching between benchmarks. Resume only within the same benchmark run.
