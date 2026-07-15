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

Skill Mechanism is a small preplanned lifecycle benchmark. It verifies a first implementation, promotes a Skill from real verifier evidence, then requires that Skill to be loaded before a second implementation:

```powershell
python -m agent.main --benchmark skill_mechanism --task-file eval\benchmarks\skill_mechanism\task.md --tasks-json eval\benchmarks\skill_mechanism\tasks.json --provider openai-compatible --max-steps 16
```

Resume if needed:

```powershell
python -m agent.main --benchmark skill_mechanism --task-file eval\benchmarks\skill_mechanism\task.md --tasks-json eval\benchmarks\skill_mechanism\tasks.json --provider openai-compatible --max-steps 16 --resume
```

Run its final evaluator with:

```powershell
python eval\benchmarks\skill_mechanism\hidden_acceptance.py
```

Do not use `--resume` when switching between benchmarks. Resume only within the same benchmark run.
