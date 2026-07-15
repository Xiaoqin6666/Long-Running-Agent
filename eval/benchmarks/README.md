# Benchmarks

Each benchmark task directory owns only Agent-visible input files and its generated workspace. Optional experimenter-only evaluators live separately under `eval/manual_evaluators/` and are run manually after the autonomous project has ended.

```text
eval/benchmarks/
  issue_tracker/
    task.md
    tasks.json
    workspace/
  todo_counter/
    project_spec.md
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

After the autonomous run, optionally run its manual evaluator with:

```powershell
python eval\manual_evaluators\skill_mechanism\evaluate.py
```

Do not use `--resume` when switching between benchmarks. Resume only within the same benchmark run.
