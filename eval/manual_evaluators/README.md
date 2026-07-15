# Manual Evaluators

These evaluators are experimenter-only checks. They deliberately live outside `eval/benchmarks/<id>/` so they are absent from the Agent's task directory during autonomous execution.

The Agent Harness never discovers or invokes these files. Run an evaluator yourself only after the corresponding project run has ended:

```powershell
python eval\manual_evaluators\issue_tracker\evaluate.py
python eval\manual_evaluators\todo_counter\evaluate.py
python eval\manual_evaluators\skill_mechanism\evaluate.py
```

Evaluator results are external experiment measurements. They do not gate `verify` or `finish`, modify the task graph, or create repair nodes.
