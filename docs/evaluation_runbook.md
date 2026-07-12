# Evaluation Runbook

This runbook defines the first long-running evaluation task for the Problem 3 harness.

## Task

Use:

```text
eval/benchmarks/issue_tracker/task.md
```

with task graph:

```text
eval/benchmarks/issue_tracker/tasks.json
```

The task asks the agent to build a standard-library Python issue tracker CLI under:

```text
eval/benchmarks/issue_tracker/workspace/
```

## Main-System Run

Configure an OpenAI-compatible provider:

```powershell
$env:LONG_AGENT_API_KEY="..."
$env:LONG_AGENT_BASE_URL="https://api.openai.com/v1"
$env:LONG_AGENT_MODEL="gpt-4.1-mini"
```

Start the first Worker session:

```powershell
python -m agent.main --benchmark issue_tracker --task-file eval\benchmarks\issue_tracker\task.md --tasks-json eval\benchmarks\issue_tracker\tasks.json --provider openai-compatible --max-steps 12
```

Resume after max steps or handoff:

```powershell
python -m agent.main --benchmark issue_tracker --task-file eval\benchmarks\issue_tracker\task.md --tasks-json eval\benchmarks\issue_tracker\tasks.json --provider openai-compatible --max-steps 12 --resume
```

Repeat resume runs until the task graph reaches `completed`, `stopped_with_failure`, or `requires_human_intervention`.

Benchmark execution never owns the host Long-Running Agent repository. `git add` and `git commit` are rejected in benchmark mode, and host worktree cleanliness is not part of benchmark completion. Do not commit harness source, root `state/`, or unrelated traces to make a benchmark finish.

The handoff threshold is an artificial experiment control: `session_budget_tokens * handoff_threshold`. With the default `64000 * 0.75`, a Worker prepares handoff after roughly `48000` estimated tokens. The handoff Markdown is only a concise resume index; structured details are written to the active benchmark state directory, for example `state/benchmarks/issue_tracker/handoff_payload.json`.

## Local Checks

Run the task-specific hidden acceptance:

```powershell
python eval\benchmarks\issue_tracker\hidden_acceptance.py
```

During autonomous benchmark execution, the harness selects this script from the active `benchmark_id`. The Worker cannot read or modify it, and verifier/trace state records only a redacted pass/fail summary rather than hidden output.

Run trace metrics after each session:

```powershell
python eval\metrics.py state\benchmarks\issue_tracker\traces\<trace-file>.jsonl --tasks state\benchmarks\issue_tracker\runtime_tasks.json
```
## Output Locations

- Framework source under test: `agent/`, `tests/`, `eval/metrics.py`, and task/eval definitions
- Task spec: `eval/benchmarks/issue_tracker/task.md`
- Source task graph: `eval/benchmarks/issue_tracker/tasks.json`
- Runtime task graph and final task states: `state/benchmarks/issue_tracker/runtime_tasks.json`
- Generated implementation output: `eval/benchmarks/issue_tracker/workspace/`
- Current Worker state: `state/benchmarks/issue_tracker/current_task.json`
- Session handoff: `state/benchmarks/issue_tracker/handoff.md`
- Structured handoff payload: `state/benchmarks/issue_tracker/handoff_payload.json`
- Latest verifier report: `state/benchmarks/issue_tracker/verifier_report.md`
- Hard/Soft memory: `state/benchmarks/issue_tracker/hard_memory.md`, `state/benchmarks/issue_tracker/soft_memory.md`
- Skills: `state/benchmarks/issue_tracker/skills/`
- Raw traces: `state/benchmarks/issue_tracker/traces/run_*.jsonl`
- Metrics JSON: terminal output from `python eval\metrics.py ...`
- Task-specific hidden acceptance output: terminal JSON from `python eval\benchmarks\issue_tracker\hidden_acceptance.py`

Each benchmark now owns its own directory under `eval/benchmarks/<benchmark_name>/`. Keep benchmark definitions and generated application output together:

```text
eval/benchmarks/<benchmark_name>/
  project_spec.md or task.md
  tasks.json               # only for preplanned benchmarks
  hidden_acceptance.py     # optional benchmark-local final check
  workspace/               # generated app output, ignored by Git
```

`eval/benchmarks/*/workspace/` and `state/benchmarks/*/traces/` are generated run artifacts and are ignored by Git. This keeps Issue Tracker and Todo Counter from seeing each other's files during list/read/search actions.

Benchmark-local state is isolated by `--benchmark`:

```text
state/
  benchmarks/
    issue_tracker/
      current_task.json
      runtime_tasks.json       # preplanned flow only
      generated_tasks.json     # autonomous Initializer flow only
      project_spec.md
      init.sh
      rejected_candidates/
        generated_tasks.json
      handoff.md
      handoff_payload.json
      verifier_report.md
      memory.md
      hard_memory.md
      soft_memory.md
      skills/
      traces/
    todo_counter/
      current_task.json
      runtime_tasks.json       # preplanned flow only
      generated_tasks.json     # autonomous Initializer flow only
      project_spec.md
      init.sh
      rejected_candidates/
        generated_tasks.json
      handoff.md
      handoff_payload.json
      verifier_report.md
      memory.md
      hard_memory.md
      soft_memory.md
      skills/
      traces/
```

If `--benchmark` is omitted, the CLI infers it from paths under `eval/benchmarks/<benchmark_name>/...`. Passing `--benchmark` explicitly is still recommended for reproducible experiments.

Task-specific expected files must be declared in the task graph with `expected_artifacts`, and task-specific test commands should be declared with `verification_commands`. The agent harness reads those fields generically; it should not hard-code file names such as `store.py`, `cli.py`, or any benchmark-specific implementation details.

For coding tasks, split artifacts by ownership when tests are involved:

- `implementation_artifacts`: source files the Worker should repair first after failed acceptance.
- `worker_test_artifacts`: tests the Worker may create or revise before they are promoted into acceptance evidence.
- `acceptance_artifacts`: tests or scripts used as part of the agreed contract.
- `frozen_acceptance_artifacts`: acceptance tests that are read-only for the Worker unless the harness explicitly records `allow_test_repair=true`.
- `test_policy`: records whether worker tests are mutable before contract freeze and whether acceptance-test repair requires verifier approval.

After a failed acceptance command, the repair gate preserves already-read diagnostic files across repeated failures. If both the failing test and implementation source have already been read, the next action is forced toward `write` or `edit` on an implementation artifact rather than repeatedly reading the same files.

## Project-Spec Initializer Flow

For benchmark cases where the agent should plan autonomously, provide only a project specification:

```powershell
python -m agent.main --benchmark todo_counter --project-spec eval\benchmarks\todo_counter\project_spec.md --provider openai-compatible --max-steps 12
```

When `--project-spec` is used without `--tasks-json`, the harness starts a one-time `INIT` task. The initializer must produce:

- `state/benchmarks/todo_counter/project_spec.md`
- `state/benchmarks/todo_counter/generated_tasks.json`
- `state/benchmarks/todo_counter/init.sh`

The last path is the run-local benchmark initializer script. It is distinct from the tracked repository-root `init.sh`, which bootstraps the Long-Running Agent harness itself. Benchmark INIT must never overwrite the repository-root script. Generated application code and public tests belong under `eval/benchmarks/todo_counter/workspace/`, not under `state/`.

`INIT` does not negotiate a Worker acceptance contract. The harness permits it to write only those three benchmark-local initializer artifacts and permits shell execution only for the deterministic INIT verification command. Application code, tests, skeletons, and workspace files remain forbidden until Orchestrator selects the first ordinary Worker task.

`answer` and `finish` cannot terminate INIT. Once all artifacts pass deterministic validation, the guard forces the INIT verification command; after that command succeeds, it forces `verify`. Only Verifier PASS marks INIT completed, and Orchestrator selects the first Worker task before any token-budget handoff is written.

Before INIT can pass, the harness validates the generated JSON schema, task ids, dependencies, initial statuses, acceptance criteria, verification commands, hidden-test isolation, and artifact paths. For this benchmark, every generated application artifact must remain under `eval/benchmarks/todo_counter/workspace/` as required by the project specification.

After `state/benchmarks/todo_counter/generated_tasks.json` exists and the initializer passes verification, Orchestrator uses that generated task graph for ordinary Worker scheduling. This keeps benchmark input implementation-independent while still making the generated plan durable and inspectable.

## Baseline And Ablation Plan

Use the same task file and task graph for every condition. Before each condition, reset `eval/benchmarks/issue_tracker/tasks.json` to its initial pending state and remove `eval/benchmarks/issue_tracker/workspace/`.

Suggested conditions:

- main system: all mechanisms enabled;
- no handoff: disable session handoff threshold or set a very large budget;
- no verifier: allow finish without verifier gate only in an experimental branch;
- no memory/skill: exclude memory and skill files from Startup/Persistent Context;
- no explicit task state: run a simple ReAct-style loop without `tasks.json`.

Compare with:

```powershell
python eval\metrics.py state\benchmarks\issue_tracker\traces\<trace-file>.jsonl --tasks state\benchmarks\issue_tracker\runtime_tasks.json
```
