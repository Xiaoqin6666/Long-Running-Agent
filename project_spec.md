# Long-Running Agent Project Specification

## Goal

Build a research-oriented coding-agent harness that can plan, implement, verify, hand off, resume, and evaluate work across multiple model context windows with minimal human intervention.

## Roles

- Initializer / Planner converts a project specification into a durable task graph and run-local initializer entrypoint.
- Orchestrator selects exactly one ready task and owns task-state transitions.
- Worker executes one selected coding task at a time.
- Verifier independently evaluates acceptance evidence and is the only role that can authorize completion.
- Project Terminator distinguishes successful completion, stopped-with-failure, and human-intervention outcomes.

## Repository Bootstrap Artifacts

These tracked root files describe and bootstrap the Long-Running Agent harness itself:

- `project_spec.md`: this framework specification.
- `tasks.json`: the framework's own development task graph.
- `init.sh`: static repository setup, test, compile, and smoke entrypoint.

Benchmark agents must not overwrite these files.

## Benchmark Inputs

Each benchmark owns an input directory:

```text
eval/benchmarks/<benchmark_id>/
  project_spec.md or task.md
  tasks.json                 # optional preplanned source graph
  hidden_acceptance.py       # evaluator-owned; Worker cannot read or modify
  workspace/                 # generated application and public tests
```

The source project specification and preplanned task graph are read-only benchmark inputs.

## Benchmark Runtime State

Every benchmark run is isolated under:

```text
state/benchmarks/<benchmark_id>/
  project_spec.md
  generated_tasks.json       # autonomous Initializer flow
  runtime_tasks.json         # copied preplanned graph
  init.sh                    # run-local POSIX shell initializer entrypoint
  current_task.json
  handoff.md
  handoff_payload.json
  verifier_report.md
  memory.md
  hard_memory.md
  soft_memory.md
  skills/
  traces/
```

The run-local `init.sh` is distinct from the repository-root `init.sh`. It must begin with `#!/usr/bin/env sh` and `set -eu`. It may invoke Python commands, but it must not contain Python source code or create an application workspace under `state/`.

## Initializer Rules

- INIT may write only the three paths named by its active task: the materialized project specification, generated task graph, and run-local init script.
- INIT does not require a Worker acceptance contract.
- INIT cannot write application code, public tests, skeleton files, or workspace files.
- INIT cannot terminate through `answer` or `finish`.
- The required transition is `artifacts_ready -> verification_command_passed -> verifier_passed -> first_worker_scheduled`.
- Generated task artifacts must remain under the workspace root declared by the benchmark project specification.
- Generated task commands must respect dependency and runtime constraints from the specification and must not contain placeholder checks.

## Worker And Verifier Rules

- The Worker acts only on the Orchestrator-selected task.
- Coding starts only after Worker and Verifier agree on an implementation-independent acceptance contract.
- Worker may not mark a task completed.
- Worker-owned tests may be edited before contract freeze.
- Frozen acceptance tests are read-only unless Verifier explicitly authorizes repair.
- Hidden acceptance tests remain evaluator-owned and inaccessible to Worker.
- Verifier PASS followed by an Orchestrator transition is required for task completion.

## Context And Memory

- Always-on context contains stable role, task, tool, and completion rules.
- Startup context restores benchmark-local specification, task graph, handoff, verifier report, and Git state.
- Just-in-time context is gathered through list, search, read, and test-error evidence.
- Persistent context stores task status, verified facts, architecture decisions, failed attempts, verifier reports, commits, and next actions.
- Hard Memory contains only evidence-backed state.
- Soft Memory contains hypotheses, suspected causes, reflection, and suggested next actions.
- Skills are promoted only from verifier-confirmed success or evidence-confirmed failure.

## Long-Running Behavior

- Default Worker session budget is 64K estimated tokens.
- Handoff preparation begins at 75% of the session budget.
- A session past the threshold must not start a large new edit.
- Handoff must preserve active task, budget, evidence, failures, contracts, verifier state, and resume instructions.

## Project Completion

Successful completion requires all required tasks completed, no unresolved blocked task, regression tests passing, hidden acceptance passing, and a clean runnable repository.

Budget exhaustion, repeated critical failure, unrecoverable environment failure, all remaining tasks blocked, or repeated no-progress sessions produce `stopped_with_failure`.

Missing credentials, irreconcilable requirements, required product decisions, or unavailable external dependencies produce `requires_human_intervention` with a concrete reason.
