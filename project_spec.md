# Project Spec

## Goal

Build a training-free long-running coding agent harness for Zhejiang University REAL Lab Problem 3. The system should run from the command line, use an LLM for decisions, interact with the local environment through minimal tools, persist task state across sessions, and produce verifiable progress traces.

## Technical Constraints

- No model training is allowed.
- The core orchestration logic must be implemented in this repository rather than delegated to an existing agent framework.
- The initial implementation should use Python standard library where possible.
- The default local mode must run without API keys.
- The real model mode should use an OpenAI-compatible chat completions API.
- Runtime environment for local commands is Windows PowerShell, though portable commands are preferred.
- Trace, memory, skill, handoff, task, and contract artifacts should be file-backed.

## Architecture Roles

### Initializer / Planner

Runs once at project start. It converts a vague user request into durable project artifacts:

- `project_spec.md`: project goals, constraints, architecture roles, and global completion criteria.
- `tasks.json`: structured tasks, dependencies, priorities, and acceptance criteria.
- `init.sh`: repeatable setup and validation entrypoint.
- Initial Git commit: stable baseline after project initialization.

After initialization, the planner may update task state, but it should not repeatedly recreate the project baseline.

### Main Agent

The Main Agent is the coding agent. It works on exactly one active task per loop. It may inspect files, execute commands, write code, request verification, and finish only when the current task's acceptance criteria are satisfied.

Before generating or modifying code for a task, it must establish an acceptance contract with the verifier.

### Verifier

The Verifier independently checks claimed progress. It should not verify only what the Main Agent happened to implement. Instead, it validates against the pre-agreed acceptance contract, global project criteria, tests, and trace evidence.

## Acceptance Contract

Before the Main Agent writes code for a task, it must produce a contract containing:

- task id and scope;
- user-visible behavior to satisfy;
- files or modules likely to be touched;
- checks that must pass;
- forbidden shortcuts or out-of-scope work;
- verifier evidence required for completion.

The harness rejects code-writing actions when no contract exists for the active task.

## Global Completion Criteria

- CLI agent loop runs in offline mode.
- OpenAI-compatible provider can drive real model decisions.
- State, memory, skills, handoff, traces, and acceptance contracts are persisted on disk.
- Main Agent handles one active task at a time.
- Coding actions require a verifier-aligned acceptance contract.
- Verifier runs deterministic checks, including syntax and behavior tests.
- Experiments can be summarized from trace files.
- Documentation explains architecture, prompts, setup, and evaluation.

