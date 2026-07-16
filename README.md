# Long-Running Agent

This repository is for a training-free long-running coding agent system.

The current design focuses on a minimal but research-friendly agent harness.

- explicit task state management;
- bounded context construction and handoff;
- independent self-verification;
- filesystem-backed Skill and Memory;
- trace-driven experiments and ablations.

See [docs/problem3_agent_framework.md](docs/problem3_agent_framework.md) for the full framework design.
See [docs/system_prompts.md](docs/system_prompts.md) for role-specific Main Agent, Planner, and Verifier system prompts.
See [docs/evaluation_runbook.md](docs/evaluation_runbook.md) for the long-running evaluation task.

## Planned Milestones

1. Build a CLI agent loop.
2. Add minimal tools: list_files, search, read, edit, bash, git, verify.
3. Store task state, memory, skills, and traces on disk.
4. Add context compaction and handoff.
5. Run long-coding-task experiments and ablations.

## Quick Start

The tracked repository-root `init.sh` bootstraps the Long-Running Agent harness. An autonomous benchmark INIT generates a separate run-local script at `state/benchmarks/<benchmark_id>/init.sh`; generated application code and public tests belong under `eval/benchmarks/<benchmark_id>/workspace/`.

Run the deterministic offline loop:

```powershell
python -m agent.main "Smoke test the minimal long-running agent" --max-steps 5
```

Start an interactive terminal conversation:

```powershell
python -m agent.main --chat --provider openai-compatible --max-steps 8
```

On Windows, `--chat` opens the conversation in a dedicated terminal window. Use `--chat-inline` to keep the UI in the current terminal, which is useful for scripts and debugging.

You can also provide the first message directly:

```powershell
python -m agent.main "Inspect the current failure" --chat --provider openai-compatible
```

Use `/ask <question>` for read-only questions, `/do <task>` for project work, and `/skill` to directly add a trusted user-authored Skill. Agent-authored Skills still require verifier or trace evidence. The chat also supports `/help`, `/status`, `/history`, `/resume`, `/new`, and `/exit`. Conversation records are appended to `state/chat_history.jsonl`, or to the selected benchmark state directory.

Summarize a trace:

```powershell
python eval\metrics.py state\benchmarks\issue_tracker\traces\<trace-file>.jsonl --tasks state\benchmarks\issue_tracker\runtime_tasks.json
```

Run behavior tests:

```powershell
python -m unittest discover -s tests
```

Run the optional manual evaluator after the autonomous run has ended:在自动运行结束后运行可选的手动评估器：

```powershell
python eval\manual_evaluators\issue_tracker\evaluate.py
```

The Agent Harness never invokes this script. Its result does not gate `finish` and cannot create repair tasks.

The offline provider is intentionally simple. It exercises the harness loop without requiring an API key, so state management, tool execution, verifier gating, and trace writing can be tested first.

## API Provider

The real model provider uses an OpenAI-compatible chat completions API. Configure it with environment variables:

```powershell
$env:LONG_AGENT_API_KEY="your_api_key"
$env:LONG_AGENT_BASE_URL="https://api.openai.com/v1"
$env:LONG_AGENT_MODEL="gpt-4.1-mini"
```

Run with:

```powershell
python -m agent.main "Inspect this repo and suggest the next implementation step" --provider openai-compatible --max-steps 3
```

For long benchmark runs, let the harness automatically continue from handoff files:

```powershell
python -m agent.main --benchmark todo_counter --project-spec eval\benchmarks\todo_counter\project_spec.md --provider openai-compatible --max-steps 12 --auto-resume --max-sessions 5
```

When a session reaches the handoff threshold, the harness writes `handoff.md`, resets the per-session budget flags, starts a fresh trace, and resumes from `current_task.json` without requiring a manual `--resume` restart.

Each run writes a diagnostic log under `state/benchmarks/<benchmark_id>/logs/`. The terminal prints the exact path before provider initialization, so startup failures are recorded too. Use `--log-file path\to\run.log` to override it.

For DeepSeek or Qwen OpenAI-compatible endpoints, keep the same command and change `LONG_AGENT_BASE_URL` plus `LONG_AGENT_MODEL`.

Inspection or recommendation tasks may finish with an `answer` action instead of `finish`. Coding tasks still rely on verifier-gated `finish`.
