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

### Agent 项目运行方式

方式 1：进入交互式 agent。

```powershell
python -m agent.main --chat --benchmark MAPA --provider openai-compatible
```

`--benchmark` 设置测评任务的名字。进入 agent 之后，使用 `/agent` 模式开启新的项目，并指定项目规格文件路径；`/adjust` 模式用来对当前项目提出修改建议；`/resume` 模式在新的会话中继续当前项目。

方式 2：直接指定项目规格文件启动。

```powershell
python -m agent.main --project-spec eval\benchmarks\MAPA\task.md --provider openai-compatible --benchmark MAPA
```

`--project-spec eval\benchmarks\MAPA\task.md` 设置项目规格文件路径。`--benchmark` 设置测评任务的名字。

The memory index at `state/memory.md` is always loaded with a 200-line / 25KB cap. Full memory files are loaded only after synchronous retrieval. Set `LONG_AGENT_MEMORY_MODEL` such as `deepseek-flash` to use a cheap selector model; if it is unset or fails, the harness falls back to local keyword matching.

Summarize a trace:

```powershell
python eval\metrics.py state\benchmarks\issue_tracker\traces\<trace-file>.jsonl --tasks state\benchmarks\issue_tracker\runtime_tasks.json
```

Token usage is recorded separately from the handoff budget estimate. Each trace event includes `token_usage` for that agent step, `session_token_usage` for the active session totals, and `total_token_usage` across all sessions. The durable `current_task.json` also stores `token_usage.totals`, `token_usage.sessions`, and per-turn `token_usage.turns`; `/status` shows the accumulated LLM input and output totals.

Inspect full model context for each action:

Every agent step writes the exact system message plus user context sent to the decision model under `state\debug_contexts\<trace-name>\step_0001.md`. Trace events include a `context_ref` field pointing at that file. The model can also call `debug_context` with `target="current"` or a step number; set `args.include_content=true` to return the snapshot content in the observation data.

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
$env:LONG_AGENT_MEMORY_MODEL="deepseek-flash"  # optional selector model
$env:LONG_AGENT_TOKEN_PRICES_JSON='{"gpt-4.1-mini":{"input_per_1m":0.0,"output_per_1m":0.0,"currency":"USD"}}'
```

If the API response includes cost or billing fields, that provider-returned cost is recorded first with `price_source="api"`. Otherwise, set `LONG_AGENT_TOKEN_PRICES_JSON` to the current price you want to use, expressed per 1M tokens; replace the `0.0` example values before relying on cost output. You can also put the same JSON object in a file and set `LONG_AGENT_TOKEN_PRICES_FILE=path\to\prices.json`. When pricing is available for the active model, each trace step's `token_usage.cost` includes input, output, and total cost. Session cost aggregates live under `token_usage.sessions[session_id].costs_by_currency`; all-session aggregates live under `token_usage.totals.costs_by_currency`. If neither API cost nor a configured model price is available, the turn is counted under `unpriced_turn_count`.

When a session reaches the handoff threshold, the harness writes `handoff.md`, resets the per-session budget flags, starts a fresh trace, and resumes from `current_task.json` without requiring a manual `--resume` restart.

Each run writes a diagnostic log under `state/benchmarks/<benchmark_id>/logs/`. The terminal prints the exact path before provider initialization, so startup failures are recorded too. Use `--log-file path\to\run.log` to override it.

For DeepSeek or Qwen OpenAI-compatible endpoints, keep the same command and change `LONG_AGENT_BASE_URL` plus `LONG_AGENT_MODEL`.

Inspection or recommendation tasks may finish with an `answer` action instead of `finish`. Coding tasks still rely on verifier-gated `finish`.
