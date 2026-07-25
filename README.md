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

在交互模式中使用 `/skill` 可以直接添加受信任的用户 Skill。Skill 文件必须在 frontmatter 中声明 `name`；`description` 等其他 frontmatter 字段和正文固定章节都是可选的，缺少 `description` 时从正文第一段推导。正文仍是自由 Markdown。Agent 自主创建的 Skill 还必须引用真实的 Verifier 或 trace 证据。

项目内置的默认 Skill 位于 `default_skills/<name>/SKILL.md`。当前随仓库提供 Anthropic
`anthropics/skills` 的 17 个 Skill，并保留各目录中的 `scripts/`、`references/`、`assets/`
和许可证文件。运行期间新增的 Skill 保存在 `state/skills/`；同名时运行状态中的 Skill
覆盖默认 Skill。来源版本和授权说明见 `default_skills/UPSTREAM.md`。

Typed Memory bodies live in `state/memories/*.md`; `state/memory.md` is a derived index rebuilt at startup and after writes. At Worker-session start, task transitions, and successful `save_memory` events, the harness sends the complete valid typed-Memory corpus to the main `LONG_AGENT_MODEL` in one isolated QA request and injects only its grounded answer and validated exact citations into the Main Agent context. The Main Agent can also call the read-only `recall_memory` action with a natural-language question. The full corpus is never silently truncated or replaced with keyword search: an oversized or invalid corpus produces an explicit error. Because every typed Memory body is sent to the configured main-model provider, do not store credentials or secrets in Memory.

Summarize a trace:

```powershell
python eval\metrics.py state\benchmarks\issue_tracker\traces\<trace-file>.jsonl --tasks state\benchmarks\issue_tracker\runtime_tasks.json
```

Token usage is recorded separately from the handoff budget estimate. Each trace event includes `token_usage` for that agent step, `session_token_usage` for the active session totals, and `total_token_usage` across all sessions. The durable `current_task.json` also stores `token_usage.totals`, `token_usage.sessions`, and per-turn `token_usage.turns`; `/status` shows the accumulated LLM input and output totals.

Inspect model context and the exact provider transcript:

Diagnostic context snapshots are written under `state\debug_contexts\<trace-name>\step_0001.md`. The exact append-only Chat Completions request/response/tool transcript is written under `state\provider_sessions\<trace-name>.jsonl`; it includes full provider `reasoning_content`, is permissioned for the current user only, and must be treated as sensitive. Normal trace events contain only the provider transcript path, SHA-256, byte length, and current `tool_call_id`. The provider transcript is reset at Worker handoff and its reasoning is not copied into handoff, Memory, Skills, or normal CLI output.

Run behavior tests:

```powershell
python -m unittest discover -s tests
```

To run the optional live DeepSeek protocol smoke test, configure the provider variables and set `LONG_AGENT_RUN_LIVE_DEEPSEEK_TEST=1` before running `python -m unittest tests.test_deepseek_protocol.DeepSeekProtocolTests.test_optional_live_deepseek_smoke`.

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
$env:LONG_AGENT_THINKING="auto"
$env:LONG_AGENT_REASONING_EFFORT="high"
$env:LONG_AGENT_PROVIDER_MAX_ATTEMPTS="3"
$env:LONG_AGENT_CONTEXT_WINDOW_TOKENS="128000"
$env:LONG_AGENT_TOKEN_PRICES_JSON='{"gpt-4.1-mini":{"input_per_1m":0.0,"output_per_1m":0.0,"currency":"USD"}}'
```

`openai-compatible` requires native Chat Completions function calling. CLI history is sent as ordinary `user` and `assistant` messages; the one-time session context contains rules, task state, handoff, Skill catalog, loaded Skills, and relevant Memory. Most compatible providers receive the forced `submit_action` wrapper. DeepSeek thinking rejects `tool_choice`, so that path receives the existing actions (`read`, `write`, `bash`, `verify`, and so on) as direct native functions; their arguments are normalized back into the same validated action structure before execution. The harness appends the matching observation plus incremental task state as a `tool` result, and content-only action JSON responses are rejected. DeepSeek endpoints/models enable thinking automatically, omit sampling temperature, round-trip `reasoning_content` exactly across tool calls, and reject any function not present in the request. Because DeepSeek's standard tool loop may return final content while the harness still requires another action, the transport can append a bounded content-only correction turn. DeepSeek can also emit parallel tool calls; the harness closes every call ID with an explicit non-execution result and asks for one action at a time, preserving its per-action state/verification boundary. Neither path fabricates a successful tool execution, and usage from all correction requests is aggregated. Set `LONG_AGENT_THINKING=disabled` only when thinking must be turned off.

If the API response includes cost or billing fields, that provider-returned cost is recorded first with `price_source="api"`. Cache hit/miss/write and reasoning-token details are retained when the provider reports them, and the original provider `usage` object is preserved on the turn as `provider_usage`. Otherwise, set `LONG_AGENT_TOKEN_PRICES_JSON` to the current price you want to use, expressed per 1M tokens; replace the `0.0` example values before relying on cost output. You can also put the same JSON object in a file and set `LONG_AGENT_TOKEN_PRICES_FILE=path\to\prices.json`. When pricing is available for the active model, each trace step's `token_usage.cost` includes input, output, and total cost. Session cost aggregates live under `token_usage.sessions[session_id].costs_by_currency`; all-session aggregates live under `token_usage.totals.costs_by_currency`. If neither API cost nor a configured model price is available, the turn is counted under `unpriced_turn_count`.

When a session reaches the handoff threshold, the harness writes `handoff.md`, resets the model transcript and per-session budget flags, starts a fresh trace, and rebuilds context from durable state, handoff, loaded Skills, and relevant Memory without requiring a manual `--resume` restart.

Each run writes a diagnostic log under `state/benchmarks/<benchmark_id>/logs/`. The terminal prints the exact path before provider initialization, so startup failures are recorded too. Use `--log-file path\to\run.log` to override it.

For DeepSeek or another OpenAI-compatible endpoint with native function calling, keep the same command and change `LONG_AGENT_BASE_URL` plus `LONG_AGENT_MODEL`.

Inspection or recommendation tasks may finish with an `answer` action instead of `finish`. Coding tasks still rely on verifier-gated `finish`.
