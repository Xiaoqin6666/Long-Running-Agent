# Long-Running Agent

This repository is for a training-free long-running coding agent system.这个存储库用于一个不需要训练的长时间运行的编码代理系统。

The current design focuses on a minimal but research-friendly agent harness:目前的设计重点是一个最小但研究友好的代理线束：

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

Run the deterministic offline loop:运行确定性脱机循环：

```powershell
python -m agent.main "Smoke test the minimal long-running agent" --max-steps 5
```

Summarize a trace:   总结一个踪迹：

```powershell
python eval\metrics.py state\benchmarks\issue_tracker\traces\<trace-file>.jsonl --tasks state\benchmarks\issue_tracker\runtime_tasks.json
```

Run behavior tests:   运行行为测试：

```powershell
python -m unittest discover -s tests
```

Run the optional manual evaluator after the autonomous run has ended:在自动运行结束后运行可选的手动评估器：

```powershell
python eval\manual_evaluators\issue_tracker\evaluate.py
```

The Agent Harness never invokes this script. Its result does not gate `finish` and cannot create repair tasks.

The offline provider is intentionally simple. It exercises the harness loop without requiring an API key, so state management, tool execution, verifier gating, and trace writing can be tested first.离线提供程序故意很简单。它在不需要API密钥的情况下执行线束循环，因此可以首先测试状态管理、工具执行、验证器门控和跟踪写入。

## API Provider

The real model provider uses an OpenAI-compatible chat completions API. Configure it with environment variables:真正的模型提供者使用与openai兼容的聊天完成API。用环境变量配置它：

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
