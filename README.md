# Long-Running Agent

This repository is for Zhejiang University REAL Lab assessment Problem 3: a training-free long-running coding agent system.

The current design focuses on a minimal but research-friendly agent harness:

- explicit task state management;
- bounded context construction and handoff;
- independent self-verification;
- filesystem-backed Skill and Memory;
- trace-driven experiments and ablations.

See [docs/problem3_agent_framework.md](docs/problem3_agent_framework.md) for the full framework design.
See [docs/system_prompts.md](docs/system_prompts.md) for role-specific Main Agent, Planner, and Verifier system prompts.

## Planned Milestones

1. Build a CLI agent loop.
2. Add minimal tools: list_files, search, read, edit, bash, git, verify.
3. Store task state, memory, skills, and traces on disk.
4. Add context compaction and handoff.
5. Run long-coding-task experiments and ablations.

## Quick Start

Run the deterministic offline loop:

```powershell
python -m agent.main "Smoke test the minimal long-running agent" --max-steps 5
```

Summarize a trace:

```powershell
python eval\metrics.py state\traces\<trace-file>.jsonl --tasks tasks.json
```

Run behavior tests:

```powershell
python -m unittest discover -s tests
```

Run hidden acceptance:

```powershell
python eval\hidden_acceptance.py
```

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

For DeepSeek or Qwen OpenAI-compatible endpoints, keep the same command and change `LONG_AGENT_BASE_URL` plus `LONG_AGENT_MODEL`.

Inspection or recommendation tasks may finish with an `answer` action instead of `finish`. Coding tasks still rely on verifier-gated `finish`.
