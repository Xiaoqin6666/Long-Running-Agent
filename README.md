# Long-Running Agent

This repository is for a training-free long-running coding agent system.

The current design focuses on a minimal but research-friendly agent harness.

- explicit task state management;
- bounded context construction and handoff;
- independent self-verification;
- filesystem-backed Skill and Memory;


## Quick Start

generated application code and public tests belong under `eval/benchmarks/<benchmark_id>/workspace/`.


### Runtime Environment
Python 3.12

### Environment Variable Configuration
```powershell
$env:LONG_AGENT_API_KEY="sk-"          
$env:LONG_AGENT_MODEL="deepseek-v4-pro"   
$env:LONG_AGENT_BASE_URL="https://api.deepseek.com/v1"
```

### Agent Project Execution Modes
Mode 1: Launch the interactive agent.

```powershell
python -m agent.main --chat --benchmark MAPA --provider openai-compatible  --auto-resume
```

The --benchmark argument configures the name of the benchmark task. After entering the agent interface, use the /agent command to initialize a new project with the file path of the project specification designated. The /adjust command submits modification proposals for the ongoing project, while the /resume command resumes the existing project within a brand-new session.

Mode 2: Start the agent by directly pointing to the project specification file.

```powershell
python -m agent.main --project-spec eval\benchmarks\MAPA\task.md --benchmark MAPA --provider openai-compatible  --auto-resume
```

`--project-spec eval\benchmarks\MAPA\task.md` sets the file path of the project specification designated. `--benchmark` configures the name of the benchmark task.

Integration Contract validation is disabled by default. Add `--integration-contract` when you want FINAL_ACCEPTANCE to generate and enforce `integration_contract.json` and `integration_results.json`.

