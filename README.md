# Long-Running Agent

这是一个无需训练、面向长时间编码任务的 Agent 系统。

当前设计以一个精简且便于研究的 Agent Harness 为核心，主要包括：

- 显式的任务状态规划和状态追踪；
- 独立的自验证机制；
- 基于文件系统的 Skill 与 Memory；


## 快速开始

### 环境要求：

Python>=3.12

### 环境变量配置：

```powershell
$env:LONG_AGENT_API_KEY="sk-"          
$env:LONG_AGENT_MODEL="deepseek-v4-pro"   
$env:LONG_AGENT_BASE_URL="https://api.deepseek.com/v1"
```
### 运行 Agent 项目
方式1：进入交互式 Agent

```powershell
python -m agent.main --chat  --benchmark MAPA  --provider openai-compatible --auto-resume
```

`--benchmark` 设置测评任务的名字
`--auto-resume` 自动切换会话
进入agent之后，默认进入/agent 模式开启新的项目，并指定项目规格文件路径，/adjust 模式用来对当前提出修改建议, /resume 模式在新的会话中继续当前项目。

方式2：直接指定项目规格文件启动

```powershell
python -m agent.main --project-spec eval\benchmarks\MAPA\task.md --provider openai-compatible --auto-resume --benchmark MAPA
```

`--project-spec eval\benchmarks\MAPA\task.md `设置项目规格文件路径
`--benchmark `设置测评任务的名字
`--auto-resume` 自动切换会话



