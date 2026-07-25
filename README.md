# Long-Running Agent

这是一个无需训练、面向长时间编码任务的 Agent 系统。

当前设计以一个精简且便于研究的 Agent Harness 为核心，主要包括：

- 显式的任务状态管理；
- 有界的上下文构建与会话交接（handoff）；
- 独立的自验证机制；
- 基于文件系统的 Skill 与 Memory；
- 由 trace 驱动的实验与消融分析。

完整框架设计见 [docs/problem3_agent_framework.md](docs/problem3_agent_framework.md)。
Main Agent、Planner 和 Verifier 的角色提示词见
[docs/system_prompts.md](docs/system_prompts.md)。
长时间运行测评任务的说明见
[docs/evaluation_runbook.md](docs/evaluation_runbook.md)。

## 规划里程碑

1. 构建 CLI Agent 循环。
2. 加入最小工具集：`list_files`、`search`、`read`、`edit`、`bash`、`git`、`verify`。
3. 将任务状态、Memory、Skill 和 trace 保存到磁盘。
4. 加入上下文压缩与 handoff。
5. 运行长时间编码任务实验和消融实验。

## 快速开始

仓库根目录中受版本控制的 `init.sh` 用于初始化 Long-Running Agent Harness。
自主测评的 INIT 阶段会另外生成本次运行专用的
`state/benchmarks/<benchmark_id>/init.sh`；生成的应用代码和公开测试应放在
`eval/benchmarks/<benchmark_id>/workspace/` 下。

### 运行 Agent 项目

方式一：进入交互式 Agent。

```powershell
python -m agent.main --chat --benchmark MAPA --provider openai-compatible
```

`--benchmark` 用于设置测评任务名称。进入 Agent 后，使用 `/agent` 模式开始新项目并指定项目规格文件路径；使用 `/adjust` 模式对当前项目提出修改要求；`/resume` 仅用于手动继续其他原因导致的未完成运行。达到 handoff 上下文阈值时，系统默认会自动创建新会话并继续执行，无需用户输入 `/resume`。

方式二：直接指定项目规格文件启动。

```powershell
python -m agent.main --project-spec eval\benchmarks\MAPA\task.md --provider openai-compatible --benchmark MAPA
```

`--project-spec eval\benchmarks\MAPA\task.md` 用于设置项目规格文件路径，
`--benchmark` 用于设置测评任务名称。

自动 handoff 默认开启且不限制会话数量。可用 `--max-sessions N` 限制一次运行最多使用的会话数，或用 `--no-auto-resume` 在写入 handoff 后停止。

### Skill

在交互模式中使用 `/skill` 可以直接添加受信任的用户 Skill。Skill 文件必须在
frontmatter 中声明 `name`；`description` 等其他 frontmatter 字段和正文固定章节均为
可选项。缺少 `description` 时，系统会从正文第一段推导。正文仍可使用自由格式的
Markdown。Agent 自主创建的 Skill 还必须引用真实的 Verifier 或 trace 证据。

项目内置的默认 Skill 位于 `default_skills/<name>/SKILL.md`。仓库当前包含来自
Anthropic `anthropics/skills` 的 17 个 Skill，并保留各目录中的 `scripts/`、
`references/`、`assets/` 和许可证文件。运行期间新增的 Skill 保存在
`state/skills/`；同名时，运行状态中的 Skill 会覆盖默认 Skill。来源版本和授权说明见
`default_skills/UPSTREAM.md`。

### Memory

类型化 Memory 的正文保存在 `state/memories/*.md` 中；`state/memory.md` 是一个派生
索引，会在启动和写入后重新构建。

在 Worker 会话开始、任务切换以及 `save_memory` 成功后，Harness 会把全部有效的类型化
Memory 一次性发送给主模型 `LONG_AGENT_MODEL`，进行隔离的问答请求；随后只把有依据的
回答和经过校验的精确引用注入 Main Agent 上下文。Main Agent 也可以使用只读的
`recall_memory` action，以自然语言提问。完整 Memory 集合不会被静默截断，也不会退化为
关键词搜索；集合过大或内容无效时，系统会明确报错。由于所有类型化 Memory 正文都会被
发送给已配置的主模型服务商，请勿在 Memory 中保存凭据或其他秘密信息。

### Trace 与模型上下文

汇总 trace：

```powershell
python eval\metrics.py state\benchmarks\issue_tracker\traces\<trace-file>.jsonl --tasks state\benchmarks\issue_tracker\runtime_tasks.json
```

Token 用量和 handoff 的预算估算分别记录。每个 trace 事件均包含当前 Agent 步骤的
`token_usage`、当前会话累计值 `session_token_usage`，以及跨会话累计值
`total_token_usage`。持久化的 `current_task.json` 还会保存
`token_usage.totals`、`token_usage.sessions` 和逐轮的 `token_usage.turns`；
`/status` 会显示累计的 LLM 输入与输出 Token。

如需检查模型上下文和发送给服务商的精确对话记录：

- 诊断上下文快照保存在
  `state\debug_contexts\<trace-name>\step_0001.md`；
- 只追加的 Chat Completions 请求、响应与工具调用记录保存在
  `state\provider_sessions\<trace-name>.jsonl`。

Provider transcript 包含完整的 `reasoning_content`，文件权限仅对当前用户开放，应按
敏感数据处理。普通 trace 事件只记录 Provider transcript 的路径、SHA-256、字节长度和
当前 `tool_call_id`。Worker handoff 时会重置 Provider transcript，其中的推理内容不会
被复制到 handoff、Memory、Skill 或普通 CLI 输出中。

## 测试

运行 Harness 行为测试：

```powershell
python -m unittest discover -s tests
```

如需运行可选的 DeepSeek 协议在线冒烟测试，请先配置 Provider 环境变量并设置
`LONG_AGENT_RUN_LIVE_DEEPSEEK_TEST=1`，然后执行：

```powershell
python -m unittest tests.test_deepseek_protocol.DeepSeekProtocolTests.test_optional_live_deepseek_smoke
```

自主运行结束后，可以执行可选的手动评估器：

```powershell
python eval\manual_evaluators\issue_tracker\evaluate.py
```

Agent Harness 不会调用此脚本。其结果不会影响 `finish` 判定，也不能创建修复任务。

离线 Provider 的实现刻意保持简单。它无需 API Key 即可运行 Harness 循环，便于先测试
状态管理、工具执行、Verifier 门禁和 trace 写入。

## API 服务配置

真实模型 Provider 使用兼容 OpenAI 的 Chat Completions API。通过以下环境变量配置：

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

`openai-compatible` 要求原生支持 Chat Completions function calling。CLI 历史会以普通
`user` 和 `assistant` 消息发送；一次性会话上下文则包含规则、任务状态、handoff、
Skill 目录、已加载的 Skill 以及相关 Memory。

大多数兼容 Provider 会收到强制使用的 `submit_action` 包装器。DeepSeek thinking 不接受
`tool_choice`，因此该路径会把现有 action（如 `read`、`write`、`bash`、`verify`）直接
作为原生 function 提供；function 参数在执行前仍会被归一化为同一套经过校验的 action
结构。Harness 会把匹配的 observation 和增量任务状态追加为 `tool` 结果，并拒绝只在
普通文本中返回 action JSON 的响应。

DeepSeek endpoint 或模型会自动开启 thinking、省略采样温度，并在工具调用之间原样回传
`reasoning_content`；模型调用请求中不存在的 function 会被拒绝。由于 DeepSeek 的标准
工具循环可能在 Harness 仍要求下一步 action 时返回最终文本，传输层可以追加一次有上限
的纯文本纠正轮次。DeepSeek 也可能并行发出多个工具调用；Harness 会为每个调用 ID 返回
明确的“未执行”结果，并要求模型每次只提交一个 action，从而保留逐 action 的状态与验证
边界。两条路径都不会伪造成功的工具执行，所有纠正请求产生的用量都会计入总量。仅当确实
需要关闭 thinking 时，才设置 `LONG_AGENT_THINKING=disabled`。

如果 API 响应包含费用或账单字段，系统会优先记录 Provider 返回的费用，并设置
`price_source="api"`。Provider 返回的缓存命中、未命中、写入和推理 Token 明细会被保留，
原始 `usage` 对象也会保存在对应轮次中。

否则，请通过 `LONG_AGENT_TOKEN_PRICES_JSON` 配置当前使用的价格，单位为每 100 万
Token；在依赖费用输出前，请替换示例中的 `0.0`。也可以把同一 JSON 对象保存到文件，并
设置 `LONG_AGENT_TOKEN_PRICES_FILE=path\to\prices.json`。当活动模型存在定价信息时，
每个 trace 步骤的 `token_usage.cost` 会包含输入、输出和总费用。会话费用汇总保存在
`token_usage.sessions[session_id].costs_by_currency`，跨会话汇总保存在
`token_usage.totals.costs_by_currency`。如果 API 没有返回费用且模型价格也未配置，该
轮次会计入 `unpriced_turn_count`。

当会话达到 handoff 阈值时，Harness 会写入 `handoff.md`，重置模型 transcript 和当前
会话的预算标志，启动新的 trace，并根据持久化状态、handoff、已加载 Skill 和相关 Memory
重新构建上下文。默认情况下，这一过程会自动完成，不需要用户输入 `/resume` 或重新使用
`--resume` 启动。

每次运行都会在 `state/benchmarks/<benchmark_id>/logs/` 下写入诊断日志。终端会在
Provider 初始化前打印准确路径，因此启动失败也会被记录。可用
`--log-file path\to\run.log` 覆盖默认位置。

对于 DeepSeek 或其他原生支持 function calling 的 OpenAI 兼容 endpoint，保持启动命令
不变，只需修改 `LONG_AGENT_BASE_URL` 和 `LONG_AGENT_MODEL`。

检查或建议类任务可以通过 `answer` action 结束；编码任务仍必须通过 Verifier 门禁后才能
执行 `finish`。
