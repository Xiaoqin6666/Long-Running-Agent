from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from agent.chat import ChatConfig, ChatMessage, InteractiveCLI, launch_chat_window
from agent.context import ContextBuilder
from agent.loop import AgentLoop, RunResult
from agent.main import build_parser, resolve_optional_task
from agent.memory import parse_memory
from agent.planner import TaskState, create_initial_state
from agent.skills import parse_skill
from agent.tools import ToolResult


class ChatCLITests(unittest.TestCase):
    def test_parser_accepts_chat_without_task(self) -> None:
        args = build_parser().parse_args(["--chat"])
        self.assertTrue(args.chat)
        self.assertIsNone(resolve_optional_task(args))

    def test_parser_accepts_inline_chat(self) -> None:
        args = build_parser().parse_args(["--chat", "--chat-inline"])
        self.assertTrue(args.chat_inline)

    def test_chat_project_spec_is_not_initial_message(self) -> None:
        args = build_parser().parse_args(["--chat", "--project-spec", "eval/benchmarks/budget_management/task.md"])
        self.assertIsNone(resolve_optional_task(args))

    def test_chat_command_runs_read_only_turn_without_starting_agent_flow(self) -> None:
        outputs: list[str] = []
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            output_fn=outputs.append,
            use_color=False,
        )
        with patch.object(cli, "_run_agent_project_flow") as run_agent:
            with patch.object(cli, "_run_turn") as run_turn:
                cli._handle_command("/chat show status")
        run_agent.assert_not_called()
        run_turn.assert_called_once_with("show status", interaction_mode="question")
        self.assertEqual(cli.active_mode, "chat")

    def test_agent_command_starts_project_flow(self) -> None:
        outputs: list[str] = []
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            output_fn=outputs.append,
            use_color=False,
        )
        with patch.object(cli, "_run_agent_project_flow") as run_agent:
            cli._handle_command("/agent build a budget app")
        run_agent.assert_called_once_with("build a budget app")
        self.assertEqual(cli.active_mode, "agent")

    def test_agent_mode_prompt_requests_direct_requirements(self) -> None:
        outputs: list[str] = []
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            output_fn=outputs.append,
            use_color=False,
        )
        cli._handle_command("/agent")
        self.assertIn("Use Shift+Enter/new lines as needed", outputs[-1])
        self.assertIn("/send", outputs[-1])
        self.assertNotIn("/ask", outputs[-1])
        self.assertIn("project spec file", outputs[-1])

    def test_plain_message_requires_selected_mode(self) -> None:
        outputs: list[str] = []
        inputs = iter(["show status", "/exit"])
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            input_fn=lambda prompt: next(inputs),
            output_fn=outputs.append,
            use_color=False,
        )
        with patch.object(cli, "_run_agent_project_flow") as run_agent:
            self.assertEqual(cli.run(), 0)
        run_agent.assert_not_called()
        self.assertIn("Choose /chat for conversation or /agent for autonomous project work.", outputs)

    def test_plain_message_uses_active_mode(self) -> None:
        outputs: list[str] = []
        inputs = iter(["/chat", "show status", "/agent", "build a budget app", "/send", "/exit"])
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            input_fn=lambda prompt: next(inputs),
            output_fn=outputs.append,
            use_color=False,
        )
        with patch.object(cli, "_run_agent_project_flow") as run_agent:
            with patch.object(cli, "_run_turn") as run_turn:
                self.assertEqual(cli.run(), 0)
        run_agent.assert_called_once_with("build a budget app")
        run_turn.assert_called_once_with("show status", interaction_mode="question")

    def test_agent_mode_collects_multiline_requirements_until_send(self) -> None:
        outputs: list[str] = []
        inputs = iter(["/agent", "line one", "line two", "/send", "/exit"])
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            input_fn=lambda prompt: next(inputs),
            output_fn=outputs.append,
            use_color=False,
        )
        with patch.object(cli, "_run_agent_project_flow") as run_agent:
            self.assertEqual(cli.run(), 0)
        run_agent.assert_called_once_with("line one\nline two")
        self.assertEqual(cli.agent_draft, [])
        self.assertTrue(any("Added to agent requirements draft" in output for output in outputs))

    def test_agent_mode_clear_discards_collected_requirements(self) -> None:
        outputs: list[str] = []
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            output_fn=outputs.append,
            use_color=False,
        )
        cli._handle_command("/agent")
        cli._handle_plain_message("line one")
        cli._handle_command("/clear")
        self.assertEqual(cli.agent_draft, [])
        self.assertIn("Cleared collected agent requirements.", outputs)

    def test_chat_and_agent_modes_switch_directly(self) -> None:
        outputs: list[str] = []
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            output_fn=outputs.append,
            use_color=False,
        )
        cli._handle_command("/chat")
        self.assertEqual(cli.active_mode, "chat")
        cli._handle_command("/agent")
        self.assertEqual(cli.active_mode, "agent")
        cli._handle_command("/chat")
        self.assertEqual(cli.active_mode, "chat")
        cli._handle_command("/mode")
        self.assertIn("Current mode: chat.", outputs)

    def test_legacy_ask_and_do_aliases_select_new_modes(self) -> None:
        outputs: list[str] = []
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            output_fn=outputs.append,
            use_color=False,
        )
        with patch.object(cli, "_run_agent_project_flow") as run_agent:
            with patch.object(cli, "_run_turn") as run_turn:
                cli._handle_command("/ask chat only")
                cli._handle_command("/do start work")
        run_turn.assert_called_once_with("chat only", interaction_mode="question")
        run_agent.assert_called_once_with("start work")

    def test_agent_project_flow_writes_project_spec_and_starts_initializer(self) -> None:
        outputs: list[str] = []
        root = Path.cwd() / ".tmp_tests" / f"chat-agent-{uuid4().hex}"
        root.mkdir(parents=True)
        cli = InteractiveCLI(
            ChatConfig(root=root, provider="offline", max_steps=1, benchmark_id="sample"),
            output_fn=outputs.append,
            use_color=False,
        )
        run_result = RunResult(
            completed=True,
            steps=1,
            trace_path=root / "state" / "benchmarks" / "sample" / "traces" / "run.jsonl",
            state_path=root / "state" / "benchmarks" / "sample" / "current_task.json",
            message="done",
        )
        try:
            with patch("agent.chat.build_project_spec", return_value="# Built Spec\n") as build_spec:
                with patch.object(AgentLoop, "run", return_value=run_result) as run_loop:
                    cli._handle_command("/agent build a budget app")
            build_spec.assert_called_once()
            run_loop.assert_called_once()
            spec_path = root / "state" / "benchmarks" / "sample" / "project_spec.md"
            self.assertEqual(spec_path.read_text(encoding="utf-8"), "# Built Spec\n")
            self.assertTrue(any("project spec saved" in output for output in outputs))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_agent_project_flow_reads_project_spec_path_from_message(self) -> None:
        outputs: list[str] = []
        root = Path.cwd() / ".tmp_tests" / f"chat-agent-spec-{uuid4().hex}"
        spec_path = root / "eval" / "benchmarks" / "sample" / "task.md"
        spec_path.parent.mkdir(parents=True)
        spec_path.write_text("Build sample app.", encoding="utf-8")
        cli = InteractiveCLI(
            ChatConfig(root=root, provider="offline", max_steps=1),
            output_fn=outputs.append,
            use_color=False,
        )
        run_result = RunResult(
            completed=True,
            steps=1,
            trace_path=root / "state" / "benchmarks" / "sample" / "traces" / "run.jsonl",
            state_path=root / "state" / "benchmarks" / "sample" / "current_task.json",
            message="done",
        )
        try:
            with patch("agent.chat.build_project_spec") as build_spec:
                with patch.object(AgentLoop, "run", return_value=run_result) as run_loop:
                    cli._handle_command(f"/agent requirements are described by this text: `{spec_path}`")
            build_spec.assert_not_called()
            run_loop.assert_called_once()
            self.assertEqual(cli.config.benchmark_id, "sample")
            self.assertEqual(cli.state_dir, root / "state" / "benchmarks" / "sample")
            materialized = root / "state" / "benchmarks" / "sample" / "project_spec.md"
            self.assertEqual(materialized.read_text(encoding="utf-8"), "Build sample app.")
            self.assertTrue(any("project spec source" in output for output in outputs))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_agent_project_flow_accepts_direct_project_spec_path(self) -> None:
        outputs: list[str] = []
        root = Path.cwd() / ".tmp_tests" / f"chat-agent-direct-spec-{uuid4().hex}"
        spec_path = root / "specs" / "project_spec.md"
        spec_path.parent.mkdir(parents=True)
        spec_path.write_text("# Direct Spec\n", encoding="utf-8")
        cli = InteractiveCLI(
            ChatConfig(root=root, provider="offline", max_steps=1),
            output_fn=outputs.append,
            use_color=False,
        )
        run_result = RunResult(
            completed=True,
            steps=1,
            trace_path=root / "state" / "traces" / "run.jsonl",
            state_path=root / "state" / "current_task.json",
            message="done",
        )
        try:
            with patch("agent.chat.build_project_spec") as build_spec:
                with patch.object(AgentLoop, "run", return_value=run_result):
                    cli._handle_command(f"/agent {spec_path}")
            build_spec.assert_not_called()
            self.assertEqual((root / "state" / "project_spec.md").read_text(encoding="utf-8"), "# Direct Spec\n")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_skill_command_collects_guided_fields(self) -> None:
        outputs: list[str] = []
        inputs = iter(["debug-failures", "Diagnose repeat failures", "Inspect the first traceback", "Use on failing tests"])
        root = Path.cwd() / ".tmp_tests" / f"chat-skill-{uuid4().hex}"
        root.mkdir(parents=True)
        cli = InteractiveCLI(
            ChatConfig(root=root, provider="offline", max_steps=1),
            input_fn=lambda prompt: next(inputs),
            output_fn=outputs.append,
            use_color=False,
        )
        try:
            with patch.object(cli, "_run_agent_project_flow") as run_agent:
                cli._handle_command("/skill")
            run_agent.assert_not_called()
            skill_path = root / "state" / "skills" / "debug-failures.md"
            skill = parse_skill(skill_path.read_text(encoding="utf-8"))
            self.assertEqual(skill.name, "debug-failures")
            self.assertEqual(skill.description, "Diagnose repeat failures")
            self.assertEqual(skill.instruction, "Inspect the first traceback")
            self.assertEqual(skill.examples, "Use on failing tests")
            self.assertTrue(any("Skill saved:" in output for output in outputs))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_skill_command_requires_fields_and_can_cancel(self) -> None:
        outputs: list[str] = []
        inputs = iter(["", "/cancel"])
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            input_fn=lambda prompt: next(inputs),
            output_fn=outputs.append,
            use_color=False,
        )
        with patch.object(cli, "_run_agent_project_flow") as run_agent:
            cli._handle_command("/skill")
        run_agent.assert_not_called()
        self.assertIn("Name is required.", outputs)
        self.assertIn("Skill setup cancelled.", outputs)

    def test_memory_command_collects_guided_fields(self) -> None:
        outputs: list[str] = []
        inputs = iter(
            [
                "go-react-profile",
                "User is experienced in Go and new to React",
                "user",
                "User has ten years of Go backend experience and is new to React.",
            ]
        )
        root = Path.cwd() / ".tmp_tests" / f"chat-memory-{uuid4().hex}"
        root.mkdir(parents=True)
        cli = InteractiveCLI(
            ChatConfig(root=root, provider="offline", max_steps=1),
            input_fn=lambda prompt: next(inputs),
            output_fn=outputs.append,
            use_color=False,
        )
        try:
            with patch.object(cli, "_run_agent_project_flow") as run_agent:
                cli._handle_command("/memory")
            run_agent.assert_not_called()
            memory_path = root / "state" / "memories" / "go-react-profile.md"
            memory = parse_memory(memory_path.read_text(encoding="utf-8"))
            self.assertEqual(memory.name, "go-react-profile")
            self.assertEqual(memory.description, "User is experienced in Go and new to React")
            self.assertEqual(memory.type, "user")
            self.assertIn("ten years of Go", memory.content)
            self.assertTrue(any("Memory saved:" in output for output in outputs))
            self.assertFalse((root / "state" / "hard_memory.md").exists())
            self.assertFalse((root / "state" / "soft_memory.md").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_context_keeps_active_task_and_conversation_separate(self) -> None:
        state = create_initial_state("Fix it and run the focused test")
        state.user_goal = "T3: Persistence layer"
        state.conversation_messages = [
            {"role": "user", "content": "Inspect the failing test"},
            {"role": "assistant", "content": "The parser rejects null end values."},
            {"role": "user", "content": "Fix it and run the focused test"},
        ]
        context = ContextBuilder(Path.cwd())._working_context(state)
        self.assertIn("# Active Task\nT3: Persistence layer", context)
        self.assertIn("# User Conversation", context)
        self.assertIn("## Conversation Turn 1", context)
        self.assertIn("User:\nInspect the failing test", context)
        self.assertIn("Agent:\nThe parser rejects null end values.", context)
        self.assertIn("## Conversation Turn 2", context)
        self.assertIn("Latest User Message:\nFix it and run the focused test", context)
        restored = TaskState.from_dict(state.to_dict())
        self.assertEqual(restored.conversation_messages, state.conversation_messages)

    def test_conversation_context_keeps_all_messages_without_truncation(self) -> None:
        state = create_initial_state("Explain everything")
        old_message = "old-start-" + ("x" * 9000) + "-old-end"
        state.conversation_messages = [
            {"role": "user", "content": old_message},
            {"role": "assistant", "content": "First answer."},
            {"role": "user", "content": "Latest question."},
        ]
        context = ContextBuilder(Path.cwd())._conversation_context(state)
        self.assertIn("old-start-", context)
        self.assertIn("-old-end", context)
        self.assertIn("First answer.", context)
        self.assertIn("Latest User Message:\nLatest question.", context)

    def test_make_loop_uses_only_current_session_messages(self) -> None:
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            use_color=False,
        )
        cli.messages = [
            ChatMessage("user", "old session question"),
            ChatMessage("assistant", "old session answer"),
            ChatMessage("user", "new session question"),
        ]
        cli.context_message_start = 2

        loop = cli._make_loop(
            "Resume task",
            resume=True,
            include_conversation=True,
            interaction_mode="question",
        )

        self.assertEqual(loop.conversation_messages, [{"role": "user", "content": "new session question"}])

    def test_commands_work_without_starting_agent_loop(self) -> None:
        outputs: list[str] = []
        inputs = iter(["/status", "/history", "/new", "/exit"])
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd() / "missing-chat-root", provider="offline", max_steps=1),
            input_fn=lambda prompt: next(inputs),
            output_fn=outputs.append,
        )
        with patch.object(cli, "_append_history") as append_history:
            self.assertEqual(cli.run(), 0)
        self.assertTrue(any("No durable agent state" in line for line in outputs))
        self.assertTrue(any("No messages" in line for line in outputs))
        append_history.assert_called_once()

    def test_tool_events_render_thought_and_start_without_results(self) -> None:
        outputs: list[str] = []
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            output_fn=outputs.append,
            use_color=False,
        )
        cli._show_event(
            {
                "type": "tool_start",
                "step": 2,
                "action": "read",
                "target": "README.md",
                "thought_summary": "Inspect the README without truncating the full reasoning summary.",
            }
        )
        cli._show_event({"type": "tool_result", "step": 2, "action": "read", "ok": True, "summary": "Read 20 lines."})
        self.assertEqual(outputs[0], "      Inspect the README without truncating the full reasoning summary.")
        self.assertEqual(outputs[1], "      calling read README.md")
        self.assertEqual(len(outputs), 2)

    def test_tool_event_thought_summary_is_not_truncated(self) -> None:
        outputs: list[str] = []
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            output_fn=outputs.append,
            use_color=False,
        )
        long_summary = "Start " + ("x" * 180) + " end"
        cli._show_event(
            {
                "type": "tool_start",
                "step": 2,
                "action": "search",
                "target": "needle",
                "thought_summary": long_summary,
            }
        )
        self.assertEqual(outputs[0], f"      {long_summary}")
        self.assertNotIn("thought_summary", outputs[0])
        self.assertEqual(outputs[1], "      calling search needle")

    def test_tool_event_target_is_not_truncated(self) -> None:
        outputs: list[str] = []
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            output_fn=outputs.append,
            use_color=False,
        )
        long_target = (
            "eval/benchmarks/budget_management/workspace/src/transaction_service.py "
            "lines 1-240 for periodic transaction implementation details"
        )
        cli._show_event(
            {
                "type": "tool_start",
                "step": 2,
                "action": "read",
                "target": long_target,
            }
        )
        self.assertEqual(outputs[0], f"      calling read {long_target}")
        self.assertNotIn("...", outputs[0])

    def test_verify_action_is_not_rendered(self) -> None:
        outputs: list[str] = []
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            output_fn=outputs.append,
            use_color=False,
        )
        cli._show_event(
            {
                "type": "tool_start",
                "step": 3,
                "action": "verify",
                "target": "default",
                "thought_summary": "Run independent verification.",
            }
        )
        cli._show_event(
            {
                "type": "tool_result",
                "step": 3,
                "action": "verify",
                "ok": False,
                "summary": "Protocol error: The read operation timed out",
            }
        )
        self.assertEqual(outputs, ["      Run independent verification."])

    def test_answer_action_is_not_rendered_as_tool_call(self) -> None:
        outputs: list[str] = []
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            output_fn=outputs.append,
            use_color=False,
        )
        cli._show_event({"type": "tool_start", "step": 1, "action": "answer", "target": ""})
        cli._show_event({"type": "tool_result", "step": 1, "action": "answer", "ok": True})
        self.assertEqual(outputs, [])

    def test_answer_does_not_complete_contract_managed_build_task(self) -> None:
        loop = AgentLoop(root=Path.cwd(), task="Status", max_steps=1)
        state = create_initial_state("Status")
        state.task_id = "T5"
        state.nodes = [
            {
                "id": "T5",
                "title": "Build feature",
                "status": "in_progress",
                "evidence": [],
                "contract_managed": True,
            }
        ]
        loop._update_state(
            state,
            {"action": "answer", "target": "", "args": {"answer": "Still in progress."}},
            ToolResult(True, "Final answer produced.", {"answer": "Still in progress."}),
        )
        self.assertEqual(state.nodes[0]["status"], "in_progress")
        self.assertEqual(state.evidence_sources[-1]["evidence_type"], "user_response")

    def test_question_mode_rejects_project_progress_actions(self) -> None:
        loop = AgentLoop(root=Path.cwd(), task="Status", max_steps=1)
        state = create_initial_state("Status")
        state.interaction_mode = "question"
        observation = loop._execute_action(
            {"action": "contract", "target": "T5", "args": {}},
            state,
        )
        self.assertFalse(observation.ok)
        self.assertTrue(observation.data["interactive_question"])

    def test_question_mode_can_answer_during_initializer(self) -> None:
        loop = AgentLoop(root=Path.cwd(), task="INIT", max_steps=1)
        state = create_initial_state("Status")
        state.task_id = "INIT"
        state.nodes = [{"id": "INIT", "status": "in_progress", "evidence": []}]
        state.interaction_mode = "question"
        observation = loop._execute_action(
            {"action": "answer", "target": "", "args": {"answer": "Initialization is still running."}},
            state,
        )
        self.assertTrue(observation.ok)

    @patch("agent.chat.os.name", "nt")
    @patch("agent.chat.subprocess.Popen")
    def test_launch_chat_window_starts_child_console(self, popen) -> None:
        self.assertTrue(launch_chat_window(["--chat", "--provider", "offline"], Path.cwd()))
        command = popen.call_args.args[0]
        self.assertEqual(command[:3], [__import__("sys").executable, "-m", "agent.main"])
        self.assertIn("--chat-child", command)
        self.assertEqual(popen.call_args.kwargs["creationflags"], __import__("subprocess").CREATE_NEW_CONSOLE)


if __name__ == "__main__":
    unittest.main()
