from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import call, patch
from uuid import uuid4

from agent.chat import ChatConfig, ChatMessage, InteractiveCLI, launch_chat_window
from agent.context import ContextBuilder
from agent.loop import AgentLoop
from agent.main import build_parser, resolve_optional_task
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

    def test_explicit_ask_and_do_commands_select_mode(self) -> None:
        outputs: list[str] = []
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            output_fn=outputs.append,
            use_color=False,
        )
        with patch.object(cli, "_run_turn") as run_turn:
            cli._handle_command("/ask 显示现在task的执行进度")
            cli._handle_command("/do 继续执行当前任务")
        self.assertEqual(
            run_turn.call_args_list,
            [
                call("显示现在task的执行进度", interaction_mode="question"),
                call("继续执行当前任务", interaction_mode="work"),
            ],
        )

    def test_plain_message_requires_explicit_mode(self) -> None:
        outputs: list[str] = []
        inputs = iter(["显示进度", "/exit"])
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            input_fn=lambda prompt: next(inputs),
            output_fn=outputs.append,
            use_color=False,
        )
        self.assertEqual(cli.run(), 0)
        self.assertIn("Choose an explicit mode: /ask <question> or /do <task>.", outputs)

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
            with patch.object(cli, "_run_turn") as run_turn:
                cli._handle_command("/skill")
            run_turn.assert_not_called()
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
        with patch.object(cli, "_run_turn") as run_turn:
            cli._handle_command("/skill")
        run_turn.assert_not_called()
        self.assertIn("Name is required.", outputs)
        self.assertIn("Skill setup cancelled.", outputs)

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
        self.assertIn("Agent: The parser rejects null end values.", context)
        self.assertIn("Latest User Message: Fix it and run the focused test", context)
        restored = TaskState.from_dict(state.to_dict())
        self.assertEqual(restored.conversation_messages, state.conversation_messages)

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

    def test_tool_events_are_rendered_as_start_and_result(self) -> None:
        outputs: list[str] = []
        cli = InteractiveCLI(
            ChatConfig(root=Path.cwd(), provider="offline", max_steps=1),
            output_fn=outputs.append,
            use_color=False,
        )
        cli._show_event({"type": "tool_start", "step": 2, "action": "read", "target": "README.md"})
        cli._show_event({"type": "tool_result", "step": 2, "action": "read", "ok": True, "summary": "Read 20 lines."})
        self.assertEqual(outputs[0], "   2  calling read README.md")
        self.assertEqual(outputs[1], "      result OK")
        self.assertEqual(outputs[2], "      Read 20 lines.")

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
