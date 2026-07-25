from __future__ import annotations

import io
import json
import os
import socket
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from agent.llm import (
    OpenAICompatibleDecisionMaker,
    ProviderProtocolError,
    ProviderRequestError,
    configure_auxiliary_payload,
    create_decision_maker,
)
from agent.loop import AgentLoop
from agent.memory_retrieval import MemoryHeader, MemoryRetriever, RetrievedMemories
from agent.planner import create_initial_state
from agent.spec_builder import OpenAICompatibleSpecBuilder
from agent.tools import ToolResult
from agent.token_usage import normalize_token_usage
from agent.ui_contract import OpenAICompatibleUIContractBuilder


def _action(target: str = ".") -> dict[str, object]:
    return {
        "thought_summary": "Inspect.",
        "action": "list_files",
        "target": target,
        "args": {},
        "expected_observation": "Workspace entries.",
        "risk": "low",
    }


def _native_response(
    call_id: str,
    *,
    action: dict[str, object] | None = None,
    reasoning: str = "",
    usage: dict[str, object] | None = None,
) -> dict[str, object]:
    message: dict[str, object] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "submit_action",
                    "arguments": json.dumps(action or _action()),
                },
            }
        ],
    }
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "choices": [{"message": message}],
        "usage": usage
        or {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _direct_native_response(
    call_id: str,
    *,
    action: dict[str, object] | None = None,
    reasoning: str = "",
    usage: dict[str, object] | None = None,
) -> dict[str, object]:
    normalized = dict(action or _action())
    function_name = str(normalized.pop("action"))
    message: dict[str, object] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(normalized),
                },
            }
        ],
    }
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "choices": [{"message": message}],
        "usage": usage
        or {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback

    def read(self) -> bytes:
        return self.body


class _LocalChatServer:
    """Minimal one-request HTTP server used to exercise the real urllib transport."""

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.request_payload: dict[str, object] = {}
        self.request_path = ""
        self.error: BaseException | None = None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(1)
        self.socket.settimeout(5)
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self) -> "_LocalChatServer":
        self.thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.thread.join(timeout=5)
        self.socket.close()
        if self.thread.is_alive():
            raise AssertionError("Local fake HTTP server did not stop.")
        if self.error is not None:
            raise self.error

    @property
    def base_url(self) -> str:
        host, port = self.socket.getsockname()
        return f"http://{host}:{port}/v1"

    def _serve(self) -> None:
        try:
            connection, _ = self.socket.accept()
            with connection:
                connection.settimeout(5)
                raw = b""
                while b"\r\n\r\n" not in raw:
                    raw += connection.recv(65536)
                header_bytes, body = raw.split(b"\r\n\r\n", 1)
                header_lines = header_bytes.decode("iso-8859-1").splitlines()
                self.request_path = header_lines[0].split(" ")[1]
                headers = {}
                for line in header_lines[1:]:
                    name, separator, value = line.partition(":")
                    if separator:
                        headers[name.strip().lower()] = value.strip()
                content_length = int(headers.get("content-length", "0"))
                while len(body) < content_length:
                    body += connection.recv(65536)
                self.request_payload = json.loads(body[:content_length].decode("utf-8"))
                response_body = json.dumps(self.response).encode("utf-8")
                connection.sendall(
                    (
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: application/json\r\n"
                        f"Content-Length: {len(response_body)}\r\n"
                        "Connection: close\r\n"
                        "\r\n"
                    ).encode("ascii")
                    + response_body
                )
        except BaseException as exc:
            self.error = exc


class DeepSeekProtocolTests(unittest.TestCase):
    def test_offline_provider_remains_deterministic_without_native_transcript(self) -> None:
        decision_maker = create_decision_maker("offline")
        action = decision_maker.next_action(
            "Offline context.",
            create_initial_state("Inspect"),
        )

        self.assertFalse(decision_maker.uses_native_tools)
        self.assertEqual(action["action"], "update_plan")

    def test_real_transport_uses_local_fake_http_server(self) -> None:
        with _LocalChatServer(_native_response("call-local-http")) as server:
            decision_maker = OpenAICompatibleDecisionMaker(
                api_key="test-key",
                base_url=server.base_url,
                model="test-model",
            )
            action = decision_maker.next_action(
                "Local HTTP transport context.",
                create_initial_state("Inspect"),
            )

        self.assertEqual(action["action"], "list_files")
        self.assertEqual(server.request_path, "/v1/chat/completions")
        self.assertEqual(
            server.request_payload["tools"][0]["function"]["name"],
            "submit_action",
        )
        self.assertEqual(
            server.request_payload["tool_choice"],
            {"type": "function", "function": {"name": "submit_action"}},
        )
        self.assertEqual(
            [message["role"] for message in server.request_payload["messages"]],
            ["system", "user"],
        )

    def test_native_messages_round_trip_reasoning_and_tool_result(self) -> None:
        responses = [
            _direct_native_response("call-1", reasoning="private reasoning one"),
            _direct_native_response("call-2", reasoning="private reasoning two"),
        ]
        responses[0]["choices"][0]["message"]["content"] = "Visible tool-call preface."
        captured_payloads: list[dict[str, object]] = []
        decision_maker = OpenAICompatibleDecisionMaker(
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )
        decision_maker._post_chat_completions = lambda payload: (
            captured_payloads.append(json.loads(json.dumps(payload))) or responses.pop(0)
        )
        state = create_initial_state("Inspect workspace")
        conversation = [
            {"role": "user", "content": "Previous user turn."},
            {"role": "assistant", "content": "Previous visible answer."},
            {"role": "user", "content": "Current user turn."},
        ]
        decision_maker.begin_session("Harness context without flattened chat.", conversation)

        first_action = decision_maker.next_action("ignored after begin", state)
        decision_maker.record_tool_result(
            {"ok": True, "summary": "Listed.", "data": {"entries": []}},
            {"task_id": "current", "handoff_ready": False},
        )
        second_action = decision_maker.next_action("ignored after begin", state)

        self.assertEqual(first_action["action"], "list_files")
        self.assertEqual(second_action["action"], "list_files")
        first_payload, second_payload = captured_payloads
        direct_tool_names = {
            tool["function"]["name"] for tool in first_payload["tools"]
        }
        self.assertIn("read", direct_tool_names)
        self.assertIn("write", direct_tool_names)
        self.assertNotIn("submit_action", direct_tool_names)
        self.assertEqual(first_payload["thinking"], {"type": "enabled"})
        self.assertEqual(first_payload["reasoning_effort"], "high")
        self.assertNotIn("temperature", first_payload)
        self.assertEqual(
            [message["role"] for message in first_payload["messages"]],
            ["system", "user", "assistant", "user", "user"],
        )
        self.assertEqual(
            second_payload["messages"][: len(first_payload["messages"])],
            first_payload["messages"],
        )
        self.assertEqual(second_payload["tools"], first_payload["tools"])
        self.assertNotIn("tool_choice", first_payload)
        self.assertNotIn("tool_choice", second_payload)
        assistant_message = second_payload["messages"][-2]
        tool_message = second_payload["messages"][-1]
        self.assertEqual(assistant_message["reasoning_content"], "private reasoning one")
        self.assertEqual(assistant_message["content"], "Visible tool-call preface.")
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "call-1")
        self.assertEqual(json.loads(tool_message["content"])["state_update"]["task_id"], "current")

    def test_deepseek_direct_action_accepts_minimal_declared_arguments(self) -> None:
        response = _direct_native_response("call-direct-read")
        response["choices"][0]["message"]["tool_calls"][0]["function"] = {
            "name": "read",
            "arguments": json.dumps({"target": "README.md"}),
        }
        decision_maker = OpenAICompatibleDecisionMaker(
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
        )
        decision_maker._post_chat_completions = lambda payload: response

        action = decision_maker.next_action(
            "Read the project overview.",
            create_initial_state("Inspect"),
        )

        self.assertEqual(action["action"], "read")
        self.assertEqual(action["target"], "README.md")
        self.assertEqual(action["args"], {})
        self.assertEqual(action["risk"], "medium")

    def test_deepseek_content_only_response_is_reprompted_as_a_native_turn(self) -> None:
        responses = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I should continue inspecting.",
                            "reasoning_content": "private content-only reasoning",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
            _direct_native_response("call-after-content"),
        ]
        payloads: list[dict[str, object]] = []
        decision_maker = OpenAICompatibleDecisionMaker(
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
        )
        decision_maker._post_chat_completions = lambda payload: (
            payloads.append(json.loads(json.dumps(payload))) or responses.pop(0)
        )

        action = decision_maker.next_action(
            "Inspect the workspace.",
            create_initial_state("Inspect"),
        )

        self.assertEqual(action["action"], "list_files")
        self.assertEqual(len(payloads), 2)
        self.assertEqual(
            [message["role"] for message in payloads[1]["messages"]],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(
            payloads[1]["messages"][-2]["reasoning_content"],
            "private content-only reasoning",
        )
        self.assertIn("protocol correction", payloads[1]["messages"][-1]["content"])
        self.assertEqual(decision_maker.last_token_usage["input_tokens"], 20)
        self.assertEqual(decision_maker.last_token_usage["output_tokens"], 10)
        self.assertEqual(
            decision_maker.last_token_usage["provider_usage"]["request_count"],
            2,
        )

    def test_deepseek_parallel_tool_calls_are_closed_and_reprompted(self) -> None:
        parallel = _direct_native_response("call-parallel-1")
        second_call = json.loads(
            json.dumps(parallel["choices"][0]["message"]["tool_calls"][0])
        )
        second_call["id"] = "call-parallel-2"
        second_call["function"]["arguments"] = json.dumps(
            {
                "thought_summary": "Inspect another directory.",
                "target": "state",
                "args": {"recursive": False},
                "expected_observation": "State entries.",
                "risk": "low",
            }
        )
        parallel["choices"][0]["message"]["tool_calls"].append(second_call)
        responses = [
            parallel,
            _direct_native_response("call-after-parallel"),
        ]
        payloads: list[dict[str, object]] = []
        decision_maker = OpenAICompatibleDecisionMaker(
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
        )
        decision_maker._post_chat_completions = lambda payload: (
            payloads.append(json.loads(json.dumps(payload))) or responses.pop(0)
        )

        action = decision_maker.next_action(
            "Inspect one directory at a time.",
            create_initial_state("Inspect"),
        )

        self.assertEqual(action["action"], "list_files")
        self.assertEqual(len(payloads), 2)
        self.assertEqual(
            [message["role"] for message in payloads[1]["messages"]],
            ["system", "user", "assistant", "tool", "tool"],
        )
        tool_messages = payloads[1]["messages"][-2:]
        self.assertEqual(
            [message["tool_call_id"] for message in tool_messages],
            ["call-parallel-1", "call-parallel-2"],
        )
        self.assertTrue(
            all(
                json.loads(message["content"])["observation"]["data"][
                    "parallel_tool_calls_rejected"
                ]
                for message in tool_messages
            )
        )

    def test_deepseek_invalid_tool_arguments_are_closed_and_reprompted(self) -> None:
        malformed = _direct_native_response("call-invalid-json")
        malformed["choices"][0]["message"]["tool_calls"][0]["function"] = {
            "name": "write",
            "arguments": (
                '{"thought_summary":"Write a test.","target":"tests/test_ui.py",'
                '"args":{"content":"self.assertIn("width=400", html)"},'
                '"expected_observation":"Test written.","risk":"low"}'
            ),
        }
        responses = [
            malformed,
            _direct_native_response("call-after-invalid-json"),
        ]
        payloads: list[dict[str, object]] = []
        decision_maker = OpenAICompatibleDecisionMaker(
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
        )
        decision_maker._post_chat_completions = lambda payload: (
            payloads.append(json.loads(json.dumps(payload))) or responses.pop(0)
        )

        action = decision_maker.next_action(
            "Write one test artifact.",
            create_initial_state("Inspect"),
        )

        self.assertEqual(action["action"], "list_files")
        self.assertEqual(len(payloads), 2)
        self.assertEqual(
            [message["role"] for message in payloads[1]["messages"]],
            ["system", "user", "assistant", "tool"],
        )
        assistant_message = payloads[1]["messages"][-2]
        tool_message = payloads[1]["messages"][-1]
        self.assertEqual(tool_message["tool_call_id"], "call-invalid-json")
        self.assertEqual(
            assistant_message["tool_calls"][0]["function"]["arguments"],
            malformed["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        )
        rejection = json.loads(tool_message["content"])
        self.assertFalse(rejection["observation"]["ok"])
        self.assertTrue(
            rejection["observation"]["data"]["invalid_tool_arguments_json"]
        )
        self.assertEqual(
            rejection["observation"]["data"]["function_name"],
            "write",
        )
        self.assertIn(
            "was not executed",
            rejection["state_update"]["required_next_action"],
        )
        self.assertEqual(
            decision_maker.last_token_usage["provider_usage"]["request_count"],
            2,
        )

    def test_content_json_without_tool_call_is_rejected(self) -> None:
        decision_maker = OpenAICompatibleDecisionMaker(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
        )
        decision_maker._post_chat_completions = lambda payload: {
            "choices": [{"message": {"role": "assistant", "content": json.dumps(_action())}}]
        }

        with self.assertRaisesRegex(ProviderProtocolError, "exactly one native tool call"):
            decision_maker.next_action("Context", create_initial_state("Inspect"))

    def test_schema_error_stops_without_fabricating_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Inspect", max_steps=3)
            decision_maker = OpenAICompatibleDecisionMaker(
                api_key="test-key",
                base_url="https://example.invalid/v1",
                model="test-model",
            )
            decision_maker._post_chat_completions = lambda payload: {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(_action()),
                        }
                    }
                ]
            }
            loop.decision_maker = decision_maker
            with self.assertLogs("long_agent", level="ERROR"):
                result = loop.run()

            state = json.loads(loop.state_path.read_text(encoding="utf-8"))
            transcript_events = [
                json.loads(line)
                for line in loop.provider_session_path.read_text(encoding="utf-8").splitlines()
            ]
            handoff_exists = loop.handoff_path.exists()

        self.assertEqual(result.steps, 1)
        self.assertEqual(state["last_action"]["action"], "provider_error")
        self.assertEqual(
            state["last_observation"]["data"]["provider_transcript"],
            f"state/provider_sessions/{loop.provider_session_path.name}",
        )
        self.assertNotIn("tool_result", [event["type"] for event in transcript_events])
        self.assertTrue(handoff_exists)

    def test_executor_exception_closes_native_tool_call_before_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Inspect", max_steps=3)
            decision_maker = OpenAICompatibleDecisionMaker(
                api_key="test-key",
                base_url="https://example.invalid/v1",
                model="test-model",
            )
            decision_maker._post_chat_completions = lambda payload: _native_response(
                "call-executor-error"
            )
            loop.decision_maker = decision_maker
            loop._execute_action = lambda action, state: (_ for _ in ()).throw(
                RuntimeError("executor exploded")
            )
            with self.assertLogs("long_agent", level="ERROR"):
                result = loop.run()

            state = json.loads(loop.state_path.read_text(encoding="utf-8"))
            transcript_events = [
                json.loads(line)
                for line in loop.provider_session_path.read_text(encoding="utf-8").splitlines()
            ]
            ledger = json.loads(loop.tool_call_ledger_path.read_text(encoding="utf-8"))

        self.assertEqual(result.steps, 1)
        self.assertEqual(state["last_action"]["action"], "list_files")
        self.assertIn("executor exploded", state["last_observation"]["summary"])
        self.assertEqual(transcript_events[-1]["type"], "tool_result")
        self.assertEqual(
            transcript_events[-1]["message"]["tool_call_id"],
            "call-executor-error",
        )
        self.assertEqual(
            ledger["calls"]["call-executor-error"]["status"],
            "completed",
        )

    def test_context_preflight_blocks_request_before_network_call(self) -> None:
        decision_maker = OpenAICompatibleDecisionMaker(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            context_window_tokens=1,
        )
        called = False

        def fake_post(payload: dict[str, object]) -> dict[str, object]:
            nonlocal called
            called = True
            return _native_response("unexpected")

        decision_maker._post_chat_completions = fake_post
        with self.assertRaisesRegex(ProviderRequestError, "preflight exceeded"):
            decision_maker.next_action("Context", create_initial_state("Inspect"))
        self.assertFalse(called)

    def test_auxiliary_deepseek_calls_explicitly_disable_thinking(self) -> None:
        payload = {"model": "deepseek-chat", "temperature": 0}
        configure_auxiliary_payload(
            payload,
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )

        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_all_single_shot_deepseek_helpers_disable_thinking(self) -> None:
        captured: list[dict[str, object]] = []
        spec_builder = OpenAICompatibleSpecBuilder(
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )
        spec_builder._post_chat_completions = lambda payload: (
            captured.append(json.loads(json.dumps(payload)))
            or {"choices": [{"message": {"content": "# Project"}}]}
        )
        spec_builder.build([{"role": "user", "content": "Build it."}])

        ui_builder = OpenAICompatibleUIContractBuilder(
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )
        ui_builder._post_chat_completions = lambda payload: (
            captured.append(json.loads(json.dumps(payload)))
            or {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "kind": "ui_contract",
                                    "version": 1,
                                    "generated_by": "llm",
                                    "contracts": [],
                                }
                            )
                        }
                    }
                ]
            }
        )
        ui_builder.build([])

        with tempfile.TemporaryDirectory() as tmp:
            retriever = MemoryRetriever(
                Path(tmp),
                api_key="test-key",
                base_url="https://api.deepseek.com/v1",
                model="deepseek-chat",
            )
            retriever._post_chat_completions = lambda payload: (
                captured.append(json.loads(json.dumps(payload)))
                or {"choices": [{"message": {"content": '["preference.md"]'}}]}
            )
            selected = retriever._select_with_model(
                "Use the preference.",
                [
                    MemoryHeader(
                        filename="preference.md",
                        name="preference",
                        description="User preference",
                        type="user",
                        mtime_ms=0,
                    )
                ],
            )

        self.assertEqual(selected, ["preference.md"])
        self.assertEqual(len(captured), 3)
        self.assertTrue(
            all(payload["thinking"] == {"type": "disabled"} for payload in captured)
        )

    def test_deepseek_reasoning_effort_is_normalized(self) -> None:
        decision_maker = OpenAICompatibleDecisionMaker(
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            reasoning_effort="xhigh",
        )
        decision_maker.begin_session("Context")
        payload = decision_maker._native_payload()

        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertNotIn("tool_choice", payload)

    def test_auto_thinking_omits_deepseek_fields_for_other_endpoints(self) -> None:
        decision_maker = OpenAICompatibleDecisionMaker(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-4.1-mini",
        )
        decision_maker.begin_session("Context")
        payload = decision_maker._native_payload()

        self.assertNotIn("thinking", payload)
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(payload["temperature"], 0.1)

    def test_explicitly_disabled_deepseek_thinking_keeps_sampling_parameter(self) -> None:
        decision_maker = OpenAICompatibleDecisionMaker(
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            thinking=False,
        )
        decision_maker.begin_session("Context")
        payload = decision_maker._native_payload()

        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["temperature"], 0.1)
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(
            payload["tool_choice"],
            {"type": "function", "function": {"name": "submit_action"}},
        )

    def test_multiple_tool_calls_are_rejected(self) -> None:
        response = _native_response("call-1")
        message = response["choices"][0]["message"]
        message["tool_calls"].append(message["tool_calls"][0])
        decision_maker = OpenAICompatibleDecisionMaker(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
        )
        decision_maker._post_chat_completions = lambda payload: response

        with self.assertRaisesRegex(ProviderProtocolError, "received 2"):
            decision_maker.next_action("Context", create_initial_state("Inspect"))

    def test_malformed_native_action_schema_stops_before_appending_assistant(self) -> None:
        malformed = _action()
        malformed.pop("risk")
        decision_maker = OpenAICompatibleDecisionMaker(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
        )
        decision_maker._post_chat_completions = lambda payload: _native_response(
            "call-malformed",
            action=malformed,
        )

        with self.assertRaisesRegex(ProviderProtocolError, "missing required fields: risk"):
            decision_maker.next_action("Context", create_initial_state("Inspect"))

        self.assertEqual(
            [message["role"] for message in decision_maker.messages],
            ["system", "user"],
        )

    def test_malformed_assistant_and_tool_call_schema_are_rejected(self) -> None:
        invalid_cases = [
            ("role", "Provider response message role must be assistant"),
            ("reasoning", "reasoning_content must be a string or null"),
            ("tool_type", "Native tool call type must be function"),
        ]
        for case, expected in invalid_cases:
            with self.subTest(case=case):
                response = _native_response(f"call-{case}")
                message = response["choices"][0]["message"]
                if case == "role":
                    message["role"] = "user"
                elif case == "reasoning":
                    message["reasoning_content"] = {"private": "invalid"}
                else:
                    message["tool_calls"][0]["type"] = "custom"
                decision_maker = OpenAICompatibleDecisionMaker(
                    api_key="test-key",
                    base_url="https://example.invalid/v1",
                    model="test-model",
                )
                decision_maker._post_chat_completions = lambda payload, value=response: value

                with self.assertRaisesRegex(ProviderProtocolError, expected):
                    decision_maker.next_action(
                        "Context",
                        create_initial_state("Inspect"),
                    )

                self.assertEqual(
                    [item["role"] for item in decision_maker.messages],
                    ["system", "user"],
                )

    def test_retryable_429_is_retried_but_400_is_not(self) -> None:
        retry_error = HTTPError(
            "https://example.invalid/v1/chat/completions",
            429,
            "slow down",
            {"Retry-After": "0"},
            io.BytesIO(b'{"error":{"message":"slow down"}}'),
        )
        decision_maker = OpenAICompatibleDecisionMaker(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            max_attempts=2,
        )
        with patch(
            "agent.llm.urllib.request.urlopen",
            side_effect=[retry_error, _FakeHTTPResponse(_native_response("call-after-retry"))],
        ) as urlopen:
            action = decision_maker.next_action("Context", create_initial_state("Inspect"))
            self.assertEqual(action["action"], "list_files")
            self.assertEqual(urlopen.call_count, 2)
            first_request = urlopen.call_args_list[0].args[0]
            second_request = urlopen.call_args_list[1].args[0]
            self.assertEqual(first_request.data, second_request.data)

        for status in (400, 401, 403, 404):
            with self.subTest(status=status):
                permanent_error = HTTPError(
                    "https://example.invalid/v1/chat/completions",
                    status,
                    "permanent error",
                    {},
                    io.BytesIO(b'{"error":{"message":"permanent"}}'),
                )
                decision_maker = OpenAICompatibleDecisionMaker(
                    api_key="test-key",
                    base_url="https://example.invalid/v1",
                    model="test-model",
                    max_attempts=3,
                )
                with patch(
                    "agent.llm.urllib.request.urlopen",
                    side_effect=permanent_error,
                ) as urlopen:
                    with self.assertRaises(ProviderRequestError) as caught:
                        decision_maker.next_action(
                            "Context",
                            create_initial_state("Inspect"),
                        )
                    self.assertEqual(caught.exception.status_code, status)
                    self.assertFalse(caught.exception.retryable)
                    self.assertEqual(urlopen.call_count, 1)

        for status in (408, 500):
            with self.subTest(status=status):
                retryable_error = HTTPError(
                    "https://example.invalid/v1/chat/completions",
                    status,
                    "retryable error",
                    {"Retry-After": "0"},
                    io.BytesIO(b'{"error":{"message":"retryable"}}'),
                )
                decision_maker = OpenAICompatibleDecisionMaker(
                    api_key="test-key",
                    base_url="https://example.invalid/v1",
                    model="test-model",
                    max_attempts=2,
                )
                with patch(
                    "agent.llm.urllib.request.urlopen",
                    side_effect=[
                        retryable_error,
                        _FakeHTTPResponse(_native_response(f"call-{status}")),
                    ],
                ) as urlopen:
                    action = decision_maker.next_action(
                        "Context",
                        create_initial_state("Inspect"),
                    )
                    self.assertEqual(action["action"], "list_files")
                    self.assertEqual(urlopen.call_count, 2)

    def test_network_retry_exhaustion_uses_three_identical_requests(self) -> None:
        decision_maker = OpenAICompatibleDecisionMaker(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            max_attempts=3,
        )
        with patch(
            "agent.llm.urllib.request.urlopen",
            side_effect=[
                URLError("network down"),
                URLError("network down"),
                URLError("network down"),
            ],
        ) as urlopen, patch("agent.llm.time.sleep"):
            with self.assertRaises(ProviderRequestError) as caught:
                decision_maker.next_action("Context", create_initial_state("Inspect"))

        self.assertTrue(caught.exception.retryable)
        self.assertEqual(urlopen.call_count, 3)
        request_bodies = [call.args[0].data for call in urlopen.call_args_list]
        self.assertEqual(request_bodies, [request_bodies[0]] * 3)

        capped = OpenAICompatibleDecisionMaker(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            max_attempts=99,
        )
        self.assertEqual(capped.max_attempts, 3)

    def test_provider_transcript_keeps_full_reasoning_with_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "provider_sessions" / "trace.jsonl"
            decision_maker = OpenAICompatibleDecisionMaker(
                api_key="test-key",
                base_url="https://api.deepseek.com/v1",
                model="deepseek-chat",
            )
            decision_maker.set_transcript_path(path)
            decision_maker._post_chat_completions = lambda payload: _direct_native_response(
                "call-secret",
                reasoning="full private reasoning",
            )
            decision_maker.next_action("Context", create_initial_state("Inspect"))

            transcript = path.read_text(encoding="utf-8")
            mode = stat.S_IMODE(path.stat().st_mode)
            directory_mode = stat.S_IMODE(path.parent.stat().st_mode)

        self.assertIn("full private reasoning", transcript)
        self.assertEqual(mode, 0o600)
        self.assertEqual(directory_mode, 0o700)

    def test_token_usage_keeps_cache_and_reasoning_details(self) -> None:
        usage = normalize_token_usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "total_tokens": 130,
                "prompt_cache_hit_tokens": 70,
                "prompt_cache_miss_tokens": 30,
                "cache_creation_input_tokens": 8,
                "completion_tokens_details": {"reasoning_tokens": 20},
                "provider_extension": {"cache_tier": "disk"},
            }
        )

        self.assertEqual(usage["cache_hit_tokens"], 70)
        self.assertEqual(usage["cache_miss_tokens"], 30)
        self.assertEqual(usage["cache_write_tokens"], 8)
        self.assertEqual(usage["reasoning_tokens"], 20)
        self.assertEqual(
            usage["provider_usage"]["provider_extension"],
            {"cache_tier": "disk"},
        )

    def test_loop_updates_state_before_appending_tool_result(self) -> None:
        class NativeDecisionMaker:
            uses_native_tools = True
            fatal_protocol_errors = True
            model = "test-model"
            last_token_usage = None
            last_tool_call_id = "call-state"

            def set_transcript_path(self, path: Path) -> None:
                self.path = path

            def begin_session(self, context: str, conversation: list[dict[str, str]]) -> None:
                self.context = context
                self.conversation = conversation

            def next_action(self, context: str, state: object) -> dict[str, object]:
                del context, state
                self.last_tool_call_id = "call-state"
                return _action()

            def request_token_estimate(self) -> int:
                return 10

            def record_tool_result(
                self,
                observation: dict[str, object],
                state_update: dict[str, object],
            ) -> None:
                self.observation = observation
                self.state_update = state_update

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Inspect", max_steps=1)
            loop._ensure_state_files()
            decision_maker = NativeDecisionMaker()
            loop.decision_maker = decision_maker
            state = create_initial_state("Inspect")
            loop._run_one_session(state)

        self.assertIn("iterations=1", decision_maker.state_update["state_summary"])
        self.assertEqual(
            decision_maker.observation["summary"],
            state.last_observation["summary"],
        )
        self.assertNotIn("last_action", decision_maker.state_update)
        self.assertNotIn("last_observation", decision_maker.state_update)

    def test_provider_protocol_error_stops_session(self) -> None:
        class BrokenDecisionMaker:
            uses_native_tools = True
            fatal_protocol_errors = True
            model = "broken"
            last_token_usage = None
            calls = 0

            def set_transcript_path(self, path: Path) -> None:
                del path

            def begin_session(self, context: str, conversation: list[dict[str, str]]) -> None:
                del context, conversation

            def next_action(self, context: str, state: object) -> dict[str, object]:
                del context, state
                self.calls += 1
                raise ProviderProtocolError("missing tool call")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Inspect", max_steps=3)
            loop._ensure_state_files()
            decision_maker = BrokenDecisionMaker()
            loop.decision_maker = decision_maker
            state = create_initial_state("Inspect")
            with self.assertLogs("long_agent", level="ERROR"):
                result = loop._run_one_session(state)

        self.assertEqual(decision_maker.calls, 1)
        self.assertEqual(result.steps, 1)
        self.assertEqual(state.last_action["action"], "provider_error")
        self.assertIn("missing tool call", result.message)

    def test_tool_call_ledger_replays_only_identical_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Inspect", max_steps=1)
            loop._ensure_state_files()
            action = _action()
            observation = ToolResult(True, "Listed.", {"entries": []})
            loop._reserve_tool_call("call-interrupted", action)
            interrupted = loop._tool_call_replay("call-interrupted", action)
            loop._record_tool_call_result("call-ledger", action, observation)

            replay = loop._tool_call_replay("call-ledger", action)
            with self.assertRaisesRegex(ProviderProtocolError, "reused with different arguments"):
                loop._tool_call_replay("call-ledger", _action("different"))
            mode = stat.S_IMODE(loop.tool_call_ledger_path.stat().st_mode)

        self.assertEqual(replay, observation.to_dict())
        self.assertTrue(interrupted["data"]["duplicate_execution_prevented"])
        self.assertEqual(mode, 0o600)

    def test_corrupt_tool_call_ledger_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Inspect", max_steps=1)
            loop._ensure_state_files()
            loop.tool_call_ledger_path.write_text("{broken", encoding="utf-8")

            with self.assertRaisesRegex(
                ProviderProtocolError,
                "prior execution outcome cannot be determined",
            ):
                loop._tool_call_replay("call-after-corruption", _action())

    def test_memory_retrieval_runs_once_without_refresh_event(self) -> None:
        class CountingRetriever:
            calls = 0

            def retrieve(self, query: str) -> RetrievedMemories:
                del query
                self.calls += 1
                return RetrievedMemories([], [], "none")

        class FixedDecisionMaker:
            uses_native_tools = False
            fatal_protocol_errors = False
            model = "fixed"
            last_token_usage = None

            def next_action(self, context: str, state: object) -> dict[str, object]:
                del context, state
                return _action()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Inspect", max_steps=2)
            loop._ensure_state_files()
            retriever = CountingRetriever()
            loop.memory_retriever = retriever
            loop.decision_maker = FixedDecisionMaker()
            state = create_initial_state("Inspect")
            loop._run_one_session(state)

        self.assertEqual(retriever.calls, 1)

    def test_task_transition_refreshes_memory_selection(self) -> None:
        class CountingRetriever:
            calls = 0

            def retrieve(self, query: str) -> RetrievedMemories:
                del query
                self.calls += 1
                return RetrievedMemories([], [], "none")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Build feature", max_steps=1)
            loop._ensure_state_files()
            retriever = CountingRetriever()
            loop.memory_retriever = retriever
            state = create_initial_state("Build feature")
            loop._run_one_session(state)

        self.assertEqual(state.last_action["action"], "update_plan")
        self.assertEqual(retriever.calls, 2)

    def test_new_worker_session_retrieves_memory_again(self) -> None:
        class CountingRetriever:
            calls = 0

            def retrieve(self, query: str) -> RetrievedMemories:
                del query
                self.calls += 1
                return RetrievedMemories([], [], f"selection-{self.calls}")

        class FixedDecisionMaker:
            uses_native_tools = False
            fatal_protocol_errors = False
            model = "fixed"
            last_token_usage = None

            def next_action(self, context: str, state: object) -> dict[str, object]:
                del context, state
                return _action()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Inspect", max_steps=1)
            loop._ensure_state_files()
            retriever = CountingRetriever()
            loop.memory_retriever = retriever
            loop.decision_maker = FixedDecisionMaker()
            state = create_initial_state("Inspect")
            first_session = loop._run_one_session(state)
            resumed_state = loop._prepare_auto_resume_session()
            second_session = loop._run_one_session(resumed_state)

        self.assertEqual(first_session.steps, 1)
        self.assertEqual(second_session.steps, 1)
        self.assertEqual(retriever.calls, 2)

    def test_successful_save_memory_triggers_refresh(self) -> None:
        class CountingRetriever:
            calls = 0

            def retrieve(self, query: str) -> RetrievedMemories:
                del query
                self.calls += 1
                return RetrievedMemories([], [], "none")

        class SaveMemoryDecisionMaker:
            uses_native_tools = False
            fatal_protocol_errors = False
            model = "fixed"
            last_token_usage = None

            def next_action(self, context: str, state: object) -> dict[str, object]:
                del context, state
                return {
                    "thought_summary": "Save durable feedback.",
                    "action": "save_memory",
                    "target": "real-db-tests",
                    "args": {
                        "name": "real-db-tests",
                        "description": "Integration tests use a real database",
                        "type": "feedback",
                        "content": "Integration tests must use a real database.",
                        "why": "Mocks missed migration failures.",
                        "how_to_apply": "Connect tests to the real test database.",
                    },
                    "expected_observation": "Memory is saved.",
                    "risk": "low",
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Remember preference", max_steps=1)
            loop._ensure_state_files()
            retriever = CountingRetriever()
            loop.memory_retriever = retriever
            loop.decision_maker = SaveMemoryDecisionMaker()
            state = create_initial_state("Remember preference")
            loop._run_one_session(state)

        self.assertTrue(state.last_observation["ok"])
        self.assertEqual(retriever.calls, 2)

    def test_skill_catalog_load_result_and_handoff_hash_rebuild(self) -> None:
        load_action = {
            "thought_summary": "Load the matching workflow.",
            "action": "load_skill",
            "target": "coding",
            "args": {},
            "expected_observation": "Skill content.",
            "risk": "low",
        }
        responses = [
            _native_response("call-load-skill", action=load_action),
            _native_response("call-after-skill"),
        ]
        captured_payloads: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Inspect", max_steps=2)
            loop._ensure_state_files()
            decision_maker = OpenAICompatibleDecisionMaker(
                api_key="test-key",
                base_url="https://example.invalid/v1",
                model="test-model",
            )
            decision_maker._post_chat_completions = lambda payload: (
                captured_payloads.append(json.loads(json.dumps(payload)))
                or responses.pop(0)
            )
            loop.decision_maker = decision_maker
            state = create_initial_state("Inspect")
            loop._run_one_session(state)

            rebuilt = loop.context_builder.build_session_context(state)
            state.loaded_skills[0]["content_hash"] = "stale-hash"
            invalidated = loop.context_builder.build_session_context(state)

        initial_context = captured_payloads[0]["messages"][-1]["content"]
        first_tool_message = captured_payloads[1]["messages"][-1]
        first_tool_payload = json.loads(first_tool_message["content"])
        self.assertIn("- coding:", initial_context)
        self.assertNotIn("## Skill: coding", initial_context)
        self.assertEqual(first_tool_message["role"], "tool")
        self.assertIn("# Instructions", first_tool_payload["observation"]["data"]["content"])
        self.assertEqual(
            sum(message["role"] == "tool" for message in captured_payloads[1]["messages"]),
            1,
        )
        self.assertIn("## Skill: coding", rebuilt)
        self.assertIn("# Instructions", rebuilt)
        self.assertNotIn("## Skill: coding", invalidated)
        self.assertIn("Invalidated Skills", invalidated)

    def test_pending_skill_reflection_is_in_tool_state_update(self) -> None:
        verify_action = {
            "thought_summary": "Verify the task.",
            "action": "verify",
            "target": "default",
            "args": {},
            "expected_observation": "Verifier result.",
            "risk": "low",
        }

        class NativeDecisionMaker:
            uses_native_tools = True
            fatal_protocol_errors = True
            model = "test-model"
            last_token_usage = None
            last_tool_call_id = "call-verify"

            def set_transcript_path(self, path: Path) -> None:
                del path

            def begin_session(self, context: str, conversation: list[dict[str, str]]) -> None:
                del context, conversation

            def next_action(self, context: str, state: object) -> dict[str, object]:
                del context, state
                self.last_tool_call_id = "call-verify"
                return verify_action

            def request_token_estimate(self) -> int:
                return 10

            def record_tool_result(
                self,
                observation: dict[str, object],
                state_update: dict[str, object],
            ) -> None:
                self.state_update = state_update

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Inspect", max_steps=1)
            loop._ensure_state_files()
            decision_maker = NativeDecisionMaker()
            loop.decision_maker = decision_maker
            loop._execute_action = lambda action, state: ToolResult(
                True,
                "Verifier passed.",
                {
                    "report_id": "VR-REFLECTION",
                    "archived_verifier_report": "state/verifier_reports/VR-REFLECTION.json",
                },
            )
            loop._apply_orchestrator_selection = lambda state: None
            state = create_initial_state("Inspect")
            state.task_session_ids["T1"] = [f"old-{index}" for index in range(6)]
            loop._run_one_session(state)

        self.assertEqual(
            decision_maker.state_update["pending_skill_review"]["report_id"],
            "VR-REFLECTION",
        )

    def test_incremental_state_contains_new_active_task_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Build", max_steps=1)
            state = create_initial_state("Build")
            state.task_id = "T2"
            state.user_goal = "T2: Implement persistence"
            state.acceptance_criteria = ["Focused persistence test passes."]
            state.nodes = [
                {
                    "id": "T2",
                    "title": "Implement persistence",
                    "status": "in_progress",
                    "expected_artifacts": ["app/storage.py"],
                    "evidence": [],
                }
            ]
            state.acceptance_contracts = [
                {
                    "task_id": "T2",
                    "summary": "Persistence contract",
                    "status": "agreed",
                }
            ]

            incremental = loop.context_builder.build_incremental_state(
                state,
                include_task_context=True,
            )

        self.assertEqual(incremental["task_id"], "T2")
        self.assertEqual(incremental["user_goal"], "T2: Implement persistence")
        self.assertEqual(
            incremental["active_task"]["expected_artifacts"],
            ["app/storage.py"],
        )
        self.assertEqual(
            incremental["active_acceptance_contracts"][0]["summary"],
            "Persistence contract",
        )

    def test_incremental_state_omits_repeated_large_task_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Build", max_steps=1)
            state = create_initial_state("Build")
            state.task_id = "T2"
            state.nodes = [
                {
                    "id": "T2",
                    "title": "Implement persistence",
                    "status": "in_progress",
                    "requirements": ["x" * 30_000],
                }
            ]
            state.acceptance_contracts = [
                {
                    "task_id": "T2",
                    "summary": "y" * 30_000,
                    "status": "agreed",
                }
            ]

            incremental = loop.context_builder.build_incremental_state(state)

        self.assertNotIn("active_task", incremental)
        self.assertNotIn("active_acceptance_contracts", incremental)
        self.assertNotIn("acceptance_criteria", incremental)
        self.assertLess(len(json.dumps(incremental)), 5_000)

    def test_handoff_does_not_copy_provider_reasoning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            emitted_events: list[dict[str, object]] = []
            loop = AgentLoop(
                root=root,
                task="Inspect",
                max_steps=1,
                conversation_messages=[
                    {"role": "user", "content": "Earlier question."},
                    {"role": "assistant", "content": "Earlier visible answer."},
                    {"role": "user", "content": "Current request."},
                ],
                event_handler=emitted_events.append,
            )
            decision_maker = OpenAICompatibleDecisionMaker(
                api_key="test-key",
                base_url="https://api.deepseek.com/v1",
                model="deepseek-chat",
            )
            captured_payload: dict[str, object] = {}

            def fake_post(payload: dict[str, object]) -> dict[str, object]:
                captured_payload.update(json.loads(json.dumps(payload)))
                return _direct_native_response(
                    "call-handoff",
                    reasoning="reasoning must remain provider-local",
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "provider_extension": {"cache_tier": "disk"},
                    },
                )

            decision_maker._post_chat_completions = fake_post
            loop.decision_maker = decision_maker
            loop.run()

            provider_transcript = loop.provider_session_path.read_text(encoding="utf-8")
            handoff = loop.handoff_path.read_text(encoding="utf-8")
            handoff_payload = loop.handoff_payload_path.read_text(encoding="utf-8")
            trace_text = loop.trace_path.read_text(encoding="utf-8")
            trace_event = loop._load_trace_events(loop.trace_path)[0]
            durable_state = json.loads(loop.state_path.read_text(encoding="utf-8"))
            memory_and_skill_text = "\n".join(
                path.read_text(encoding="utf-8")
                for directory in (loop.state_dir / "memories", loop.state_dir / "skills")
                for path in directory.rglob("*.md")
            )

        self.assertIn("reasoning must remain provider-local", provider_transcript)
        self.assertNotIn("reasoning must remain provider-local", handoff)
        self.assertNotIn("reasoning must remain provider-local", handoff_payload)
        self.assertNotIn("reasoning must remain provider-local", trace_text)
        self.assertNotIn(
            "reasoning must remain provider-local",
            json.dumps(durable_state),
        )
        self.assertNotIn("reasoning must remain provider-local", memory_and_skill_text)
        self.assertNotIn(
            "reasoning must remain provider-local",
            json.dumps(emitted_events),
        )
        self.assertGreater(trace_event["provider_session_ref"]["bytes"], 0)
        self.assertEqual(len(trace_event["provider_session_ref"]["sha256"]), 64)
        self.assertEqual(
            trace_event["provider_session_ref"]["tool_call_id"],
            "call-handoff",
        )
        self.assertEqual(
            durable_state["token_usage"]["turns"][0]["provider_usage"]["provider_extension"],
            {"cache_tier": "disk"},
        )
        self.assertEqual(
            [message["role"] for message in captured_payload["messages"]],
            ["system", "user", "assistant", "user", "user"],
        )
        self.assertNotIn("# User Conversation", captured_payload["messages"][-1]["content"])

    def test_auto_resume_resets_provider_session_and_uses_new_transcript(self) -> None:
        class ResettableDecisionMaker:
            reset_calls = 0

            def reset_session(self) -> None:
                self.reset_calls += 1

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Inspect", max_steps=1, resume=True)
            loop._ensure_state_files()
            state = create_initial_state("Inspect")
            loop._write_state(state)
            old_path = loop.provider_session_path
            decision_maker = ResettableDecisionMaker()
            loop.decision_maker = decision_maker

            loop._prepare_auto_resume_session()

        self.assertEqual(decision_maker.reset_calls, 1)
        self.assertNotEqual(loop.provider_session_path, old_path)

    def test_reset_session_drops_old_reasoning_and_tool_messages(self) -> None:
        decision_maker = OpenAICompatibleDecisionMaker(
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )
        decision_maker._post_chat_completions = lambda payload: _direct_native_response(
            "call-old-session",
            reasoning="old session private reasoning",
        )
        state = create_initial_state("Inspect")
        decision_maker.next_action("Old session context.", state)
        decision_maker.record_tool_result(
            {"ok": True, "summary": "Done.", "data": {}},
            {"task_id": "T1"},
        )

        decision_maker.reset_session()
        decision_maker.begin_session("New handoff-derived context.")
        new_payload = decision_maker._native_payload()
        serialized = json.dumps(new_payload)

        self.assertEqual(
            [message["role"] for message in new_payload["messages"]],
            ["system", "user"],
        )
        self.assertNotIn("old session private reasoning", serialized)
        self.assertNotIn("call-old-session", serialized)

    @unittest.skipUnless(
        os.environ.get("LONG_AGENT_RUN_LIVE_DEEPSEEK_TEST") == "1",
        "Set LONG_AGENT_RUN_LIVE_DEEPSEEK_TEST=1 for the optional live smoke test.",
    )
    def test_optional_live_deepseek_smoke(self) -> None:
        decision_maker = OpenAICompatibleDecisionMaker.from_env()
        action = decision_maker.next_action(
            (
                "This is a protocol smoke test. Call the declared list_files function once with "
                "target='.', args={}, expected_observation='Workspace entries.', and risk=low."
            ),
            create_initial_state("Protocol smoke test"),
        )

        self.assertEqual(action["action"], "list_files")


if __name__ == "__main__":
    unittest.main()
