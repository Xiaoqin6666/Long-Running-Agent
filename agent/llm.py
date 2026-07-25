from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from agent.planner import TaskState
from agent.prompts import MAIN_AGENT_SYSTEM_PROMPT
from agent.token_usage import normalize_response_usage


ACTION_TOOL_NAME = "submit_action"
ACTION_ARGUMENT_FIELDS = {
    "thought_summary",
    "action",
    "target",
    "args",
    "expected_observation",
    "risk",
}

ALLOWED_ACTIONS = {
    "answer",
    "bash",
    "contract",
    "edit",
    "git",
    "list_files",
    "read",
    "recall_memory",
    "load_skill",
    "save_memory",
    "save_skill",
    "dismiss_skill",
    "skill",
    "write",
    "search",
    "update_plan",
    "verify",
    "finish",
}

ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": ACTION_TOOL_NAME,
        "description": (
            "Submit exactly one harness action. The harness validates and executes it, then returns "
            "the observation as a tool message."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "thought_summary": {
                    "type": "string",
                    "description": "Brief visible reasoning summary; never include hidden chain of thought.",
                },
                "action": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
                "target": {"type": "string"},
                "args": {"type": "object"},
                "expected_observation": {"type": "string"},
                "risk": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": [
                "thought_summary",
                "action",
                "target",
                "args",
                "expected_observation",
                "risk",
            ],
            "additionalProperties": False,
        },
    },
}

ACTION_DESCRIPTIONS = {
    "answer": "Return the final evidence-based answer in args.answer.",
    "bash": "Run one command from the repository root. Put the command in target.",
    "contract": (
        "Create or update the active task acceptance contract. For a generated task whose contract is "
        "pending_verification_procedure, keep target as the active task id and set args to "
        '{"verification_procedure":{"working_directory":"<workspace-relative directory>",'
        '"commands":["python -m unittest discover -s tests"]}}. Every verification command must start '
        "with python, python3, python.exe, or py followed by -c or -m; direct script commands such as "
        "'python3 tests/test_ui.py' are rejected. Update only the verification procedure; do not return "
        "to INIT or recreate the task graph."
    ),
    "dismiss_skill": "Dismiss the pending Skill Reflection with a reason in args.",
    "edit": "Apply one bounded edit to target using the replacement fields in args.",
    "finish": "Request project-level completion after all verification gates pass.",
    "git": "Run one allowed git status, diff, log, show, branch, add, or commit operation.",
    "list_files": "List target directory entries; configure recursion and limit in args.",
    "load_skill": "Load one Skill named by target.",
    "read": "Read target file content; use query or line bounds in args.",
    "recall_memory": "Answer the natural-language question in target from the complete typed-Memory corpus.",
    "save_memory": "Save one validated typed Memory using fields in args.",
    "save_skill": "Save one evidence-backed reusable Skill using fields in args.",
    "search": "Search for target text under the path specified in args.",
    "skill": "Compatibility alias for saving one evidence-backed reusable Skill.",
    "update_plan": "Update the harness plan for target.",
    "verify": "Run the independent verifier for the active task.",
    "write": "Create, overwrite, or append target with content and mode in args.",
}


def _direct_action_tool(action_name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": action_name,
            "description": ACTION_DESCRIPTIONS[action_name],
            "parameters": {
                "type": "object",
                "properties": {
                    "thought_summary": {
                        "type": "string",
                        "description": "Brief visible reasoning summary; never include hidden chain of thought.",
                    },
                    "target": {"type": "string"},
                    "args": {"type": "object"},
                    "expected_observation": {"type": "string"},
                    "risk": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": [
                    "thought_summary",
                    "target",
                    "args",
                    "expected_observation",
                    "risk",
                ],
                "additionalProperties": False,
            },
        },
    }


DIRECT_ACTION_TOOLS = [
    _direct_action_tool(action_name) for action_name in sorted(ALLOWED_ACTIONS)
]


class ProviderRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class ProviderProtocolError(RuntimeError):
    pass


def create_decision_maker(provider: str):
    if provider == "offline":
        return OfflineDecisionMaker()
    if provider == "openai-compatible":
        return OpenAICompatibleDecisionMaker.from_env()
    raise ValueError(f"Unsupported provider: {provider}")


class OfflineDecisionMaker:
    """Deterministic decision maker for local smoke tests.

    This is a stand-in for an LLM provider. It exercises the same harness
    interface without network access or API keys.
    """

    last_token_usage: dict[str, Any] | None = None
    uses_native_tools = False
    fatal_protocol_errors = False
    model = "offline"

    def next_action(self, context: str, state: TaskState) -> dict[str, Any]:
        self.last_token_usage = None
        del context
        if state.iterations == 0:
            return {
                "thought_summary": "Create the initial explicit plan.",
                "action": "update_plan",
                "target": "current_task",
                "args": {},
                "expected_observation": "Plan state is initialized.",
                "risk": "low",
            }
        if state.iterations == 1:
            return {
                "thought_summary": "Inspect the README as a bounded observation.",
                "action": "read",
                "target": "README.md",
                "args": {"start": 1, "end": 120},
                "expected_observation": "Read project overview.",
                "risk": "low",
            }
        if state.iterations == 2:
            return {
                "thought_summary": "Run independent verifier.",
                "action": "verify",
                "target": "default",
                "args": {},
                "expected_observation": "Verifier reports whether state and trace exist.",
                "risk": "low",
            }
        return {
            "thought_summary": "Attempt finish after verification.",
            "action": "finish",
            "target": "current_task",
            "args": {},
            "expected_observation": "Harness accepts finish only if checks pass.",
            "risk": "low",
        }


class OpenAICompatibleDecisionMaker:
    """Decision maker for OpenAI-compatible chat completions APIs."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 60,
        temperature: float = 0.1,
        thinking: bool | None = None,
        reasoning_effort: str = "high",
        max_attempts: int = 3,
        context_window_tokens: int = 128_000,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.thinking = self._is_deepseek() if thinking is None else bool(thinking)
        self.reasoning_effort = self._normalize_reasoning_effort(reasoning_effort)
        self.max_attempts = min(3, max(1, max_attempts))
        self.context_window_tokens = max(1, context_window_tokens)
        self.uses_native_tools = True
        self.fatal_protocol_errors = True
        self.last_token_usage: dict[str, Any] | None = None
        self.last_tool_call_id = ""
        self.last_request_payload: dict[str, Any] = {}
        self.last_assistant_message: dict[str, Any] = {}
        self.messages: list[dict[str, Any]] = []
        self._session_started = False
        self._transcript_path: Path | None = None
        self._transcript_hasher = hashlib.sha256()
        self._transcript_bytes = 0

    @classmethod
    def from_env(cls) -> "OpenAICompatibleDecisionMaker":
        api_key = os.environ.get("LONG_AGENT_API_KEY")
        if not api_key:
            raise RuntimeError("Missing LONG_AGENT_API_KEY.")
        return cls(
            api_key=api_key,
            base_url=os.environ.get("LONG_AGENT_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("LONG_AGENT_MODEL", "gpt-4.1-mini"),
            timeout=int(os.environ.get("LONG_AGENT_TIMEOUT", "60")),
            temperature=float(os.environ.get("LONG_AGENT_TEMPERATURE", "0.1")),
            thinking=_thinking_from_env(os.environ.get("LONG_AGENT_THINKING", "auto")),
            reasoning_effort=os.environ.get("LONG_AGENT_REASONING_EFFORT", "high"),
            max_attempts=int(os.environ.get("LONG_AGENT_PROVIDER_MAX_ATTEMPTS", "3")),
            context_window_tokens=int(os.environ.get("LONG_AGENT_CONTEXT_WINDOW_TOKENS", "128000")),
        )

    def set_transcript_path(self, path: Path | None) -> None:
        self._transcript_path = path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.parent.chmod(0o700)
        except OSError:
            pass
        if not path.exists():
            path.touch(mode=0o600)
        try:
            path.chmod(0o600)
        except OSError:
            pass
        self._transcript_hasher = hashlib.sha256()
        self._transcript_bytes = 0
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    self._transcript_hasher.update(chunk)
                    self._transcript_bytes += len(chunk)
        except OSError:
            pass

    def begin_session(self, context: str, conversation_messages: list[dict[str, str]] | None = None) -> None:
        self.last_tool_call_id = ""
        self.last_assistant_message = {}
        self.messages = []
        self._session_started = True
        provider_tool_instruction = (
            "For this provider, each allowed action is exposed as its own function. "
            "Call exactly one of those declared functions and do not call submit_action."
            if self._uses_direct_action_tools()
            else (
                "For this provider, submit_action is the only declared function. "
                "Call submit_action exactly once."
            )
        )
        self.messages.append(
            {
                "role": "system",
                "content": f"{MAIN_AGENT_SYSTEM_PROMPT} {provider_tool_instruction}",
            }
        )
        for message in conversation_messages or []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                self.messages.append({"role": role, "content": content})
        self.messages.append({"role": "user", "content": context})
        self._append_transcript(
            {
                "type": "session_start",
                "model": self.model,
                "message_protocol": "native-tools",
                "thinking": self.thinking,
            }
        )

    def reset_session(self) -> None:
        self.messages = []
        self._session_started = False
        self.last_tool_call_id = ""
        self.last_assistant_message = {}

    def next_action(self, context: str, state: TaskState) -> dict[str, Any]:
        self.last_token_usage = None
        self.last_tool_call_id = ""
        if not self._session_started:
            self.begin_session(context, state.conversation_messages)
        response_usages: list[dict[str, Any]] = []
        for protocol_reprompt in range(3):
            payload = self._native_payload()
            estimated_tokens = self.estimate_payload_tokens(payload)
            if estimated_tokens >= self.context_window_tokens:
                raise ProviderRequestError(
                    (
                        "Provider request preflight exceeded the configured context window "
                        f"({estimated_tokens} >= {self.context_window_tokens} estimated tokens)."
                    ),
                    retryable=False,
                )
            self.last_request_payload = _json_copy(payload)
            self._append_transcript({"type": "request", "payload": self.last_request_payload})
            try:
                response = self._post_chat_completions(payload)
            except Exception as exc:
                error_event: dict[str, Any] = {
                    "type": "provider_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                if isinstance(exc, ProviderRequestError):
                    error_event.update(
                        {
                            "status_code": exc.status_code,
                            "retryable": exc.retryable,
                        }
                    )
                self._append_transcript(error_event)
                self.last_token_usage = _merge_response_usages(response_usages)
                raise
            self._append_transcript({"type": "response", "response": response})
            response_usage = normalize_response_usage(response, source="api")
            if response_usage:
                response_usages.append(response_usage)
            self.last_token_usage = _merge_response_usages(response_usages)
            message = self._response_message(response)
            try:
                action = validate_action(self._parse_native_action(message), state)
            except ProviderProtocolError:
                if self._uses_direct_action_tools() and protocol_reprompt < 2:
                    parallel_tool_results = self._parallel_tool_rejection_messages(
                        message
                    )
                    if parallel_tool_results:
                        assistant_message = _assistant_message_for_roundtrip(message)
                        self.messages.append(_json_copy(assistant_message))
                        for tool_message in parallel_tool_results:
                            self.messages.append(tool_message)
                            self._append_transcript(
                                {"type": "tool_result", "message": tool_message}
                            )
                        self._append_transcript(
                            {
                                "type": "parallel_tool_calls_reprompt",
                                "attempt": protocol_reprompt + 1,
                                "tool_call_count": len(parallel_tool_results),
                                "assistant_message": assistant_message,
                            }
                        )
                        continue
                if (
                    self._uses_direct_action_tools()
                    and protocol_reprompt < 2
                    and self._is_content_only_message(message)
                ):
                    assistant_message = _assistant_message_for_roundtrip(message)
                    self.messages.append(_json_copy(assistant_message))
                    correction = {
                        "role": "user",
                        "content": (
                            "Harness protocol correction: the previous response did not call a tool, "
                            "so no action was executed. Continue by calling exactly one function declared "
                            "in the API tools. Do not return another content-only response."
                        ),
                    }
                    self.messages.append(correction)
                    self._append_transcript(
                        {
                            "type": "content_only_reprompt",
                            "attempt": protocol_reprompt + 1,
                            "assistant_message": assistant_message,
                            "correction": correction,
                        }
                    )
                    continue
                if self._uses_direct_action_tools() and protocol_reprompt < 2:
                    invalid_arguments_result = (
                        self._invalid_tool_arguments_rejection_message(message)
                    )
                    if invalid_arguments_result:
                        assistant_message = _assistant_message_for_roundtrip(message)
                        self.messages.append(_json_copy(assistant_message))
                        self.messages.append(invalid_arguments_result)
                        self._append_transcript(
                            {
                                "type": "tool_result",
                                "message": invalid_arguments_result,
                            }
                        )
                        self._append_transcript(
                            {
                                "type": "invalid_tool_arguments_reprompt",
                                "attempt": protocol_reprompt + 1,
                                "assistant_message": assistant_message,
                            }
                        )
                        continue
                raise
            except ValueError as exc:
                raise ProviderProtocolError(f"Native action validation failed: {exc}") from exc
            self.last_assistant_message = _assistant_message_for_roundtrip(message)
            self.messages.append(_json_copy(self.last_assistant_message))
            return action
        raise ProviderProtocolError(
            "Provider did not produce a native tool call after content-only correction."
        )

    def record_tool_result(self, observation: dict[str, Any], state_update: dict[str, Any]) -> None:
        if not self.last_tool_call_id:
            raise ProviderProtocolError("Cannot append a tool result without a matching tool_call id.")
        content = json.dumps(
            {
                "observation": observation,
                "state_update": state_update,
            },
            ensure_ascii=False,
        )
        message = {
            "role": "tool",
            "tool_call_id": self.last_tool_call_id,
            "content": content,
        }
        self.messages.append(message)
        self._append_transcript({"type": "tool_result", "message": message})

    def request_token_estimate(self) -> int:
        if not self._session_started:
            return 0
        return self.estimate_payload_tokens(self._native_payload())

    @staticmethod
    def estimate_payload_tokens(payload: dict[str, Any]) -> int:
        return max(1, len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))) // 4)

    def _native_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _json_copy(self.messages),
            "tools": _json_copy(self._native_tools()),
        }
        # DeepSeek thinking rejects every tested tool_choice value. Expose the
        # existing harness actions as native functions and validate the selected
        # function locally. Other providers retain the single forced wrapper.
        if not self._uses_direct_action_tools():
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": ACTION_TOOL_NAME},
            }
        if self.thinking:
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = self.reasoning_effort
        else:
            payload["temperature"] = self.temperature
            if self._is_deepseek():
                payload["thinking"] = {"type": "disabled"}
        return payload

    def _native_tools(self) -> list[dict[str, Any]]:
        if self._uses_direct_action_tools():
            return DIRECT_ACTION_TOOLS
        return [ACTION_TOOL]

    def _uses_direct_action_tools(self) -> bool:
        return self.thinking and self._is_deepseek()

    @staticmethod
    def _is_content_only_message(message: dict[str, Any]) -> bool:
        tool_calls = message.get("tool_calls")
        content = message.get("content")
        no_tool_calls = (
            tool_calls is None
            or tool_calls == ()
            or isinstance(tool_calls, list)
            and not tool_calls
        )
        return no_tool_calls and isinstance(content, str) and bool(content.strip())

    @staticmethod
    def _parallel_tool_rejection_messages(
        message: dict[str, Any],
    ) -> list[dict[str, Any]]:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) <= 1:
            return []
        call_ids: list[str] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                return []
            call_id = tool_call.get("id")
            if not isinstance(call_id, str) or not call_id.strip():
                return []
            call_ids.append(call_id)
        if len(call_ids) != len(set(call_ids)):
            return []
        summary = (
            "Parallel tool-call batch rejected without execution: this harness "
            "updates and verifies durable state after each action. Call exactly "
            "one declared function in the next response."
        )
        return [
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(
                    {
                        "observation": {
                            "ok": False,
                            "summary": summary,
                            "data": {
                                "parallel_tool_calls_rejected": True,
                                "tool_call_count": len(call_ids),
                                "counts_as_progress": False,
                            },
                        },
                        "state_update": {
                            "required_next_action": (
                                "Call exactly one declared function. The other "
                                "actions can be requested after its tool result."
                            )
                        },
                    },
                    ensure_ascii=False,
                ),
            }
            for call_id in call_ids
        ]

    def _invalid_tool_arguments_rejection_message(
        self,
        message: dict[str, Any],
    ) -> dict[str, Any] | None:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            return None
        tool_call = tool_calls[0]
        if not isinstance(tool_call, dict) or tool_call.get("type") != "function":
            return None
        call_id = tool_call.get("id")
        function = tool_call.get("function")
        if not isinstance(call_id, str) or not call_id.strip() or not isinstance(function, dict):
            return None
        function_name = function.get("name")
        declared_tool_names = {
            str(tool.get("function", {}).get("name", ""))
            for tool in self._native_tools()
        }
        if not isinstance(function_name, str) or function_name not in declared_tool_names:
            return None
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            return None
        try:
            json.loads(arguments)
        except json.JSONDecodeError as exc:
            json_error = str(exc)
        else:
            return None
        summary = (
            "Tool call rejected without execution: function.arguments was not valid JSON. "
            "Call the function again with a complete JSON object."
        )
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(
                {
                    "observation": {
                        "ok": False,
                        "summary": summary,
                        "data": {
                            "invalid_tool_arguments_json": True,
                            "function_name": function_name,
                            "json_error": json_error,
                            "counts_as_progress": False,
                        },
                    },
                    "state_update": {
                        "required_next_action": (
                            "Call exactly one declared function again with valid JSON arguments. "
                            "The rejected call was not executed. Escape every quote, backslash, "
                            "and newline inside string fields such as args.content."
                        )
                    },
                },
                ensure_ascii=False,
            ),
        }

    def _response_message(self, response: dict[str, Any]) -> dict[str, Any]:
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderProtocolError("Provider response is missing choices[0].message.") from exc
        if not isinstance(message, dict):
            raise ProviderProtocolError("Provider response message must be an object.")
        return message

    def _parse_native_action(self, message: dict[str, Any]) -> dict[str, Any]:
        if message.get("role") != "assistant":
            raise ProviderProtocolError(
                "Provider response message role must be assistant."
            )
        for field in ("content", "reasoning_content"):
            value = message.get(field)
            if field in message and value is not None and not isinstance(value, str):
                raise ProviderProtocolError(
                    f"Provider response {field} must be a string or null."
                )
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            count = len(tool_calls) if isinstance(tool_calls, list) else 0
            raise ProviderProtocolError(f"Expected exactly one native tool call, received {count}.")
        tool_call = tool_calls[0]
        if not isinstance(tool_call, dict):
            raise ProviderProtocolError("Native tool call must be an object.")
        if tool_call.get("type") != "function":
            raise ProviderProtocolError("Native tool call type must be function.")
        call_id = str(tool_call.get("id", "")).strip()
        function = tool_call.get("function")
        if not call_id or not isinstance(function, dict):
            raise ProviderProtocolError("Native tool call is missing id or function.")
        function_name = str(function.get("name", ""))
        declared_tool_names = {
            str(tool.get("function", {}).get("name", ""))
            for tool in self._native_tools()
        }
        if function_name not in declared_tool_names:
            raise ProviderProtocolError(f"Unsupported native tool: {function_name}")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise ProviderProtocolError("Native tool arguments must be a JSON string.")
        try:
            decoded_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ProviderProtocolError(f"Native tool arguments are invalid JSON: {exc}") from exc
        if not isinstance(decoded_arguments, dict):
            raise ProviderProtocolError("Native tool arguments must decode to an object.")
        action = (
            decoded_arguments
            if function_name == ACTION_TOOL_NAME
            else self._direct_action_from_arguments(
                function_name,
                decoded_arguments,
                message,
            )
        )
        missing = sorted(ACTION_ARGUMENT_FIELDS - set(action))
        extra = sorted(set(action) - ACTION_ARGUMENT_FIELDS)
        if missing:
            raise ProviderProtocolError(
                f"Native tool arguments are missing required fields: {', '.join(missing)}."
            )
        if extra:
            raise ProviderProtocolError(
                f"Native tool arguments contain unsupported fields: {', '.join(extra)}."
            )
        string_fields = ACTION_ARGUMENT_FIELDS - {"args"}
        invalid_strings = sorted(
            field for field in string_fields if not isinstance(action.get(field), str)
        )
        if invalid_strings:
            raise ProviderProtocolError(
                "Native tool argument fields must be strings: "
                + ", ".join(invalid_strings)
                + "."
            )
        if not isinstance(action.get("args"), dict):
            raise ProviderProtocolError("Native tool args must be an object.")
        if action.get("action") not in ALLOWED_ACTIONS:
            raise ProviderProtocolError(
                f"Unsupported native action: {action.get('action')}"
            )
        if action.get("risk") not in {"low", "medium", "high"}:
            raise ProviderProtocolError(
                "Native tool risk must be low, medium, or high."
            )
        self.last_tool_call_id = call_id
        return action

    def _direct_action_from_arguments(
        self,
        function_name: str,
        arguments: dict[str, Any],
        message: dict[str, Any],
    ) -> dict[str, Any]:
        allowed_fields = ACTION_ARGUMENT_FIELDS - {"action"}
        extra = sorted(set(arguments) - allowed_fields)
        if extra:
            raise ProviderProtocolError(
                "Native action tool arguments contain unsupported fields: "
                + ", ".join(extra)
                + "."
            )
        content = message.get("content")
        thought_summary = arguments.get("thought_summary")
        if not isinstance(thought_summary, str) or not thought_summary.strip():
            thought_summary = (
                content
                if isinstance(content, str) and content.strip()
                else f"Use the {function_name} action."
            )
        target = arguments.get("target", "")
        expected_observation = arguments.get("expected_observation", "")
        risk = arguments.get("risk", "medium")
        args = arguments.get("args", {})
        for field, value in (
            ("target", target),
            ("expected_observation", expected_observation),
            ("risk", risk),
        ):
            if not isinstance(value, str):
                raise ProviderProtocolError(
                    f"Native action tool {field} must be a string."
                )
        if not isinstance(args, dict):
            raise ProviderProtocolError("Native action tool args must be an object.")
        return {
            "thought_summary": thought_summary,
            "action": function_name,
            "target": target,
            "args": args,
            "expected_observation": expected_observation,
            "risk": risk,
        }

    def _is_deepseek(self) -> bool:
        marker = f"{self.base_url} {self.model}".lower()
        return "deepseek" in marker

    @staticmethod
    def _normalize_reasoning_effort(value: str) -> str:
        normalized = str(value or "high").strip().lower()
        if normalized == "xhigh":
            return "max"
        if normalized in {"low", "medium"}:
            return "high"
        return normalized if normalized in {"high", "max"} else "high"

    def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(1, self.max_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                retry_after = exc.headers.get("Retry-After")
                exc.close()
                retryable = exc.code in {408, 429} or exc.code >= 500
                error = ProviderRequestError(
                    f"API HTTP {exc.code}: {body}",
                    status_code=exc.code,
                    retryable=retryable,
                )
                if not retryable or attempt >= self.max_attempts:
                    raise error from exc
                time.sleep(_retry_delay_seconds(attempt, retry_after))
            except (urllib.error.URLError, TimeoutError) as exc:
                error = ProviderRequestError(f"API request failed: {exc}", retryable=True)
                if attempt >= self.max_attempts:
                    raise error from exc
                time.sleep(_retry_delay_seconds(attempt))
        raise ProviderRequestError("API request failed after retries.", retryable=True)

    def _append_transcript(self, event: dict[str, Any]) -> None:
        if self._transcript_path is None:
            return
        encoded = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        with self._transcript_path.open("ab") as handle:
            handle.write(encoded)
        self._transcript_hasher.update(encoded)
        self._transcript_bytes += len(encoded)

    def transcript_metadata(self) -> dict[str, Any]:
        return {
            "bytes": self._transcript_bytes,
            "sha256": self._transcript_hasher.hexdigest() if self._transcript_bytes else "",
        }


def configure_auxiliary_payload(
    payload: dict[str, Any],
    *,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    """Keep single-shot helper calls out of DeepSeek thinking mode."""
    marker = f"{base_url} {model}".lower()
    if "deepseek" in marker:
        payload["thinking"] = {"type": "disabled"}
    return payload


def _thinking_from_env(raw: str) -> bool | None:
    value = str(raw or "auto").strip().lower()
    if value in {"auto", ""}:
        return None
    if value in {"1", "true", "yes", "on", "enabled"}:
        return True
    if value in {"0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError("LONG_AGENT_THINKING must be auto, enabled, or disabled.")


def _retry_delay_seconds(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return min(60.0, max(0.0, float(retry_after)))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return min(
                    60.0,
                    max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds()),
                )
            except (TypeError, ValueError, OverflowError):
                pass
    return min(60.0, float(2 ** (attempt - 1)))


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _assistant_message_for_roundtrip(message: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"role": "assistant"}
    for key in ("content", "reasoning_content", "tool_calls"):
        if key in message:
            result[key] = _json_copy(message[key])
    return result


def _merge_response_usages(usages: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not usages:
        return None
    if len(usages) == 1:
        return _json_copy(usages[0])
    numeric_fields = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "cache_hit_tokens",
        "cache_miss_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    )
    merged: dict[str, Any] = {"source": "api"}
    for field in numeric_fields:
        values = [usage.get(field) for usage in usages if usage.get(field) is not None]
        if values:
            merged[field] = sum(int(value or 0) for value in values)
    provider_usages = [
        _json_copy(usage["provider_usage"])
        for usage in usages
        if isinstance(usage.get("provider_usage"), dict)
    ]
    if provider_usages:
        merged["provider_usage"] = {
            "request_count": len(provider_usages),
            "requests": provider_usages,
        }
    costs = [
        usage["cost"]
        for usage in usages
        if isinstance(usage.get("cost"), dict)
        and usage["cost"].get("available") is True
    ]
    currencies = {str(cost.get("currency", "USD")) for cost in costs}
    if len(costs) == len(usages) and len(currencies) == 1:
        merged["cost"] = {
            "available": True,
            "currency": currencies.pop(),
            "input_cost": sum(float(cost.get("input_cost", 0.0) or 0.0) for cost in costs),
            "output_cost": sum(float(cost.get("output_cost", 0.0) or 0.0) for cost in costs),
            "total_cost": sum(float(cost.get("total_cost", 0.0) or 0.0) for cost in costs),
            "price_source": "api",
        }
    return merged


def validate_action(action: dict[str, Any], state: TaskState) -> dict[str, Any]:
    del state
    raw_args = action.get("args", {})
    normalized = {
        "thought_summary": str(action.get("thought_summary", ""))[:1000],
        "action": str(action.get("action", "")),
        "target": str(action.get("target", "")),
        "args": normalize_args(str(action.get("action", "")), raw_args),
        "expected_observation": str(action.get("expected_observation", ""))[:1000],
        "risk": str(action.get("risk", "medium")),
    }
    if normalized["action"] not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported model action: {normalized['action']}")
    if normalized["risk"] not in {"low", "medium", "high"}:
        normalized["risk"] = "medium"
    return normalized


def normalize_args(action_name: str, raw_args: Any) -> dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        if action_name == "bash":
            return {"command": raw_args}
        if action_name == "read":
            return {"note": raw_args}
        if action_name == "search":
            return {"path": ".", "note": raw_args}
        return {"value": raw_args}
    if isinstance(raw_args, list):
        return {"items": raw_args}
    return {"value": str(raw_args)}
