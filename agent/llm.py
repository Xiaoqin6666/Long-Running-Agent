from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from agent.planner import TaskState
from agent.prompts import MAIN_AGENT_SYSTEM_PROMPT


ALLOWED_ACTIONS = {
    "answer",
    "bash",
    "contract",
    "edit",
    "git",
    "list_files",
    "read",
    "skill",
    "write",
    "search",
    "update_plan",
    "verify",
    "finish",
}


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

    def next_action(self, context: str, state: TaskState) -> dict[str, Any]:
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
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

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
        )

    def next_action(self, context: str, state: TaskState) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": MAIN_AGENT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": context,
                },
            ],
        }
        response = self._post_chat_completions(payload)
        content = response["choices"][0]["message"]["content"]
        action = parse_action_json(content)
        return validate_action(action, state)

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
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc


def parse_action_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    data = _loads_json_object(text)
    if data is None:
        extracted = _extract_first_json_object(text)
        if extracted is not None:
            data = _loads_json_object(extracted)
    if data is None:
        raise ValueError(f"Model did not return valid JSON: {text[:500]}")
    if not isinstance(data, dict):
        raise ValueError("Model action must be a JSON object.")
    return data


def _loads_json_object(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return text[start : start + end]


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
