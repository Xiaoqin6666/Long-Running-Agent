from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


SPEC_BUILDER_PROMPT = """You convert a user's project conversation into a durable project_spec.md.

Return Markdown only. Do not return JSON. Do not include code fences.

The specification must be concrete enough for an autonomous coding agent to:
- generate a task graph,
- implement the project,
- write tests,
- run verification,
- know where generated application artifacts should live.

Preserve explicit paths, constraints, and user preferences from the conversation.
"""


def build_project_spec(
    provider: str,
    messages: list[dict[str, str]],
) -> str:
    if provider == "offline":
        return _deterministic_project_spec(messages)
    if provider == "openai-compatible":
        return OpenAICompatibleSpecBuilder.from_env().build(messages)
    raise ValueError(f"Unsupported provider: {provider}")


def _deterministic_project_spec(messages: list[dict[str, str]]) -> str:
    user_messages = [
        str(message.get("content", "")).strip()
        for message in messages
        if str(message.get("role", "")).strip().lower() == "user" and str(message.get("content", "")).strip()
    ]
    lines = ["# Project Specification", ""]
    if user_messages:
        lines.extend(["## Conversation Requirements", ""])
        lines.extend(f"- {message}" for message in user_messages)
        lines.append("")
    lines.extend(
        [
            "## Implementation Expectations",
            "",
            "- Generate a concrete task graph before implementation.",
            "- Implement the requested behavior in the configured workspace.",
            "- Add tests or verification commands that prove the requested behavior.",
            "- Keep generated artifacts inside the intended project or benchmark workspace.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


@dataclass
class OpenAICompatibleSpecBuilder:
    api_key: str
    base_url: str
    model: str
    timeout: int = 60
    temperature: float = 0.1

    @classmethod
    def from_env(cls) -> "OpenAICompatibleSpecBuilder":
        api_key = os.environ.get("LONG_AGENT_API_KEY")
        if not api_key:
            raise RuntimeError("Missing LONG_AGENT_API_KEY.")
        return cls(
            api_key=api_key,
            base_url=os.environ.get("LONG_AGENT_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            model=os.environ.get("LONG_AGENT_MODEL", "gpt-4.1-mini"),
            timeout=int(os.environ.get("LONG_AGENT_TIMEOUT", "60")),
            temperature=float(os.environ.get("LONG_AGENT_TEMPERATURE", "0.1")),
        )

    def build(self, messages: list[dict[str, str]]) -> str:
        content = self._render_user_content(messages)
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": SPEC_BUILDER_PROMPT},
                {"role": "user", "content": content},
            ],
        }
        response = self._post_chat_completions(payload)
        spec = str(response["choices"][0]["message"]["content"]).strip()
        return _strip_markdown_fence(spec) + "\n"

    def _render_user_content(self, messages: list[dict[str, str]]) -> str:
        conversation = "\n\n".join(
            f"{str(message.get('role', '')).strip().lower()}: {str(message.get('content', '')).strip()}"
            for message in messages
            if str(message.get("role", "")).strip().lower() in {"user", "assistant"}
            and str(message.get("content", "")).strip()
        )
        sections = [
            "# Project Conversation",
            conversation or "No conversation messages were provided.",
        ]
        return "\n\n".join(sections)

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


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped
