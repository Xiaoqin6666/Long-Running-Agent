from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


UI_CONTRACT_VERSION = 1
UI_CONTRACT_REQUIRED_FIELDS = (
    "entry_points",
    "buttons",
    "inputs",
    "dialogs",
    "data_display",
    "empty_state",
    "success_refresh",
)
UI_CONTRACT_APPLICABILITY_VALUES = ("required", "indirect", "not_applicable")
UI_CONTRACT_SURFACE_VALUES = ("widget", "dialog", "report", "none")

UI_CONTRACT_GENERATOR_PROMPT = """You generate verifier-owned UI contracts from project requirements.

Return JSON only. Do not include Markdown fences or commentary.

Required top-level shape:
{
  "kind": "ui_contract",
  "version": 1,
  "generated_by": "llm",
  "contracts": [
    {
      "requirement_id": "same id as input requirement",
      "source": "same source as input requirement",
      "priority": "same priority as input requirement",
      "type": "same type as input requirement",
      "requirement_text": "same text as input requirement",
      "ui_applicability": "required | indirect | not_applicable",
      "ui_surface": "widget | dialog | report | none",
      "ui_contract": {
        "entry_points": ["where the user starts this workflow in the UI"],
        "buttons": ["required visible button/menu command/action control labels"],
        "inputs": ["required input fields, selectors, upload controls, sliders, or 'No dedicated input required; ...'"],
        "dialogs": ["required dialogs/modals/confirmations, or 'No modal required; inline feedback is acceptable'"],
        "data_display": ["where resulting data/status/errors must be visible"],
        "empty_state": "what the UI displays when prerequisite data or result data is empty",
        "success_refresh": "how the UI visibly updates after a successful operation"
      }
    }
  ]
}

Rules:
- Produce exactly one contract for every input requirement.
- The UI contract is a user-interface contract, not a test plan and not code.
- Set ui_applicability='required' only when this requirement has direct user-facing UI that must be checked against the fields below.
- Set ui_applicability='indirect' when the requirement is business/service/persistence behavior that is triggered or surfaced through UI, but should not require every widget category below.
- Set ui_applicability='not_applicable' when the requirement has no meaningful UI surface; use ui_surface='none'.
- Make labels and controls concrete enough for a GUI verifier or human reviewer to check.
- If a requirement is mainly service logic or persistence, still describe the user-visible surface or explicitly state that no dedicated UI is required and where status/errors are surfaced.
- Every required field must be present and non-empty.
- Do not invent product features beyond what the requirement implies.
"""


def build_ui_contract(
    requirements: list[dict[str, Any]],
    *,
    provider: str = "offline",
) -> dict[str, Any]:
    requirement_items = _contract_requirements(requirements)
    if provider == "offline":
        return _build_offline_contract(requirement_items)
    if provider == "openai-compatible":
        contract = OpenAICompatibleUIContractBuilder.from_env().build(requirement_items)
        errors = validate_ui_contract({"requirements": requirement_items}, contract)
        if errors:
            raise RuntimeError("LLM-generated UI Contract failed schema validation: " + "; ".join(errors))
        return contract
    raise ValueError(f"Unsupported provider: {provider}")


def write_ui_contract(
    path: Path,
    requirements: list[dict[str, Any]],
    *,
    provider: str = "offline",
) -> dict[str, Any]:
    contract = build_ui_contract(requirements, provider=provider)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return contract


def validate_ui_contract(requirements_data: dict[str, Any], contract_data: dict[str, Any]) -> list[str]:
    requirements = requirements_data.get("requirements", []) if isinstance(requirements_data, dict) else []
    if not isinstance(requirements, list):
        return ["UI Contract validation requires requirements.json requirements to be a list."]
    if not isinstance(contract_data, dict):
        return ["UI Contract must be a JSON object."]
    errors: list[str] = []
    if contract_data.get("kind") != "ui_contract":
        errors.append("UI Contract kind must be 'ui_contract'.")
    if contract_data.get("version") != UI_CONTRACT_VERSION:
        errors.append(f"UI Contract version must be {UI_CONTRACT_VERSION}.")
    raw_contracts = contract_data.get("contracts", [])
    if not isinstance(raw_contracts, list):
        return errors + ["UI Contract contracts must be a list."]
    contract_by_id = {
        str(item.get("requirement_id", "")).strip(): item
        for item in raw_contracts
        if isinstance(item, dict) and str(item.get("requirement_id", "")).strip()
    }
    for requirement in _contract_requirements(requirements):
        requirement_id = str(requirement.get("id", "")).strip()
        contract = contract_by_id.get(requirement_id)
        if not contract:
            errors.append(f"UI Contract missing requirement {requirement_id}.")
            continue
        errors.extend(_validate_contract_entry(requirement_id, contract))
    extra_ids = sorted(set(contract_by_id) - {str(item.get("id", "")).strip() for item in _contract_requirements(requirements)})
    if extra_ids:
        errors.append(f"UI Contract includes unknown requirement ids: {', '.join(extra_ids)}.")
    return errors


@dataclass
class OpenAICompatibleUIContractBuilder:
    api_key: str
    base_url: str
    model: str
    timeout: int = 60
    temperature: float = 0.1

    @classmethod
    def from_env(cls) -> "OpenAICompatibleUIContractBuilder":
        api_key = os.environ.get("LONG_AGENT_API_KEY")
        if not api_key:
            raise RuntimeError("Missing LONG_AGENT_API_KEY.")
        return cls(
            api_key=api_key,
            base_url=os.environ.get("LONG_AGENT_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            model=os.environ.get("LONG_AGENT_UI_CONTRACT_MODEL", os.environ.get("LONG_AGENT_MODEL", "gpt-4.1-mini")),
            timeout=int(os.environ.get("LONG_AGENT_UI_CONTRACT_TIMEOUT", os.environ.get("LONG_AGENT_TIMEOUT", "60"))),
            temperature=float(
                os.environ.get("LONG_AGENT_UI_CONTRACT_TEMPERATURE", os.environ.get("LONG_AGENT_TEMPERATURE", "0.1"))
            ),
        )

    def build(self, requirements: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": UI_CONTRACT_GENERATOR_PROMPT},
                {"role": "user", "content": self._render_user_content(requirements)},
            ],
        }
        response = self._post_chat_completions(payload)
        content = str(response["choices"][0]["message"]["content"]).strip()
        parsed = _loads_json_object(_strip_markdown_fence(content))
        if not isinstance(parsed, dict):
            extracted = _extract_first_json_object(content)
            parsed = _loads_json_object(extracted) if extracted else None
        if not isinstance(parsed, dict):
            raise RuntimeError(f"LLM did not return a JSON object for UI Contract: {content[:500]}")
        return _normalize_contract_payload(parsed, requirements, generated_by="llm")

    def _render_user_content(self, requirements: list[dict[str, Any]]) -> str:
        return json.dumps({"requirements": requirements}, ensure_ascii=False, indent=2)

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


def _contract_requirements(requirements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(requirement)
        for requirement in requirements
        if isinstance(requirement, dict) and str(requirement.get("id", "")).strip()
    ]


def _validate_contract_entry(requirement_id: str, contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    ui_contract = contract.get("ui_contract")
    if not isinstance(ui_contract, dict):
        return [f"UI Contract for {requirement_id} must include ui_contract object."]
    applicability = str(contract.get("ui_applicability", "")).strip().lower()
    if applicability and applicability not in UI_CONTRACT_APPLICABILITY_VALUES:
        errors.append(
            f"UI Contract for {requirement_id}.ui_applicability must be one of: "
            + ", ".join(UI_CONTRACT_APPLICABILITY_VALUES)
        )
    surface = str(contract.get("ui_surface", "")).strip().lower()
    if surface and surface not in UI_CONTRACT_SURFACE_VALUES:
        errors.append(
            f"UI Contract for {requirement_id}.ui_surface must be one of: "
            + ", ".join(UI_CONTRACT_SURFACE_VALUES)
        )
    for field in UI_CONTRACT_REQUIRED_FIELDS:
        value = ui_contract.get(field)
        if isinstance(value, list):
            if not any(str(item).strip() for item in value):
                errors.append(f"UI Contract for {requirement_id}.{field} must be non-empty.")
        elif not str(value or "").strip():
            errors.append(f"UI Contract for {requirement_id}.{field} must be non-empty.")
    return errors


def _normalize_contract_payload(
    payload: dict[str, Any],
    requirements: list[dict[str, Any]],
    *,
    generated_by: str,
) -> dict[str, Any]:
    raw_contracts = payload.get("contracts", [])
    if not isinstance(raw_contracts, list):
        raw_contracts = []
    raw_by_id = {
        str(contract.get("requirement_id", "")).strip(): contract
        for contract in raw_contracts
        if isinstance(contract, dict) and str(contract.get("requirement_id", "")).strip()
    }
    contracts = []
    for requirement in _contract_requirements(requirements):
        requirement_id = str(requirement.get("id", "")).strip()
        raw = raw_by_id.get(requirement_id, {})
        ui_contract = raw.get("ui_contract", {}) if isinstance(raw, dict) else {}
        raw_applicability = raw.get("ui_applicability") if isinstance(raw, dict) else None
        raw_surface = raw.get("ui_surface") if isinstance(raw, dict) else None
        contracts.append(
            {
                "requirement_id": requirement_id,
                "source": str(requirement.get("source", "")).strip(),
                "priority": str(requirement.get("priority", "")).strip(),
                "type": str(requirement.get("type", "")).strip(),
                "requirement_text": str(requirement.get("text", "")).strip(),
                "ui_applicability": _normalize_ui_applicability(raw_applicability, requirement),
                "ui_surface": _normalize_ui_surface(raw_surface, requirement),
                "ui_contract": _normalize_ui_contract(ui_contract if isinstance(ui_contract, dict) else {}),
            }
        )
    return {
        "kind": "ui_contract",
        "version": UI_CONTRACT_VERSION,
        "generated_by": generated_by,
        "contracts": contracts,
    }


def _normalize_ui_contract(ui_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "entry_points": _coerce_string_list(ui_contract.get("entry_points")),
        "buttons": _coerce_string_list(ui_contract.get("buttons")),
        "inputs": _coerce_string_list(ui_contract.get("inputs")),
        "dialogs": _coerce_string_list(ui_contract.get("dialogs")),
        "data_display": _coerce_string_list(ui_contract.get("data_display")),
        "empty_state": str(ui_contract.get("empty_state", "")).strip(),
        "success_refresh": str(ui_contract.get("success_refresh", "")).strip(),
    }


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _build_offline_contract(requirements: list[dict[str, Any]]) -> dict[str, Any]:
    contracts = []
    for requirement in requirements:
        text = _requirement_text(requirement)
        contracts.append(
            {
                "requirement_id": str(requirement.get("id", "")).strip(),
                "source": str(requirement.get("source", "")).strip(),
                "priority": str(requirement.get("priority", "")).strip(),
                "type": str(requirement.get("type", "")).strip(),
                "requirement_text": str(requirement.get("text", "")).strip(),
                "ui_applicability": _default_ui_applicability(requirement),
                "ui_surface": _default_ui_surface(requirement),
                "ui_contract": _offline_ui_contract(text),
            }
        )
    return {
        "kind": "ui_contract",
        "version": UI_CONTRACT_VERSION,
        "generated_by": "offline",
        "contracts": contracts,
    }


def _offline_ui_contract(text: str) -> dict[str, Any]:
    lowered = text.lower()
    is_employee_add = ("employee" in lowered or "员工" in text) and any(marker in text for marker in ("添加", "新增", "录入"))
    if is_employee_add:
        return {
            "entry_points": ["Employee management panel or tab"],
            "buttons": ["Add"],
            "inputs": ["Employee name input", "Skills selector", "Weekly capacity input"],
            "dialogs": ["No modal dialog required; inline UI feedback is acceptable"],
            "data_display": ["Employee list/detail view"],
            "empty_state": "When no employees exist, the UI shows an explicit empty employee list state.",
            "success_refresh": "After success, the employee list/detail view refreshes immediately.",
        }
    return {
        "entry_points": ["Primary application window or relevant feature panel"],
        "buttons": ["Feature action control"],
        "inputs": ["Feature-specific input control or no dedicated input required"],
        "dialogs": ["No modal dialog required; inline UI feedback is acceptable"],
        "data_display": ["Relevant state, result, or error feedback area"],
        "empty_state": "When prerequisite data is absent, the UI shows a stable empty state instead of stale data or an exception.",
        "success_refresh": "After success, the visible UI reflects the new state without requiring an application restart.",
    }


def _normalize_ui_applicability(value: object, requirement: dict[str, Any]) -> str:
    text = str(value or "").strip().lower()
    if text in UI_CONTRACT_APPLICABILITY_VALUES:
        return text
    return _default_ui_applicability(requirement)


def _normalize_ui_surface(value: object, requirement: dict[str, Any]) -> str:
    text = str(value or "").strip().lower()
    if text in UI_CONTRACT_SURFACE_VALUES:
        return text
    return _default_ui_surface(requirement)


def _default_ui_applicability(requirement: dict[str, Any]) -> str:
    requirement_type = str(requirement.get("type", "")).strip().lower()
    text = _requirement_text(requirement).lower()
    if requirement_type == "gui_workflow":
        return "required"
    if requirement_type == "report":
        return "required"
    if requirement_type == "persistence":
        if any(marker in text for marker in ("export", "import", "导出", "导入", "瀵煎嚭", "瀵煎叆")):
            return "required"
        return "indirect"
    if any(marker in text for marker in ("button", "dialog", "form", "screen", "gui", "ui", "按钮", "弹窗", "界面")):
        return "indirect"
    return "not_applicable"


def _default_ui_surface(requirement: dict[str, Any]) -> str:
    applicability = _default_ui_applicability(requirement)
    if applicability == "not_applicable":
        return "none"
    requirement_type = str(requirement.get("type", "")).strip().lower()
    if requirement_type == "report":
        return "report"
    if requirement_type == "persistence":
        return "dialog" if applicability == "required" else "none"
    return "widget"


def _requirement_text(requirement: dict[str, Any]) -> str:
    return " ".join(
        str(requirement.get(key, "")).strip()
        for key in ("id", "text", "type", "acceptance_intent")
        if str(requirement.get(key, "")).strip()
    )


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
        if match:
            return match.group(1).strip()
    return stripped


def _loads_json_object(text: str | None) -> Any | None:
    if text is None:
        return None
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
