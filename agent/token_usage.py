from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


TOKEN_USAGE_SCHEMA = "long-agent.token-usage.v1"


def estimate_tokens(value: object) -> int:
    text = str(value or "")
    if not text:
        return 0
    return max(1, len(text) // 4)


def normalize_token_usage(raw_usage: object, source: str = "api") -> dict[str, Any] | None:
    if not isinstance(raw_usage, dict):
        return None

    input_tokens = _optional_int(raw_usage.get("input_tokens", raw_usage.get("prompt_tokens")))
    output_tokens = _optional_int(raw_usage.get("output_tokens", raw_usage.get("completion_tokens")))
    total_tokens = _optional_int(raw_usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    normalized = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "source": source,
    }
    api_cost = extract_api_cost(raw_usage)
    if api_cost:
        normalized["cost"] = api_cost
    return normalized


def normalize_response_usage(response: object, source: str = "api") -> dict[str, Any] | None:
    if not isinstance(response, dict):
        return None
    usage = normalize_token_usage(response.get("usage"), source=source)
    if not usage:
        return None
    api_cost = extract_api_cost(response.get("usage")) or extract_api_cost(response)
    if api_cost:
        usage["cost"] = api_cost
    return usage


def estimate_turn_usage(system_prompt: str, context: str, action: dict[str, Any]) -> dict[str, Any]:
    input_tokens = estimate_tokens(system_prompt) + estimate_tokens(context)
    output_tokens = estimate_tokens(json.dumps(action, ensure_ascii=False))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "source": "estimated",
    }


def initialize_token_usage(existing: object | None = None) -> dict[str, Any]:
    if isinstance(existing, dict):
        usage = dict(existing)
    else:
        usage = {}
    usage.setdefault("schema", TOKEN_USAGE_SCHEMA)
    usage.setdefault("totals", _empty_totals())
    usage.setdefault("sessions", {})
    usage.setdefault("turns", [])
    if not isinstance(usage["totals"], dict):
        usage["totals"] = _empty_totals()
    for key, value in _empty_totals().items():
        usage["totals"].setdefault(key, value)
    if not isinstance(usage["sessions"], dict):
        usage["sessions"] = {}
    if not isinstance(usage["turns"], list):
        usage["turns"] = []
    return usage


def record_turn_usage(
    token_usage: dict[str, Any],
    *,
    session_id: str,
    step: int,
    task_id: str,
    provider: str,
    model: str,
    operation_type: str,
    usage: dict[str, Any],
    pricing: dict[str, Any] | None = None,
    recorded_at: str,
) -> dict[str, Any]:
    normalized = initialize_token_usage(token_usage)
    input_tokens = _int_or_zero(usage.get("input_tokens"))
    output_tokens = _int_or_zero(usage.get("output_tokens"))
    record = {
        "session_id": session_id,
        "step": step,
        "task_id": task_id,
        "provider": provider,
        "model": model,
        "operation_type": operation_type,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": _total_or_sum(usage),
        "source": str(usage.get("source", "unknown")),
        "cost": _usage_cost_or_calculated(usage, model, input_tokens, output_tokens, pricing or {}),
        "recorded_at": recorded_at,
    }
    normalized["turns"].append(record)

    _add_to_totals(normalized["totals"], record, recorded_at)
    sessions = normalized["sessions"]
    session_totals = sessions.setdefault(
        session_id,
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "turn_count": 0,
            "started_at": recorded_at,
            "updated_at": recorded_at,
        },
    )
    _add_to_totals(session_totals, record, recorded_at)
    return record


def _empty_totals() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "turn_count": 0,
        "costs_by_currency": {},
        "unpriced_turn_count": 0,
    }


def _add_to_totals(totals: dict[str, Any], record: dict[str, Any], recorded_at: str) -> None:
    totals["input_tokens"] = _int_or_zero(totals.get("input_tokens")) + record["input_tokens"]
    totals["output_tokens"] = _int_or_zero(totals.get("output_tokens")) + record["output_tokens"]
    totals["total_tokens"] = _int_or_zero(totals.get("total_tokens")) + record["total_tokens"]
    totals["turn_count"] = _int_or_zero(totals.get("turn_count")) + 1
    cost = record.get("cost", {})
    if isinstance(cost, dict) and cost.get("available"):
        currency = str(cost.get("currency", "USD") or "USD")
        costs_by_currency = totals.setdefault("costs_by_currency", {})
        if not isinstance(costs_by_currency, dict):
            costs_by_currency = {}
            totals["costs_by_currency"] = costs_by_currency
        currency_totals = costs_by_currency.setdefault(
            currency,
            {
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
                "turn_count": 0,
            },
        )
        currency_totals["input_cost"] = _round_cost(
            float(currency_totals.get("input_cost", 0.0) or 0.0) + float(cost.get("input_cost", 0.0) or 0.0)
        )
        currency_totals["output_cost"] = _round_cost(
            float(currency_totals.get("output_cost", 0.0) or 0.0) + float(cost.get("output_cost", 0.0) or 0.0)
        )
        currency_totals["total_cost"] = _round_cost(
            float(currency_totals.get("total_cost", 0.0) or 0.0) + float(cost.get("total_cost", 0.0) or 0.0)
        )
        currency_totals["turn_count"] = _int_or_zero(currency_totals.get("turn_count")) + 1
    else:
        totals["unpriced_turn_count"] = _int_or_zero(totals.get("unpriced_turn_count")) + 1
    totals["updated_at"] = recorded_at


def load_pricing_from_env() -> dict[str, Any]:
    raw_json = os.environ.get("LONG_AGENT_TOKEN_PRICES_JSON", "").strip()
    if raw_json:
        return normalize_pricing_table(_loads_json_dict(raw_json), source="LONG_AGENT_TOKEN_PRICES_JSON")

    raw_path = os.environ.get("LONG_AGENT_TOKEN_PRICES_FILE", "").strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        try:
            return normalize_pricing_table(
                _loads_json_dict(path.read_text(encoding="utf-8")),
                source=f"file:{path}",
            )
        except OSError:
            return {}

    return {}


def normalize_pricing_table(raw: object, source: str = "configured") -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, Any] = {}
    for model, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        model_name = str(model).strip()
        if not model_name:
            continue
        input_per_1m = _optional_float(
            entry.get("input_per_1m", entry.get("input_cost_per_1m", entry.get("prompt_per_1m")))
        )
        output_per_1m = _optional_float(
            entry.get("output_per_1m", entry.get("output_cost_per_1m", entry.get("completion_per_1m")))
        )
        if input_per_1m is None or output_per_1m is None:
            continue
        normalized[model_name] = {
            "input_per_1m": input_per_1m,
            "output_per_1m": output_per_1m,
            "currency": str(entry.get("currency", "USD") or "USD"),
            "source": str(entry.get("source", source) or source),
        }
    return normalized


def calculate_usage_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: dict[str, Any],
) -> dict[str, Any]:
    price = pricing.get(model) if isinstance(pricing, dict) else None
    if not isinstance(price, dict):
        return {
            "available": False,
            "reason": "missing_model_price",
        }

    input_per_1m = _optional_float(price.get("input_per_1m"))
    output_per_1m = _optional_float(price.get("output_per_1m"))
    if input_per_1m is None or output_per_1m is None:
        return {
            "available": False,
            "reason": "invalid_model_price",
        }

    input_cost = input_tokens / 1_000_000 * input_per_1m
    output_cost = output_tokens / 1_000_000 * output_per_1m
    return {
        "available": True,
        "currency": str(price.get("currency", "USD") or "USD"),
        "input_cost": _round_cost(input_cost),
        "output_cost": _round_cost(output_cost),
        "total_cost": _round_cost(input_cost + output_cost),
        "input_per_1m": input_per_1m,
        "output_per_1m": output_per_1m,
        "price_source": str(price.get("source", "configured") or "configured"),
    }


def _usage_cost_or_calculated(
    usage: dict[str, Any],
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: dict[str, Any],
) -> dict[str, Any]:
    cost = usage.get("cost")
    if isinstance(cost, dict) and cost.get("available"):
        return cost
    return calculate_usage_cost(model, input_tokens, output_tokens, pricing)


def extract_api_cost(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    for key in ("cost", "cost_details", "billing", "billing_details"):
        nested = raw.get(key)
        if isinstance(nested, dict):
            cost = _cost_from_mapping(nested)
            if cost:
                return cost
        elif key == "cost":
            total = _optional_float(nested)
            if total is not None:
                return _api_cost_record(total_cost=total, raw=raw)

    cost = _cost_from_mapping(raw)
    if cost:
        return cost
    return None


def _cost_from_mapping(raw: dict[str, Any]) -> dict[str, Any] | None:
    input_cost = _optional_float(raw.get("input_cost", raw.get("prompt_cost")))
    output_cost = _optional_float(raw.get("output_cost", raw.get("completion_cost")))
    total_cost = _optional_float(
        raw.get(
            "total_cost",
            raw.get("cost", raw.get("billed_cost", raw.get("amount", raw.get("estimated_cost")))),
        )
    )
    if total_cost is None and input_cost is not None and output_cost is not None:
        total_cost = input_cost + output_cost
    if input_cost is None and output_cost is None and total_cost is None:
        return None
    return _api_cost_record(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost,
        raw=raw,
    )


def _api_cost_record(
    *,
    raw: dict[str, Any],
    input_cost: float | None = None,
    output_cost: float | None = None,
    total_cost: float | None = None,
) -> dict[str, Any]:
    return {
        "available": True,
        "currency": str(raw.get("currency", "USD") or "USD"),
        "input_cost": _round_cost(input_cost or 0.0),
        "output_cost": _round_cost(output_cost or 0.0),
        "total_cost": _round_cost(total_cost or 0.0),
        "price_source": "api",
    }


def _total_or_sum(usage: dict[str, Any]) -> int:
    total = _optional_int(usage.get("total_tokens"))
    if total is not None:
        return total
    return _int_or_zero(usage.get("input_tokens")) + _int_or_zero(usage.get("output_tokens"))


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: object) -> int:
    parsed = _optional_int(value)
    return parsed if parsed is not None else 0


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _round_cost(value: float) -> float:
    return round(value, 12)


def _loads_json_dict(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
