"""Stable model routes exposed to the FlowPilot GUI and API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any

import flora_translate.config as cfg


@dataclass(frozen=True)
class ModelRoute:
    route_id: str
    label: str
    provider: str
    model: str
    base_url: str | None = None
    local: bool = False

    def public(self) -> dict:
        return asdict(self)


def model_routes() -> dict[str, ModelRoute]:
    routes = (
        ModelRoute(
            route_id="claude-opus-4-6",
            label="Claude Opus 4.6",
            provider="anthropic",
            model="claude-opus-4-6",
        ),
        ModelRoute(
            route_id="claude-sonnet-4-6",
            label="Claude Sonnet 4.6",
            provider="anthropic",
            model="claude-sonnet-4-6",
        ),
        ModelRoute(
            route_id="gpt-4o",
            label="GPT-4o",
            provider="openai",
            model="gpt-4o",
        ),
        ModelRoute(
            route_id="qwen3.6-27b",
            label="Qwen3.6-27B (local)",
            provider="ollama",
            model="/models/Qwen3.6-27B",
            base_url=os.getenv(
                "FLOWPILOT_QWEN36_BASE_URL",
                "http://127.0.0.1:8000/v1",
            ),
            local=True,
        ),
        ModelRoute(
            route_id="qwen3.8-27b",
            label="Qwen3.8-27B (local)",
            provider="ollama",
            model="/models/Qwen3.8-27B",
            base_url=os.getenv(
                "FLOWPILOT_QWEN38_BASE_URL",
                "http://127.0.0.1:8001/v1",
            ),
            local=True,
        ),
    )
    return {route.route_id: route for route in routes}


def route_for_id(route_id: str | None, *, role: str) -> ModelRoute:
    routes = model_routes()
    selected = route_id or default_route_ids()[role]
    if selected not in routes:
        raise ValueError(f"Unknown FlowPilot model route: {selected}")
    return routes[selected]


def default_route_ids() -> dict[str, str]:
    routes = model_routes()

    def match(model: str, fallback: str) -> str:
        for route_id, route in routes.items():
            if route.model == model:
                return route_id
        return fallback

    downstream_model = (
        cfg.ENGINE_MODEL_OPENAI
        if cfg.ENGINE_PROVIDER == "openai"
        else cfg.ENGINE_MODEL_ANTHROPIC
        if cfg.ENGINE_PROVIDER == "anthropic"
        else cfg.ENGINE_MODEL_OLLAMA
    )
    return {
        "upstream": match(cfg.MODEL_CHEMISTRY_AGENT, "claude-opus-4-6"),
        "downstream": match(downstream_model, "gpt-4o"),
    }


def runtime_model_options(
    upstream_route_id: str | None,
    downstream_route_id: str | None,
) -> dict:
    upstream = route_for_id(upstream_route_id, role="upstream")
    downstream = route_for_id(downstream_route_id, role="downstream")
    endpoints = {
        route.model: route.base_url
        for route in (upstream, downstream)
        if route.base_url
    }
    return {
        "upstream_model": upstream.model,
        "upstream_provider": upstream.provider,
        "downstream_model": downstream.model,
        "downstream_provider": downstream.provider,
        "model_endpoints": endpoints,
    }


def model_route_availability(route: ModelRoute, *, probe_local: bool = True) -> dict[str, Any]:
    """Report whether a model route can be used by the current process."""

    if route.provider == "openai":
        available = bool(os.getenv("OPENAI_API_KEY", "").strip())
        return {
            "available": available,
            "availability": "configured" if available else "unavailable",
            "reason": "" if available else "OPENAI_API_KEY is not configured.",
        }
    if route.provider == "anthropic":
        available = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
        return {
            "available": available,
            "availability": "configured" if available else "unavailable",
            "reason": "" if available else "ANTHROPIC_API_KEY is not configured.",
        }
    if not route.base_url:
        return {
            "available": False,
            "availability": "unavailable",
            "reason": "Local model endpoint is not configured.",
        }
    if not probe_local:
        return {"available": True, "availability": "configured", "reason": ""}
    try:
        import httpx

        response = httpx.get(
            f"{route.base_url.rstrip('/')}/models",
            timeout=3.0,
            trust_env=False,
        )
        response.raise_for_status()
        model_ids = {
            str(item.get("id"))
            for item in response.json().get("data", [])
            if isinstance(item, dict)
        }
        available = route.model in model_ids
        return {
            "available": available,
            "availability": "reachable" if available else "unavailable",
            "reason": "" if available else f"Endpoint does not advertise {route.model}.",
        }
    except Exception as exc:
        return {
            "available": False,
            "availability": "unavailable",
            "reason": f"Endpoint is unreachable: {str(exc)[:160]}",
        }


def model_route_statuses(*, probe_local: bool = True) -> dict[str, dict[str, Any]]:
    """Return availability metadata for every declared GUI route."""

    return {
        route_id: model_route_availability(route, probe_local=probe_local)
        for route_id, route in model_routes().items()
    }


def available_default_route_ids(statuses: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Choose usable defaults without silently selecting a dead route."""

    configured = default_route_ids()

    def choose(role: str, fallbacks: tuple[str, ...]) -> str:
        preferred = configured[role]
        if statuses.get(preferred, {}).get("available"):
            return preferred
        for route_id in fallbacks:
            if statuses.get(route_id, {}).get("available"):
                return route_id
        return preferred

    return {
        "upstream": choose(
            "upstream",
            ("claude-opus-4-6", "claude-sonnet-4-6", "qwen3.6-27b", "qwen3.8-27b"),
        ),
        "downstream": choose(
            "downstream",
            ("claude-sonnet-4-6", "qwen3.6-27b", "claude-opus-4-6", "qwen3.8-27b"),
        ),
    }
