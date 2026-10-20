"""
FLORA ENGINE — LLM provider abstraction.

Single entry point for all council LLM calls.
Provider is configured via ENGINE_PROVIDER in flora_translate/config.py:

    ENGINE_PROVIDER = "anthropic"   →  Claude Sonnet (ENGINE_MODEL_ANTHROPIC)
    ENGINE_PROVIDER = "openai"      →  GPT-4o        (ENGINE_MODEL_OPENAI)

To switch: change ENGINE_PROVIDER in config.py. That is the only change needed.
"""

from __future__ import annotations

import json
import logging
import math
import re
import ssl
import time
from dataclasses import dataclass

import anthropic

from flora_translate.config import (
    ENGINE_PROVIDER,
    ENGINE_MODEL_ANTHROPIC,
    ENGINE_MODEL_OPENAI,
    ENGINE_MODEL_OLLAMA,
    OLLAMA_BASE_URL,
    ENGINE_MAX_ROUNDS,
)

logger = logging.getLogger("flora.engine.llm_agents")

_ANTHROPIC_CLIENT = None
_OPENAI_CLIENT    = None
_OLLAMA_CLIENT    = None
_OPENAI_COMPAT_CLIENTS: dict[str, object] = {}
_LLM_OBSERVER     = None
_RUNTIME_OVERRIDES: dict = {}
_MODEL_ENDPOINT_OVERRIDES: dict[str, str] = {}


@dataclass
class TextGenerationResult:
    text: str
    provider: str
    model: str
    usage: dict
    stop_reason: str | None = None
    finish_reason: str | None = None


def set_llm_observer(observer) -> None:
    """Register an optional observer(event_dict) for benchmark telemetry."""
    global _LLM_OBSERVER
    _LLM_OBSERVER = observer


def clear_llm_observer() -> None:
    global _LLM_OBSERVER
    _LLM_OBSERVER = None


def set_llm_runtime_overrides(**kwargs) -> None:
    """Set optional runtime overrides such as temperature or seed."""
    global _RUNTIME_OVERRIDES
    _RUNTIME_OVERRIDES = {k: v for k, v in kwargs.items() if v is not None}


def clear_llm_runtime_overrides() -> None:
    global _RUNTIME_OVERRIDES
    _RUNTIME_OVERRIDES = {}


def get_llm_runtime_overrides() -> dict:
    return dict(_RUNTIME_OVERRIDES)


def set_model_endpoint_overrides(overrides: dict[str, str] | None) -> None:
    """Route selected local/OpenAI-compatible models to their own endpoints."""
    global _MODEL_ENDPOINT_OVERRIDES
    _MODEL_ENDPOINT_OVERRIDES = {
        str(model): str(base_url).rstrip("/")
        for model, base_url in (overrides or {}).items()
        if model and base_url
    }


def get_model_endpoint_overrides() -> dict[str, str]:
    return dict(_MODEL_ENDPOINT_OVERRIDES)


def _uses_responses_api(model: str) -> bool:
    name = (model or "").lower()
    return name == "gpt-6-astra" or name.startswith("gpt-6-astra-")


def _supports_explicit_temperature(model: str) -> bool:
    """Return whether models in the supported publication set accept temperature."""
    return not _uses_responses_api(model)


def _runtime_kwargs(provider: str, model: str) -> dict:
    """Translate benchmark sampling controls into provider-compatible kwargs."""
    kwargs: dict = {}
    if (
        "temperature" in _RUNTIME_OVERRIDES
        and _supports_explicit_temperature(model)
    ):
        kwargs["temperature"] = _RUNTIME_OVERRIDES["temperature"]
    if provider in {"openai", "ollama"} and "seed" in _RUNTIME_OVERRIDES and not _uses_responses_api(model):
        kwargs["seed"] = _RUNTIME_OVERRIDES["seed"]
    return kwargs


def _openai_token_limit(model: str, max_tokens: int) -> dict:
    """Use the Chat Completions token-limit parameter for GPT-4o."""
    return {"max_tokens": max_tokens}


def _openai_tool_kwargs(model: str) -> dict:
    """Return optional GPT-4o Chat Completions tool controls."""
    return {}


def _anthropic_text(content: list) -> str:
    """Extract final text while ignoring adaptive-thinking and tool blocks."""
    return "\n".join(
        block.text.strip()
        for block in content
        if getattr(block, "type", "") == "text"
        and isinstance(getattr(block, "text", None), str)
        and block.text.strip()
    ).strip()


def _emit_llm_event(event: dict) -> None:
    if _LLM_OBSERVER is None:
        return
    try:
        _LLM_OBSERVER(event)
    except Exception as exc:
        logger.debug("LLM observer raised: %s", exc)


def _usage_to_dict(resp) -> dict:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}

    def _get(name: str):
        value = getattr(usage, name, None)
        return value if isinstance(value, (int, float)) else None

    data = {
        "input_tokens": _get("input_tokens"),
        "output_tokens": _get("output_tokens"),
        "prompt_tokens": _get("prompt_tokens"),
        "completion_tokens": _get("completion_tokens"),
        "total_tokens": _get("total_tokens"),
        "cache_creation_input_tokens": _get("cache_creation_input_tokens"),
        "cache_read_input_tokens": _get("cache_read_input_tokens"),
    }
    for name in ("input_tokens_details", "output_tokens_details"):
        details = getattr(usage, name, None)
        if details is not None:
            data[name] = details.model_dump() if hasattr(details, "model_dump") else dict(details)
    return {k: v for k, v in data.items() if v is not None}


def _base_event(
    *,
    api_name: str,
    provider: str,
    model: str,
    max_tokens: int,
    system: str,
    user_content: str,
) -> dict:
    event = {
        "api_name": api_name,
        "provider": provider,
        "model": model,
        "max_tokens": max_tokens,
        "system_chars": len(system or ""),
        "user_chars": len(user_content or ""),
    }
    if "temperature" in _RUNTIME_OVERRIDES and _supports_explicit_temperature(model):
        event["temperature"] = _RUNTIME_OVERRIDES["temperature"]
    if "seed" in _RUNTIME_OVERRIDES and not _uses_responses_api(model):
        event["seed"] = _RUNTIME_OVERRIDES["seed"]
    if _uses_responses_api(model):
        event["reasoning_effort"] = _RUNTIME_OVERRIDES.get("reasoning_effort", "medium")
        event["sampling_controls_applied"] = False
    if _RUNTIME_OVERRIDES.get("capture_content"):
        event["system_prompt"] = system
        event["user_prompt"] = user_content
    return event


def _content_event(text: str | None, raw_text: str | None = None) -> dict:
    """Return raw model content only for explicitly instrumented benchmark runs."""
    if not _RUNTIME_OVERRIDES.get("capture_content"):
        return {}
    event = {"response_text": text or ""}
    if raw_text is not None and raw_text != text:
        event["raw_response_text"] = raw_text
    return event


def _strip_thinking(text: str) -> str:
    """Remove local-model reasoning blocks while preserving the final answer."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.IGNORECASE)
    return cleaned.strip()


def emit_component_llm_event(
    *,
    component: str,
    provider: str,
    model: str,
    max_tokens: int,
    system: str,
    user_content: str,
    resp,
    started: float,
    extra: dict | None = None,
) -> None:
    event = {
        **_base_event(
            api_name=component,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            system=system,
            user_content=user_content,
        ),
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "usage": _usage_to_dict(resp),
    }
    if extra:
        event.update(extra)
    _emit_llm_event(event)


def _get_anthropic_client() -> anthropic.Anthropic:
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        _ANTHROPIC_CLIENT = anthropic.Anthropic(
            http_client=build_verified_httpx_client(),
        )
    return _ANTHROPIC_CLIENT


def build_verified_httpx_client():
    """Build a verified client using the host's configured system trust store."""
    import httpx

    # The laboratory network adds a trusted TLS inspection certificate to the
    # host CA store. Forcing certifi bypasses that trust configuration and
    # produces CERTIFICATE_VERIFY_FAILED despite a valid system-trusted chain.
    ssl_context = ssl.create_default_context()
    # Preserve certificate and hostname verification. Some enterprise proxy
    # chains omit AKI, which Python 3.13's optional strict flag rejects.
    strict_flag = getattr(ssl, "VERIFY_X509_STRICT", 0)
    if strict_flag:
        ssl_context.verify_flags &= ~strict_flag
    return httpx.Client(verify=ssl_context)


def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        import openai
        _OPENAI_CLIENT = openai.OpenAI(http_client=build_verified_httpx_client())
    return _OPENAI_CLIENT


def _get_ollama_client(model: str | None = None):
    """OpenAI-compatible client for the selected local model endpoint."""
    global _OLLAMA_CLIENT
    base_url = _MODEL_ENDPOINT_OVERRIDES.get(str(model or ""), OLLAMA_BASE_URL).rstrip("/")
    if base_url == OLLAMA_BASE_URL.rstrip("/") and _OLLAMA_CLIENT is not None:
        return _OLLAMA_CLIENT
    if base_url in _OPENAI_COMPAT_CLIENTS:
        return _OPENAI_COMPAT_CLIENTS[base_url]
    if _OLLAMA_CLIENT is None or base_url != OLLAMA_BASE_URL.rstrip("/"):
        import httpx
        import openai
        client = openai.OpenAI(
            base_url=base_url,
            api_key="ollama",          # required by the library, ignored by Ollama
            # The laboratory endpoint is plain HTTP. An explicit transport keeps
            # httpx from loading a host SSL_CERT_FILE that may not exist inside
            # the execution environment and is irrelevant for this connection.
            http_client=httpx.Client(verify=False, trust_env=False),
        )
        _OPENAI_COMPAT_CLIENTS[base_url] = client
        if base_url == OLLAMA_BASE_URL.rstrip("/"):
            _OLLAMA_CLIENT = client
        return client
    return _OLLAMA_CLIENT


def _retry_llm_request(fn, *, provider: str, model: str, api_name: str, attempts: int = 3):
    """Retry transient provider failures with useful context for the GUI."""

    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            retryable = (
                "connection" in msg.lower()
                or "timeout" in msg.lower()
                or "temporarily" in msg.lower()
                or "rate" in msg.lower()
                or "overloaded" in msg.lower()
                or exc.__class__.__name__.lower() in {
                    "apiconnectionerror",
                    "apitimeouterror",
                    "ratelimiterror",
                    "overloadederror",
                }
            )
            if not retryable or attempt == attempts:
                break
            delay_s = min(2 ** (attempt - 1), 8)
            logger.warning(
                "LLM request failed during %s (%s/%s), retry %d/%d in %ss: %s",
                api_name,
                provider,
                model,
                attempt + 1,
                attempts,
                delay_s,
                exc,
            )
            time.sleep(delay_s)
    raise RuntimeError(
        f"LLM request failed during {api_name} ({provider}/{model}) "
        f"after {attempts} attempts: {last_exc}"
    ) from last_exc


def get_max_rounds() -> int:
    """Return the configured number of council rounds for the active provider."""
    return ENGINE_MAX_ROUNDS.get(ENGINE_PROVIDER, 2)


def infer_provider_for_model(model: str, provider: str | None = None) -> str:
    """Infer provider from model name unless explicitly given."""
    if provider:
        return provider
    name = (model or "").strip().lower()
    if name.startswith("claude"):
        return "anthropic"
    if (
        name.startswith("gpt-")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
        or name.startswith("chatgpt-")
    ):
        return "openai"
    return "ollama"


def _stringify_messages(messages: list[dict]) -> str:
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.append(json.dumps(content, default=str))
    return "\n".join(parts)


def _responses_request(*, model, system, inputs, max_tokens, api_name, json_schema=None, **extra):
    # Reasoning consumes the same output budget as visible text. Preserve ample
    # headroom for both architectures without changing legacy model requests.
    limit = max(max_tokens, 16384)
    effort = _RUNTIME_OVERRIDES.get("reasoning_effort", "medium")
    if effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError(f"Unsupported Astra reasoning effort: {effort}")
    kwargs = {
        "model": model,
        "instructions": system,
        "input": inputs,
        "max_output_tokens": limit,
        "reasoning": {"effort": effort},
        "store": False,
        # Carry encrypted reasoning across tool turns without server storage.
        "include": ["reasoning.encrypted_content"],
        **extra,
    }
    schema = json_schema or _RUNTIME_OVERRIDES.get("json_schema")
    if schema:
        kwargs["text"] = {"format": {
            "type": "json_schema", "name": "flowpilot_benchmark_output",
            "strict": False, "schema": schema,
        }}
    elif _RUNTIME_OVERRIDES.get("json_mode"):
        kwargs["text"] = {"format": {"type": "json_object"}}
    started = time.perf_counter()
    def complete_stream():
        # Streaming avoids an idle connection while long reasoning requests run.
        # Only a terminal response is returned; partial output is never a design.
        with _get_openai_client().responses.create(**kwargs, stream=True) as stream:
            for event in stream:
                if event.type in {"response.completed", "response.incomplete", "response.failed"}:
                    return event.response
                if event.type == "error":
                    raise RuntimeError(f"Astra streaming error: {event.message}")
        raise RuntimeError("Astra stream ended without a terminal response")

    resp = _retry_llm_request(
        complete_stream,
        provider="openai", model=model, api_name=api_name,
    )
    content = (resp.output_text or "").strip()
    calls = [item for item in resp.output if item.type == "function_call"]
    event = {
        **_base_event(api_name=api_name, provider="openai", model=model,
                      max_tokens=limit, system=system,
                      user_content=_stringify_messages(inputs)),
        "api": "responses", "streaming": True, "response_id": resp.id,
        "returned_model": resp.model, "requested_max_tokens": max_tokens,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "usage": _usage_to_dict(resp), "response_chars": len(content),
        "finish_reason": resp.status, "tool_calls_requested": len(calls),
        **_content_event(content),
    }
    if _RUNTIME_OVERRIDES.get("capture_content"):
        event["request_input"] = inputs
        event["tool_definitions"] = extra.get("tools", [])
        event["response_format"] = kwargs.get("text")
        event["tool_calls"] = [item.model_dump() for item in calls]
    _emit_llm_event(event)
    if resp.status != "completed":
        raise RuntimeError(f"{api_name}: Astra response {resp.status}: {resp.incomplete_details}")
    if not content and not calls:
        raise RuntimeError(f"{api_name}: Astra returned no text or tool calls (possible refusal)")
    return resp


def _responses_tool_loop(system, user_content, tools, tool_executor, max_tokens, max_tool_turns):
    inputs = [{"role": "user", "content": user_content}]
    definitions = [{
        "type": "function", "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool["input_schema"], "strict": False,
    } for tool in tools]
    log = []
    for turn in range(max_tool_turns + 1):
        final = turn == max_tool_turns
        resp = _responses_request(
            model=ENGINE_MODEL_OPENAI, system=system, inputs=inputs,
            max_tokens=max_tokens,
            api_name="call_llm_with_tools_force_final" if final else "call_llm_with_tools_turn",
            tools=definitions, tool_choice="none" if final else "auto",
        )
        calls = [item for item in resp.output if item.type == "function_call"]
        if not calls:
            return resp.output_text.strip(), log
        if final:
            raise RuntimeError("Astra requested tools after the tool-turn limit")
        inputs.extend(item.model_dump(exclude_none=True) for item in resp.output)
        for call in calls:
            arguments = json.loads(call.arguments)
            if not isinstance(arguments, dict) or call.name not in {tool["name"] for tool in tools}:
                raise ValueError("Invalid Astra tool request")
            result = tool_executor(call.name, arguments)
            log.append({"tool": call.name, "input": arguments, "result": result})
            inputs.append({"type": "function_call_output", "call_id": call.call_id,
                           "output": json.dumps(result)})
    raise RuntimeError("Astra tool loop ended without a final response")


def _bounded_local_messages(
    system: str,
    messages: list[dict],
    *,
    max_tokens: int,
) -> tuple[str, list[dict]]:
    """Keep local-model requests safely below the deployed 32k context."""

    input_token_budget = max(4096, 32768 - max_tokens - 2048)
    # Chemistry and JSON tokenize densely. Two characters per token is a
    # deliberately conservative estimate for the deployed Qwen endpoints.
    char_budget = input_token_budget * 2
    total_chars = len(system or "") + sum(
        len(content) if isinstance(content := msg.get("content", ""), str)
        else len(json.dumps(content, default=str))
        for msg in messages
    )
    if total_chars <= char_budget:
        return system, messages

    remaining = max(2000, char_budget - len(system or ""))
    bounded = [dict(msg) for msg in messages]
    text_indices = [
        index for index, msg in enumerate(bounded)
        if isinstance(msg.get("content"), str)
    ]
    if not text_indices:
        return system, messages
    per_message = max(1000, math.floor(remaining / len(text_indices)))
    marker = "\n\n[FlowPilot compacted repeated context for local-model limits.]\n\n"
    for index in text_indices:
        content = bounded[index]["content"]
        if len(content) <= per_message:
            continue
        usable = max(500, per_message - len(marker))
        head = math.floor(usable * 0.65)
        bounded[index]["content"] = content[:head] + marker + content[-(usable - head):]
    logger.warning(
        "Local prompt compacted from %s to at most %s characters for context safety",
        total_chars,
        char_budget,
    )
    return system, bounded


def call_model_messages(
    *,
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int,
    provider: str | None = None,
    api_name: str = "call_model_messages",
    json_schema: dict | None = None,
) -> TextGenerationResult:
    """Provider-agnostic text generation for upstream modules."""
    resolved_provider = infer_provider_for_model(model, provider)
    user_content = _stringify_messages(messages)

    if resolved_provider == "openai":
        if _uses_responses_api(model):
            resp = _responses_request(
                model=model, system=system, inputs=messages, max_tokens=max_tokens,
                api_name=api_name, json_schema=json_schema,
            )
            return TextGenerationResult(
                text=resp.output_text.strip(), provider="openai", model=model,
                usage=_usage_to_dict(resp), finish_reason=resp.status,
            )
        kwargs = _runtime_kwargs("openai", model)
        effective_json_schema = json_schema or _RUNTIME_OVERRIDES.get("json_schema")
        if effective_json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "flowpilot_benchmark_output",
                    "strict": False,
                    "schema": effective_json_schema,
                },
            }
        elif _RUNTIME_OVERRIDES.get("json_mode"):
            kwargs["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        resp = _retry_llm_request(
            lambda: _get_openai_client().chat.completions.create(
                model=model,
                **_openai_token_limit(model, max_tokens),
                messages=[{"role": "system", "content": system}, *messages],
                **kwargs,
            ),
            provider="openai",
            model=model,
            api_name=api_name,
        )
        content = (resp.choices[0].message.content or "").strip()
        usage = _usage_to_dict(resp)
        finish_reason = getattr(resp.choices[0], "finish_reason", None)
        _emit_llm_event({
            **_base_event(
                api_name=api_name,
                provider="openai",
                model=model,
                max_tokens=max_tokens,
                system=system,
                user_content=user_content,
            ),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "usage": usage,
            "response_chars": len(content),
            "finish_reason": finish_reason,
            **_content_event(content),
        })
        return TextGenerationResult(
            text=content,
            provider="openai",
            model=model,
            usage=usage,
            finish_reason=finish_reason,
        )

    if resolved_provider == "ollama":
        client = _get_ollama_client(model)
        system, messages = _bounded_local_messages(
            system,
            messages,
            max_tokens=max_tokens,
        )
        user_content = _stringify_messages(messages)
        kwargs = {}
        if "temperature" in _RUNTIME_OVERRIDES:
            kwargs["temperature"] = _RUNTIME_OVERRIDES["temperature"]
        if "seed" in _RUNTIME_OVERRIDES:
            kwargs["seed"] = _RUNTIME_OVERRIDES["seed"]
        effective_json_schema = json_schema or _RUNTIME_OVERRIDES.get("json_schema")
        if effective_json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "flowpilot_benchmark_output",
                    "strict": False,
                    "schema": effective_json_schema,
                },
            }
        elif _RUNTIME_OVERRIDES.get("json_mode"):
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        resp = _retry_llm_request(
            lambda: client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                stream=False,
                messages=[{"role": "system", "content": system}, *[
                    {
                        **msg,
                        "content": f"/no_think\n{msg['content']}" if idx == len(messages) - 1 and isinstance(msg.get("content"), str) else msg.get("content", "")
                    }
                    for idx, msg in enumerate(messages)
                ]],
                extra_body={
                    "think": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                **kwargs,
            ),
            provider="ollama",
            model=model,
            api_name=api_name,
        )
        choice = resp.choices[0] if resp.choices else None
        content = choice.message.content if choice else None
        if not content and choice:
            extra = getattr(choice.message, "model_extra", None) or {}
            reasoning = extra.get("reasoning") or getattr(choice.message, "reasoning", None)
            if reasoning:
                logger.info("Ollama: content empty — extracting from reasoning field")
                content = reasoning
        if not content:
            content = ""
        raw_content = content.strip()
        content = _strip_thinking(raw_content)
        usage = _usage_to_dict(resp)
        finish_reason = getattr(choice, "finish_reason", None) if choice else None
        _emit_llm_event({
            **_base_event(
                api_name=api_name,
                provider="ollama",
                model=model,
                max_tokens=max_tokens,
                system=system,
                user_content=user_content,
            ),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "usage": usage,
            "response_chars": len(content),
            "finish_reason": finish_reason,
            "empty_content": not bool(content),
            **_content_event(content, raw_content),
        })
        return TextGenerationResult(
            text=content,
            provider="ollama",
            model=model,
            usage=usage,
            finish_reason=finish_reason,
        )

    kwargs = _runtime_kwargs("anthropic", model)
    effective_json_schema = json_schema or _RUNTIME_OVERRIDES.get("json_schema")
    if effective_json_schema:
        kwargs["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": effective_json_schema,
            }
        }
    started = time.perf_counter()
    resp = _retry_llm_request(
        lambda: _get_anthropic_client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            **kwargs,
        ),
        provider="anthropic",
        model=model,
        api_name=api_name,
    )
    content = _anthropic_text(resp.content)
    usage = _usage_to_dict(resp)
    stop_reason = getattr(resp, "stop_reason", None)
    _emit_llm_event({
        **_base_event(
            api_name=api_name,
            provider="anthropic",
            model=model,
            max_tokens=max_tokens,
            system=system,
            user_content=user_content,
        ),
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "usage": usage,
        "response_chars": len(content),
        "stop_reason": stop_reason,
        **_content_event(content),
    })
    return TextGenerationResult(
        text=content,
        provider="anthropic",
        model=model,
        usage=usage,
        stop_reason=stop_reason,
    )


def call_model_text(
    *,
    model: str,
    system: str,
    user_content: str,
    max_tokens: int,
    provider: str | None = None,
    api_name: str = "call_model_text",
    json_schema: dict | None = None,
) -> TextGenerationResult:
    return call_model_messages(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user_content}],
        max_tokens=max_tokens,
        provider=provider,
        api_name=api_name,
        json_schema=json_schema,
    )


def call_llm(system: str, user_content: str, max_tokens: int) -> str:
    """Route a council LLM call to the configured provider.

    Provider is set by ENGINE_PROVIDER in config.py:
      "anthropic" → Claude Sonnet
      "openai"    → GPT-4o (or ENGINE_MODEL_OPENAI)
      "ollama"    → local model at OLLAMA_BASE_URL
    """
    if ENGINE_PROVIDER in ("openai",):
        if _uses_responses_api(ENGINE_MODEL_OPENAI):
            return call_model_text(
                model=ENGINE_MODEL_OPENAI, system=system, user_content=user_content,
                max_tokens=max_tokens, provider="openai", api_name="call_llm",
            ).text
        kwargs = _runtime_kwargs("openai", ENGINE_MODEL_OPENAI)
        started = time.perf_counter()
        resp = _retry_llm_request(
            lambda: _get_openai_client().chat.completions.create(
                model=ENGINE_MODEL_OPENAI,
                **_openai_token_limit(ENGINE_MODEL_OPENAI, max_tokens),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_content},
                ],
                **kwargs,
            ),
            provider="openai",
            model=ENGINE_MODEL_OPENAI,
            api_name="call_llm",
        )
        content = resp.choices[0].message.content.strip()
        _emit_llm_event({
            **_base_event(
                api_name="call_llm",
                provider="openai",
                model=ENGINE_MODEL_OPENAI,
                max_tokens=max_tokens,
                system=system,
                user_content=user_content,
            ),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "usage": _usage_to_dict(resp),
            "response_chars": len(content),
            **_content_event(content),
        })
        return content

    elif ENGINE_PROVIDER == "ollama":
        client = _get_ollama_client(ENGINE_MODEL_OLLAMA)
        extra_body = {
            "think": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        kwargs = {}
        if "temperature" in _RUNTIME_OVERRIDES:
            kwargs["temperature"] = _RUNTIME_OVERRIDES["temperature"]
        if "seed" in _RUNTIME_OVERRIDES:
            kwargs["seed"] = _RUNTIME_OVERRIDES["seed"]

        # gemma4 (and similar) are "thinking" models: they spend all tokens on an
        # internal scratchpad ("reasoning" field) and never write to "content".
        # Disable thinking via the Ollama API flag (0.9+) AND via the /no_think
        # user-message prefix (all versions that support it).
        started = time.perf_counter()
        resp = _retry_llm_request(
            lambda: client.chat.completions.create(
                model=ENGINE_MODEL_OLLAMA,
                max_tokens=max_tokens,
                stream=False,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": f"/no_think\n{user_content}"},
                ],
                extra_body=extra_body,
                **kwargs,
            ),
            provider="ollama",
            model=ENGINE_MODEL_OLLAMA,
            api_name="call_llm",
        )

        choice  = resp.choices[0] if resp.choices else None
        content = choice.message.content if choice else None

        # Fallback: if thinking was NOT disabled, the answer is in "reasoning"
        if not content and choice:
            extra     = getattr(choice.message, "model_extra", None) or {}
            reasoning = extra.get("reasoning") or getattr(choice.message, "reasoning", None)
            if reasoning:
                logger.info("Ollama: content empty — extracting from reasoning field")
                content = reasoning

        if not content:
            logger.error(
                "Ollama returned no content (model=%s finish_reason=%s)",
                ENGINE_MODEL_OLLAMA,
                choice.finish_reason if choice else "N/A",
            )
            _emit_llm_event({
                **_base_event(
                    api_name="call_llm",
                    provider="ollama",
                    model=ENGINE_MODEL_OLLAMA,
                    max_tokens=max_tokens,
                    system=system,
                    user_content=user_content,
                ),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "usage": _usage_to_dict(resp),
                "response_chars": 0,
                "finish_reason": choice.finish_reason if choice else None,
                "empty_content": True,
                **_content_event(""),
            })
            return ""

        raw_content = content.strip()
        content = _strip_thinking(raw_content)
        _emit_llm_event({
            **_base_event(
                api_name="call_llm",
                provider="ollama",
                model=ENGINE_MODEL_OLLAMA,
                max_tokens=max_tokens,
                system=system,
                user_content=user_content,
            ),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "usage": _usage_to_dict(resp),
            "response_chars": len(content),
            "finish_reason": choice.finish_reason if choice else None,
            **_content_event(content, raw_content),
        })
        return content

    else:  # "anthropic" (default)
        kwargs = _runtime_kwargs("anthropic", ENGINE_MODEL_ANTHROPIC)
        started = time.perf_counter()
        resp = _retry_llm_request(
            lambda: _get_anthropic_client().messages.create(
                model=ENGINE_MODEL_ANTHROPIC,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_content}],
                **kwargs,
            ),
            provider="anthropic",
            model=ENGINE_MODEL_ANTHROPIC,
            api_name="call_llm",
        )
        content = _anthropic_text(resp.content)
        _emit_llm_event({
            **_base_event(
                api_name="call_llm",
                provider="anthropic",
                model=ENGINE_MODEL_ANTHROPIC,
                max_tokens=max_tokens,
                system=system,
                user_content=user_content,
            ),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "usage": _usage_to_dict(resp),
            "response_chars": len(content),
            "stop_reason": getattr(resp, "stop_reason", None),
            **_content_event(content),
        })
        return content


def call_llm_with_tools(
    system: str,
    user_content: str,
    tools: list[dict],
    tool_executor,
    max_tokens: int,
    max_tool_turns: int = 4,
) -> tuple[str, list[dict]]:
    """Route a tool-using council LLM call to the configured provider.

    Returns (final_text, tool_calls_log) where each log entry is
    {"tool": name, "input": dict, "result": dict}.

    For Ollama: gracefully falls back to call_llm() with empty log
    (most local models don't reliably support tools).
    """
    tool_calls_log: list[dict] = []

    if ENGINE_PROVIDER == "openai" and _uses_responses_api(ENGINE_MODEL_OPENAI):
        return _responses_tool_loop(
            system, user_content, tools, tool_executor, max_tokens, max_tool_turns,
        )

    # ── Ollama fallback (no reliable tool support) ───────────────────────────
    if ENGINE_PROVIDER == "ollama":
        logger.info("call_llm_with_tools: Ollama detected — falling back to call_llm (no tools)")
        try:
            text = call_llm(system, user_content, max_tokens=max_tokens)
        except Exception as e:
            logger.warning("call_llm_with_tools Ollama fallback failed: %s", e)
            text = ""
        return text, tool_calls_log

    # ── Anthropic tool loop ──────────────────────────────────────────────────
    if ENGINE_PROVIDER not in ("openai",):  # default: anthropic
        try:
            client = _get_anthropic_client()
            messages = [{"role": "user", "content": user_content}]
            for _ in range(max_tool_turns):
                kwargs = _runtime_kwargs("anthropic", ENGINE_MODEL_ANTHROPIC)
                started = time.perf_counter()
                resp = client.messages.create(
                    model=ENGINE_MODEL_ANTHROPIC,
                    max_tokens=max_tokens,
                    system=system,
                    tools=tools,
                    messages=messages,
                    **kwargs,
                )
                _emit_llm_event({
                    **_base_event(
                        api_name="call_llm_with_tools_turn",
                        provider="anthropic",
                        model=ENGINE_MODEL_ANTHROPIC,
                        max_tokens=max_tokens,
                        system=system,
                        user_content=user_content,
                    ),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "usage": _usage_to_dict(resp),
                    "tool_calls_requested": len([b for b in resp.content if getattr(b, "type", "") == "tool_use"]),
                    "stop_reason": getattr(resp, "stop_reason", None),
                    **_content_event(_anthropic_text(resp.content)),
                })
                if resp.stop_reason != "tool_use":
                    final_text = _anthropic_text(resp.content)
                    return final_text, tool_calls_log
                # Execute all tool_use blocks
                tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]
                tool_results = []
                for block in tool_use_blocks:
                    result = tool_executor(block.name, dict(block.input))
                    tool_calls_log.append({
                        "tool": block.name,
                        "input": dict(block.input),
                        "result": result,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": tool_results})
            # Tool turns exhausted — force one final text-only response using
            # accumulated tool results already in the message history.
            logger.warning(
                "call_llm_with_tools: exhausted %d tool turns — forcing final text response",
                max_tool_turns,
            )
            try:
                force_msgs = messages + [{
                    "role": "user",
                    "content": (
                        "You have completed all necessary tool calls. "
                        "Now produce your final JSON response. "
                        "Do NOT call any more tools. Output only the JSON."
                    ),
                }]
                started = time.perf_counter()
                final_resp = client.messages.create(
                    model=ENGINE_MODEL_ANTHROPIC,
                    max_tokens=max_tokens,
                    system=system,
                    tools=tools,
                    tool_choice={"type": "none"},
                    messages=force_msgs,
                    **kwargs,
                )
                _emit_llm_event({
                    **_base_event(
                        api_name="call_llm_with_tools_force_final",
                        provider="anthropic",
                        model=ENGINE_MODEL_ANTHROPIC,
                        max_tokens=max_tokens,
                        system=system,
                        user_content=user_content,
                    ),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "usage": _usage_to_dict(final_resp),
                    "tool_calls_requested": 0,
                    "stop_reason": getattr(final_resp, "stop_reason", None),
                    **_content_event(_anthropic_text(final_resp.content)),
                })
                final_text = _anthropic_text(final_resp.content)
                if final_text:
                    return final_text, tool_calls_log
            except Exception as force_err:
                logger.warning("call_llm_with_tools: forced final call failed: %s", force_err)
            return "", tool_calls_log
        except Exception as e:
            logger.warning("call_llm_with_tools (anthropic) failed: %s — falling back to call_llm", e)
            try:
                text = call_llm(system, user_content, max_tokens=max_tokens)
            except Exception as e2:
                logger.warning("call_llm_with_tools fallback also failed: %s", e2)
                text = ""
            return text, tool_calls_log

    # ── OpenAI tool loop ─────────────────────────────────────────────────────
    try:
        from flora_translate.engine.tool_definitions import to_openai_tools
        client = _get_openai_client()
        openai_tools = to_openai_tools(tools)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        for _ in range(max_tool_turns):
            kwargs = _runtime_kwargs("openai", ENGINE_MODEL_OPENAI)
            started = time.perf_counter()
            resp = client.chat.completions.create(
                model=ENGINE_MODEL_OPENAI,
                **_openai_token_limit(ENGINE_MODEL_OPENAI, max_tokens),
                **_openai_tool_kwargs(ENGINE_MODEL_OPENAI),
                tools=openai_tools,
                tool_choice="auto",
                messages=messages,
                **kwargs,
            )
            choice = resp.choices[0]
            _emit_llm_event({
                **_base_event(
                    api_name="call_llm_with_tools_turn",
                    provider="openai",
                    model=ENGINE_MODEL_OPENAI,
                    max_tokens=max_tokens,
                    system=system,
                    user_content=user_content,
                ),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "usage": _usage_to_dict(resp),
                "tool_calls_requested": len(choice.message.tool_calls or []),
                "finish_reason": choice.finish_reason,
                **_content_event((choice.message.content or "").strip()),
            })
            msg = choice.message
            if not msg.tool_calls:
                return (msg.content or "").strip(), tool_calls_log
            # Execute all tool calls
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]})
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_input = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    fn_input = {}
                result = tool_executor(fn_name, fn_input)
                tool_calls_log.append({
                    "tool": fn_name,
                    "input": fn_input,
                    "result": result,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })
        logger.warning("call_llm_with_tools: exhausted %d tool turns without final text", max_tool_turns)
        return "", tool_calls_log
    except Exception as e:
        logger.warning("call_llm_with_tools (openai) failed: %s — falling back to call_llm", e)
        try:
            text = call_llm(system, user_content, max_tokens=max_tokens)
        except Exception as e2:
            logger.warning("call_llm_with_tools fallback also failed: %s", e2)
            text = ""
        return text, tool_calls_log
