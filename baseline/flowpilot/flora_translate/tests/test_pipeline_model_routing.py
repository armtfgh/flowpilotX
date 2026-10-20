from __future__ import annotations

import flora_translate.config as cfg
from flora_translate.engine import llm_agents
from flora_translate.model_catalog import runtime_model_options
from flora_translate.pipeline_runtime import (
    PipelineRuntimeOptions,
    runtime_model_routing,
)


def test_catalog_maps_gui_choices_to_runtime_roles() -> None:
    options = runtime_model_options("claude-opus-4-6", "gpt-4o")

    assert options["upstream_model"] == "claude-opus-4-6"
    assert options["upstream_provider"] == "anthropic"
    assert options["downstream_model"] == "gpt-4o"
    assert options["downstream_provider"] == "openai"


def test_runtime_routing_changes_real_pipeline_models_and_restores_defaults() -> None:
    saved = (
        cfg.MODEL_INPUT_PARSER,
        cfg.MODEL_CHEMISTRY_AGENT,
        cfg.MODEL_TRANSLATION,
        cfg.MODEL_OUTPUT_FORMATTER,
        cfg.MODEL_CONVERSATION_AGENT,
        llm_agents.ENGINE_PROVIDER,
        llm_agents.ENGINE_MODEL_OPENAI,
        llm_agents.get_model_endpoint_overrides(),
    )
    runtime = PipelineRuntimeOptions.coerce(
        runtime_model_options("claude-sonnet-4-6", "gpt-4o")
    )

    with runtime_model_routing(runtime):
        assert cfg.MODEL_INPUT_PARSER == "claude-sonnet-4-6"
        assert cfg.MODEL_CHEMISTRY_AGENT == "claude-sonnet-4-6"
        assert cfg.MODEL_TRANSLATION == "gpt-4o"
        assert cfg.MODEL_OUTPUT_FORMATTER == "gpt-4o"
        assert cfg.MODEL_CONVERSATION_AGENT == "gpt-4o"
        assert llm_agents.ENGINE_PROVIDER == "openai"
        assert llm_agents.ENGINE_MODEL_OPENAI == "gpt-4o"

    assert (
        cfg.MODEL_INPUT_PARSER,
        cfg.MODEL_CHEMISTRY_AGENT,
        cfg.MODEL_TRANSLATION,
        cfg.MODEL_OUTPUT_FORMATTER,
        cfg.MODEL_CONVERSATION_AGENT,
        llm_agents.ENGINE_PROVIDER,
        llm_agents.ENGINE_MODEL_OPENAI,
        llm_agents.get_model_endpoint_overrides(),
    ) == saved


def test_each_local_model_uses_its_configured_vllm_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("FLOWPILOT_QWEN36_BASE_URL", "http://127.0.0.1:9101/v1")
    monkeypatch.setenv("FLOWPILOT_QWEN38_BASE_URL", "http://127.0.0.1:9102/v1")
    options = runtime_model_options("qwen3.6-27b", "qwen3.8-27b")
    runtime = PipelineRuntimeOptions.coerce(options)

    with runtime_model_routing(runtime):
        assert cfg.MODEL_INPUT_PARSER == "/models/Qwen3.6-27B"
        assert cfg.MODEL_CHEMISTRY_AGENT == "/models/Qwen3.6-27B"
        assert cfg.MODEL_TRANSLATION == "/models/Qwen3.8-27B"
        assert cfg.MODEL_OUTPUT_FORMATTER == "/models/Qwen3.8-27B"
        assert cfg.MODEL_CONVERSATION_AGENT == "/models/Qwen3.8-27B"
        assert llm_agents.ENGINE_PROVIDER == "ollama"
        assert llm_agents.ENGINE_MODEL_OLLAMA == "/models/Qwen3.8-27B"
        endpoints = llm_agents.get_model_endpoint_overrides()
        assert endpoints["/models/Qwen3.6-27B"] == "http://127.0.0.1:9101/v1"
        assert endpoints["/models/Qwen3.8-27B"] == "http://127.0.0.1:9102/v1"
