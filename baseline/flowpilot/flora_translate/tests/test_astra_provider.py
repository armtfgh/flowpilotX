from types import SimpleNamespace
from unittest.mock import Mock
from contextlib import nullcontext

import pytest

from flora_translate.engine import llm_agents as llm


def response(text="OK", status="completed", output=None):
    return SimpleNamespace(
        id="resp_test", model="gpt-6-astra", output_text=text,
        status=status, output=output or [], incomplete_details="max_output_tokens",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15,
                              input_tokens_details={"cached_tokens": 2},
                              output_tokens_details={"reasoning_tokens": 3}),
    )


def stream_response(*args, **kwargs):
    result = response(*args, **kwargs)
    return nullcontext([SimpleNamespace(type=f"response.{result.status}", response=result)])


@pytest.fixture
def client(monkeypatch):
    client = Mock()
    client.responses.create.return_value = stream_response()
    monkeypatch.setattr(llm, "_OPENAI_CLIENT", client)
    monkeypatch.setattr(llm, "ENGINE_PROVIDER", "openai")
    monkeypatch.setattr(llm, "ENGINE_MODEL_OPENAI", "gpt-6-astra")
    monkeypatch.setattr(llm, "_RUNTIME_OVERRIDES", {"temperature": .2, "seed": 42})
    return client


def test_astra_request_and_effective_telemetry(client, monkeypatch):
    events = []
    monkeypatch.setattr(llm, "_LLM_OBSERVER", events.append)
    result = llm.call_model_text(model="gpt-6-astra", system="System", user_content="Input", max_tokens=4096)
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["reasoning"] == {"effort": "medium"}
    assert kwargs["max_output_tokens"] == 16384
    assert kwargs["store"] is False
    assert kwargs["stream"] is True
    assert not {"temperature", "seed", "max_tokens", "top_p"} & kwargs.keys()
    assert not {"temperature", "seed"} & events[0].keys()
    assert events[0]["usage"]["output_tokens_details"]["reasoning_tokens"] == 3
    assert result.text == "OK"
    client.chat.completions.create.assert_not_called()


def test_astra_structured_output(client):
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    llm.call_model_text(model="gpt-6-astra", system="S", user_content="JSON", max_tokens=1000, json_schema=schema)
    assert client.responses.create.call_args.kwargs["text"]["format"]["schema"] == schema


def test_council_uses_same_model(client):
    assert llm.call_llm("S", "U", 1000) == "OK"
    assert client.responses.create.call_args.kwargs["model"] == "gpt-6-astra"


def test_tool_loop_round_trip_and_forced_final(client):
    item = Mock(type="function_call", call_id="call_1", arguments='{"x": 2}')
    item.name = "calculate"
    item.model_dump.return_value = {"type": "function_call", "call_id": "call_1", "name": "calculate", "arguments": '{"x": 2}'}
    client.responses.create.side_effect = [stream_response("", output=[item]), stream_response("Done")]
    executor = Mock(return_value={"answer": 4})
    tools = [{"name": "calculate", "description": "Compute", "input_schema": {"type": "object"}}]
    text, log = llm.call_llm_with_tools("S", "U", tools, executor, 1000, max_tool_turns=1)
    executor.assert_called_once_with("calculate", {"x": 2})
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["tool_choice"] == "none"
    assert kwargs["input"][-1] == {"type": "function_call_output", "call_id": "call_1", "output": '{"answer": 4}'}
    assert text == "Done" and log[0]["result"] == {"answer": 4}
    client.chat.completions.create.assert_not_called()


def test_incomplete_is_not_silently_accepted(client):
    client.responses.create.return_value = stream_response("partial", "incomplete")
    with pytest.raises(RuntimeError, match="incomplete"):
        llm.call_llm("S", "U", 1000)


def test_tool_provider_failure_is_not_hidden_by_fallback(client):
    client.responses.create.side_effect = ValueError("invalid request")
    with pytest.raises(RuntimeError, match="invalid request"):
        llm.call_llm_with_tools("S", "U", [], Mock(), 1000)
    client.chat.completions.create.assert_not_called()


def test_unterminated_stream_is_not_a_complete_design(client):
    client.responses.create.return_value = nullcontext([])
    with pytest.raises(RuntimeError, match="terminal response"):
        llm.call_llm("S", "U", 1000)


def test_legacy_gpt4o_request_unchanged(client):
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="OK"), finish_reason="stop")], usage=None,
    )
    llm.call_model_text(model="gpt-4o", system="S", user_content="U", max_tokens=1000)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] == 1000
    assert kwargs["temperature"] == .2 and kwargs["seed"] == 42
    client.responses.create.assert_not_called()
