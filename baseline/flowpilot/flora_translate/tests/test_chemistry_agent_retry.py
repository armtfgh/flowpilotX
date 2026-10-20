from types import SimpleNamespace

from flora_translate import chemistry_agent


def test_truncated_chemistry_output_retries_with_compact_context(monkeypatch) -> None:
    calls = []

    def fake_call_model_text(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(
                text='<JSON>{"reaction_name": "truncated',
                stop_reason="max_tokens",
                finish_reason=None,
            )
        return SimpleNamespace(
            text='<JSON>{"reaction_name": "complete"}</JSON>',
            stop_reason="end_turn",
            finish_reason=None,
        )

    monkeypatch.setattr(chemistry_agent, "call_model_text", fake_call_model_text)
    monkeypatch.setattr(chemistry_agent.cfg, "MODEL_CHEMISTRY_AGENT", "test-model")
    monkeypatch.setattr(chemistry_agent.cfg, "MODEL_TRANSLATION", "test-model")

    full_system = chemistry_agent.CHEMISTRY_SYSTEM + "\n\n## FLOW CHEMISTRY HANDBOOK RULES\nrule"
    result = chemistry_agent.ChemistryReasoningAgent()._call_with_retry(full_system, "input")

    assert len(calls) == 2
    assert "FLOW CHEMISTRY HANDBOOK RULES" in calls[0]["system"]
    assert calls[1]["system"].endswith(
        "Fundamentals rules were omitted in this fallback call because the larger "
        "prompt failed at the provider connection layer."
    )
    assert calls[0]["max_tokens"] == chemistry_agent.cfg.CHEMISTRY_MAX_TOKENS
    assert calls[1]["max_tokens"] == max(
        chemistry_agent.cfg.CHEMISTRY_MAX_TOKENS * 2, 16384
    )
    assert '"complete"' in result
