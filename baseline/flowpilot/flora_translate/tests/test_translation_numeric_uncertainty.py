import json
from types import SimpleNamespace

from flora_translate.translation_llm import TranslationLLM


def test_schema_repair_preserves_scientific_instructions_and_unknown_numbers(monkeypatch):
    calls = []
    invalid = {"residence_time_in_channel_min": "quantity unresolved"}
    fixed = {"residence_time_in_channel_min": None,
             "reasoning_per_field": {"residence_time_in_channel_min": "quantity unresolved"}}

    def model(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps(invalid if len(calls) < 3 else fixed))

    monkeypatch.setattr("flora_translate.translation_llm.call_model_text", model)
    result = TranslationLLM().generate("Preserve batch air as a source fact.", "Source protocol")
    assert len(calls) == 3
    assert all("Preserve batch air" in call["system"] for call in calls)
    assert "VALIDATION ERROR" in calls[-1]["user_content"]
    assert "SCHEMA" in calls[-1]["user_content"]
    assert "Do not replace unknown quantities with guessed numbers" in calls[-1]["user_content"]
    assert result.residence_time_in_channel_min is None
    assert result.reasoning_per_field["residence_time_in_channel_min"] == "quantity unresolved"
