import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from flora_translate.schemas import FlowProposal
from flora_translate.translation_llm import TranslationLLM


@pytest.mark.parametrize("confidence", [0.85, 85, "MEDIUM"])
def test_numeric_confidence_is_preserved_without_repair_calls(confidence, monkeypatch):
    calls = []
    def respond(**kwargs):
        calls.append(kwargs["api_name"])
        return SimpleNamespace(text=json.dumps({
            "residence_time_min": 10, "residence_time_basis": "liquid-only",
            "flow_rate_mL_min": 0.1, "reactor_volume_mL": 1,
            "temperature_C": 40, "BPR_bar": 3, "concentration_M": 0.1,
            "reactor_type": "coil", "tubing_ID_mm": 1, "tubing_material": "PFA",
            "confidence": confidence,
        }))
    monkeypatch.setattr("flora_translate.translation_llm.call_model_text", respond)
    proposal = TranslationLLM().generate("system", "user")
    assert proposal.confidence == str(confidence)
    assert len(calls) == 1
    assert proposal.engine_validated is False
    assert proposal.flow_rate_mL_min == 0.1


def test_arbitrary_objects_are_not_coerced_to_confidence():
    with pytest.raises(ValidationError):
        FlowProposal(confidence={"unexpected": []})


@pytest.mark.parametrize("field", ["pre_reactor_steps", "post_reactor_steps", "literature_analogies", "safety_flags"])
def test_text_metadata_accepts_one_string_or_structured_notes_without_losing_fields(field):
    text = "One complete instruction, including both clauses."
    assert getattr(FlowProposal(**{field: text}), field) == [text]
    note = {"step": "collection", "description": "Collect a sample.", "equipment": "collector-1", "required": True}
    for value in (note, [note], {"items": [note]}):
        proposal = FlowProposal(**TranslationLLM._normalize_proposal_data({field: value}))
        assert json.loads(getattr(proposal, field)[0]) == note


def test_structured_streams_are_not_serialized_as_text_metadata():
    data = TranslationLLM._normalize_proposal_data({"streams": {"stream_label": "A", "contents": ["substrate"], "flow_rate_mL_min": 0.1}})
    proposal = FlowProposal(**data)
    assert proposal.streams[0].flow_rate_mL_min == 0.1
    with pytest.raises(ValidationError):
        FlowProposal(flow_rate_mL_min={"ambiguous": [1, 2]})


def test_metadata_shape_differences_do_not_trigger_model_repair(monkeypatch):
    calls = []
    def respond(**kwargs):
        calls.append(kwargs["api_name"])
        return SimpleNamespace(text=json.dumps({"confidence": 0.85, "pre_reactor_steps": "Prepare feed.",
            "post_reactor_steps": [{"step": "collection", "description": "Collect after review.", "equipment": "C1"}],
            "reasoning_per_field": {"flow": {"source": "inventory", "value": 0.1}},
            "literature_analogies": "One source, not individual characters."}))
    monkeypatch.setattr("flora_translate.translation_llm.call_model_text", respond)
    proposal = TranslationLLM().generate("system", "user")
    assert len(calls) == 1
    assert proposal.literature_analogies == ["One source, not individual characters."]
    assert json.loads(proposal.post_reactor_steps[0])["equipment"] == "C1"
