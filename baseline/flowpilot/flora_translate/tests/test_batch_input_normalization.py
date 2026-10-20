import json
from types import SimpleNamespace

import pytest

from flora_translate.batch_normalization import normalize_batch_numeric_fields
from flora_translate.input_parser import InputParser


def test_multistep_numeric_objects_reduce_to_batch_record_scalars():
    normalized = normalize_batch_numeric_fields(
        {
            "reaction_time_h": {
                "step_1_h": 4,
                "step_2_h": 6,
                "total_h": 10,
            },
            "yield_pct": {
                "step_1_pct": 82,
                "step_2_pct": 71,
                "overall_yield_pct": 68,
            },
            "temperature_C": {"step_1_C": 25, "step_2_C": 40},
        }
    )

    assert normalized["reaction_time_h"] == 10
    assert normalized["yield_pct"] == 68
    assert normalized["temperature_C"] == 25


def test_multistep_time_sums_stages_and_converts_minutes_without_total():
    normalized = normalize_batch_numeric_fields(
        {
            "reaction_time_h": {
                "step_1_min": 30,
                "step_2_h": 1.5,
            },
            "yield_pct": {"stage_1_pct": 80, "stage_2_pct": 72},
        }
    )

    assert normalized["reaction_time_h"] == pytest.approx(2.0)
    assert normalized["yield_pct"] == 72


def test_commercial_input_parser_accepts_structured_json_fields(monkeypatch):
    payload = {
        "reaction_description": "Two-step photoredox process",
        "reaction_time_h": {"step_1_h": 4, "step_2_h": 6, "total_h": 10},
        "yield_pct": {"final_yield_pct": 68},
        "temperature_C": {"step_1_C": 25, "step_2_C": 40},
        "additives": [],
    }
    monkeypatch.setattr(
        "flora_translate.input_parser.call_model_text",
        lambda **_: SimpleNamespace(text=json.dumps(payload)),
    )

    record = InputParser().parse("A free-text two-step chemistry protocol")

    assert record.reaction_time_h == 10
    assert record.yield_pct == 68
    assert record.temperature_C == 25


def test_input_parser_requests_provider_native_json_schema(monkeypatch):
    captured = {}

    def fake_call(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text=json.dumps({
            "reaction_description": "Thermal reaction",
            "photocatalyst": None,
            "catalyst_loading_mol_pct": None,
            "base": None,
            "solvent": "methanol",
            "temperature_C": 60,
            "reaction_time_h": 1,
            "concentration_M": 0.1,
            "scale_mmol": 1,
            "yield_pct": None,
            "light_source": None,
            "wavelength_nm": None,
            "additives": [],
            "atmosphere": "N2",
        }))

    monkeypatch.setattr("flora_translate.input_parser.call_model_text", fake_call)

    record = InputParser().parse("A thermal reaction under nitrogen.")

    assert captured["json_schema"]["additionalProperties"] is False
    assert captured["max_tokens"] == 2048
    assert record.temperature_C == 60


def test_direct_json_input_uses_same_normalization():
    record = InputParser().parse(
        {
            "reaction_description": "Two-step process",
            "reaction_time_h": {"step_1_h": 4, "step_2_h": 6},
            "yield_pct": {"stage_1_pct": 80, "stage_2_pct": 72},
            "temperature_C": "40 C",
            "solvent": {"step_1": "EtOH", "step_2": "EtOH/water"},
        }
    )

    assert record.reaction_time_h == 10
    assert record.yield_pct == 72
    assert record.temperature_C == 40
    assert record.solvent == "step_1: EtOH; step_2: EtOH/water"
    assert '"step_2": "EtOH/water"' in record.raw_text
