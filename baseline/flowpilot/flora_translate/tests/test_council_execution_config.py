from __future__ import annotations

import pytest

from flora_translate.engine.council_v4 import chief, scoring
from flora_translate.engine.council_v4.execution_config import CouncilExecutionConfig
from flora_translate.pipeline_runtime import PipelineRuntimeOptions


def test_execution_config_round_trip_and_validation():
    config = CouncilExecutionConfig.coerce(
        {
            "condition_id": "minus_fluidics",
            "enabled_scoring_agents": ["chemistry", "kinetics", "safety"],
            "enable_skeptic": False,
        }
    )
    assert config.provenance()["enabled_scoring_agents"] == (
        "chemistry",
        "kinetics",
        "safety",
    )
    runtime = PipelineRuntimeOptions(council_execution=config.provenance())
    assert runtime.provenance()["council_execution"]["condition_id"] == "minus_fluidics"
    with pytest.raises(ValueError, match="Unknown council scoring agents"):
        CouncilExecutionConfig(enabled_scoring_agents=("chemistry", "unknown"))


def test_disabled_scoring_agents_make_no_calls(monkeypatch):
    calls: list[str] = []

    def fake(domain, score_key):
        def run(**_kwargs):
            calls.append(domain)
            return "ok", [{"candidate_id": 1, score_key: 0.8, "verdict": "PASS"}], []
        return run

    monkeypatch.setattr(scoring, "run_chemistry_scoring", fake("chemistry", "combined_score"))
    monkeypatch.setattr(scoring, "run_kinetics_scoring", fake("kinetics", "kinetics_score"))
    monkeypatch.setattr(scoring, "run_fluidics_scoring", fake("fluidics", "fluidics_score"))
    monkeypatch.setattr(scoring, "run_safety_scoring", fake("safety", "safety_score"))
    monkeypatch.setattr(
        scoring,
        "enrich_scoring_with_flow_values",
        lambda _candidates, result, **_kwargs: result,
    )

    result = scoring.run_domain_scoring(
        candidates=[{"id": 1}],
        table_markdown="",
        chemistry_brief="",
        objectives="balanced",
        is_photochem=False,
        pump_max_bar=10.0,
        enabled_agents=("chemistry", "safety"),
    )

    assert calls == ["chemistry", "safety"]
    assert result["kinetics_scores"] == []
    assert result["fluidics_scores"] == []
    assert result["disabled_scoring_agents"] == ["fluidics", "kinetics"]


def test_weighting_renormalizes_remaining_agents():
    candidate = {
        "id": 1,
        "tau_min": 10.0,
        "d_mm": 1.0,
        "L_m": 1.0,
        "Q_mL_min": 0.1,
        "deltaP_bar": 0.1,
        "Re": 10.0,
    }
    scoring_result = {
        "enabled_scoring_agents": ["chemistry"],
        "chemistry_scores": [{"candidate_id": 1, "combined_score": 0.8}],
        "kinetics_scores": [],
        "fluidics_scores": [],
        "safety_scores": [],
        "process_value_scores": [],
    }
    rows = chief.compute_weighted_scores(
        [candidate], scoring_result, "balanced", set()
    )
    row = rows[0]
    expected = (
        chief._WEIGHTS["chemistry"] * row["chemistry"]
        + chief._WEIGHTS["geometry"] * row["geometry"]
    ) / (chief._WEIGHTS["chemistry"] + chief._WEIGHTS["geometry"])
    assert row["legacy_domain_combined"] == pytest.approx(expected, abs=1e-4)
    assert row["enabled_scoring_agents"] == ["chemistry"]


def test_bypassed_skeptic_is_explicit():
    audit = chief._bypassed_skeptic_audit([{"id": 1}, {"id": 2}])
    assert audit["verdict"] == "BYPASSED"
    assert audit["module_disabled"] is True
    assert audit["council_may_proceed"] is True
    assert audit["disqualify_ids"] == []
