import json
from pathlib import Path

import pytest

from flora_translate.engine.council_v4 import scoring
from flora_translate.engine.council_v4.chief import _parse_chief


@pytest.fixture(scope="module")
def response_shapes() -> dict[str, str]:
    path = Path(__file__).parent / "fixtures" / "council_response_shapes.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "shape_name",
    ["canonical", "provider_aliases", "id_keyed", "nested"],
)
def test_score_parser_normalizes_provider_equivalent_shapes(
    response_shapes: dict[str, str],
    shape_name: str,
) -> None:
    overall, rows = scoring._parse_score_response(response_shapes[shape_name])

    assert len(rows) == 1
    candidate_id = rows[0].get(
        "candidate_id", rows[0].get("candidateId", rows[0].get("id"))
    )
    assert int(candidate_id) == 1
    assert rows[0]["kinetics_score"] == pytest.approx(0.81)
    if shape_name != "nested":
        assert overall == "ok"


def test_score_parser_recovers_only_complete_rows_from_truncated_output(
    response_shapes: dict[str, str],
) -> None:
    _, rows = scoring._parse_score_response(
        response_shapes["truncated_after_complete_row"]
    )

    assert rows == [{"candidate_id": 1, "kinetics_score": 0.81}]


def test_claude_uses_bounded_scoring_even_when_legacy_flag_is_false(monkeypatch) -> None:
    monkeypatch.setattr(scoring.llm_agents, "ENGINE_PROVIDER", "anthropic")
    monkeypatch.setattr(
        scoring.llm_agents, "ENGINE_MODEL_ANTHROPIC", "claude-sonnet-4-6"
    )
    called = {}

    def fake_bounded(**kwargs):
        called.update(kwargs)
        return "bounded", [{"candidate_id": 1, "kinetics_score": 0.8}], []

    monkeypatch.setattr(scoring, "_run_scoring_agent_claude_compact", fake_bounded)

    overall, rows, tool_calls = scoring.run_kinetics_scoring(
        candidates=[{"id": 1}],
        table_markdown="table",
        chemistry_brief="brief",
        objectives="balanced",
        is_photochem=False,
        pump_max_bar=10.0,
        benchmark_claude_compact_mode=False,
    )

    assert overall == "bounded"
    assert rows[0]["candidate_id"] == 1
    assert tool_calls == []
    assert called["domain"] == "kinetics"


def test_chief_parser_normalizes_alias_and_recovers_truncated_selection() -> None:
    assert _parse_chief('{"winner_id":"2","selection_rationale":"ok"}')[
        "selected_candidate_id"
    ] == "2"

    truncated = '{"selected_candidate_id": 3, "selection_rationale": "unfinished'
    parsed = _parse_chief(truncated)
    assert parsed["selected_candidate_id"] == 3
    assert parsed["response_incomplete"] is True
