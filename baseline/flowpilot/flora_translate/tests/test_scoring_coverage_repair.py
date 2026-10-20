from __future__ import annotations

from flora_translate.engine.council_v4 import scoring


def _candidate(candidate_id: int) -> dict:
    return {
        "id": candidate_id,
        "tau_min": 3.0,
        "d_mm": 1.0,
        "Q_mL_min": 0.1,
        "V_R_mL": 0.3,
        "L_m": 0.4,
        "Re": 12.0,
        "delta_P_bar": 0.01,
        "r_mix": 0.1,
        "expected_conversion": 0.8,
        "productivity_mg_h": 20.0,
    }


def test_equal_sized_batch_repairs_only_missing_candidate(monkeypatch) -> None:
    calls: list[set[int]] = []

    def fake_run_scoring_agent(
        agent_name,
        system_prompt,
        context,
        tools,
        valid_ids,
        max_tokens,
    ):
        ids = {int(value) for value in valid_ids}
        calls.append(ids)
        returned_id = min(ids)
        return (
            "scored",
            [{"candidate_id": returned_id, "score": 0.8, "verdict": "PASS"}],
            [],
        )

    monkeypatch.setattr(scoring, "_run_scoring_agent", fake_run_scoring_agent)
    candidates = [_candidate(1), _candidate(2)]

    _, scores, _ = scoring._run_scoring_agent_batched(
        domain="fluidics",
        agent_name="Dr. Fluidics",
        system_prompt="system",
        candidates=candidates,
        chemistry_brief="brief",
        objectives="screen",
        is_photochem=False,
        pump_max_bar=20.0,
        tools=[],
        max_tokens=100,
        batch_size=len(candidates),
        strict_coverage=True,
    )

    assert calls == [{1, 2}, {2}]
    assert [entry["candidate_id"] for entry in scores] == [1, 2]


def test_batched_strict_coverage_uses_labeled_deterministic_fallback(monkeypatch) -> None:
    def no_scores(*args, **kwargs):
        return "", [], []

    monkeypatch.setattr(scoring, "_run_scoring_agent", no_scores)
    candidates = [_candidate(1), _candidate(2)]

    overall, scores, events = scoring._run_scoring_agent_batched(
        domain="kinetics",
        agent_name="Dr. Kinetics",
        system_prompt="system",
        candidates=candidates,
        chemistry_brief="brief",
        objectives="screen",
        is_photochem=False,
        pump_max_bar=20.0,
        tools=[],
        max_tokens=100,
        batch_size=2,
        strict_coverage=True,
    )

    assert [entry["candidate_id"] for entry in scores] == [1, 2]
    assert all(entry["kinetics_score"] == 0.65 for entry in scores)
    assert "no domain-expert inference is claimed" in overall
    assert events[-1]["name"] == "deterministic_scoring_coverage_fallback"


def test_singleton_local_score_restores_requested_candidate_id() -> None:
    scores = [{"candidate_id": 1, "score": 0.75, "verdict": "PASS"}]

    cleaned = scoring._clean_local_scores(scores, {7}, "score")

    assert cleaned == [{"candidate_id": 7, "score": 0.75, "verdict": "PASS"}]


def test_multi_candidate_local_scores_do_not_remap_unknown_ids() -> None:
    scores = [{"candidate_id": 1, "score": 0.75}, {"candidate_id": 99, "score": 0.5}]

    cleaned = scoring._clean_local_scores(scores, {1, 2}, "score")

    assert cleaned == [{"candidate_id": 1, "score": 0.75}]


def test_claude_compact_strict_coverage_uses_labeled_deterministic_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setattr(scoring, "call_llm", lambda *args, **kwargs: "not json")
    candidates = [_candidate(1), _candidate(2)]

    overall, scores, events = scoring._run_scoring_agent_claude_compact(
        domain="fluidics",
        agent_name="Dr. Fluidics",
        candidates=candidates,
        chemistry_brief="brief",
        objectives="screen",
        is_photochem=False,
        pump_max_bar=20.0,
        batch_size=2,
        strict_coverage=True,
    )

    assert [entry["candidate_id"] for entry in scores] == [1, 2]
    assert all(entry["fluidics_score"] == 0.65 for entry in scores)
    assert all(
        entry["score_provenance"]["source"] == "deterministic_coverage_fallback"
        for entry in scores
    )
    assert "no domain-expert inference is claimed" in overall
    assert events == [{
        "name": "deterministic_scoring_coverage_fallback",
        "domain": "fluidics",
        "candidate_ids": [1, 2],
        "reason": "LLM score missing after bounded retries",
    }]


def test_deterministic_coverage_fallback_preserves_hard_fluidics_gate() -> None:
    candidate = _candidate(7)
    candidate["L_m"] = 31.0

    score = scoring._deterministic_coverage_score("fluidics", candidate)

    assert score["candidate_id"] == 7
    assert score["fluidics_score"] == 0.10
    assert score["verdict"] == "BLOCK"
