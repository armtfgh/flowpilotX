import json

import pytest
from pydantic import ValidationError

from flora_translate.schemas import DesignInputPackage
from flora_translate.intake_agent import intake_context_block
from flora_translate.scientific_objective import resolve_objective, screen_patterns, pool_fingerprint, answer_effects
from flora_translate.tests.test_scientific_council import case
from flora_translate.engine.council_v4.scientific import build_screen_pool


@pytest.mark.parametrize("text,priority", [
    ("Develop a safe flow process with high yield", "balanced"),
    ("Maximize the final amide yield across the complete two-stage sequence.", "yield_priority"),
    ("  MAXIMISE   the final amide YIELD. ", "yield_priority"),
    ("Prioritize yield over throughput", "yield_priority"),
    ("Maximize throughput", "throughput_priority"),
    ("Identify a process that reduces the overall processing time while maintaining high final yield.", "throughput_priority"),
    ("Do not reduce the overall processing time", "balanced"),
    ("Maximize yield; reduce overall processing time", "balanced"),
    ("Do not maximize throughput", "balanced"),
    ("Never prioritize yield over safety", "balanced"),
    ("Maximize yield; maximize throughput", "balanced"),
    ("Maximize yield and throughput", "balanced"),
    ("Maximize yield and conversion", "yield_priority"),
])
def test_deterministic_intent_and_negation(text, priority):
    assert resolve_objective(text)["priority"] == priority
    assert resolve_objective(text) == resolve_objective(text)


def test_explicit_priority_roundtrip_and_prompt():
    p = DesignInputPackage(objective="High yield", screening_priority="throughput_priority")
    assert DesignInputPackage.model_validate_json(p.model_dump_json()) == p
    assert resolve_objective(p.objective, p)["basis"] == "chemist_selected"
    assert resolve_objective(p.objective, p)["priority"] == "throughput_priority"
    assert "Screening priority selection: throughput_priority" in intake_context_block(p)
    from flora_translate.intake_agent import IntakeAgent
    p.raw_protocol = "Heat acid in solvent at 80 C for 30 min."
    assert IntakeAgent().analyze(existing_package=p, use_llm=False).screening_priority == p.screening_priority
    with pytest.raises(ValidationError):
        DesignInputPackage(screening_priority="invented")


@pytest.mark.parametrize("count", [1, 2])
def test_patterns_keep_twelve_unique_experiments_and_baseline(count):
    for priority in ("balanced", "yield_priority", "throughput_priority"):
        patterns = screen_patterns(priority, count)
        assert len(patterns) == len(set(patterns)) == 12
        assert (1,) * count in patterns


def test_objective_changes_physical_pool_with_identical_chemistry_and_inventory():
    p, b, plan, inv = case()
    pools = {}
    for objective in ("High yield", "Maximize yield", "Maximize throughput"):
        intent = resolve_objective(objective)
        pool, _ = build_screen_pool(p, b, plan, inv, objective_policy=intent)
        again, _ = build_screen_pool(p, b, plan, inv, objective_policy=intent)
        assert pool_fingerprint(pool) == pool_fingerprint(again)
        assert len(pool) == 12
        assert all(all(c["validation"]["checks"].values()) for c in pool)
        assert all(c["pressure_headroom"]["passed"] for c in pool)
        pools[intent["priority"]] = pool
    assert len({pool_fingerprint(v) for v in pools.values()}) == 3
    # A priority does not erase the common control experiments or force a winner.
    assert pools["balanced"][3]["proposal"] == pools["yield_priority"][3]["proposal"]
    assert sum(min(c["objective_fit"]["stage_hold_ratios"]) >= 1.5 for c in pools["yield_priority"]) > sum(min(c["objective_fit"]["stage_hold_ratios"]) >= 1.5 for c in pools["balanced"])


def test_hypotheses_do_not_silently_become_hard_time_limits():
    objective = "Maximize yield"
    assert resolve_objective(objective, {"hypotheses": ["Longer is better"]}) == resolve_objective(objective, {"hypotheses": ["Shorter is better"]})
    p = {"answers": [{"question_id": "Q-HYP-001", "status": "answered", "answer": "Longer might help"}]}
    effect = answer_effects(p, resolve_objective(objective), {})
    assert "not measured rates" in effect["questions"][0]["effect"]


def test_same_semantic_priority_produces_same_screen():
    first = resolve_objective("Maximize final amide yield.")
    second = resolve_objective("Prioritize final product yield over throughput.")
    assert screen_patterns(first["priority"], 2) == screen_patterns(second["priority"], 2)


def test_council_comparisons_use_actual_not_target_hold_ratios():
    from flora_translate.engine.council_v4.scientific import comparison_facts
    p, b, plan, inv = case()
    pool, _ = build_screen_pool(p, b, plan, inv)
    facts = comparison_facts(pool, plan.scientific_context)
    assert len(facts["candidates"]) == 12
    for source, row in zip(pool, facts["candidates"]):
        assert row["total_stage_time_min"] == sum(s["residence_time_min"] for s in source["proposal"]["stage_parameters"])
        for stage in row["actual_stage_hold_ratios"]:
            assert stage["ratio"] == stage["flow_min"] / stage["batch_min"]
    assert "union is not" in facts["equipment_boundary"]
    assert "do not compare cooled" in facts["scope"]


def test_temperature_reference_is_original_protocol_not_model_proposal():
    from flora_translate.scientific_evidence import source_context
    from flora_translate.scientific_objective import source_stage_temperature
    p, b, plan, inv = case()
    b.raw_text = "Acid, DPDTC and DMAP were stirred at 95 deg C for 30 min. Benzylamine was added and heated at 105 C for another 30 min."
    for stage in plan.stages:
        stage.temperature_C = 80
    plan.scientific_context = source_context(b, plan)
    assert [source_stage_temperature(plan, n) for n in (1, 2)] == [95, 105]
    assert [h["temperature_quote"] for h in plan.scientific_context["timed_holds"]] == ["95 deg C", "105 C"]
    for h in plan.scientific_context["timed_holds"]:
        start = h["temperature_source_start"]
        assert b.raw_text[start:start + len(h["temperature_quote"])] == h["temperature_quote"]
    # A model proposing 80 C must not erase the source temperature deviation.
    pool, _ = build_screen_pool(p, b, plan, inv)
    assert all([d["batch_temperature_C"] for d in c["temperature_deviations"]] == [95, 105] for c in pool)


def test_mixed_case_numbered_solvent_expansion_matches_exact_protocol_alias():
    from flora_translate.scientific_evidence import aliases, identify
    name = "2-MeTHF (2-methyltetrahydrofuran)"
    assert "2-MeTHF" in aliases(name)
    component = {"name": "2-MeTHF", "aliases": ["2-MeTHF"]}
    assert identify(name, [component]) == component
    assert identify("3-MeTHF (3-methyltetrahydrofuran)", [component]) is None
    assert identify("2-methyltetrahydrofuran (2-MeTHF)", [component]) == component
    assert "DPDTC" in aliases("DPDTC(di(2-pyridyl) dithiocarbonate)")
    assert "PPh3" not in aliases("Pd(PPh3)")


@pytest.mark.parametrize("fault", ["alignment", "alternatives", "invented_question"])
def test_unexplained_selection_or_invented_answer_reference_is_not_published(monkeypatch, tmp_path, fault):
    from flora_translate.engine.council_v4.scientific import run_scientific_council
    p, b, plan, inv = case()
    monkeypatch.chdir(tmp_path)
    def fake(system, user, max_tokens):
        request = json.loads(user)
        assert request["context"]["objective_policy"]["priority"] == "yield_priority"
        role = request["role"]
        if role.startswith("Dr"):
            return json.dumps({"reviews": [{"candidate_id": n, "recommendation": "acceptable", "hard_violation": False, "justification": "Unknown kinetics"} for n in range(1, 13)]})
        if role == "Skeptic":
            return json.dumps({"vetoes": [], "assessment": "No known hard violations"})
        choice = {"candidate_id": 7, "justification": "Explore longer times", "objective_alignment": "Tests time adequacy without claiming higher yield", "alternatives": [{"candidate_id": n, "reason_not_selected": "Shorter control for later comparison"} for n in (1, 4)]}
        if fault == "alignment":
            choice.pop("objective_alignment")
        elif fault == "alternatives":
            choice["alternatives"][0]["candidate_id"] = 99
        else:
            choice["answer_impacts"] = [{"question_id": "Q-INVENTED", "effect": "Asserted influence"}]
        return json.dumps(choice)
    monkeypatch.setattr("flora_translate.engine.llm_agents.call_llm", fake)
    with pytest.raises(ValueError):
        run_scientific_council(p, b, plan, inv, [], "Maximize yield")
    audit = json.loads(next(tmp_path.glob("outputs/scientific_council/*/audit.json")).read_text())
    assert len(audit["calls"]) == 6
    assert "selected_candidate_id" not in audit
