import json
import pytest

from flora_translate.tests.test_scientific_evidence import example
from flora_translate.schemas import FlowProposal, LabInventory, StreamAssignment
from flora_translate.chemistry_contract import reconcile_chemistry_plan
from flora_translate.engine.council_v4.scientific import build_screen_pool, signature, run_scientific_council
from flora_translate.design_realizer import realize_executable_design


def test_json_review_retry_preserves_raw_and_increases_budget():
    from unittest.mock import Mock
    from flora_translate.engine.council_v4.scientific import _call_json_review
    call = Mock(side_effect=['{"assessment":"truncated', '{"assessment":"valid", "vetoes":[]}'])
    audit = {"calls": []}
    save = Mock()
    result = _call_json_review(call, 'system', {"role":"Skeptic"}, 4500, audit, save)
    assert result == {"assessment":"valid", "vetoes":[]}
    assert [c.args[2] for c in call.call_args_list] == [4500, 9000]
    assert 'parse_error' in audit['calls'][0]
    assert audit['calls'][0]['raw_response'] == '{"assessment":"truncated'
    assert len(audit['calls']) == 2


def test_json_review_retry_is_bounded_and_does_not_fabricate_success():
    from unittest.mock import Mock
    from flora_translate.engine.council_v4.scientific import _call_json_review
    call = Mock(return_value='not JSON')
    audit = {"calls": []}
    with pytest.raises(ValueError, match='after two recorded attempts'):
        _call_json_review(call, 'system', {"role":"Chief"}, 4500, audit, lambda:None)
    assert call.call_count == 2
    assert all('parse_error' in c for c in audit['calls'])


def case():
    b, plan = example()
    for st in plan.stages:
        st.temperature_C = 95
    plan, _ = reconcile_chemistry_plan(b, plan, scientific=True)
    for st in plan.stages:
        for feed in st.feed_streams:
            feed.concentration_M = 0.5 if st.stage_number == 1 else 2.1
            feed.molar_equiv = 1 if st.stage_number == 1 else 1.05
    p = FlowProposal(flow_rate_mL_min=0.25, residence_time_min=10, reactor_volume_mL=10,
        tubing_ID_mm=1.016, tubing_material="PFA", BPR_bar=2, temperature_C=95, concentration_M=0.5,
        streams=[StreamAssignment(stream_label="A", contents=["Acid", "DPDTC", "DMAP"], concentration_M=0.5, molar_equiv=1),
                 StreamAssignment(stream_label="B", contents=["Benzylamine"], concentration_M=2.1, molar_equiv=1.05, introduction_stage=2)])
    inv = LabInventory.model_validate({"pumps": [{"equipment_id": "pump", "name": "Syringe", "type": "syringe", "quantity": 3,
         "min_flow_rate_mL_min": 0.0001, "max_flow_rate_mL_min": 10, "max_pressure_bar": 5}],
         "reactors": [{"equipment_id": f"coil{v}", "type": "coil", "material": "PFA", "quantity": 2,
               "volume_mL": v, "ID_mm": 1.016, "max_temperature_C": 80, "max_pressure_bar": 10} for v in (2, 5, 10)],
         "tubing": [{"equipment_id": "tubing", "material": "PFA", "ID_mm": 1.016, "max_temperature_C": 80, "max_pressure_bar": 10, "transparent": True}],
         "pressure_controllers": [{"equipment_id": "bpr", "type": "BPR", "setpoints_bar": [2], "max_pressure_bar": 5}]})
    return p, b, plan, inv


def test_twelve_distinct_reproducible_complete_designs_and_preservation():
    p, b, plan, inv = case()
    pool, rejected = build_screen_pool(p, b, plan, inv)
    again, _ = build_screen_pool(p, b, plan, inv)
    assert pool == again
    assert len(pool) == len({json.dumps(c["proposal"], sort_keys=True) for c in pool}) == 12
    for row in pool:
        prop = FlowProposal.model_validate(row["proposal"])
        assert row["engineering"]["complete"]
        assert len(row["temperature_deviations"]) == 2
        for stage in prop.stage_parameters:
            assert stage["residence_time_min"] == pytest.approx(stage["reactor_volume_mL"] / stage["Q_liquid_mL_min"], rel=2e-5)
            assert stage["temperature_C"] == 80
        rerun, _, val = realize_executable_design(prop, batch_record=b, chemistry_plan=plan.model_copy(deep=True), inventory=inv)
        assert signature(rerun) == signature(prop)
        assert all(val["checks"].values())


def test_constraints_change_pool_without_inventing_reactors():
    p, b, plan, inv = case()
    first, _ = build_screen_pool(p, b, plan, inv)
    inv.reactors = [r for r in inv.reactors if r.volume_mL >= 5]
    second, _ = build_screen_pool(p, b, plan, inv)
    assert first != second
    assert all(s["reactor_volume_mL"] in (5, 10) for row in second for s in row["proposal"]["stage_parameters"])


def test_too_narrow_inventory_does_not_duplicate_candidates():
    p, b, plan, inv = case()
    inv.reactors = [r for r in inv.reactors if r.volume_mL == 10]
    with pytest.raises(ValueError, match="12 distinct feasible"):
        build_screen_pool(p, b, plan, inv)


def test_council_requires_all_four_reviews_and_retains_winner(monkeypatch, tmp_path):
    p, b, plan, inv = case()
    monkeypatch.chdir(tmp_path)
    roles = []
    def fake(system, user, max_tokens):
        q = json.loads(user)
        roles.append(q["role"])
        assert len(q["context"]["candidates"]) == 12
        if q["role"].startswith("Dr"):
            return json.dumps({"reviews": [{"candidate_id": n, "recommendation": "reject" if n == 1 else "acceptable", "hard_violation": False,
                  "justification": "A screening hypothesis, not measured performance.", "uncertainties": ["yield unknown"]} for n in range(1, 13)]})
        if q["role"] == "Skeptic":
            return json.dumps({"vetoes": [], "assessment": "Review temperature", "required_measurements": ["stage1 conversion"]})
        assert 1 in q["eligible_ids"], "A negative preference is not a hard constraint"
        return json.dumps({"candidate_id": 4, "justification": "Balanced initial screen", "objective_alignment": "Baseline screen with unknown yield", "alternatives": [{"candidate_id": n, "reason_not_selected": "Different exposure for subsequent testing"} for n in (2, 7)], "answer_impacts": [], "limitations": ["No measured kinetics"], "next_measurements": ["yield"]})
    monkeypatch.setattr("flora_translate.engine.llm_agents.call_llm", fake)
    candidate, calc = run_scientific_council(p, b, plan, inv, [], "High yield without isolation")
    assert len(roles) == 6
    assert len(candidate.proposal.scientific_design["calls"]) == 6
    assert candidate.proposal.scientific_design["selected_signature"] == signature(candidate.proposal)
    assert candidate.proposal.scientific_design["selected_candidate_id"] == 4
    assert calc.rate_constant is None


def test_missing_review_blocks_and_saves_raw_failure(monkeypatch, tmp_path):
    p, b, plan, inv = case()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("flora_translate.engine.llm_agents.call_llm", lambda *a: '{"reviews": []}')
    with pytest.raises(ValueError, match="12 complete"):
        run_scientific_council(p, b, plan, inv, [], "High yield")
    audit = json.loads(next(tmp_path.glob("outputs/scientific_council/*/audit.json")).read_text())
    assert audit["calls"][0]["raw_response"] == '{"reviews": []}'


def test_photochemical_screen_without_lights_is_rejected():
    p, b, plan, inv = case()
    plan.stages[0].requires_light = True
    with pytest.raises(ValueError, match="12 distinct feasible"):
        build_screen_pool(p, b, plan, inv)


def test_bpr_at_pump_limit_is_not_feasible_and_lower_inventory_setting_is_screened():
    p, b, plan, inv = case()
    p.BPR_bar = 5
    inv.pressure_controllers[0].setpoints_bar = [2, 5]
    pool, rejected = build_screen_pool(p, b, plan, inv)
    assert all(c["proposal"]["BPR_bar"] == 2 for c in pool)
    assert any("pump/reactor pressure headroom" in c["reasons"] for c in rejected)
