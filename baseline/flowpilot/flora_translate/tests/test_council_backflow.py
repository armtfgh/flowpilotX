from copy import deepcopy
import json

import pytest

from flora_translate.engine.council_v4.backflow import assess_topology, oxygen_option, revised_oxygen_pool, review_and_revise
from flora_translate.schemas import FlowProposal, LabInventory


def topology(valve_position="gas", direction="toward reactor"):
    ops = [{"op_id": "r1", "op_type": "photoreactor"}, {"op_id": "mfc", "op_type": "mfc"},
           {"op_id": "mix", "op_type": "mixer"}, {"op_id": "r2", "op_type": "photoreactor"}]
    edges = [{"stream_id": "liquid", "from_op": "r1", "to_op": "mix", "stream_type": "liquid"},
             {"stream_id": "gas", "from_op": "mfc", "to_op": "mix", "stream_type": "gas"},
             {"stream_id": "out", "from_op": "mix", "to_op": "r2", "stream_type": "gas_liquid"}]
    if valve_position:
        idx = 1 if valve_position == "gas" else 0
        ops.append({"op_id": "cv", "op_type": "check_valve", "parameters": {"direction": direction}})
        before = dict(edges[idx], stream_id="to_cv", to_op="cv")
        edges[idx]["from_op"] = "cv"
        edges.append(before)
    return {"unit_operations": ops, "streams": edges}


def test_gas_valve_does_not_protect_first_reactor():
    result = assess_topology(topology(), FlowProposal(BPR_bar=7), LabInventory())
    exposed = [b for b in result["branches"] if b["exposed_operation_id"] == "r1"]
    assert exposed[0]["status"] == "unprotected_reverse_path"
    assert exposed[0]["reverse_path"] == ["mix", "r1"]
    assert result["laboratory_execution_status"] == "review_required"
    assert any(b["status"] == "protection_declared_requires_verification" for b in result["branches"])


def test_liquid_valve_protects_only_its_branch_and_never_claims_lab_safety():
    result = assess_topology(topology("liquid"), FlowProposal(), LabInventory())
    by_target = {b["exposed_operation_id"]: b for b in result["branches"]}
    assert by_target["r1"]["status"] == "protection_declared_requires_verification"
    assert by_target["mfc"]["status"] == "unprotected_reverse_path"
    assert all(f["resolved"] is False for f in result["findings"])


def test_wrong_direction_does_not_clear_risk():
    result = assess_topology(topology("liquid", "reverse"), FlowProposal(), LabInventory())
    assert any(b["exposed_operation_id"] == "r1" and b["status"] == "unprotected_reverse_path" for b in result["branches"])


def test_non_process_edges_and_liquid_only_do_not_trigger():
    t = topology(None)
    t["streams"][1]["connection_type"] = "utility"
    assert not assess_topology(t, FlowProposal(), LabInventory())["applicable"]
    t["streams"][1].update(connection_type="process", stream_type="liquid")
    assert not assess_topology(t, FlowProposal(), LabInventory())["applicable"]


def test_detector_is_order_independent_and_handles_cycles():
    t = topology()
    first = assess_topology(t, FlowProposal(), LabInventory())
    t["streams"].reverse()
    t["unit_operations"].reverse()
    assert first == assess_topology(t, FlowProposal(), LabInventory())
    t["streams"].append({"stream_id": "cycle", "from_op": "mix", "to_op": "cv", "stream_type": "gas"})
    assert assess_topology(t, FlowProposal(), LabInventory())["applicable"]


@pytest.fixture(scope="module")
def air_case():
    from flora_translate.tests.test_scientific_gas import photo_case
    from flora_translate.engine.council_v4.scientific import build_screen_pool
    p, b, plan, inv = photo_case()
    gas = plan.scientific_context["gas_delivery"]
    gas.update(species="air", reagent_mole_fraction=0.21, identity_source="protocol_fact")
    for feed in [*plan.stream_logic, *[f for s in plan.stages for f in s.feed_streams]]:
        if feed.phase == "gas":
            feed.reagents = ["air"]
            feed.gas_reagent_mole_fraction = 0.21
    for feed in p.streams:
        if feed.phase == "gas":
            feed.contents = ["air"]
            feed.gas_reagent_mole_fraction = 0.21
    inv.gas_hardware[0].gas = "O2, Air"
    # Synthetic wide-range fixture; real KHU hardware is not modified.
    inv.gas_hardware[0].min_flow_sccm = 0.001
    pool, _ = build_screen_pool(p, b, plan, inv)
    return pool, b, plan, inv


def test_no_override_of_chemist_confirmed_air_or_prohibitions(air_case):
    _, _, plan, inv = deepcopy(air_case)
    assert oxygen_option(plan, inv)["eligible"]
    plan.scientific_context["gas_delivery"]["identity_source"] = "chemist_answer"
    assert not oxygen_option(plan, inv)["eligible"]
    plan.scientific_context["gas_delivery"]["identity_source"] = "protocol_fact"
    assert not oxygen_option(plan, inv, {"operating_limits": "Do not use pure oxygen"})["eligible"]
    assert not oxygen_option(plan, inv, {"answers": [{"question_id": "Q-GAS-001", "status": "answered", "answer": "air"}]})["eligible"]
    inv.gas_hardware[0].gas = "Air, N2"
    assert not oxygen_option(plan, inv)["eligible"]


def test_oxygen_revision_closes_all_twelve_and_preserves_evidence(air_case):
    pool, batch, plan, inv = deepcopy(air_case)
    original = plan.model_dump()
    revised, new_plan, checks = revised_oxygen_pool(pool, batch, plan, inv)
    assert len(revised) == len(checks) == 12
    assert plan.model_dump() == original
    assert new_plan.scientific_context["gas_delivery"]["requires_chemist_confirmation"]
    for before, after in zip(pool, revised):
        g0 = next(s for s in before["proposal"]["streams"] if s["phase"] == "gas")
        g1 = next(s for s in after["proposal"]["streams"] if s["phase"] == "gas")
        assert g1["gas_flow_sccm"] == pytest.approx(g0["gas_flow_sccm"] * 0.21, rel=1e-6, abs=1e-6)
        assert g1["gas_reagent_mole_fraction"] == 1
        assert "confirmation required" in g1["pump_role"]
        assert "AIR" not in g1["feed_group"]
        assert "Council" in " ".join(g1["source_evidence"])
        assert any(f["kind"] == "credible_risk" for f in after["backflow_assessment"]["findings"])
        assert after["engineering"]["complete"]
        for stage in after["proposal"]["stage_parameters"]:
            assert stage["residence_time_min"] == pytest.approx(stage["reactor_volume_mL"] / (stage["Q_liquid_mL_min"] + stage["Q_gas_sccm"]), rel=1e-5)


def test_council_can_retain_air_no_forced_oxygen(air_case):
    pool, batch, plan, inv = deepcopy(air_case)
    audit = {}
    def retain(system, request):
        return {"action": "retain", "justification": "Resolve pressure controls first.",
                "addressed_finding_ids": [request["assessment"]["findings"][0]["finding_id"]],
                "required_controls": ["Review liquid-branch protection"], "required_confirmations": ["Gas pressure"],
                "limitations": ["Not an operating-pressure simulation"]}
    new, _, = review_and_revise(pool, batch, plan, inv, None, retain, audit, lambda: None)
    assert audit["backflow_review"]["application_status"] == "retained"
    assert all(next(s for s in r["proposal"]["streams"] if s["phase"] == "gas")["gas_reagent_mole_fraction"] == 0.21 for r in new)


def test_unachievable_oxygen_mfc_flow_does_not_silently_increase_equivalents(air_case):
    pool, batch, plan, inv = deepcopy(air_case)
    inv.gas_hardware[0].min_flow_sccm = 0.01
    with pytest.raises(ValueError, match="preserve stage and delivered oxygen"):
        revised_oxygen_pool(pool, batch, plan, inv)


def test_invalid_council_action_is_not_applied(air_case):
    pool, batch, plan, inv = deepcopy(air_case)
    audit = {}
    with pytest.raises(ValueError, match="Invalid council"):
        review_and_revise(pool, batch, plan, inv, None,
            lambda *_: {"action": "install_unavailable_membrane"}, audit, lambda: None)
    assert audit["backflow_review"]["decision"]["action"] == "install_unavailable_membrane"


@pytest.mark.parametrize("enabled", [False, True])
def test_pipeline_feature_toggle_and_final_gas_consistency(air_case, monkeypatch, tmp_path, enabled):
    import flora_translate.main as pipeline
    from flora_translate.schemas import DesignInputPackage
    pool, batch, plan, inv = deepcopy(air_case)
    p = FlowProposal.model_validate(pool[0]["proposal"])
    package = DesignInputPackage(raw_protocol=batch.raw_text, extracted_batch_fields=batch.model_dump(),
        objective="balanced experimental screen", inventory_constraints=inv.model_dump(),
        engineering_requirements={"gas": plan.scientific_context["gas_delivery"]}, ready_for_design=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "analyze_batch_chemistry", lambda *a, **kw: plan.model_copy(deep=True))
    monkeypatch.setattr(pipeline.VectorRetriever, "retrieve", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline.AnalogySelector, "select", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline.TranslationLLM, "generate", lambda *a, **kw: p.model_copy(deep=True))
    roles = []
    def fake(system, user, tokens):
        q = json.loads(user)
        roles.append(q["role"])
        if q["role"] == "FlowOperability":
            return json.dumps({"action": "propose_pure_oxygen", "justification": "Compare lower gas volume; backflow remains unresolved.",
                "addressed_finding_ids": [q["assessment"]["findings"][0]["finding_id"]],
                "required_controls": ["Liquid-branch protection"], "required_confirmations": ["Oxygen service"],
                "limitations": ["No laboratory validation"]})
        if q["role"].startswith("Dr"):
            assert len(q["context"]["candidates"]) == 12
            return json.dumps({"reviews": [{"candidate_id": n, "recommendation": "acceptable", "hard_violation": False,
                "justification": "Unvalidated screen", "uncertainties": ["Branch protection"]} for n in range(1, 13)]})
        if q["role"] == "Skeptic":
            return json.dumps({"vetoes": [], "assessment": "Unvalidated", "required_measurements": ["pressure"]})
        return json.dumps({"candidate_id": 4, "justification": "Initial screen", "objective_alignment": "Balanced unvalidated screen",
            "alternatives": [{"candidate_id": n, "reason_not_selected": "Different exposure"} for n in (2, 7)],
            "answer_impacts": [], "limitations": ["Backflow remains unresolved"], "next_measurements": ["pressure"]})
    monkeypatch.setattr("flora_translate.engine.llm_agents.call_llm", fake)
    result = pipeline.translate(batch.raw_text, intake_package=package, runtime_options={"design_policy": "scientific_v2",
        "candidate_budget": 12, "council_backflow_review": enabled})
    assert result["final_design"]["status"] == "executable", result["final_design"]["consistency"]
    assert ("FlowOperability" in roles) is enabled
    gas = next(s for s in result["proposal"]["streams"] if s["phase"] == "gas")
    assert gas["gas_reagent_mole_fraction"] == (1 if enabled else 0.21)
    assert result["scientific_assessment"]["selected_design_preserved"]
    assert result["intake_package"]["engineering_requirements"]["gas"]["species"] == "air"
    assert all(s["closure"] for s in result["result_report"]["stages"])
    if enabled:
        assert result["backflow_assessment"]["revision_status"] == "proposed_gas_source_revision"
        assert result["result_report"]["flow_operability"]["laboratory_execution_status"] == "review_required"
        topology_gas = next(o for o in result["process_topology"]["unit_operations"] if o["op_type"] == "mfc")
        assert "air" not in str(topology_gas["parameters"].get("contents", [])).lower()
        assert "Protocol-authorized" not in str(topology_gas)
        assert "Council-proposed" in str(topology_gas)
        assert result["chemistry_plan"]["scientific_context"]["gas_delivery"]["identity_source"] == "council_screening_proposal"
