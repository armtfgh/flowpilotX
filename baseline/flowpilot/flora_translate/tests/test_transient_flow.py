from copy import deepcopy
import json
import math

import pytest

from flora_translate.engine.council_v4.transient_flow import (
    HydraulicNetwork, P_STP, SCENARIOS, TransientProfile,
    illustrative_profile, simulate, tube_resistance,
)


@pytest.fixture
def network():
    return HydraulicNetwork(liquid_feed_mL_min=0.02, gas_feed_STP_mL_min=0.426909,
        upstream_volume_mL=2, upstream_id_mm=1, downstream_volume_mL=20,
        downstream_id_mm=1, temperature_C=25, BPR_bar_g=7, gas_valve_cracking_bar=1)


def test_poiseuille_units_and_geometry():
    assert tube_resistance(2, 1, 1) == pytest.approx(0.01729214867)
    assert tube_resistance(4, 1, 1) == pytest.approx(tube_resistance(2, 1, 1) * 2)
    # Fixed volume changes length as well: R scales as diameter^-6.
    assert tube_resistance(2, 0.5, 1) == pytest.approx(tube_resistance(2, 1, 1) * 64)
    with pytest.raises(ValueError):
        tube_resistance(1, 0, 1)


@pytest.mark.parametrize("bad", [None, {}, {"gas_supply_bar_g": 9}, {**illustrative_profile().model_dump(), "gas_plenum_mL": float("nan")}])
def test_missing_or_invalid_dynamics_cannot_be_invented(network, bad):
    with pytest.raises(ValueError):
        simulate(network, bad, "steady")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_conservation_finite_and_reproducible(network, scenario):
    p = illustrative_profile()
    result = simulate(network, p, scenario)
    assert result == simulate(network, p, scenario)
    assert result["status"] == "simulated_unvalidated"
    assert result["numerical_checks"]["max_liquid_balance_error_mL"] < 1e-7
    assert result["numerical_checks"]["max_gas_balance_error_STP_mL"] < 1e-7
    assert result["backflow_probability"] is None
    assert result["laboratory_execution_status"] == "review_required"
    assert all(math.isfinite(v) for row in result["trace"] for v in row.values())


def test_equilibrium_and_gauge_to_absolute_conversion(network):
    r = simulate(network, illustrative_profile(), "steady")
    assert not r["reverse_flow_predicted"]
    trace = r["trace"]
    assert trace[-1]["liquid_branch_mL_min"] == pytest.approx(network.liquid_feed_mL_min)
    assert trace[-1]["gas_to_junction_STP_mL_min"] == pytest.approx(network.gas_feed_STP_mL_min)
    assert trace[0]["junction_bar_g"] == pytest.approx(trace[-1]["junction_bar_g"], abs=1e-8)
    res = r["resistances_bar_min_mL"]["downstream_effective"]
    pressure_abs = trace[0]["junction_bar_g"] + P_STP
    gas_actual = network.gas_feed_STP_mL_min * P_STP / pressure_abs * 298.15 / 273.15
    assert trace[0]["junction_bar_g"] - network.BPR_bar_g == pytest.approx(res * (0.02 + gas_actual))


def test_conditional_reversal_is_sensitive_to_gas_volume(network):
    p = illustrative_profile()
    big = simulate(network, p, "gas_first_start")
    small = simulate(network, p.model_copy(update={"gas_plenum_mL": 0.02}), "gas_first_start")
    assert not big["reverse_flow_predicted"]
    assert small["reverse_flow_predicted"]
    assert small["reverse_displacement_uL"] > 0
    assert small["gas_reaching_upstream_reactor"] == "not_modeled"
    assert not small["BPR_opening_pressure_reached"]
    assert "operating pressure was not reached" in small["window_note"]


def test_ideal_liquid_check_and_leaking_valve_differ(network):
    p = illustrative_profile().model_copy(update={"gas_plenum_mL": 0.02})
    ideal = network.model_copy(update={"liquid_valve_cracking_bar": 0.1})
    leaking = ideal.model_copy(update={"liquid_valve_reverse_leak_mL_min_bar": 0.001})
    first = simulate(ideal, p, "gas_first_start")
    second = simulate(leaking, p, "gas_first_start")
    assert not first["reverse_flow_predicted"]
    assert second["reverse_flow_predicted"]
    assert first["laboratory_execution_status"] == "review_required"


def test_time_step_and_tolerance_convergence(network):
    p = illustrative_profile().model_copy(update={"gas_plenum_mL": 0.02})
    first = simulate(network, p, "gas_first_start")
    refined = simulate(network, p, "gas_first_start", max_step_s=0.1, rtol=1e-9)
    assert first["reverse_displacement_uL"] == pytest.approx(refined["reverse_displacement_uL"], rel=1e-4)
    assert first["peak_junction_bar_g"] == pytest.approx(refined["peak_junction_bar_g"], rel=1e-5)


def test_oxygen_reduction_is_not_a_guarantee(network):
    p = illustrative_profile().model_copy(update={"gas_plenum_mL": 0.001, "startup_liquid_delay_s": 20})
    oxygen = network.model_copy(update={"gas_feed_STP_mL_min": network.gas_feed_STP_mL_min * 0.21})
    r = simulate(oxygen, p, "gas_first_start")
    assert r["reverse_flow_predicted"]


def test_supply_dp_is_not_bpr_minus_valve_cracking(network):
    p = illustrative_profile().model_copy(update={"gas_supply_bar_g": 8})
    result = simulate(network, p, "steady")
    assert result["status"] == "not_assessable"
    assert "initial steady state" in result["reason"]
    assert result["backflow_probability"] is None


def test_invalid_model_state_is_not_success(network):
    p = illustrative_profile().model_copy(update={"gas_plenum_mL": 0.001, "liquid_compliance_mL_bar": 10})
    tiny = network.model_copy(update={"upstream_volume_mL": 0.00001})
    r = simulate(tiny, p, "gas_first_start")
    assert r["status"] == "invalid_simulation"
    assert not r["numerical_checks"]["solver_completed"]


def test_runtime_opt_in_and_validation():
    from flora_translate.pipeline_runtime import PipelineRuntimeOptions
    assert PipelineRuntimeOptions().council_physics_profile is None
    with pytest.raises(ValueError, match="require scientific_v2"):
        PipelineRuntimeOptions(council_physics_profile=illustrative_profile().model_dump())
    with pytest.raises(ValueError):
        PipelineRuntimeOptions(design_policy="scientific_v2", council_physics_profile={})
    options = PipelineRuntimeOptions(design_policy="scientific_v2", council_physics_profile=illustrative_profile().model_dump())
    assert options.provenance()["council_physics_profile"]["provenance"] == "illustrative_assumptions"


def test_final_warnings_are_not_dependent_on_chief_text():
    from flora_translate.engine.council_v4.physics_tools import selected_physics_findings
    screen = {"candidate_id": 1, "simulations": [{"variant": "as_designed", "status": "simulated_unvalidated",
        "scenario": "gas_overshoot", "reverse_flow_predicted": True, "reverse_displacement_uL": 0.4,
        "peak_gas_plenum_bar_g": 9.75, "MFC_outlet_rating_screen": {"exceeded_under_assumptions": True,
            "declared_max_pressure_bar": 9}, "evidence_id": "test_evidence"}]}
    findings = selected_physics_findings(screen)
    assert {r["finding_id"] for r in findings} == {"BF-PHYSICS:C1:MFC-PRESSURE", "BF-PHYSICS:C1:LIQUID-REVERSE"}
    assert all(not r["resolved"] and not r["measured_evidence"] for r in findings)
    assert "as-designed" in findings[1]["message"]
    screen["simulations"][0]["variant"] = "hypothetical_liquid_check"
    assert selected_physics_findings(screen)[0]["finding_id"].endswith("INCOMPLETE")


@pytest.fixture(scope="module")
def session_data():
    from flora_translate.tests.test_scientific_gas import photo_case
    from flora_translate.engine.council_v4.scientific import build_screen_pool
    p, b, plan, inv = photo_case()
    plan.scientific_context["gas_delivery"].update(species="air", reagent_mole_fraction=0.21, identity_source="protocol_fact")
    for feed in [*plan.stream_logic, *[f for s in plan.stages for f in s.feed_streams]]:
        if feed.phase == "gas":
            feed.reagents, feed.gas_reagent_mole_fraction = ["air"], 0.21
    for feed in p.streams:
        if feed.phase == "gas":
            feed.contents, feed.gas_reagent_mole_fraction = ["air"], 0.21
    inv.gas_hardware[0].gas = "O2, Air"
    inv.gas_hardware[0].min_flow_sccm = 0.0001
    inv.gas_hardware[0].required_outlet_accessory_type = "check_valve"
    from flora_translate.schemas import SafetyAccessorySpec
    inv.safety_accessories.append(SafetyAccessorySpec(equipment_id="test_gas_valve", type="check_valve",
        quantity=1, cracking_pressure_bar=0.1, max_pressure_bar=10))
    pool, _ = build_screen_pool(p, b, plan, inv)
    profile = illustrative_profile().model_dump()
    profile["pump_pressure_limit_bar_g"] = min(x.max_pressure_bar for x in inv.pumps)
    return p, b, plan, inv, pool, profile


def test_tool_roles_whitelist_assumptions_and_no_mutation(session_data, tmp_path):
    from flora_translate.engine.council_v4.physics_tools import PhysicsToolSession, network_from_candidate, compact_tool_result
    _, b, plan, inv, pool, profile = deepcopy(session_data)
    before = json.dumps(pool, sort_keys=True)
    session = PhysicsToolSession(pool, b, plan, inv, None, profile, tmp_path)
    assert all(r["status"] == "supported_unvalidated" for r in session.inputs), session.inputs
    request = {"tool_name": "simulate_flow_transients", "candidate_ids": [1]}
    with pytest.raises(ValueError, match="assigned only"):
        session.execute("NewAgent", request)
    with pytest.raises(ValueError, match="all 12"):
        session.execute("DrFluidics", request)
    with pytest.raises(ValueError, match="unapproved tool arguments"):
        session.execute("DrSafety", {**request, "gas_supply_bar_g": 500})
    result = session.execute("DrSafety", request)
    assert result["assumption_profile"] == profile
    assert result["candidate_results"][0]["alternative_parameters"]["hypothetical_liquid_check"]["liquid_check_cracking_bar"] == profile["hypothetical_liquid_valve_cracking_bar"]
    assert result["candidate_results"][0]["hypothetical_valve_inventory_item_id"] is None
    assert not result["candidate_results"][0]["hypothetical_valve_service_verified"]
    assert result["candidate_results"][0]["simulations"]
    assert "representative_simulations" in compact_tool_result(result)["candidate_results"][0]
    assert json.dumps(pool, sort_keys=True) == before
    assert len(session.manifest()["tool_calls"]) == 1
    with pytest.raises(ValueError, match="One bounded"):
        session.execute("DrSafety", request)
    roles = session.inputs[0]["equipment_roles"]
    assert roles["gas_branch"]["type"] == "MFC"
    assert roles["liquid_branch"]["type"] == "liquid_pump"
    bad = deepcopy(pool[0])
    next(s for s in bad["proposal"]["streams"] if s["phase"] == "gas")["introduction_stage"] = 1
    with pytest.raises(ValueError, match="two serial"):
        network_from_candidate(bad, session.assessments[0], inv, TransientProfile.model_validate(profile))


def test_full_pipeline_uses_existing_agents_and_publishes_same_design(session_data, monkeypatch, tmp_path):
    import flora_translate.main as pipeline
    from flora_translate.schemas import DesignInputPackage
    p, b, plan, inv, _, profile = deepcopy(session_data)
    package = DesignInputPackage(raw_protocol=b.raw_text, extracted_batch_fields=b.model_dump(),
        objective="balanced screen", inventory_constraints=inv.model_dump(),
        engineering_requirements={"gas": plan.scientific_context["gas_delivery"]}, ready_for_design=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "analyze_batch_chemistry", lambda *a, **kw: plan.model_copy(deep=True))
    monkeypatch.setattr(pipeline.VectorRetriever, "retrieve", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline.AnalogySelector, "select", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline.TranslationLLM, "generate", lambda *a, **kw: p.model_copy(deep=True))
    calls = []
    def fake(system, user, tokens):
        q = json.loads(user)
        calls.append((q["role"], q.get("phase", "review")))
        if q.get("phase") == "tool_request":
            return json.dumps({"tool_name": "simulate_flow_transients", "candidate_ids": list(range(1, 13))})
        if q["role"].startswith("Dr"):
            response = {"reviews": [{"candidate_id": i, "recommendation": "acceptable", "hard_violation": False,
                "justification": "Synthetic wiring test", "uncertainties": ["Not calibrated"]} for i in range(1, 13)]}
            if q["role"] in {"DrFluidics", "DrSafety"}:
                r = q["context"]["physics_tool_results"][q["role"]]
                response["physics_assessment"] = {"tool_result_id": r["result_id"],
                    "evidence_ids": [r["candidate_results"][0]["representative_simulations"][0]["evidence_id"]],
                    "proposed_alternatives": ["Check liquid valve, unapproved"], "limitations": ["Assumed dynamics"]}
            return json.dumps(response)
        assert set(q["context"]["physics_reviewer_assessments"]) == {"DrFluidics", "DrSafety"}
        if q["role"] == "Skeptic":
            return json.dumps({"vetoes": [], "assessment": "Synthetic test", "required_measurements": ["pressure"]})
        return json.dumps({"candidate_id": 1, "justification": "Synthetic choice", "objective_alignment": "Screen only",
            "alternatives": [{"candidate_id": i, "reason_not_selected": "Synthetic"} for i in (2, 3)],
            "answer_impacts": [], "limitations": ["Assumptions"], "next_measurements": ["pressure"]})
    monkeypatch.setattr("flora_translate.engine.llm_agents.call_llm", fake)
    result = pipeline.translate(b.raw_text, intake_package=package,
        runtime_options={"design_policy": "scientific_v2", "council_physics_profile": profile})
    assert len(calls) == 8
    assert {r for r, _ in calls} == {"DrChemistry", "DrKinetics", "DrFluidics", "DrSafety", "Skeptic", "Chief"}
    review = result["scientific_assessment"]["physics_review"]
    assert len(review["tool_calls"]) == 2
    assert set(review["reviewer_assessments"]) == {"DrFluidics", "DrSafety"}
    assert result["final_design"]["status"] == "executable"
    assert result["scientific_assessment"]["selected_design_preserved"]
    assert not review["design_modification_applied"]
    assert result["result_report"]["flow_operability"]["physics_screen"]["backflow_probability"] is None
    assert next(s for s in result["proposal"]["streams"] if s["phase"] == "gas")["gas_reagent_mole_fraction"] == 0.21
