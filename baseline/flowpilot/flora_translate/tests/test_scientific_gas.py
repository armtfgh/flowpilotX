from types import SimpleNamespace
import json
from pathlib import Path
import pytest

from flora_translate.scientific_gas import gas_context
from flora_translate.residence_time_basis import stp_gas_flow_for_equiv
from flora_translate.schemas import StreamAssignment, FlowProposal


def test_stp_equivalents_do_not_change_with_pressure():
    rate = stp_gas_flow_for_equiv(0.02, 0.1, 2.0)
    p = FlowProposal(BPR_bar=2, streams=[StreamAssignment(stream_label="G", phase="gas", contents=["O2"],
        molar_equiv=2, gas_reagent_mole_fraction=1, gas_flow_sccm=rate)])
    calc = SimpleNamespace(concentration_M=0.1, temperature_C=40, residence_time_min=30)
    first = gas_context(calc, 0.02, p, None, True)
    p.BPR_bar = 8
    second = gas_context(calc, 0.02, p, None, True)
    assert first["gas_sccm"] == second["gas_sccm"] == rate
    assert first["gas_equiv_supplied"] == pytest.approx(2)
    assert second["gas_equiv_supplied"] == pytest.approx(2)
    assert second["gas_actual_mL_min"] < first["gas_actual_mL_min"]
    assert "kLa_s" not in first


def test_frozen_gas_does_not_invent_missing_feed():
    p = FlowProposal(streams=[StreamAssignment(stream_label="G", phase="gas", contents=["O2"], molar_equiv=2)])
    calc = SimpleNamespace(concentration_M=0.1, temperature_C=40, residence_time_min=30)
    with pytest.raises(ValueError, match="explicit inlet"):
        gas_context(calc, 0.02, p, None, True)


def test_gas_target_is_not_replaced_by_supplied_equivalents():
    rate = stp_gas_flow_for_equiv(0.02, 0.1, 1.0)
    p = FlowProposal(BPR_bar=2, streams=[StreamAssignment(stream_label="G", phase="gas",
        contents=["O2 (pure gas, proposed)"], molar_equiv=2, gas_flow_sccm=rate)])
    calc = SimpleNamespace(concentration_M=0.1, temperature_C=40, residence_time_min=30)
    context = gas_context(calc, 0.02, p, None, True)
    assert context["target_gas_equiv_inlet"] == 2
    assert context["gas_equiv_supplied"] == pytest.approx(1)
    assert context["oxygen_fraction"] == 1
    with pytest.raises(ValueError, match="Positive liquid flow"):
        gas_context(calc, 0, p, None, True)


def test_explicit_gas_ratio_required_before_schema_defaults():
    from flora_translate.scientific_evidence import explicit_gas_ratios
    raw = {"stream_logic": [{"stream_label": "G", "phase": "gas", "molar_equiv": None}]}
    with pytest.raises(ValueError, match="must not default"):
        explicit_gas_ratios(raw)
    raw["stream_logic"][0]["molar_equiv"] = 3
    raw["stages"] = [{"feed_streams": [{"stream_label": "G", "phase": "gas"}]}]
    parsed = explicit_gas_ratios(raw)
    assert parsed["stages"][0]["feed_streams"][0]["molar_equiv"] == 3
    assert "molar_equiv" not in raw["stages"][0]["feed_streams"][0]
    raw["stages"][0]["feed_streams"][0]["molar_equiv"] = 2
    with pytest.raises(ValueError, match="Conflicting"):
        explicit_gas_ratios(raw)


def test_only_confirmed_gas_identity_can_override_batch_air():
    from flora_translate.design_realizer import _protocol_reagent_gas
    from flora_translate.schemas import BatchRecord, ChemistryPlan
    b = BatchRecord(raw_text="Oxidation under ambient air.")
    p = ChemistryPlan(scientific_context={"gas_delivery": {"species": "O2", "identity_source": "model_inference"}})
    assert _protocol_reagent_gas(b, p) == "air"
    p.scientific_context["gas_delivery"]["identity_source"] = "chemist_answer"
    assert _protocol_reagent_gas(b, p) == "O2"


def test_state_descriptor_does_not_change_component_identity():
    from flora_translate.scientific_evidence import aliases
    assert "pH 9 buffer" in aliases("pH 9 buffer (aqueous)")
    assert "pH 7 buffer" not in aliases("pH 9 buffer (aqueous)")
    assert "copper sulfate" not in aliases("copper sulfate (pentahydrate)")
    assert "EtOH" in aliases("Ethanol")
    assert "ethanol" in aliases("EtOH")
    assert "MeOH" not in aliases("Ethanol")
    assert "THF" not in aliases("2-MeTHF")
    assert "pH 9 buffer" in aliases("pH 9 aqueous buffer")
    assert "pH 7 buffer" not in aliases("pH 9 aqueous buffer")


def test_active_air_exposure_is_recognized_after_an_inert_stage():
    from flora_translate.chemistry_contract import _protocol_reagent_gases
    from flora_translate.schemas import BatchRecord
    b = BatchRecord(atmosphere="argon", raw_text="React under argon. Remove the cap to expose the reaction mixture to ambient air.")
    assert _protocol_reagent_gases(b) == ["air"]
    b.raw_text = "Do not expose the reaction mixture to ambient air."
    assert _protocol_reagent_gases(b) == []


def photo_case():
    from flora_translate.schemas import BatchRecord, ChemistryPlan, LabInventory
    data = json.loads((Path(__file__).parent / "fixtures/scientific_giese.json").read_text())
    return (FlowProposal.model_validate(data["proposal"]), BatchRecord.model_validate(data["batch"]),
            ChemistryPlan.model_validate(data["plan"]), LabInventory.model_validate(data["inventory"]))


def test_twelve_photo_gas_screens_are_reproducible_and_closed():
    from flora_translate.engine.council_v4.scientific import build_screen_pool, signature
    from flora_translate.design_realizer import realize_executable_design
    from flora_translate.residence_time_basis import gas_equiv_from_stp_flow
    p, b, plan, inv = photo_case()
    pool, _ = build_screen_pool(p, b, plan, inv)
    again, _ = build_screen_pool(p, b, plan, inv)
    assert pool == again
    assert len(pool) == 12
    for candidate in pool:
        p = FlowProposal.model_validate(candidate["proposal"])
        assert candidate["engineering"]["complete"]
        assert candidate["pressure_headroom"]["passed"]
        assert candidate["inventory_allocation"]["checks"]["all_required_operations_assigned"]
        stages = p.stage_parameters
        assert stages[0]["Q_gas_sccm"] == 0
        assert stages[1]["Q_gas_sccm"] > 0
        assert len({s["light_equipment_id"] for s in stages}) == 2
        assert all(425 <= s["wavelength_nm"] <= 477 for s in stages)
        gas = next(s for s in p.streams if s.phase == "gas")
        liquid = next(s for s in p.streams if s.phase == "liquid")
        assert gas.introduction_stage == 2
        assert gas.molar_equiv == pytest.approx(gas_equiv_from_stp_flow(gas.gas_flow_sccm, liquid.flow_rate_mL_min, liquid.concentration_M, gas.gas_reagent_mole_fraction), abs=0.0001)
        for stage in stages:
            assert stage["residence_time_min"] == pytest.approx(stage["reactor_volume_mL"] / (stage["Q_liquid_mL_min"] + stage["Q_gas_sccm"]), abs=0.001)
        for stage in candidate["engineering"]["stages"]:
            assert stage["calculations"]["rate_constant"] is None
            assert stage["calculations"]["consistent"], stage["calculations"]["consistency_notes"]
        gas_calc = candidate["engineering"]["stages"][1]["calculations"]
        assert gas_calc["kLa_s"] is None
        assert gas_calc["o2_transfer_sufficiency"] is None
        from flora_translate.design_calculator import DesignCalculations, StepResult
        from dataclasses import fields
        allowed = {f.name for f in fields(DesignCalculations)}
        formatted_calc = {k: v for k, v in gas_calc.items() if k in allowed}
        formatted_calc["steps"] = [StepResult(**s) for s in gas_calc["steps"]]
        block = DesignCalculations(**formatted_calc).to_prompt_block()
        assert "oxygen transfer sufficiency unknown" in block
        final, _, _ = realize_executable_design(p, batch_record=b, chemistry_plan=plan.model_copy(deep=True), inventory=inv, operating_limits=plan.scientific_context["operating_limits"])
        assert signature(final) == signature(p)


@pytest.mark.parametrize("missing", ["light_sources", "gas_hardware"])
def test_missing_required_hardware_does_not_get_a_fabricated_assignment(missing):
    from flora_translate.engine.council_v4.scientific import build_screen_pool
    p, b, plan, inv = photo_case()
    setattr(inv, missing, [])
    with pytest.raises(ValueError):
        build_screen_pool(p, b, plan, inv)


def test_full_pipeline_preserves_council_gas_pressure_and_streams(monkeypatch, tmp_path):
    import flora_translate.main as pipeline
    from flora_translate.design_calculator import DesignCalculator
    from flora_translate.engine.council_v4.scientific import build_screen_pool, signature
    from flora_translate.schemas import DesignCandidate, DesignInputPackage
    p, b, plan, inv = photo_case()
    package = DesignInputPackage(raw_protocol=b.raw_text, extracted_batch_fields=b.model_dump(),
        objective="balanced experimental screen", inventory_constraints=inv.model_dump(),
        engineering_requirements={"gas": plan.scientific_context["gas_delivery"]}, ready_for_design=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "analyze_batch_chemistry", lambda *a, **kw: plan.model_copy(deep=True))
    monkeypatch.setattr(pipeline.VectorRetriever, "retrieve", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline.AnalogySelector, "select", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline.TranslationLLM, "generate", lambda *a, **kw: p.model_copy(deep=True))
    selected = {}

    def council(self, proposal, batch, analogies, inventory, **kwargs):
        chemistry = kwargs["chemistry_plan"]
        pool, _ = build_screen_pool(proposal, batch, chemistry, inventory)
        chosen = FlowProposal.model_validate(pool[0]["proposal"])
        selected.update(signature(chosen))
        chosen.scientific_design.update({"candidates": pool, "candidate_count": 12,
            "selected_candidate_id": pool[0]["candidate_id"], "selected_signature": signature(chosen),
            "chief": {"justification": "Synthetic selection for lifecycle regression only."}})
        return DesignCandidate(proposal=chosen, chemistry_plan=chemistry), DesignCalculator().run(batch, chemistry_plan=chemistry, proposal=chosen, inventory=inventory)

    monkeypatch.setattr(pipeline.CouncilV4, "run", council)
    result = pipeline.translate(b.raw_text, intake_package=package, runtime_options={"design_policy": "scientific_v2", "candidate_budget": 12})
    assert result["final_design"]["status"] == "executable", result["final_design"]["consistency"]
    assert result["proposal"]["BPR_bar"] == 2
    assert signature(FlowProposal.model_validate(result["proposal"])) == selected
    assert all(row["closure"] for row in result["result_report"]["stages"])


def test_mixed_stage_tubing_is_checked_per_assigned_reactor():
    from flora_translate.engine.council_v4.scientific import build_screen_pool
    from flora_translate.design_realizer import _tubing_or_integrated_path_feasible
    from flora_translate.inventory_allocator import InventoryAllocator
    p, batch, plan, inv = photo_case()
    pool, _ = build_screen_pool(p, batch, plan, inv)
    candidate = FlowProposal.model_validate(pool[0]["proposal"])
    candidate.tubing_material = "PFA (Stage 1); FEP (Stage 2)"
    assert _tubing_or_integrated_path_feasible(candidate, inv)
    allocator = InventoryAllocator(inv, candidate)
    allocator._allocate_tubing()
    assert not allocator.unresolved
    assert len(allocator.assignments) == len(candidate.stage_parameters)
    candidate.stage_parameters[0]["material"] = "undeclared material"
    assert not _tubing_or_integrated_path_feasible(candidate, inv)
    allocator = InventoryAllocator(inv, candidate)
    allocator._allocate_tubing()
    assert any(row["operation_id"] == "st1_reactor" for row in allocator.unresolved)


def test_user_can_request_first_stage_gas_without_rewriting_batch_evidence():
    from flora_translate.chemistry_contract import reconcile_chemistry_plan
    from flora_translate.intake_agent import apply_intake_requirements_to_chemistry_plan
    from flora_translate.schemas import DesignInputPackage
    from flora_translate.engine.council_v4.scientific import build_screen_pool
    p, b, plan, inv = photo_case()
    gas = dict(plan.scientific_context["gas_delivery"], introduction_stage=1, introduction_stage_source="chemist_answer")
    updated, report = reconcile_chemistry_plan(b, plan, scientific=True,
        hard_constraints={"gas_introduction_requirement": gas})
    updated, _ = apply_intake_requirements_to_chemistry_plan(updated, DesignInputPackage(engineering_requirements={"gas": gas}))
    assert not updated.scientific_context["issues"]
    assert updated.scientific_context["gas_stage_deviation"]["chemical_compatibility_established"] is False
    air = next(c for c in updated.scientific_context["components"] if c["name"] == "air")
    assert air["addition_stages"] == [2]
    assert all(f.introduction_stage == 1 for f in updated.stream_logic if f.phase == "gas")
    assert not inv.reactor_trains
    pool, _ = build_screen_pool(p, b, updated, inv)
    assert len(pool) == 12
    assert all(all(s["Q_gas_sccm"] > 0 for s in row["proposal"]["stage_parameters"]) for row in pool)
    from flora_translate.main import _build_translate_topology
    from flora_translate.topology_compiler import compile_inventory_topology
    from flora_translate.final_design_contract import _build_executable_process_graph
    for row in pool:
        candidate = FlowProposal.model_validate(row["proposal"])
        topology = _build_translate_topology(candidate, updated, b, inv)
        compiled, _ = compile_inventory_topology(topology, proposal=candidate, inventory=inv)
        graph = _build_executable_process_graph(compiled.model_dump())
        for stage, final_stage in zip(graph.stages, candidate.stage_parameters):
            assert stage.residence_flow_mL_min == pytest.approx(final_stage["Q_liquid_mL_min"] + final_stage["Q_gas_sccm"])
    # Availability of connectors does not create extra reactors or a new volume.
    assert not inv.reactor_trains
    inv.standard_reactor_connectors_available = False
    with pytest.raises(ValueError, match="pre-council topology allocation incomplete"):
        build_screen_pool(p, b, updated, inv)
