"""Review decisions and fixed hardware must survive delivery unchanged."""
import pytest
from flora_translate.engine.council_v4.chief import _select_winner_id, _apply_winner, _parse_chief
from flora_translate.engine.council_v4.inventory_candidates import generate_inventory_candidates, materialize
from flora_translate.design_realizer import realize_executable_design
from flora_translate.residence_time_basis import requested_residence_time_basis, gas_time_basis_consistent
from flora_translate.schemas import BatchRecord, FlowProposal, LabInventory, StreamAssignment


def candidates():
    inventory = LabInventory.model_validate({
        "reactors": [{"equipment_id": "bed", "name": "fixed packed reactor", "type": "packed-bed", "material": "stainless steel", "volume_mL": 2., "ID_mm": 4.5, "allowed_temperatures_C": [25.], "max_pressure_bar": 40}],
        "pumps": [{"equipment_id": "pump", "name": "liquid pump", "type": "HPLC", "min_flow_rate_mL_min": .1, "max_flow_rate_mL_min": 1., "max_pressure_bar": 40}],
        "BPR_available": [3.],
    })
    proposal = FlowProposal(reactor_volume_mL=20., tubing_ID_mm=.75, flow_rate_mL_min=.2, BPR_bar=3,
        streams=[StreamAssignment(stream_label="A", contents=["substrate (0.1 M)"], solvent="methanol", concentration_M=.1)])
    return generate_inventory_candidates(proposal, BatchRecord(solvent="methanol"), None, inventory, [],
        count=2, tau_kinetics_min=30, assumed_MW=200, IF_used=1, pump_max_bar=40, is_photochem=False, batch_time_min=60)


def test_inventory_pool_uses_available_bed_and_feed_closed_times():
    rows = candidates()["survivors"]
    assert len(rows) == 2
    assert len({r["Q_mL_min"] for r in rows}) == 2
    for row in rows:
        assert row["V_R_mL"] == 2.
        assert row["d_mm"] == 4.5
        assert row["equipment_id"] == "bed"
        assert row["tau_min"] * row["Q_mL_min"] == pytest.approx(2.)
        assert "packed-bed hydraulics unresolved" in row["hydraulic_model"]


def test_revision_cannot_reintroduce_unavailable_diameter_or_volume():
    row = candidates()["survivors"][0]
    for patch in [{"d_mm": .75}, {"V_R_mL": 20.}, {"temperature_C": 100.}]:
        with pytest.raises(ValueError):
            materialize(row["_realization_context"], patch=patch)
    revised = materialize(row["_realization_context"], patch={"tau_min": 20.})
    assert revised["Q_mL_min"] == pytest.approx(.1)
    assert revised["V_R_mL"] == 2.


def test_apply_winner_preserves_reviewed_feeds_despite_conflicting_chief_fields():
    row = candidates()["survivors"][0]
    proposal, _ = _apply_winner(FlowProposal(), row, {"BPR_bar": "20"},
        chief_data={"pump_flowrates": [{"stream_label": "A", "Q_mL_min": .9}]})
    assert proposal.flow_rate_mL_min == row["Q_mL_min"]
    assert proposal.BPR_bar == row["BPR_bar"]
    assert proposal.reactor_volume_mL == 2.


def test_explicit_rejection_survives_good_weighted_scores():
    scores = [{"candidate_id": 1, "combined": .9, "safety": .9, "disqualified": False}]
    assert _select_winner_id({"selected_candidate_id": None, "selection_rationale": "No candidate matches inventory"}, scores, {1}) is None
    assert _select_winner_id({}, scores, {1}) == 1
    assert _parse_chief('{"selected_candidate_id": null, "winner_id": 1}')["selected_candidate_id"] is None
    assert _parse_chief('{"selected_candidate_id": null, "rationale": "truncated')["selected_candidate_id"] is None


def test_numerical_realization_cannot_clear_a_council_block():
    row = candidates()["survivors"][0]
    context = row["_realization_context"]
    proposal = FlowProposal.model_validate(context["proposal"])
    proposal.inventory_constraints["council_selection_block"] = "Unresolved inventory/safety conflict"
    from flora_translate.final_design_validator import finalize_design
    proposal, _, _ = finalize_design(proposal, batch_record=BatchRecord.model_validate(context["batch"]),
        chemistry_plan=None, inventory=LabInventory.model_validate(context["inventory"]), analogies=[])
    assert proposal.inventory_constraints["council_selection_block"] == "Unresolved inventory/safety conflict"
    _, audit, validation = realize_executable_design(proposal, batch_record=BatchRecord.model_validate(context["batch"]),
        chemistry_plan=None, inventory=LabInventory.model_validate(context["inventory"]))
    assert audit["status"] == "incomplete"
    assert not validation["checks"]["design_realization_complete"]


def test_metering_instruction_does_not_set_reactor_time_basis():
    assert requested_residence_time_basis("Report air flow at inlet/STP") == "unknown"
    assert requested_residence_time_basis("Use liquid flow and reactor volume to define residence time") == "liquid_only"
    assert requested_residence_time_basis("Distinguish nominal empty-bed space time from phase contact time") == "liquid_only"


def test_declared_liquid_time_is_valid_for_a_gas_process_but_bad_closure_is_not():
    data = {"flow_rate_mL_min": .1, "reactor_volume_mL": 1., "residence_time_min": 10.,
            "residence_time_basis": "liquid-only", "streams": [{"phase": "gas", "gas_flow_sccm": 1.}]}
    assert gas_time_basis_consistent(data)
    data["residence_time_min"] = .909
    assert not gas_time_basis_consistent(data)


@pytest.mark.parametrize("role", ["Supported metal source", "Supported reaction material"])
def test_inventory_candidate_keeps_declared_bed_material_out_of_feed(role):
    context = candidates()["survivors"][0]["_realization_context"]
    context["plan"] = {"reagents": [{"name": "bed material", "role": role,
        "equiv_or_loading": "Flow: retained in the supplied packed bed; mass unresolved"}]}
    context["proposal"]["streams"][0]["contents"].append("bed material")
    row = materialize(context)
    assert "bed material" not in row["streams"][0]["contents"]
    assert any(r["name"] == "bed material" for r in row["stationary_components"])
