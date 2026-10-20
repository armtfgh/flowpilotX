"""Gas identity and time-basis invariants across calculation and graph delivery."""
from copy import deepcopy

import pytest

from flora_translate.design_calculator import DesignCalculator
from flora_translate.design_realizer import _close_single_stage_time, _solve_gas_stream_rates
from flora_translate.final_design_contract import (
    _build_executable_process_graph,
    _gas_composition_and_equivalent_issues,
)
from flora_translate.main import _build_singlestep_topology
from flora_translate.residence_time_basis import (
    STP_MOLAR_VOLUME_ML_PER_MMOL,
    gas_reagent_fraction,
)
from flora_translate.schemas import BatchRecord, FlowProposal, StreamAssignment


def proposal(gas="air", fraction=1.0, q=0.1):
    return FlowProposal(
        reactor_volume_mL=3.0, flow_rate_mL_min=q, concentration_M=0.1,
        temperature_C=25, BPR_bar=3, reactor_type="coil",
        streams=[
            StreamAssignment(stream_label="A", contents=["substrate"], solvent="solvent",
                             concentration_M=0.1, flow_rate_mL_min=q, phase="liquid"),
            StreamAssignment(stream_label="G", contents=[gas], phase="gas",
                             gas_reagent_mole_fraction=fraction, molar_equiv=2),
        ],
    )


@pytest.mark.parametrize("contents", ["air", "oxygen in air (21 mol% O2)", "Molecular oxygen in air"])
def test_calculator_does_not_promote_oxygen_in_air_to_pure_oxygen(contents):
    p = proposal(contents)
    assert DesignCalculator._detect_gas_species(None, None, p) == ("air", 0.21)


@pytest.mark.parametrize("gas,fraction", [("air", 0.21), ("O2", 1.0), ("H2", 1.0), ("H2", 0.05)])
@pytest.mark.parametrize("q", [0.000013, 0.1, 0.75])
def test_solver_to_diagram_to_executable_graph_preserves_basis(gas, fraction, q):
    p = proposal(gas, fraction, q)
    decisions, issues = [], []
    _solve_gas_stream_rates(p, {}, None, decisions, issues)
    _close_single_stage_time(p, decisions)
    assert not issues
    topo = _build_singlestep_topology(p, None, BatchRecord())
    for op in topo.unit_operations:
        op.inventory_item_id = "test_" + op.op_id
        op.assignment_status = "assigned"
    topo.compilation_status = "inventory_assigned"
    graph = _build_executable_process_graph(topo.model_dump())
    stage = graph.stages[0]
    assert "inlet/STP" in stage.residence_time_basis
    assert stage.residence_flow_mL_min == pytest.approx(q + p.streams[1].gas_flow_sccm)
    assert stage.residence_time_min == pytest.approx(3 / stage.residence_flow_mL_min, abs=1e-6)
    assert stage.residence_time_min == p.residence_time_min
    # JSON serialization must not lose units/bases or quantize small liquid flows.
    restored = type(graph).model_validate_json(graph.model_dump_json())
    assert restored == graph
    assert not _gas_composition_and_equivalent_issues(p.model_dump())


def test_realizer_reconciles_and_audits_stale_pure_fraction_for_air():
    p = proposal("air", 1.0)
    decisions, issues = [], []
    _solve_gas_stream_rates(p, {}, None, decisions, issues)
    gas = p.streams[1]
    assert gas.gas_reagent_mole_fraction == 0.21
    assert gas.gas_flow_sccm == pytest.approx(0.1 * 0.1 * 2 * STP_MOLAR_VOLUME_ML_PER_MMOL / 0.21, abs=1e-6)
    assert gas.molar_equiv == pytest.approx(2)
    assert p.multiphase_metrics["gas_oxygen_fraction"] == 0.21
    assert p.multiphase_metrics["o2_supply_mmol_min"] == pytest.approx(0.02)
    assert any(d["decision"] == "reconcile_gas_composition" for d in decisions)
    assert not issues


def test_final_validator_independently_rejects_air_as_pure_oxygen():
    p = proposal("air", 1.0).model_dump()
    p["streams"][1].update(gas_flow_sccm=0.1 * 0.1 * 2 * STP_MOLAR_VOLUME_ML_PER_MMOL)
    codes = {x["code"] for x in _gas_composition_and_equivalent_issues(p)}
    assert "FINAL-GAS-COMPOSITION-INCONSISTENT" in codes
    assert "FINAL-GAS-EQUIVALENT-CLOSURE" in codes


def test_missing_gas_stage_basis_is_not_silently_assumed_liquid_only():
    from flora_translate.tests.test_executable_artifact_contract import _hydrogen_result
    topology = deepcopy(_hydrogen_result()["process_topology"])
    reactor = next(op for op in topology["unit_operations"] if "reactor" in op["op_type"])
    del reactor["parameters"]["residence_time_basis"]
    # Even if tau accidentally closes under liquid-only, an ambiguous gas graph fails.
    reactor["parameters"]["residence_time_min"] = 20.0
    with pytest.raises(ValueError, match="explicit residence-time basis"):
        _build_executable_process_graph(topology)


@pytest.mark.parametrize("value", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_custom_fraction_bounds(value):
    with pytest.raises(ValueError, match="mole fraction"):
        gas_reagent_fraction("H2/N2 mixture", value)
