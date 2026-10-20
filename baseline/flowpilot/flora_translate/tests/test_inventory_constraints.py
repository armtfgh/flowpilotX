from pathlib import Path

from flora_translate.engine.design_space import DesignSpaceSearch
from flora_translate.inventory_constraints import enforce_reactor_inventory, inventory_prompt_block
from flora_translate.residence_time_basis import gas_equiv_from_stp_flow
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory, StreamAssignment


THQ_INVENTORY = Path("flora_translate/data/lab_inventory_thq.json")


class _StubCalc:
    is_gas_liquid = True
    residence_time_min = 67.68
    tau_analogy_min = None
    tau_class_min = None
    tau_kinetics_min = 67.68
    rate_constant = None
    intensification_factor = 13.3
    concentration_M = 0.5
    bpr_pressure_bar = 6.0
    extinction_coefficient_M_cm = None


def _thq_batch() -> BatchRecord:
    return BatchRecord(
        reaction_description="Photochemical aerobic oxidation of tetrahydroquinoline to quinoline with O2.",
        solvent="DMSO",
        temperature_C=40,
        reaction_time_h=15,
        concentration_M=0.5,
        scale_mmol=0.2,
        atmosphere="O2",
        wavelength_nm=450,
    )


def _thq_plan() -> ChemistryPlan:
    return ChemistryPlan(
        reaction_class="photochemical aerobic oxidation",
        mechanism_type="O2-mediated photochemical oxidation",
        o2_is_reagent=True,
        recommended_wavelength_nm=450,
    )


def test_design_space_uses_only_inventory_reactor_volumes_for_thq():
    inventory = LabInventory.from_json(str(THQ_INVENTORY))
    candidates = DesignSpaceSearch().run(
        batch_record=_thq_batch(),
        chemistry_plan=_thq_plan(),
        calculations=_StubCalc(),
        inventory=inventory,
        reaction_class="photochemical aerobic oxidation",
    )

    assert candidates
    volumes = {round(c.V_R_mL) for c in candidates}
    assert volumes <= {10, 15, 20, 30}
    assert {round(c.d_mm, 3) for c in candidates} == {1.0}
    assert all(c.inventory_reactor_name for c in candidates)


def test_inventory_enforcement_snaps_krict_like_design_to_available_reactor():
    inventory = LabInventory.from_json(str(THQ_INVENTORY))
    proposal = FlowProposal(
        residence_time_min=67.68,
        residence_time_in_channel_min=67.68,
        residence_time_inlet_min=15.277,
        residence_time_basis="in-channel total actual flow",
        flow_rate_mL_min=0.02835,
        reactor_volume_mL=10.74,
        temperature_C=40,
        concentration_M=0.5,
        BPR_bar=6,
        tubing_ID_mm=0.75,
        tubing_material="FEP",
        wavelength_nm=450,
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="substrate solution",
                contents=["6-methyl-THQ"],
                solvent="DMSO",
                concentration_M=0.5,
                flow_rate_mL_min=0.02835,
            ),
            StreamAssignment(
                stream_label="G",
                pump_role="O2 gas feed",
                contents=["O2"],
                phase="gas",
                flow_rate_mL_min=0.13034,
                gas_flow_actual_mL_min=0.13034,
                gas_flow_sccm=0.67469,
            ),
        ],
    )

    revised, report = enforce_reactor_inventory(proposal, inventory)

    assert report["applied"]
    assert revised.reactor_volume_mL == 10.0
    assert revised.tubing_ID_mm == 1.0
    assert revised.inventory_selection["system"] == "Vapourtec System"
    assert revised.residence_time_basis == "inlet/STP apparent residence time"
    assert revised.residence_time_min == revised.residence_time_inlet_min == 15.277
    assert revised.residence_time_in_channel_min > revised.residence_time_inlet_min
    assert revised.flow_rate_mL_min < proposal.flow_rate_mL_min
    assert revised.streams[1].gas_flow_sccm is not None
    assert revised.residence_time_inlet_min is not None


def test_inventory_enforcement_preserves_inlet_stp_residence_basis():
    inventory = LabInventory.from_json(str(THQ_INVENTORY))
    proposal = FlowProposal(
        residence_time_min=67.0,
        residence_time_inlet_min=67.0,
        residence_time_in_channel_min=137.0,
        residence_time_basis="inlet/STP apparent residence time",
        flow_rate_mL_min=0.014,
        reactor_volume_mL=10.0,
        temperature_C=40,
        concentration_M=0.5,
        BPR_bar=6,
        tubing_ID_mm=1.0,
        tubing_material="FEP",
        wavelength_nm=450,
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="substrate solution",
                contents=["6-methyl-THQ"],
                solvent="DMSO",
                concentration_M=0.5,
                flow_rate_mL_min=0.014,
            ),
            StreamAssignment(
                stream_label="G",
                pump_role="O2 gas feed",
                contents=["O2"],
                phase="gas",
                flow_rate_mL_min=0.059,
                gas_flow_actual_mL_min=0.059,
                gas_flow_sccm=0.153,
            ),
        ],
    )

    revised, report = enforce_reactor_inventory(proposal, inventory)

    assert report["applied"]
    assert revised.reactor_volume_mL == 10.0
    assert revised.residence_time_basis == "inlet/STP apparent residence time"
    assert revised.residence_time_min == 67.0
    assert revised.residence_time_inlet_min == 67.0
    assert revised.residence_time_in_channel_min > 120.0
    assert revised.streams[1].gas_flow_sccm > revised.streams[1].gas_flow_actual_mL_min


def test_inventory_enforcement_recomputes_o2_from_target_equiv():
    inventory = LabInventory.from_json(str(THQ_INVENTORY))
    proposal = FlowProposal(
        residence_time_min=68.6,
        residence_time_inlet_min=68.6,
        residence_time_in_channel_min=300.0,
        residence_time_basis="inlet/STP apparent residence time",
        flow_rate_mL_min=0.06,
        reactor_volume_mL=10.0,
        temperature_C=40,
        concentration_M=0.5,
        BPR_bar=6,
        tubing_ID_mm=1.0,
        tubing_material="FEP",
        wavelength_nm=450,
        evidence_calibration={
            "best_run_id": "entry_05_krict_2_1",
            "best_tau_inlet_min": 38.6,
            "best_response_pct": 27.0,
            "target_response_pct": 75.0,
            "recommended_conditions": {
                "residence_time_basis": "inlet/STP apparent residence time",
                "residence_time_min": 68.6,
                "residence_time_inlet_min": 68.6,
                "gas_equiv_inlet": 2.0,
            },
        },
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="substrate solution",
                contents=["6-methyl-THQ"],
                solvent="DMSO",
                concentration_M=0.5,
                flow_rate_mL_min=0.06,
            ),
            StreamAssignment(
                stream_label="G",
                pump_role="O2 gas feed",
                contents=["O2"],
                phase="gas",
                flow_rate_mL_min=0.04,
                gas_flow_actual_mL_min=0.04,
                gas_flow_sccm=0.04,
            ),
        ],
    )

    revised, report = enforce_reactor_inventory(proposal, inventory)
    gas = revised.streams[1]
    supplied_equiv = gas_equiv_from_stp_flow(
        gas.gas_flow_sccm,
        revised.flow_rate_mL_min,
        revised.concentration_M,
        1.0,
    )

    assert report["applied"]
    assert revised.residence_time_inlet_min == 68.6
    assert abs(supplied_equiv - 2.0) < 0.02
    assert gas.gas_flow_sccm > gas.gas_flow_actual_mL_min
    assert gas.molar_equiv == 2.0


def test_manual_reactor_enforces_ten_microliter_pump_minimum_and_preserves_three_bar():
    inventory = LabInventory.from_json(str(THQ_INVENTORY))
    inventory.reactors = [
        reactor
        for reactor in inventory.reactors
        if reactor.system == "Manual Setup" and reactor.volume_mL <= 20
    ]
    proposal = FlowProposal(
        residence_time_min=98.9,
        residence_time_inlet_min=98.9,
        residence_time_basis="inlet/STP apparent residence time",
        flow_rate_mL_min=0.0086,
        reactor_volume_mL=20.0,
        temperature_C=40,
        concentration_M=0.5,
        BPR_bar=3,
        tubing_ID_mm=1.0,
        tubing_material="FEP",
        wavelength_nm=448,
        evidence_calibration={
            "recommended_conditions": {
                "residence_time_inlet_min": 98.9,
                "gas_equiv_inlet": 2.0,
            },
        },
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="substrate solution",
                contents=["6-methyl-THQ"],
                solvent="DMSO",
                concentration_M=0.5,
                flow_rate_mL_min=0.0086,
            ),
            StreamAssignment(
                stream_label="G",
                pump_role="O2 gas feed",
                contents=["O2"],
                phase="gas",
                molar_equiv=2.0,
                gas_flow_actual_mL_min=0.06,
                gas_flow_sccm=0.19,
            ),
        ],
    )

    revised, report = enforce_reactor_inventory(proposal, inventory)
    flow_report = revised.inventory_constraints["flow_recalculation"]

    assert report["applied"]
    assert revised.inventory_selection["system"] == "Manual Setup"
    assert revised.flow_rate_mL_min == 0.01
    assert revised.BPR_bar == 3.0
    assert revised.residence_time_inlet_min < 98.9
    assert revised.residence_time_inlet_min > 68.9
    assert flow_report["pump_flow_clamped"] is True
    assert flow_report["pump_min_flow_rate_mL_min"] == 0.01
    assert flow_report["o2_equiv_supplied"] == 2.0


def test_serial_30_ml_manual_reactor_makes_evidence_screen_pump_feasible():
    inventory = LabInventory.from_json(str(THQ_INVENTORY))
    proposal = FlowProposal(
        residence_time_min=98.9,
        residence_time_inlet_min=98.9,
        residence_time_basis="inlet/STP apparent residence time",
        flow_rate_mL_min=0.00782,
        reactor_volume_mL=20.0,
        temperature_C=40,
        concentration_M=0.5,
        BPR_bar=3,
        tubing_ID_mm=1.0,
        tubing_material="FEP",
        wavelength_nm=448,
        evidence_calibration={
            "best_tau_min": 68.9,
            "anchor_conditions": {
                "reactor_volume_mL": 20.0,
                "gas_equiv_inlet": 2.0,
            },
            "recommended_conditions": {
                "residence_time_inlet_min": 98.9,
                "gas_equiv_inlet": 2.0,
            },
        },
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="substrate solution",
                contents=["6-methyl-THQ"],
                solvent="DMSO",
                concentration_M=0.5,
                flow_rate_mL_min=0.00782,
            ),
            StreamAssignment(
                stream_label="G",
                pump_role="O2 gas feed",
                contents=["O2"],
                phase="gas",
                molar_equiv=2.0,
                gas_flow_actual_mL_min=0.05,
                gas_flow_sccm=0.175,
            ),
        ],
    )

    revised, report = enforce_reactor_inventory(proposal, inventory)
    flow_report = revised.inventory_constraints["flow_recalculation"]

    assert report["applied"]
    assert revised.reactor_volume_mL == 30.0
    assert revised.inventory_selection["system"] == "Manual Setup"
    assert revised.inventory_selection["component_volumes_mL"] == [20.0, 10.0]
    assert revised.residence_time_inlet_min == 98.9
    assert revised.flow_rate_mL_min > 0.01
    assert revised.BPR_bar == 3.0
    assert flow_report["pump_flow_clamped"] is False
    assert flow_report["o2_equiv_supplied"] == 2.0


def test_inventory_prompt_exposes_pump_flow_limits():
    inventory = LabInventory.from_json(str(THQ_INVENTORY))

    prompt = inventory_prompt_block(inventory)

    assert "Manual syringe/HPLC pump setup: 0.01-5 mL/min" in prompt
    assert "systems: Manual Setup" in prompt
