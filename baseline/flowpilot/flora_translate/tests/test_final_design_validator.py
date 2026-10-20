from pathlib import Path

import pytest

from flora_translate.design_calculator import DesignCalculator
from flora_translate.final_design_validator import finalize_design
from flora_translate.residence_time_basis import actual_gas_flow_from_stp
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    FlowProposal,
    GasHardwareSpec,
    LabInventory,
    PumpSpec,
    ReactorSpec,
    StreamAssignment,
    TubingSpec,
)


INVENTORY_PATH = Path("flora_translate/data/lab_inventory.json")


def test_pmpsch2tms_liquid_feed_is_not_misclassified_as_hydrogen():
    stream = StreamAssignment(
        stream_label="A",
        pump_role="Single liquid feed pump",
        contents=[
            "PMPSCH2TMS",
            "Acrylonitrile",
            "Ir(dF(CF3)ppy)2(dtbpy)PF6",
        ],
        solvent="EtOH / pH 9 aqueous buffer, pre-degassed under argon",
        phase="gas",
    )

    assert not DesignCalculator._stream_assignment_is_gas(stream)


def test_gas_calculations_use_flow_temperature_and_explicit_proposal_species():
    batch = BatchRecord(
        reaction_description="Photoredox reaction followed by oxidation in air.",
        solvent="EtOH",
        temperature_C=25,
        reaction_time_h=2,
        concentration_M=0.05,
    )
    plan = ChemistryPlan(
        reaction_class="aerobic oxidation",
        o2_is_reagent=True,
    )
    proposal = FlowProposal(
        residence_time_min=20,
        flow_rate_mL_min=0.2,
        reactor_volume_mL=10,
        concentration_M=0.05,
        temperature_C=40,
        BPR_bar=3,
        tubing_ID_mm=1.0,
        tubing_material="PFA",
        streams=[
            StreamAssignment(
                stream_label="C",
                pump_role="O2 gas injection",
                contents=["pure O2"],
                phase="gas",
                gas_flow_sccm=0.25,
                gas_flow_actual_mL_min=0.1,
                molar_equiv=1.0,
            )
        ],
    )

    calc = DesignCalculator().run(
        batch,
        chemistry_plan=plan,
        proposal=proposal,
        target_flow_rate_mL_min=proposal.flow_rate_mL_min,
        target_tubing_ID_mm=proposal.tubing_ID_mm,
        target_residence_time_min=proposal.residence_time_min,
    )

    assert calc.temperature_C == 40
    assert calc.gas_species == "O2"
    assert calc.gas_oxygen_fraction == 1.0
    assert calc.gas_flow_actual_mL_min == pytest.approx(
        actual_gas_flow_from_stp(calc.gas_flow_sccm, 40, 3),
        abs=1e-4,
    )


def test_finalizer_atomically_enforces_inventory_and_geometry_closure():
    inventory = LabInventory.from_json(str(INVENTORY_PATH))
    batch = BatchRecord(
        reaction_description="Knoevenagel condensation in ethanol for 1 h.",
        solvent="EtOH",
        temperature_C=40,
        reaction_time_h=1,
        concentration_M=0.2,
    )
    plan = ChemistryPlan(
        reaction_class="Knoevenagel condensation",
        mechanism_type="thermal",
    )
    council_proposal = FlowProposal(
        residence_time_min=12,
        flow_rate_mL_min=0.31,
        reactor_volume_mL=3.72,
        concentration_M=0.2,
        temperature_C=40,
        BPR_bar=2,
        tubing_ID_mm=0.75,
        tubing_material="PTFE",
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="substrate solution",
                solvent="EtOH",
                concentration_M=0.2,
                flow_rate_mL_min=0.31,
            )
        ],
    )

    final, calculations, report = finalize_design(
        council_proposal,
        batch_record=batch,
        chemistry_plan=plan,
        analogies=[],
        inventory=inventory,
    )

    available_reactors = {
        (reactor.volume_mL, reactor.ID_mm, reactor.material)
        for reactor in inventory.reactors
    }
    assert (
        final.reactor_volume_mL,
        final.tubing_ID_mm,
        final.tubing_material,
    ) in available_reactors
    assert report["status"] == "ready"
    assert report["checks"]["reactor_inventory_match"]
    assert report["checks"]["pump_flow_feasible"]
    assert report["checks"]["tubing_feasible"]
    assert report["checks"]["geometry_closure"]
    assert calculations.tubing_ID_mm == final.tubing_ID_mm
    assert abs(calculations.reactor_volume_mL - final.reactor_volume_mL) < 0.02


def test_finalizer_adds_missing_air_stream_and_both_residence_times():
    inventory = LabInventory.from_json(str(INVENTORY_PATH))
    batch = BatchRecord(
        reaction_description="The substrate solution was exposed to air for 2 h.",
        solvent="water",
        temperature_C=25,
        reaction_time_h=2,
        concentration_M=0.1,
    )
    plan = ChemistryPlan(
        reaction_class="aerobic oxidation",
        mechanism_type="oxidation",
    )
    council_proposal = FlowProposal(
        residence_time_min=20,
        residence_time_basis="inlet/STP apparent residence time",
        flow_rate_mL_min=0.1,
        reactor_volume_mL=2,
        concentration_M=0.1,
        temperature_C=25,
        BPR_bar=0,
        tubing_ID_mm=0.75,
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="substrate solution",
                solvent="water",
                concentration_M=0.1,
                flow_rate_mL_min=0.1,
            )
        ],
    )

    final, _, report = finalize_design(
        council_proposal,
        batch_record=batch,
        chemistry_plan=plan,
        analogies=[],
        inventory=inventory,
    )

    gas = next(stream for stream in final.streams if stream.phase == "gas")
    assert gas.contents == ["air"]
    assert gas.gas_flow_sccm > 0
    assert gas.gas_flow_actual_mL_min > 0
    assert gas.molar_equiv > 0
    assert final.residence_time_inlet_min > 0
    assert final.residence_time_in_channel_min > 0
    assert report["checks"]["gas_bookkeeping_complete"]


def test_finalizer_uses_flow_concentration_for_hydrogen_inventory_closure():
    inventory = LabInventory(
        pumps=[
            PumpSpec(
                name="HPLC pump",
                type="HPLC",
                min_flow_rate_mL_min=0.01,
                max_flow_rate_mL_min=5,
                max_pressure_bar=50,
            )
        ],
        tubing=[
            TubingSpec(
                material="SS",
                ID_mm=1.0,
                max_pressure_bar=50,
                max_temperature_C=120,
                transparent=False,
            )
        ],
        BPR_available=[10],
        gas_hardware=[
            GasHardwareSpec(
                name="Hydrogen MFC",
                type="MFC",
                gas="H2",
                min_flow_sccm=0.1,
                max_flow_sccm=100,
                max_pressure_bar=20,
            )
        ],
        reactors=[
            ReactorSpec(
                type="packed_bed",
                material="SS",
                volume_mL=5,
                ID_mm=1.0,
                max_temperature_C=120,
                notes="Certified for hydrogen service.",
            )
        ],
    )
    batch = BatchRecord(
        reaction_description="Hydrogenolysis under H2.",
        solvent="MeOH",
        temperature_C=60,
        reaction_time_h=5,
        concentration_M=0.1,
        atmosphere="H2",
    )
    plan = ChemistryPlan(
        reaction_class="hydrogenolysis",
        mechanism_type="gas-liquid-solid hydrogenation",
    )
    council_proposal = FlowProposal(
        residence_time_min=22.9,
        residence_time_basis="in-channel pressure-corrected total residence time",
        flow_rate_mL_min=0.125,
        reactor_volume_mL=7.1,
        concentration_M=0.05,
        temperature_C=60,
        BPR_bar=10,
        tubing_ID_mm=1.0,
        tubing_material="SS",
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="HPLC liquid feed to gas-liquid mixer",
                contents=["substrate", "methanol solvent"],
                phase="gas",
                concentration_M=0.05,
            ),
            StreamAssignment(
                stream_label="G",
                pump_role="Hydrogen MFC",
                contents=["H2"],
                phase="gas",
                molar_equiv=5.95,
            ),
        ],
    )

    final, calculations, report = finalize_design(
        council_proposal,
        batch_record=batch,
        chemistry_plan=plan,
        analogies=[],
        inventory=inventory,
    )

    liquid = next(stream for stream in final.streams if stream.stream_label == "A")
    gas = next(stream for stream in final.streams if stream.stream_label == "G")
    assert liquid.phase == "liquid"
    assert liquid.gas_flow_sccm is None
    assert gas.phase == "gas"
    assert calculations.concentration_M == 0.05
    assert report["checks"]["geometry_closure"]
    assert report["checks"]["calculation_matches_serialized_design"]
    assert report["status"] == "ready"
