from __future__ import annotations

import json
from pathlib import Path

import pytest

from flora_translate.design_disposition import (
    apply_design_disposition_gate,
    evaluate_design_disposition,
)
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    FlowProposal,
    GasHardwareSpec,
    LabInventory,
    MixerSpec,
    ProcessStage,
    ReactorSpec,
    StreamAssignment,
)


SCENARIOS_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "design_disposition_scenarios.json"
)


def _suite() -> dict:
    return json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))


def _inventory(inventory_id: str) -> LabInventory:
    return LabInventory.model_validate(_suite()["inventories"][inventory_id])


def _case(case_id: str) -> dict:
    return next(case for case in _suite()["cases"] if case["case_id"] == case_id)


def _validation_ready() -> dict:
    return {
        "status": "ready",
        "checks": {
            "reactor_inventory_match": True,
            "pump_flow_feasible": True,
            "tubing_feasible": True,
            "geometry_closure": True,
            "calculation_matches_serialized_design": True,
            "gas_bookkeeping_complete": True,
        },
        "unresolved_reasons": [],
    }


def _proposal(**updates) -> FlowProposal:
    data = {
        "residence_time_min": 10.0,
        "flow_rate_mL_min": 0.5,
        "temperature_C": 50.0,
        "concentration_M": 0.1,
        "BPR_bar": 5.0,
        "reactor_type": "coil",
        "tubing_material": "PFA",
        "tubing_ID_mm": 1.0,
        "reactor_volume_mL": 5.0,
        "engine_validated": True,
        "confidence": "MEDIUM",
    }
    data.update(updates)
    return FlowProposal(**data)


def test_unconfirmed_khu_hardware_blocks_gas_multistage_candidate():
    inventory = LabInventory(
        BPR_available=[],
        gas_hardware=[
            GasHardwareSpec(
                name="KHU oxygen mass-flow controller",
                type="MFC",
                gas="O2",
                min_flow_sccm=0.01,
                max_flow_sccm=10.0,
                max_pressure_bar=9.0,
            )
        ],
        reactors=[
            ReactorSpec(
                name="KHU PFA coil 10 mL",
                type="coil",
                material="PFA",
                volume_mL=10.0,
                ID_mm=1.016,
            )
        ],
    )
    proposal = _proposal(
        BPR_bar=3.0,
        tubing_material="PFA",
        tubing_ID_mm=1.016,
        reactor_volume_mL=10.0,
        streams=[
            StreamAssignment(
                stream_label="C",
                pump_role="O2 gas injection",
                contents=["pure O2"],
                phase="gas",
                gas_flow_sccm=1.0,
                gas_flow_actual_mL_min=0.3,
            )
        ],
    )
    plan = ChemistryPlan(
        reaction_class="two-stage photoredox oxidation",
        n_stages=2,
        stages=[
            ProcessStage(stage_number=1, stage_name="photoredox addition"),
            ProcessStage(stage_number=2, stage_name="aerobic oxidation"),
        ],
    )

    decision = evaluate_design_disposition(
        proposal,
        final_validation=_validation_ready(),
        inventory=inventory,
        batch_record=BatchRecord(reaction_description="Two-stage O2 process"),
        chemistry_plan=plan,
        hard_constraints={"BPR_settings_not_specified_in_source": True},
    )

    failure_ids = {finding.finding_id for finding in decision.hard_failures}
    assert decision.recommended_disposition == "BLOCK"
    assert "INVENTORY-BPR-UNCONFIRMED" in failure_ids
    assert "INVENTORY-MULTISTAGE-TOPOLOGY" in failure_ids


def test_undeclared_passive_mixer_does_not_hard_block_gas_design():
    inventory = LabInventory(
        BPR_available=[5.0],
        gas_hardware=[
            GasHardwareSpec(
                name="O2 MFC",
                type="MFC",
                gas="O2",
                min_flow_sccm=0.01,
                max_flow_sccm=10.0,
                max_pressure_bar=9.0,
            )
        ],
    )
    proposal = _proposal(
        streams=[
            StreamAssignment(
                stream_label="G",
                contents=["O2"],
                phase="gas",
                gas_flow_sccm=1.0,
            )
        ]
    )

    decision = evaluate_design_disposition(
        proposal,
        final_validation=_validation_ready(),
        inventory=inventory,
        batch_record=BatchRecord(),
        chemistry_plan=ChemistryPlan(o2_is_reagent=True),
    )

    assert "INVENTORY-GAS-LIQUID-MIXER" not in {
        finding.finding_id for finding in decision.hard_failures
    }


def test_explicitly_unavailable_mixer_hard_blocks_gas_design():
    inventory = LabInventory(
        BPR_available=[5.0],
        gas_hardware=[
            GasHardwareSpec(
                name="O2 MFC",
                type="MFC",
                gas="O2",
                min_flow_sccm=0.01,
                max_flow_sccm=10.0,
                max_pressure_bar=9.0,
            )
        ],
        mixers=[
            MixerSpec(
                equipment_id="mixer_unavailable",
                name="T-mixer",
                service_status="unavailable",
            )
        ],
    )
    proposal = _proposal(
        streams=[
            StreamAssignment(
                stream_label="G",
                contents=["O2"],
                phase="gas",
                gas_flow_sccm=1.0,
            )
        ]
    )

    decision = evaluate_design_disposition(
        proposal,
        final_validation=_validation_ready(),
        inventory=inventory,
        batch_record=BatchRecord(),
        chemistry_plan=ChemistryPlan(o2_is_reagent=True),
    )

    assert "INVENTORY-GAS-LIQUID-MIXER" in {
        finding.finding_id for finding in decision.hard_failures
    }


def _decision(
    case_id: str,
    inventory_id: str,
    proposal: FlowProposal,
    *,
    batch: BatchRecord | None = None,
    plan: ChemistryPlan | None = None,
):
    case = _case(case_id)
    return evaluate_design_disposition(
        proposal,
        final_validation=_validation_ready(),
        inventory=_inventory(inventory_id),
        batch_record=batch
        or BatchRecord(
            reaction_description=case["protocol"],
            raw_text=case["protocol"],
        ),
        chemistry_plan=plan or ChemistryPlan(),
        objective=case["objective"],
        hard_constraints=case["hard_constraints"],
    )


@pytest.mark.parametrize(
    ("case_id", "inventory_id", "proposal", "batch", "plan", "expected"),
    [
        (
            "suzuki_feasible",
            "thermal_feasible",
            _proposal(temperature_C=50),
            None,
            ChemistryPlan(reaction_class="Suzuki-Miyaura"),
            "SCREEN",
        ),
        (
            "suzuki_infeasible",
            "thermal_infeasible_40c",
            _proposal(temperature_C=40),
            None,
            ChemistryPlan(reaction_class="Suzuki-Miyaura"),
            "BLOCK",
        ),
        (
            "photo_oxidation_feasible",
            "photo_420_feasible",
            _proposal(
                temperature_C=21,
                tubing_material="FEP",
                wavelength_nm=420,
                streams=[
                    StreamAssignment(
                        stream_label="G",
                        pump_role="air feed",
                        contents=["air"],
                        phase="gas",
                        gas_flow_sccm=1.0,
                        gas_flow_actual_mL_min=0.2,
                    )
                ],
            ),
            None,
            ChemistryPlan(reaction_class="aerobic oxidation", o2_is_reagent=True),
            "SCREEN",
        ),
        (
            "photo_oxidation_infeasible",
            "photo_525_infeasible",
            _proposal(
                temperature_C=21,
                tubing_material="FEP",
                wavelength_nm=525,
                streams=[
                    StreamAssignment(
                        stream_label="G",
                        pump_role="air feed",
                        contents=["air"],
                        phase="gas",
                        gas_flow_sccm=1.0,
                        gas_flow_actual_mL_min=0.2,
                    )
                ],
            ),
            None,
            ChemistryPlan(reaction_class="aerobic oxidation", o2_is_reagent=True),
            "BLOCK",
        ),
        (
            "hydrogenolysis_feasible",
            "hydrogen_feasible",
            _proposal(
                temperature_C=60,
                BPR_bar=10,
                tubing_material="SS",
                reactor_type="packed_bed",
                streams=[
                    StreamAssignment(
                        stream_label="G",
                        pump_role="hydrogen feed",
                        contents=["H2"],
                        phase="gas",
                        gas_flow_sccm=1.0,
                        gas_flow_actual_mL_min=0.1,
                    )
                ],
            ),
            BatchRecord(
                reaction_description="Hydrogenolysis over Pd catalyst",
                raw_text="Hydrogen is required.",
            ),
            ChemistryPlan(reaction_class="hydrogenolysis"),
            "SCREEN",
        ),
        (
            "hydrogenolysis_infeasible",
            "hydrogen_unavailable",
            _proposal(
                temperature_C=60,
                BPR_bar=10,
                tubing_material="SS",
                reactor_type="packed_bed",
                streams=[
                    StreamAssignment(
                        stream_label="G",
                        pump_role="hydrogen feed",
                        contents=["H2"],
                        phase="gas",
                        gas_flow_sccm=1.0,
                        gas_flow_actual_mL_min=0.1,
                    )
                ],
            ),
            BatchRecord(
                reaction_description="Hydrogenolysis over Pd catalyst",
                raw_text="Hydrogen is required.",
            ),
            ChemistryPlan(reaction_class="hydrogenolysis"),
            "BLOCK",
        ),
        (
            "dinitration_feasible",
            "nitration_feasible",
            _proposal(
                temperature_C=60,
                tubing_material="PTFE",
                tubing_ID_mm=0.5,
            ),
            BatchRecord(
                reaction_description="Dinitration with concentrated nitric acid",
                raw_text="Dinitration with nitric acid.",
            ),
            ChemistryPlan(reaction_class="dinitration"),
            "SCREEN",
        ),
        (
            "dinitration_infeasible",
            "nitration_prohibited",
            _proposal(temperature_C=60, tubing_material="FEP"),
            BatchRecord(
                reaction_description="Dinitration with concentrated nitric acid",
                raw_text="Dinitration with nitric acid.",
            ),
            ChemistryPlan(reaction_class="dinitration"),
            "BLOCK",
        ),
        (
            "multistep_feasible",
            "multistep_feasible",
            _proposal(
                temperature_C=80,
                tubing_material="PFA",
                reactor_type="two_stage_coil_train",
                reactor_volume_mL=15,
            ),
            None,
            ChemistryPlan(
                reaction_class="two-step oxidation and amidation",
                n_stages=2,
                stages=[ProcessStage(stage_number=1), ProcessStage(stage_number=2)],
            ),
            "SCREEN",
        ),
        (
            "multistep_infeasible",
            "multistep_infeasible",
            _proposal(
                temperature_C=80,
                tubing_material="PFA",
                reactor_type="single_coil",
            ),
            None,
            ChemistryPlan(
                reaction_class="two-step oxidation and amidation",
                n_stages=2,
                stages=[ProcessStage(stage_number=1), ProcessStage(stage_number=2)],
            ),
            "BLOCK",
        ),
    ],
)
def test_paired_inventory_disposition(
    case_id,
    inventory_id,
    proposal,
    batch,
    plan,
    expected,
):
    decision = _decision(
        case_id,
        inventory_id,
        proposal,
        batch=batch,
        plan=plan,
    )
    assert decision.recommended_disposition == expected
    assert bool(decision.hard_failures) is (expected == "BLOCK")


def test_final_validation_failure_is_blocking():
    validation = _validation_ready()
    validation["checks"]["pump_flow_feasible"] = False
    proposal = _proposal()

    decision = evaluate_design_disposition(
        proposal,
        final_validation=validation,
        inventory=None,
        batch_record=None,
        chemistry_plan=None,
    )

    assert decision.recommended_disposition == "BLOCK"
    assert decision.hard_failures[0].finding_id == "FINAL-PUMP_FLOW_FEASIBLE"


def test_screen_required_alone_does_not_become_block():
    proposal = _proposal(
        engine_validated=False,
        safety_flags=["SCREEN_REQUIRED: kinetic anchor uncertain"],
    )

    decision = evaluate_design_disposition(
        proposal,
        final_validation=_validation_ready(),
        inventory=None,
        batch_record=None,
        chemistry_plan=None,
    )

    assert decision.recommended_disposition == "SCREEN"
    assert not decision.hard_failures


def test_apply_gate_sets_one_authoritative_top_level_disposition():
    proposal = _proposal(temperature_C=40)
    validation = _validation_ready()
    result = {
        "proposal": proposal.model_dump(),
        "final_validation": json.loads(json.dumps(validation)),
    }
    case = _case("suzuki_infeasible")

    decision = apply_design_disposition_gate(
        result,
        proposal=proposal,
        final_validation=validation,
        inventory=_inventory("thermal_infeasible_40c"),
        batch_record=BatchRecord(reaction_description=case["protocol"]),
        chemistry_plan=ChemistryPlan(reaction_class="Suzuki-Miyaura"),
        objective=case["objective"],
        hard_constraints=case["hard_constraints"],
    )

    assert decision.recommended_disposition == "BLOCK"
    assert result["recommended_disposition"] == "BLOCK"
    assert result["reported_disposition"] == "BLOCK"
    assert result["proposal"]["recommended_disposition"] == "BLOCK"
    assert result["final_validation"]["status"] == "blocked"
    assert result["final_validation"]["design_disposition"] == result["design_disposition"]
    assert result["proposal"]["engine_validated"] is False
