from flora_translate.main import (
    _build_translate_topology,
    _sync_stage_hardware_from_compiled_topology,
    _topology_matches_serialized_proposal,
)
from flora_translate.multistage_inventory import reconcile_multistage_inventory, _temperature_for_light
import pytest
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    FlowProposal,
    GasHardwareSpec,
    LabInventory,
    LightSourceSpec,
    PressureControllerSpec,
    ProcessStage,
    PumpSpec,
    ReactorSpec,
    StreamAssignment,
    StreamLogic,
    TubingSpec,
)
from flora_translate.topology_compiler import compile_inventory_topology


def _inventory():
    return LabInventory(
        schema_version="flowpilot_lab_inventory_v3.0",
        strict_assignment=True,
        pumps=[
            PumpSpec(
                equipment_id="pump",
                name="Liquid pump",
                type="syringe",
                min_flow_rate_mL_min=0.01,
                max_flow_rate_mL_min=1.0,
                max_pressure_bar=10,
            )
        ],
        tubing=[
            TubingSpec(
                equipment_id="pfa_tubing",
                name="PFA tubing",
                material="PFA",
                ID_mm=1.0,
                max_pressure_bar=10,
                max_temperature_C=80,
            ),
            TubingSpec(
                equipment_id="fep_tubing",
                name="FEP tubing",
                material="FEP",
                ID_mm=1.0,
                max_pressure_bar=10,
                max_temperature_C=50,
            ),
        ],
        reactors=[
            ReactorSpec(
                equipment_id="stage1_10",
                name="Stage 1 PFA 10 mL",
                type="coil",
                material="PFA",
                volume_mL=10,
                ID_mm=1.0,
                max_pressure_bar=10,
                max_temperature_C=80,
            ),
            ReactorSpec(
                equipment_id="stage2_20",
                name="Stage 2 FEP 20 mL",
                type="coil",
                material="FEP",
                volume_mL=20,
                ID_mm=1.0,
                max_pressure_bar=10,
                max_temperature_C=50,
            ),
        ],
        light_sources=[
            LightSourceSpec(
                equipment_id="uv_450",
                name="UV-150 450 nm",
                wavelength_nm=450,
                compatible_reactor="coil",
                min_temperature_C=10,
                max_temperature_C=80,
            ),
            LightSourceSpec(
                equipment_id="manual_448",
                name="Manual 448 nm",
                wavelength_nm=448,
                compatible_reactor="coil",
                allowed_temperatures_C=[40, 50],
            ),
        ],
        gas_hardware=[
            GasHardwareSpec(
                equipment_id="o2_mfc",
                name="O2 MFC",
                type="MFC",
                gas="O2",
                min_flow_sccm=0.01,
                max_flow_sccm=10,
                max_pressure_bar=10,
            )
        ],
        pressure_controllers=[
            PressureControllerSpec(
                equipment_id="bpr_7",
                name="7 bar BPR",
                setpoints_bar=[7],
            )
        ],
    )


@pytest.mark.parametrize("requested,expected", [(25, 40), (50, 50), (95, 80)])
def test_profile_allowed_temperature_range_is_hard_even_with_broader_equipment_range(requested, expected):
    light = LightSourceSpec(name="Vapourtec UV-150 450 nm", wavelength_nm=450, compatible_reactor="coil", min_temperature_C=10, max_temperature_C=100)
    limits = {"photoreactor_limits": {"Vapourtec UV-150": {"allowed_temperature_C": {"minimum": 40, "maximum": 80}}}}
    assert _temperature_for_light(requested, light, limits) == expected


def test_temperature_discrete_settings_intersect_profile_range():
    light = LightSourceSpec(name="Manual blue", wavelength_nm=448, compatible_reactor="coil", allowed_temperatures_C=[40, 50])
    limits = {"photoreactor_limits": {"Manual blue": {"allowed_temperature_C": {"minimum": 45, "maximum": 60}}}}
    assert _temperature_for_light(40, light, limits) == 50
    limits["photoreactor_limits"]["Manual blue"]["allowed_temperature_C"]["minimum"] = 55
    with pytest.raises(ValueError, match="No available temperature setting"):
        _temperature_for_light(40, light, limits)


def _plan():
    return ChemistryPlan(
        reaction_name="Two-stage photochemistry",
        n_stages=2,
        stages=[
            ProcessStage(
                stage_number=1,
                stage_name="Anaerobic photoreaction",
                temperature_C=25,
                requires_light=True,
                wavelength_nm=450,
                feed_streams=[
                    StreamLogic(stream_label="A", reagents=["substrate"])
                ],
            ),
            ProcessStage(
                stage_number=2,
                stage_name="Aerobic photooxidation",
                temperature_C=25,
                requires_light=True,
                wavelength_nm=450,
                feed_streams=[
                    StreamLogic(stream_label="G", reagents=["O2 gas"], phase="gas")
                ],
            ),
        ],
    )


def _proposal():
    return FlowProposal(
        residence_time_min=28.6,
        flow_rate_mL_min=0.1,
        temperature_C=25,
        concentration_M=0.1,
        BPR_bar=7,
        tubing_material="PFA",
        tubing_ID_mm=1.0,
        reactor_volume_mL=10,
        wavelength_nm=450,
        light_setup=(
            "Stage 1 uv_450 with stage1_10. "
            "Stage 2 manual_448 with stage2_20."
        ),
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="liquid feed",
                flow_rate_mL_min=0.1,
            ),
            StreamAssignment(
                stream_label="G",
                pump_role="O2 feed",
                contents=["O2"],
                phase="gas",
                gas_flow_sccm=0.2,
                gas_flow_actual_mL_min=0.05,
                flow_rate_mL_min=0.05,
            ),
        ],
    )


def test_multistage_reconciliation_uses_exact_inventory_and_closes_times():
    proposal, report = reconcile_multistage_inventory(
        _proposal(), _plan(), _inventory()
    )

    assert report["status"] == "complete"
    assert report["used_reactor_ids"] == ["stage1_10", "stage2_20"]
    assert report["used_light_ids"] == ["uv_450", "manual_448"]
    assert proposal.reactor_volume_mL == 30
    for stage in proposal.stage_parameters:
        assert stage["flow_rate_mL_min"] == stage["Q_liquid_mL_min"]
        assert stage["notes"].startswith("Deterministic stage closure:")


def test_multistage_reconciliation_removes_stale_model_run_instructions():
    initial = _proposal()
    initial.stage_parameters = [
        {
            "stage_number": 1,
            "reactor_equipment_id": "stage1_10",
            "flow_rate_mL_min": 99.0,
            "pump_stream_A_equipment_id": "STALE-PUMP",
            "notes": "Stale model calculation: 999 min.",
        },
        {
            "stage_number": 2,
            "reactor_equipment_id": "stage2_20",
            "bpr_equipment_id": "STALE-BPR",
            "notes": "Stale model calculation: 999 min.",
        },
    ]

    proposal, _ = reconcile_multistage_inventory(initial, _plan(), _inventory())

    first, second = proposal.stage_parameters
    assert first["flow_rate_mL_min"] == first["Q_liquid_mL_min"] == 0.1
    assert "pump_stream_A_equipment_id" not in first
    assert "bpr_equipment_id" not in second
    assert "999 min" not in first["notes"]
    assert "999 min" not in second["notes"]
    assert proposal.stage_parameters[0]["residence_time_inlet_min"] == 100
    assert proposal.stage_parameters[1]["residence_time_inlet_min"] == 66.6667
    assert proposal.stage_parameters[1]["residence_time_in_channel_min"] == 133.3333
    assert proposal.stage_parameters[1]["temperature_C"] == 40
    assert proposal.stage_parameters[1]["wavelength_nm"] == 448


def test_stage_prose_cannot_turn_liquid_feed_into_oxygen_stream():
    plan = _plan()
    plan.stages[0].feed_streams[0].reasoning = (
        "Keep this feed oxygen-free to avoid premature O2 ingress."
    )

    proposal, _ = reconcile_multistage_inventory(
        _proposal(), plan, _inventory()
    )

    stage_one = proposal.stage_parameters[0]
    stage_two = proposal.stage_parameters[1]
    assert plan.stages[0].feed_streams[0].phase == "liquid"
    assert stage_one["Q_gas_sccm"] == 0
    assert stage_one["Q_gas_actual_mL_min"] == 0
    assert stage_one["residence_time_inlet_min"] == 100
    assert stage_one["residence_time_in_channel_min"] == 100
    assert stage_two["Q_gas_sccm"] == 0.2


def test_khu_stagewise_residence_time_arithmetic_uses_correct_gas_basis():
    proposal = _proposal()
    proposal.flow_rate_mL_min = 0.43454
    proposal.streams[0].flow_rate_mL_min = 0.43454
    proposal.streams[1].flow_rate_mL_min = 0.3066
    proposal.streams[1].gas_flow_sccm = 0.9739
    proposal.streams[1].gas_flow_actual_mL_min = 0.3066
    plan = _plan()
    plan.stages[0].feed_streams[0].reasoning = "Exclude O2 from Stage 1."

    reconciled, _ = reconcile_multistage_inventory(
        proposal, plan, _inventory()
    )

    stage_one, stage_two = reconciled.stage_parameters
    assert stage_one["residence_time_inlet_min"] == 23.0128
    assert stage_one["residence_time_in_channel_min"] == 23.0128
    assert stage_two["residence_time_inlet_min"] == 14.2001
    assert stage_two["residence_time_in_channel_min"] == 26.9855
    assert reconciled.residence_time_inlet_min == 37.2129
    assert reconciled.residence_time_in_channel_min == 49.9983


def test_repeated_stream_label_is_introduced_once_then_carried_forward():
    plan = _plan()
    plan.stages[1].feed_streams.insert(
        0,
        StreamLogic(
            stream_label="A",
            reagents=["substrate solution carried from Stage 1"],
        ),
    )

    reconciled, _ = reconcile_multistage_inventory(
        _proposal(), plan, _inventory()
    )

    assert reconciled.stage_parameters[0]["Q_liquid_mL_min"] == 0.1
    assert reconciled.stage_parameters[1]["Q_liquid_mL_min"] == 0.1


def test_multistage_reconciliation_uses_frozen_temperature_limits_for_old_inventory():
    inventory = _inventory()
    inventory.light_sources[1].allowed_temperatures_C = []

    proposal, _ = reconcile_multistage_inventory(
        _proposal(),
        _plan(),
        inventory,
        operating_limits={
            "photoreactor_limits": {
                "KHU manual blue photoreactor": {
                    "allowed_temperatures_C": [40.0, 50.0]
                }
            }
        },
    )

    assert proposal.stage_parameters[1]["temperature_C"] == 40


def test_multistage_topology_preserves_exact_stage_hardware():
    inventory = _inventory()
    plan = _plan()
    proposal, _ = reconcile_multistage_inventory(_proposal(), plan, inventory)
    topology = _build_translate_topology(proposal, plan, BatchRecord())
    compiled, allocation = compile_inventory_topology(
        topology,
        proposal=proposal,
        inventory=inventory,
    )

    reactors = [
        operation
        for operation in compiled.unit_operations
        if operation.op_type == "coil_reactor"
    ]
    assert [item.parameters["volume_mL"] for item in reactors] == [10, 20]
    assert [item.parameters["material"] for item in reactors] == ["PFA", "FEP"]
    assert [item.inventory_item_id for item in reactors] == ["stage1_10", "stage2_20"]
    assert [item.parameters["residence_time_min"] for item in reactors] == [100.0, 66.67]
    assert reactors[1].parameters["residence_time_inlet_min"] == 66.6667
    assert reactors[1].parameters["residence_time_in_channel_min"] == 133.3333
    assert reactors[1].parameters["Q_gas_sccm"] == 0.2
    assert _topology_matches_serialized_proposal(topology, proposal)
    unresolved = {
        (item["operation_id"], item["category"])
        for item in allocation["unresolved_requirements"]
    }
    assert unresolved == set()
    assert allocation["status"] == "complete_with_assumptions"
    assert allocation["assumed_standard_accessories"][0]["operation_id"] == "st2_mixer"


def test_compiled_topology_replaces_stale_model_written_stage_hardware_ids():
    inventory = _inventory()
    plan = _plan()
    proposal, _ = reconcile_multistage_inventory(_proposal(), plan, inventory)
    proposal.stage_parameters[0]["pump_equipment_id"] = "STALE-PUMP"
    proposal.stage_parameters[1]["gas_equipment_id"] = "STALE-MFC"
    proposal.stage_parameters[1]["BPR_equipment_id"] = "STALE-BPR"
    topology = _build_translate_topology(proposal, plan, BatchRecord())
    compiled, _ = compile_inventory_topology(
        topology,
        proposal=proposal,
        inventory=inventory,
    )

    assert _sync_stage_hardware_from_compiled_topology(proposal, compiled)
    first, second = proposal.stage_parameters
    assert first["pump_equipment_id"] == "pump"
    assert second["gas_equipment_id"] == "o2_mfc"
    assert second["BPR_equipment_id"] == "bpr_7"
    assert second["mixer_assignment_status"] == "assumed_standard_accessory"


def test_reconciliation_replaces_exhausted_duplicate_with_referenced_fallback():
    proposal = _proposal()
    proposal.stage_parameters = [
        {
            "stage_number": 1,
            "reactor_equipment_id": "stage1_10",
            "light_equipment_id": "uv_450",
        },
        {
            "stage_number": 2,
            "reactor_equipment_id": "stage1_10",
            "light_equipment_id": "uv_450",
            "residence_time_min": 999,
        },
    ]

    reconciled, report = reconcile_multistage_inventory(
        proposal, _plan(), _inventory()
    )

    assert report["status"] == "complete"
    assert report["used_reactor_ids"] == ["stage1_10", "stage2_20"]
    assert report["used_light_ids"] == ["uv_450", "manual_448"]
    assert reconciled.stage_parameters[1]["residence_time_inlet_min"] == 66.6667


def test_unresolved_stage_does_not_retain_stale_run_values():
    inventory = _inventory()
    inventory.reactors = inventory.reactors[:1]
    inventory.light_sources = inventory.light_sources[:1]
    proposal = _proposal()
    proposal.stage_parameters = [
        {
            "stage_number": 1,
            "reactor_equipment_id": "stage1_10",
            "light_equipment_id": "uv_450",
        },
        {
            "stage_number": 2,
            "reactor_equipment_id": "stage1_10",
            "reactor_volume_mL": 10,
            "residence_time_inlet_min": 100,
            "Q_liquid_mL_min": 0.1,
        },
    ]

    reconciled, report = reconcile_multistage_inventory(
        proposal, _plan(), inventory
    )

    assert report["status"] == "incomplete"
    stage_two = reconciled.stage_parameters[1]
    assert stage_two["inventory_resolved"] is False
    assert "reactor_equipment_id" not in stage_two
    assert "reactor_volume_mL" not in stage_two
    assert "residence_time_inlet_min" not in stage_two
    assert "Q_liquid_mL_min" not in stage_two
