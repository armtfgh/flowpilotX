from flora_translate.schemas import (
    ChemistryPlan,
    GasHardwareSpec,
    LabInventory,
    LightSourceSpec,
    MixerSpec,
    PressureControllerSpec,
    ProcessStage,
    PumpSpec,
    ReactorSpec,
    StreamLogic,
)
from flora_translate.topology_preflight import analyze_topology_requirements


def _plan():
    return ChemistryPlan(
        reaction_name="Two-stage photochemistry",
        n_stages=2,
        stages=[
            ProcessStage(
                stage_number=1,
                stage_name="oxygen-free photoreaction",
                requires_light=True,
                wavelength_nm=450,
                feed_streams=[StreamLogic(stream_label="A", reagents=["substrate"])],
            ),
            ProcessStage(
                stage_number=2,
                stage_name="aerobic oxidation",
                requires_light=True,
                wavelength_nm=450,
                feed_streams=[StreamLogic(stream_label="G", reagents=["O2"])],
            ),
        ],
    )


def _inventory(*, mixer: bool):
    return LabInventory(
        schema_version="flowpilot_lab_inventory_v3.0",
        strict_assignment=True,
        pumps=[
            PumpSpec(
                equipment_id="pump",
                name="Pump",
                quantity=1,
                type="syringe",
                min_flow_rate_mL_min=0.01,
                max_flow_rate_mL_min=1,
                max_pressure_bar=8,
            )
        ],
        gas_hardware=[
            GasHardwareSpec(
                equipment_id="o2_mfc",
                name="O2 MFC",
                type="MFC",
                gas="O2",
            )
        ],
        mixers=[
            MixerSpec(equipment_id="mixer", name="T-mixer")
        ] if mixer else [],
        reactors=[
            ReactorSpec(
                equipment_id="reactor_a",
                name="Reactor A",
                quantity=2,
                type="coil",
                material="PFA",
                volume_mL=10,
                ID_mm=1,
            )
        ],
        light_sources=[
            LightSourceSpec(
                equipment_id="light",
                name="450 nm LED",
                quantity=2,
                wavelength_nm=450,
                compatible_reactor="coil",
            )
        ],
        pressure_controllers=[
            PressureControllerSpec(equipment_id="bpr", name="BPR")
        ],
    )


def test_undeclared_passive_mixer_allows_design_with_verification_assumption():
    topology, report = analyze_topology_requirements(_plan(), _inventory(mixer=False))

    assert report["status"] == "ready_with_assumptions"
    assert not report["unresolved_requirements"]
    assert [
        (item["requirement_id"], item["status"])
        for item in report["assumed_standard_accessories"]
    ] == [
        ("REQ-MIXERS", "assumed_standard_accessory")
    ]
    mixer = next(op for op in topology.unit_operations if op.op_type == "mixer")
    assert mixer.assignment_status == "assumed_standard_accessory"
    assert not any(
        item["requirement_id"] == "REQ-DIRECT-REACTOR-CONNECTIONS"
        for item in report["requirements"]
    )


def test_direct_reactor_connections_use_stocked_accessories():
    plan = _plan()
    gas = plan.stages[1].feed_streams.pop()
    gas.phase = "gas"
    gas.introduction_stage = 1
    plan.stages[0].feed_streams.append(gas)
    inventory = _inventory(mixer=True)
    _, report = analyze_topology_requirements(plan, inventory)
    connection = next(row for row in report["requirements"] if row["requirement_id"] == "REQ-DIRECT-REACTOR-CONNECTIONS")
    assert connection["status"] == "assumed_standard_accessory"
    assert not report["unresolved_requirements"]
    inventory.capability_status["connectors"] = "unavailable"
    _, blocked = analyze_topology_requirements(plan, inventory)
    assert any(row["requirement_id"] == "REQ-DIRECT-REACTOR-CONNECTIONS" for row in blocked["unresolved_requirements"])


def test_explicitly_unavailable_mixer_blocks_before_design():
    inventory = _inventory(mixer=False)
    inventory.mixers = [
        MixerSpec(
            equipment_id="mixer_unavailable",
            name="T-mixer",
            service_status="unavailable",
        )
    ]

    _, report = analyze_topology_requirements(_plan(), inventory)

    assert report["status"] == "infeasible"
    assert [item["requirement_id"] for item in report["unresolved_requirements"]] == [
        "REQ-MIXERS"
    ]


def test_declared_mixer_allows_numerical_design_to_proceed():
    topology, report = analyze_topology_requirements(_plan(), _inventory(mixer=True))

    assert report["status"] == "ready"
    assert not report["unresolved_requirements"]
    assert topology.compilation_status == "preflight_ready"


def test_oxygen_exclusion_prose_does_not_add_stage_one_gas_feed():
    plan = _plan()
    plan.stages[0].feed_streams[0].reasoning = "Exclude O2 from this stage."

    topology, _ = analyze_topology_requirements(plan, _inventory(mixer=True))

    gas_operations = [op for op in topology.unit_operations if op.op_type == "mfc"]
    assert [operation.op_id for operation in gas_operations] == ["st2_gas_1"]
    assert plan.o2_is_reagent is True


def test_auxiliary_nitrogen_purge_does_not_require_second_mfc():
    plan = ChemistryPlan(
        reaction_name="Hydrogenolysis",
        stream_logic=[
            StreamLogic(stream_label="A", reagents=["substrate", "methanol"]),
            StreamLogic(stream_label="B", reagents=["H2"], phase="gas"),
            StreamLogic(
                stream_label="C",
                reagents=["nitrogen"],
                phase="gas",
                reasoning="Startup purge/leak check and shutdown displacement.",
            ),
        ],
    )
    inventory = _inventory(mixer=True)
    inventory.gas_hardware[0].gas = "H2"

    topology, report = analyze_topology_requirements(plan, inventory)

    gas_operations = [op for op in topology.unit_operations if op.op_type == "mfc"]
    assert len(gas_operations) == 1
    gas_requirement = next(
        item for item in report["requirements"] if item["requirement_id"] == "REQ-GAS-MFC"
    )
    assert gas_requirement["required_count"] == 1
    assert report["status"] == "ready"
