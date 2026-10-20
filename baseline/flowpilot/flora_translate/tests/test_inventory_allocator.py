from flora_translate.inventory_allocator import InventoryAllocator, _gas_matches
from flora_translate.schemas import (
    ConnectorSpec,
    FlowProposal,
    GasHardwareSpec,
    LAB_INVENTORY_SCHEMA_VERSION,
    LabInventory,
    LightSourceSpec,
    MixerSpec,
    PressureControllerSpec,
    ProcessTopology,
    PumpSpec,
    ReactorSpec,
    ReactorTrainSpec,
    StreamConnection,
    TubingSpec,
    UnitOperation,
)


def _proposal(**overrides):
    values = {
        "flow_rate_mL_min": 0.05,
        "reactor_volume_mL": 10,
        "tubing_ID_mm": 1.0,
        "tubing_material": "PFA",
        "BPR_bar": 6,
        "temperature_C": 40,
        "wavelength_nm": 450,
        "inventory_selection": {"system": "Manual", "equipment_id": "reactor_10"},
    }
    values.update(overrides)
    return FlowProposal(**values)


def _strict_inventory(*, mixer=True, pump_quantity=1):
    return LabInventory(
        schema_version=LAB_INVENTORY_SCHEMA_VERSION,
        strict_assignment=True,
        pumps=[
            PumpSpec(
                equipment_id="pump_1",
                name="Syringe pump",
                quantity=pump_quantity,
                type="syringe",
                min_flow_rate_mL_min=0.01,
                max_flow_rate_mL_min=1,
                max_pressure_bar=8,
                compatible_systems=["Manual"],
            )
        ],
        tubing=[
            TubingSpec(
                equipment_id="tubing_1",
                name="PFA tubing",
                material="PFA",
                ID_mm=1.0,
                max_pressure_bar=8,
                max_temperature_C=80,
            )
        ],
        reactors=[
            ReactorSpec(
                equipment_id="reactor_10",
                name="10 mL PFA reactor",
                system="Manual",
                type="coil",
                material="PFA",
                volume_mL=10,
                ID_mm=1.0,
                max_pressure_bar=8,
                max_temperature_C=80,
            )
        ],
        mixers=[
            MixerSpec(
                equipment_id="mixer_1",
                name="PFA T-mixer",
                type="T-mixer",
                material="PFA",
                max_inputs=2,
                max_pressure_bar=8,
                compatible_systems=["Manual"],
            )
        ] if mixer else [],
        pressure_controllers=[
            PressureControllerSpec(
                equipment_id="bpr_1",
                name="6 bar BPR",
                setpoints_bar=[6],
                compatible_systems=["Manual"],
            )
        ],
        gas_hardware=[
            GasHardwareSpec(
                equipment_id="mfc_1",
                name="O2 MFC",
                type="MFC",
                gas="O2",
                min_flow_sccm=0.01,
                max_flow_sccm=10,
                max_pressure_bar=8,
            )
        ],
        light_sources=[
            LightSourceSpec(
                equipment_id="light_1",
                name="450 nm LED",
                wavelength_nm=450,
                compatible_reactor="coil",
                compatible_systems=["Manual"],
            )
        ],
    )


def _gas_liquid_topology():
    operations = [
        UnitOperation(op_id="pump_a", op_type="pump", label="Liquid feed", parameters={"flow_rate_mL_min": 0.05}),
        UnitOperation(op_id="mfc_o2", op_type="mfc", label="O2 feed", parameters={"contents": ["O2"], "gas_flow_sccm": 0.2}),
        UnitOperation(op_id="mixer_1", op_type="mixer", label="Mixer", parameters={}),
        UnitOperation(op_id="reactor_1", op_type="photoreactor", label="Reactor", parameters={"volume_mL": 10, "ID_mm": 1.0, "material": "PFA", "temperature_C": 40}),
        UnitOperation(op_id="led_1", op_type="led_module", label="LED", parameters={"wavelength_nm": 450}),
        UnitOperation(op_id="bpr_1", op_type="bpr", label="BPR", parameters={"pressure_bar": 6}),
        UnitOperation(op_id="collector_1", op_type="collector", label="Collection"),
    ]
    streams = [
        StreamConnection(stream_id="s1", from_op="pump_a", to_op="mixer_1"),
        StreamConnection(stream_id="s2", from_op="mfc_o2", to_op="mixer_1", stream_type="gas"),
        StreamConnection(stream_id="s3", from_op="mixer_1", to_op="reactor_1"),
        StreamConnection(stream_id="s4", from_op="reactor_1", to_op="bpr_1"),
        StreamConnection(stream_id="s5", from_op="bpr_1", to_op="collector_1"),
    ]
    return ProcessTopology(
        topology_id="gas_liquid",
        unit_operations=operations,
        streams=streams,
        reactor_volume_mL=10,
    )


def test_strict_allocator_assigns_every_physical_operation():
    compiled, report = InventoryAllocator(_strict_inventory(), _proposal()).compile(
        _gas_liquid_topology()
    )

    assert report["status"] == "complete"
    assert not report["unresolved_requirements"]
    assert report["checks"]["no_unknown_equipment_ids"]
    assert compiled.compilation_status == "inventory_assigned"
    assigned = {
        operation.op_id: operation.inventory_item_id
        for operation in compiled.unit_operations
        if operation.assignment_status == "assigned"
    }
    assert assigned == {
        "pump_a": "pump_1",
        "mfc_o2": "mfc_1",
        "mixer_1": "mixer_1",
        "reactor_1": "reactor_10",
        "led_1": "light_1",
        "bpr_1": "bpr_1",
    }


def test_component_family_uses_reactor_compatible_installations():
    inventory = _strict_inventory()
    inventory.reactors[0].system = "KHU tubing reactor"
    inventory.reactors[0].compatible_systems = ["Manual"]
    proposal = _proposal(
        inventory_selection={
            "equipment_id": "reactor_10",
            "system": "KHU tubing reactor",
        }
    )

    _, report = InventoryAllocator(inventory, proposal).compile(
        _gas_liquid_topology()
    )

    assert report["status"] == "complete"
    assert not report["unresolved_requirements"]


def test_undeclared_mixer_uses_visible_standard_accessory_assumption():
    compiled, report = InventoryAllocator(
        _strict_inventory(mixer=False), _proposal()
    ).compile(_gas_liquid_topology())

    assert report["status"] == "complete_with_assumptions"
    assert not report["unresolved_requirements"]
    assert report["assumed_standard_accessories"][0]["operation_id"] == "mixer_1"
    mixer = next(op for op in compiled.unit_operations if op.op_id == "mixer_1")
    assert mixer.assignment_status == "assumed_standard_accessory"
    assert mixer.parameters["pre_run_verification_required"] is True


def test_explicitly_unavailable_mixer_blocks_strict_assignment():
    inventory = _strict_inventory(mixer=False)
    inventory.mixers = [
        MixerSpec(
            equipment_id="mixer_unavailable",
            name="Unavailable mixer",
            service_status="unavailable",
        )
    ]

    _, report = InventoryAllocator(inventory, _proposal()).compile(
        _gas_liquid_topology()
    )

    assert report["status"] == "incomplete"
    assert any(
        item["operation_id"] == "mixer_1"
        for item in report["unresolved_requirements"]
    )


def test_equipment_quantity_cannot_be_reused_for_two_pumps():
    topology = _gas_liquid_topology()
    topology.unit_operations.insert(
        1,
        UnitOperation(op_id="pump_b", op_type="pump", label="Second feed", parameters={"flow_rate_mL_min": 0.05}),
    )
    topology.streams.append(
        StreamConnection(stream_id="s6", from_op="pump_b", to_op="mixer_1")
    )
    topology.unit_operations[3].parameters["max_inputs"] = 3
    inventory = _strict_inventory(pump_quantity=1)
    inventory.mixers[0].max_inputs = 3

    _, report = InventoryAllocator(inventory, _proposal()).compile(topology)

    assert report["status"] == "incomplete"
    assert any(item["operation_id"] == "pump_b" for item in report["unresolved_requirements"])


def _two_reactor_inventory(with_train: bool):
    inventory = _strict_inventory()
    inventory.reactors = [
        ReactorSpec(equipment_id="r20", name="20 mL reactor", system="Manual", type="coil", material="PFA", volume_mL=20, ID_mm=1, max_pressure_bar=8, max_temperature_C=80),
        ReactorSpec(equipment_id="r10", name="10 mL reactor", system="Manual", type="coil", material="PFA", volume_mL=10, ID_mm=1, max_pressure_bar=8, max_temperature_C=80),
    ]
    inventory.connectors = [
        ConnectorSpec(equipment_id="union_1", name="PFA union", material="PFA", max_pressure_bar=8)
    ]
    inventory.reactor_trains = [
        ReactorTrainSpec(
            equipment_id="train_30",
            name="30 mL serial train",
            component_reactor_ids=["r20", "r10"],
            connector_ids=["union_1"],
            total_volume_mL=30,
            max_pressure_bar=8,
        )
    ] if with_train else []
    return inventory


def _two_reactor_topology():
    return ProcessTopology(
        topology_id="two_stage",
        reactor_volume_mL=30,
        unit_operations=[
            UnitOperation(op_id="r1", op_type="coil_reactor", label="Stage 1", parameters={"volume_mL": 20, "ID_mm": 1, "material": "PFA", "temperature_C": 40}),
            UnitOperation(op_id="r2", op_type="coil_reactor", label="Stage 2", parameters={"volume_mL": 10, "ID_mm": 1, "material": "PFA", "temperature_C": 40}),
            UnitOperation(op_id="collector", op_type="collector", label="Collection"),
        ],
        streams=[
            StreamConnection(stream_id="s1", from_op="r1", to_op="r2"),
            StreamConnection(stream_id="s2", from_op="r2", to_op="collector"),
        ],
    )


def test_declared_serial_train_assigns_connectors_and_passes():
    proposal = _proposal(reactor_volume_mL=30, inventory_selection={"system": "Manual"})
    _, report = InventoryAllocator(
        _two_reactor_inventory(with_train=True), proposal
    ).compile(_two_reactor_topology())

    assert report["status"] == "complete"
    train_assignment = next(
        item for item in report["assignments"] if item["assignment_id"] == "assignment_reactor_train"
    )
    assert train_assignment["equipment_item_ids"] == ["train_30", "union_1"]


def test_undeclared_serial_train_blocks():
    proposal = _proposal(reactor_volume_mL=30, inventory_selection={"system": "Manual"})
    inventory = _two_reactor_inventory(with_train=False)
    inventory.standard_reactor_connectors_available = False
    _, report = InventoryAllocator(
        inventory, proposal
    ).compile(_two_reactor_topology())

    assert report["status"] == "incomplete"
    assert any(
        item["requirement_id"] == "INV-REACTOR-TRAIN"
        for item in report["unresolved_requirements"]
    )


def test_stocked_connectors_allow_direct_stages_without_declared_train():
    inventory = _two_reactor_inventory(with_train=False)
    inventory.connectors = []
    proposal = _proposal(reactor_volume_mL=30, inventory_selection={"system": "Manual"})
    _, report = InventoryAllocator(inventory, proposal).compile(_two_reactor_topology())
    assert report["status"] == "complete_with_assumptions"
    assert report["checks"]["serial_reactor_train_valid"]
    assert report["checks"]["no_unknown_equipment_ids"]
    connector = next(row for row in report["instrument_manifest"] if row["category"] == "connectors")
    assert connector["requires_pre_run_verification"]
    assert not inventory.reactor_trains


def test_explicitly_unavailable_connectors_override_stocked_policy():
    inventory = _two_reactor_inventory(with_train=False)
    inventory.capability_status["connectors"] = "unavailable"
    proposal = _proposal(reactor_volume_mL=30, inventory_selection={"system": "Manual"})
    _, report = InventoryAllocator(inventory, proposal).compile(_two_reactor_topology())
    assert any(row["requirement_id"] == "INV-REACTOR-TRAIN" for row in report["unresolved_requirements"])


def test_mixer_separated_reactors_do_not_require_declared_serial_train():
    inventory = _two_reactor_inventory(with_train=False)
    topology = _two_reactor_topology()
    topology.unit_operations.insert(
        1,
        UnitOperation(
            op_id="interstage_mixer",
            op_type="mixer",
            label="Interstage mixer",
            parameters={"Q_inlet_mL_min": 0.05},
        ),
    )
    topology.unit_operations.insert(
        2,
        UnitOperation(
            op_id="interstage_gas",
            op_type="mfc",
            label="Interstage O2",
            parameters={"contents": ["O2"], "gas_flow_sccm": 0.2},
        ),
    )
    topology.streams = [
        StreamConnection(stream_id="s1", from_op="r1", to_op="interstage_mixer"),
        StreamConnection(
            stream_id="s2",
            from_op="interstage_gas",
            to_op="interstage_mixer",
            stream_type="gas",
        ),
        StreamConnection(stream_id="s3", from_op="interstage_mixer", to_op="r2"),
        StreamConnection(stream_id="s4", from_op="r2", to_op="collector"),
    ]
    proposal = _proposal(
        reactor_volume_mL=30,
        inventory_selection={"system": "Manual"},
    )

    _, report = InventoryAllocator(inventory, proposal).compile(topology)

    assert not any(
        item["requirement_id"] == "INV-REACTOR-TRAIN"
        for item in report["unresolved_requirements"]
    )


def test_gas_formula_and_word_names_are_equivalent():
    assert _gas_matches("hydrogen", "H2")
    assert _gas_matches("O2 reagent gas", "oxygen")
    assert _gas_matches("nitrogen purge", "N2")
    assert not _gas_matches("hydrogen", "O2")
