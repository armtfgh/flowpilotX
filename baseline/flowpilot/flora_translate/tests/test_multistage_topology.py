from flora_translate.main import (
    _build_multistep_topology,
    _topology_matches_serialized_proposal,
)
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    FlowProposal,
    ProcessStage,
    StreamAssignment,
    StreamLogic,
)


def test_multistage_topology_allocates_global_council_tau_and_keeps_air_as_mfc():
    batch = BatchRecord(
        reaction_description="One-pot photoredox Giese addition + aerobic oxidation",
        reaction_time_h=10.0,
        concentration_M=0.1,
        scale_mmol=0.2,
        solvent="EtOH / pH 9 buffer",
        temperature_C=25,
    )
    plan = ChemistryPlan(
        reaction_name="One-pot photoredox Giese addition + aerobic oxidation",
        reaction_class="photoredox aerobic oxidation",
        n_stages=2,
        stages=[
            ProcessStage(
                stage_number=1,
                stage_name="Giese radical addition",
                reaction_type="photoredox radical addition",
                reactor_type="coil",
                temperature_C=25,
                requires_light=True,
                wavelength_nm=452,
                batch_time_h=4.0,
                solvent="EtOH / pH 9 buffer",
                atmosphere="Ar",
                oxygen_sensitive=True,
                deoxygenation_required=True,
                feed_streams=[
                    StreamLogic(
                        stream_label="A",
                        reagents=["PMPSCH2TMS", "Ir photocatalyst"],
                        reasoning="O2-free substrate/catalyst solution",
                        molar_equiv=1.0,
                    ),
                    StreamLogic(
                        stream_label="B",
                        reagents=["acrylonitrile"],
                        reasoning="Michael acceptor",
                        molar_equiv=2.0,
                    ),
                ],
            ),
            ProcessStage(
                stage_number=2,
                stage_name="Aerobic sulfide oxidation",
                reaction_type="photoredox aerobic oxidation",
                reactor_type="coil",
                temperature_C=25,
                requires_light=True,
                wavelength_nm=452,
                batch_time_h=6.0,
                solvent="EtOH / pH 9 buffer",
                atmosphere="air",
                feed_streams=[
                    StreamLogic(
                        stream_label="C",
                        reagents=["Air"],
                        reasoning="O2 gas feed",
                        molar_equiv=1.0,
                    )
                ],
            ),
        ],
    )
    proposal = FlowProposal(
        residence_time_min=38.38,
        flow_rate_mL_min=0.2234,
        tubing_ID_mm=0.5,
        reactor_volume_mL=8.574,
        temperature_C=25,
        concentration_M=0.1,
        BPR_bar=5,
        tubing_material="FEP",
        wavelength_nm=452,
        streams=[
            StreamAssignment(
                stream_label="A",
                contents=["PMPSCH2TMS", "Ir photocatalyst"],
                solvent="EtOH / pH 9 buffer",
                flow_rate_mL_min=0.0745,
            ),
            StreamAssignment(
                stream_label="B",
                contents=["acrylonitrile"],
                solvent="EtOH / pH 9 buffer",
                flow_rate_mL_min=0.1489,
            ),
            StreamAssignment(
                stream_label="C",
                pump_role="gas injection",
                contents=["Air"],
                flow_rate_mL_min=0.0,
            ),
        ],
        # Legacy bad state: only stage 1 received the full council tau.
        stage_parameters=[{"stage_number": 1, "residence_time_min": 38.38, "d_mm": 0.5}],
    )

    topology = _build_multistep_topology(proposal, plan, batch)
    reactors = [
        op for op in topology.unit_operations
        if op.op_type == "coil_reactor"
    ]
    assert [r.parameters["residence_time_min"] for r in reactors] == [15.35, 23.03]
    assert [r.parameters["Q_inlet_mL_min"] for r in reactors] == [0.2234, 0.2234]

    gas_feed = next(op for op in topology.unit_operations if op.op_id == "pump_c")
    assert gas_feed.op_type == "mfc"
    assert gas_feed.parameters["flow_rate_mL_min"] == 0.0
    assert any(
        op.op_type == "deoxygenation_unit"
        for op in topology.unit_operations
    )

    offline_proposal = proposal.model_copy(
        update={
            "deoxygenation_method": (
                "Offline argon sparging followed by transfer to sealed "
                "argon-blanketed reservoirs. No inline degasser used."
            ),
            "pre_reactor_steps": [
                "Prepare pre-degassed Stage 1 feeds before pumping."
            ],
        }
    )
    offline_topology = _build_multistep_topology(
        offline_proposal,
        plan,
        batch,
    )
    assert not any(
        op.op_type == "deoxygenation_unit"
        for op in offline_topology.unit_operations
    )

    inventory_train_proposal = offline_proposal.model_copy(
        update={
            "inventory_selection": {
                "component_volumes_mL": [4.25, 4.25],
            },
        }
    )
    inventory_topology = _build_multistep_topology(
        inventory_train_proposal,
        plan,
        batch,
    )
    inventory_reactors = [
        op
        for op in inventory_topology.unit_operations
        if op.op_type == "coil_reactor"
    ]
    assert [op.parameters["volume_mL"] for op in inventory_reactors] == [
        4.25,
        4.25,
    ]
    assert inventory_topology.reactor_volume_mL == 8.5
    assert all(
        op.parameters.get("material") != "PEEK"
        for op in inventory_topology.unit_operations
    )

    pfa_inventory_proposal = inventory_train_proposal.model_copy(
        update={
            "BPR_bar": 3.0,
            "tubing_material": "PFA",
            "inventory_selection": {
                "material": "PFA",
                "component_volumes_mL": [4.25, 4.25],
            },
        }
    )
    pfa_topology = _build_multistep_topology(
        pfa_inventory_proposal,
        plan,
        batch,
    )
    pfa_reactors = [
        op for op in pfa_topology.unit_operations if op.op_type == "coil_reactor"
    ]
    assert {op.parameters["material"] for op in pfa_reactors} == {"PFA"}
    assert {op.parameters["temperature_C"] for op in pfa_reactors} == {25.0}
    bpr = next(op for op in pfa_topology.unit_operations if op.op_type == "bpr")
    assert bpr.parameters["pressure_bar"] == 3.0


def test_downstream_liquid_feed_is_not_double_counted_in_stage_one():
    batch = BatchRecord(
        reaction_description="Two-stage amide coupling",
        reaction_time_h=1.0,
        concentration_M=0.5,
        temperature_C=80,
    )
    plan = ChemistryPlan(
        reaction_name="Two-stage amide coupling",
        reaction_class="amide coupling",
        n_stages=2,
        stages=[
            ProcessStage(
                stage_number=1,
                stage_name="Activation",
                reaction_type="thioester formation",
                reactor_type="coil",
                temperature_C=80,
                solvent="2-MeTHF",
                feed_streams=[
                    StreamLogic(
                        stream_label="A",
                        reagents=["acid", "DPDTC", "DMAP"],
                        concentration_M=0.5,
                    )
                ],
            ),
            ProcessStage(
                stage_number=2,
                stage_name="Aminolysis",
                reaction_type="nucleophilic acyl substitution",
                reactor_type="coil",
                temperature_C=80,
                solvent="2-MeTHF",
                feed_streams=[
                    StreamLogic(
                        stream_label="B",
                        reagents=["amine"],
                        concentration_M=0.5,
                    )
                ],
            ),
        ],
    )
    proposal = FlowProposal(
        residence_time_min=12.5,
        residence_time_inlet_min=12.5,
        residence_time_in_channel_min=12.5,
        flow_rate_mL_min=2.0,
        tubing_ID_mm=1.0,
        reactor_volume_mL=15,
        temperature_C=80,
        concentration_M=0.5,
        BPR_bar=2.5,
        tubing_material="PFA",
        streams=[
            StreamAssignment(
                stream_label="A",
                contents=["acid", "DPDTC", "DMAP"],
                solvent="2-MeTHF",
                flow_rate_mL_min=1.0,
            ),
            StreamAssignment(
                stream_label="B",
                contents=["amine"],
                solvent="2-MeTHF",
                flow_rate_mL_min=1.0,
            ),
        ],
        stage_parameters=[
            {
                "stage_number": 1,
                "reactor_volume_mL": 10.0,
                "residence_time_min": 10.0,
                "residence_time_inlet_min": 10.0,
                "residence_time_in_channel_min": 10.0,
                "Q_liquid_mL_min": 1.0,
                "d_mm": 1.0,
            },
            {
                "stage_number": 2,
                "reactor_volume_mL": 5.0,
                "residence_time_min": 2.5,
                "residence_time_inlet_min": 2.5,
                "residence_time_in_channel_min": 2.5,
                "Q_liquid_mL_min": 2.0,
                "d_mm": 1.0,
            },
        ],
    )

    topology = _build_multistep_topology(proposal, plan, batch)
    pumps = {
        operation.op_id: operation.parameters["flow_rate_mL_min"]
        for operation in topology.unit_operations
        if operation.op_type == "pump"
    }
    reactors = [
        operation
        for operation in topology.unit_operations
        if operation.op_type == "coil_reactor"
    ]

    assert pumps == {"pump_a": 1.0, "pump_b": 1.0}
    assert [item.parameters["Q_inlet_mL_min"] for item in reactors] == [1.0, 2.0]
    assert topology.total_flow_rate_mL_min == 2.0
    assert _topology_matches_serialized_proposal(topology, proposal)


def test_feed_workup_wording_cannot_suppress_finalized_reaction_stage():
    batch = BatchRecord(
        reaction_description="Two-stage oxidative amidation",
        reaction_time_h=1.0,
        concentration_M=0.667,
        temperature_C=80,
    )
    plan = ChemistryPlan(
        reaction_name="Two-stage oxidative amidation",
        n_stages=2,
        stages=[
            ProcessStage(
                stage_number=1,
                stage_name="Imine formation",
                reaction_type="condensation",
                temperature_C=70,
                feed_streams=[
                    StreamLogic(stream_label="A", reagents=["benzaldehyde"]),
                    StreamLogic(stream_label="B", reagents=["morpholine"]),
                ],
            ),
            ProcessStage(
                stage_number=2,
                stage_name="Oxidative amidation",
                reaction_type="oxidative amidation",
                temperature_C=80,
                feed_streams=[
                    StreamLogic(
                        stream_label="C",
                        reagents=["aqueous TBHP"],
                        reasoning="Oxidant addition before the later aqueous workup",
                    )
                ],
                post_stage_action="aqueous workup after reaction",
            ),
        ],
    )
    proposal = FlowProposal(
        residence_time_min=36.7432,
        flow_rate_mL_min=0.435999,
        reactor_volume_mL=15.04,
        concentration_M=0.667,
        temperature_C=80,
        tubing_ID_mm=1.0,
        streams=[
            StreamAssignment(stream_label="A", flow_rate_mL_min=0.145333),
            StreamAssignment(stream_label="B", flow_rate_mL_min=0.145333),
            StreamAssignment(stream_label="C", flow_rate_mL_min=0.145333),
        ],
        stage_parameters=[
            {
                "stage_number": 1,
                "reactor_equipment_id": "reactor_1",
                "reactor_volume_mL": 1.96,
                "residence_time_min": 6.7431,
                "Q_liquid_mL_min": 0.290666,
            },
            {
                "stage_number": 2,
                "reactor_equipment_id": "reactor_2",
                "reactor_volume_mL": 13.08,
                "residence_time_min": 30.0001,
                "Q_liquid_mL_min": 0.435999,
            },
        ],
    )

    topology = _build_multistep_topology(proposal, plan, batch)
    reactors = [
        operation
        for operation in topology.unit_operations
        if operation.op_type == "coil_reactor"
    ]

    assert len(reactors) == 2
    assert [item.parameters["volume_mL"] for item in reactors] == [1.96, 13.08]
    assert topology.reactor_volume_mL == 15.04
    assert _topology_matches_serialized_proposal(topology, proposal)


def test_final_gas_substitution_overrides_stale_plan_gas_identity_in_topology():
    batch = BatchRecord(reaction_description="Aerobic oxidation")
    plan = ChemistryPlan(
        n_stages=2,
        stages=[
            ProcessStage(
                stage_number=1,
                stage_name="Reaction",
                feed_streams=[StreamLogic(stream_label="A", reagents=["substrate"])],
            ),
            ProcessStage(
                stage_number=2,
                stage_name="Aerobic oxidation",
                feed_streams=[
                    StreamLogic(
                        stream_label="B",
                        reagents=["Air"],
                        phase="gas",
                        introduction_stage=2,
                    )
                ],
            ),
        ],
    )
    proposal = FlowProposal(
        residence_time_min=20,
        flow_rate_mL_min=0.1,
        reactor_volume_mL=2,
        BPR_bar=7,
        streams=[
            StreamAssignment(stream_label="A", flow_rate_mL_min=0.1),
            StreamAssignment(
                stream_label="B",
                pump_role="Pure O2 oxidant feed (inventory substitution)",
                contents=["O2"],
                phase="gas",
                gas_flow_sccm=0.2,
                gas_flow_actual_mL_min=0.03,
                pump_equipment_id="o2_mfc",
                introduction_stage=2,
            ),
        ],
        stage_parameters=[
            {"stage_number": 1, "reactor_volume_mL": 1, "residence_time_min": 10},
            {"stage_number": 2, "reactor_volume_mL": 1, "residence_time_min": 10},
        ],
    )

    topology = _build_multistep_topology(proposal, plan, batch)
    gas = next(operation for operation in topology.unit_operations if operation.op_type == "mfc")

    assert gas.parameters["contents"] == ["O2"]
    assert gas.parameters["inventory_equipment_id"] == "o2_mfc"
    assert "Pure O2" in gas.label
