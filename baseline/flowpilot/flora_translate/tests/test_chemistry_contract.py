from flora_translate.chemistry_contract import reconcile_chemistry_plan
from flora_translate.design_realizer import _normalized_streams, _reconcile_candidate_operations
from flora_translate.design_calculator import _extract_gas_equiv
from flora_translate.main import _build_multistep_topology
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    FlowProposal,
    ProcessStage,
    ReagentRole,
    StreamAssignment,
    StreamLogic,
)
from flora_translate.topology_preflight import build_requirements_topology, _required_counts


def test_hydrogenolysis_merges_model_split_liquids_and_keeps_protocol_h2():
    batch = BatchRecord(
        reaction_description="Catalytic hydrogenolysis under hydrogen gas.",
        atmosphere="H2",
        raw_text="Substrate, catalyst, and methanol were charged together and stirred under H2.",
    )
    plan = ChemistryPlan(
        reaction_name="Hydrogenolysis",
        stream_logic=[
            StreamLogic(stream_label="A", reagents=["substrate"], phase="liquid"),
            StreamLogic(stream_label="B", reagents=["catalyst slurry"], phase="liquid"),
            StreamLogic(stream_label="G", reagents=["H2"], phase="gas"),
        ],
    )

    reconciled, report = reconcile_chemistry_plan(batch, plan)
    topology = build_requirements_topology(reconciled)

    assert report["required_reagent_gases"] == ["H2"]
    assert _required_counts(topology)["pumps"] == 1
    assert _required_counts(topology)["gas_hardware"] == 1
    assert reconciled.stream_logic[0].reagents == ["substrate", "catalyst slurry"]
    assert reconciled.canonical_contract is not None


def test_cuaac_removes_hallucinated_light_and_oxygen_requirements():
    batch = BatchRecord(
        reaction_description="Copper-catalyzed azide-alkyne cycloaddition.",
        atmosphere="nitrogen",
        raw_text="Azide, alkyne, CuSO4, and sodium ascorbate were stirred at 40 C.",
    )
    plan = ChemistryPlan(
        reaction_name="CuAAC",
        recommended_wavelength_nm=450,
        light_sensitive_reagents=["copper catalyst"],
        stages=[
            ProcessStage(
                stage_number=1,
                requires_light=True,
                wavelength_nm=450,
                feed_streams=[
                    StreamLogic(stream_label="A", reagents=["azide"], phase="liquid"),
                    StreamLogic(stream_label="B", reagents=["alkyne"], phase="liquid"),
                    StreamLogic(stream_label="G", reagents=["O2"], phase="gas"),
                ],
            )
        ],
    )

    reconciled, report = reconcile_chemistry_plan(batch, plan)
    counts = _required_counts(build_requirements_topology(reconciled))

    assert report["required_reagent_gases"] == []
    assert counts["pumps"] == 1
    assert counts["gas_hardware"] == 0
    assert counts["light_sources"] == 0
    assert not reconciled.stages[0].requires_light
    assert any(
        item["decision"] == "remove_unsupported_reagent_gas"
        for item in report["decisions"]
    )


def test_explicit_two_feed_constraint_preserves_two_pumps():
    batch = BatchRecord(raw_text="The reagents are combined at the reactor inlet.")
    plan = ChemistryPlan(
        stream_logic=[
            StreamLogic(stream_label="A", reagents=["substrate"]),
            StreamLogic(stream_label="B", reagents=["unstable reagent"]),
        ]
    )

    reconciled, _ = reconcile_chemistry_plan(
        batch,
        plan,
        hard_constraints="Use two separate liquid feeds through a T-mixer.",
    )

    assert _required_counts(build_requirements_topology(reconciled))["pumps"] == 2
    assert all(feed.separate_feed_required for feed in reconciled.stream_logic)


def test_photochemical_protocol_preserves_light_requirement():
    batch = BatchRecord(
        raw_text="The reaction was irradiated with a 450 nm LED for 8 h.",
        light_source="450 nm LED",
        wavelength_nm=450,
    )
    plan = ChemistryPlan(
        stream_logic=[StreamLogic(stream_label="A", reagents=["reaction mixture"])],
    )

    reconciled, _ = reconcile_chemistry_plan(batch, plan)
    counts = _required_counts(build_requirements_topology(reconciled))

    assert counts["light_sources"] == 1
    assert reconciled.stages[0].requires_light


def test_empty_first_stage_restores_protocol_anchored_liquid_feed_and_photocatalyst():
    catalyst = "Ir(dF(CF3)ppy)2(dtbpy)PF6"
    batch = BatchRecord(
        raw_text=(
            "Substrate A and acrylonitrile were charged in ethanol. "
            f"Photocatalyst: {catalyst}, 0.5 mol%. The mixture was irradiated."
        ),
        photocatalyst=catalyst,
        catalyst_loading_mol_pct=0.5,
        concentration_M=0.1,
    )
    plan = ChemistryPlan(
        n_stages=2,
        reagents=[
            ReagentRole(name="Substrate A", role="substrate"),
            ReagentRole(name=catalyst, role="substrate"),
        ],
        stages=[
            ProcessStage(stage_number=1, stage_name="photoredox", feed_streams=[]),
            ProcessStage(stage_number=2, stage_name="oxidation", feed_streams=[]),
        ],
    )

    reconciled, report = reconcile_chemistry_plan(batch, plan)

    feed = reconciled.stages[0].feed_streams[0]
    assert feed.phase == "liquid"
    assert feed.requirement_authority == "protocol_fact"
    assert feed.concentration_M == 0.1
    assert "Substrate A" in feed.reagents
    assert next(item for item in reconciled.reagents if item.name == catalyst).role == "photocatalyst"
    assert {item["decision"] for item in report["decisions"]} >= {
        "restore_missing_initial_liquid_feed",
        "restore_protocol_photocatalyst_role",
    }


def test_inventory_pump_notes_do_not_hide_qwen_first_stage_feed():
    catalyst = "Ir(dF(CF3)ppy)2(dtbpy)PF6"
    batch = BatchRecord(
        raw_text=(
            "Substrate/radical precursor: PMPSCH2TMS (1a), 0.20 mmol, 1.0 equiv. "
            "Michael acceptor: Acrylonitrile (2a), 2.0 equiv. "
            f"Photocatalyst: {catalyst}, 0.5 mol%. "
            "Step 1: irradiate under argon. Step 2: open to air and irradiate."
        ),
        photocatalyst=catalyst,
        catalyst_loading_mol_pct=0.5,
        concentration_M=0.1,
    )
    repeated = StreamLogic(
        stream_label="A",
        reagents=[catalyst, "PMPSCH2TMS (1a)", "Acrylonitrile (2a)"],
        introduction_stage=1,
        separate_feed_required=True,
        requirement_authority="hard_constraint",
    )
    plan = ChemistryPlan(
        n_stages=2,
        reagents=[ReagentRole(name=catalyst, role="substrate")],
        stages=[
            ProcessStage(
                stage_number=1,
                stage_name="Giese addition",
                feed_streams=[repeated],
            ),
            ProcessStage(
                stage_number=2,
                stage_name="Aerobic oxidation",
                feed_streams=[repeated.model_copy(deep=True)],
            ),
        ],
    )

    reconciled, report = reconcile_chemistry_plan(
        batch,
        plan,
        hard_constraints={
            "inventory_constraints": {
                "pump_notes": "Two pumps are available for the Vapourtec system."
            },
            "runtime_hard_constraints": [],
        },
    )

    stage_one = reconciled.stages[0].feed_streams
    stage_two = reconciled.stages[1].feed_streams
    assert report["explicit_separate_liquid_feeds"] is False
    assert len(stage_one) == 1
    assert any(item.startswith(catalyst + " (0.5 mol%)") for item in stage_one[0].reagents)
    assert len([item for item in stage_one[0].reagents if item.startswith("PMPSCH2TMS")]) == 1
    assert len([item for item in stage_one[0].reagents if item.startswith("Acrylonitrile")]) == 1
    assert all(feed.stream_label != "A" for feed in stage_two)
    decisions = {item["decision"] for item in report["decisions"]}
    assert "remove_duplicate_cross_stage_feed" in decisions

    proposal = FlowProposal(
        residence_time_min=25,
        flow_rate_mL_min=0.05,
        reactor_volume_mL=1.25,
        streams=[
            StreamAssignment(
                stream_label="A",
                contents=stage_one[0].reagents,
                phase="liquid",
                flow_rate_mL_min=0.05,
            )
        ],
    )
    topology = _build_multistep_topology(proposal, reconciled, batch)
    assert any(op.op_id == "st1_reactor" for op in topology.unit_operations)
    assert any(stream.to_op == "st1_reactor" for stream in topology.streams)


def test_downstream_candidate_cannot_reintroduce_contract_extras():
    batch = BatchRecord(raw_text="Substrate and catalyst were stirred at 50 C.")
    reconciled, _ = reconcile_chemistry_plan(
        batch,
        ChemistryPlan(
            stream_logic=[StreamLogic(stream_label="A", reagents=["reaction mixture"])]
        ),
    )
    decisions = []
    streams = _normalized_streams(
        [
            StreamAssignment(stream_label="A", contents=["reaction mixture"]),
            StreamAssignment(stream_label="G", contents=["O2"], phase="gas"),
        ],
        reconciled,
        batch.raw_text,
        decisions=decisions,
    )
    proposal = FlowProposal(light_setup="450 nm LED", wavelength_nm=450)
    _reconcile_candidate_operations(proposal, reconciled, decisions)

    assert [stream.stream_label for stream in streams] == ["A"]
    assert proposal.light_setup == ""
    assert proposal.wavelength_nm is None
    assert {item["decision"] for item in decisions} >= {
        "remove_stream_outside_canonical_contract",
        "remove_light_outside_canonical_contract",
    }


def test_multistage_protocol_does_not_spread_light_to_thermal_stage():
    batch = BatchRecord(raw_text="Stage 1 was irradiated with a 450 nm LED, then heated in stage 2.")
    plan = ChemistryPlan(
        n_stages=2,
        stages=[
            ProcessStage(
                stage_number=1,
                stage_name="Photochemical step",
                requires_light=True,
                wavelength_nm=450,
                feed_streams=[StreamLogic(stream_label="A", reagents=["substrate"])],
            ),
            ProcessStage(
                stage_number=2,
                stage_name="Thermal step",
                requires_light=False,
                feed_streams=[
                    StreamLogic(
                        stream_label="A",
                        reagents=["stage 1 outlet"],
                        delivery_mode="carried_from_previous",
                        introduction_stage=1,
                    )
                ],
            ),
        ],
    )

    reconciled, report = reconcile_chemistry_plan(batch, plan)

    assert report["light_required_stages"] == [1]
    assert [stage.requires_light for stage in reconciled.stages] == [True, False]
    assert _required_counts(build_requirements_topology(reconciled))["light_sources"] == 1


def test_stationary_packed_bed_catalyst_does_not_consume_pump_slot():
    batch = BatchRecord(
        raw_text="The methanolic substrate was contacted with H2 over Pd/C in a packed-bed reactor.",
        atmosphere="H2",
    )
    plan = ChemistryPlan(
        stream_logic=[
            StreamLogic(stream_label="A", reagents=["substrate in methanol"], phase="liquid"),
            StreamLogic(stream_label="G", reagents=["H2"], phase="gas"),
            StreamLogic(
                stream_label="C",
                reagents=["Pd/C"],
                phase="solid",
                reasoning="Immobilized packed-bed catalyst; not pumped.",
            ),
        ]
    )

    reconciled, report = reconcile_chemistry_plan(batch, plan)
    counts = _required_counts(build_requirements_topology(reconciled))

    assert counts["pumps"] == 1
    catalyst = next(feed for feed in reconciled.stream_logic if feed.stream_label == "C")
    assert catalyst.accepted_requirement is False
    assert any(
        row["decision"] == "classify_stationary_bed_material_as_non_pumped"
        for row in report["decisions"]
    )


def test_hydrogen_pressure_atmosphere_is_recognized_as_reagent_gas():
    batch = BatchRecord(
        raw_text="The substrate was contacted with hydrogen over Pd/C.",
        atmosphere="hydrogen, 21 bar",
    )
    plan = ChemistryPlan(
        stream_logic=[
            StreamLogic(stream_label="A", reagents=["substrate"]),
            StreamLogic(stream_label="G", reagents=["hydrogen gas"], phase="gas"),
        ]
    )

    reconciled, report = reconcile_chemistry_plan(batch, plan)

    assert report["required_reagent_gases"] == ["H2"]
    assert _required_counts(build_requirements_topology(reconciled))["gas_hardware"] == 1
    gas = next(feed for feed in reconciled.stream_logic if feed.phase == "gas")
    assert gas.molar_equiv == 1.0
    assert gas.molar_equiv_basis == "deterministic_screening_assumption"


def test_protocol_explicit_gas_equivalents_override_model_calculation():
    batch = BatchRecord(raw_text="O2 gas (2.0 equiv) was bubbled through the mixture.")
    plan = ChemistryPlan(
        stream_logic=[
            StreamLogic(stream_label="A", reagents=["substrate"]),
            StreamLogic(
                stream_label="G",
                reagents=["O2 (7.5 equiv inferred from a model flow)"],
                phase="gas",
                molar_equiv=7.5,
                reasoning="Model calculation gives 7.5 equiv.",
            ),
        ]
    )

    reconciled, _ = reconcile_chemistry_plan(batch, plan)
    gas = next(feed for feed in reconciled.stream_logic if feed.phase == "gas")

    assert gas.molar_equiv == 2.0
    assert gas.molar_equiv_basis == "protocol_fact"
    assert _extract_gas_equiv(reconciled, batch_record=batch) == 2.0


def test_model_reasoning_cannot_become_protocol_gas_equivalents():
    batch = BatchRecord(raw_text="The substrate was contacted with hydrogen at 21 bar.")
    plan = ChemistryPlan(
        stream_logic=[
            StreamLogic(
                stream_label="G",
                reagents=["H2"],
                phase="gas",
                molar_equiv=8.9,
                molar_equiv_basis="model_inference",
                reasoning="Calculated as 8.9 equiv from a proposed flow.",
            )
        ]
    )

    assert _extract_gas_equiv(plan, batch_record=batch) == 1.0


def test_microwave_heating_does_not_create_photochemical_light_requirement():
    batch = BatchRecord(
        raw_text="The mixture was heated under microwave irradiation at 150 C.",
        light_source="microwave irradiation (dielectric heating, not photochemical)",
    )
    plan = ChemistryPlan(
        recommended_wavelength_nm=450,
        stages=[
            ProcessStage(
                stage_number=1,
                requires_light=True,
                wavelength_nm=450,
                feed_streams=[StreamLogic(stream_label="A", reagents=["reaction mixture"])],
            )
        ],
    )

    reconciled, report = reconcile_chemistry_plan(batch, plan)

    assert report["light_required"] is False
    assert _required_counts(build_requirements_topology(reconciled))["light_sources"] == 0


def test_excluded_zero_equivalent_component_cannot_survive_into_final_stream():
    batch = BatchRecord(
        raw_text="Acetic acid is optional and must not be introduced unless explicitly justified."
    )
    plan = ChemistryPlan(
        stream_logic=[
            StreamLogic(
                stream_label="A",
                reagents=["substrate (1.0 equiv)", "methanol", "acetic acid (0 equiv - deliberately excluded)"],
            )
        ]
    )
    reconciled, report = reconcile_chemistry_plan(batch, plan)
    decisions = []
    streams = _normalized_streams(
        [
            StreamAssignment(
                stream_label="A",
                contents=["substrate", "methanol", "acetic acid (0 equiv - deliberately excluded)"],
            )
        ],
        reconciled,
        batch.raw_text,
        decisions=decisions,
    )

    assert all("acetic" not in item.lower() for item in streams[0].contents)
    assert any("substrate" in item.lower() for item in streams[0].contents)
    assert any(row["decision"] == "remove_explicitly_excluded_component" for row in report["decisions"])
    assert any(row["decision"] == "restore_canonical_stream_composition" for row in decisions)
