from __future__ import annotations

from flora_translate.design_calculator import DesignCalculator, _extract_gas_equiv
from flora_translate.chemistry_agent import _normalize_plan_data
from flora_translate.engine.sampling import L_MAX_BENCH_M, compute_metrics, generate_candidates, hard_filter
from flora_translate.engine.council_v4.chief import _intensification_feasibility_precheck
from flora_translate.engine.council_v4.skeptic import _verify_v_r_equals_tau_q
from flora_translate.main import (
    _build_multistep_topology,
    _build_singlestep_topology,
    _reconcile_final_bpr,
    _sync_final_stream_flowrates,
)
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    FlowProposal,
    LabInventory,
    PressureControllerSpec,
    ProcessStage,
    StreamAssignment,
    StreamLogic,
)


def test_gas_liquid_calculation_corrects_volume_and_adds_o2_metrics():
    batch = BatchRecord(
        reaction_description="Photoredox aerobic oxidation with molecular oxygen from air.",
        solvent="EtOH / water",
        temperature_C=25,
        reaction_time_h=6,
        concentration_M=0.1,
        scale_mmol=0.2,
        atmosphere="air",
    )
    plan = ChemistryPlan(
        reaction_class="photoredox oxidation",
        mechanism_type="gas-liquid photoredox oxidation",
        stages=[
            ProcessStage(
                stage_number=1,
                stage_name="Aerobic oxidation",
                atmosphere="air",
                requires_light=True,
                feed_streams=[
                    StreamLogic(stream_label="A", reagents=["sulfide solution"], concentration_M=0.1),
                    StreamLogic(stream_label="G", reagents=["air"], phase="gas", reasoning="O2 feed"),
                ],
            )
        ],
    )
    proposal = FlowProposal(
        residence_time_min=10,
        flow_rate_mL_min=1.0,
        concentration_M=0.1,
        temperature_C=25,
        BPR_bar=5.0,
        tubing_ID_mm=1.0,
        streams=[
            StreamAssignment(stream_label="A", pump_role="liquid substrate", solvent="EtOH/water", concentration_M=0.1),
            StreamAssignment(stream_label="G", pump_role="air gas feed", contents=["air"], phase="gas"),
        ],
    )

    calc = DesignCalculator().run(batch, chemistry_plan=plan, proposal=proposal)
    proposal = DesignCalculator.annotate_proposal_with_calculations(proposal, calc)
    proposal.BPR_bar = 0.0
    topology = _build_multistep_topology(proposal, plan, batch)

    assert calc.is_gas_liquid
    assert calc.gas_species == "air"
    assert calc.gas_flow_sccm > 0
    assert calc.gas_flow_actual_mL_min > 0
    assert calc.gas_holdup > 0
    assert calc.reactor_volume_mL > calc.liquid_holdup_volume_mL
    assert abs(calc.o2_equiv_supplied - 1.0) < 0.02
    assert abs(calc.target_gas_equiv_inlet - 1.0) < 0.02
    assert calc.UA_W_K > 0
    bprs = [op for op in topology.unit_operations if op.op_type == "bpr"]
    assert bprs
    assert bprs[-1].parameters["pressure_bar"] >= 5.0


def test_liquid_oxidant_does_not_invent_an_air_feed():
    batch = BatchRecord(
        reaction_description="TBHP oxidation of benzyl alcohol followed by amidation.",
        raw_text=(
            "Benzyl alcohol was oxidized with TBHP, NaBr, and sulfuric acid "
            "in dioxane before morpholine addition."
        ),
        solvent="dioxane",
        temperature_C=80,
        reaction_time_h=0.5,
        concentration_M=0.667,
        atmosphere=None,
    )
    plan = ChemistryPlan(
        reaction_class="Sequential alcohol oxidation and oxidative amidation",
        mechanism_type="Bromide-mediated peroxide oxidation",
        stages=[
            ProcessStage(
                stage_number=1,
                stage_name="Liquid-phase oxidation",
                atmosphere="Air",
                feed_streams=[
                    StreamLogic(
                        stream_label="A",
                        reagents=["benzyl alcohol", "TBHP"],
                        phase="liquid",
                    )
                ],
            )
        ],
        stream_logic=[
            StreamLogic(
                stream_label="A",
                reagents=["benzyl alcohol", "TBHP"],
                phase="liquid",
            )
        ],
    )
    proposal = FlowProposal(
        residence_time_min=30,
        flow_rate_mL_min=0.5,
        reactor_volume_mL=15,
        concentration_M=0.667,
        temperature_C=80,
        BPR_bar=5,
        tubing_ID_mm=1.0,
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="liquid oxidation feed",
                contents=["benzyl alcohol", "TBHP"],
                solvent="dioxane",
                phase="liquid",
            )
        ],
    )

    calc = DesignCalculator().run(batch, chemistry_plan=plan, proposal=proposal)

    assert not calc.is_gas_liquid
    assert calc.gas_flow_sccm == 0
    assert calc.gas_flow_actual_mL_min == 0


def test_o2_mfc_is_recomputed_from_target_equivalents_not_llm_stream_guess():
    batch = BatchRecord(
        reaction_description="THQ aerobic oxidation with O2 (2.0 equiv) in DMSO.",
        solvent="DMSO",
        temperature_C=40,
        reaction_time_h=15,
        concentration_M=0.5,
        atmosphere="O2",
    )
    plan = ChemistryPlan(
        reaction_class="photochemical aerobic oxidation",
        mechanism_type="O2-mediated photochemical oxidation",
        o2_is_reagent=True,
        stream_logic=[
            StreamLogic(stream_label="A", reagents=["THQ"], concentration_M=0.5, phase="liquid"),
            StreamLogic(stream_label="G", reagents=["O2"], phase="gas", molar_equiv=2.0),
        ],
    )
    proposal = FlowProposal(
        residence_time_min=68.6,
        residence_time_basis="inlet/STP apparent residence time",
        flow_rate_mL_min=0.06,
        concentration_M=0.5,
        temperature_C=40,
        BPR_bar=6.0,
        tubing_ID_mm=1.0,
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="substrate solution",
                contents=["THQ"],
                solvent="DMSO",
                concentration_M=0.5,
                flow_rate_mL_min=0.06,
            ),
            StreamAssignment(
                stream_label="G",
                pump_role="O2 gas feed",
                contents=["O2"],
                phase="gas",
                gas_flow_sccm=0.04,
                gas_flow_actual_mL_min=0.04,
                flow_rate_mL_min=0.04,
            ),
        ],
    )

    calc = DesignCalculator().run(batch, chemistry_plan=plan, proposal=proposal)
    proposal = DesignCalculator.annotate_proposal_with_calculations(proposal, calc)

    assert calc.gas_flow_recomputed_from_equiv
    assert abs(calc.gas_flow_sccm - 1.344) < 0.02
    assert abs(calc.o2_equiv_supplied - 2.0) < 0.02
    assert calc.gas_flow_actual_mL_min > 0.2
    assert proposal.streams[1].gas_flow_sccm > 1.3
    assert proposal.streams[1].molar_equiv == 2.0


def test_unstated_air_equivalents_ignore_model_inference():
    batch = BatchRecord(
        reaction_description="The solution was exposed to air.",
        raw_text="The solution was exposed to air and irradiated for 5 min.",
        atmosphere="air",
    )
    plan = ChemistryPlan(
        reaction_class="aerobic photooxidation",
        stream_logic=[
            StreamLogic(
                stream_label="G",
                reagents=["air"],
                phase="gas",
                reasoning="An inferred feed would provide 1.12 equiv O2.",
                molar_equiv=5.33,
            )
        ],
    )

    assert _extract_gas_equiv(plan, batch_record=batch) == 1.0


def test_confirmed_air_equivalents_are_not_shadowed_by_formula_substrings():
    protocol = (
        "PMPSCH2TMS (1a), 0.20 mmol, 1.0 equiv, and acrylonitrile, "
        "2.0 equiv, are irradiated oxygen-free. Stage 2 is open to air."
    )
    batch = BatchRecord(
        reaction_description=protocol,
        raw_text=protocol,
        atmosphere="air",
    )
    plan = ChemistryPlan(
        reaction_class="aerobic photooxidation",
        stream_logic=[
            StreamLogic(
                stream_label="G",
                reagents=["air"],
                phase="gas",
                molar_equiv=2.0,
                molar_equiv_basis="chemist_confirmed_intake_inlet_stp",
                gas_reagent_mole_fraction=0.21,
            )
        ],
    )

    assert _extract_gas_equiv(plan, batch_record=batch) == 2.0


def test_exposed_to_air_protocol_triggers_stp_gas_bookkeeping():
    batch = BatchRecord(
        reaction_description="Methionine oxidation to methionine sulfoxide.",
        raw_text="The methionine solution was exposed to air and stirred for 2 h.",
        solvent="water",
        temperature_C=25,
        reaction_time_h=2,
        concentration_M=0.1,
    )
    plan = ChemistryPlan(
        reaction_class="aerobic oxidation",
        mechanism_type="oxidation",
    )
    proposal = FlowProposal(
        residence_time_min=20,
        residence_time_basis="inlet/STP apparent residence time",
        flow_rate_mL_min=0.1,
        concentration_M=0.1,
        temperature_C=25,
        BPR_bar=5,
        tubing_ID_mm=1.0,
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="substrate solution",
                contents=["methionine"],
                solvent="water",
                concentration_M=0.1,
                flow_rate_mL_min=0.1,
            )
        ],
    )

    calc = DesignCalculator().run(batch, chemistry_plan=plan, proposal=proposal)
    annotated = DesignCalculator.annotate_proposal_with_calculations(proposal, calc)

    assert calc.is_gas_liquid
    assert calc.gas_species == "air"
    assert calc.target_gas_equiv_inlet > 0
    assert calc.gas_equiv_supplied > 0
    gas = next(stream for stream in annotated.streams if stream.phase == "gas")
    assert gas.contents == ["air"]
    assert gas.gas_flow_sccm > 0
    assert gas.gas_flow_actual_mL_min > 0


def test_hydrogen_reagent_has_nonzero_generic_gas_equivalents():
    batch = BatchRecord(
        reaction_description="Hydrogenolysis under H2 at 5 bar for 4 h.",
        solvent="EtOH",
        temperature_C=30,
        reaction_time_h=4,
        concentration_M=0.2,
        atmosphere="H2",
    )
    plan = ChemistryPlan(
        reaction_class="hydrogenolysis",
        mechanism_type="gas-liquid hydrogenation",
        stream_logic=[
            StreamLogic(
                stream_label="G",
                reagents=["H2"],
                phase="gas",
                molar_equiv=1.5,
            )
        ],
    )
    proposal = FlowProposal(
        residence_time_min=15,
        residence_time_basis="inlet/STP apparent residence time",
        flow_rate_mL_min=0.1,
        concentration_M=0.2,
        temperature_C=30,
        BPR_bar=5,
        tubing_ID_mm=1.0,
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="substrate solution",
                solvent="EtOH",
                concentration_M=0.2,
                flow_rate_mL_min=0.1,
            ),
            StreamAssignment(
                stream_label="G",
                pump_role="H2 gas feed",
                contents=["H2"],
                phase="gas",
            ),
        ],
    )

    calc = DesignCalculator().run(batch, chemistry_plan=plan, proposal=proposal)

    assert calc.is_gas_liquid
    assert calc.gas_species == "H2"
    assert abs(calc.target_gas_equiv_inlet - 1.5) < 0.02
    assert abs(calc.gas_equiv_supplied - 1.5) < 0.02
    assert calc.gas_flow_sccm > 0
    assert calc.gas_flow_actual_mL_min > 0


def test_under_air_is_not_a_gas_feed_for_nonoxidative_snar():
    batch = BatchRecord(
        reaction_description="4-Fluoronitrobenzene reacts with piperazine.",
        raw_text=(
            "The mixture was heated at 120 C under air for 6 h and then "
            "worked up."
        ),
        solvent="DMF",
        temperature_C=120,
        reaction_time_h=6,
        concentration_M=0.3,
        atmosphere="air",
    )
    plan = ChemistryPlan(
        reaction_class="Nucleophilic Aromatic Substitution (SNAr)",
        mechanism_type="nucleophilic substitution",
        stream_logic=[
            StreamLogic(
                stream_label="A",
                reagents=["4-fluoronitrobenzene"],
                phase="liquid",
            ),
            StreamLogic(
                stream_label="B",
                reagents=["piperazine"],
                phase="liquid",
            ),
        ],
    )
    proposal = FlowProposal(
        residence_time_min=30,
        flow_rate_mL_min=0.1,
        concentration_M=0.3,
        temperature_C=120,
        BPR_bar=5,
        tubing_ID_mm=1.0,
    )

    calc = DesignCalculator().run(batch, chemistry_plan=plan, proposal=proposal)

    assert not calc.is_gas_liquid
    assert calc.gas_flow_sccm == 0
    assert calc.target_gas_equiv_inlet == 0


def test_n2_blanket_does_not_trigger_multiphase_and_quench_is_not_reactor():
    batch = BatchRecord(
        reaction_description="Alpha bromination under N2 followed by thiosulfate quench. Aldehyde is O2-sensitive.",
        solvent="MeCN",
        temperature_C=25,
        reaction_time_h=0.5,
        concentration_M=0.667,
        scale_mmol=10,
        atmosphere="N2",
    )
    plan = ChemistryPlan(
        reaction_class="electrophilic alpha bromination",
        mechanism_type="thermal",
        quench_required=True,
        quench_reagent="Na2S2O3",
    )
    proposal = FlowProposal(
        residence_time_min=5,
        flow_rate_mL_min=1.0,
        concentration_M=0.667,
        temperature_C=25,
        tubing_ID_mm=1.0,
        streams=[
            StreamAssignment(stream_label="A", pump_role="aldehyde in MeCN", solvent="MeCN", concentration_M=0.667, flow_rate_mL_min=0.5),
            StreamAssignment(stream_label="B", pump_role="Br2 in MeCN", solvent="MeCN", concentration_M=0.667, flow_rate_mL_min=0.5),
            StreamAssignment(stream_label="Q", pump_role="inline quench", contents=["Na2S2O3"], solvent="water", flow_rate_mL_min=0.1),
        ],
    )

    calc = DesignCalculator().run(batch, chemistry_plan=plan, proposal=proposal)
    topology = _build_singlestep_topology(proposal, plan, batch)

    assert not calc.is_gas_liquid
    assert not any(op.op_type == "mfc" for op in topology.unit_operations)
    reaction_reactors = [
        op for op in topology.unit_operations
        if op.op_type in {"coil_reactor", "photoreactor", "chip_reactor"}
    ]
    assert len(reaction_reactors) == 1
    assert any(op.op_id == "quench_mixer" for op in topology.unit_operations)


def test_multistage_packed_bed_is_reactor_not_filter():
    batch = BatchRecord(
        reaction_description="Two-stage liquid reaction followed by H2 packed-bed hydrogenation.",
        solvent="MeCN",
        temperature_C=25,
        reaction_time_h=2,
        concentration_M=0.5,
        scale_mmol=5,
    )
    plan = ChemistryPlan(
        reaction_class="cascade hydrogenation",
        mechanism_type="gas-liquid hydrogenation",
        stages=[
            ProcessStage(
                stage_number=1,
                stage_name="Liquid activation",
                reactor_type="coil",
                feed_streams=[
                    StreamLogic(stream_label="A", reagents=["substrate"], concentration_M=0.5),
                    StreamLogic(stream_label="B", reagents=["activator"], concentration_M=0.5),
                ],
            ),
            ProcessStage(
                stage_number=2,
                stage_name="Hydrogenation",
                reactor_type="packed_bed",
                feed_streams=[
                    StreamLogic(stream_label="D", reagents=["H2 gas"], phase="gas"),
                ],
            ),
        ],
    )
    proposal = FlowProposal(
        residence_time_min=20,
        flow_rate_mL_min=0.5,
        concentration_M=0.5,
        temperature_C=25,
        BPR_bar=7.0,
        tubing_ID_mm=0.75,
        streams=[
            StreamAssignment(stream_label="A", pump_role="substrate", concentration_M=0.5),
            StreamAssignment(stream_label="B", pump_role="activator", concentration_M=0.5),
            StreamAssignment(stream_label="D", pump_role="H2 gas", contents=["H2 gas"], phase="gas"),
        ],
        multiphase_metrics={
            "gas_holdup": 0.5,
            "gas_flow_sccm": 10.0,
            "gas_flow_actual_mL_min": 2.0,
        },
    )

    topology = _build_multistep_topology(proposal, plan, batch)

    assert any(op.op_type == "packed_bed_reactor" for op in topology.unit_operations)
    assert not any(
        op.op_type == "inline_filter" and "hydrogenation" in op.label.lower()
        for op in topology.unit_operations
    )


def test_gas_stream_with_none_solvent_stays_single_mfc():
    batch = BatchRecord(
        reaction_description="Activated alkene chlorination with Cl2 gas feed followed by thiosulfate quench.",
        solvent="DCM",
        temperature_C=-20,
        reaction_time_h=0.5,
        concentration_M=0.2,
        scale_mmol=5,
    )
    plan = ChemistryPlan(
        reaction_class="chlorination",
        mechanism_type="gas-liquid electrophilic addition",
    )
    proposal = FlowProposal(
        residence_time_min=3,
        flow_rate_mL_min=1.0,
        concentration_M=0.2,
        temperature_C=-20,
        tubing_ID_mm=0.5,
        BPR_bar=7.0,
        streams=[
            StreamAssignment(stream_label="A", pump_role="substrate pump", contents=["alkene"], solvent="DCM", flow_rate_mL_min=1.0),
            StreamAssignment(stream_label="G", pump_role="gas mass flow controller", contents=["Cl2 gas"], solvent="none", flow_rate_mL_min=0.05),
        ],
    )

    calc = DesignCalculator().run(batch, chemistry_plan=plan, proposal=proposal)
    proposal = DesignCalculator.annotate_proposal_with_calculations(proposal, calc)
    topology = _build_singlestep_topology(proposal, plan, batch)

    assert calc.is_gas_liquid
    assert calc.gas_species == "Cl2"
    gas_ops = [op for op in topology.unit_operations if op.parameters.get("stream") == "G"]
    assert len(gas_ops) == 1
    assert gas_ops[0].op_type == "mfc"


def test_singlestep_packed_bed_topology_preserves_reactor_type():
    proposal = FlowProposal(
        residence_time_min=4,
        flow_rate_mL_min=1.0,
        concentration_M=0.2,
        temperature_C=25,
        tubing_ID_mm=0.75,
        reactor_type="packed_bed",
        tubing_material="stainless steel",
        streams=[
            StreamAssignment(stream_label="A", pump_role="substrate", contents=["nitroarene"], solvent="EtOH", flow_rate_mL_min=1.0),
            StreamAssignment(stream_label="B", pump_role="H2 gas feed", contents=["H2 gas"], phase="gas", flow_rate_mL_min=0.1),
        ],
    )

    topology = _build_singlestep_topology(proposal, ChemistryPlan(), BatchRecord())

    assert any(op.op_type == "packed_bed_reactor" for op in topology.unit_operations)


def test_gas_liquid_hard_filter_rejects_high_pressure_microtube():
    metrics = {
        "L_m": 10.0,
        "V_R_mL": 5.0,
        "Re": 50.0,
        "delta_P_bar": 120.0,
        "Q_mL_min": 1.0,
        "tau_min": 5.0,
        "d_mm": 0.28,
        "Da_mass": 0.1,
        "r_mix": 0.01,
        "expected_conversion": 0.9,
    }

    feasible, violations, _ = hard_filter(
        metrics,
        is_photochem=False,
        is_gas_liquid=True,
        pump_max_bar=200.0,
        BPR_bar=120.0,
    )

    assert not feasible
    assert any("gas-liquid ΔP" in v for v in violations)
    assert any("gas-liquid slug-flow practical floor" in v for v in violations)


def test_quench_stream_mentioning_gas_reagent_is_not_mfc():
    proposal = FlowProposal(
        residence_time_min=5,
        flow_rate_mL_min=1.0,
        concentration_M=0.2,
        temperature_C=25,
        tubing_ID_mm=0.75,
        streams=[
            StreamAssignment(stream_label="A", pump_role="substrate feed", contents=["alkene"], solvent="DCM", flow_rate_mL_min=1.0),
            StreamAssignment(stream_label="G", pump_role="Cl2 gas feed", contents=["Cl2"], phase="gas", flow_rate_mL_min=0.1),
            StreamAssignment(stream_label="Q", pump_role="quench stream for excess Cl2 destruction", contents=["Na2S2O3 aq"], flow_rate_mL_min=0.2),
        ],
        multiphase_metrics={"gas_flow_sccm": 10.0, "gas_flow_actual_mL_min": 0.1},
    )
    plan = ChemistryPlan(quench_required=True)

    topology = _build_singlestep_topology(proposal, plan, BatchRecord())
    quench = next(op for op in topology.unit_operations if op.parameters.get("stream") == "Q")

    assert quench.op_type == "pump"


def test_gas_stream_allows_null_solvent_from_llm_json():
    proposal = FlowProposal(
        residence_time_min=30,
        flow_rate_mL_min=0.1,
        streams=[
            {"stream_label": "A", "pump_role": "substrate in DMSO", "solvent": "DMSO"},
            {
                "stream_label": "G",
                "pump_role": "O2 gas feed",
                "contents": "O2",
                "phase": "gas",
                "solvent": None,
            },
        ],
    )

    assert proposal.streams[1].solvent == ""
    assert proposal.streams[1].contents == ["O2"]
    assert proposal.streams[1].phase == "gas"


def test_gas_liquid_candidate_metrics_include_holdup_and_high_pressure_warning():
    metrics = compute_metrics(
        tau_min=20.7,
        d_mm=0.5,
        Q_mL_min=0.1518,
        solvent="EtOH",
        temperature_C=25,
        concentration_M=0.1,
        assumed_MW=300.0,
        IF_used=6.0,
        tau_kinetics_min=20.0,
        pump_max_bar=400.0,
        is_photochem=True,
        is_gas_liquid=True,
        BPR_bar=7.0,
    )

    assert metrics["V_R_mL"] > metrics["liquid_holdup_volume_mL"]
    assert metrics["two_phase_multiplier"] > 1.0
    assert metrics["required_bpr_bar"] > 10.0
    feasible, violations, warnings = hard_filter(
        metrics,
        is_photochem=True,
        is_gas_liquid=True,
        pump_max_bar=400.0,
        BPR_bar=7.0,
    )
    assert not feasible
    assert not any("gas-service pressure ceiling" in violation for violation in violations)
    assert any("routine operating range" in warning for warning in warnings)


def test_gas_liquid_candidate_rejects_pressure_above_available_ceiling():
    metrics = compute_metrics(
        tau_min=10.0,
        d_mm=1.0,
        Q_mL_min=0.1,
        solvent="EtOH",
        temperature_C=25,
        concentration_M=0.1,
        assumed_MW=300.0,
        IF_used=6.0,
        tau_kinetics_min=20.0,
        pump_max_bar=100.0,
        is_photochem=False,
        is_gas_liquid=True,
        BPR_bar=60.0,
        target_gas_equiv_inlet=1.0,
        gas_reagent_fraction=1.0,
    )

    feasible, violations, _ = hard_filter(
        metrics,
        is_photochem=False,
        is_gas_liquid=True,
        pump_max_bar=100.0,
        BPR_bar=60.0,
        gas_liquid_max_bpr_bar=50.0,
    )

    assert not feasible
    assert any("available gas-service pressure ceiling" in v for v in violations)


def test_pure_hydrogen_basis_uses_less_gas_holdup_than_air_basis():
    common = dict(
        tau_min=10.0,
        d_mm=1.0,
        Q_mL_min=0.1,
        solvent="EtOH",
        temperature_C=25,
        concentration_M=0.1,
        assumed_MW=300.0,
        IF_used=6.0,
        tau_kinetics_min=20.0,
        pump_max_bar=50.0,
        is_photochem=False,
        is_gas_liquid=True,
        BPR_bar=10.0,
    )

    hydrogen = compute_metrics(
        **common,
        target_gas_equiv_inlet=1.0,
        gas_reagent_fraction=1.0,
    )
    air = compute_metrics(
        **common,
        target_gas_equiv_inlet=1.0,
        gas_reagent_fraction=0.21,
    )

    assert hydrogen["gas_holdup"] < air["gas_holdup"]
    assert hydrogen["gas_flow_actual_mL_min"] < air["gas_flow_actual_mL_min"]


def test_candidate_rejects_flow_above_selected_pump_maximum():
    metrics = compute_metrics(
        tau_min=2.0,
        d_mm=1.0,
        Q_mL_min=11.2,
        solvent="EtOH",
        temperature_C=25,
        concentration_M=0.1,
        assumed_MW=300.0,
        IF_used=6.0,
        tau_kinetics_min=20.0,
        pump_max_bar=50.0,
        is_photochem=False,
    )

    feasible, violations, _ = hard_filter(
        metrics,
        is_photochem=False,
        is_gas_liquid=False,
        pump_max_bar=50.0,
        max_flow_rate_mL_min=5.0,
    )

    assert not feasible
    assert any("selected pump maximum" in violation for violation in violations)


def test_candidate_diagnostics_distinguish_pump_maximum_from_floor():
    _, infeasible = generate_candidates(
        tau_center_min=2.0,
        tau_lit_min=None,
        solvent="EtOH",
        temperature_C=25,
        concentration_M=0.1,
        assumed_MW=300.0,
        IF_used=1.0,
        tau_kinetics_min=2.0,
        pump_max_bar=50.0,
        is_photochem=False,
        is_gas_liquid=False,
        min_flow_rate_mL_min=0.000001,
        max_flow_rate_mL_min=0.000001,
    )

    assert infeasible
    assert any(
        "Q_ceiling" in candidate.get("primary_kill_categories", [])
        for candidate in infeasible
    )


def test_final_bpr_reconciliation_does_not_promote_stale_calculator_bpr():
    result = {
        "proposal": {"BPR_bar": 7.0},
        "design_calculations": {
            "is_gas_liquid": True,
            "bpr_pressure_bar": 23.4,
        },
    }

    _reconcile_final_bpr(result)

    assert result["proposal"]["BPR_bar"] == 7.0
    assert result["design_calculations"]["bpr_pressure_bar"] == 7.0
    assert "validation warning" in result["design_calculations"]["bpr_reconciliation_note"]


def test_final_bpr_reconciliation_uses_declared_inventory_setpoint():
    result = {
        "proposal": {"BPR_bar": 2.5},
        "design_calculations": {
            "is_gas_liquid": True,
            "bpr_pressure_bar": 3.0,
        },
    }
    inventory = LabInventory(
        BPR_available=[1.5, 2.5, 7.0, 16.0],
        pressure_controllers=[
            PressureControllerSpec(
                equipment_id="bpr_7",
                name="7 bar BPR",
                setpoints_bar=[7.0],
            )
        ],
    )

    _reconcile_final_bpr(result, inventory=inventory)

    assert result["proposal"]["BPR_bar"] == 7.0
    assert result["design_calculations"]["bpr_pressure_bar"] == 7.0


def test_gas_liquid_candidate_generation_uses_total_tube_volume_basis():
    feasible, infeasible = generate_candidates(
        tau_center_min=8.3,
        tau_lit_min=None,
        solvent="EtOH",
        temperature_C=25,
        concentration_M=0.1,
        assumed_MW=300.0,
        IF_used=6.0,
        tau_kinetics_min=16.67,
        pump_max_bar=400.0,
        is_photochem=True,
        is_gas_liquid=True,
        BPR_bar=7.0,
        tau_low_factor=0.3,
        tau_high_factor=2.0,
        n_tau=5,
        d_exclude_above_mm=1.0,
        L_fractions=[0.4, 0.6, 0.8, 0.95],
        N_target=12,
    )

    assert feasible
    assert all(c["L_m"] <= L_MAX_BENCH_M for c in feasible)
    assert all(c["required_bpr_bar"] <= 10.0 for c in feasible)
    assert all(c["violations"] for c in infeasible)


def test_candidate_generation_honors_measured_residence_time_floor():
    feasible, _ = generate_candidates(
        tau_center_min=85.4,
        tau_lit_min=None,
        solvent="DMSO",
        temperature_C=40,
        concentration_M=0.5,
        assumed_MW=143.19,
        IF_used=6.0,
        tau_kinetics_min=85.4,
        pump_max_bar=8.0,
        is_photochem=True,
        is_gas_liquid=True,
        BPR_bar=3.0,
        tau_low_factor=0.3,
        tau_high_factor=2.0,
        n_tau=5,
        d_exclude_above_mm=1.0,
        L_fractions=[0.4, 0.6, 0.8],
        N_target=12,
        min_tau_min=68.9,
        min_flow_rate_mL_min=0.01,
    )

    assert feasible
    assert all(candidate["tau_min"] >= 68.9 for candidate in feasible)


def test_chemistry_plan_normalization_drops_null_mandate_values():
    normalized = _normalize_plan_data({
        "intensification_mandate": {
            "tau_reduction_target": None,
            "minimum_flow_advantage": None,
        },
    })

    plan = ChemistryPlan(**normalized)

    assert plan.intensification_mandate.tau_reduction_target == 2.0
    assert plan.intensification_mandate.minimum_flow_advantage == "productivity"


def test_chemistry_plan_normalization_repairs_nested_stream_numbers():
    normalized = _normalize_plan_data({
        "incompatible_pairs": ["TBHP + sulfuric acid"],
        "stream_logic": [
            {
                "stream_label": "B",
                "reagents": "heterogeneous catalyst",
                "molar_equiv": "heterogeneous",
                "concentration_M": "0.25",
            }
        ],
        "stages": [
            {
                "stage_id": 1,
                "feed_streams": [
                    {
                        "stream_label": "A",
                        "reagents": [{"name": "substrate"}],
                        "molar_equiv": "1.0",
                        "gas_flow_sccm": "not applicable",
                    }
                ],
            }
        ],
    })

    assert normalized["stream_logic"][0]["reagents"] == ["heterogeneous catalyst"]
    assert normalized["incompatible_pairs"] == [["TBHP", "sulfuric acid"]]
    assert normalized["stream_logic"][0]["molar_equiv"] == 1.0
    assert normalized["stream_logic"][0]["concentration_M"] == 0.25
    assert normalized["stages"][0]["feed_streams"][0]["reagents"] == ["substrate"]
    assert normalized["stages"][0]["feed_streams"][0]["gas_flow_sccm"] is None


def test_chemistry_plan_normalization_repairs_null_stream_equivalents():
    normalized = _normalize_plan_data({
        "stream_logic": [
            {
                "stream_label": "B",
                "reagents": ["oxidant"],
                "molar_equiv": None,
            }
        ],
        "stages": [
            {
                "stage_id": 1,
                "feed_streams": [
                    {
                        "stream_label": "C",
                        "reagents": ["amine"],
                        "molar_equiv": None,
                    }
                ],
            }
        ],
    })

    assert normalized["stream_logic"][0]["molar_equiv"] == 1.0
    assert normalized["stages"][0]["feed_streams"][0]["molar_equiv"] == 1.0
    ChemistryPlan(**normalized)


def test_skeptic_volume_audit_uses_liquid_holdup_for_gas_liquid():
    errors = _verify_v_r_equals_tau_q([
        {
            "id": 1,
            "tau_min": 25.0,
            "Q_mL_min": 0.1272,
            "liquid_holdup_volume_mL": 3.18,
            "V_R_mL": 21.2,
            "gas_holdup": 0.85,
        }
    ])

    assert errors == []


def test_final_stream_flowrates_sync_to_authoritative_liquid_q():
    proposal = FlowProposal(
        residence_time_min=93.8,
        flow_rate_mL_min=0.13397,
        streams=[
            StreamAssignment(
                stream_label="A",
                pump_role="substrate solution feed",
                contents=["substrate"],
                solvent="DMSO",
                concentration_M=0.5,
                flow_rate_mL_min=0.433,
            ),
            StreamAssignment(
                stream_label="B",
                pump_role="oxidant gas feed",
                contents=["O2"],
                phase="gas",
                flow_rate_mL_min=0.0793,
            ),
        ],
    )
    holder = type("Holder", (), {"proposal": proposal})()
    result = {
        "proposal": proposal.model_dump(),
        "design_calculations": {
            "is_gas_liquid": True,
            "liquid_flow_rate_mL_min": 0.13397,
            "gas_flow_sccm": 4.504,
            "gas_flow_actual_mL_min": 0.0793,
        },
    }

    _sync_final_stream_flowrates(result, holder)

    streams = result["proposal"]["streams"]
    assert streams[0]["flow_rate_mL_min"] == 0.13397
    assert streams[1]["flow_rate_mL_min"] == 0.0793
    assert streams[1]["gas_flow_sccm"] == 4.504
    assert holder.proposal.streams[0].flow_rate_mL_min == 0.13397


def test_explicit_o2_mfc_stream_overrides_wrong_liquid_phase_tag():
    stream = StreamAssignment(
        stream_label="B",
        pump_role="O2 gas feed via MFC",
        contents=["O2 gas"],
        solvent="N/A (gas stream)",
        phase="liquid",
    )

    assert DesignCalculator._stream_assignment_is_gas(stream)


def test_marginal_intensification_conflict_proceeds_as_screen_required():
    diagnostic = _intensification_feasibility_precheck(
        batch_time_min=900.0,
        tau_kinetics_min=187.5,
        intensification_mandate={"tau_reduction_target": 5.0},
        translation_policy="intensify",
    )

    assert diagnostic is not None
    assert diagnostic["status"] == "KINETIC_ANCHOR_UNCERTAIN_SCREEN_REQUIRED"
    assert diagnostic["hard_block"] is False
    assert diagnostic["required_to_ceiling_ratio"] == 1.042


def test_candidate_under_ceiling_prevents_pre_council_hard_block():
    diagnostic = _intensification_feasibility_precheck(
        batch_time_min=900.0,
        tau_kinetics_min=187.5,
        intensification_mandate={"tau_reduction_target": 5.0},
        translation_policy="intensify",
        candidate_tau_min=93.8,
    )

    assert diagnostic is not None
    assert diagnostic["status"] == "KINETIC_ANCHOR_UNCERTAIN_SCREEN_REQUIRED"
    assert diagnostic["hard_block"] is False
    assert diagnostic["candidate_tau_min"] == 93.8
    assert diagnostic["candidate_projected_conversion"] >= 0.5


def test_large_intensification_conflict_still_blocks_before_council():
    diagnostic = _intensification_feasibility_precheck(
        batch_time_min=900.0,
        tau_kinetics_min=300.0,
        intensification_mandate={"tau_reduction_target": 5.0},
        translation_policy="intensify",
    )

    assert diagnostic is not None
    assert diagnostic["status"] == "INFEASIBLE_WITH_CURRENT_KINETIC_ANCHOR"
    assert diagnostic["hard_block"] is True


def test_bad_low_conversion_candidate_does_not_bypass_hard_block():
    diagnostic = _intensification_feasibility_precheck(
        batch_time_min=900.0,
        tau_kinetics_min=300.0,
        intensification_mandate={"tau_reduction_target": 5.0},
        translation_policy="intensify",
        candidate_tau_min=40.0,
    )

    assert diagnostic is not None
    assert diagnostic["status"] == "INFEASIBLE_WITH_CURRENT_KINETIC_ANCHOR"
    assert diagnostic["hard_block"] is True
    assert diagnostic["candidate_projected_conversion"] < 0.5
