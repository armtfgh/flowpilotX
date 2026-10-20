from flora_translate.experiment_loop import (
    ActualConditions,
    ExperimentalOutcomes,
    ExperimentResult,
    calibrate_experimental_campaign,
    extract_experiments_from_text,
    refine_from_experimental_campaign,
    refine_from_experiment,
)


def _base_result():
    return {
        "design_version": 1,
        "confidence": "MEDIUM",
        "explanation": "Initial FLORA design.",
        "proposal": {
            "residence_time_min": 10.0,
            "flow_rate_mL_min": 0.2,
            "reactor_volume_mL": 2.0,
            "temperature_C": 40.0,
            "concentration_M": 0.1,
            "BPR_bar": 5.0,
            "reactor_type": "photoreactor coil",
            "tubing_material": "FEP",
            "tubing_ID_mm": 0.8,
            "pre_reactor_steps": ["degas stream A with N2"],
            "post_reactor_steps": [],
            "chemistry_notes": "Photoredox reaction.",
            "safety_flags": [],
            "streams": [
                {
                    "stream_label": "A",
                    "pump_role": "substrate + photocatalyst",
                    "contents": ["substrate", "photocatalyst"],
                    "solvent": "MeCN",
                    "concentration_M": 0.1,
                    "flow_rate_mL_min": 0.1,
                    "phase": "liquid",
                    "reasoning": "Main reaction stream.",
                },
                {
                    "stream_label": "B",
                    "pump_role": "base",
                    "contents": ["base"],
                    "solvent": "MeCN",
                    "concentration_M": 0.1,
                    "flow_rate_mL_min": 0.1,
                    "phase": "liquid",
                    "reasoning": "Separate base stream.",
                },
            ],
            "reasoning_per_field": {},
        },
    }


def _thq_base_result():
    result = _base_result()
    proposal = result["proposal"]
    proposal.update(
        {
            "residence_time_min": 34.5,
            "flow_rate_mL_min": 0.0656,
            "reactor_volume_mL": 13.27,
            "temperature_C": 40.0,
            "concentration_M": 0.5,
            "BPR_bar": 6.0,
            "tubing_ID_mm": 1.0,
            "streams": [
                {
                    "stream_label": "A",
                    "pump_role": "substrate in DMSO",
                    "contents": ["6-methyl-1,2,3,4-tetrahydroquinoline"],
                    "solvent": "DMSO",
                    "concentration_M": 0.5,
                    "flow_rate_mL_min": 0.0656,
                    "phase": "liquid",
                    "reasoning": "Starting short screen.",
                },
                {
                    "stream_label": "B",
                    "pump_role": "O2",
                    "contents": ["O2"],
                    "solvent": "",
                    "concentration_M": None,
                    "flow_rate_mL_min": 0.32,
                    "phase": "gas",
                    "gas_flow_actual_mL_min": 0.32,
                    "gas_flow_sccm": 1.6565,
                    "reasoning": "Starting short screen.",
                },
            ],
        }
    )
    return result


def _thq_experiments():
    rows = [
        ("khu_manual", 40.0, 0.25, 3.0, 0.014, 0.059, 0.1530, 1.95, 59.91, 137.08, 10.00, None, 52.0, 1.016),
        ("krict_2", 40.0, 0.50, 6.0, 0.0509, 0.234, 1.2113, 2.12, 8.51, 37.68, 10.74, 83.0, 12.0, 0.75),
        ("krict_6", 40.0, 0.50, 6.0, 0.0656, 0.320, 1.6565, 2.25, 7.71, 34.42, 13.27, 85.0, 10.0, 1.00),
    ]
    experiments = []
    for idx, row in enumerate(rows, 1):
        (
            run_id,
            temperature,
            concentration,
            pressure,
            substrate_q,
            gas_channel,
            gas_stp,
            gas_equiv,
            t_inlet,
            t_channel,
            volume,
            sm_pct,
            product_pct,
            tubing_id,
        ) = row
        experiments.append(
            ExperimentResult(
                run_id=run_id,
                design_version=1,
                actual_conditions=ActualConditions(
                    residence_time_inlet_min=t_inlet,
                    residence_time_in_channel_min=t_channel,
                    residence_time_basis="in_channel",
                    flow_rate_mL_min=substrate_q,
                    substrate_flow_mL_min=substrate_q,
                    gas_flow_in_channel_mL_min=gas_channel,
                    gas_flow_stp_mL_min=gas_stp,
                    gas_equiv_inlet=gas_equiv,
                    temperature_C=temperature,
                    concentration_M=concentration,
                    tubing_ID_mm=tubing_id,
                    reactor_volume_mL=volume,
                    BPR_bar=pressure,
                ),
                outcomes=ExperimentalOutcomes(
                    product_pct=product_pct,
                    yield_pct=product_pct,
                    starting_material_pct=sm_pct,
                ),
            )
        )
    return experiments


def _krict_inlet_basis_experiments():
    return [
        ExperimentResult(
            run_id="krict_2",
            design_version=1,
            actual_conditions=ActualConditions(
                residence_time_inlet_min=8.51,
                residence_time_in_channel_min=37.68,
                residence_time_basis="inlet_stp",
                flow_rate_mL_min=0.0509,
                substrate_flow_mL_min=0.0509,
                gas_flow_in_channel_mL_min=0.234,
                gas_flow_stp_mL_min=1.2113,
                gas_equiv_inlet=2.12,
                temperature_C=40.0,
                concentration_M=0.5,
                tubing_ID_mm=0.75,
                reactor_volume_mL=10.74,
                BPR_bar=6.0,
            ),
            outcomes=ExperimentalOutcomes(product_pct=12.0, yield_pct=12.0, starting_material_pct=83.0),
        ),
        ExperimentResult(
            run_id="krict_6",
            design_version=1,
            actual_conditions=ActualConditions(
                residence_time_inlet_min=7.71,
                residence_time_in_channel_min=34.42,
                residence_time_basis="inlet_stp",
                flow_rate_mL_min=0.0656,
                substrate_flow_mL_min=0.0656,
                gas_flow_in_channel_mL_min=0.320,
                gas_flow_stp_mL_min=1.6565,
                gas_equiv_inlet=2.25,
                temperature_C=40.0,
                concentration_M=0.5,
                tubing_ID_mm=1.0,
                reactor_volume_mL=13.27,
                BPR_bar=6.0,
            ),
            outcomes=ExperimentalOutcomes(product_pct=10.0, yield_pct=10.0, starting_material_pct=85.0),
        ),
    ]


def test_low_conversion_increases_residence_time_and_preserves_volume():
    experiment = ExperimentResult(
        run_id="run_01",
        design_version=1,
        actual_conditions=ActualConditions(
            residence_time_min=10.0,
            flow_rate_mL_min=0.2,
            reactor_volume_mL=2.0,
            temperature_C=40.0,
            concentration_M=0.1,
        ),
        outcomes=ExperimentalOutcomes(
            yield_pct=28.0,
            conversion_pct=45.0,
            selectivity_pct=92.0,
        ),
    )

    closed_loop = refine_from_experiment(_base_result(), experiment)

    assert closed_loop.decision.failure_modes == ["kinetic_underconversion"]
    proposal = closed_loop.refined_result["proposal"]
    assert proposal["residence_time_min"] == 38.515
    assert proposal["reactor_volume_mL"] == 2.0
    assert proposal["flow_rate_mL_min"] == 0.05193
    assert closed_loop.refined_result["design_version"] == 2


def test_precipitation_and_pressure_drift_add_screening_controls():
    experiment = ExperimentResult(
        run_id="run_01",
        design_version=1,
        actual_conditions=ActualConditions(
            residence_time_min=10.0,
            flow_rate_mL_min=0.2,
            reactor_volume_mL=2.0,
            temperature_C=40.0,
            concentration_M=0.1,
            BPR_bar=5.0,
        ),
        outcomes=ExperimentalOutcomes(
            yield_pct=42.0,
            conversion_pct=88.0,
            selectivity_pct=62.0,
            pressure_bar=9.0,
            pressure_drift_bar=4.0,
            precipitation_observed=True,
        ),
    )

    closed_loop = refine_from_experiment(_base_result(), experiment)
    proposal = closed_loop.refined_result["proposal"]

    assert "selectivity_loss" in closed_loop.decision.failure_modes
    assert "solubility_or_fouling" in closed_loop.decision.failure_modes
    assert "pressure_instability" in closed_loop.decision.failure_modes
    assert proposal["temperature_C"] == 30.0
    assert proposal["concentration_M"] == 0.07
    assert proposal["BPR_bar"] == 7.0
    assert "inline 2 um filter before reactor" in proposal["pre_reactor_steps"]
    assert closed_loop.decision.status == "screen_required"
    assert any("pressure instability" in flag for flag in proposal["safety_flags"])


def test_three_cycle_case_reaches_converged_design_version():
    result = _base_result()
    cycles = [
        ExperimentalOutcomes(yield_pct=30.0, conversion_pct=48.0, selectivity_pct=90.0),
        ExperimentalOutcomes(yield_pct=58.0, conversion_pct=90.0, selectivity_pct=74.0),
        ExperimentalOutcomes(yield_pct=84.0, conversion_pct=94.0, selectivity_pct=91.0),
    ]
    last = None
    for idx, outcomes in enumerate(cycles, 1):
        proposal = result["proposal"]
        experiment = ExperimentResult(
            run_id=f"run_{idx:02d}",
            design_version=result["design_version"],
            actual_conditions=ActualConditions(
                residence_time_min=proposal["residence_time_min"],
                flow_rate_mL_min=proposal["flow_rate_mL_min"],
                reactor_volume_mL=proposal["reactor_volume_mL"],
                temperature_C=proposal["temperature_C"],
                concentration_M=proposal["concentration_M"],
                BPR_bar=proposal["BPR_bar"],
            ),
            outcomes=outcomes,
        )
        last = refine_from_experiment(result, experiment)
        result = last.refined_result

    assert last is not None
    assert last.decision.status == "converged"
    assert result["design_version"] == 4
    assert result["proposal"]["confidence"] == "MEDIUM"


def test_campaign_calibration_builds_thq_residence_time_ladder():
    calibration = calibrate_experimental_campaign(
        _thq_experiments(),
        target_yield_pct=75.0,
        target_conversion_pct=75.0,
    )

    assert calibration.n_usable == 3
    assert calibration.best_run_id == "khu_manual"
    assert calibration.primary_residence_time_basis == "inlet/STP apparent residence time"
    assert calibration.best_tau_inlet_min == 59.91
    assert calibration.best_tau_in_channel_min == 137.08
    assert 89.0 < calibration.recommended_tau_inlet_min < 91.0
    assert 205.0 < calibration.recommended_tau_in_channel_min < 207.0
    assert 112.0 < calibration.target_tau_inlet_min < 114.0
    assert 258.0 < calibration.target_tau_in_channel_min < 260.5
    assert calibration.design_ladder[0]["label"] == "best_observed_anchor"
    assert calibration.design_ladder[-1]["label"] == "target_estimate"


def test_campaign_refinement_overrides_short_intensification_design():
    closed_loop = refine_from_experimental_campaign(
        _thq_base_result(),
        _thq_experiments(),
        target_yield_pct=75.0,
        target_conversion_pct=75.0,
    )

    proposal = closed_loop.refined_result["proposal"]
    next_exp = closed_loop.decision.next_experiment
    assert 89.0 < proposal["residence_time_min"] < 91.0
    assert proposal["residence_time_min"] == proposal["residence_time_inlet_min"]
    assert proposal["residence_time_in_channel_min"] > 205.0
    assert proposal["residence_time_basis"] == "inlet/STP apparent residence time"
    assert proposal["flow_rate_mL_min"] < 0.014
    assert next_exp["evidence_calibration"]["best_tau_in_channel_min"] == 137.08
    assert next_exp["evidence_calibration"]["target_tau_in_channel_min"] > 258.0
    assert "experiment_calibrated_kinetics" in closed_loop.decision.failure_modes


def test_krict_only_campaign_can_calibrate_on_inlet_stp_basis():
    calibration = calibrate_experimental_campaign(
        _krict_inlet_basis_experiments(),
        target_yield_pct=75.0,
        target_conversion_pct=75.0,
    )

    assert calibration.n_usable == 2
    assert calibration.primary_residence_time_basis == "inlet/STP apparent residence time"
    assert calibration.best_run_id == "krict_2"
    assert calibration.best_tau_min == 8.51
    assert calibration.best_tau_inlet_min == 8.51
    assert calibration.best_tau_in_channel_min == 37.68
    assert 38.0 < calibration.recommended_tau_inlet_min < 39.0
    assert 91.0 < calibration.target_tau_inlet_min < 93.5

    recommended = calibration.recommended_conditions
    assert recommended["residence_time_min"] == recommended["residence_time_inlet_min"]
    assert recommended["residence_time_in_channel_min"] > 160.0
    assert recommended["gas_flow_stp_mL_min"] > recommended["gas_flow_in_channel_mL_min"]
    assert abs(
        recommended["gas_flow_stp_mL_min"] / recommended["substrate_flow_mL_min"]
        - 1.2113 / 0.0509
    ) < 0.05


def test_campaign_refinement_preserves_inlet_basis_for_krict_only_feedback():
    result = _thq_base_result()
    result["proposal"]["residence_time_basis"] = "inlet/STP apparent residence time"
    result["proposal"]["residence_time_min"] = 8.51
    result["proposal"]["residence_time_inlet_min"] = 8.51
    result["proposal"]["residence_time_in_channel_min"] = 37.68

    closed_loop = refine_from_experimental_campaign(
        result,
        _krict_inlet_basis_experiments(),
        target_yield_pct=75.0,
        target_conversion_pct=75.0,
    )

    proposal = closed_loop.refined_result["proposal"]
    assert proposal["residence_time_basis"] == "inlet/STP apparent residence time"
    assert proposal["residence_time_min"] == proposal["residence_time_inlet_min"]
    assert 38.0 < proposal["residence_time_inlet_min"] < 39.0
    assert proposal["residence_time_in_channel_min"] > 160.0
    assert proposal["gas_flow_sccm"] > proposal["gas_flow_actual_mL_min"]
    assert "experiment_calibrated_kinetics" in closed_loop.decision.failure_modes


def test_extract_experiments_from_prompt_text():
    prompt = """
Entry 1, KHU manual:
T = 40 C
c = 0.25 M
P = 3 bar
substrate flow = 0.014 mL/min
O2 in-channel = 0.059 mL/min
O2 inlet/STP = 0.1530 mL/min
O2 equiv inlet = 1.95
t inlet = 59.91 min
t in-channel = 137.08 min
reactor volume = 10.00 mL
product = 52%
tubing ID = 1.016 mm

Entry 2, KRICT 6:
T = 40 C
c = 0.50 M
P = 6 bar
substrate flow = 0.0656 mL/min
O2 in-channel = 0.320 mL/min
O2 inlet/STP = 1.6565 mL/min
O2 equiv inlet = 2.25
t inlet = 7.71 min
t in-channel = 34.42 min
reactor volume = 13.27 mL
starting material = 85%
product = 10%
tubing ID = 1.00 mm
"""
    experiments = extract_experiments_from_text(prompt)

    assert len(experiments) == 2
    assert experiments[0].run_id == "entry_01_khu_manual"
    assert experiments[0].actual_conditions.residence_time_in_channel_min == 137.08
    assert experiments[0].actual_conditions.residence_time_basis == "inlet_stp"
    assert experiments[0].outcomes.yield_pct == 52.0
    assert experiments[1].outcomes.conversion_pct == 15.0


def test_extract_compact_krict_rows_and_calibrate_longer_inlet_screen():
    prompt = """
1- Entry 3: Note KRICT 2, T 40, c 0.50, P 6.0 (5+1), Substrate 0.0509, O₂ In-channel 0.234, O₂ Inlet/STP 1.2113, O₂ equiv 2.12, t inlet 8.51, t in-channel 37.68, Reactor Volume 10.74, Starting Material 83, Product 12, Tubing ID 0.75

2- Entry 4: Note KRICT 6, T 40, c 0.50, P 6.0 (5+1), Substrate 0.0656, O₂ In-channel 0.320, O₂ Inlet/STP 1.6565, O₂ equiv 2.25, t inlet 7.71, t in-channel 34.42, Reactor Volume 13.27, Starting Material 85, Product 10, Tubing ID 1.00

3- Entry 5: Note KRICT 2-1, T 40, c 0.50, P 6.0 (5+1), Substrate 0.01047, O₂ In-channel 0.0489, O₂ Inlet/STP 0.2492, O₂ equiv ~2, t inlet 38.6, t in-channel 168.43, Reactor Volume 10.00, Starting Material -, Product 27, Tubing ID 1.016
"""
    experiments = extract_experiments_from_text(prompt)

    assert len(experiments) == 3
    entry_5 = experiments[-1]
    assert entry_5.run_id == "entry_05_krict_2_1"
    assert entry_5.actual_conditions.substrate_flow_mL_min == 0.01047
    assert entry_5.actual_conditions.gas_flow_stp_mL_min == 0.2492
    assert entry_5.actual_conditions.gas_flow_in_channel_mL_min == 0.0489
    assert entry_5.actual_conditions.residence_time_inlet_min == 38.6
    assert entry_5.actual_conditions.residence_time_in_channel_min == 168.43
    assert entry_5.outcomes.product_pct == 27.0

    calibration = calibrate_experimental_campaign(
        experiments,
        target_yield_pct=75.0,
        target_conversion_pct=75.0,
        residence_time_basis="inlet_stp",
    )

    assert calibration.best_run_id == entry_5.run_id
    assert calibration.best_tau_inlet_min == 38.6
    assert calibration.recommended_tau_inlet_min > calibration.best_tau_inlet_min
    assert 68.0 < calibration.recommended_tau_inlet_min < 69.0
    assert calibration.target_tau_inlet_min > 160.0
    assert calibration.recommended_conditions["residence_time_in_channel_min"] > 290.0
    assert calibration.recommended_conditions["gas_flow_stp_mL_min"] > calibration.recommended_conditions["gas_flow_in_channel_mL_min"]
