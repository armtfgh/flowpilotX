"""Local review artifacts from immutable, pre-outcome campaign nominations.

One authoritative condition list feeds every table and results form. No model
execution, equipment control, laboratory release, or external upload occurs.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import html
import importlib.util
import json
from pathlib import Path
import shutil

from .provenance import digest, file_hash, write_json
from .candidate_space import R_ML_BAR_PER_MMOL_K

STATUS = "conditional_review_not_released_for_experiment"
TIME_NOTE = ("V/Q_L is liquid-only space time, not measured liquid contact time or elapsed run duration. "
             "Mixture times at 2 and 4 bar absolute are illustrative calculations under unconfirmed pressures, not BPR settings.")
COUPLING_NOTE = ("Changing the O2 supply level also changes gas flow, estimated mixture transit and potentially liquid holdup, "
                 "mass transfer and photon exposure, even at fixed V/Q_L. The comparison therefore does not isolate an oxygen-only causal effect. "
                 "Supplied O2 is not measured uptake; 0.5 mol O2/mol sulfide is not automatically substoichiometric or sulfone-suppressing.")
COMPARISON_NOTE = ("Compare the two frozen ranked nominations from each arm under matched evaluation opportunities. "
                   "Do not compare the best of the six informed screen conditions against two omitted-history nominations. "
                   "Overlapping conditions represent convergence and one shared measurement, not independent evidence. "
                   "The independent repeat assesses repeatability; it is not an extra opportunity to select the better result.")


def _load(campaign_dir: Path):
    input_path = campaign_dir / "input_freeze.json"
    nominations_path = campaign_dir / "nominations_freeze.json"
    frozen_input = json.loads(input_path.read_text())
    frozen_nominations = json.loads(nominations_path.read_text())
    snapshot = frozen_input["snapshot"]
    snapshot_hash = digest(snapshot)
    if frozen_input.get("snapshot_sha256") != snapshot_hash:
        raise ValueError("Input freeze snapshot hash mismatch")
    if frozen_nominations.get("input_freeze_sha256") != snapshot_hash:
        raise ValueError("Nominations do not belong to this frozen input snapshot")
    if frozen_input.get("flow_outcomes_seen") is not False or frozen_nominations.get("flow_outcomes_seen") is not False:
        raise ValueError("This exporter requires explicit pre-outcome freezes")
    if frozen_nominations.get("laboratory_released") is not False:
        raise ValueError("This exporter is for conditional, unreleased nominations")
    pool = snapshot["pool"]
    candidates = {c["candidate_id"]: c for c in pool["candidates"]}
    if len(candidates) != len(pool["candidates"]):
        raise ValueError("Duplicate candidate identities in frozen pool")
    for arm in ["history_informed", "history_omitted"]:
        chief = frozen_nominations["arms"][arm]["Chief"]
        core = chief["recommended_core_candidate_ids"]
        if len(core) != 6 or len(set(core)) != 6 or not any(set(g["candidate_ids"]) == set(core) for g in pool["grid_groups"]):
            raise ValueError(f"{arm} core is not a complete eligible frozen 3x2 grid")
        if any(candidates[c]["status"] != "eligible_under_declared_assumptions" for c in core):
            raise ValueError("A nomination uses a rejected candidate")
        nominees = chief["comparison_nomination_ids"]
        if len(nominees) != 2 or len(set(nominees)) != 2 or not set(nominees) <= set(core):
            raise ValueError(f"{arm} requires two distinct ranked nominations from its core")
        if chief["repeat_candidate_id"] not in core:
            raise ValueError("Repeat reference is not a core condition")
        reviews = chief["candidate_reviews"]
        if len(reviews) != 6 or {r["candidate_id"] for r in reviews} != set(core):
            raise ValueError("Chief reviews must cover exactly its six selected candidates")
    return frozen_input, frozen_nominations


def _authoritative_conditions(snapshot: dict, nominations: dict):
    pool = snapshot["pool"]
    by_id = {c["candidate_id"]: c for c in pool["candidates"]}
    chiefs = {arm: nominations["arms"][arm]["Chief"] for arm in ["history_informed", "history_omitted"]}
    reviews = {arm: {r["candidate_id"]: r for r in chief["candidate_reviews"]} for arm, chief in chiefs.items()}
    gas_assumptions = {a["id"]: a for a in pool["scenario"]["assumptions"]}
    rows = []
    def append(cid, condition_id, arm, repeat_of=None):
        c = by_id[cid]
        row = {"condition_id": condition_id, "candidate_id": cid, "scenario_id": c["scenario_id"],
            "table_order_only": len(rows) + 1, "proposed_execution_order": None, "comparison_arm": arm,
            "repeat_of_condition_id": repeat_of,
            "preparation_type": "independent_preparation_and_operation" if repeat_of else "independent_condition",
            "status": STATUS, "source_configuration_sha256": pool["configuration_sha256"],
            "time_coordinate": c["time_coordinate"], "oxygen_coordinate": c["oxygen_coordinate"],
            "nominal_oxygen_equivalents": c["nominal_oxygen_equivalents"],
            "proposed_setpoints": deepcopy(c["proposed_setpoints"]), "calculated": deepcopy(c["calculated"]),
            "review": deepcopy(reviews[arm][cid]), "actual_conditions": None, "actual_outcomes": None}
        row["gas_reference_assumptions"] = {
            "temperature_K": gas_assumptions["A_GAS_REFERENCE"]["temperature_K"],
            "pressure_bar_abs": gas_assumptions["A_GAS_REFERENCE"]["pressure_bar_abs"],
            "oxygen_mole_fraction": gas_assumptions["A_AIR_COMPOSITION"]["oxygen_mole_fraction"],
            "status": "assumed_unconfirmed"}
        rows.append(row)
        return row
    primary = {}
    for i, cid in enumerate(chiefs["history_informed"]["recommended_core_candidate_ids"], 1):
        primary[cid] = append(cid, f"HI{i:02d}", "history_informed")["condition_id"]
    repeat_id = chiefs["history_informed"]["repeat_candidate_id"]
    append(repeat_id, "HI-R01", "history_informed", primary[repeat_id])
    added = 0
    for cid in chiefs["history_omitted"]["comparison_nomination_ids"]:
        if cid not in primary:
            added += 1
            primary[cid] = append(cid, f"HO{added:02d}", "history_omitted")["condition_id"]
    comparison = {"status": "frozen_two_versus_two_before_flow_outcomes", "rule": COMPARISON_NOTE,
        "arms": {arm: [{"rank": rank, "candidate_id": cid, "condition_id": primary[cid],
            "review": deepcopy(reviews[arm][cid])} for rank, cid in enumerate(chief["comparison_nomination_ids"], 1)]
            for arm, chief in chiefs.items()},
        "shared_nominated_candidate_ids": sorted(set(chiefs["history_informed"]["comparison_nomination_ids"]) &
                                               set(chiefs["history_omitted"]["comparison_nomination_ids"])),
        "omitted_nominations_already_in_informed_core": [cid for cid in chiefs["history_omitted"]["comparison_nomination_ids"]
            if cid in chiefs["history_informed"]["recommended_core_candidate_ids"]],
        "additional_omitted_runs": added, "independent_repeat_condition_id": "HI-R01",
        "omitted_alternative_screen_execution_status": "The omitted-history arm also nominated a full six-condition alternative and repeat for the matched recommendation task. Only its two ranked comparison nominations are included in this combined experimental package; its other conditions and repeat are retained alternatives, not extra scheduled runs.",
        "repeat_aggregation_rule": "Use the original nominated run for primary matched comparison; report independent repeat separately unless an alternative rule is frozen before outcomes.",
        "execution_order": "Unassigned; finalize randomized or blocked order before outcomes. Table order is not laboratory instruction.",
        "history_boundary": "Both arms share the recipe, physical pool, instructions, role routing and opportunity. Only the informed arm receives optimization history. Full visible inputs are archived.",
        "interpretation_limit": "Two nominations per arm illustrate prospective choices; they cannot establish general superiority across chemistries."}
    return rows, comparison


def _flat(row):
    proposed, calculated = row["proposed_setpoints"], row["calculated"]
    illustrations = {d["pressure_bar_abs"]: d for d in calculated["mixture_transit"]["illustrations"]}
    return {
        "condition_id": row["condition_id"], "candidate_id": row["candidate_id"], "comparison_arm": row["comparison_arm"],
        "table_order_only": row["table_order_only"], "proposed_execution_order": None,
        "repeat_of_condition_id": row["repeat_of_condition_id"], "preparation_type": row["preparation_type"],
        "status": STATUS, "scenario_id": row["scenario_id"], "time_coordinate": row["time_coordinate"],
        "oxygen_coordinate": row["oxygen_coordinate"], "liquid_only_space_time_min": calculated["liquid_only_space_time_min"],
        "proposed_liquid_flow_mL_min": proposed["liquid_flow_mL_min"],
        "proposed_gas_reference_flow_mL_min": proposed["gas_reference_flow_mL_min"],
        "nominal_oxygen_equivalents": row["nominal_oxygen_equivalents"],
        "calculated_oxygen_equivalents_supplied": calculated["oxygen_equivalents_supplied"],
        "oxygen_relative_quantization_error": calculated["oxygen_relative_quantization_error"],
        "assumed_gas_reference_temperature_K": row["gas_reference_assumptions"]["temperature_K"],
        "assumed_gas_reference_pressure_bar_abs": row["gas_reference_assumptions"]["pressure_bar_abs"],
        "assumed_air_oxygen_mole_fraction": row["gas_reference_assumptions"]["oxygen_mole_fraction"],
        "mixture_transit_min_at_unconfirmed_2bar_abs_illustration": illustrations[2.0]["no_slip_mixture_transit_min_estimate"],
        "mixture_transit_min_at_unconfirmed_4bar_abs_illustration": illustrations[4.0]["no_slip_mixture_transit_min_estimate"],
        "inlet_reference_volume_index_min": calculated["inlet_reference_volume_index_min"],
        "actual_liquid_contact_time_min": None, "proposed_temperature_C": proposed["temperature_C"],
        "selected_BPR": None, "reactor_pressure_bar_abs": None, "actual_light_power_setting": None,
        "substrate_mmol_min": calculated["substrate_mmol_min"], "catalyst_mmol_min": calculated["catalyst_mmol_min"],
        "oxygen_mmol_min_supplied": calculated["oxygen_mmol_min_supplied"],
        "feed_mL_per_elapsed_hour_at_continuous_setting": calculated["feed_mL_per_elapsed_hour_at_continuous_setting"],
        "substrate_mmol_per_elapsed_hour_at_continuous_setting": calculated["substrate_mmol_per_elapsed_hour_at_continuous_setting"],
        "catalyst_mmol_per_elapsed_hour_at_continuous_setting": calculated["catalyst_mmol_per_elapsed_hour_at_continuous_setting"],
        "review_verdict": row["review"]["verdict"], "rationale": row["review"]["rationale"],
        "citations": " | ".join(row["review"].get("citations", [])),
        "source_configuration_sha256": row["source_configuration_sha256"],
    }


def _results(rows):
    templates = []
    for row in rows:
        templates.append({"condition_id": row["condition_id"], "candidate_id": row["candidate_id"],
            "repeat_of_condition_id": row["repeat_of_condition_id"], "proposed_conditions": deepcopy(row),
            "sample_id": None, "actual_execution_order": None, "actual_run_datetime": None,
            "actual_liquid_flow_mL_min": None, "actual_gas_flow_display": None, "actual_gas_units_reference": None,
            "actual_gas_identity_calibration": None, "actual_reactor_pressure_bar_abs": None,
            "actual_pressure_sensor_location": None, "actual_temperature_C": None, "actual_light_setting": None,
            "actual_substrate_concentration_M": None, "actual_catalyst_loading_mol_percent": None,
            "actual_solvent_composition": None, "actual_reactor_id": None, "actual_reactor_volume_mL": None,
            "actual_wavelength_nm": None,
            "actual_feed_mode": None, "actual_sample_loop_volume_mL": None, "actual_priming_volume_mL": None,
            "actual_feed_elapsed_min": None, "actual_collection_window": None, "collected_volume_mL": None,
            "residual_sulfide_pct": None, "sulfoxide_pct": None, "sulfone_pct": None,
            "other_products_pct": None, "unknown_components_pct": None,
            "analytical_method": None, "analytical_basis": None, "internal_standard": None,
            "calibration_details": None, "analytical_uncertainty": None, "sample_handling": None,
            "time_from_collection_to_analysis": None, "repeat_type_actual": None,
            "oxidation_continues_after_collection_observed": None,
            "actual_independent_preparation": None, "gas_behavior_observed": None,
            "precipitation_observed": None, "leak_observed": None, "pressure_instability_observed": None,
            "deviations": None, "analyst_notes": None})
    return templates


def _write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _diagram_nodes():
    return [
        {"key": "air", "x": 25, "y": 25, "lines": ["Air supply", "composition assumed"], "unknown": True},
        {"key": "regulator", "x": 255, "y": 25, "lines": ["Regulator", "ID / supply P pending"], "unknown": True},
        {"key": "mfc", "x": 485, "y": 25, "lines": ["MFC: mfc_ffc00", "calibration pending"], "unknown": False},
        {"key": "check", "x": 715, "y": 25, "lines": ["Check: cv-3301", "pressure rating pending"], "unknown": False},
        {"key": "feed", "x": 25, "y": 205, "lines": ["Premixed EtOH feed", "mode / volume pending"], "unknown": True},
        {"key": "pump", "x": 255, "y": 205, "lines": ["R-series pump", "pump_r_series_hplc"], "unknown": False},
        {"key": "mixer", "x": 485, "y": 205, "lines": ["T-mixer: p-713", "gas + liquid"], "unknown": False},
        {"key": "coil", "x": 715, "y": 205, "lines": ["Coil: coil_row5", "UV150 450 nm / 25 C"], "unknown": False},
        {"key": "sensor", "x": 715, "y": 385, "lines": ["Pressure sensor", "ID / location pending"], "unknown": True},
        {"key": "bpr", "x": 485, "y": 385, "lines": ["BPR", "ID / basis / P pending"], "unknown": True},
        {"key": "collector", "x": 255, "y": 385, "lines": ["Collector", "handling pending"], "unknown": True},
    ]


def _diagram_paths():
    return [[(225, 70), (255, 70)], [(455, 70), (485, 70)], [(685, 70), (715, 70)],
        [(815, 115), (815, 160), (585, 160), (585, 205)],
        [(225, 250), (255, 250)], [(455, 250), (485, 250)], [(685, 250), (715, 250)],
        [(815, 295), (815, 385)], [(715, 430), (685, 430)], [(485, 430), (455, 430)]]


def _svg():
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="550" viewBox="0 0 960 550" role="img" aria-label="Conditional apparatus proposal, unresolved assignments dashed">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#264653"/></marker></defs>',
        '<rect width="960" height="550" fill="white"/>']
    for path in _diagram_paths():
        out.append('<polyline points="' + " ".join(f"{x},{y}" for x, y in path) + '" fill="none" stroke="#264653" stroke-width="3" marker-end="url(#arrow)"/>')
    for node in _diagram_nodes():
        dash = ' stroke-dasharray="9 6"' if node["unknown"] else ""
        out.append(f'<rect x="{node["x"]}" y="{node["y"]}" width="200" height="90" rx="10" fill="#eef6f5" stroke="#264653" stroke-width="2"{dash}/>')
        for i, line in enumerate(node["lines"]):
            out.append(f'<text x="{node["x"] + 100}" y="{node["y"] + 34 + i * 29}" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(line)}</text>')
    out.extend(['<text x="25" y="510" font-family="sans-serif" font-size="18">Solid: proposed inventory options; dashed: undefined details. No actual assignments confirmed.</text>',
        '<text x="25" y="539" font-family="sans-serif" font-size="18">Pressure is unresolved. Illustrative 2 / 4 bar absolute calculations are not BPR choices.</text>', '</svg>'])
    return "\n".join(out)


def _num(value):
    return "pending" if value is None else f"{value:.4g}" if isinstance(value, (int, float)) else str(value)


def _html_table(headers, rows):
    return '<table><thead><tr>' + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + '</tr></thead><tbody>' + "".join(
        '<tr>' + "".join(f"<td>{html.escape(_num(c))}</td>" for c in row) + '</tr>' for row in rows) + '</tbody></table>'


def _tables(flat):
    conditions = [[r["condition_id"], r["candidate_id"], r["liquid_only_space_time_min"], r["proposed_liquid_flow_mL_min"],
        r["proposed_gas_reference_flow_mL_min"], r["nominal_oxygen_equivalents"], r["calculated_oxygen_equivalents_supplied"]] for r in flat]
    rates = [[r["condition_id"], r["mixture_transit_min_at_unconfirmed_2bar_abs_illustration"],
        r["mixture_transit_min_at_unconfirmed_4bar_abs_illustration"], r["feed_mL_per_elapsed_hour_at_continuous_setting"],
        r["substrate_mmol_per_elapsed_hour_at_continuous_setting"], r["catalyst_mmol_per_elapsed_hour_at_continuous_setting"]] for r in flat]
    return conditions, rates


def _methods_text(snapshot):
    configuration = snapshot.get("configuration", {})
    routes = "; ".join(f"{role}: {model}" for role, model in configuration.get("routing", {}).items()) or "routing preserved in input freeze"
    return (f"Frozen model routing: {routes}. Reasoning effort: {configuration.get('reasoning_effort', 'not specified')}; "
            f"service tier: {configuration.get('service_tier', 'not specified')}; "
            f"maximum campaign budget USD {configuration.get('maximum_cost_USD', 'not specified')}. "
            "Both arms used the same common instructions and physical candidate pool. Batch observations were not fitted as flow kinetics; no monotonic yield/time rule or forced time reduction was used.")


def _calculation_basis(snapshot, rows):
    scenario = snapshot["pool"]["scenario"]
    first = rows[0]
    reference = first["gas_reference_assumptions"]
    calculated = first["calculated"]
    return {
        "schema_version": "khu_report_calculation_basis_v1",
        "status": "calculated_basis_under_frozen_unconfirmed_assumptions",
        "R_value": R_ML_BAR_PER_MMOL_K, "R_units": "mL bar / (mmol K)",
        "variable_units": {"C": "mol/L, numerically mmol/mL", "Q_L": "mL/min liquid",
            "Q_ref": "mL/min gas at declared reference", "V": "mL nominal reactor volume",
            "P_ref": "bar absolute reference pressure", "P_abs": "bar absolute reactor pressure",
            "T_ref": "K reference temperature", "T_rxn": "K reactor temperature",
            "x_O2": "mole fraction O2 in supplied gas", "n_substrate": "mmol/min",
            "n_gas": "mmol/min total supplied gas", "oxygen_equivalents": "mol O2 supplied / mol sulfide supplied",
            "tau_L": "min liquid-only space time", "tau_mix": "min ideal-gas no-slip mixture transit estimate"},
        "equations": {"substrate_molar_flow": "n_substrate = C * Q_L",
            "catalyst_molar_flow": "n_catalyst = n_substrate * catalyst_loading_mol_percent / 100",
            "total_gas_molar_flow": "n_gas = P_ref * Q_ref / (R * T_ref)",
            "oxygen_equivalents": "oxygen_equivalents = x_O2 * n_gas / (C * Q_L)",
            "reactor_gas_volume_flow": "Q_gas_rxn = Q_ref * (P_ref / P_abs) * (T_rxn / T_ref)",
            "liquid_only_space_time": "tau_L = V / Q_L",
            "mixture_transit_estimate": "tau_mix = V / (Q_L + Q_gas_rxn)"},
        "gas_reference_assumptions": deepcopy(reference),
        "rounding_policy": {"liquid_increment_mL_min": scenario["hardware_choices"]["pump"]["source_reported_inventory_record"]["flow_rate_increment_mL_min"],
            "gas_reference_increment_mL_min": scenario["hardware_choices"]["mfc"]["source_reported_inventory_record"]["flow_rate_increment_sccm"],
            "calculation_sequence": "Apply controller increments first; recompute molar flows, supplied equivalents and time indices from the quantized setpoints. Do not clamp infeasible gas settings.",
            "display_only": "Human-readable numerical tables use four significant digits; CSV/JSON retain stored numerical precision."},
        "pressure_boundary": "P_abs remains unknown. The 2 and 4 bar absolute examples are unconfirmed illustrations, never selected BPR settings or measured pressure.",
        "model_limitations": "Ideal gas, constant inlet gas flow and no slip; dissolution, consumption and axial pressure gradients are not modeled. Supply is not uptake. Neither tau_L nor tau_mix is measured liquid contact time or experimental duration.",
        "worked_example": {"condition_id": first["condition_id"], "candidate_id": first["candidate_id"],
            "inputs": {"C_mol_L": scenario["shared_recipe"]["substrate_concentration_M"],
                "Q_L_mL_min": first["proposed_setpoints"]["liquid_flow_mL_min"],
                "Q_ref_mL_min": first["proposed_setpoints"]["gas_reference_flow_mL_min"],
                "V_mL": scenario["geometry"]["volume_mL"], "P_ref_bar_abs": reference["pressure_bar_abs"],
                "T_ref_K": reference["temperature_K"], "T_rxn_K": first["proposed_setpoints"]["temperature_C"] + 273.15,
                "x_O2": reference["oxygen_mole_fraction"]},
            "outputs": {"substrate_mmol_min": calculated["substrate_mmol_min"],
                "total_gas_mmol_min": calculated["total_gas_mmol_min"],
                "oxygen_equivalents_supplied": calculated["oxygen_equivalents_supplied"],
                "liquid_only_space_time_min": calculated["liquid_only_space_time_min"],
                "illustrative_mixture_transit": deepcopy(calculated["mixture_transit"]["illustrations"])}}}


def _formula_lines(basis):
    example = basis["worked_example"]
    output = example["outputs"]
    return [
        "n_substrate = C * Q_L; n_gas = P_ref * Q_ref / (R * T_ref)",
        "O2 equivalents = x_O2 * n_gas / (C * Q_L)",
        "Q_gas,rxn = Q_ref * (P_ref / P_abs) * (T_rxn / T_ref)",
        "tau_L = V / Q_L; tau_mix = V / (Q_L + Q_gas,rxn)",
        f"R = {basis['R_value']:.12g} mL bar/(mmol K); pressures absolute, temperatures K; C in mmol/mL.",
        "Controller settings are rounded first; all quantities are recomputed. Supply is not uptake; 2/4 bar are illustrations only.",
        f"Example {example['condition_id']} ({example['candidate_id']}): substrate {_num(output['substrate_mmol_min'])} mmol/min; total gas {_num(output['total_gas_mmol_min'])} mmol/min; O2 {_num(output['oxygen_equivalents_supplied'])} eq."]


def _report_html(snapshot, nominations, rows, comparison, preparation, calculation_basis):
    flat = [_flat(row) for row in rows]
    conditions, rates = _tables(flat)
    scenario = snapshot["pool"]["scenario"]
    chunks = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><title>KHU conditional first-screen review</title>',
        '<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 24px;color:#15323b}h1{font-size:30px}h2{margin-top:34px;color:#16685f}table{border-collapse:collapse;width:100%;font-size:14px;margin:14px 0}th,td{border:1px solid #c9d8db;padding:8px;text-align:left;vertical-align:top}th{background:#e7f1f0}img{width:100%;height:auto}.status{padding:16px;border-left:6px solid #b66b24;background:#fff4e5}.small{font-size:13px;color:#42606a}li{margin:7px 0}code{overflow-wrap:anywhere}a{color:#16685f}</style></head><body>',
        '<h1>KHU first homogeneous acridine flow screen</h1>',
        '<p class="status"><strong>Conditional review package - not released for laboratory execution.</strong> These are frozen model nominations and calculated settings under explicit assumptions. No flow outcomes or experimental validation are reported.</p>',
        f'<p>{len(rows)} planned runs represent six informed screen conditions, one independently prepared/operated repeat, and {comparison["additional_omitted_runs"]} additional unique history-omitted nominations. Table order is not a finalized run order.</p>',
        '<h2>Conditional condition table</h2><p>' + html.escape(TIME_NOTE) + '</p>',
        _html_table(["Run", "Candidate", "V/Q_L (min)", "Q_L (mL/min)", "Gas reference (mL/min)", "Nominal O2 eq", "Calculated O2 eq"], conditions),
        '<p>' + html.escape(COUPLING_NOTE) + '</p>',
        '<p>Gas calculations assume air O2 mole fraction 0.2095 and reference 273.15 K / 1.01325 bar absolute. Proposed temperature is 25 C. BPR, actual gas calibration/reference and light power setting are unresolved.</p>',
        '<h2>Apparatus proposal</h2><img src="apparatus.svg" alt="Proposed apparatus, undefined components dashed"><p>Proposed coil: 10 mL PFA, 1 mm ID; proposed module: light_uv150_row10. Coil geometry is source-reported nominal. Calculated tubing length does not establish illuminated length or liquid holdup.</p>',
        '<h2>Illustrative time indices and scalable material rates</h2>',
        _html_table(["Run", "Mixture min at assumed 2 bar abs", "Mixture min at assumed 4 bar abs", "Feed mL/h", "Substrate mmol/h", "Catalyst mmol/h"], rates),
        '<p>Both pressures are illustrations, not selected BPR settings. Rates apply only while continuous feed is actually delivered. No run duration, equilibration period, collection window or total material consumption is inferred from V/Q_L.</p>',
        '<h2>Calculation basis</h2><p>' + '<br>'.join(html.escape(line) for line in _formula_lines(calculation_basis)) + '</p><p class="small">Complete equations, units and the unrounded worked example are in calculation_basis.json.</p>',
        '<h2>Scalable feed preparation</h2><p>' + html.escape(preparation["instruction"]) + '</p>',
        '<p>' + html.escape(preparation["mass_basis_explanation"]) + '</p>',
        '<p>Post-freeze calculation references, checked 13 October 2026: <a href="https://www.sigmaaldrich.com/SG/en/product/sial/t28002">thioanisole molecular weight</a>; <a href="https://www.sigmaaldrich.com/US/en/product/sigma/01640">acridine molecular weight</a>. These references were added to the report calculation basis, not to either frozen model input. Total preparation volume, solubility and handling require laboratory review.</p>',
        '<h2>Fair comparison and repeated conditions</h2><p>' + html.escape(COMPARISON_NOTE) + '</p>']
    matched = [[arm, item["rank"], item["candidate_id"], item["condition_id"]]
        for arm, items in comparison["arms"].items() for item in items]
    chunks.append(_html_table(["Arm", "Frozen rank", "Nomination", "Measurement condition"], matched))
    chunks.append('<p>' + html.escape(comparison["history_boundary"] + " " + comparison["repeat_aggregation_rule"]) + '</p>')
    chunks.append('<h2>Condition rationale and traceable citations</h2>')
    chunks.append(_html_table(["Run", "Verdict", "Frozen Chief rationale", "Visible citations"],
        [[r["condition_id"], r["review"]["verdict"], r["review"]["rationale"], "; ".join(r["review"].get("citations", []))] for r in rows]))
    for arm in ["history_informed", "history_omitted"]:
        chief = nominations["arms"][arm]["Chief"]
        chunks.append('<h2>' + html.escape(arm.replace("_", " ").title()) + ' council conclusion</h2><p>' + html.escape(chief.get("conclusion", "")) + '</p>')
        if arm == "history_omitted":
            chunks.append('<p>' + html.escape(comparison["omitted_alternative_screen_execution_status"]) + '</p>')
        chunks.append(_html_table(["Type", "Finding", "Citations"], [[f.get("kind"), f.get("statement"), "; ".join(f.get("citations", []))] for f in chief.get("findings", [])]))
        chunks.append('<ul>' + ''.join('<li>' + html.escape(x) + '</li>' for x in chief.get("limitations", [])) + '</ul>')
    chunks.append('<h2>Assumptions and remaining laboratory confirmations</h2><ul>' + ''.join('<li>' + html.escape(a["id"] + ": " + a.get("explanation", "")) + '</li>' for a in scenario["assumptions"]) + '</ul>')
    chunks.append('<ul>' + ''.join('<li>' + html.escape(x) + '</li>' for x in scenario["unresolved_release_items"]) + '</ul>')
    chunks.append('<h2>Results return</h2><p>Use results_template.csv or results_template.json. Record actual conditions separately, including residual sulfide, sulfoxide, sulfone, other/unknown products, analytical method/calibration, uncertainty and sample handling. Blank/unknown operational observations are not false. Preserve reported percentages above 100; do not normalize mass closure.</p>')
    rejected = [c for c in snapshot["pool"]["candidates"] if c["status"] == "rejected"]
    chunks.append('<h2>Retained pool rejections</h2>' + _html_table(["Candidate", "Reason"], [[c["candidate_id"], "; ".join(c["rejection_reasons"])] for c in rejected]))
    chunks.append('<h2>Methods and archive</h2><p>The immutable input and nomination freezes, full pool including rejections, evidence coverage, complete arm reviews and API audit are preserved. Source hashes and artifact hashes are in manifest.json. Internal review agreement does not establish laboratory readiness or chemical performance.</p>')
    chunks.append('<p>' + html.escape(_methods_text(snapshot)) + '</p>')
    audit = nominations.get("api_audit", {})
    chunks.append('<p class="small">Recorded attempts: ' + html.escape(str(audit.get("attempt_count", "unknown"))) + '; known token cost USD: ' + html.escape(str(audit.get("known_token_cost_USD", "unknown"))) + '; accounted cost USD: ' + html.escape(str(audit.get("accounted_cost_USD", "unknown"))) + '.</p>')
    chunks.append('<p class="small">Input snapshot SHA-256: <code>' + digest(snapshot) + '</code><br>Pool configuration SHA-256: <code>' + snapshot["pool"]["configuration_sha256"] + '</code></p></body></html>')
    return "\n".join(chunks)


def _pdf(output, snapshot, nominations, rows, comparison, preparation, calculation_basis):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak
    from reportlab.graphics.shapes import Drawing, Rect, String, PolyLine
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font = "Helvetica"
    for path in [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf")]:
        if path.exists():
            pdfmetrics.registerFont(TTFont("KHUReportSans", str(path)))
            font = "KHUReportSans"
            break
    styles = getSampleStyleSheet()
    for key in ["Normal", "BodyText", "Title", "Heading1", "Heading2"]:
        styles[key].fontName = font
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 13
    styles["BodyText"].spaceAfter = 6
    styles["Heading2"].fontSize = 12
    styles["Heading2"].leading = 16
    styles["Heading2"].textColor = colors.HexColor("#16685f")
    from reportlab.lib.styles import ParagraphStyle
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=7.4, leading=10, spaceAfter=0)
    story = []
    def p(text, style="BodyText"):
        return Paragraph(html.escape(str(text)).replace("\n", "<br/>"), styles[style])
    def heading(text):
        story.append(p(text, "Heading2"))
    def table(headers, data, widths=None):
        cells = [[Paragraph(html.escape(_num(value)), small) for value in row] for row in [headers, *data]]
        obj = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        obj.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7f1f0")),
            ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#b9cccf")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        return obj
    usable = A4[0] - 72
    flat = [_flat(row) for row in rows]
    conditions, rates = _tables(flat)
    story += [p("KHU first homogeneous acridine flow screen", "Title"),
        p("CONDITIONAL REVIEW - NOT RELEASED FOR LABORATORY EXECUTION", "Heading2"),
        p("Frozen model nominations and calculated settings under explicit assumptions. No measured flow outcomes or experimental validation are reported."),
        p(f"{len(rows)} proposed runs: six informed conditions, one independent repeat and {comparison['additional_omitted_runs']} additional unique omitted-history nominations. Table order is not a finalized execution order.")]
    heading("Conditional settings")
    story += [p(TIME_NOTE), table(["Run", "Candidate", "V/Q_L min", "Q_L mL/min", "Gas ref. mL/min", "Nominal O2 eq", "Calc. O2 eq"], conditions, [53, 58, 72, 75, 86, 83, usable-427]),
        p(COUPLING_NOTE),
        p("Air O2 fraction 0.2095 and gas reference 273.15 K / 1.01325 bar absolute are unconfirmed assumptions. Proposed temperature is 25 C. BPR and light power setting remain unresolved.")]
    heading("Scalable feed preparation")
    story += [p(preparation["instruction"]), p(preparation["mass_basis_explanation"]),
        p("Molecular weights are post-freeze report references, checked 13 October 2026: thioanisole 124.20 g/mol (Sigma T28002); acridine 179.22 g/mol (Sigma 01640). Actual bottle purity is unknown. Total campaign volume depends on operating/collection windows."), PageBreak()]
    heading("Apparatus proposal and unknown assignments")
    drawing = Drawing(960, 550)
    for path in _diagram_paths():
        drawing.add(PolyLine([coordinate for x, y in path for coordinate in [x, 550-y]], strokeColor=colors.HexColor("#264653"), strokeWidth=3))
        (x1, y1), (x2, y2) = path[-2:]
        # Small V-shaped arrow at each path end, using the same topology as SVG.
        if x1 == x2:
            sign = 1 if y2 > y1 else -1
            drawing.add(PolyLine([x2-7, 550-y2+sign*10, x2, 550-y2, x2+7, 550-y2+sign*10], strokeColor=colors.HexColor("#264653"), strokeWidth=3))
        else:
            sign = 1 if x2 > x1 else -1
            drawing.add(PolyLine([x2-sign*10, 550-y2-7, x2, 550-y2, x2-sign*10, 550-y2+7], strokeColor=colors.HexColor("#264653"), strokeWidth=3))
    for node in _diagram_nodes():
        drawing.add(Rect(node["x"], 550-node["y"]-90, 200, 90, rx=10, ry=10,
            fillColor=colors.HexColor("#eef6f5"), strokeColor=colors.HexColor("#264653"),
            strokeWidth=2, strokeDashArray=[9, 6] if node["unknown"] else None))
        for i, line in enumerate(node["lines"]):
            drawing.add(String(node["x"]+100, 550-node["y"]-34-i*29, line, textAnchor="middle", fontName=font, fontSize=17))
    drawing.add(String(25, 40, "Solid: proposed inventory option. Dashed: undefined details. Actual assignments remain unknown.", fontName=font, fontSize=17))
    drawing.add(String(25, 12, "Reactor pressure / BPR unresolved; 2 and 4 bar absolute are illustrations only.", fontName=font, fontSize=17))
    drawing.scale(usable/960, usable/960)
    drawing.width, drawing.height = usable, 550*usable/960
    story += [drawing, p("Proposed inventory IDs: pump_r_series_hplc; mfc_ffc00; cv-3301; p-713; coil_row5; light_uv150_row10. Nominal reactor: PFA 10 mL, 1 mm ID. Actual illuminated length and liquid holdup remain unknown.")]
    heading("Illustrative mixture times and continuous-feed rates")
    story += [table(["Run", "Mix min: assumed 2 bar abs", "Mix min: assumed 4 bar abs", "Feed mL/h", "Substrate mmol/h", "Catalyst mmol/h"], rates, [53, 104, 104, 80, 91, usable-432]),
        p("Mixture transit assumes ideal gas and no slip, with no gas dissolution/consumption correction. Neither illustrative pressure is a BPR choice. Material rates apply only during actual continuous feed; they are not run totals.")]
    heading("Calculation basis")
    story.append(Paragraph('<br/>'.join(html.escape(line) for line in _formula_lines(calculation_basis)), small))
    story.append(PageBreak())
    heading("Matched comparison, repeat and information boundary")
    story += [p(COMPARISON_NOTE), table(["Arm", "Rank", "Candidate", "Condition"], [[arm.replace("history_", ""), item["rank"], item["candidate_id"], item["condition_id"]]
        for arm, items in comparison["arms"].items() for item in items], [200, 55, 130, usable-385]),
        p(comparison["history_boundary"]), p(comparison["repeat_aggregation_rule"])]
    heading("Rationale and visible evidence citations")
    story.append(table(["Run / verdict", "Frozen Chief rationale and citations"], [[r["condition_id"] + " / " + r["review"]["verdict"], r["review"]["rationale"] + " Citations: " + "; ".join(r["review"].get("citations", []))] for r in rows], [95, usable-95]))
    for arm in ["history_informed", "history_omitted"]:
        story.append(PageBreak())
        chief = nominations["arms"][arm]["Chief"]
        heading(arm.replace("_", " ").title() + " council conclusion")
        story.append(p(chief.get("conclusion", "")))
        if arm == "history_omitted":
            story.append(p(comparison["omitted_alternative_screen_execution_status"]))
        for finding in chief.get("findings", []):
            story.append(p(finding.get("kind", "") + ": " + finding.get("statement", "") + " [" + "; ".join(finding.get("citations", [])) + "]"))
        heading("Retained limitations")
        story.extend(p("- " + limitation) for limitation in chief.get("limitations", []))
    story.append(PageBreak())
    heading("Assumptions and laboratory confirmations")
    scenario = snapshot["pool"]["scenario"]
    story.extend(p(a["id"] + ": " + a.get("explanation", "")) for a in scenario["assumptions"])
    heading("Still required before release")
    story.extend(p("- " + x) for x in scenario["unresolved_release_items"])
    heading("Results and analytical definitions")
    story.append(p("Return results_template.csv/json with actual settings and residual sulfide, sulfoxide, sulfone, other/unknown products separately. Record analytical method, internal standard/calibration, uncertainty, sample handling, collection-to-analysis delay and operating observations. Unknown observations remain blank/null, not false. Preserve values above 100; do not normalize component recovery."))
    heading("Retained pool rejections")
    story.append(table(["Candidate", "Rejection reason"], [[c["candidate_id"], "; ".join(c["rejection_reasons"])] for c in snapshot["pool"]["candidates"] if c["status"] == "rejected"], [72, usable-72]))
    heading("Methods and immutable archive")
    story.append(p(_methods_text(snapshot)))
    audit = nominations.get("api_audit", {})
    story.append(p(f"Recorded API attempts: {audit.get('attempt_count', 'unknown')}; known token cost USD {audit.get('known_token_cost_USD', 'unknown')}; accounted cost USD {audit.get('accounted_cost_USD', 'unknown')}. Original freezes, all arm reviews, complete visible inputs, evidence coverage, pool rejections, configuration and API audit are retained. The manifest records original and exported artifact hashes."))
    story.append(p("Input snapshot SHA-256: " + digest(snapshot)))
    story.append(p("Post-freeze molecular-weight references (not provided to the models): https://www.sigmaaldrich.com/SG/en/product/sial/t28002 ; https://www.sigmaaldrich.com/US/en/product/sigma/01640"))
    def footer(canvas, doc):
        canvas.setFont(font, 7)
        canvas.setFillColor(colors.HexColor("#42606a"))
        canvas.drawString(36, 23, "KHU conditional first screen | not a laboratory-release protocol")
        canvas.drawRightString(A4[0]-36, 23, f"Page {doc.page}")
    SimpleDocTemplate(str(output), pagesize=A4, rightMargin=36, leftMargin=36,
        topMargin=32, bottomMargin=38, title="KHU conditional first-screen review", author="KHU study workspace").build(story, onFirstPage=footer, onLaterPages=footer)


def export_campaign_report(root, campaign_dir, output_dir, pdf=False):
    """Export new local review artifacts, refusing overwrite or changed freezes."""
    root, campaign_dir, output_dir = Path(root), Path(campaign_dir), Path(output_dir)
    if pdf and importlib.util.find_spec("reportlab") is None:
        raise RuntimeError("PDF export requires reportlab in the selected Python environment")
    frozen_input, nominations = _load(campaign_dir)
    snapshot = frozen_input["snapshot"]
    rows, comparison = _authoritative_conditions(snapshot, nominations)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Review exports are immutable; choose a new empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    original_hashes = {name: file_hash(campaign_dir / name) for name in ["input_freeze.json", "nominations_freeze.json"]}
    recipe = snapshot["pool"]["scenario"]["shared_recipe"]
    if (recipe["substrate"], recipe["catalyst"], recipe["substrate_concentration_M"], recipe["catalyst_loading_mol_percent"]) != ("thioanisole", "acridine", .25, 3.):
        raise ValueError("Review the feed preparation references for a changed frozen recipe before exporting")
    preparation = {"status": "scalable_proposed_feed_not_actual_preparation", "final_volume_example_mL": 10.0,
        "substrate": "thioanisole", "substrate_concentration_M": .25, "substrate_mmol_example": 2.5,
        "catalyst": "acridine", "catalyst_loading_mol_percent": 3., "catalyst_mmol_example": .075,
        "catalyst_concentration_mM": 7.5, "solvent": "EtOH", "actual_weighed_masses_mg": None,
        "molecular_weight_g_mol": {"thioanisole": 124.20, "acridine": 179.22},
        "pure_equivalent_mass_mg": {"thioanisole": 310.50, "acridine": 13.4415},
        "actual_bottle_purity_fraction": {"thioanisole": None, "acridine": None},
        "actual_bottle_identity_confirmed": None, "solubility_confirmed": None,
        "mass_formulas": {"pure_equivalent_mass_mg": "amount_mmol * molecular_weight_g_mol",
            "weighed_mass_mg": "pure_equivalent_mass_mg / actual_bottle_purity_fraction (0 < fraction <= 1)"},
        "post_freeze_calculation_references": [
            {"compound": "thioanisole", "molecular_weight_g_mol": 124.20, "checked_on": "2026-10-13", "url": "https://www.sigmaaldrich.com/SG/en/product/sial/t28002"},
            {"compound": "acridine", "molecular_weight_g_mol": 179.22, "checked_on": "2026-10-13", "url": "https://www.sigmaaldrich.com/US/en/product/sigma/01640"}],
        "reference_boundary": "Added only for post-freeze report calculations; not part of either arm's frozen model input. Vendor product purity is not assigned to KHU material.",
        "instruction": "Scalable example only: for a final 10 mL feed at 0.25 M thioanisole and 3 mol% acridine, use 2.5 mmol thioanisole and 0.075 mmol acridine, then make to the final 10 mL with EtOH after confirming compatibility and solubility. Scale both amounts linearly with final volume. This example does not establish the amount required for the campaign.",
        "mass_basis_explanation": "The 10 mL example corresponds to 310.50 mg thioanisole and 13.4415 mg acridine as pure-equivalent masses. Actual weighed mass equals pure-equivalent mass divided by the actual bottle purity fraction. Bottle identity, purity and solubility remain unknown; vendor product purities are not assumed. No density, neat volume, quench recipe or collection duration is inferred.",
        "total_campaign_feed_volume_mL": None, "total_substrate_mmol": None, "total_catalyst_mmol": None,
        "missing_for_totals": ["feed mode", "priming and collection windows", "carrier/flush volumes", "reserve", "material availability"]}
    payload = {"status": STATUS, "input_snapshot_sha256": digest(snapshot),
        "nomination_freeze_file_sha256": original_hashes["nominations_freeze.json"], "conditions": rows,
        "scenario": snapshot["pool"]["scenario"], "comparison": comparison, "feed_preparation": preparation,
        "time_basis_warning": TIME_NOTE, "coupled_factor_warning": COUPLING_NOTE}
    calculation_basis = _calculation_basis(snapshot, rows)
    write_json(output_dir / "authoritative_conditions.json", payload)
    write_json(output_dir / "calculation_basis.json", calculation_basis)
    _write_csv(output_dir / "conditions.csv", [_flat(row) for row in rows])
    templates = _results(rows)
    write_json(output_dir / "results_template.json", {"status": "blank_actuals_unknown_not_false", "records": templates})
    _write_csv(output_dir / "results_template.csv", [{k: v for k, v in t.items() if k != "proposed_conditions"} for t in templates])
    write_json(output_dir / "feed_preparation.json", preparation)
    write_json(output_dir / "preserved_pool.json", snapshot["pool"])
    selected_ids = {r["candidate_id"] for r in rows}
    write_json(output_dir / "pool_selection_dispositions.json", [{"candidate_id": c["candidate_id"],
        "disposition": "selected_for_proposed_measurement" if c["candidate_id"] in selected_ids else
            "rejected_by_pool_constraints" if c["status"] == "rejected" else "eligible_not_selected",
        "pool_rejection_reasons": c["rejection_reasons"]} for c in snapshot["pool"]["candidates"]])
    write_json(output_dir / "api_audit.json", nominations.get("api_audit", {}))
    for filename in original_hashes:
        shutil.copyfile(campaign_dir / filename, output_dir / filename)
    (output_dir / "apparatus.svg").write_text(_svg(), encoding="utf-8")
    (output_dir / "campaign_report.html").write_text(_report_html(snapshot, nominations, rows, comparison, preparation, calculation_basis), encoding="utf-8")
    if pdf:
        _pdf(output_dir / "campaign_report.pdf", snapshot, nominations, rows, comparison, preparation, calculation_basis)
    if any(file_hash(campaign_dir / name) != old_hash for name, old_hash in original_hashes.items()):
        raise ValueError("A campaign freeze changed during export; do not use this incomplete package")
    files = {p.name: {"sha256": file_hash(p), "size_bytes": p.stat().st_size} for p in sorted(output_dir.iterdir()) if p.is_file()}
    manifest = {"schema_version": "khu_campaign_report_manifest_v1", "status": STATUS,
        "source_campaign_directory": str(campaign_dir.resolve()), "original_freeze_sha256": original_hashes,
        "input_snapshot_sha256": digest(snapshot), "source_hashes": snapshot.get("source_hashes"),
        "exporter_source_sha256": file_hash(Path(__file__)), "files": files,
        "condition_count": len(rows), "unique_condition_count": len({r["candidate_id"] for r in rows}),
        "comparison": comparison, "api_audit": {k: v for k, v in nominations.get("api_audit", {}).items() if k != "entries"},
        "pdf_visual_QA": "required_before_delivery" if pdf else "not_requested",
        "manifest_integrity_note": "This manifest hashes all other exported files; it is not self-hashed."}
    write_json(output_dir / "manifest.json", manifest)
    return {"status": STATUS, "output_dir": str(output_dir.resolve()), "files": list(files) + ["manifest.json"],
        "condition_count": len(rows), "unique_condition_count": manifest["unique_condition_count"],
        "comparison": comparison, "manifest_path": str((output_dir / "manifest.json").resolve()),
        "pdf_path": str((output_dir / "campaign_report.pdf").resolve()) if pdf else None}
