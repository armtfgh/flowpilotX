"""Explicit conditional gas/liquid screen; no models, history or lab control.

Unlike the archived liquid-throughput grid, the caller declares a short
exploratory *mixture-transit* window and a particular inventory BPR.  Inlet
oxygen equivalents use only the declared gas reference state. Reactor gas
volume is used separately for a nominal ideal-gas/no-slip transit estimate.
Nothing here confirms an installed assembly, pressure convention or calibration.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import json

from .candidate_space import R_ML_BAR_PER_MMOL_K
from .inventory import absolute_pressure, finite_number
from .provenance import digest, file_hash


SOURCES = ["KHU_inventory_complete_20260918_v7.json", "KHU_inventory_complete_20260915.xlsx",
           "data/raw/inventory/source_review_20260915/inventory_source_cells.md",
           "data/raw/inventory/source_review_20260915/khu_clarification.md", "docs/INVENTORY_SOURCES.md"]
ROLES = {"pump": ("pumps", "pump_id"), "mfc": ("gas_hardware", "mfc_id"),
         "coil": ("reactors", "coil_id"), "light": ("light_sources", "light_id"),
         "mixer": ("mixers", "mixer_id"), "check_valve": ("safety_accessories", "check_valve_id"),
         "bpr": ("pressure_controllers", "bpr_id")}


def _number(value, name, *, positive=False, nonnegative=False):
    return finite_number(value, name, positive=positive, nonnegative=nonnegative)


def _item(inventory, category, equipment_id):
    matches = [r for r in inventory[category] if r.get("equipment_id") == equipment_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one source inventory record for {equipment_id}")
    record = deepcopy(matches[0])
    if record.get("service_status") != "available" or record.get("quantity", 0) < 1:
        raise ValueError(f"Inventory does not report {equipment_id} available")
    return record


def _scenario(inventory, config):
    selected = {role: _item(inventory, category, config["selected_equipment"][key])
                for role, (category, key) in ROLES.items()}
    coil, light, pump, mfc, bpr = [selected[k] for k in ["coil", "light", "pump", "mfc", "bpr"]]
    a, chemistry = config["assumptions"], deepcopy(config["chemistry"])
    temperature = _number(config["operating"]["temperature_C"], "temperature")
    _number(temperature + 273.15, "absolute temperature", positive=True)
    for key in ["substrate", "solvent", "catalyst"]:
        if not chemistry.get(key):
            raise ValueError(f"Explicit chemistry {key} required")
    _number(chemistry["substrate_concentration_M"], "concentration", positive=True)
    _number(chemistry["catalyst_loading_mol_percent"], "catalyst loading", nonnegative=True)
    if chemistry.get("additional_liquid_streams"):
        raise ValueError("Only one premixed liquid feed is supported")
    for key in ["volume_mL", "ID_mm"]:
        _number(coil[key], key, positive=True)
    if light.get("module_id") not in coil.get("photoreactor_module_ids", []):
        raise ValueError("Selected coil is not declared compatible with the selected light module")
    if coil["volume_mL"] > light["max_reactor_volume_mL"]:
        raise ValueError("Selected coil exceeds light module volume capacity")
    if light.get("compatible_pump_platforms") and pump.get("platform_id") not in light["compatible_pump_platforms"]:
        raise ValueError("Selected pump platform is incompatible with the light module")
    if not light["min_temperature_C"] <= temperature <= min(light["max_temperature_C"], coil["max_temperature_C"]):
        raise ValueError("Proposed temperature is outside source-reported component limits")
    gas = str(a["gas_identity"]).strip().lower()
    if gas not in {g.strip().lower() for g in mfc["gas"].split(",")}:
        raise ValueError("Selected gas is not listed for this MFC")
    x = _number(a["oxygen_mole_fraction"], "oxygen mole fraction", positive=True)
    if x > 1 or (gas == "air" and not .19 <= x <= .22) or (gas in {"o2", "oxygen"} and x != 1):
        raise ValueError("Declared oxygen fraction is inconsistent with air/pure-O2 identity")
    if gas not in {"air", "o2", "oxygen"}:
        raise ValueError("This oxidation screen requires an explicit oxygen-bearing gas identity")
    reference = {"temperature_K": _number(a["gas_reference_temperature_K"], "gas reference temperature", positive=True),
                 "pressure_bar_abs": _number(a["gas_reference_pressure_bar_abs"], "gas reference pressure", positive=True),
                 "oxygen_mole_fraction": x, "gas_identity": gas, "status": "assumed_unconfirmed",
                 "source_display_label": "sccm (STP), workbook MFC!C3",
                 "source_exact_reference_temperature_K": None, "source_exact_reference_pressure_bar_abs": None,
                 "actual_calibration": None}
    if bpr["type"] != "fixed" or len(bpr["setpoints_bar"]) != 1:
        raise ValueError("This corrected screen requires a fixed BPR with one source-reported setting")
    basis = a["bpr_label_basis"]
    ambient = _number(a["ambient_pressure_bar_abs"], "ambient pressure assumption", positive=True)
    pressure = absolute_pressure(bpr["setpoints_bar"][0], "bar", basis, ambient)
    if a["pressure_drop_model"] != "uniform_BPR_setpoint_approximation":
        raise ValueError("Unknown pressure model; no measured pressure-drop relation is implemented")
    controllers = {}
    for role, keys in [("pump", ("min_flow_rate_mL_min", "max_flow_rate_mL_min", "flow_rate_increment_mL_min")),
                       ("mfc", ("min_flow_sccm", "max_flow_sccm", "flow_rate_increment_sccm"))]:
        low, high, increment = [_number(selected[role][key], f"{role} {key}", positive=True) for key in keys]
        if high < low or math.ceil(low / increment - 1e-10) > math.floor(high / increment + 1e-10):
            raise ValueError(f"{role} has no usable grid point")
        controllers[role] = {"min": low, "max": high, "increment": increment, "origin": 0.,
                             "status": "source_reported_range_and_increment_not_accuracy",
                             "grid_basis": "Multiples of reported increment, September clarification item 4"}
    # Pressure checks remain partial: unknown rating bases cannot be certified.
    # A requirement above even the gauge interpretation is unequivocally outside
    # the supplied rating; lower values merely survive this conditional screen.
    cracking = _number(selected["check_valve"]["cracking_pressure_bar"], "check-valve cracking pressure", nonnegative=True)
    rating_checks = []
    for role, requirement in [("pump", pressure), ("coil", pressure), ("mixer", pressure), ("mfc", pressure + cracking)]:
        rating = selected[role].get("max_pressure_bar")
        if rating is None:
            rating_checks.append({"role": role, "status": "unknown_rating", "required_nominal_pressure_bar_abs": requirement})
            continue
        rating = _number(rating, f"{role} pressure rating", positive=True)
        if requirement > rating + ambient:
            raise ValueError(f"Nominal {role} pressure exceeds both possible absolute/gauge interpretations of source rating")
        rating_checks.append({"role": role, "source_rating_bar": rating, "source_rating_basis": None,
                              "required_nominal_pressure_bar_abs": requirement,
                              "status": "conditional_numeric_check_only" if requirement <= rating else "rating_basis_dependent"})
    return {"status": "proposed_conditional_not_assigned", "selected_equipment": {
                role: {**record, "selection_status": "proposed", "actual_assignment": None}
                for role, record in selected.items()},
            "chemistry": chemistry, "temperature_C": temperature, "reactor_volume_mL": coil["volume_mL"],
            "reactor_ID_mm": coil["ID_mm"], "gas_reference": reference, "controllers": controllers,
            "pressure": {"bpr_equipment_id": bpr["equipment_id"], "label_value_bar": bpr["setpoints_bar"][0],
                         "label_basis_source": None, "label_basis_assumption": basis,
                         "ambient_pressure_bar_abs_assumed": ambient, "reactor_pressure_bar_abs_assumed": pressure,
                         "approximation": a["pressure_drop_model"], "actual_reactor_pressure_bar_abs": None,
                         "minimum_mfc_outlet_pressure_bar_abs_before_line_losses": pressure + cracking,
                         "check_valve_cracking_differential_bar_source_reported": cracking,
                         "rating_checks": rating_checks, "pressure_envelope_verified": False,
                         "actual_gas_supply_pressure_bar_abs": None, "required_mfc_differential_bar": None,
                         "sensor_assignment": None, "check_valve_pressure_rating_bar": selected["check_valve"].get("max_pressure_bar")},
            "actual_illuminated_volume_mL": None, "actual_light_power_setting": None,
            "actual_liquid_holdup_mL": None, "actual_liquid_residence_time_min": None}


def calculate_coupled_values(scenario: dict, liquid_flow_mL_min: float, gas_reference_flow_mL_min: float) -> dict:
    """Calculate supplied molar equivalents and separately expanded gas volume.

    This pure arithmetic function never promotes assumptions to confirmations.
    Controller-grid feasibility is assessed by the screen solver separately.
    """
    ql = _number(liquid_flow_mL_min, "liquid flow", positive=True)
    qref = _number(gas_reference_flow_mL_min, "reference gas flow", positive=True)
    ref, chem = scenario["gas_reference"], scenario["chemistry"]
    t0 = _number(ref["temperature_K"], "reference temperature", positive=True)
    p0 = _number(ref["pressure_bar_abs"], "reference pressure", positive=True)
    x = _number(ref["oxygen_mole_fraction"], "oxygen fraction", positive=True)
    if x > 1:
        raise ValueError("Oxygen mole fraction exceeds one")
    pressure = _number(scenario["pressure"]["reactor_pressure_bar_abs_assumed"], "reactor pressure", positive=True)
    temperature = _number(scenario["temperature_C"] + 273.15, "reactor absolute temperature", positive=True)
    volume = _number(scenario["reactor_volume_mL"], "reactor volume", positive=True)
    c = _number(chem["substrate_concentration_M"], "concentration", positive=True)
    loading = _number(chem["catalyst_loading_mol_percent"], "catalyst loading", nonnegative=True)
    gas = p0 * qref / (R_ML_BAR_PER_MMOL_K * t0)
    oxygen, substrate = x * gas, c * ql
    qgas = qref * p0 / pressure * temperature / t0
    values = {"status": "calculated_under_proposed_and_assumed_conditions",
              "liquid_flow_mL_min": ql, "gas_reference_flow_mL_min": qref,
              "oxygen_reference_volume_flow_mL_min": x * qref,
              "oxygen_reference_volume_meaning": "O2 component's partial volume at the same reference state; not an additional gas feed",
              "total_gas_mmol_min": gas, "oxygen_mmol_min_supplied": oxygen,
              "substrate_mmol_min": substrate, "catalyst_mmol_min": substrate * loading / 100,
              "catalyst_concentration_mM": c * loading * 10,
              "oxygen_equivalents_supplied": oxygen / substrate,
              "gas_reactor_flow_mL_min_estimate": qgas,
              "no_slip_mixture_transit_min": volume / (ql + qgas),
              "liquid_only_space_time_min": volume / ql,
              "inlet_reference_volume_index_min": volume / (ql + qref),
              "gas_fraction_no_slip_inlet_estimate": qgas / (ql + qgas),
              "reactor_pressure_bar_abs_assumed": pressure, "gas_reference_temperature_K_assumed": t0,
              "gas_reference_pressure_bar_abs_assumed": p0, "gas_oxygen_mole_fraction_assumed": x,
              "feed_mL_per_elapsed_hour": ql * 60,
              "substrate_mmol_per_elapsed_hour": substrate * 60,
              "catalyst_mmol_per_elapsed_hour": substrate * loading / 100 * 60,
              "actual_liquid_contact_time_min": None, "actual_elapsed_collection_time_min": None}
    if any(not math.isfinite(v) for v in values.values() if isinstance(v, (float, int))):
        raise ValueError("Nonfinite calculated result")
    return values


def _grid_interval(controller, low, high):
    increment = controller["increment"]
    low, high = max(low, controller["min"]), min(high, controller["max"])
    return math.ceil(low / increment - 1e-10), math.floor(high / increment + 1e-10)


def _solve(scenario, config, time, equivalent, condition_id):
    tt, et = config["tolerances"]["time_relative"], config["tolerances"]["oxygen_relative"]
    pump, mfc = scenario["controllers"]["pump"], scenario["controllers"]["mfc"]
    ref = scenario["gas_reference"]
    c, v = scenario["chemistry"]["substrate_concentration_M"], scenario["reactor_volume_mL"]
    alpha = c * equivalent * R_ML_BAR_PER_MMOL_K * ref["temperature_K"] / (ref["pressure_bar_abs"] * ref["oxygen_mole_fraction"])
    k = ref["pressure_bar_abs"] / scenario["pressure"]["reactor_pressure_bar_abs_assumed"] * (scenario["temperature_C"] + 273.15) / ref["temperature_K"]
    ideal_ql = v / time / (1 + k * alpha)
    ideal_qg = alpha * ideal_ql
    qlow = v / (time * (1 + tt)) / (1 + k * alpha * (1 + et))
    qhigh = v / (time * (1 - tt)) / (1 + k * alpha * (1 - et))
    if any(not math.isfinite(x) for x in [alpha, k, ideal_ql, ideal_qg, qlow, qhigh]):
        raise ValueError("Nonfinite coupled target calculation")
    first, last = _grid_interval(pump, qlow, qhigh)
    result = {"condition_id": condition_id, "candidate_id": condition_id, "status": "rejected",
              "target": {"nominal_mixture_transit_min": time, "inlet_oxygen_equivalents": equivalent},
              "ideal_unquantized": {"liquid_flow_mL_min": ideal_ql, "gas_reference_flow_mL_min": ideal_qg},
              "proposed_setpoints": None, "calculated": None, "rejection_reasons": [],
              "quantization_search": [], "admissible_liquid_flow_interval_mL_min": [qlow, qhigh],
              "actual": {key: None for key in ["liquid_flow_mL_min", "gas_reference_flow_mL_min",
                         "gas_reference_temperature_K", "gas_reference_pressure_bar_abs", "temperature_C",
                         "bpr_equipment_id", "bpr_label_basis", "reactor_pressure_bar_abs", "liquid_holdup_mL",
                         "liquid_residence_time_min", "elapsed_collection_time_min", "residual_sulfide_percent",
                         "sulfoxide_percent", "sulfone_percent", "analytical_method"]}}
    required_pairs = 0
    for index in range(first, last + 1):
        ql = round(index * pump["increment"], 12)
        gl = max(alpha * (1 - et) * ql, (v / (time * (1 + tt)) - ql) / k)
        gh = min(alpha * (1 + et) * ql, (v / (time * (1 - tt)) - ql) / k)
        gi, gj = _grid_interval(mfc, gl, gh)
        count = max(0, gj - gi + 1)
        required_pairs += count
        result["quantization_search"].append({"liquid_flow_mL_min": ql,
              "gas_feasible_interval_reference_mL_min": [gl, gh], "gas_grid_first_index": gi,
              "gas_grid_last_index": gj, "pair_count": count, "tested_pair_count": 0,
              "accepted_pair_count": 0, "best_pair": None,
              "rejection_reason": "No gas controller point satisfies both tolerances" if not count else None})
    result["required_pair_evaluations"] = required_pairs
    if required_pairs > config["max_grid_pair_evaluations"]:
        result["rejection_reasons"] = ["Declared grid-pair evaluation budget exceeded; no partial selection returned"]
        return result
    best = None
    for trial in result["quantization_search"]:
        ql = trial["liquid_flow_mL_min"]
        trial_best = None
        for gi in range(trial["gas_grid_first_index"], trial["gas_grid_last_index"] + 1):
            qg = round(gi * mfc["increment"], 12)
            calc = calculate_coupled_values(scenario, ql, qg)
            terr, eerr = abs(calc["no_slip_mixture_transit_min"] / time - 1), abs(calc["oxygen_equivalents_supplied"] / equivalent - 1)
            trial["tested_pair_count"] += 1
            if terr > tt + 1e-12 or eerr > et + 1e-12:
                continue
            trial["accepted_pair_count"] += 1
            score = (max(terr, eerr), terr + eerr, ql, qg)
            if trial_best is None or score < trial_best[0]:
                trial_best = (score, calc, terr, eerr)
            if best is None or score < best[0]:
                best = (score, calc, terr, eerr)
        if trial_best:
            trial["best_pair"] = {"gas_reference_flow_mL_min": trial_best[1]["gas_reference_flow_mL_min"],
                                  "time_relative_error": trial_best[2], "oxygen_relative_error": trial_best[3]}
    if best is None:
        result["rejection_reasons"] = ["No controller-grid pair meets both declared target tolerances"]
        for role, ideal in [("pump", ideal_ql), ("mfc", ideal_qg)]:
            controller = scenario["controllers"][role]
            if ideal < controller["min"]:
                result["rejection_reasons"].append(f"Ideal {role} flow below source-reported minimum")
            if ideal > controller["max"]:
                result["rejection_reasons"].append(f"Ideal {role} flow exceeds source-reported maximum")
        return result
    pressure = scenario["pressure"]
    result.update(status="conditional_proposal", calculated=best[1], realized_time_relative_error=best[2],
                  realized_oxygen_relative_error=best[3], proposed_setpoints={
                      "status": "proposed_not_actual", "liquid_flow_mL_min": best[1]["liquid_flow_mL_min"],
                      "gas_reference_flow_mL_min": best[1]["gas_reference_flow_mL_min"], "temperature_C": scenario["temperature_C"],
                      "bpr_equipment_id": pressure["bpr_equipment_id"], "bpr_label_value_bar": pressure["label_value_bar"],
                      "bpr_label_basis_assumption": pressure["label_basis_assumption"]})
    return result


def _inventory_review(inventory, scenario):
    selected = {x["equipment_id"] for x in scenario["selected_equipment"].values()}
    light = scenario["selected_equipment"]["light"]
    review = {}
    for category, records in inventory.items():
        if not isinstance(records, list):
            continue
        entries = []
        for record in records:
            if not isinstance(record, dict) or "equipment_id" not in record:
                continue
            eid = record["equipment_id"]
            disposition, reason = "not_selected", "Catalog alternative retained; not evidence of chemical inferiority"
            if eid in selected:
                disposition, reason = "proposed_selection", "Explicit configured choice; actual assignment remains unconfirmed"
            elif category == "reactors":
                if record.get("volume_mL", 0) > light["max_reactor_volume_mL"]:
                    disposition, reason = "outside_selected_module_capacity", "Volume exceeds selected module capacity, regardless of generic compatibility listing"
                elif light["module_id"] not in record.get("photoreactor_module_ids", []):
                    disposition, reason = "not_declared_compatible", "Selected module is absent from coil compatibility list"
                else:
                    reason = "Compatible stock option; geometry/material/photon exposure would need a separate comparison"
            elif category == "light_sources":
                reason = "Alternative wavelength/module/temperature envelope changes the shared blue-light scenario"
            elif category == "pressure_controllers":
                reason = "Alternative pressure changes gas expansion and supply headroom; source pressure basis is unspecified"
            elif category == "pumps":
                reason = "Alternative platform/controller/pressure range; selected R-series follows study setup and avoids mixed-platform assumptions"
            entries.append({"equipment_id": eid, "name": record.get("name"), "disposition": disposition,
                            "reason": reason, "source_reported_record": deepcopy(record)})
        review[category] = entries
    return {"scope": "All inventory categories retained and reviewed against the configured scenario; no claim that the chosen apparatus is uniquely possible",
            "categories": review, "empty_categories_are_unknown_assignments": [k for k, v in review.items() if not v]}


def build_corrected_design(root: str | Path, config: dict) -> dict:
    """Return six explicit coupled proposals plus an independent-repeat record.

    Required keys are documented by the test fixture. There are no defaults for
    time targets, coil, BPR, pressure/reference conventions or tolerances.
    Computational attempts are retained as reproducible grid intervals/counts
    and their best pairs, rather than dumping a large Cartesian product.
    """
    root, config = Path(root), deepcopy(config)
    config_hash = digest(config)  # rejects NaN/Infinity before any calculation
    if not config.get("design_id") or config.get("version") is None:
        raise ValueError("Explicit design identity and version required")
    times = [_number(x, "mixture transit target", positive=True) for x in config["target_mixture_transit_min"]]
    oxygens = [_number(x, "inlet oxygen equivalents target", positive=True) for x in config["target_inlet_oxygen_equivalents"]]
    if len(times) != 3 or len(set(times)) != 3 or len(oxygens) != 2 or len(set(oxygens)) != 2:
        raise ValueError("Exactly three distinct transit targets and two distinct oxygen targets required")
    for key in ["time_relative", "oxygen_relative"]:
        if not 0 <= _number(config["tolerances"][key], key, nonnegative=True) < 1:
            raise ValueError("Tolerances must be finite and in [0,1)")
    if type(config["max_grid_pair_evaluations"]) is not int or config["max_grid_pair_evaluations"] <= 0:
        raise ValueError("Positive integer max_grid_pair_evaluations required")
    for key in ["time_window", "coil_choice", "bpr_choice", "oxygen_levels"]:
        if not str(config["rationale"].get(key, "")).strip():
            raise ValueError(f"Explicit {key} rationale required")
    sources = {path: file_hash(root / path) for path in SOURCES}
    profile = json.loads((root / SOURCES[0]).read_text())
    inventory = profile["lab_inventory"]
    scenario = _scenario(inventory, config)
    conditions = [_solve(scenario, config, time, eq, f"KD{i * 2 + j + 1:02d}")
                  for i, time in enumerate(times) for j, eq in enumerate(oxygens)]
    repeat_id = config["repeat_condition_id"]
    if repeat_id not in {c["condition_id"] for c in conditions}:
        raise ValueError("Independent repeat must refer to one of the six conditions")
    alternatives = []
    for alternate_id in ["coil_row3", "coil_row5"]:
        if alternate_id == config["selected_equipment"]["coil_id"]:
            continue
        alternative_config = deepcopy(config)
        alternative_config["selected_equipment"]["coil_id"] = alternate_id
        alternate_scenario = _scenario(inventory, alternative_config)
        attempts = [_solve(alternate_scenario, alternative_config, t, e, f"ALT-{alternate_id}-{i * 2 + j + 1:02d}")
                    for i, t in enumerate(times) for j, e in enumerate(oxygens)]
        alternatives.append({"equipment_id": alternate_id, "configuration_sha256": digest(alternative_config),
                             "configuration": alternative_config, "scenario": alternate_scenario,
                             "status": "all_targets_conditionally_feasible" if all(c["calculated"] for c in attempts) else "cannot_meet_all_targets_on_declared_grid",
                             "conditions": attempts, "interpretation": "Comparison under the same targets and assumptions only; not a general rejection of this coil"})
    feasible = all(c["calculated"] for c in conditions)
    realized = [(c["calculated"]["liquid_flow_mL_min"], c["calculated"]["gas_reference_flow_mL_min"]) for c in conditions if c["calculated"]]
    if feasible and len(set(realized)) != 6:
        raise ValueError("Distinct design targets collapsed to duplicate controller settings")
    return {"schema_version": "khu_corrected_coupled_design_v1", "design_id": config["design_id"],
            "status": "conditional_proposals_ready" if feasible else "infeasible_requested_design",
            "configuration": config, "configuration_sha256": config_hash, "source_hashes": sources,
            "source_inventory_profile_id": profile.get("profile_id"), "scenario": scenario,
            "source_operating_constraints": deepcopy(profile.get("operating_constraints", {})),
            "inventory_review": _inventory_review(inventory, scenario), "conditions": conditions,
            "independent_repeat": {"condition_id": "KD-R01", "repeat_of": repeat_id,
                                   "status": "proposed_independent_repeat" if feasible else "blocked_by_infeasible_design",
                                   "preparation": "independently re-establish preparation and operation", "execution_order": None},
            "alternative_evaluations": alternatives,
            "calculation_basis": {"R": R_ML_BAR_PER_MMOL_K, "R_units": "mL bar / (mmol K)",
                "equations": {"substrate_mmol_min": "C_M * Q_L_mL_min",
                    "total_gas_mmol_min": "P_ref_bar_abs * Q_ref_mL_min / (R * T_ref_K)",
                    "oxygen_equivalents_supplied": "xO2 * total_gas_mmol_min / substrate_mmol_min",
                    "oxygen_reference_volume_flow": "xO2 * Q_ref (component volume, not additional feed)",
                    "reactor_gas_flow": "Q_ref * (P_ref / P_reactor_abs) * (T_reactor_K / T_ref_K)",
                    "nominal_mixture_transit_min": "V_nominal / (Q_L + Q_gas_reactor)",
                    "liquid_only_space_time_min": "V_nominal / Q_L; bookkeeping index, not residence time",
                    "joint_ideal_solution": "alpha=C*O2eq*R*T_ref/(P_ref*xO2); k=(P_ref/P_abs)*(T_rxn/T_ref); QL=V/[tau*(1+k*alpha)]; Qref=alpha*QL"},
                "oxygen_pressure_independence": "Changing reactor pressure alone does not alter inlet reference molar flow or supplied oxygen equivalents",
                "quantization": "Enumerate every controller pair in the analytic intersection of time and oxygen tolerance bounds. Recalculate after grid rounding; never clamp an infeasible ideal pair.",
                "selection_score": "Minimum of (maximum relative error, sum of relative errors, liquid flow, reference gas flow) over all admissible grid pairs",
                "search_archive": "Per-liquid admissible gas-index intervals, pair counts and best pair reconstruct every evaluated pair without a million-pair archive",
                "laboratory_readiness": "Conditional catalog arithmetic only; no confirmed pressure/calibration/assembly or measured residence"},
            "limitations": ["Transit window is an explicit exploratory engineering choice, not batch-derived kinetics, a validated optimum or a promised speedup.",
                "If all conditions retain substantial sulfide, the window has not established sufficient exposure; that does not establish failure of the chemistry. Consider a longer follow-up only in light of measured flow outcomes.",
                "Actual gas calibration and exact STP convention are unknown. Declared reference T/P and dry-air composition are assumptions.",
                "BPR identity is proposed; gauge interpretation and ambient pressure are assumptions, not documented source conventions.",
                "Nominal mixture transit assumes uniform BPR pressure, ideal gas, constant inlet gas flow and no slip; uptake, dissolution, solvent vapor and pressure gradients are not modeled.",
                "At equal nominal transit, differing gas delivery still changes liquid fraction, mass transfer and photon exposure; this does not isolate all oxygen effects.",
                "Pump/MFC resolution is not accuracy. Actual pressure, holdup, RTD, illumination, supply/regulator/sensor arrangement and check-valve rating remain unconfirmed.",
                "Feed mode, stabilization, collection and handling are unresolved; rate calculations do not specify total material demand or experimental duration.",
                "These are offline corrected proposals, not new model nominations. Preserve the original campaign and do not present these as a prospective history-informed versus history-omitted comparison."],
            "laboratory_released": False, "experimental_validation": False, "paid_model_calls": 0,
            "history_read_by_generator": False, "comparison_nominations": None, "condition_ranks": None}
