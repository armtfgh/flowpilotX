"""Coupled, explicit-reference gas/liquid calculations for exploratory screens.

Only one premixed substrate/catalyst liquid stream is implemented.  No kinetic
fit, mass-transfer prediction, pressure-drop correlation or efficacy estimate
is made.  Controller quantization is followed by full numerical recalculation.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .inventory import configuration_hash, finite_number, reactor_pressure, validate_apparatus

R_ML_BAR_PER_MMOL_K = 0.0831446261815324  # numerically L bar / mol K
TIME_BASES = {"liquid_only_space_time_min", "no_slip_mixture_transit_min"}


def _chemistry(chemistry: dict):
    c = finite_number(chemistry.get("substrate_concentration_M"), "substrate concentration", positive=True)
    loading = finite_number(chemistry.get("catalyst_loading_mol_percent"), "catalyst loading", nonnegative=True)
    if chemistry.get("additional_liquid_streams", []):
        raise ValueError("Additional liquid feeds require a dilution-aware extension; this solver accepts one premixed feed")
    return c, loading


def _calculate(profile: dict, chemistry: dict, q_liquid: float, q_gas_ref: float) -> dict:
    c, loading = _chemistry(chemistry)
    mfc, gas, coil = [profile[k] for k in ["mfc", "gas", "reactor"]]
    p = reactor_pressure(profile)
    t = profile["operating"]["temperature_C"] + 273.15
    molar_volume = R_ML_BAR_PER_MMOL_K * mfc["reference_temperature_K"] / mfc["reference_pressure_bar_abs"]
    finite_number(molar_volume, "calculated reference molar volume", positive=True)
    q_gas_reactor = q_gas_ref * mfc["reference_pressure_bar_abs"] / p * t / mfc["reference_temperature_K"]
    substrate = c * q_liquid
    total_gas = q_gas_ref / molar_volume
    oxygen = total_gas * gas["o2_mole_fraction"]
    v = coil["volume_mL"]
    values = {
        "quantity_status": "calculated_from_proposed_setpoints", "liquid_flow_mL_min": q_liquid,
        "gas_reference_flow_mL_min": q_gas_ref, "gas_reactor_flow_mL_min_estimate": q_gas_reactor,
        "substrate_mmol_min": substrate, "catalyst_mmol_min": substrate * loading / 100,
        "catalyst_concentration_mM": c * loading * 10, "total_gas_mmol_min": total_gas,
        "oxygen_mmol_min_supplied": oxygen, "oxygen_equivalents_supplied": oxygen / substrate,
        "reactor_pressure_bar_abs": p, "reactor_pressure_basis": profile["pressure"]["reactor_pressure_basis"],
        "pressure_value_status": "assumed_for_proposed_condition_from_declared_reference",
        "actual_condition_pressure_bar_abs": None,
        "measured_pressure_reference_observation_id": profile["pressure"].get("measured_pressure_observation_id"),
        "reference_molar_volume_mL_per_mmol": molar_volume,
        "liquid_only_space_time_min": v / q_liquid,
        "no_slip_mixture_transit_min": v / (q_liquid + q_gas_reactor),
        "inlet_reference_volume_index_min": v / (q_liquid + q_gas_ref),
        "actual_liquid_contact_time_min": None, "measured_liquid_holdup": None,
        "coil_length_from_volume_and_ID_m": v * 4 / (math.pi * coil["ID_mm"] ** 2),
        "gas_fraction_no_slip_inlet_estimate": q_gas_reactor / (q_liquid + q_gas_reactor),
        "limitations": ["O2 delivery is supply, not uptake or oxygen-atom equivalents.",
            "Mixture transit assumes ideal gas, constant inlet gas flow and no slip; dissolution/consumption/pressure gradients are not modeled.",
            "Liquid-only space time and inlet-reference volume index are bookkeeping indices, not measured liquid contact time.",
            "Coil length calculated from internal volume and ID assumes cylindrical tubing; illuminated length is separately recorded."]}
    for key, value in values.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f"Nonfinite calculated quantity: {key}")
    return values


def calculate_setpoints(profile: dict, chemistry: dict, liquid_flow_mL_min, gas_reference_flow_mL_min) -> dict:
    """Verify a pair of explicit controller settings and compute all indices."""
    issues = validate_apparatus(profile)
    if issues:
        raise ValueError("; ".join(issues))
    ql = finite_number(liquid_flow_mL_min, "liquid flow", positive=True)
    qg = finite_number(gas_reference_flow_mL_min, "gas flow", positive=True)
    for value, name in [(ql, "pump"), (qg, "mfc")]:
        c = profile[name]
        if not c["min_flow_mL_min"] - 1e-12 <= value <= c["max_flow_mL_min"] + 1e-12:
            raise ValueError(f"{name} flow outside confirmed usable range")
        n = (value - c["grid_origin_mL_min"]) / c["increment_mL_min"]
        if abs(n - round(n)) > 1e-7:
            raise ValueError(f"{name} setting is not on its confirmed controller grid")
    return _calculate(profile, chemistry, ql, qg)


def _indices(controller: dict, low: float, high: float):
    origin, increment = controller["grid_origin_mL_min"], controller["increment_mL_min"]
    low = max(low, controller["min_flow_mL_min"])
    high = min(high, controller["max_flow_mL_min"])
    return math.ceil((low - origin) / increment - 1e-10), math.floor((high - origin) / increment + 1e-10)


def solve_candidate(profile: dict, chemistry: dict, target: dict) -> dict:
    """Solve time and supplied O2 jointly, then search declared controller grids.

    ``target`` requires time_min, time_basis, oxygen_equivalents,
    time_relative_tolerance, oxygen_relative_tolerance, max_grid_evaluations.
    Every searched liquid-grid point and gas pair is retained. A finite bound
    that is exceeded fails closed, without silently returning a partial search.
    """
    record = {"candidate_id": target.get("candidate_id"), "target": deepcopy(target),
        "status": "rejected", "rejection_reasons": [], "ideal_unquantized": None,
        "calculated": None, "quantization_search": [], "experimental_validation": False,
        "grid_budget_basis": "liquid_controller_points_each_with_at_most_two_gas_pairs",
        "quantization_selection_method": "Search every potentially feasible liquid grid point; test floor/ceiling oxygen-ratio gas settings clamped to the feasible gas interval; select lowest observed maximum relative error, then summed error, then flows. This is a feasibility search, not a global error minimizer."}
    try:
        record["configuration_sha256"] = configuration_hash({"apparatus": profile, "chemistry": chemistry, "target": target})
    except (ValueError, TypeError) as exc:
        record["configuration_sha256"] = None
        record["rejection_reasons"] = [f"Non-JSON/nonfinite configuration: {exc}"]
        return record
    issues = validate_apparatus(profile)
    if issues:
        record["status"] = "blocked"
        record["rejection_reasons"] = issues
        return record
    try:
        concentration, _ = _chemistry(chemistry)
        tau = finite_number(target.get("time_min"), "target time", positive=True)
        equivalent = finite_number(target.get("oxygen_equivalents"), "target O2 equivalents", positive=True)
        tt = finite_number(target.get("time_relative_tolerance"), "time relative tolerance", nonnegative=True)
        et = finite_number(target.get("oxygen_relative_tolerance"), "oxygen relative tolerance", nonnegative=True)
        if max(tt, et) >= 1:
            raise ValueError("Relative tolerances must be less than 1")
        basis = target.get("time_basis")
        if basis not in TIME_BASES:
            raise ValueError("Unknown/mismatched time basis; use liquid_only_space_time_min or no_slip_mixture_transit_min")
        budget = target.get("max_grid_evaluations")
        if type(budget) is not int or budget <= 0:
            raise ValueError("max_grid_evaluations must be an explicit positive integer")
        mfc, pump, gas, coil = [profile[k] for k in ["mfc", "pump", "gas", "reactor"]]
        molar_volume = R_ML_BAR_PER_MMOL_K * mfc["reference_temperature_K"] / mfc["reference_pressure_bar_abs"]
        alpha = concentration * equivalent * molar_volume / gas["o2_mole_fraction"]
        k = mfc["reference_pressure_bar_abs"] / reactor_pressure(profile) * (profile["operating"]["temperature_C"] + 273.15) / mfc["reference_temperature_K"]
        factor = k if basis == "no_slip_mixture_transit_min" else 0
        ideal_ql = coil["volume_mL"] / tau / (1 + factor * alpha)
        ideal_qg = alpha * ideal_ql
        record["ideal_unquantized"] = {"liquid_flow_mL_min": ideal_ql, "gas_reference_flow_mL_min": ideal_qg,
            "status": "calculated_not_controller_settings"}
        # These are necessary bounds for any pair inside both target tolerances.
        ql_low = coil["volume_mL"] / (tau * (1 + tt)) / (1 + factor * alpha * (1 + et))
        ql_high = coil["volume_mL"] / (tau * (1 - tt)) / (1 + factor * alpha * (1 - et))
        first, last = _indices(pump, ql_low, ql_high)
        record["required_liquid_grid_points"] = max(0, last - first + 1)
        record["maximum_gas_pairs_if_searched"] = 2 * record["required_liquid_grid_points"]
        if last - first + 1 > budget:
            raise ValueError("Declared quantization search budget exceeded; increase it explicitly and preserve this rejected attempt")
        best = None
        for index in range(first, last + 1):
            ql = pump["grid_origin_mL_min"] + index * pump["increment_mL_min"]
            gl, gh = alpha * (1 - et) * ql, alpha * (1 + et) * ql
            if factor:
                gl = max(gl, (coil["volume_mL"] / (tau * (1 + tt)) - ql) / factor)
                gh = min(gh, (coil["volume_mL"] / (tau * (1 - tt)) - ql) / factor)
            gi, gj = _indices(mfc, gl, gh)
            trial = {"liquid_flow_mL_min": ql, "gas_interval_mL_min": [gl, gh], "tested_pairs": []}
            if gi > gj:
                trial["rejection_reason"] = "No usable MFC grid point satisfies both target tolerances at this liquid setting"
            else:
                desired = (alpha * ql - mfc["grid_origin_mL_min"]) / mfc["increment_mL_min"]
                # Nearest ratio settings inside the feasible interval; the full
                # interval is recorded, so no unsearched feasibility is implied.
                for gindex in sorted({max(gi, min(gj, math.floor(desired))), max(gi, min(gj, math.ceil(desired)))}):
                    qg = mfc["grid_origin_mL_min"] + gindex * mfc["increment_mL_min"]
                    calc = _calculate(profile, chemistry, ql, qg)
                    terr = abs(calc[basis] / tau - 1)
                    eerr = abs(calc["oxygen_equivalents_supplied"] / equivalent - 1)
                    score = (max(terr, eerr), terr + eerr, ql, qg)
                    accepted = terr <= tt + 1e-12 and eerr <= et + 1e-12
                    trial["tested_pairs"].append({"gas_reference_flow_mL_min": qg, "time_relative_error": terr,
                        "oxygen_relative_error": eerr, "accepted": accepted})
                    if accepted and (best is None or score < best[0]):
                        best = (score, calc, terr, eerr)
            record["quantization_search"].append(trial)
        if best is None:
            reasons = ["No controller-grid pair realizes the requested time/O2 combination within declared tolerances"]
            for ideal, name in [(ideal_ql, "pump"), (ideal_qg, "mfc")]:
                if ideal < profile[name]["min_flow_mL_min"]:
                    reasons.append(f"Ideal {name} flow is below confirmed usable minimum")
                if ideal > profile[name]["max_flow_mL_min"]:
                    reasons.append(f"Ideal {name} flow exceeds confirmed usable maximum")
            record["rejection_reasons"] = reasons
        else:
            record.update(status="engineering_feasible_requires_review", calculated=best[1],
                realized_time_relative_error=best[2], realized_oxygen_relative_error=best[3])
            record["setpoints"] = {"liquid_flow_mL_min": best[1]["liquid_flow_mL_min"],
                "gas_reference_flow_mL_min": best[1]["gas_reference_flow_mL_min"], "status": "proposed_not_actual"}
    except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError) as exc:
        record["rejection_reasons"].append(str(exc))
    return record
