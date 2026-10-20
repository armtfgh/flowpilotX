"""Inventory-bounded, explicitly conditional common pool for prospective arms.

This module deliberately does not read optimization history, use batch times,
fit kinetics, or mark apparatus assignments as confirmed. It proposes one
catalog-supported scenario and calculates controller settings under visible
gas-reference assumptions. Laboratory release remains a separate decision.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

from .candidate_space import R_ML_BAR_PER_MMOL_K
from .inventory import configuration_hash, finite_number

SCENARIO_ID = "conditional_air_uv150_pfa10_v1"
OXYGEN_LEVELS = [0.5, 1.0, 2.0, 4.0]
LIQUID_COORDINATE_COUNT = 7
OXYGEN_RELATIVE_TOLERANCE = 0.03


def _quantize(value: float, increment: float, *, floor=False) -> float:
    n = Decimal(str(value)) / Decimal(str(increment))
    return float(n.to_integral_value(rounding=ROUND_FLOOR if floor else ROUND_HALF_UP) * Decimal(str(increment)))


def _source(root: Path, relative: str) -> dict:
    path = root / relative
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _equipment(inventory: dict, category: str, equipment_id: str) -> dict:
    matches = [x for x in inventory[category] if x.get("equipment_id") == equipment_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one inventory record for {equipment_id}")
    value = deepcopy(matches[0])
    if value.get("service_status") != "available" or value.get("quantity", 0) < 1:
        raise ValueError(f"Source inventory does not report {equipment_id} as available")
    return value


def build_conditional_pool(root: str | Path) -> dict:
    """Build and retain all 7x4 targets and every complete 3x2 subset.

    Source inventory entries are copied as evidence, not assigned as actual.
    Eligible candidates meet the reported flow grids, geometry and temperature
    envelope under the declared assumptions; pressure suitability is unresolved.
    This function performs no writes, model calls or laboratory actions.
    """
    root = Path(root)
    source_paths = ["KHU_inventory_complete_20260918_v7.json", "KHU_inventory_complete_20260915.xlsx",
        "data/raw/inventory/source_review_20260915/khu_clarification.md",
        "data/raw/inventory/source_review_20260915/inventory_source_cells.md", "docs/INVENTORY_SOURCES.md"]
    sources = [_source(root, path) for path in source_paths]
    profile = json.loads((root / source_paths[0]).read_text())
    inventory = profile["lab_inventory"]
    chosen = {
        "pump": _equipment(inventory, "pumps", "pump_r_series_hplc"),
        "reactor": _equipment(inventory, "reactors", "coil_row5"),
        "light": _equipment(inventory, "light_sources", "light_uv150_row10"),
        "mfc": _equipment(inventory, "gas_hardware", "mfc_ffc00"),
        "mixer": _equipment(inventory, "mixers", "p-713"),
        "check_valve": _equipment(inventory, "safety_accessories", "cv-3301"),
    }
    pump, coil, light, mfc = [chosen[k] for k in ["pump", "reactor", "light", "mfc"]]
    for record, keys in [(pump, ["min_flow_rate_mL_min", "max_flow_rate_mL_min", "flow_rate_increment_mL_min"]),
        (mfc, ["min_flow_sccm", "max_flow_sccm", "flow_rate_increment_sccm"]),
        (coil, ["volume_mL", "ID_mm"]), (light, ["max_reactor_volume_mL"])]:
        for key in keys:
            finite_number(record[key], key, positive=True)
    if (coil["material"], coil["volume_mL"], coil["ID_mm"]) != ("PFA", 10.0, 1.0):
        raise ValueError("Selected scenario requires the source-reported PFA 10 mL, 1 mm coil; review inventory change explicitly")
    if "uv150" not in coil["photoreactor_module_ids"] or coil["volume_mL"] > light["max_reactor_volume_mL"]:
        raise ValueError("Selected coil is not compatible with the UV150 capacity")
    if "r_series" not in light["compatible_pump_platforms"] or light["wavelength_nm"] != 450.0:
        raise ValueError("Selected scenario requires the inventory R-series/450 nm module combination")
    if "air" not in {x.strip().lower() for x in mfc["gas"].split(",")}:
        raise ValueError("Source inventory does not list air among MFC supported gases")

    concentration, loading, temperature_C = 0.25, 3.0, 25.0
    reference_T, reference_P, oxygen_fraction = 273.15, 1.01325, 0.2095
    if not light["min_temperature_C"] <= temperature_C <= min(light["max_temperature_C"], coil["max_temperature_C"]):
        raise ValueError("Proposed temperature is outside source-reported module/coil limits")
    molar_volume = R_ML_BAR_PER_MMOL_K * reference_T / reference_P
    # The fastest coordinate is derived from controller ceilings at the lowest
    # supply level. The remaining coordinates span the pump's reported minimum.
    fastest_bound = min(pump["max_flow_rate_mL_min"],
        mfc["max_flow_sccm"] * oxygen_fraction / (concentration * OXYGEN_LEVELS[0] * molar_volume))
    fastest = _quantize(fastest_bound, pump["flow_rate_increment_mL_min"], floor=True)
    slowest = pump["min_flow_rate_mL_min"]
    if fastest < slowest:
        raise ValueError("Reported controllers cannot realize even the lowest oxygen level at minimum pump flow")
    flows = [_quantize(fastest * (slowest / fastest) ** (i / (LIQUID_COORDINATE_COUNT - 1)),
                       pump["flow_rate_increment_mL_min"]) for i in range(LIQUID_COORDINATE_COUNT)]
    if len(set(flows)) != LIQUID_COORDINATE_COUNT or min(flows) < slowest:
        raise ValueError("Controller quantization does not yield seven distinct usable liquid coordinates; review pool policy")

    assumptions = [
        {"id": "A_GAS_REFERENCE", "status": "assumed_unconfirmed", "temperature_K": reference_T,
         "pressure_bar_abs": reference_P, "explanation": "A specific interpretation of the inventory's STP label; actual MFC reference state must be confirmed."},
        {"id": "A_AIR_COMPOSITION", "status": "assumed_unconfirmed", "gas": "air", "oxygen_mole_fraction": oxygen_fraction,
         "explanation": "Declared dry-air planning composition; actual selected gas/calibration and composition remain unconfirmed."},
        {"id": "A_FLOW_GRID_ORIGIN", "status": "assumed_unconfirmed", "liquid_grid_origin_mL_min": 0.0,
         "gas_grid_origin_reference_mL_min": 0.0, "explanation": "Controller settings are treated as multiples of source-reported increments, consistent with the archived clarification."},
        {"id": "A_GAS_VOLUME_MODEL", "status": "screening_model_assumption", "model": "ideal gas, constant inlet gas molar flow, no slip",
         "explanation": "Dissolution, uptake, pressure gradients and phase slip are not modeled; this is not actual liquid contact time."},
        {"id": "A_PRESSURE_ILLUSTRATIONS", "status": "assumed_unconfirmed_not_selected_BPR", "pressure_bar_abs": [2.0, 4.0],
         "explanation": "Mathematical illustrations only. Neither is an assigned BPR or a pressure measurement."},
    ]
    scenario = {
        "scenario_id": SCENARIO_ID, "status": "proposed_conditional_scenario_not_assigned_or_released",
        "hardware_choices": {key: {"proposed_equipment_id": value["equipment_id"], "choice_status": "proposed",
            "source_reported_inventory_record": value, "actual_assignment": None} for key, value in chosen.items()},
        "choice_rationale": "One PFA 10 mL coil uses the reported maximum UV150 capacity with a listed R-series-compatible 450 nm module. This apparatus choice uses inventory constraints only, not optimization outcomes.",
        "proposed_temperature_C": temperature_C,
        "temperature_status": "proposed_at_source_reported_module_minimum_not_measured",
        "bpr_assignment": None, "reactor_pressure_bar_abs": None, "pressure_basis_confirmed": None,
        "source_reported_bpr_options": deepcopy(inventory["pressure_controllers"]),
        "source_reported_standard_reactor_connectors_available": inventory.get("standard_reactor_connectors_available"),
        "shared_recipe": {"status": "disclosed_common_recipe_information_not_optimization_outcomes",
            "substrate": "thioanisole", "solvent": "EtOH", "substrate_concentration_M": concentration,
            "catalyst": "acridine", "catalyst_loading_mol_percent": loading,
            "gas_planning_preference": "air", "additional_liquid_streams": []},
        "feed_calculations": {"quantity_status": "calculated_from_proposed_recipe",
            "substrate_mmol_per_mL_feed": concentration,
            "catalyst_mmol_per_mL_feed": concentration * loading / 100,
            "catalyst_concentration_mM": concentration * loading * 10,
            "basis": "Amounts scale with final feed volume. No total run volume or actual feed preparation is assumed."},
        "geometry": {"volume_mL": coil["volume_mL"], "ID_mm": coil["ID_mm"],
            "geometry_status": "source_reported_nominal_not_newly_measured",
            "calculated_total_tubing_length_m": 4 * coil["volume_mL"] / (math.pi * coil["ID_mm"] ** 2),
            "actual_illuminated_length_m": None, "actual_liquid_holdup": None, "measured_RTD": None},
        "light_reporting": {"source_reported_measured_wavelength_nm": light["wavelength_nm"],
            "source_reported_radiant_power_W": light["power_W"],
            "source_reported_irradiance_mW_cm2": light["intensity_mW_cm2"],
            "actual_power_setting": None, "current_photon_flux_calibration": None},
        "assumptions": assumptions,
        "unresolved_release_items": [
            "Actual selected gas/calibration, MFC reference temperature/pressure and current reliable operating range",
            "Selected BPR identity, pressure basis, and mapping between slide 20/40/100 psi and inventory 1.5/2.5/7 bar labels",
            "Current apparatus assignment, feed compatibility, temperature-control performance and illuminated geometry",
            "Pressure sensor identity/location; gas supply/regulator settings; check-valve and connection pressure ratings/headroom",
            "Continuous feed versus sample loop, carrier/loop volume if applicable, priming, collection and sample handling",
            "Available material, analytical capacity and agreed sulfone acceptance criterion"],
        "physical_checks_scope": "Flow grids, reported component availability, module/coil capacity and catalog temperature compatibility only. Pressure envelope and operating readiness are not verified.",
    }
    policy = {
        "policy_version": "conditional_pool_policy_v1", "independent_of_optimization_history": True,
        "optimization_history_files_read": [], "batch_time_used": False,
        "minimum_time_basis": "Derived only from chosen nominal reactor volume and maximum feasible quantized liquid flow at lowest O2 supply level",
        "coordinate_generation": "Seven logarithmically spaced liquid flows from MFC-limited fastest flow to reported pump minimum; half-up quantization to pump increment",
        "fastest_liquid_flow_bound_mL_min": fastest_bound, "liquid_flow_coordinates_mL_min": flows,
        "liquid_only_space_time_coordinates_min": [coil["volume_mL"] / q for q in flows],
        "oxygen_target_levels": deepcopy(OXYGEN_LEVELS),
        "oxygen_level_policy": "Prospective supply exploration at 0.5, 1, 2 and 4 mol O2/mol sulfide; no inferred uptake or yield prediction",
        "oxygen_relative_tolerance_after_quantization": OXYGEN_RELATIVE_TOLERANCE,
        "gas_quantization": "Nearest source-reported MFC increment; never clamp to minimum or maximum",
        "time_basis": "liquid_only_space_time_min",
        "screen_structure": "Select a complete three-liquid-coordinate by two-nominal-O2-level rectangle, then independently repeat a selected condition",
        "non_experimental_time_warning": "V/QL can span 13–1000 min here; these are liquid-only space-time coordinates, not measured two-phase contact times or required elapsed collection durations.",
    }
    config = {"sources": sources, "scenario": scenario, "generation_policy": policy}
    result = {"schema_version": "khu_conditional_candidate_pool_v1", "status": "conditional_planning_only_not_executable",
        **config, "configuration_sha256": configuration_hash(config), "candidates": [],
        "feasible_candidates": [], "grid_groups": [], "experimental_validation": False,
        "candidate_status_definition": "eligible_under_declared_assumptions does not mean apparatus-verified, released or experimentally validated"}
    lookup = {}
    for ti, ql in enumerate(flows, 1):
        for oi, equivalent in enumerate(OXYGEN_LEVELS, 1):
            cid = f"CP{(ti - 1) * len(OXYGEN_LEVELS) + oi:02d}"
            ideal_qg = concentration * ql * equivalent * molar_volume / oxygen_fraction
            qg = _quantize(ideal_qg, mfc["flow_rate_increment_sccm"])
            total_gas = qg / molar_volume
            oxygen = total_gas * oxygen_fraction
            substrate = concentration * ql
            actual_equivalent = oxygen / substrate
            error = abs(actual_equivalent / equivalent - 1)
            reasons = []
            if not pump["min_flow_rate_mL_min"] <= ql <= pump["max_flow_rate_mL_min"]:
                reasons.append("Quantized liquid setting lies outside source-reported pump range")
            if not mfc["min_flow_sccm"] <= qg <= mfc["max_flow_sccm"]:
                reasons.append("Quantized gas setting lies outside source-reported MFC range; no clamping was performed")
            if error > OXYGEN_RELATIVE_TOLERANCE:
                reasons.append("Quantized O2 delivery differs from nominal level by more than declared 3% tolerance")
            coefficient = qg * reference_P * (temperature_C + 273.15) / reference_T
            transit = {
                "status": "parametric_estimate_not_actual_contact_time",
                "pressure_variable": "P_abs_bar: positive absolute reactor pressure, unconfirmed",
                "gas_reactor_flow_formula": "gas_reactor_flow_mL_min = gas_volume_pressure_coefficient / P_abs_bar",
                "gas_volume_pressure_coefficient_mL_bar_min": coefficient,
                "mixture_transit_formula": "nominal_volume_mL / (liquid_flow_mL_min + gas_volume_pressure_coefficient / P_abs_bar)",
                "nominal_volume_mL": coil["volume_mL"], "liquid_flow_mL_min": ql,
                "illustrations": [{"pressure_bar_abs": pressure, "pressure_status": "assumed_illustration_not_selected_BPR",
                    "gas_reactor_flow_mL_min_estimate": coefficient / pressure,
                    "no_slip_mixture_transit_min_estimate": coil["volume_mL"] / (ql + coefficient / pressure)}
                    for pressure in [2.0, 4.0]],
            }
            candidate = {
                "candidate_id": cid, "scenario_id": SCENARIO_ID, "time_coordinate": f"T{ti}",
                "oxygen_coordinate": f"O{oi}", "nominal_oxygen_equivalents": equivalent,
                "status": "rejected" if reasons else "eligible_under_declared_assumptions",
                "rejection_reasons": reasons, "configuration_sha256": result["configuration_sha256"],
                "proposed_setpoints": {"status": "proposed_not_actual", "liquid_flow_mL_min": ql,
                    "gas_reference_flow_mL_min": qg, "gas_controller_display_label": "sccm under A_GAS_REFERENCE",
                    "temperature_C": temperature_C, "bpr_setting": None, "light_power_setting": None},
                "ideal_unquantized_gas_reference_flow_mL_min": ideal_qg,
                "calculated": {"quantity_status": "calculated_under_explicit_unconfirmed_assumptions",
                    "liquid_only_space_time_min": coil["volume_mL"] / ql,
                    "inlet_reference_volume_index_min": coil["volume_mL"] / (ql + qg),
                    "actual_liquid_contact_time_min": None, "mixture_transit": transit,
                    "substrate_mmol_min": substrate, "catalyst_mmol_min": substrate * loading / 100,
                    "total_gas_mmol_min": total_gas, "oxygen_mmol_min_supplied": oxygen,
                    "oxygen_equivalents_supplied": actual_equivalent, "oxygen_relative_quantization_error": error,
                    "feed_mL_per_elapsed_hour_at_continuous_setting": ql * 60,
                    "substrate_mmol_per_elapsed_hour_at_continuous_setting": substrate * 60,
                    "catalyst_mmol_per_elapsed_hour_at_continuous_setting": substrate * loading / 100 * 60,
                    "rate_basis_warning": "Scalable rates only. Feed duration, collection duration, steady state, priming and total consumption are not established."},
                "hypothesis_status": "No predicted chemical outcome; selected conditions would explore time and supplied oxygen",
            }
            result["candidates"].append(candidate)
            if not reasons:
                result["feasible_candidates"].append(cid)
                lookup[(ti, oi)] = cid
    for times in combinations(range(1, LIQUID_COORDINATE_COUNT + 1), 3):
        for oxygens in combinations(range(1, len(OXYGEN_LEVELS) + 1), 2):
            if all((ti, oi) in lookup for ti in times for oi in oxygens):
                result["grid_groups"].append({"group_id": f"GRID{len(result['grid_groups']) + 1:03d}",
                    "scenario_id": SCENARIO_ID, "candidate_ids": [lookup[(ti, oi)] for ti in times for oi in oxygens],
                    "time_coordinates": [f"T{ti}" for ti in times], "oxygen_coordinates": [f"O{oi}" for oi in oxygens],
                    "time_coordinate_values": [coil["volume_mL"] / flows[ti - 1] for ti in times],
                    "oxygen_coordinate_values": [OXYGEN_LEVELS[oi - 1] for oi in oxygens],
                    "time_basis": "liquid_only_space_time_min", "oxygen_values_basis": "nominal supply levels; realized equivalents differ by recorded controller quantization"})
    result["summary"] = {"requested_candidates": len(result["candidates"]),
        "eligible_conditional_candidates": len(result["feasible_candidates"]),
        "rejected_candidates": sum(c["status"] == "rejected" for c in result["candidates"]),
        "complete_three_by_two_groups": len(result["grid_groups"])}
    return result
