"""Feed quantities and elapsed/material budgets, with explicit missing inputs."""
from __future__ import annotations

import math


def _number(name, value, zero=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


def feed_preparation(volume_mL, substrate_concentration_M, catalyst_loading_mol_percent, properties=None):
    """Single premixed stock; solvent is made up to final volume, not added volume.

    Molecular masses, purity and densities must be supplied with source strings.
    Missing properties leave masses/aliquot volumes unknown, not guessed.
    """
    c = _number("substrate concentration", substrate_concentration_M)
    loading = _number("catalyst loading", catalyst_loading_mol_percent)
    cat_c = c * loading / 100
    result = {"basis": "calculated from proposed stock recipe; preparation not performed",
              "final_stock_volume_mL": volume_mL, "substrate_concentration_M": c,
              "catalyst_loading_mol_percent": loading, "catalyst_concentration_M": cat_c,
              "substrate_mmol": None, "catalyst_mmol": None,
              "solvent_instruction": "Dissolve and make up to the final stock volume with the specified solvent after solubility confirmation.",
              "properties": properties or {}, "missing": []}
    if volume_mL is None:
        result["missing"].append("final_stock_volume_mL")
        return result
    v = _number("stock volume", volume_mL)
    result.update(substrate_mmol=c*v, catalyst_mmol=cat_c*v)
    for component in ("substrate", "catalyst"):
        prop = (properties or {}).get(component, {})
        result[f"{component}_mass_mg"] = None
        result[f"{component}_aliquot_mL"] = None
        if not prop.get("source") or prop.get("molecular_mass_g_mol") is None or prop.get("purity_fraction") is None:
            result["missing"].append(f"{component}: sourced molecular mass and purity for weighed quantity")
            continue
        mw = _number("molecular mass", prop["molecular_mass_g_mol"])
        purity = _number("purity", prop["purity_fraction"])
        if purity > 1:
            raise ValueError("purity fraction must not exceed 1")
        mass = result[f"{component}_mmol"] * mw / purity
        result[f"{component}_mass_mg"] = mass
        if prop.get("density_g_mL") is not None:
            density = _number("density", prop["density_g_mL"])
            result[f"{component}_aliquot_mL"] = mass / (1000*density)
    return result


def validate_budget_inputs(config):
    """Unknown capacities remain unknown; negative/nonfinite capacities reject."""
    for key in ("extra_stock_reserve_mL", "available_substrate_mmol", "available_catalyst_mmol"):
        if config.get(key) is not None: _number(key, config[key], zero=True)
    if config.get("final_stock_volume_mL") is not None:
        _number("final_stock_volume_mL", config["final_stock_volume_mL"])


def run_consumption(*, liquid_flow_mL_min, gas_reference_flow_mL_min,
                    substrate_concentration_M, catalyst_loading_mol_percent,
                    feed_mode, sampling):
    """Compute only explicit windows. Unknown durations cannot become zero.

    Continuous mode requires priming feed volume, equilibration/collection times,
    flush solvent volume, flush time and handling time. Sample-loop mode requires
    explicit total injected feed and carrier volumes and elapsed/gas-on times;
    no steady plateau, productivity or collection window is inferred.
    """
    required = {
        "continuous": ["priming_feed_mL", "equilibration_min", "collection_min", "flush_solvent_mL", "flush_min", "handling_min", "gas_on_min"],
        "sample_loop": ["injected_feed_mL", "carrier_solvent_mL", "flush_solvent_mL", "elapsed_min", "gas_on_min", "collection_window_definition"],
    }
    if feed_mode not in required:
        return {"status": "pending", "missing": ["feed_mode"], "feed_volume_mL": None, "elapsed_min": None}
    missing = [k for k in required[feed_mode] if sampling.get(k) is None]
    if liquid_flow_mL_min is None: missing.append("liquid_flow_mL_min")
    if gas_reference_flow_mL_min is None: missing.append("gas_reference_flow_mL_min")
    if missing:
        return {"status": "pending", "missing": missing, "feed_volume_mL": None, "elapsed_min": None}
    q = _number("liquid flow", liquid_flow_mL_min)
    gas = _number("gas flow", gas_reference_flow_mL_min)
    values = {k: _number(k, sampling[k], zero=True) for k in required[feed_mode] if k != "collection_window_definition"}
    if feed_mode == "continuous":
        _number("collection_min", values["collection_min"])
        feed = values["priming_feed_mL"] + q*(values["equilibration_min"]+values["collection_min"])
        elapsed = values["priming_feed_mL"]/q + values["equilibration_min"] + values["collection_min"] + values["flush_min"] + values["handling_min"]
        collection = q*values["collection_min"]
        carrier = 0.0
    else:
        if not str(sampling["collection_window_definition"]).strip():
            raise ValueError("sample-loop collection window must be explicitly defined")
        _number("injected_feed_mL", values["injected_feed_mL"])
        _number("elapsed_min", values["elapsed_min"])
        feed, carrier, elapsed, collection = values["injected_feed_mL"], values["carrier_solvent_mL"], values["elapsed_min"], None
    if values["gas_on_min"] > elapsed:
        raise ValueError("gas-on duration exceeds complete run elapsed duration")
    c = _number("concentration", substrate_concentration_M)
    loading = _number("catalyst loading", catalyst_loading_mol_percent)
    return {"status": "calculated_from_explicit_windows", "feed_mode": feed_mode,
            "feed_volume_mL": feed, "collected_liquid_mL_estimate": collection,
            "additional_carrier_solvent_mL": carrier, "flush_solvent_mL": values["flush_solvent_mL"],
            "substrate_mmol": c*feed, "catalyst_mmol": c*feed*loading/100,
            "gas_reference_volume_mL": gas*values["gas_on_min"], "elapsed_min": elapsed,
            "note": "Stock solution volume is not neat solvent volume. Gas volume uses the declared MFC reference state. Sampling representativeness remains unvalidated."}
