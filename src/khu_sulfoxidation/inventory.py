"""Study apparatus evidence and explicit assignments; no inherited settings.

An inventory listing is evidence of a possible resource, not confirmation of
the assembly used for this study.  All pressures used by the calculator are
absolute; an explicit ambient pressure is required to convert a gauge label.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

PSI_TO_BAR = 0.06894757293168


def configuration_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite_number(value, name: str, *, positive=False, nonnegative=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else " and nonnegative" if nonnegative else ""))
    return result


def absolute_pressure(value, unit: str, basis: str, ambient_bar_abs=None) -> float:
    pressure = finite_number(value, "pressure", nonnegative=True)
    if unit not in {"bar", "psi"}:
        raise ValueError("Pressure unit must be explicitly bar or psi")
    pressure *= PSI_TO_BAR if unit == "psi" else 1
    if basis == "gauge":
        pressure += finite_number(ambient_bar_abs, "ambient pressure", positive=True)
    elif basis != "absolute":
        raise ValueError("Pressure basis must be explicitly gauge or absolute")
    return finite_number(pressure, "absolute pressure", positive=True)


def _source(root: Path, path: Path, **extra) -> dict:
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **extra}


def reconcile_inventory(root: str | Path) -> dict:
    """Capture prior inventory and new-slide evidence without assigning hardware."""
    root = Path(root)
    old_path = root / "data/raw/inventory/KHU_inventory_20260915_v6.json"
    old = json.loads(old_path.read_text())["lab_inventory"]
    ppt = next((root / "data/raw/khu").glob("*.pptx"))
    wanted = {"pump_r_series_hplc", "mfc_ffc00", "light_uv150_row10", "cv-3301", "p-713", "p-632"}
    catalog = [{"category": category, "record": deepcopy(record), "assignment_status": "unassigned_prior_inventory_reference"}
               for category, records in old.items() if isinstance(records, list)
               for record in records if isinstance(record, dict)
               and (record.get("equipment_id") in wanted or category in {"reactors", "pressure_controllers"})]
    def pending(**fields):
        return {"status": "unconfirmed", "confirmation_source": None, **fields}
    profile = {
        "schema_version": "khu_apparatus_v1", "profile_id": "khu_homogeneous_pending_v1",
        "status": "blocked_pending_lab_confirmation", "confirmation_source": None,
        "scope": "homogeneous_single_premixed_liquid_feed",
        "sources": [_source(root, old_path), _source(root, root / "data/raw/inventory/Inventory_final.xlsx"),
                    _source(root, root / "data/raw/inventory/KHU_previous_study_v7.json", role="reference_only_no_settings_inherited"),
                    _source(root, ppt, inspected_slides=[5, 6, 7])],
        "prior_inventory_evidence": catalog,
        "new_slide_evidence": {"slide_5": {"pump": "Vapourtec R-series HPLC", "mfc": "Bronkhorst FF-C00",
            "connecting_tubing": "PFA 0.04 inch ID", "bpr_options_psi": [20, 40, 100],
            "bpr_nominal_bar_conversions": [n * PSI_TO_BAR for n in [20, 40, 100]], "pressure_basis": None},
            "slide_6": {"topology": ["premixed_liquid_feed", "pump", "gas_liquid_mixer", "illuminated_coil", "BPR", "collection"],
                "drawn_gas": "O2", "UV150_wavelengths_nm_as_labelled": [360, 380, 395, 420, 450, 470, 520],
                "manual_strip_nm_as_labelled": 454, "tubing_material_options": ["PFA", "ETFE", "FEP"],
                "tubing_diameters_inch_as_labelled": [0.04, 0.062, 0.093], "installed_coil_identity": None},
            "slide_7": {"scope": "heterogeneous_later_phase_excluded_from_homogeneous_assignment"}},
        "reconciliation_issues": [
            {"id": "gas_identity", "issue": "Slide draws O2; correspondence planning preference is air. Selected calibration and gas remain unconfirmed."},
            {"id": "bpr_labels", "issue": "New 20/40/100 psi labels convert to 1.378951/2.757903/6.894757 bar, not the old 1.5/2.5/7 bar entries. Do not merge or inherit 7 bar."},
            {"id": "temperature", "issue": "Prior UV150 range is 25–80 C; batch anchor is 24 C. Confirm achievable setpoint without silently changing temperature."},
            {"id": "mfc_reference", "issue": "Workbook MFC!C3 says 0.01–10 sccm (STP), but exact reference T/P, selected gas calibration and reliable current range require confirmation."},
            {"id": "pump_grid", "issue": "Workbook Pump!E5/F5/G5 reports 0.01/10/0.01 mL/min; usable minimum differs from increment. Confirm current channel and service."},
            {"id": "coil_assignment", "issue": "Prior list contains multiple coils; slide diameters are options, not an installed UV150 coil. Confirm physical coil, volume, ID, illuminated length and light compatibility."},
            {"id": "gas_accessory", "issue": "Prior Accessories!row3 calls for CV-3301 between MFC and mixer, cracking differential 1 bar; pressure rating and required supply headroom remain unknown."}],
        "pump": pending(id=None, available=None, min_flow_mL_min=None, max_flow_mL_min=None,
                        increment_mL_min=None, grid_origin_mL_min=None, max_pressure_bar_abs=None, max_pressure_exclusive=None),
        "mfc": pending(id=None, available=None, min_flow_mL_min=None, max_flow_mL_min=None,
            increment_mL_min=None, grid_origin_mL_min=None, display_unit=None, calibration_id=None,
            calibration_gas_identity=None, reference_temperature_K=None, reference_pressure_bar_abs=None,
            max_outlet_pressure_bar_abs=None, max_inlet_pressure_bar_abs=None, min_pressure_drop_bar=None),
        "gas": pending(identity=None, o2_mole_fraction=None, supply_pressure_bar_abs=None),
        "reactor": pending(id=None, available=None, material=None, volume_mL=None, volume_status=None,
            ID_mm=None, illuminated_length_m=None, illuminated_length_status=None,
            compatible_light_ids=None, max_pressure_bar_abs=None, max_pressure_exclusive=None,
            min_temperature_C=None, max_temperature_C=None, geometry_relative_tolerance=None),
        "light": pending(id=None, available=None, wavelength_nm=None, min_temperature_C=None,
                         max_temperature_C=None, max_reactor_volume_mL=None, power_setting=None),
        "pressure": pending(bpr_id=None, available_bpr_ids=None, available_bpr_settings=None, setpoint_value=None, unit=None, basis=None,
            ambient_bar_abs=None, reactor_pressure_basis=None, measured_reactor_pressure_bar_abs=None,
            approved_max_operating_pressure_bar_abs=None, pressure_drop_allowance_bar=None,
            sensor_id=None, sensor_location=None, sensor_max_pressure_bar_abs=None),
        "connections": pending(mixer_id=None, check_valve_id=None, check_valve_cracking_pressure_bar=None,
            minimum_component_rating_bar_abs=None, minimum_gas_inlet_component_rating_bar_abs=None,
            minimum_gas_outlet_component_rating_bar_abs=None, regulator_id=None, collection_arrangement=None),
        "operating": pending(temperature_C=None, feed_mode=None, sample_loop_volume_mL=None),
        "assumptions": ["Ideal gas conversion is a screening calculation, not gas uptake.",
            "Constant inlet gas flow and no phase slip are estimates; actual liquid contact time remains unknown.",
            "Pressure drop allowance is declared, not a calculated or measured two-phase pressure drop."]}
    new_inventory = root / "khu_inventory.json"
    if new_inventory.exists():
        content = json.loads(new_inventory.read_text())
        profile["sources"].append(_source(root, new_inventory, role="new_user_inventory_requires_schema_and_assignment_review"))
        profile["new_user_inventory_evidence"] = content
        if content.get("schema_version") == "flowpilot_inventory_profile_v3.0":
            catalog = content.get("lab_inventory", {})
            profile["new_inventory_review"] = {
                "profile_id": content.get("profile_id"), "source_updated_at": content.get("updated_at_utc"),
                "reviewed_schema": "flowpilot_inventory_profile_v3.0", "inherited_assignments": [],
                "r_series_candidates": [r for r in catalog.get("pumps", []) if "r-series" in r.get("name", "").lower()],
                "mfc_candidates": [r for r in catalog.get("gas_hardware", []) if r.get("type") == "MFC"],
                "available_mixer_records": catalog.get("mixers", []),
                "source_validation": content.get("validation"),
                "interpretation": "Newly supplied file has August metadata; retain source conflicts explicitly instead of treating file arrival as apparatus-setting confirmation."}
            profile["planning_preferences"] = {"status": "source_supported_options_not_actual_assignments",
                "pump_id": "KHU-PUMP-VAPOURTEC-R-SERIES", "light_id": "KHU-LIGHT-UV150-450",
                "source": "khu_inventory.json", "coil_id": None, "bpr_id": None, "gas_calibration_id": None}
            profile["reconciliation_issues"].extend([
                {"id": "new_inventory_gas", "issue": "New inventory MFC is explicitly O2-tagged. It does not confirm air calibration or exact gas reference T/P."},
                {"id": "new_inventory_mixer", "issue": "New inventory defines no mixer or check-valve records and warns that multiple-inlet designs are blocked. September workbook and new PPT name such components; physical assembly must be reconciled."},
                {"id": "new_inventory_geometry", "issue": "New inventory warns of 0.1 mm UV150 tubing text conflicting with 1.0 mm reactor table. Do not infer an installed coil from either statement."},
                {"id": "new_inventory_temperature", "issue": "New JSON describes UV150 35–80 C standard and 10–35 C with additional cooling, conflicting with September 25–80 C. Neither confirms 24 C operation in this assembly."},
                {"id": "new_inventory_bpr", "issue": "New inventory fixed 1.5/2.5/7/8/16 bar BPR list still differs from PPT 20/40/100 psi, and gives no gauge/absolute basis."}])
        profile["reconciliation_issues"].append({"id": "new_inventory_assignment_review", "issue":
            "New khu_inventory.json preserved verbatim as evidence. An inventory list alone does not establish selected installed equipment, calibration or operating settings; review its schema and explicit confirmations before assignment."})
    return profile


def reactor_pressure(profile: dict) -> float:
    p = profile["pressure"]
    nominal = absolute_pressure(p["setpoint_value"], p["unit"], p["basis"], p.get("ambient_bar_abs"))
    if p["reactor_pressure_basis"] == "setpoint_approximation":
        return nominal
    if p["reactor_pressure_basis"] == "measured":
        if not p.get("measured_pressure_observation_id"):
            raise ValueError("Measured pressure requires a traceable observation ID; a reference measurement is not a measurement of every proposed condition")
        return finite_number(p.get("measured_reactor_pressure_bar_abs"), "measured reactor pressure", positive=True)
    raise ValueError("reactor_pressure_basis must be measured or setpoint_approximation")


def validate_apparatus(profile: dict) -> list[str]:
    """Return all essential missing/invalid assignment details, never guess them."""
    issues = []
    if profile.get("status") != "confirmed_for_engineering" or not profile.get("confirmation_source"):
        issues.append("Apparatus profile requires current study confirmation and source; inventory availability alone is insufficient")
    if profile.get("scope") != "homogeneous_single_premixed_liquid_feed":
        issues.append("This apparatus validator supports only the homogeneous single premixed liquid feed scope")
    required = {
        "pump": ["id", "available", "min_flow_mL_min", "max_flow_mL_min", "increment_mL_min", "grid_origin_mL_min", "max_pressure_bar_abs", "max_pressure_exclusive"],
        "mfc": ["id", "available", "min_flow_mL_min", "max_flow_mL_min", "increment_mL_min", "grid_origin_mL_min", "display_unit", "calibration_id", "calibration_gas_identity", "reference_temperature_K", "reference_pressure_bar_abs", "max_outlet_pressure_bar_abs", "max_inlet_pressure_bar_abs", "min_pressure_drop_bar"],
        "gas": ["identity", "o2_mole_fraction", "supply_pressure_bar_abs"],
        "reactor": ["id", "available", "material", "volume_mL", "volume_status", "ID_mm", "illuminated_length_m", "illuminated_length_status", "compatible_light_ids", "max_pressure_bar_abs", "max_pressure_exclusive", "min_temperature_C", "max_temperature_C", "geometry_relative_tolerance"],
        "light": ["id", "available", "wavelength_nm", "min_temperature_C", "max_temperature_C", "max_reactor_volume_mL", "power_setting"],
        "pressure": ["bpr_id", "available_bpr_ids", "available_bpr_settings", "setpoint_value", "unit", "basis", "reactor_pressure_basis", "approved_max_operating_pressure_bar_abs", "pressure_drop_allowance_bar", "sensor_id", "sensor_location", "sensor_max_pressure_bar_abs"],
        "connections": ["mixer_id", "check_valve_id", "check_valve_cracking_pressure_bar", "minimum_component_rating_bar_abs", "minimum_gas_inlet_component_rating_bar_abs", "minimum_gas_outlet_component_rating_bar_abs", "regulator_id", "collection_arrangement"],
        "operating": ["temperature_C", "feed_mode"],
    }
    for section, keys in required.items():
        record = profile.get(section, {})
        if record.get("status") != "confirmed" or not record.get("confirmation_source"):
            issues.append(f"{section}: current study confirmation and source required")
        for key in keys:
            if record.get(key) is None or record.get(key) == "" or record.get(key) == []:
                issues.append(f"{section}.{key}: required")
        if "available" in keys and record.get("available") is not True:
            issues.append(f"{section}: assigned equipment must be available")
    if issues:
        return issues
    try:
        for section in ["pump", "mfc"]:
            c = profile[section]
            for key in ["min_flow_mL_min", "max_flow_mL_min", "increment_mL_min"]:
                finite_number(c[key], f"{section}.{key}", positive=True)
            finite_number(c["grid_origin_mL_min"], f"{section}.grid_origin_mL_min", nonnegative=True)
            if c["max_flow_mL_min"] < c["min_flow_mL_min"]:
                raise ValueError(f"{section}: usable maximum below minimum")
            first_grid = (c["min_flow_mL_min"] - c["grid_origin_mL_min"]) / c["increment_mL_min"]
            last_grid = (c["max_flow_mL_min"] - c["grid_origin_mL_min"]) / c["increment_mL_min"]
            finite_number(first_grid, f"{section} lower controller grid index")
            finite_number(last_grid, f"{section} upper controller grid index")
            if math.ceil(first_grid - 1e-10) > math.floor(last_grid + 1e-10):
                raise ValueError(f"{section}: no controller grid setting lies within its confirmed usable range")
        mfc, gas, coil, light = [profile[k] for k in ["mfc", "gas", "reactor", "light"]]
        if mfc["display_unit"] != "mL/min_at_reference":
            raise ValueError("MFC display units must be explicitly converted to mL/min_at_reference")
        if mfc["calibration_gas_identity"] != gas["identity"]:
            raise ValueError("MFC calibration gas identity differs from supplied gas")
        fraction = finite_number(gas["o2_mole_fraction"], "O2 mole fraction", positive=True)
        if fraction > 1 or (gas["identity"].lower() == "air" and not 0.19 <= fraction <= 0.21):
            raise ValueError("O2 fraction inconsistent with gas identity (enriched mixtures must be named explicitly)")
        if gas["identity"].lower() in {"n2", "nitrogen", "co2", "argon"}:
            raise ValueError("Supplied gas identity is not an oxygen-containing feed")
        if gas["identity"].lower() in {"o2", "oxygen", "pure_o2"} and fraction != 1.0:
            raise ValueError("Pure O2 identity requires unit fraction; name a declared purity/mixture explicitly")
        for section, keys in {"mfc": ["reference_temperature_K", "reference_pressure_bar_abs", "max_outlet_pressure_bar_abs", "max_inlet_pressure_bar_abs"],
            "gas": ["supply_pressure_bar_abs"], "reactor": ["volume_mL", "ID_mm", "illuminated_length_m", "max_pressure_bar_abs"],
            "light": ["wavelength_nm", "max_reactor_volume_mL"], "pump": ["max_pressure_bar_abs"],
            "pressure": ["approved_max_operating_pressure_bar_abs", "sensor_max_pressure_bar_abs"],
            "connections": ["minimum_component_rating_bar_abs", "minimum_gas_inlet_component_rating_bar_abs", "minimum_gas_outlet_component_rating_bar_abs"]}.items():
            for key in keys:
                finite_number(profile[section][key], f"{section}.{key}", positive=True)
        for section, key in [("mfc", "min_pressure_drop_bar"), ("pressure", "pressure_drop_allowance_bar"), ("connections", "check_valve_cracking_pressure_bar")]:
            finite_number(profile[section][key], f"{section}.{key}", nonnegative=True)
        if coil["volume_status"] not in {"measured", "nominal"} or coil["illuminated_length_status"] not in {"measured", "nominal", "calculated_from_volume_and_ID"}:
            raise ValueError("Coil geometry status must distinguish measured, nominal and calculated quantities")
        geometry_tolerance = finite_number(coil["geometry_relative_tolerance"], "geometry relative tolerance", nonnegative=True)
        if geometry_tolerance >= 1:
            raise ValueError("Geometry relative tolerance must be less than one")
        total_length = 4 * coil["volume_mL"] / (math.pi * coil["ID_mm"] ** 2)
        finite_number(total_length, "coil length calculated from volume and ID", positive=True)
        if coil["illuminated_length_m"] > total_length * (1 + geometry_tolerance):
            raise ValueError("Illuminated length exceeds total coil length calculated from volume and ID")
        if light["id"] not in coil["compatible_light_ids"] or coil["volume_mL"] > light["max_reactor_volume_mL"]:
            raise ValueError("Selected coil is incompatible with selected light/module capacity")
        t = finite_number(profile["operating"]["temperature_C"], "temperature_C")
        low = finite_number(light["min_temperature_C"], "minimum module temperature")
        high = finite_number(light["max_temperature_C"], "maximum module temperature")
        if not -273.15 < t or not low <= t <= high:
            raise ValueError("Temperature outside confirmed module range")
        coil_low = finite_number(coil["min_temperature_C"], "minimum reactor temperature")
        coil_high = finite_number(coil["max_temperature_C"], "maximum reactor temperature")
        if not coil_low <= t <= coil_high:
            raise ValueError("Temperature outside confirmed reactor-material range")
        p = reactor_pressure(profile)
        if profile["pressure"]["bpr_id"] not in profile["pressure"]["available_bpr_ids"]:
            raise ValueError("Selected BPR is not assigned as available")
        pressure = profile["pressure"]
        selected_label = absolute_pressure(pressure["setpoint_value"], pressure["unit"], pressure["basis"], pressure.get("ambient_bar_abs"))
        options = [o for o in pressure["available_bpr_settings"] if o.get("bpr_id") == pressure["bpr_id"]]
        if not any(abs(absolute_pressure(o["value"], o["unit"], o["basis"], pressure.get("ambient_bar_abs")) - selected_label) <= 1e-8 for o in options):
            raise ValueError("Selected pressure is not a confirmed setting for the assigned BPR identity")
        upstream = p + profile["pressure"]["pressure_drop_allowance_bar"]
        for part in [profile["pump"], coil]:
            if type(part["max_pressure_exclusive"]) is not bool:
                raise ValueError("Pressure limit exclusivity must be explicit boolean")
            if part["max_pressure_exclusive"] and upstream >= part["max_pressure_bar_abs"]:
                raise ValueError("Pressure reaches an exclusive equipment rating")
        for rating in [profile["pump"]["max_pressure_bar_abs"], coil["max_pressure_bar_abs"], profile["connections"]["minimum_component_rating_bar_abs"], profile["pressure"]["approved_max_operating_pressure_bar_abs"], profile["pressure"]["sensor_max_pressure_bar_abs"]]:
            if upstream > rating:
                raise ValueError("Reactor pressure plus declared drop allowance exceeds an assigned pressure limit")
        gas_outlet = upstream + profile["connections"]["check_valve_cracking_pressure_bar"]
        if gas_outlet > profile["connections"]["minimum_gas_outlet_component_rating_bar_abs"]:
            raise ValueError("Gas outlet pressure plus check-valve differential exceeds gas-side component rating")
        if gas["supply_pressure_bar_abs"] > profile["connections"]["minimum_gas_inlet_component_rating_bar_abs"]:
            raise ValueError("Gas supply pressure exceeds gas-inlet component rating")
        if gas["supply_pressure_bar_abs"] > mfc["max_inlet_pressure_bar_abs"]:
            raise ValueError("Gas supply pressure exceeds confirmed MFC inlet rating")
        if gas_outlet > mfc["max_outlet_pressure_bar_abs"] or gas_outlet + mfc["min_pressure_drop_bar"] > gas["supply_pressure_bar_abs"]:
            raise ValueError("Insufficient MFC outlet/supply pressure for check valve and declared pressure allowance")
        mode = profile["operating"]["feed_mode"]
        if mode not in {"continuous", "sample_loop"}:
            raise ValueError("Feed mode must be continuous or sample_loop")
        if mode == "sample_loop":
            finite_number(profile["operating"].get("sample_loop_volume_mL"), "sample loop volume", positive=True)
    except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError) as exc:
        issues.append(str(exc))
    return issues
