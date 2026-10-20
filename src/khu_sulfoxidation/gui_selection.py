"""Deterministic, local GUI proposals using the corrected coupled-flow solver.

No model, baseline provider, file write or equipment operation is invoked. Text
and historical outcomes are preserved as context, not interpreted into settings.
The returned pool is the complete bounded prospective search, not an exhaustive
enumeration of all chemically or mechanically possible experiments.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from .corrected_design import SOURCES, _scenario, _solve
from .inventory import finite_number
from .provenance import digest, file_hash


MAX_DESIGNS = 12
VERSION = "flowpilotx_local_selection_v1"
DEFAULT_CHEMISTRY = {"substrate": "thioanisole", "solvent": "EtOH", "concentration_M": .25,
                     "catalyst": "acridine", "catalyst_mol_percent": 3.}
DEFAULT_CONSTRAINTS = {"time_min": 2., "time_max": 10., "oxygen_min": .5, "oxygen_max": 1.,
                       "coil_mL": 5., "pressure_bar_gauge": 1.5, "temperature_C": 25.}
TEMPLATE = "configs/corrected_screen_2026-10-13_v1.json"


class GuiDesignError(ValueError):
    """Validation/infeasibility with an archiveable, JSON-compatible audit."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {"status": "invalid_request", "rejection_reasons": [message],
                                   "candidate_pool": []}


def _log_grid(low, high, count):
    if low == high:
        return [low]
    if count == 1:
        return [math.exp((math.log(low) + math.log(high)) / 2)]
    return [low if i == 0 else high if i == count - 1 else
            math.exp(math.log(low) + (math.log(high) - math.log(low)) * i / (count - 1))
            for i in range(count)]


def _distinct(values):
    return list(dict.fromkeys(values))


def _settings(row):
    return (row["calculated"]["liquid_flow_mL_min"], row["calculated"]["gas_reference_flow_mL_min"])


def _normalize(request):
    if not isinstance(request, dict):
        raise ValueError("Request must be a JSON object")
    digest(request)  # reject non-JSON and nonfinite inputs without changing them
    n = request.get("requested_count", 6)
    if type(n) is not int or not 1 <= n <= MAX_DESIGNS:
        raise ValueError("requested_count must be an integer from 1 to 12")
    repeat = request.get("include_repeat", True)
    if type(repeat) is not bool:
        raise ValueError("include_repeat must be a boolean")
    for key in ["protocol_text", "research_notes"]:
        if not isinstance(request.get(key, ""), str):
            raise ValueError(f"{key} must be text")
    chemistry = deepcopy(DEFAULT_CHEMISTRY)
    constraints = deepcopy(DEFAULT_CONSTRAINTS)
    for key, result in [("chemistry", chemistry), ("constraints", constraints)]:
        updates = request.get(key, {})
        if not isinstance(updates, dict) or set(updates) - set(result):
            raise ValueError(f"{key} has unsupported fields or is not an object")
        result.update(updates)
    identities = {"substrate": {"thioanisole", "methyl phenyl sulfide", "methyl phenyl sulphide"},
                  "solvent": {"etoh", "ethanol"}, "catalyst": {"acridine"}}
    for key, aliases in identities.items():
        if not isinstance(chemistry[key], str) or chemistry[key].strip().lower() not in aliases:
            raise ValueError("This local engineering adapter supports thioanisole / neutral acridine / EtOH only; "
                             "other chemistry requires a separate reviewed adapter")
        chemistry[key] = DEFAULT_CHEMISTRY[key]
    chemistry["concentration_M"] = finite_number(chemistry["concentration_M"], "concentration_M", positive=True)
    chemistry["catalyst_mol_percent"] = finite_number(chemistry["catalyst_mol_percent"], "catalyst_mol_percent", nonnegative=True)
    for key in constraints:
        constraints[key] = finite_number(constraints[key], key, positive=key != "temperature_C")
    for low, high in [("time_min", "time_max"), ("oxygen_min", "oxygen_max")]:
        if constraints[low] > constraints[high]:
            raise ValueError(f"{low} must not exceed {high}")
    summary = request.get("history_summary", {})
    if not isinstance(summary, dict):
        raise ValueError("history_summary must be an object")
    ids = summary.get("evidence_ids", [])
    if not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("history_summary.evidence_ids must be a list of nonempty strings")
    return n, repeat, chemistry, constraints, _distinct(ids)


def _config(root, chemistry, constraints):
    template = json.loads((root / TEMPLATE).read_text())
    expected_reference = {"gas_identity": "air", "oxygen_mole_fraction": .2095,
                          "gas_reference_temperature_K": 273.15, "gas_reference_pressure_bar_abs": 1.01325,
                          "bpr_label_basis": "gauge", "ambient_pressure_bar_abs": 1.01325}
    if any(template["assumptions"].get(k) != v for k, v in expected_reference.items()):
        raise ValueError("The GUI air/STP/gauge contract differs from its source configuration; explicit adapter review required")
    profile = json.loads((root / SOURCES[0]).read_text())
    inventory = profile["lab_inventory"]
    equipment = deepcopy(template["selected_equipment"])
    # Select an actual listed geometry, never fabricate a coil or BPR label.
    choices = [r for r in inventory["reactors"] if r.get("material") == "PFA" and r.get("ID_mm") == 1
               and r.get("volume_mL") == constraints["coil_mL"] and r.get("service_status") == "available"
               and r.get("quantity", 0) > 0]
    if len(choices) != 1:
        raise ValueError("coil_mL must identify one available PFA / 1 mm-ID inventory coil (2, 5 or 10 mL)")
    equipment["coil_id"] = choices[0]["equipment_id"]
    choices = [r for r in inventory["pressure_controllers"] if r.get("type") == "fixed"
               and r.get("setpoints_bar") == [constraints["pressure_bar_gauge"]]
               and r.get("service_status") == "available" and r.get("quantity", 0) > 0]
    if len(choices) != 1:
        raise ValueError("pressure_bar_gauge must match an available fixed inventory BPR label; "
                         "its gauge interpretation remains an assumption")
    equipment["bpr_id"] = choices[0]["equipment_id"]
    config = {"schema_version": VERSION, "selected_equipment": equipment,
              "chemistry": {"substrate": chemistry["substrate"], "solvent": chemistry["solvent"],
                  "substrate_concentration_M": chemistry["concentration_M"], "catalyst": chemistry["catalyst"],
                  "catalyst_loading_mol_percent": chemistry["catalyst_mol_percent"]},
              "operating": {"temperature_C": constraints["temperature_C"]},
              "constraints": constraints, "assumptions": deepcopy(template["assumptions"]),
              "tolerances": {"time_relative": .03, "oxygen_relative": .03},
              "max_grid_pair_evaluations": 100000}
    return config, _scenario(inventory, config), profile


def _selection(pool, primary_times, count, paired, middle_oxygen, low, high):
    eligible = [r for r in pool if r.get("calculated")]
    selected, used = [], set()
    groups = {}
    for row in eligible:
        groups.setdefault(row["target"]["nominal_mixture_transit_min"], []).append(row)
    def position(time):
        return 0.5 if low == high else (math.log(time) - math.log(low)) / (math.log(high) - math.log(low))
    def group_available(rows):
        keys = [_settings(r) for r in rows]
        return len(set(keys)) == len(keys) and not used.intersection(keys)
    if paired:
        endpoints = sorted({r["target"]["inlet_oxygen_equivalents"] for r in pool})
        groups = {t: [r for r in rows if r["target"]["inlet_oxygen_equivalents"] in (endpoints[0], endpoints[-1])]
                  for t, rows in groups.items()}
        groups = {t: rows for t, rows in groups.items() if len(rows) == 2}
        required = count // 2
        for time in primary_times:
            rows = groups.get(time, [])
            if len(selected) < required * 2 and rows and group_available(rows):
                selected.extend(rows)
                used.update(_settings(r) for r in rows)
        while len(selected) < required * 2:
            available = [(t, rows) for t, rows in groups.items() if group_available(rows)]
            if not available:
                break
            occupied = [position(r["target"]["nominal_mixture_transit_min"]) for r in selected]
            t, rows = min(available, key=lambda tr: (
                -min([abs(position(tr[0]) - x) for x in occupied] or [abs(position(tr[0]) - .5)]), tr[0]))
            selected.extend(rows)
            used.update(_settings(r) for r in rows)
        if len(selected) != required * 2:
            return []
    # Odd counts add one middle-oxygen probe; a fixed oxygen range is a 1-D screen.
    singles = [r for r in eligible if not paired or r["target"]["inlet_oxygen_equivalents"] == middle_oxygen]
    preferred = {time: i for i, time in enumerate(primary_times)}
    while len(selected) < count:
        available = [r for r in singles if _settings(r) not in used]
        if not available:
            break
        if paired or count == 1:
            key = lambda r: (abs(position(r["target"]["nominal_mixture_transit_min"]) - .5), r["candidate_id"])
        else:
            occupied = [position(r["target"]["nominal_mixture_transit_min"]) for r in selected]
            key = lambda r: (0 if r["target"]["nominal_mixture_transit_min"] in preferred else 1,
                             preferred.get(r["target"]["nominal_mixture_transit_min"], 100),
                             -min([abs(position(r["target"]["nominal_mixture_transit_min"]) - x) for x in occupied] or [0]),
                             r["candidate_id"])
        row = min(available, key=key)
        selected.append(row)
        used.add(_settings(row))
    return sorted(selected, key=lambda r: (r["target"]["nominal_mixture_transit_min"], r["target"]["inlet_oxygen_equivalents"]))


def _annotate_roles(conditions, repeat_id):
    """Explain each prospective comparison without inferring a response trend."""
    times = sorted({c["target_time_min"] for c in conditions})
    oxygen_levels = sorted({c["target_oxygen_eq"] for c in conditions})
    def location(value, values):
        return "fixed" if len(values) == 1 else "lower" if value == values[0] else "upper" if value == values[-1] else "interior"
    for row in conditions:
        time, oxygen = row["target_time_min"], row["target_oxygen_eq"]
        tpos, opos = location(time, times), location(oxygen, oxygen_levels)
        same_time = [c for c in conditions if c["target_time_min"] == time and c["id"] != row["id"]]
        paired = next((c for c in same_time if opos in {"lower", "upper"} and
                       c["target_oxygen_eq"] == oxygen_levels[-1 if opos == "lower" else 0]), None)
        companion = paired or min(same_time, key=lambda c: (abs(c["target_oxygen_eq"] - oxygen), c["id"]), default=None)
        same_oxygen = [c for c in conditions if c["target_oxygen_eq"] == oxygen and c["target_time_min"] != time]
        time_comparison = min(same_oxygen, key=lambda c: (abs(math.log(c["target_time_min"] / time)), c["target_time_min"]), default=None)
        if len(conditions) == 1:
            question = "Initial flow probe"
            purpose = "Establish one measured starting point; a single unique condition cannot map time or oxygen dependence."
        elif tpos == "fixed":
            question = {"lower": "Lower oxygen boundary", "upper": "Upper oxygen boundary", "interior": "Oxygen response"}[opos]
            purpose = "Cover this part of the selected oxygen range at one shared nominal transit target and inspect all three analytical components."
        elif opos == "fixed":
            question = {"lower": "Incomplete oxidation", "upper": "Longer exposure", "interior": "Exposure response"}[tpos]
            purpose = "Cover this part of the selected transit range at one shared inlet oxygen target and inspect residual sulfide, sulfoxide and sulfone."
        else:
            roles = {
                ("lower", "lower"): ("Incomplete oxidation", "Probe residual sulfide at the shortest selected transit and lower oxygen boundary; this establishes a low-exposure comparison point without assuming insufficient conversion."),
                ("lower", "upper"): ("Oxygen contrast", "Complete the short-transit oxygen contrast to assess whether the three-component outcome changes with the higher inlet supply."),
                ("interior", "lower"): ("Exposure response", "Add an interior transit coordinate on the lower-oxygen branch, so a trend need not be inferred from endpoints alone."),
                ("interior", "upper"): ("Middle oxygen contrast", "Add an interior coordinate on the higher-oxygen branch and complete the oxygen comparison away from the time boundaries."),
                ("upper", "lower"): ("Longer exposure", "Extend the lower-oxygen branch to the longest selected transit and inspect whether the component distribution differs from a shorter exposure."),
                ("upper", "upper"): ("Combined upper bound", "Cover the joint upper corner of the selected time and oxygen targets; inspect remaining sulfide, desired sulfoxide and possible sulfone without predicting which increases."),
            }
            question, purpose = roles.get((tpos, opos), ("Intermediate oxygen probe", "Add a distinct interior oxygen coordinate to check what a two-level contrast could miss."))
        if row["id"] == repeat_id and tpos == "interior" and opos == "upper":
            question = "Repeat reference"
        comparisons = []
        if companion:
            comparisons.append(f"Compare with {companion['id']} at the same {time:g} min nominal transit target and a different inlet O2 target.")
        if time_comparison:
            comparisons.append(f"Compare with {time_comparison['id']} at the same {oxygen:g} inlet O2 equivalents and a different nominal transit target.")
        if row["id"] == repeat_id:
            comparisons.append("The separate independent repeat checks local preparation, operation and analytical variability at this condition.")
        row.update(question=question, design_role=f"{tpos}_time_{opos}_oxygen",
                   companion_id=companion["id"] if companion else None,
                   paired_condition_id=paired["id"] if paired else None,
                   time_comparison_id=time_comparison["id"] if time_comparison else None,
                   rationale=(f"{time:g} min / {oxygen:g} inlet O2 equivalents: {purpose} " + " ".join(comparisons)
                              + " Flow adjustments also change phase fraction and transport, so these comparisons do not isolate a kinetic or oxygen-only causal effect. "
                              "Targets are configured engineering choices; historical rows remain shared review context, not condition-specific predictions."))


def generate_gui_designs(request: dict, project_root: Path) -> dict:
    """Return exactly N unique conditional designs, or raise GuiDesignError.

    Failure ``.details`` includes all attempted targets for the server's archive.
    Requested time/O2 bounds describe targets; achieved values may differ by the
    explicitly reported 3% quantization tolerance. Repeats never count toward N.
    """
    root = Path(project_root)
    audit = {"schema_version": VERSION, "status": "invalid_request", "candidate_pool": []}
    try:
        n, include_repeat, chemistry, bounds, history_ids = _normalize(request)
        audit["request_sha256"] = digest(request)
        config, scenario, profile = _config(root, chemistry, bounds)
        fixed_time = bounds["time_min"] == bounds["time_max"]
        paired = bounds["oxygen_min"] < bounds["oxygen_max"] and not fixed_time and n > 1
        primary_times = _log_grid(bounds["time_min"], bounds["time_max"], max(1, n // 2) if paired else n)
        if n == 6 and bounds["time_min"] == 2 and bounds["time_max"] == 10 and paired:
            primary_times = [2., 5., 10.]
        times = _distinct(primary_times + _log_grid(bounds["time_min"], bounds["time_max"], 25))
        middle = bounds["oxygen_min"] / 2 + bounds["oxygen_max"] / 2
        oxygens = _distinct([bounds["oxygen_min"], bounds["oxygen_max"]] + ([middle] if n % 2 else []))
        if n == 1:
            oxygens = [middle]
        elif fixed_time:
            oxygens = _distinct(_log_grid(bounds["oxygen_min"], bounds["oxygen_max"], n)
                                 + _log_grid(bounds["oxygen_min"], bounds["oxygen_max"], 25))
        config.update(requested_count=n, include_repeat=include_repeat, primary_time_targets=primary_times,
                      pool_time_targets=times, pool_oxygen_targets=oxygens,
                      selection_version=VERSION, range_semantics="target_bounds_with_3_percent_realized_tolerance")
        sources = {p: file_hash(root / p) for p in SOURCES + [TEMPLATE,
                   "src/khu_sulfoxidation/corrected_design.py", "src/khu_sulfoxidation/gui_selection.py"]}
        audit.update(status="searching", configuration=config, configuration_sha256=digest(config),
                     source_hashes=sources, scenario=scenario)
        for time in times:
            for oxygen in oxygens:
                cid = f"GP{len(audit['candidate_pool']) + 1:03d}"
                try:
                    row = _solve(scenario, config, time, oxygen, cid)
                except (ValueError, OverflowError, ZeroDivisionError) as exc:
                    row = {"candidate_id": cid, "condition_id": cid, "status": "rejected",
                           "target": {"nominal_mixture_transit_min": time, "inlet_oxygen_equivalents": oxygen},
                           "calculated": None, "rejection_reasons": [str(exc)]}
                row["selection_disposition"] = "not_selected" if row["calculated"] else "engineering_rejected"
                row["selection_reason"] = "Feasible proposal; no chemical ranking assigned" if row["calculated"] else "Target fails declared engineering constraints"
                audit["candidate_pool"].append(row)
        selected = _selection(audit["candidate_pool"], primary_times, n, paired, middle,
                              bounds["time_min"], bounds["time_max"])
        if len(selected) != n:
            message = (f"Cannot form {n} unique designs within the declared candidate pool, controller grids and 3% tolerances"
                       + (" while retaining complete oxygen-endpoint pairs" if paired else "") + ". No partial screen returned.")
            audit.update(status="infeasible_requested_design", rejection_reasons=[message], conditions=[], repeat=None)
            raise GuiDesignError(message, audit)
        policy = ("Local deterministic engineering selection; no ML score or yield prediction. "
                  "Six designs with a 2–10 min target range prioritize the existing 2/5/10 min coordinates. "
                  "Other counts use logarithmically spaced primary time targets. With distinct oxygen bounds, "
                  "whole low/high-oxygen pairs are retained at each selected time; odd counts add one middle-oxygen probe nearest "
                  "the logarithmic time midpoint. A fixed oxygen range uses a one-dimensional time screen; "
                  "a fixed time uses logarithmic oxygen coordinates with the same primary-target priority. "
                  "Infeasible primary targets are replaced by complete feasible pairs with maximum minimum log-time distance "
                  "from selected pairs; ties prefer lower target time. Within each target, the corrected solver minimizes "
                  "(maximum relative error, sum of relative errors, liquid flow, gas flow) over its entire admissible grid. "
                  "Controller settings must be unique. The bounded pool contains 25 log-time coordinates plus primary targets; "
                  "failure means this declared search cannot supply the requested screen, not global chemical infeasibility.")
        conditions = []
        hardware_ids = ["inventory:" + r["equipment_id"] for r in scenario["selected_equipment"].values()]
        for i, row in enumerate(selected, 1):
            row["selection_disposition"] = "selected"
            row["selection_reason"] = "Part of the requested deterministic time/oxygen coverage; not a predicted best-yield condition"
            c = row["calculated"]
            conditions.append({"id": f"FPX{i:02d}", "candidate_id": row["candidate_id"],
                "target_time_min": row["target"]["nominal_mixture_transit_min"],
                "target_oxygen_eq": row["target"]["inlet_oxygen_equivalents"],
                "liquid_mL_min": c["liquid_flow_mL_min"], "air_STP_mL_min": c["gas_reference_flow_mL_min"],
                "oxygen_STP_mL_min": c["oxygen_reference_volume_flow_mL_min"],
                "oxygen_eq": c["oxygen_equivalents_supplied"], "transit_min": c["no_slip_mixture_transit_min"],
                "rationale": row["selection_reason"] + ". Numerical targets come from the configured range; supplied batch context does not define flow kinetics.",
                "evidence_ids": hardware_ids + history_ids, "history_evidence_role": "supplied_review_context_only_not_condition_specific_prediction",
                "calculated": deepcopy(c), "proposed_setpoints": deepcopy(row["proposed_setpoints"]),
                "actual": deepcopy(row["actual"]), "status": "conditional_proposal"})
        repeat, repeated = None, None
        if include_repeat:
            selected_times = sorted({c["target_time_min"] for c in conditions})
            center = selected_times[len(selected_times) // 2]
            repeated = max((c for c in conditions if c["target_time_min"] == center), key=lambda c: c["target_oxygen_eq"])
        _annotate_roles(conditions, repeated["id"] if repeated else None)
        if repeated:
            repeat = deepcopy(repeated)
            repeat.update(id="FPX-R01", repeat_of=repeated["id"], status="proposed_independent_repeat",
                          question="Independent reproducibility", design_role="independent_repeat",
                          companion_id=repeated["id"], paired_condition_id=None, time_comparison_id=None,
                          rationale="Independently re-establish the middle selected time / higher selected oxygen condition to assess local reproducibility; not included in the requested unique-design count.")
        departures = [k for k in ["concentration_M", "catalyst_mol_percent"] if chemistry[k] != DEFAULT_CHEMISTRY[k]]
        assumptions = [
            "All settings are proposals. Inventory availability does not confirm installed hardware or calibration.",
            "Air is assumed to contain 0.2095 mole fraction O2; STP is explicitly 273.15 K and 1.01325 bar absolute. Installed MFC reference/calibration is unknown.",
            "The selected fixed BPR label is interpreted as gauge; source pressure basis is unspecified. Uniform reactor pressure is label plus assumed 1.01325 bar ambient, not a measurement.",
            "Primary time is nominal ideal-gas, no-slip gas–liquid transit V/(QL+Qg at reactor pressure/temperature). V/QL is a separate liquid-only space-time index, not measured liquid residence.",
            "Inlet O2 equivalents use reference-state molar supply only. Compression changes estimated transit, not inlet equivalents. O2 uptake, dissolution, slip, vapor and pressure gradients are not modeled.",
            "Time and oxygen ranges bound targets; grid-realized values can lie up to 3% outside a target. Resolution is not accuracy.",
            "The time window is exploratory, not a batch-derived speedup or validated kinetic interval. Residual sulfide throughout the screen can justify considering longer exposure after observing results.",
            "Illumination, pressure envelope, gas supply/sensor/collection assignments, stabilization, sampling and actual residence remain unresolved. Rate calculations do not establish total material demand.",
            "Protocol text, research notes and historical records are preserved for review. This local generator does not interpret them as instructions or adjust numerical settings from outcomes."]
        if departures:
            assumptions.append("Departure from the default batch-supported recipe: " + ", ".join(departures) +
                               ". Arithmetic updates do not establish solubility, performance or laboratory suitability.")
        audit.update(status="conditional", conditions=conditions, repeat=repeat, requested_count=n,
                     unique_design_count=len(conditions), total_proposed_runs=n + int(include_repeat),
                     source_inventory_profile_id=profile.get("profile_id"), selection_policy=policy,
                     assumptions=assumptions, chemistry_departures_from_default=departures,
                     evidence_boundary={"history_used_for_numerical_selection": False, "history_evidence_ids": history_ids,
                         "history_context_sha256": digest({k: request[k] for k in ["history_summary", "history_records", "history_ids"] if k in request}),
                         "protocol_text_interpreted": False, "research_notes_interpreted": False,
                         "independent_history_omitted_comparison": False},
                     text_context={k: request.get(k, "") for k in ["protocol_text", "research_notes"]},
                     laboratory_released=False, experimental_validation=False, paid_model_calls=0)
        return audit
    except GuiDesignError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        audit.update(status="invalid_request", rejection_reasons=[str(exc)])
        raise GuiDesignError(str(exc), audit) from exc
