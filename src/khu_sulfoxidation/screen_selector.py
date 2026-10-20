"""Deterministic complete-grid first screens with preserved rejected targets.

This module selects engineering-feasible coverage. It does not rank predicted
yields or convert batch time into flow kinetics. A failed grid is not silently
reduced to a misleading partial screen; one-level fallback is explicit.
"""
from __future__ import annotations

from copy import deepcopy

from .candidate_space import solve_candidate
from .inventory import configuration_hash, finite_number


def generate_screen(profile: dict, chemistry: dict, config: dict) -> dict:
    times = config.get("times_min", [])
    oxygen = config.get("oxygen_equivalents", [])
    design = config.get("design")
    if design == "three_times_two_oxygen_levels":
        if len(times) != 3 or len(oxygen) != 2:
            raise ValueError("The core screen requires exactly three time coordinates and two oxygen levels")
    elif design == "one_oxygen_level_fallback":
        if not 3 <= len(times) <= 5 or len(oxygen) != 1:
            raise ValueError("Fallback requires three to five time coordinates and one oxygen level")
    else:
        raise ValueError("Screen design must explicitly name the 3x2 grid or the one-level fallback")
    for values, label in [(times, "time"), (oxygen, "oxygen")]:
        known = [finite_number(v, label, positive=True) for v in values if v is not None]
        if len(set(known)) != len(known):
            raise ValueError(f"Repeated {label} coordinates are not unique conditions; use the explicit repeat")
    if not config.get("screen_id") or config.get("version") is None:
        raise ValueError("A screen ID and configuration version are required")
    snapshot = {"apparatus": deepcopy(profile), "chemistry": deepcopy(chemistry), "screen": deepcopy(config)}
    result = {"schema_version": "khu_screen_v1", "screen_id": config["screen_id"],
        "status": "blocked", "configuration_snapshot": snapshot,
        "configuration_sha256": configuration_hash(snapshot), "candidates": [],
        "selected_runs": [], "planned_slots": [], "rejection_reasons": [],
        "selection_policy": "Complete requested grid; choose lowest observed relative error among tested feasible controller pairs; no yield prediction or global-optimality claim",
        "oxygen_sensitivity_tested": False,
        "oxygen_sensitivity_planned": len(oxygen) == 2,
        "experimental_validation": False,
        "run_order_status": "proposed_grid_order_requires_lab_blocking_or_randomization_review"}
    seen = {}
    for ti, tau in enumerate(times, 1):
        for oi, equiv in enumerate(oxygen, 1):
            cid = f"C{len(result['candidates']) + 1:02d}"
            target = {"candidate_id": cid, "time_min": tau, "oxygen_equivalents": equiv,
                **{key: config.get(key) for key in ["time_basis", "time_relative_tolerance", "oxygen_relative_tolerance", "max_grid_evaluations"]}}
            candidate = solve_candidate(profile, chemistry, target)
            candidate.update(time_coordinate=f"T{ti}", oxygen_coordinate=f"O{oi}",
                rationale="Explore time and supplied oxygen while preserving fixed chemistry/apparatus; measured outcomes are unknown",
                comparison_arm=config.get("comparison_arm"))
            if candidate["status"] == "engineering_feasible_requires_review":
                pair = tuple(candidate["setpoints"][k] for k in ["liquid_flow_mL_min", "gas_reference_flow_mL_min"])
                if pair in seen:
                    candidate["status"] = "rejected"
                    candidate["rejection_reasons"].append(f"Quantized settings duplicate {seen[pair]}; not a distinct core condition")
                else:
                    seen[pair] = cid
            result["candidates"].append(candidate)
            result["planned_slots"].append({"run_id": cid, "candidate_id": cid, "repeat_of": None,
                "time_coordinate": f"T{ti}", "oxygen_coordinate": f"O{oi}",
                "target_time_min": tau, "target_oxygen_equivalents": equiv,
                "time_basis": config.get("time_basis"), "status": candidate["status"],
                "preparation_type": "independent_condition", "comparison_arm": config.get("comparison_arm")})
    reference = config.get("repeat_candidate_id")
    if reference not in {c["candidate_id"] for c in result["candidates"]}:
        raise ValueError("repeat_candidate_id must explicitly identify a core condition")
    reference_slot = next(s for s in result["planned_slots"] if s["candidate_id"] == reference)
    result["planned_slots"].append({**deepcopy(reference_slot), "run_id": "R01", "repeat_of": reference,
        "preparation_type": "independent_preparation_and_operation", "actual_repeat_independence_confirmed": None})
    failed = [c for c in result["candidates"] if c["status"] != "engineering_feasible_requires_review"]
    if failed:
        result["status"] = "blocked" if any(c["status"] == "blocked" for c in failed) else "infeasible_requested_grid"
        result["rejection_reasons"] = [{"candidate_id": c["candidate_id"], "reasons": c["rejection_reasons"]} for c in failed]
    else:
        result["status"] = "engineering_feasible_requires_review"
        by_id = {c["candidate_id"]: c for c in result["candidates"]}
        for order, slot in enumerate(result["planned_slots"], 1):
            candidate = by_id[slot["candidate_id"]]
            result["selected_runs"].append({**deepcopy(slot), "proposed_execution_order": order,
                "configuration_sha256": result["configuration_sha256"], "target": deepcopy(candidate["target"]),
                "setpoints": deepcopy(candidate["setpoints"]), "calculated": deepcopy(candidate["calculated"]),
                "actual_conditions": None, "actual_outcomes": None})
    return result
