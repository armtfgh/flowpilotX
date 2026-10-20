"""Flow-value metrics used by Designer, Skeptic, and Chief."""

from __future__ import annotations

from typing import Any

import flora_translate.config as cfg


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def mandate_dict(mandate: Any) -> dict:
    if mandate is None:
        return {}
    if hasattr(mandate, "model_dump"):
        return mandate.model_dump()
    if isinstance(mandate, dict):
        return dict(mandate)
    return {}


def compute_flow_sense_report(
    candidate: dict,
    *,
    batch_time_min: float | None,
    batch_concentration_M: float | None = None,
    solvent_name: str | None = None,
    intensification_mandate: Any = None,
) -> dict:
    """Calculate deterministic evidence for whether a candidate is flow-worthy."""
    mandate = mandate_dict(intensification_mandate)
    target = max(_safe_float(mandate.get("tau_reduction_target"), 2.0), 1.0)
    advantage = str(mandate.get("minimum_flow_advantage") or "productivity")

    tau = _safe_float(candidate.get("tau_min"))
    Q = _safe_float(candidate.get("Q_mL_min"))
    d_mm = _safe_float(candidate.get("d_mm"), 1.0)
    L_m = _safe_float(candidate.get("L_m"))
    V_R = _safe_float(candidate.get("V_R_mL"))
    Re = _safe_float(candidate.get("Re"))
    X = _safe_float(candidate.get("expected_conversion"))
    concentration = _safe_float(candidate.get("concentration_M"), 0.0)

    tau_ratio = None
    achieved_reduction = 0.0
    reduction_achievement = 0.0
    boundary_hugging = False
    if batch_time_min and batch_time_min > 0 and tau > 0:
        tau_ratio = tau / batch_time_min
        achieved_reduction = batch_time_min / tau
        reduction_achievement = achieved_reduction / target
        boundary_hugging = tau_ratio >= cfg.BATCH_PROXIMITY_THRESHOLD

    # Circular tube surface-to-volume ratio = 4/d.
    surface_to_volume_m_inv = 4000.0 / d_mm if d_mm > 0 else 0.0

    solvent_lower = (solvent_name or "").lower()
    viscous = any(token in solvent_lower for token in ("des", "glycol", "eg", "glycerol", "tbab"))
    flags: list[str] = []
    if tau_ratio is not None:
        if tau_ratio >= 1.0:
            flags.append("NO_TAU_REDUCTION")
        elif tau_ratio >= cfg.BATCH_PROXIMITY_THRESHOLD:
            flags.append("BATCH_TIME_BOUNDARY_HUGGING")
        elif reduction_achievement < 0.3:
            flags.append("WEAK_TAU_REDUCTION")
    if viscous and Q < 0.1 and L_m > 10.0:
        flags.append("ULTRA_LOW_FLOW_VISCOUS_MEDIA")
    if Q < 0.05:
        flags.append("BELOW_PRACTICAL_Q")
    if L_m > 15.0:
        flags.append("LONG_COIL")
    if batch_concentration_M and concentration and concentration < 0.5 * batch_concentration_M:
        flags.append("SUSPICIOUS_DILUTION")

    tau_score = _clamp01(reduction_achievement)
    heat_score = _clamp01((surface_to_volume_m_inv - 200.0) / 2500.0)
    if Re > 100:
        heat_score = max(heat_score, 0.7)
    if d_mm <= 1.0:
        heat_score = max(heat_score, 0.6)

    safety_score = 0.4
    if V_R <= 10.0:
        safety_score += 0.2
    if L_m <= 15.0:
        safety_score += 0.15
    if "ULTRA_LOW_FLOW_VISCOUS_MEDIA" in flags or "LONG_COIL" in flags:
        safety_score -= 0.25
    safety_score = _clamp01(safety_score)

    selectivity_score = 0.4
    if d_mm <= 1.0 and L_m <= 15.0:
        selectivity_score = 0.6
    if Q < 0.08 and L_m > 10.0:
        selectivity_score = 0.2

    domain_proxy = {
        "selectivity": selectivity_score,
        "heat_transfer": heat_score,
        "hazardous_intermediate": safety_score,
        "safety": safety_score,
        "productivity": tau_score,
    }.get(advantage, tau_score)

    process_value_score = _clamp01(
        0.35 * tau_score
        + 0.20 * heat_score
        + 0.25 * safety_score
        + 0.20 * selectivity_score
    )
    if boundary_hugging:
        process_value_score *= 0.6
    if X and X < 0.5:
        process_value_score *= 0.7

    return {
        "batch_time_min": batch_time_min,
        "tau_min": tau,
        "tau_ratio": round(tau_ratio, 4) if tau_ratio is not None else None,
        "achieved_reduction_factor": round(achieved_reduction, 3),
        "target_reduction_factor": round(target, 3),
        "reduction_achievement": round(reduction_achievement, 3),
        "boundary_hugging": boundary_hugging,
        "Q_mL_min": Q,
        "L_m": L_m,
        "V_R_mL": V_R,
        "surface_to_volume_m_inv": round(surface_to_volume_m_inv, 1),
        "primary_flow_advantage": advantage,
        "primary_advantage_proxy_score": round(domain_proxy, 3),
        "process_value_score": round(process_value_score, 3),
        "automatic_flags": flags,
    }


def attach_flow_sense_reports(
    candidates: list[dict],
    *,
    batch_time_min: float | None,
    batch_concentration_M: float | None = None,
    solvent_name: str | None = None,
    intensification_mandate: Any = None,
) -> list[dict]:
    for candidate in candidates:
        candidate["flow_sense_report"] = compute_flow_sense_report(
            candidate,
            batch_time_min=batch_time_min,
            batch_concentration_M=batch_concentration_M,
            solvent_name=solvent_name,
            intensification_mandate=intensification_mandate,
        )
    return candidates


def domain_flow_value_defaults(candidate: dict, intensification_mandate: Any = None) -> dict:
    report = candidate.get("flow_sense_report") or {}
    target = _safe_float(report.get("target_reduction_factor"), 2.0)
    achieved = _safe_float(report.get("achieved_reduction_factor"))
    reduction_achievement = achieved / target if target > 0 else 0.0
    tau_value = _clamp01(reduction_achievement)

    return {
        "selectivity_flow_value": _clamp01(0.6 if _safe_float(candidate.get("d_mm"), 1.0) <= 1.0 else 0.4),
        "tau_reduction_flow_value": tau_value,
        "achieved_reduction_factor": round(achieved, 3),
        "target_reduction_factor": round(target, 3),
        "reduction_achievement": round(reduction_achievement, 3),
        "heat_transfer_flow_value": _clamp01(_safe_float(report.get("surface_to_volume_m_inv")) / 3000.0),
        "safety_delta_flow_value": _clamp01(0.7 if _safe_float(candidate.get("V_R_mL")) <= 10.0 else 0.4),
    }


def get_score_entry(entries: list[dict], candidate_id: int) -> dict:
    return next((e for e in entries if e.get("candidate_id") == candidate_id), {})


def enrich_scoring_with_flow_values(
    candidates: list[dict],
    scoring: dict,
    *,
    intensification_mandate: Any = None,
) -> dict:
    """Ensure all four scoring lists carry flow-value fields."""
    by_id = {int(c.get("id")): c for c in candidates if c.get("id") is not None}
    for cid, candidate in by_id.items():
        defaults = domain_flow_value_defaults(candidate, intensification_mandate)
        mapping = [
            ("chemistry_scores", "selectivity_flow_value"),
            ("kinetics_scores", "tau_reduction_flow_value"),
            ("fluidics_scores", "heat_transfer_flow_value"),
            ("safety_scores", "safety_delta_flow_value"),
        ]
        for list_key, field in mapping:
            entry = get_score_entry(scoring.get(list_key, []), cid)
            if not entry:
                continue
            # Flow-value scores are code-owned evidence. LLM agents may explain
            # them, but weak/local models frequently overstate intensification.
            entry[field] = _clamp01(defaults[field])
            if list_key == "kinetics_scores":
                entry["achieved_reduction_factor"] = defaults["achieved_reduction_factor"]
                entry["target_reduction_factor"] = defaults["target_reduction_factor"]
                entry["reduction_achievement"] = defaults["reduction_achievement"]
    scoring["process_value_scores"] = compute_pvs_for_candidates(candidates, scoring)
    return scoring


def compute_pvs_for_candidates(candidates: list[dict], scoring: dict) -> list[dict]:
    rows: list[dict] = []
    for candidate in candidates:
        cid = int(candidate.get("id", 0))
        chem = _safe_float(get_score_entry(scoring.get("chemistry_scores", []), cid).get("selectivity_flow_value"), 0.0)
        kin = _safe_float(get_score_entry(scoring.get("kinetics_scores", []), cid).get("tau_reduction_flow_value"), 0.0)
        flu = _safe_float(get_score_entry(scoring.get("fluidics_scores", []), cid).get("heat_transfer_flow_value"), 0.0)
        saf = _safe_float(get_score_entry(scoring.get("safety_scores", []), cid).get("safety_delta_flow_value"), 0.0)
        pvs = (
            cfg.PVS_WEIGHT_SELECTIVITY * chem
            + cfg.PVS_WEIGHT_TAU * kin
            + cfg.PVS_WEIGHT_HEAT * flu
            + cfg.PVS_WEIGHT_SAFETY * saf
        )
        rows.append({
            "candidate_id": cid,
            "PVS": round(_clamp01(pvs), 4),
            "selectivity_flow_value": round(_clamp01(chem), 4),
            "tau_reduction_flow_value": round(_clamp01(kin), 4),
            "heat_transfer_flow_value": round(_clamp01(flu), 4),
            "safety_delta_flow_value": round(_clamp01(saf), 4),
        })
    rows.sort(key=lambda row: row["PVS"], reverse=True)
    return rows


def pvs_for_candidate(scoring: dict, candidate_id: int) -> float:
    for row in scoring.get("process_value_scores", []):
        if int(row.get("candidate_id", 0)) == int(candidate_id):
            return _safe_float(row.get("PVS"), 0.0)
    return 0.0
