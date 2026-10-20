"""Closed-loop experimental feedback and deterministic design refinement.

This module turns lab observations from a flow experiment into a revised
FLORA design suggestion. It is intentionally deterministic: LLM agents can
interpret richer free text later, but the first pass must be reproducible,
testable, and safe to run without API keys.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from flora_translate.residence_time_basis import (
    IN_CHANNEL_BASIS,
    INLET_STP_BASIS,
    LIQUID_ONLY_BASIS,
    UNKNOWN_BASIS,
    actual_gas_flow_from_stp,
    normalize_residence_time_basis,
    residence_time_basis_label,
    stp_gas_flow_from_actual,
)


DEFAULT_TARGET_YIELD = 80.0
DEFAULT_TARGET_CONVERSION = 90.0
DEFAULT_TARGET_SELECTIVITY = 85.0


class ActualConditions(BaseModel):
    """Actual conditions used in one experimental run."""

    residence_time_min: Optional[float] = None
    residence_time_inlet_min: Optional[float] = None
    residence_time_in_channel_min: Optional[float] = None
    residence_time_basis: str = ""  # inlet | in_channel | liquid_only | unknown
    flow_rate_mL_min: Optional[float] = None
    substrate_flow_mL_min: Optional[float] = None
    gas_flow_in_channel_mL_min: Optional[float] = None
    gas_flow_stp_mL_min: Optional[float] = None
    gas_equiv_inlet: Optional[float] = None
    temperature_C: Optional[float] = None
    concentration_M: Optional[float] = None
    tubing_ID_mm: Optional[float] = None
    reactor_volume_mL: Optional[float] = None
    BPR_bar: Optional[float] = None
    wavelength_nm: Optional[float] = None
    light_power_W: Optional[float] = None


class ExperimentalOutcomes(BaseModel):
    """Measured results and operational observations from the lab."""

    yield_pct: Optional[float] = None
    product_pct: Optional[float] = None
    starting_material_pct: Optional[float] = None
    conversion_pct: Optional[float] = None
    selectivity_pct: Optional[float] = None
    pressure_bar: Optional[float] = None
    pressure_drift_bar: Optional[float] = None
    clogging_observed: bool = False
    precipitation_observed: bool = False
    gas_liquid_stability: str = ""  # stable | slugging | flooding | unknown
    impurity_notes: str = ""
    analytical_method: str = ""
    notes: str = ""


class ExperimentResult(BaseModel):
    """One closed-loop run tied to a design version."""

    run_id: str = ""
    design_version: int = 1
    actual_conditions: ActualConditions = Field(default_factory=ActualConditions)
    outcomes: ExperimentalOutcomes = Field(default_factory=ExperimentalOutcomes)
    free_text_observations: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class RefinementDecision(BaseModel):
    """Machine-readable diagnosis and next design recommendation."""

    design_version_in: int = 1
    design_version_out: int = 2
    score: float = 0.0
    status: str = "needs_refinement"  # converged | needs_refinement | screen_required
    diagnosis: str = ""
    failure_modes: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    revision_instructions: str = ""
    parameter_changes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    safety_flags: list[str] = Field(default_factory=list)
    next_experiment: dict[str, Any] = Field(default_factory=dict)


class ClosedLoopResult(BaseModel):
    """Result package returned after analyzing one experiment."""

    experiment: ExperimentResult
    decision: RefinementDecision
    refined_result: dict


class CampaignCalibration(BaseModel):
    """Evidence-backed calibration from multiple experimental observations."""

    n_experiments: int = 0
    n_usable: int = 0
    response_metric: str = ""
    target_response_pct: float = 0.0
    primary_residence_time_basis: str = ""
    best_run_id: str = ""
    best_response_pct: Optional[float] = None
    best_tau_min: Optional[float] = None
    target_tau_min: Optional[float] = None
    recommended_tau_min: Optional[float] = None
    best_tau_inlet_min: Optional[float] = None
    best_tau_in_channel_min: Optional[float] = None
    target_tau_inlet_min: Optional[float] = None
    apparent_rate_min_inv: Optional[float] = None
    target_tau_in_channel_min: Optional[float] = None
    recommended_tau_inlet_min: Optional[float] = None
    recommended_tau_in_channel_min: Optional[float] = None
    anchor_conditions: dict[str, Any] = Field(default_factory=dict)
    recommended_conditions: dict[str, Any] = Field(default_factory=dict)
    design_ladder: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def refine_from_experimental_campaign(
    current_result: dict,
    experiments: list[ExperimentResult],
    target_yield_pct: float = DEFAULT_TARGET_YIELD,
    target_conversion_pct: float = DEFAULT_TARGET_CONVERSION,
    target_selectivity_pct: float = DEFAULT_TARGET_SELECTIVITY,
) -> ClosedLoopResult:
    """Refine a design from a complete experimental campaign.

    This is the batch/table-oriented entry point for the GUI and scripts. It
    reuses ``refine_from_experiment`` but passes all previous observations as
    campaign history so multi-run kinetic calibration can override optimistic
    first-pass intensification assumptions.
    """

    if not experiments:
        raise ValueError("experiments must contain at least one ExperimentResult")
    history = [{"experiment": exp.model_dump()} for exp in experiments[:-1]]
    return refine_from_experiment(
        current_result=current_result,
        experiment=experiments[-1],
        campaign_history=history,
        target_yield_pct=target_yield_pct,
        target_conversion_pct=target_conversion_pct,
        target_selectivity_pct=target_selectivity_pct,
    )


def extract_experiments_from_text(text: str, design_version: int = 1) -> list[ExperimentResult]:
    """Extract structured Entry-style experimental feedback from free text."""

    text = str(text or "")
    if not re.search(r"\bEntry\s+\d+", text, flags=re.IGNORECASE):
        return []

    matches = list(re.finditer(r"\bEntry\s+(\d+)\s*,?\s*([^:\n]*)\s*:", text, flags=re.IGNORECASE))
    experiments: list[ExperimentResult] = []
    for idx, match in enumerate(matches):
        entry_no = int(match.group(1))
        note = (match.group(2) or f"entry_{entry_no}").strip() or f"entry_{entry_no}"
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]
        note_match = re.search(r"\bNote\s+([^,:\n]+)", block, flags=re.IGNORECASE)
        if note.startswith("entry_") and note_match:
            note = note_match.group(1).strip() or note

        product_pct = _extract_percent(block, r"\bproduct\b")
        isolated_yield_pct = _extract_percent(block, r"\bisolated\s+yield\b")
        starting_material_pct = _extract_percent(block, r"\bstarting\s+material\b")
        conversion_pct = None
        if starting_material_pct is not None:
            conversion_pct = _bounded(100.0 - starting_material_pct, 0.0, 100.0)
        substrate_flow = (
            _extract_number(block, r"\bsubstrate\s+flow\b")
            or _extract_number(block, r"\bsubstrate\b")
        )

        experiment = ExperimentResult(
            run_id=f"entry_{entry_no:02d}_{_slug(note)}",
            design_version=design_version,
            actual_conditions=ActualConditions(
                residence_time_inlet_min=_extract_number(block, r"\bt\s+inlet\b"),
                residence_time_in_channel_min=_extract_number(block, r"\bt\s+in-channel\b"),
                residence_time_basis="inlet_stp",
                flow_rate_mL_min=substrate_flow,
                substrate_flow_mL_min=substrate_flow,
                gas_flow_in_channel_mL_min=_extract_number(block, r"\bO[2₂]\s+in-channel\b"),
                gas_flow_stp_mL_min=_extract_number(block, r"\bO[2₂]\s+inlet/STP\b"),
                gas_equiv_inlet=_extract_number(block, r"\bO[2₂]\s+equiv(?:\s+inlet)?\b"),
                temperature_C=_extract_number(block, r"\bT\b"),
                concentration_M=_extract_number(block, r"\bc\b"),
                tubing_ID_mm=_extract_number(block, r"\btubing\s+ID\b"),
                reactor_volume_mL=_extract_number(block, r"\breactor\s+volume\b"),
                BPR_bar=_extract_number(block, r"\bP\b"),
            ),
            outcomes=ExperimentalOutcomes(
                yield_pct=isolated_yield_pct if isolated_yield_pct is not None else product_pct,
                product_pct=product_pct,
                starting_material_pct=starting_material_pct,
                conversion_pct=conversion_pct,
                notes=note,
            ),
            free_text_observations=block.strip(),
        )
        if (
            experiment.actual_conditions.residence_time_in_channel_min is not None
            and (experiment.outcomes.yield_pct is not None or experiment.outcomes.product_pct is not None)
        ):
            experiments.append(experiment)
    return experiments


def calibrate_experimental_campaign(
    experiments: list[ExperimentResult],
    target_yield_pct: float = DEFAULT_TARGET_YIELD,
    target_conversion_pct: float = DEFAULT_TARGET_CONVERSION,
    residence_time_basis: str | None = None,
) -> CampaignCalibration:
    """Fit an apparent one-parameter response model from campaign data.

    The model is deliberately simple and conservative: it uses the best
    observed response as the anchor, then estimates the residence time required
    to reach the target response. This avoids letting poor early screen points
    dominate a later, better empirical anchor.
    """

    preferred_basis = _campaign_residence_time_basis(experiments, residence_time_basis)
    usable: list[dict[str, Any]] = []
    for exp in experiments:
        normalized = _normalize_outcomes(exp.outcomes)
        metric, response = _select_campaign_response(normalized)
        tau, basis = _residence_time_for_basis(exp.actual_conditions, preferred_basis)
        if not metric or response is None or tau <= 0:
            continue
        response = _bounded(response, 0.01, 98.0)
        usable.append(
            {
                "experiment": exp,
                "metric": metric,
                "response_pct": response,
                "tau_min": tau,
                "residence_time_basis": basis,
                "k_min_inv": _first_order_rate(response, tau),
            }
        )

    calibration = CampaignCalibration(
        n_experiments=len(experiments),
        n_usable=len(usable),
        primary_residence_time_basis=residence_time_basis_label(preferred_basis),
    )
    if len(usable) < 2:
        calibration.notes.append(
            "Need at least two usable experiments with response and residence time on the selected basis for campaign calibration."
        )
        return calibration

    usable.sort(key=lambda item: item["response_pct"], reverse=True)
    best = usable[0]
    best_exp: ExperimentResult = best["experiment"]
    target_response = target_yield_pct if best["metric"] in {"yield_pct", "product_pct"} else target_conversion_pct
    target_response = _bounded(target_response, 1.0, 95.0)
    k_best = best["k_min_inv"]
    target_tau = _tau_for_response(target_response, k_best)
    best_tau = best["tau_min"]

    if best["response_pct"] >= target_response:
        recommended_tau = best_tau
    else:
        # Move toward the fitted target without making a single experimental
        # jump too large. For the THQ data this gives an intermediate ~190 min
        # screen before the fitted ~260 min target point.
        recommended_tau = min(target_tau, max(best_tau * 1.4, best_tau + 30.0))

    anchor = _conditions_for_tau(best_exp.actual_conditions, best_tau, preferred_basis)
    recommended = _conditions_for_tau(best_exp.actual_conditions, recommended_tau, preferred_basis)
    target_conditions = _conditions_for_tau(best_exp.actual_conditions, target_tau, preferred_basis)

    ladder = [
        {
            "label": "best_observed_anchor",
            "purpose": "Reproduce the best measured condition before claiming improvement.",
            **anchor,
        }
    ]
    if recommended_tau > best_tau * 1.05:
        ladder.append(
            {
                "label": "next_intermediate_screen",
                "purpose": "Step toward the target while limiting the next experimental jump.",
                **recommended,
            }
        )
    if target_tau > recommended_tau * 1.05:
        ladder.append(
            {
                "label": "target_estimate",
                "purpose": f"Fitted first-order estimate for {target_response:.1f}% {best['metric']}.",
                **target_conditions,
            }
        )

    calibration.response_metric = best["metric"]
    calibration.target_response_pct = round(target_response, 3)
    calibration.best_run_id = best_exp.run_id
    calibration.best_response_pct = round(best["response_pct"], 3)
    calibration.best_tau_min = round(best_tau, 3)
    calibration.target_tau_min = round(target_tau, 3)
    calibration.recommended_tau_min = round(recommended_tau, 3)
    calibration.best_tau_inlet_min = best_exp.actual_conditions.residence_time_inlet_min
    calibration.best_tau_in_channel_min = best_exp.actual_conditions.residence_time_in_channel_min
    calibration.apparent_rate_min_inv = round(k_best, 6)
    if preferred_basis == INLET_STP_BASIS:
        calibration.target_tau_inlet_min = round(target_tau, 3)
        calibration.recommended_tau_inlet_min = round(recommended_tau, 3)
        calibration.target_tau_in_channel_min = target_conditions.get(
            "residence_time_in_channel_min"
        )
        calibration.recommended_tau_in_channel_min = recommended.get(
            "residence_time_in_channel_min"
        )
    else:
        calibration.target_tau_in_channel_min = round(target_tau, 3)
        calibration.recommended_tau_in_channel_min = round(recommended_tau, 3)
    calibration.anchor_conditions = anchor
    calibration.recommended_conditions = recommended
    calibration.design_ladder = ladder
    calibration.notes.extend(
        [
            "Experimental campaign data override the first-pass intensification mandate.",
            f"Residence time is calibrated on {residence_time_basis_label(preferred_basis)}.",
            "The fitted target is a screening estimate, not a final kinetic model.",
        ]
    )
    return calibration


def refine_from_experiment(
    current_result: dict,
    experiment: ExperimentResult,
    campaign_history: Optional[list[dict]] = None,
    target_yield_pct: float = DEFAULT_TARGET_YIELD,
    target_conversion_pct: float = DEFAULT_TARGET_CONVERSION,
    target_selectivity_pct: float = DEFAULT_TARGET_SELECTIVITY,
) -> ClosedLoopResult:
    """Analyze lab feedback and return a revised result dict.

    The revised result preserves the original FLORA output structure while
    updating the proposal fields needed for the next experimental run.
    """

    result = deepcopy(current_result)
    proposal = result.setdefault("proposal", {})
    history = campaign_history or []

    version_in = int(result.get("design_version", proposal.get("design_version", 1)) or 1)
    version_out = version_in + 1

    outcomes = _normalize_outcomes(experiment.outcomes)
    actual = experiment.actual_conditions
    failure_modes = _detect_failure_modes(
        proposal=proposal,
        outcomes=outcomes,
        target_yield_pct=target_yield_pct,
        target_conversion_pct=target_conversion_pct,
        target_selectivity_pct=target_selectivity_pct,
    )
    score = _objective_score(outcomes)

    parameter_changes: dict[str, dict[str, Any]] = {}
    recommended_actions: list[str] = []
    safety_flags: list[str] = list(proposal.get("safety_flags") or [])

    actual_tau, selected_basis = _residence_time_for_basis(actual)
    if actual_tau <= 0:
        actual_tau, selected_basis = _residence_time_for_basis(
            actual,
            proposal.get("residence_time_basis"),
        )
    base_tau = _num(actual_tau, proposal.get("residence_time_min"), 10.0)
    base_q = _num(
        actual.substrate_flow_mL_min,
        actual.flow_rate_mL_min,
        proposal.get("flow_rate_mL_min"),
        0.1,
    )
    base_volume = _num(
        actual.reactor_volume_mL,
        proposal.get("reactor_volume_mL"),
        base_tau * base_q,
    )
    base_temp = _num(actual.temperature_C, proposal.get("temperature_C"), 25.0)
    base_conc = _num(actual.concentration_M, proposal.get("concentration_M"), 0.1)
    base_bpr = _num(actual.BPR_bar, proposal.get("BPR_bar"), 0.0)
    base_id = _num(actual.tubing_ID_mm, proposal.get("tubing_ID_mm"), 1.0)

    tau = base_tau
    temperature = base_temp
    concentration = base_conc
    bpr = base_bpr
    tubing_id = base_id

    if "kinetic_underconversion" in failure_modes:
        tau = _revised_tau_for_underconversion(
            base_tau=base_tau,
            conversion_pct=outcomes.conversion_pct,
            target_conversion_pct=target_conversion_pct,
        )
        recommended_actions.append(
            "Increase in-channel residence time before changing chemistry because conversion is below target."
        )

    if "low_yield_unknown_conversion" in failure_modes:
        tau = _revised_tau_for_low_yield(
            base_tau=base_tau,
            yield_pct=outcomes.yield_pct,
            target_yield_pct=target_yield_pct,
        )
        recommended_actions.append(
            "Yield is below target but conversion/selectivity are incomplete; run a longer residence-time screen."
        )

    if "selectivity_loss" in failure_modes:
        temperature = _bounded(base_temp - 10.0, lower=-20.0, upper=180.0)
        tau = _bounded(min(tau, base_tau * 0.85), lower=1.0, upper=240.0)
        recommended_actions.append(
            "Reduce thermal/over-reaction stress because conversion is acceptable but selectivity or yield is poor."
        )

    if "solubility_or_fouling" in failure_modes:
        concentration = _bounded(base_conc * 0.7, lower=0.005, upper=2.0)
        tubing_id = max(base_id, 1.0)
        _append_unique(proposal.setdefault("pre_reactor_steps", []), "inline 2 um filter before reactor")
        _append_unique(proposal.setdefault("pre_reactor_steps", []), "confirm complete dissolution before pumping")
        recommended_actions.append(
            "Reduce concentration and add filtration/dissolution controls because pressure drift or solids were observed."
        )

    if "pressure_instability" in failure_modes:
        if base_bpr > 0:
            bpr = min(base_bpr + 2.0, 10.0)
        tubing_id = max(tubing_id, 1.0)
        recommended_actions.append(
            "Stabilize hydraulic operation before further intensification."
        )
        safety_flags.append("SCREEN_REQUIRED: pressure instability observed during experiment")

    if "gas_liquid_instability" in failure_modes:
        if base_bpr > 0:
            bpr = min(max(base_bpr + 2.0, 5.0), 10.0)
        tubing_id = max(tubing_id, 1.6)
        recommended_actions.append(
            "Use a larger gas-liquid compatible tube and higher BPR screening point."
        )

    if not failure_modes:
        if _num(outcomes.yield_pct, 0.0) >= target_yield_pct:
            status = "converged"
            recommended_actions.append(
                "Design met the target yield; confirm reproducibility before scale-up."
            )
        else:
            status = "needs_refinement"
            tau = _bounded(base_tau * 1.2, lower=1.0, upper=240.0)
            recommended_actions.append(
                "No dominant failure mode was detected; run a mild residence-time confirmation screen."
            )
    else:
        status = "screen_required" if any("pressure" in f or "fouling" in f for f in failure_modes) else "needs_refinement"

    flow_rate, gas_update = _scaled_flows_for_tau(
        proposal=proposal,
        volume_mL=base_volume,
        tau_min=tau,
        fallback_liquid_q=base_q,
        actual=actual,
        residence_time_basis=selected_basis,
    )
    _set_change(proposal, parameter_changes, "residence_time_min", round(tau, 3))
    _set_change(proposal, parameter_changes, "residence_time_basis", residence_time_basis_label(selected_basis))
    _set_change(proposal, parameter_changes, "flow_rate_mL_min", round(flow_rate, 5))
    _set_change(proposal, parameter_changes, "reactor_volume_mL", round(base_volume, 4))
    _set_change(proposal, parameter_changes, "temperature_C", round(temperature, 2))
    _set_change(proposal, parameter_changes, "concentration_M", round(concentration, 4))
    _set_change(proposal, parameter_changes, "BPR_bar", round(bpr, 2))
    _set_change(proposal, parameter_changes, "tubing_ID_mm", round(tubing_id, 3))

    for field, value in gas_update.items():
        _set_change(proposal, parameter_changes, field, value)

    _sync_liquid_streams(proposal, flow_rate, concentration)
    _sync_gas_streams(proposal, gas_update)

    campaign_experiments = _collect_campaign_experiments(history, experiment)
    campaign_calibration = calibrate_experimental_campaign(
        campaign_experiments,
        target_yield_pct=target_yield_pct,
        target_conversion_pct=target_conversion_pct,
    )
    if campaign_calibration.n_usable >= 2:
        _apply_campaign_calibration(
            proposal=proposal,
            parameter_changes=parameter_changes,
            calibration=campaign_calibration,
        )
        if (
            campaign_calibration.best_response_pct is not None
            and campaign_calibration.best_response_pct >= campaign_calibration.target_response_pct
        ):
            status = "converged"
        else:
            status = "needs_refinement"
            _append_unique(failure_modes, "experiment_calibrated_kinetics")
        recommended_actions.insert(
            0,
            "Use evidence-calibrated residence times from the experimental campaign instead of the original intensification estimate.",
        )
        safety_flags.append("SCREEN_REQUIRED: evidence-calibrated from experimental feedback")

    proposal["safety_flags"] = _dedupe(safety_flags)
    proposal["engine_validated"] = False
    proposal["confidence"] = "MEDIUM" if status == "converged" or campaign_calibration.n_usable >= 2 else "LOW"
    proposal["chemistry_notes"] = _append_note(
        proposal.get("chemistry_notes", ""),
        f"Closed-loop refinement v{version_out}: {', '.join(failure_modes) or 'target check'}."
    )

    reasoning = proposal.setdefault("reasoning_per_field", {})
    reasoning["closed_loop_refinement"] = (
        "Updated from experimental feedback. The next run keeps deterministic "
        "volume closure and prioritizes operational stability before scale-up."
    )

    revision_instructions = _build_revision_instructions(
        failure_modes=failure_modes,
        parameter_changes=parameter_changes,
        recommended_actions=recommended_actions,
    )
    diagnosis = _build_diagnosis(outcomes, failure_modes, status)
    next_experiment = _build_next_experiment(proposal, status, version_out)

    decision = RefinementDecision(
        design_version_in=version_in,
        design_version_out=version_out,
        score=score,
        status=status,
        diagnosis=diagnosis,
        failure_modes=failure_modes,
        recommended_actions=recommended_actions,
        revision_instructions=revision_instructions,
        parameter_changes=parameter_changes,
        safety_flags=proposal["safety_flags"],
        next_experiment=next_experiment,
    )

    result["proposal"] = proposal
    result["design_version"] = version_out
    result["closed_loop"] = {
        "last_experiment": experiment.model_dump(),
        "last_decision": decision.model_dump(),
        "history_length": len(history) + 1,
    }
    if campaign_calibration.n_usable >= 2:
        result["closed_loop"]["campaign_calibration"] = campaign_calibration.model_dump()
    result["confidence"] = proposal["confidence"]
    result["explanation"] = _append_note(
        result.get("explanation", ""),
        f"Closed-loop v{version_out}: {diagnosis} Next run: {revision_instructions}",
    )

    return ClosedLoopResult(
        experiment=experiment,
        decision=decision,
        refined_result=result,
    )


def summarize_campaign(cycles: list[ClosedLoopResult]) -> dict:
    """Create a compact summary for UI tables or case-study export."""

    rows = []
    for cycle in cycles:
        exp = cycle.experiment
        decision = cycle.decision
        rows.append(
            {
                "run_id": exp.run_id,
                "design_in": decision.design_version_in,
                "design_out": decision.design_version_out,
                "yield_pct": exp.outcomes.yield_pct,
                "conversion_pct": exp.outcomes.conversion_pct,
                "selectivity_pct": exp.outcomes.selectivity_pct,
                "score": round(decision.score, 2),
                "status": decision.status,
                "failure_modes": ", ".join(decision.failure_modes),
            }
        )
    return {
        "n_cycles": len(cycles),
        "best_score": max((c.decision.score for c in cycles), default=0.0),
        "cycles": rows,
    }


def _collect_campaign_experiments(
    history: list[dict],
    experiment: ExperimentResult,
) -> list[ExperimentResult]:
    experiments: list[ExperimentResult] = []
    seen: set[str] = set()
    for item in history:
        exp_data = item.get("experiment") if isinstance(item, dict) else None
        if not exp_data:
            continue
        try:
            exp = ExperimentResult.model_validate(exp_data)
        except Exception:
            continue
        key = exp.run_id or str(len(experiments))
        if key in seen:
            continue
        seen.add(key)
        experiments.append(exp)

    key = experiment.run_id or str(len(experiments))
    if key not in seen:
        experiments.append(experiment)
    return experiments


def _select_campaign_response(outcomes: ExperimentalOutcomes) -> tuple[str, Optional[float]]:
    if outcomes.yield_pct is not None:
        return "yield_pct", outcomes.yield_pct
    if outcomes.product_pct is not None:
        return "product_pct", outcomes.product_pct
    if outcomes.conversion_pct is not None:
        return "conversion_pct", outcomes.conversion_pct
    return "", None


def _campaign_residence_time_basis(
    experiments: list[ExperimentResult],
    requested_basis: str | None = None,
) -> str:
    # A reagent-gas campaign always uses the inlet/STP convention for fitting
    # and refinement. In-channel time remains a derived diagnostic.
    if any(
        exp.actual_conditions.gas_flow_stp_mL_min
        or exp.actual_conditions.gas_flow_in_channel_mL_min
        for exp in experiments
    ):
        return INLET_STP_BASIS
    requested = normalize_residence_time_basis(requested_basis)
    if requested != UNKNOWN_BASIS:
        return requested
    counts = {
        INLET_STP_BASIS: 0,
        IN_CHANNEL_BASIS: 0,
        LIQUID_ONLY_BASIS: 0,
    }
    for exp in experiments:
        basis = normalize_residence_time_basis(exp.actual_conditions.residence_time_basis)
        if basis != UNKNOWN_BASIS:
            counts[basis] = counts.get(basis, 0) + 1
            continue
        actual = exp.actual_conditions
        if actual.residence_time_inlet_min is not None and actual.gas_flow_stp_mL_min:
            counts[INLET_STP_BASIS] += 1
        elif actual.residence_time_in_channel_min is not None:
            counts[IN_CHANNEL_BASIS] += 1
        elif actual.residence_time_min is not None:
            counts[LIQUID_ONLY_BASIS] += 1
    return max(counts.items(), key=lambda item: item[1])[0] if any(counts.values()) else LIQUID_ONLY_BASIS


def _residence_time_for_basis(
    actual: ActualConditions,
    preferred_basis: str | None = None,
) -> tuple[float, str]:
    basis = normalize_residence_time_basis(preferred_basis)
    if basis == UNKNOWN_BASIS:
        basis = normalize_residence_time_basis(actual.residence_time_basis)
    if basis == UNKNOWN_BASIS:
        if actual.residence_time_inlet_min is not None and actual.gas_flow_stp_mL_min:
            basis = INLET_STP_BASIS
        elif actual.residence_time_in_channel_min is not None:
            basis = IN_CHANNEL_BASIS
        elif actual.residence_time_min is not None:
            basis = LIQUID_ONLY_BASIS
        else:
            basis = IN_CHANNEL_BASIS

    if basis == INLET_STP_BASIS:
        tau = _num(actual.residence_time_inlet_min, 0.0)
        if tau <= 0:
            volume = _num(actual.reactor_volume_mL, 0.0)
            liquid_q = _num(actual.substrate_flow_mL_min, actual.flow_rate_mL_min, 0.0)
            gas_stp = _num(actual.gas_flow_stp_mL_min, 0.0)
            if gas_stp <= 0 and actual.gas_flow_in_channel_mL_min:
                gas_stp = stp_gas_flow_from_actual(
                    actual.gas_flow_in_channel_mL_min,
                    _num(actual.temperature_C, 25.0),
                    _num(actual.BPR_bar, 0.0),
                )
            if volume > 0 and liquid_q > 0 and gas_stp > 0:
                tau = volume / (liquid_q + gas_stp)
        return _num(tau, actual.residence_time_min, 0.0), basis
    if basis == IN_CHANNEL_BASIS:
        return _num(actual.residence_time_in_channel_min, actual.residence_time_min, 0.0), basis
    return _num(actual.residence_time_min, actual.residence_time_inlet_min, actual.residence_time_in_channel_min, 0.0), basis


def _extract_number(text: str, label_pattern: str) -> Optional[float]:
    # Accept both "label = 1.23" and compact table text "label 1.23".
    # The negative lookahead prevents treating ranges like "40.0-80.0" as
    # standalone values when the label did not immediately precede the number.
    pattern = (
        label_pattern
        + r"\s*(?:[:=]|is|=)?\s*"
        + r"[-~≈]?\s*"
        + r"([-+]?\d+(?:\.\d+)?)"
        + r"(?!\s*[-–]\s*\d)"
    )
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _extract_percent(text: str, label_pattern: str) -> Optional[float]:
    return _extract_number(text, label_pattern)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return slug or "run"


def _first_order_rate(response_pct: float, tau_min: float) -> float:
    response_fraction = _bounded(response_pct / 100.0, 0.0001, 0.98)
    return -math.log(1.0 - response_fraction) / max(tau_min, 1e-9)


def _tau_for_response(response_pct: float, k_min_inv: float) -> float:
    response_fraction = _bounded(response_pct / 100.0, 0.0001, 0.95)
    return -math.log(1.0 - response_fraction) / max(k_min_inv, 1e-9)


def _conditions_for_tau(
    actual: ActualConditions,
    tau_min: float,
    residence_time_basis: str | None = None,
) -> dict[str, Any]:
    basis = normalize_residence_time_basis(residence_time_basis)
    if basis == UNKNOWN_BASIS:
        _, basis = _residence_time_for_basis(actual)
    volume = _num(actual.reactor_volume_mL, 0.0)
    liquid_q = _num(actual.substrate_flow_mL_min, actual.flow_rate_mL_min, 0.0)
    gas_actual = _num(actual.gas_flow_in_channel_mL_min, 0.0)
    gas_stp = _num(actual.gas_flow_stp_mL_min, 0.0)
    if gas_stp <= 0 and gas_actual > 0:
        gas_stp = stp_gas_flow_from_actual(
            gas_actual,
            _num(actual.temperature_C, 25.0),
            _num(actual.BPR_bar, 0.0),
        )
    if gas_actual > 0 or gas_stp > 0:
        basis = INLET_STP_BASIS

    package: dict[str, Any] = {
        "residence_time_basis": residence_time_basis_label(basis),
        "residence_time_min": round(tau_min, 3),
        "reactor_volume_mL": round(volume, 4) if volume else None,
        "temperature_C": actual.temperature_C,
        "concentration_M": actual.concentration_M,
        "BPR_bar": actual.BPR_bar,
        "tubing_ID_mm": actual.tubing_ID_mm,
        "gas_equiv_inlet": actual.gas_equiv_inlet,
    }

    if volume <= 0 or tau_min <= 0:
        package["substrate_flow_mL_min"] = round(liquid_q, 5) if liquid_q else None
        package["gas_flow_in_channel_mL_min"] = round(gas_actual, 5) if gas_actual else None
        package["gas_flow_stp_mL_min"] = round(gas_stp, 5) if gas_stp else None
        return package

    if basis == INLET_STP_BASIS and gas_stp > 0 and liquid_q > 0:
        total_current = liquid_q + gas_stp
        total_target = volume / tau_min
        liquid_target = total_target * liquid_q / total_current
        gas_stp_target = total_target * gas_stp / total_current
        if gas_actual > 0:
            gas_actual_target = gas_actual * gas_stp_target / gas_stp
        else:
            gas_actual_target = actual_gas_flow_from_stp(
                gas_stp_target,
                _num(actual.temperature_C, 25.0),
                _num(actual.BPR_bar, 0.0),
            )
        package["substrate_flow_mL_min"] = round(liquid_target, 5)
        package["gas_flow_stp_mL_min"] = round(gas_stp_target, 5)
        package["gas_flow_in_channel_mL_min"] = round(gas_actual_target, 5)
        package["residence_time_inlet_min"] = round(tau_min, 3)
        package["residence_time_in_channel_min"] = round(
            volume / max(liquid_target + gas_actual_target, 1e-9),
            3,
        )
    elif gas_actual > 0 and liquid_q > 0:
        total_current = liquid_q + gas_actual
        total_target = volume / tau_min
        liquid_target = total_target * liquid_q / total_current
        gas_actual_target = total_target * gas_actual / total_current
        package["substrate_flow_mL_min"] = round(liquid_target, 5)
        package["gas_flow_in_channel_mL_min"] = round(gas_actual_target, 5)
        package["residence_time_in_channel_min"] = round(tau_min, 3)
        if gas_stp > 0:
            gas_stp_target = gas_stp * gas_actual_target / gas_actual
            package["gas_flow_stp_mL_min"] = round(gas_stp_target, 5)
            package["residence_time_inlet_min"] = round(
                volume / max(liquid_target + gas_stp_target, 1e-9),
                3,
            )
        else:
            package["gas_flow_stp_mL_min"] = None
    else:
        liquid_target = volume / tau_min
        package["substrate_flow_mL_min"] = round(liquid_target, 5)
        package["gas_flow_in_channel_mL_min"] = None
        package["gas_flow_stp_mL_min"] = None
        package["residence_time_inlet_min"] = round(tau_min, 3)
        package["residence_time_in_channel_min"] = round(tau_min, 3)

    return package


def _apply_campaign_calibration(
    proposal: dict,
    parameter_changes: dict[str, dict[str, Any]],
    calibration: CampaignCalibration,
) -> None:
    rec = calibration.recommended_conditions
    if not rec:
        return

    field_map = {
        "residence_time_min": rec.get("residence_time_min"),
        "residence_time_in_channel_min": rec.get("residence_time_in_channel_min"),
        "residence_time_inlet_min": rec.get("residence_time_inlet_min"),
        "residence_time_basis": rec.get("residence_time_basis"),
        "flow_rate_mL_min": rec.get("substrate_flow_mL_min"),
        "reactor_volume_mL": rec.get("reactor_volume_mL"),
        "temperature_C": rec.get("temperature_C"),
        "concentration_M": rec.get("concentration_M"),
        "BPR_bar": rec.get("BPR_bar"),
        "tubing_ID_mm": rec.get("tubing_ID_mm"),
        "gas_flow_actual_mL_min": rec.get("gas_flow_in_channel_mL_min"),
        "gas_flow_sccm": rec.get("gas_flow_stp_mL_min"),
    }
    for field, value in field_map.items():
        if value is not None:
            _set_change(proposal, parameter_changes, field, value)

    _sync_liquid_streams(
        proposal,
        _num(rec.get("substrate_flow_mL_min"), proposal.get("flow_rate_mL_min"), 0.0),
        _num(rec.get("concentration_M"), proposal.get("concentration_M"), 0.0),
    )
    _sync_gas_streams(
        proposal,
        {
            "gas_flow_actual_mL_min": rec.get("gas_flow_in_channel_mL_min"),
            "gas_flow_sccm": rec.get("gas_flow_stp_mL_min"),
        },
    )
    proposal["evidence_calibration"] = calibration.model_dump()
    proposal["chemistry_notes"] = _append_note(
        proposal.get("chemistry_notes", ""),
        "Evidence calibration: measured campaign data override unsupported residence-time intensification.",
    )
    reasoning = proposal.setdefault("reasoning_per_field", {})
    reasoning["residence_time_min"] = (
        "Evidence-calibrated from measured conversion/product response versus "
        f"{rec.get('residence_time_basis', 'the selected residence-time basis')}; "
        "the original short residence-time design is "
        "treated as an aggressive screen only."
    )


def _detect_failure_modes(
    proposal: dict,
    outcomes: ExperimentalOutcomes,
    target_yield_pct: float,
    target_conversion_pct: float,
    target_selectivity_pct: float,
) -> list[str]:
    modes: list[str] = []
    outcomes = _normalize_outcomes(outcomes)
    conversion = outcomes.conversion_pct
    yield_pct = outcomes.yield_pct
    selectivity = outcomes.selectivity_pct

    if conversion is not None and conversion < min(target_conversion_pct, 85.0):
        modes.append("kinetic_underconversion")

    if conversion is None and yield_pct is not None and yield_pct < target_yield_pct:
        modes.append("low_yield_unknown_conversion")

    selectivity_low = selectivity is not None and selectivity < target_selectivity_pct
    yield_low = yield_pct is not None and yield_pct < target_yield_pct
    conversion_ok = conversion is None or conversion >= 75.0
    if conversion_ok and (selectivity_low or (yield_low and selectivity is not None and selectivity < 90.0)):
        modes.append("selectivity_loss")

    if outcomes.clogging_observed or outcomes.precipitation_observed:
        modes.append("solubility_or_fouling")

    if outcomes.pressure_drift_bar is not None and outcomes.pressure_drift_bar >= 2.0:
        modes.append("pressure_instability")
    elif outcomes.pressure_bar is not None:
        bpr = _num(proposal.get("BPR_bar"), 0.0)
        if bpr and outcomes.pressure_bar > bpr + 3.0:
            modes.append("pressure_instability")

    gas_state = (outcomes.gas_liquid_stability or "").strip().lower()
    if gas_state and gas_state not in {"stable", "none", "n/a", "na", "unknown"}:
        modes.append("gas_liquid_instability")

    return _dedupe(modes)


def _objective_score(outcomes: ExperimentalOutcomes) -> float:
    outcomes = _normalize_outcomes(outcomes)
    y = _num(outcomes.yield_pct, 0.0)
    conv = _num(outcomes.conversion_pct, y)
    sel = _num(outcomes.selectivity_pct, y)
    pressure_penalty = 0.0
    if outcomes.pressure_drift_bar:
        pressure_penalty += min(outcomes.pressure_drift_bar * 2.0, 15.0)
    if outcomes.clogging_observed:
        pressure_penalty += 20.0
    if outcomes.precipitation_observed:
        pressure_penalty += 12.0
    score = 0.5 * y + 0.25 * conv + 0.25 * sel - pressure_penalty
    return round(_bounded(score, lower=0.0, upper=100.0), 2)


def _build_diagnosis(outcomes: ExperimentalOutcomes, modes: list[str], status: str) -> str:
    outcomes = _normalize_outcomes(outcomes)
    if status == "converged":
        return "Experimental feedback meets the current target; use the next cycle for reproducibility confirmation."
    if not modes:
        return "No single dominant failure mode was detected from the submitted measurements."

    clauses = []
    if "kinetic_underconversion" in modes:
        clauses.append(f"conversion is below target ({outcomes.conversion_pct}%)")
    if "low_yield_unknown_conversion" in modes:
        clauses.append(f"product/yield is below target ({outcomes.yield_pct}%) and conversion basis is incomplete")
    if "selectivity_loss" in modes:
        clauses.append("conversion is acceptable but yield/selectivity is not")
    if "solubility_or_fouling" in modes:
        clauses.append("solids, precipitation, or clogging were observed")
    if "pressure_instability" in modes:
        clauses.append("pressure drift indicates unstable hydraulic operation")
    if "gas_liquid_instability" in modes:
        clauses.append("gas-liquid contacting was not stable")
    return "Diagnosis: " + "; ".join(clauses) + "."


def _build_revision_instructions(
    failure_modes: list[str],
    parameter_changes: dict[str, dict[str, Any]],
    recommended_actions: list[str],
) -> str:
    changes = []
    for field, change in parameter_changes.items():
        if change.get("changed"):
            changes.append(f"{field}: {change.get('old')} -> {change.get('new')}")
    mode_text = ", ".join(failure_modes) if failure_modes else "target confirmation"
    action_text = " ".join(recommended_actions)
    change_text = "; ".join(changes) if changes else "no major numeric change"
    return f"Refine for {mode_text}. {change_text}. {action_text}".strip()


def _build_next_experiment(proposal: dict, status: str, version: int) -> dict[str, Any]:
    gas_liquid = _gas_liquid_summary(proposal)
    package = {
        "design_version": version,
        "status": status,
        "residence_time_min": proposal.get("residence_time_min"),
        "residence_time_in_channel_min": proposal.get("residence_time_in_channel_min"),
        "residence_time_inlet_min": proposal.get("residence_time_inlet_min"),
        "residence_time_basis": proposal.get("residence_time_basis"),
        "flow_rate_mL_min": proposal.get("flow_rate_mL_min"),
        "reactor_volume_mL": proposal.get("reactor_volume_mL"),
        "temperature_C": proposal.get("temperature_C"),
        "concentration_M": proposal.get("concentration_M"),
        "BPR_bar": proposal.get("BPR_bar"),
        "tubing_ID_mm": proposal.get("tubing_ID_mm"),
        "pre_reactor_steps": proposal.get("pre_reactor_steps", []),
        "acceptance_criteria": {
            "yield_pct_min": DEFAULT_TARGET_YIELD,
            "conversion_pct_min": DEFAULT_TARGET_CONVERSION,
            "selectivity_pct_min": DEFAULT_TARGET_SELECTIVITY,
            "pressure_drift_bar_max": 2.0,
            "no_clogging": True,
        },
    }
    package.update(gas_liquid)
    if proposal.get("evidence_calibration"):
        package["evidence_calibration"] = proposal["evidence_calibration"]
        recommended = proposal["evidence_calibration"].get("recommended_conditions") or {}
        for key, value in recommended.items():
            if value is not None:
                package[key] = value
    return package


def _sync_liquid_streams(proposal: dict, total_flow_rate: float, concentration: float) -> None:
    streams = proposal.get("streams") or []
    liquid_streams = [
        s for s in streams
        if str(s.get("phase", "liquid")).lower() != "gas"
        and "quench" not in str(s.get("pump_role", "")).lower()
    ]
    if not liquid_streams:
        return

    current_sum = sum(_num(s.get("flow_rate_mL_min"), 0.0) for s in liquid_streams)
    for stream in liquid_streams:
        if current_sum > 0:
            frac = _num(stream.get("flow_rate_mL_min"), 0.0) / current_sum
        else:
            frac = 1.0 / len(liquid_streams)
        stream["flow_rate_mL_min"] = round(total_flow_rate * frac, 5)
        if stream.get("concentration_M") is not None:
            stream["concentration_M"] = round(concentration, 4)
        stream["reasoning"] = _append_note(
            stream.get("reasoning", ""),
            "Closed-loop update: flow and concentration synchronized to the revised next-run design.",
        )


def _sync_gas_streams(proposal: dict, gas_update: dict[str, Any]) -> None:
    streams = proposal.get("streams") or []
    for stream in streams:
        if str(stream.get("phase", "")).lower() != "gas":
            continue
        if gas_update.get("gas_flow_actual_mL_min") is not None:
            stream["gas_flow_actual_mL_min"] = gas_update["gas_flow_actual_mL_min"]
            stream["flow_rate_mL_min"] = gas_update["gas_flow_actual_mL_min"]
        if gas_update.get("gas_flow_sccm") is not None:
            stream["gas_flow_sccm"] = gas_update["gas_flow_sccm"]
        stream["reasoning"] = _append_note(
            stream.get("reasoning", ""),
            "Closed-loop update: gas flow scaled with liquid flow to preserve O2 equivalents while targeting the revised residence-time basis.",
        )


def _normalize_outcomes(outcomes: ExperimentalOutcomes) -> ExperimentalOutcomes:
    data = outcomes.model_dump()
    if data.get("yield_pct") is None and data.get("product_pct") is not None:
        data["yield_pct"] = data["product_pct"]
    if data.get("conversion_pct") is None and data.get("starting_material_pct") is not None:
        data["conversion_pct"] = _bounded(100.0 - float(data["starting_material_pct"]), 0.0, 100.0)
    return ExperimentalOutcomes(**data)


def _revised_tau_for_underconversion(
    base_tau: float,
    conversion_pct: Optional[float],
    target_conversion_pct: float,
) -> float:
    if conversion_pct is None or conversion_pct <= 0 or conversion_pct >= target_conversion_pct:
        return _bounded(base_tau * 1.5, lower=1.0, upper=720.0)

    observed_fraction = _bounded(conversion_pct / 100.0, 0.01, 0.98)
    target_fraction = _bounded(target_conversion_pct / 100.0, 0.2, 0.95)
    try:
        k_obs = -math.log(1.0 - observed_fraction) / max(base_tau, 1e-6)
        raw_tau = -math.log(1.0 - target_fraction) / max(k_obs, 1e-9)
    except (ValueError, ZeroDivisionError):
        raw_tau = base_tau * 1.5

    # Avoid one-cycle jumps to impractically long coils. The next experiment is
    # a screen, so move at most 4x from the observed failed point.
    return _bounded(min(raw_tau, base_tau * 4.0), lower=1.0, upper=720.0)


def _revised_tau_for_low_yield(
    base_tau: float,
    yield_pct: Optional[float],
    target_yield_pct: float,
) -> float:
    if yield_pct is None or yield_pct <= 0:
        return _bounded(base_tau * 1.5, lower=1.0, upper=720.0)
    multiplier = _bounded((target_yield_pct / max(yield_pct, 1e-6)) * 1.1, 1.2, 4.0)
    return _bounded(base_tau * multiplier, lower=1.0, upper=720.0)


def _scaled_flows_for_tau(
    proposal: dict,
    volume_mL: float,
    tau_min: float,
    fallback_liquid_q: float,
    actual: ActualConditions,
    residence_time_basis: str | None = None,
) -> tuple[float, dict[str, Any]]:
    basis = normalize_residence_time_basis(residence_time_basis)
    if basis == UNKNOWN_BASIS:
        _, basis = _residence_time_for_basis(actual)
    gas_actual, gas_stp = _extract_gas_flows(proposal)
    gas_actual = _num(actual.gas_flow_in_channel_mL_min, gas_actual, 0.0)
    gas_stp = _num(actual.gas_flow_stp_mL_min, gas_stp, 0.0)
    if gas_stp <= 0 and gas_actual > 0:
        gas_stp = stp_gas_flow_from_actual(
            gas_actual,
            _num(actual.temperature_C, proposal.get("temperature_C"), 25.0),
            _num(actual.BPR_bar, proposal.get("BPR_bar"), 0.0),
        )
    if gas_actual > 0 or gas_stp > 0:
        basis = INLET_STP_BASIS
    liquid_q = _num(actual.substrate_flow_mL_min, actual.flow_rate_mL_min, fallback_liquid_q)

    if gas_actual <= 0 and not (basis == INLET_STP_BASIS and gas_stp > 0):
        return _safe_flow_from_volume(volume_mL, tau_min, fallback=fallback_liquid_q), {}

    if basis == INLET_STP_BASIS and gas_stp > 0:
        total_current = max(liquid_q + gas_stp, 1e-9)
        liquid_fraction = liquid_q / total_current
        gas_stp_fraction = gas_stp / total_current
        total_target = volume_mL / max(tau_min, 1e-9)
        liquid_target = total_target * liquid_fraction
        gas_stp_target = total_target * gas_stp_fraction
        if gas_actual > 0:
            gas_actual_target = gas_actual * gas_stp_target / gas_stp
        else:
            gas_actual_target = actual_gas_flow_from_stp(
                gas_stp_target,
                _num(actual.temperature_C, proposal.get("temperature_C"), 25.0),
                _num(actual.BPR_bar, proposal.get("BPR_bar"), 0.0),
            )
        gas_update = {
            "gas_flow_actual_mL_min": round(gas_actual_target, 5),
            "gas_flow_sccm": round(gas_stp_target, 5),
            "residence_time_inlet_min": round(tau_min, 3),
            "residence_time_in_channel_min": round(
                volume_mL / max(liquid_target + gas_actual_target, 1e-9),
                3,
            ),
        }
        return liquid_target, gas_update

    total_current = max(liquid_q + gas_actual, 1e-9)
    liquid_fraction = liquid_q / total_current
    gas_fraction = gas_actual / total_current
    total_target = volume_mL / max(tau_min, 1e-9)
    liquid_target = total_target * liquid_fraction
    gas_actual_target = total_target * gas_fraction

    gas_update = {
        "gas_flow_actual_mL_min": round(gas_actual_target, 5),
    }
    if gas_stp > 0 and gas_actual > 0:
        gas_update["gas_flow_sccm"] = round(gas_stp * (gas_actual_target / gas_actual), 5)
    elif gas_actual_target > 0:
        gas_update["gas_flow_sccm"] = round(
            stp_gas_flow_from_actual(
                gas_actual_target,
                _num(actual.temperature_C, proposal.get("temperature_C"), 25.0),
                _num(actual.BPR_bar, proposal.get("BPR_bar"), 0.0),
            ),
            5,
        )
    gas_update["residence_time_in_channel_min"] = round(
        volume_mL / max(liquid_target + gas_actual_target, 1e-9), 3
    )
    if gas_update.get("gas_flow_sccm"):
        gas_update["residence_time_inlet_min"] = round(
            volume_mL / max(liquid_target + gas_update["gas_flow_sccm"], 1e-9), 3
        )
    return liquid_target, gas_update


def _extract_gas_flows(proposal: dict) -> tuple[float, float]:
    for stream in proposal.get("streams") or []:
        if str(stream.get("phase", "")).lower() == "gas":
            return (
                _num(stream.get("gas_flow_actual_mL_min"), stream.get("flow_rate_mL_min"), 0.0),
                _num(stream.get("gas_flow_sccm"), 0.0),
            )
    return 0.0, 0.0


def _gas_liquid_summary(proposal: dict) -> dict[str, Any]:
    liquid_q = _num(proposal.get("flow_rate_mL_min"), 0.0)
    gas_actual, gas_stp = _extract_gas_flows(proposal)
    volume = _num(proposal.get("reactor_volume_mL"), 0.0)
    if gas_actual <= 0 or volume <= 0:
        return {}

    if gas_stp <= 0:
        gas_stp = stp_gas_flow_from_actual(
            gas_actual,
            _num(proposal.get("temperature_C"), 25.0),
            _num(proposal.get("BPR_bar"), 0.0),
        )
    basis = INLET_STP_BASIS
    summary = {
        "residence_time_basis": residence_time_basis_label(basis),
        "substrate_flow_mL_min": round(liquid_q, 5),
        "gas_flow_in_channel_mL_min": round(gas_actual, 5),
        "gas_flow_stp_mL_min": round(gas_stp, 5) if gas_stp else None,
        "residence_time_in_channel_min": round(volume / max(liquid_q + gas_actual, 1e-9), 3),
    }
    if gas_stp:
        summary["residence_time_inlet_min"] = round(volume / max(liquid_q + gas_stp, 1e-9), 3)
    if basis == INLET_STP_BASIS and summary.get("residence_time_inlet_min") is not None:
        summary["residence_time_min"] = summary["residence_time_inlet_min"]
    else:
        summary["residence_time_min"] = summary["residence_time_in_channel_min"]
    return summary


def _set_change(proposal: dict, changes: dict[str, dict[str, Any]], field: str, value: Any) -> None:
    old = proposal.get(field)
    proposal[field] = value
    changes[field] = {
        "old": old,
        "new": value,
        "changed": _format_value(old) != _format_value(value),
    }


def _safe_flow_from_volume(volume_mL: float, tau_min: float, fallback: float) -> float:
    if tau_min <= 0:
        return fallback
    q = volume_mL / tau_min
    return q if q > 0 else fallback


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _append_note(existing: str, note: str) -> str:
    existing = str(existing or "").strip()
    note = str(note or "").strip()
    if not existing:
        return note
    if note in existing:
        return existing
    return f"{existing}\n\n{note}"


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _bounded(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _num(*values: Any) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
