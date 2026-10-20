"""Qualitative feedback only; no monotonic yield/time model or executable update."""
from __future__ import annotations

import math
from .observations import ExperimentObservation


def qualitative_feedback(observation: ExperimentObservation, *, sulfone_limit_pct: float,
                         residual_sulfide_review_pct: float) -> dict:
    """Use caller-declared review thresholds without fitting or inventing settings."""
    for name, value in [("sulfone_limit_pct", sulfone_limit_pct),
                        ("residual_sulfide_review_pct", residual_sulfide_review_pct)]:
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative percentage")
    outcome = observation.outcomes
    diagnoses, hypotheses = [], []
    if outcome.sulfone_pct is None or outcome.residual_sulfide_pct is None or outcome.sulfoxide_pct is None:
        diagnoses.append("incomplete_response_measurement")
    if outcome.sulfone_pct is not None and outcome.sulfone_pct > sulfone_limit_pct:
        diagnoses.append("sulfone_above_declared_limit")
        hypotheses.append({"hypothesis": "Test shorter exposure or lower oxygen delivery in separately controlled comparisons.",
                           "uncertainty": "Overoxidation is a possible explanation; selectivity and transport effects are unverified."})
    if outcome.residual_sulfide_pct is not None and outcome.residual_sulfide_pct > residual_sulfide_review_pct:
        diagnoses.append("residual_sulfide_above_declared_review_level")
        hypotheses.append({"hypothesis": "Compare exposure and oxygen delivery while measuring sulfone independently.",
                           "uncertainty": "Incomplete reaction may reflect illumination, mixing, gas transfer or chemistry; longer time need not improve sulfoxide yield."})
    if outcome.flags:
        diagnoses.append("analytical_review_needed")
    if observation.actual_conditions is None:
        diagnoses.append("actual_conditions_missing")
    if observation.experiment_mode == "batch":
        diagnoses.append("batch_evidence_not_flow_kinetics")
    return {"schema_version": "khu-feedback-v1", "observation_id": observation.experiment_id,
            "experiment_mode": observation.experiment_mode,
            "status": "hypotheses_for_review", "diagnoses": diagnoses,
            "hypotheses": hypotheses, "outcomes": outcome.to_dict(),
            "thresholds": {"sulfone_limit_pct": sulfone_limit_pct,
                           "residual_sulfide_review_pct": residual_sulfide_review_pct},
            "parameter_changes": {}, "fitted_kinetics": None,
            "caveat": "No measured flow yield is predicted; a new screen must satisfy the separately confirmed apparatus and sampling constraints."}
