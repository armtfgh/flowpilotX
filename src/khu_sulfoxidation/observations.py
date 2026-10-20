"""Study observations with unaltered analytical percentages and explicit unknowns.

No response normalization, clipping, missing-to-zero coercion or kinetic fitting.
Reported component percentages need not close to 100 percent.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from typing import Any


def reported_number(value: Any) -> float | None:
    """Interpret finite numeric cells, retaining the original separately."""
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (ValueError, TypeError):
        return None
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True)
class AnalyticalOutcome:
    residual_sulfide_pct: float | None = None
    sulfoxide_pct: float | None = None
    sulfone_pct: float | None = None
    other_products_pct: float | None = None
    unknown_components_pct: float | None = None
    reported_component_sum_pct: float | None = None
    reported_conversion_pct: float | None = None
    reported_two_product_selectivity_pct: float | None = None
    raw_reported: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "raw_reported" or value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number or None")

    @property
    def apparent_conversion_pct(self) -> float | None:
        return None if self.residual_sulfide_pct is None else 100 - self.residual_sulfide_pct

    @property
    def two_product_selectivity_pct(self) -> float | None:
        if self.sulfoxide_pct is None or self.sulfone_pct is None:
            return None
        total = self.sulfoxide_pct + self.sulfone_pct
        return None if total <= 0 else 100 * self.sulfoxide_pct / total

    @property
    def component_sum_pct(self) -> float | None:
        values = [self.residual_sulfide_pct, self.sulfoxide_pct, self.sulfone_pct,
                  self.other_products_pct, self.unknown_components_pct]
        return None if any(v is None for v in values) else sum(values)

    @property
    def flags(self) -> list[str]:
        flags = []
        values = [self.residual_sulfide_pct, self.sulfoxide_pct, self.sulfone_pct,
                  self.other_products_pct, self.unknown_components_pct]
        if any(v is not None and not 0 <= v <= 100 for v in values):
            flags.append("component_percentage_outside_0_100")
        if any(v is None for v in values):
            flags.append("incomplete_component_reporting")
        total = self.component_sum_pct
        if total is not None and total > 100:
            flags.append("component_sum_above_100")
        for calculated, reported, label, tolerance in [
            (total, self.reported_component_sum_pct, "component_sum_arithmetic_mismatch", 1e-8),
            (self.apparent_conversion_pct, self.reported_conversion_pct, "conversion_arithmetic_mismatch", 1e-8),
            (self.two_product_selectivity_pct, self.reported_two_product_selectivity_pct,
             "two_product_selectivity_arithmetic_mismatch", 0.0051),
        ]:
            if calculated is not None and reported is not None and abs(calculated - reported) > tolerance:
                flags.append(label)
        if (self.sulfoxide_pct is not None and self.apparent_conversion_pct is not None
                and self.sulfoxide_pct > self.apparent_conversion_pct):
            flags.append("sulfoxide_percentage_exceeds_apparent_conversion")
        return flags

    def to_dict(self) -> dict:
        return {**asdict(self), "calculated": {
            "apparent_conversion_pct": self.apparent_conversion_pct,
            "two_product_selectivity_pct": self.two_product_selectivity_pct,
            "component_sum_pct": self.component_sum_pct,
        }, "flags": self.flags, "definitions": {
            "apparent_conversion_pct": "100 - residual_sulfide_pct; not an independent measurement",
            "two_product_selectivity_pct": "100 * sulfoxide / (sulfoxide + sulfone); excludes other and unknown components",
            "component_sum_pct": "Sum of all five reported components; not gravimetric recovery; unknown if any is missing",
        }}

    @classmethod
    def from_dict(cls, value: dict) -> "AnalyticalOutcome":
        return cls(**{name: value[name] for name in cls.__dataclass_fields__ if name in value})


@dataclass(frozen=True)
class ExperimentObservation:
    experiment_id: str
    experiment_mode: str
    proposed_conditions: dict[str, Any] = field(default_factory=dict)
    actual_conditions: dict[str, Any] | None = None
    outcomes: AnalyticalOutcome = field(default_factory=AnalyticalOutcome)
    analytical_method: str | None = None
    internal_standard: str | None = None
    uncertainty: dict[str, Any] | None = None
    sample_id: str | None = None
    source_ids: list[str] = field(default_factory=list)
    design_version: str | None = None
    repeat_of: str | None = None
    repeat_type: str | None = None
    operational_observations: dict[str, Any] = field(default_factory=lambda: {
        "precipitation_observed": None, "clogging_observed": None,
        "gas_behavior": None, "pressure_observation": None,
        "temperature_observation": None, "post_collection_oxidation_observed": None,
    })
    deviations: str | None = None
    sample_handling: str | None = None
    collected_amount: dict[str, Any] | None = None

    def __post_init__(self):
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if self.experiment_mode not in {"batch", "flow"}:
            raise ValueError("experiment_mode must be batch or flow")
        if not isinstance(self.outcomes, AnalyticalOutcome):
            raise TypeError("outcomes must be AnalyticalOutcome")
        if self.experiment_mode == "batch" and self.actual_conditions and any(
            "residence_time" in key for key in self.actual_conditions
        ):
            raise ValueError("Batch duration must not be labelled flow residence time")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["outcomes"] = self.outcomes.to_dict()
        return data


def results_template(condition_id: str, proposed_conditions: dict,
                     design_version: str | None = None) -> dict:
    """Create an empty return record; proposals never become actual settings."""
    return ExperimentObservation(condition_id, "flow", deepcopy(proposed_conditions),
                                 design_version=design_version).to_dict()
