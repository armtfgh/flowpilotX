"""Deterministic evidence summaries scoped to one catalyst system and batch mode."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math

from .history import CONDITION_FIELDS
from .observations import AnalyticalOutcome


FACTOR_FIELDS = {
    "concentration": ("concentration_mol_L",),
    "catalyst_loading": ("catalyst_loading_mol_pct",),
    "solvent": ("solvent_1", "solvent_2", "solvent_1_percentage", "solvent_ratio_basis"),
    "temperature": ("temperature_C",),
    "light": ("light_source_reported",),
    "oxidant": ("oxidant_reported",),
    "batch_duration": ("batch_duration_h",),
}
COMPARISON_SCOPE = (
    "Matched on other recorded fields only; unknown solvent-ratio basis, gas delivery, "
    "illumination, chronology, repeat relationships and other missing metadata remain "
    "unknown. Differences are descriptive observations, not causal effects or flow kinetics."
)


def assess_objective(outcome: AnalyticalOutcome, *, minimum_sulfoxide_pct: float,
                     maximum_sulfone_pct: float) -> dict:
    """Evaluate declared endpoints, never two-product selectivity as yield."""
    for value in (minimum_sulfoxide_pct, maximum_sulfone_pct):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Endpoint thresholds must be finite nonnegative numbers")
    measured = outcome.sulfoxide_pct is not None and outcome.sulfone_pct is not None
    return {"meets_reported_endpoints": (outcome.sulfoxide_pct >= minimum_sulfoxide_pct
             and outcome.sulfone_pct <= maximum_sulfone_pct) if measured else None,
            "minimum_sulfoxide_pct": minimum_sulfoxide_pct, "maximum_sulfone_pct": maximum_sulfone_pct,
            "analytical_review_flags": outcome.flags,
            "status": "descriptive reported-value check; not validated yield or overall selectivity"}


def _key(conditions: dict, exclude: set[str]) -> str:
    return json.dumps({k: v for k, v in sorted(conditions.items()) if k not in exclude},
                      sort_keys=True, ensure_ascii=False, allow_nan=False)


def _compact(record: dict) -> dict:
    return deepcopy({"record_id": record["record_id"], "source_entry": record["source_entry"],
            "catalyst_system": record["catalyst_system"],
            "version_membership": record["version_membership"], "experiment_mode": "batch",
            "conditions": record["reported_actual_conditions"], "outcomes": record["outcomes"],
            "source_record_ids": [s["source_record_id"] for s in record["source_representations"]],
            "identity_scope": record["identity_mapping"]["status"]})


def _factor_value_available(conditions: dict, factor: str) -> bool:
    # Pure-solvent rows have unreported secondary-solvent/ratio fields. Keep
    # these unknowns in the joint solvent descriptor instead of inventing 100%.
    required = ("solvent_1",) if factor == "solvent" else FACTOR_FIELDS[factor]
    return all(conditions.get(field) is not None for field in required)


def _factor_comparisons(records: list[dict]) -> dict:
    comparisons = {}
    for factor, fields in FACTOR_FIELDS.items():
        grouped = defaultdict(list)
        for record in records:
            conditions = record["reported_actual_conditions"]
            if _factor_value_available(conditions, factor):
                grouped[_key(conditions, set(fields))].append(record)
        comparisons[factor] = []
        for fixed_key, group in sorted(grouped.items()):
            levels = {_key({field: r["reported_actual_conditions"].get(field) for field in fields}, set())
                      for r in group}
            if len(levels) < 2:
                continue
            ordered = sorted(group, key=lambda r: r["record_id"])
            group_hash = hashlib.sha256(f"{factor}:{fixed_key}".encode()).hexdigest()[:16]
            comparisons[factor].append({
                "group_id": f"batch-factor-{factor}:{group_hash}", "factor": factor,
                "varied_recorded_fields": list(fields),
                "same_other_recorded_conditions": json.loads(fixed_key),
                "reported_levels": [json.loads(level) for level in sorted(levels)],
                "record_ids": [r["record_id"] for r in ordered],
                "records": [_compact(r) for r in ordered],
                "comparison_scope": COMPARISON_SCOPE,
            })
    return comparisons


def _complete_visibility(record: dict) -> dict:
    """Full canonical record with explicit interpretation eligibility.

    This retains both source representations for disputes rather than silently
    choosing a displayed workbook value as the resolved observation.
    """
    result = deepcopy(record)
    result["interpretation_eligible"] = not bool(record["disputed_source_fields"])
    result["interpretation_exclusion_reasons"] = (
        ["Conflicting export values require source adjudication"] if record["disputed_source_fields"] else [])
    result["visibility_status"] = "retained_in_full_regardless_of_match_or_analytical_flags"
    return result


def _coverage_ledger(all_records: list[dict], matched: list[dict], comparisons: dict) -> dict:
    matched_ids = {r["record_id"] for r in matched}
    memberships = defaultdict(list)
    for factor, groups in comparisons.items():
        for group in groups:
            for rid in group["record_ids"]:
                memberships[rid].append({"factor": factor, "group_id": group["group_id"]})
    rows = []
    for record in all_records:
        rid = record["record_id"]
        disputed = bool(record["disputed_source_fields"])
        rows.append({
            "record_id": rid, "source_entry": record["source_entry"], "visible": True,
            "source_record_ids": [s["source_record_id"] for s in record["source_representations"]],
            "interpretation_eligible": not disputed,
            "interpretation_exclusion_reasons": ["Conflicting export values require source adjudication"] if disputed else [],
            "in_matched_records": rid in matched_ids,
            "factor_comparison_memberships": memberships[rid],
            "factors_without_reported_value": [f for f in FACTOR_FIELDS
                if not _factor_value_available(record["reported_actual_conditions"], f)],
            "analytical_flags": list(record["outcomes"]["flags"]),
        })
    return {
        "canonical_record_count": len(all_records), "visible_record_count": len(rows),
        "interpretation_eligible_count": sum(row["interpretation_eligible"] for row in rows),
        "held_for_source_resolution_count": sum(not row["interpretation_eligible"] for row in rows),
        "matched_filter_record_count": len(matched),
        "source_representation_count": sum(len(row["source_record_ids"]) for row in rows),
        "record_count_in_any_factor_comparison": sum(bool(row["factor_comparison_memberships"]) for row in rows),
        "records_without_factor_comparison": [row["record_id"] for row in rows if not row["factor_comparison_memberships"]],
        "factor_coverage": {factor: {"comparison_group_count": len(groups),
            "record_ids": sorted({rid for group in groups for rid in group["record_ids"]})}
            for factor, groups in comparisons.items()},
        "records": rows,
        "policy": "Every canonical entry is visible. A filter, missing comparison partner, analytical flag or source dispute never removes it from all_records. Visibility does not imply ranking, causal attribution or flow-model use.",
    }


def build_evidence_packet(history: dict, system: str = "homogeneous", match: dict | None = None) -> dict:
    if system not in {"homogeneous", "heterogeneous"}:
        raise ValueError("Select one catalyst system; pooled evidence is not supported")
    match = {} if match is None else dict(match)
    allowed = set(CONDITION_FIELDS) | {"solvent_ratio_basis"}
    if set(match) - allowed:
        raise ValueError(f"Unknown condition match keys: {sorted(set(match) - allowed)}")
    all_records = history["systems"][system]["records"]
    if any(r["experiment_mode"] != "batch" or r["catalyst_system"] != system for r in all_records):
        raise ValueError("This evidence packet accepts only batch records from the selected system")
    excluded = [{"record_id": r["record_id"], "reason": "Conflicting export values require source adjudication",
                 "disputed_fields": r["disputed_source_fields"]}
                for r in all_records if r["disputed_source_fields"]]
    records = [r for r in all_records if not r["disputed_source_fields"]]
    matched = [r for r in records if all(r["reported_actual_conditions"].get(k) == v for k, v in match.items())]
    factor_comparisons = _factor_comparisons(records)
    time_groups, exact_groups, oxidant_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for record in records:
        conditions = record["reported_actual_conditions"]
        exact_groups[_key(conditions, set())].append(record)
        time_groups[_key(conditions, {"batch_duration_h"})].append(record)
        oxidant_groups[_key(conditions, {"oxidant_reported"})].append(record)
    series = []
    for key, group in sorted(time_groups.items()):
        durations = {r["reported_actual_conditions"]["batch_duration_h"] for r in group}
        if len(durations - {None}) < 2:
            continue
        sorted_group = sorted(group, key=lambda r: (r["reported_actual_conditions"]["batch_duration_h"] is None,
                                      r["reported_actual_conditions"]["batch_duration_h"] or 0, r["record_id"]))
        points = [{"record_id": r["record_id"], "batch_duration_h": r["reported_actual_conditions"]["batch_duration_h"],
                   "residual_sulfide_pct": r["outcomes"]["residual_sulfide_pct"],
                   "sulfoxide_pct": r["outcomes"]["sulfoxide_pct"], "sulfone_pct": r["outcomes"]["sulfone_pct"],
                   "flags": r["outcomes"]["flags"]} for r in sorted_group]
        decreases = [{"earlier_record_id": a["record_id"], "later_record_id": b["record_id"]}
                     for i, a in enumerate(points) for b in points[i + 1:]
                     if a["batch_duration_h"] is not None and b["batch_duration_h"] is not None
                     and a["batch_duration_h"] < b["batch_duration_h"]
                     and a["sulfoxide_pct"] is not None and b["sulfoxide_pct"] is not None
                     and a["sulfoxide_pct"] > b["sulfoxide_pct"]]
        series.append({"group_id": "batch-time-group:" + hashlib.sha256(key.encode()).hexdigest()[:16],
                       "same_other_recorded_conditions": json.loads(key), "points": points,
                       "observed_decreases_at_longer_time": decreases,
                       "relationship": "Not a confirmed time-course, independent repeat series or flow kinetic model"})
    oxidant_comparisons = []
    for key, group in sorted(oxidant_groups.items()):
        oxidants = {r["reported_actual_conditions"]["oxidant_reported"] for r in group}
        if len(oxidants - {None}) > 1:
            oxidant_comparisons.append({"same_other_recorded_conditions": json.loads(key),
                "records": [_compact(r) for r in group],
                "comparison_scope": "Matched on recorded conditions only; unrecorded gas delivery, illumination, chronology and independence remain unknown"})
    return {"schema_version": "khu-evidence-v2", "history_schema_version": history["schema_version"],
            "catalyst_system": system, "experiment_mode": "batch", "match": match,
            "source_hashes": [{"path": s["path"], "sha256": s["sha256"]} for s in history["sources"]],
            "record_count": len(all_records), "matched_records": [_compact(r) for r in matched],
            "all_records": [_complete_visibility(r) for r in all_records],
            "coverage_ledger": _coverage_ledger(all_records, matched, factor_comparisons),
            "recorded_condition_factor_comparisons": factor_comparisons,
            "time_groups": series, "recorded_condition_oxidant_comparisons": oxidant_comparisons,
            "same_recorded_condition_groups": [{"record_ids": [r["record_id"] for r in group],
                "independent_repeat_status": "unknown"} for _, group in sorted(exact_groups.items()) if len(group) > 1],
            "review_flags": dict(Counter(flag for r in records for flag in r["outcomes"]["flags"])),
            "excluded_from_interpreted_summary": excluded,
            "interpretation_limits": [
                "No yield-based exclusion, normalization or clipping is applied.",
                "Slide component identity and analytical method mapping across every raw row remain provisional.",
                "Component sum is reported analytical recovery, not independently measured mass balance.",
                "Two-product selectivity omits other and unknown components and is not the primary yield objective.",
                "Batch observations cannot establish flow kinetics, photon-flux equivalence or gas-transfer performance.",
                "Entry order does not establish chronology; equal conditions do not establish independent repeats.",
                "Groups control only recorded columns; missing operational metadata precludes fully controlled causal claims.",
                "Joint solvent comparisons retain unreported solvent proportions/bases; they do not assume a missing proportion is 100%.",
                "Complete record visibility does not imply every row identifies a separate factor effect or determines a design nomination.",
            ], "flow_kinetics": None, "predicted_flow_yield_pct": None}
