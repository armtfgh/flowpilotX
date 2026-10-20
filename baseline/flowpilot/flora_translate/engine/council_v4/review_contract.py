"""Evidence boundaries shared by candidate generation, review and selection."""
from __future__ import annotations

import json


REVIEW_POLICY = """
AUTHORITATIVE REVIEW POLICY (takes precedence over generic screening heuristics):
Follow the researcher objective and authoritative inventory. A first-screen brief
does not require a demonstrated improvement over batch or the shortest residence
time. Intensification is a hypothesis unless explicitly requested and supported.
Read model_validity before using conversion, mixing or hydraulic estimates.
Unvalidated or inapplicable models provide warnings and proposed measurements,
not hard feasibility gates or evidence of conversion. Preserve hard inventory,
material compatibility, feed-identity, topology and pressure-limit conflicts.
Do not invent missing catalyst loading, kinetics, holdup or safety measurements.
For tau_kinetics_min the convention is t90: X=1-exp(-ln(10)*tau/t90).
Chemistry may propose component_placements as a list of objects with component,
placement='stationary_reactor_phase', equipment_id and reason. This requests a
checked repair on an existing packed reactor, not permission to invent equipment,
change chemical identity or erase a rejection. Every accepted repair is realized
and reviewed again. Report unsupported repairs as unresolved blocking_issues.
"""


def validated_model(candidate: dict, name: str) -> bool:
    evidence = (candidate.get("model_validity") or {}).get(name) or {}
    return (evidence.get("applicable") is True
            and evidence.get("validated") is True
            and bool(evidence.get("evidence")))


def candidate_fingerprint(candidate: dict) -> str:
    """Operating identity, excluding IDs, scores, rationale and provenance."""
    keys = ("equipment_id", "reactor_type", "tau_min", "Q_mL_min", "d_mm",
            "V_R_mL", "concentration_M", "temperature_C", "BPR_bar",
            "tubing_material", "gas_flow_sccm", "stationary_components")
    payload = {key: candidate.get(key) for key in keys}
    payload["feeds"] = [{key: stream.get(key) for key in (
        "stream_label", "contents", "phase", "solvent", "concentration_M", "flow_rate_mL_min",
        "gas_flow_sccm", "gas_reagent_mole_fraction", "molar_equiv", "pump_equipment_id", "introduction_stage")}
        for stream in candidate.get("streams", [])]
    def rounded(value):
        if isinstance(value, float):
            return round(value, 8)
        if isinstance(value, dict):
            return {key: rounded(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rounded(item) for item in value]
        return value
    return json.dumps(rounded(payload), sort_keys=True, ensure_ascii=False)


def diverse_candidates(revised: list[dict], originals: list[dict], limit: int | None):
    """Deduplicate revisions, then restore distinct original options for re-review."""
    selected, seen, duplicate_ids, restored_ids = [], set(), [], []
    for candidate in revised:
        identity = candidate_fingerprint(candidate)
        if identity in seen:
            duplicate_ids.append(candidate.get("id"))
            continue
        seen.add(identity)
        selected.append(candidate)
    target = limit or len(revised)
    next_id = max((int(c.get("id", 0)) for c in [*originals, *revised]), default=0) + 1
    for original in originals:
        if len(selected) >= target:
            break
        identity = candidate_fingerprint(original)
        if identity in seen:
            continue
        candidate = dict(original)
        if candidate["id"] in {c["id"] for c in selected}:
            candidate["parent_id"] = candidate["id"]
            candidate["id"] = next_id
            next_id += 1
        candidate["variant_mode"] = "original_retained_for_diversity"
        selected.append(candidate)
        seen.add(identity)
        restored_ids.append(candidate["id"])
    return selected, {"duplicate_candidate_ids": duplicate_ids,
                      "restored_original_ids": restored_ids,
                      "distinct_candidate_count": len(selected)}
