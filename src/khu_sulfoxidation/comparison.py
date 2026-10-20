"""Prospective, offline input/freeze contract for the optional history contrast.

The contract isolates structured history and omits retrieval/cache mechanisms. It
cannot prove that a human put no historical inference in arbitrary free text;
its freeze gate therefore requires a recorded information-boundary audit.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

ARMS = ("history_informed", "history_omitted")
VERSION = "khu_matched_history_comparison_v1"
BOUNDARY_NOTE = (
    "With versus without batch optimization history; both arms receive the same "
    "declared standard batch recipe, apparatus, physical candidate pool, common "
    "instructions and generation settings. Structured allowlists and disjoint "
    "sources do not prove that arbitrary prose is free of historical inference. "
    "A human information-boundary audit is required before freezing."
)


def canonical_json(value: Any) -> str:
    """Exact serialization used for both saved visible input and its hash."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _keys(value: Any, required: set[str], optional: set[str] | None = None, label: str = "object") -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing = required - value.keys()
    extra = value.keys() - required - (optional or set())
    if missing or extra:
        raise ValueError(f"{label}: missing={sorted(missing)}, unknown={sorted(extra)}")


def _text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")


def _number(value: Any, label: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'}")


def _hash(value: Any, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _identifier(value: Any, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}", value):
        raise ValueError(f"{label} must be a plain stable identifier")


def _integer(value: Any, label: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def build_comparison_plan(optional_nominations: int = 1) -> dict[str, Any]:
    """An honest scaffold while apparatus, routing and endpoints are unresolved."""
    if type(optional_nominations) is not int or optional_nominations not in (0, 1, 2):
        raise ValueError("Optional matched nominations must be 0 (disabled), 1 or 2 per arm")
    return {
        "schema_version": VERSION,
        "status": "disabled_optional_comparison" if optional_nominations == 0 else "planning_only_no_nominations_or_calls",
        "enabled": optional_nominations != 0,
        "nomination_count_per_arm": optional_nominations,
        "arms": {arm: {"nominations": []} for arm in ARMS},
        "information_boundary": BOUNDARY_NOTE,
        "shared_recipe_disclosure": "The proposed 0.25 M/3 mol% acridine recipe may be shared; disclose its selection from the batch anchor.",
        "required_before_freeze": [
            "confirmed apparatus and common feasible physical candidate pool",
            "candidate-space origin declared independent of optimization history",
            "exact common instructions, settings, equal candidate/call/revision/spending budgets",
            "agreed sulfone limit and matching evaluation opportunities",
            "complete exact visible inputs, source/configuration hashes and independent audit",
            "both arms' ordered nominations before any flow outcomes",
        ],
        "retrieval": {"enabled": False, "shared_rag": False, "study_reembedding": False, "cache_context": False},
        "paid_calls_enabled": False,
        "convergence_policy": "Allow overlap; a reused measurement is one shared condition, never independent arm evidence.",
        "evaluation_policy": "Compare the first matched 1-2 nominations per arm; do not compare the best of six with one control.",
        "interpretation": "A small prospective illustration, not statistical proof across chemistries.",
    }


def _validate_common(common: dict[str, Any]) -> None:
    _keys(common, {"study_id", "configuration_version", "shared_recipe", "apparatus", "physical_candidates", "pool_origin", "endpoints", "run_order", "generation", "instructions", "shared_sources", "nomination_count"}, label="common")
    for field in ("study_id", "configuration_version"):
        _identifier(common[field], field)
    if type(common["nomination_count"]) is not int or common["nomination_count"] not in (1, 2):
        raise ValueError("nomination_count must be 1 or 2")
    recipe = common["shared_recipe"]
    _keys(recipe, {"protocol_id", "protocol_sha256", "substrate", "catalyst", "solvent", "substrate_concentration_M", "catalyst_loading_mol_percent", "batch_temperature_C", "batch_time_h", "batch_oxidant", "selection_disclosure"}, label="shared_recipe")
    _identifier(recipe["protocol_id"], "protocol_id")
    _hash(recipe["protocol_sha256"], "protocol_sha256")
    for field in ("substrate", "catalyst", "solvent", "batch_oxidant", "selection_disclosure"):
        _text(recipe[field], field)
    for field in ("substrate_concentration_M", "catalyst_loading_mol_percent", "batch_time_h"):
        _number(recipe[field], field, positive=True)
    if isinstance(recipe["batch_temperature_C"], bool) or not isinstance(recipe["batch_temperature_C"], (float, int)) or not math.isfinite(recipe["batch_temperature_C"]) or recipe["batch_temperature_C"] <= -273.15:
        raise ValueError("Invalid batch_temperature_C")
    apparatus = common["apparatus"]
    _keys(apparatus, {"profile_id", "profile_sha256", "confirmed", "eligible_equipment_ids"}, label="apparatus")
    _identifier(apparatus["profile_id"], "profile_id")
    _hash(apparatus["profile_sha256"], "profile_sha256")
    if apparatus["confirmed"] is not True:
        raise ValueError("Matched nominations require a confirmed apparatus")
    equipment = apparatus["eligible_equipment_ids"]
    if not isinstance(equipment, list) or not equipment or len(equipment) != len(set(equipment)):
        raise ValueError("Unique eligible equipment IDs are required")
    for item in equipment:
        _identifier(item, "equipment_id")
    origin = common["pool_origin"]
    _keys(origin, {"basis", "independent_of_history", "declaration", "configuration_sha256"}, label="pool_origin")
    if origin["basis"] not in ("apparatus_envelope", "prospective_user_specification") or origin["independent_of_history"] is not True:
        raise ValueError("The common physical candidate pool must be defined independently of optimization history")
    _text(origin["declaration"], "pool origin declaration")
    _hash(origin["configuration_sha256"], "pool configuration hash")
    candidates = common["physical_candidates"]
    if not isinstance(candidates, list) or len(candidates) < common["nomination_count"]:
        raise ValueError("Insufficient common physical candidates")
    seen = set()
    for candidate in candidates:
        _keys(candidate, {"candidate_id", "reactor_id", "liquid_flow_mL_min", "gas_reference_flow_mL_min", "time_basis", "time_coordinate_min", "oxygen_equivalents", "feasible", "calculation_sha256"}, label="physical candidate")
        _identifier(candidate["candidate_id"], "candidate_id")
        if candidate["candidate_id"] in seen:
            raise ValueError("Duplicate candidate ID")
        seen.add(candidate["candidate_id"])
        if candidate["reactor_id"] not in equipment or candidate["feasible"] is not True:
            raise ValueError("Every shared candidate must be verified feasible with eligible equipment")
        _hash(candidate["calculation_sha256"], "candidate calculation hash")
        for field in ("liquid_flow_mL_min", "gas_reference_flow_mL_min", "time_coordinate_min", "oxygen_equivalents"):
            _number(candidate[field], field, positive=True)
        if candidate["time_basis"] not in ("liquid_only_space_time_min", "no_slip_mixture_transit_min"):
            raise ValueError("Candidate time basis must be an explicit calculated index")
    generation = common["generation"]
    _keys(generation, {"generator_id", "provider", "model", "temperature", "seed", "candidate_budget", "max_calls_per_arm", "max_output_tokens_per_call", "revision_budget", "max_cost_USD_per_arm", "routing_config_sha256"}, label="generation")
    for field in ("generator_id", "provider", "model"):
        _text(generation[field], field)
    _hash(generation["routing_config_sha256"], "routing_config_sha256")
    _number(generation["temperature"], "temperature")
    _integer(generation["seed"], "seed")
    for field in ("candidate_budget", "max_calls_per_arm", "max_output_tokens_per_call"):
        _integer(generation[field], field, 1)
    if generation["candidate_budget"] != len(candidates):
        raise ValueError("candidate_budget must match the complete physical pool")
    _integer(generation["revision_budget"], "revision_budget")
    _number(generation["max_cost_USD_per_arm"], "max_cost_USD_per_arm")
    endpoints = common["endpoints"]
    _keys(endpoints, {"primary", "sulfone_limit_percent", "secondary"}, label="endpoints")
    if endpoints["primary"] != "measured_sulfoxide_yield_percent":
        raise ValueError("Primary endpoint must be measured sulfoxide yield")
    _number(endpoints["sulfone_limit_percent"], "sulfone_limit_percent")
    if endpoints["secondary"] != ["residual_sulfide_percent", "sulfone_percent"]:
        raise ValueError("Keep residual sulfide and sulfone separate")
    order = common["run_order"]
    _keys(order, {"method", "seed", "sample_blinding", "measurement_reuse"}, label="run_order")
    if order["method"] not in ("randomized", "blocked") or order["sample_blinding"] is not True or order["measurement_reuse"] != "one_shared_condition_not_independent_evidence":
        raise ValueError("Declare randomized/blocked order, blinded samples and shared measurement handling")
    _integer(order["seed"], "run order seed")
    _text(common["instructions"], "common instructions")
    sources = common["shared_sources"]
    if not isinstance(sources, list):
        raise ValueError("shared_sources must be a list")
    ids = {recipe["protocol_id"], apparatus["profile_id"]}
    for source in sources:
        _keys(source, {"source_id", "source_sha256", "kind", "content"}, label="shared source")
        _identifier(source["source_id"], "source_id")
        _hash(source["source_sha256"], "source_sha256")
        if source["kind"] not in ("literature", "apparatus_specification", "analytical_protocol"):
            raise ValueError("Optimization history cannot be a shared source")
        if source["source_id"] in ids:
            raise ValueError("Duplicate shared source identity")
        ids.add(source["source_id"])
        _text(source["content"], "source content")
    canonical_json(common)


def build_matched_inputs(common: dict[str, Any], history_evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce complete isolated visible inputs, without reading stores or caches."""
    _validate_common(common)
    if not isinstance(history_evidence, list) or not history_evidence:
        raise ValueError("History-informed input needs explicit batch evidence")
    shared_ids = {source["source_id"] for source in common["shared_sources"]}
    shared_ids.update((common["apparatus"]["profile_id"], common["shared_recipe"]["protocol_id"]))
    history_ids = set()
    for evidence in history_evidence:
        _keys(evidence, {"source_id", "source_sha256", "kind", "payload"}, label="history evidence")
        _identifier(evidence["source_id"], "history source_id")
        _hash(evidence["source_sha256"], "history source_sha256")
        if evidence["kind"] not in ("batch_record", "slide_evidence", "history_summary"):
            raise ValueError("Only batch/slide evidence belongs in the prospective history arm")
        if evidence["source_id"] in history_ids | shared_ids:
            raise ValueError("History sources must be unique and separate from shared source identities")
        history_ids.add(evidence["source_id"])
    # Detached JSON copies prevent a caller modifying one arm through aliasing.
    copied = json.loads(canonical_json(common))
    inputs = {}
    for arm in ARMS:
        visible = {
            "schema_version": VERSION,
            "comparison_arm": arm,
            "common": json.loads(canonical_json(copied)),
            "batch_optimization_evidence": json.loads(canonical_json(history_evidence)) if arm == ARMS[0] else [],
            "access": {"retrieval_enabled": False, "shared_rag_enabled": False, "cache_context_enabled": False, "automatic_reembedding_enabled": False},
            "information_boundary": BOUNDARY_NOTE,
        }
        inputs[arm] = {"visible_input": visible, "visible_input_sha256": sha256(visible)}
    return {"schema_version": VERSION, "common_sha256": sha256(copied), "arms": inputs}


def visible_source_ids(visible_input: dict[str, Any]) -> set[str]:
    common = visible_input["common"]
    return {common["shared_recipe"]["protocol_id"], common["apparatus"]["profile_id"]} | {item["source_id"] for item in common["shared_sources"] + visible_input["batch_optimization_evidence"]}


def validate_matched_inputs(matched_inputs: dict[str, Any]) -> None:
    _keys(matched_inputs, {"schema_version", "common_sha256", "arms"}, label="matched inputs")
    if matched_inputs["schema_version"] != VERSION or set(matched_inputs["arms"]) != set(ARMS):
        raise ValueError("Both matched arms are required")
    informed = matched_inputs["arms"][ARMS[0]]["visible_input"]
    expected = build_matched_inputs(informed["common"], informed["batch_optimization_evidence"])
    if canonical_json(matched_inputs) != canonical_json(expected):
        raise ValueError("Matched inputs differ from the allowlisted isolated input contract or their hashes")


def freeze_comparison(directory: str | Path, matched_inputs: dict[str, Any], nominations: dict[str, list[dict[str, Any]]], *, frozen_at: str, boundary_audit: dict[str, Any], outcomes_seen: bool = False) -> dict[str, Any]:
    """Write one exclusive snapshot containing both arms and every nomination.

    frozen_at and outcomes_seen are investigator attestations, not proof that
    outcomes were unavailable outside this process. Existing files are never
    overwritten. The snapshot is content addressed and includes exact inputs.
    """
    validate_matched_inputs(matched_inputs)
    if outcomes_seen is not False:
        raise ValueError("Cannot prospectively freeze after flow outcomes have been seen")
    _text(frozen_at, "frozen_at")
    _keys(boundary_audit, {"reviewer", "reviewed_at", "no_history_in_common_text", "candidate_pool_independent_of_history", "no_external_rag_or_cache", "notes"}, label="boundary audit")
    for field in ("reviewer", "reviewed_at", "notes"):
        _text(boundary_audit[field], field)
    for field in ("no_history_in_common_text", "candidate_pool_independent_of_history", "no_external_rag_or_cache"):
        if boundary_audit[field] is not True:
            raise ValueError(f"Boundary audit has not affirmed {field}")
    if set(nominations) != set(ARMS):
        raise ValueError("Freeze both arms together")
    common = matched_inputs["arms"][ARMS[0]]["visible_input"]["common"]
    allowed = {c["candidate_id"] for c in common["physical_candidates"]}
    selected = {}
    for arm in ARMS:
        rows = nominations[arm]
        if not isinstance(rows, list) or len(rows) != common["nomination_count"]:
            raise ValueError("Both arms must have the same prespecified 1-2 nominations")
        ids = []
        visible = visible_source_ids(matched_inputs["arms"][arm]["visible_input"])
        for row in rows:
            _keys(row, {"candidate_id", "rationale", "evidence_citations"}, label="nomination")
            if row["candidate_id"] not in allowed or row["candidate_id"] in ids:
                raise ValueError("Nominations must be unique feasible candidates within each arm")
            _text(row["rationale"], "rationale")
            citations = row["evidence_citations"]
            if not isinstance(citations, list) or not citations or not set(citations) <= visible:
                raise ValueError(f"{arm} nomination cites evidence not visible in that arm")
            ids.append(row["candidate_id"])
        selected[arm] = ids
    overlap = sorted(set(selected[ARMS[0]]) & set(selected[ARMS[1]]))
    payload = {
        "schema_version": VERSION,
        "status": "prospectively_frozen_pending_laboratory_release",
        "frozen_at": frozen_at,
        "flow_outcomes_seen_attestation": False,
        "boundary_audit": boundary_audit,
        "matched_inputs": matched_inputs,
        "nominations": nominations,
        "shared_condition_ids": overlap,
        "measurement_policy": "Any overlapping nomination is one shared condition; a reused measurement is not independent evidence.",
        "evaluation_policy": "Evaluate ordered matched nominations, not the best result from the full history-informed screen.",
        "interpretation": "Exploratory prospective comparison, not an experimentally validated benefit.",
    }
    digest = sha256(payload)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"comparison-{digest}.json"
    # Write to a private temporary file and atomically link it into place. This
    # provides no partial public snapshot and refuses to replace an old freeze.
    data = {"snapshot_sha256": digest, "snapshot": payload}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json(data))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink()
    return {"path": str(destination), "snapshot_sha256": digest, "shared_condition_ids": overlap}
