"""Offline review contracts and append-only imports; no model/network client.

Review responses are untrusted proposals. Structural/citation validation cannot
verify the truth of free-text chemistry or release a laboratory protocol.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .comparison import (
    ARMS, BOUNDARY_NOTE, VERSION, _hash, _integer, _keys, _number,
    _text, _validate_common, canonical_json, sha256, visible_source_ids,
)

ROLES = ("Chemistry", "KineticsSelectivity", "FluidicsPhotoreactor", "Safety", "Skeptic", "Chief")
SYSTEM_INSTRUCTIONS = (
    "Review the supplied bounded candidate pool for the KHU selective sulfide-to-sulfoxide study. "
    "Use only this complete visible input. Cite visible source IDs for observations and claims. "
    "Keep residual sulfide, desired sulfoxide and sulfone distinct. Batch observations are "
    "evidence, not measured flow kinetics. Do not assume yield increases with time, "
    "normalize analytical percentages, invent flow outcomes, force time reduction, or "
    "force different nominations between arms. Label transfer ideas as hypotheses. "
    "Report hard violations and unresolved constraints; no internal score releases a protocol. "
    "Return the exact structured response requested, retaining rejections and limitations."
)


def disabled_campaign() -> dict[str, Any]:
    return {
        "schema_version": "khu_paid_campaign_v1",
        "enabled": False,
        "campaign_id": None,
        "authorization": None,
        "routing": [],
        "maximum_cost_USD": None,
        "max_calls": None,
        "max_output_tokens_per_call": None,
        "revision_budget": None,
        "decoding": None,
        "fallback_enabled": False,
        "automatic_reembedding_enabled": False,
        "network_execution_implemented": False,
    }


def validate_paid_campaign(configuration: dict[str, Any] | None, *, spent_USD: float = 0.0, completed_calls: int = 0, reserved_next_call_USD: float = 0.0) -> dict[str, Any]:
    """Validate a prospective spending gate, never execute or authorize a call.

    A returned configuration_ready flag is only validation of supplied recorded
    authorization. The caller must establish the real authorization with the
    user; writing this JSON is not permission. This module cannot call models.
    """
    config = disabled_campaign() if configuration is None else configuration
    _keys(config, set(disabled_campaign()), label="paid campaign")
    _number(spent_USD, "spent_USD")
    _number(reserved_next_call_USD, "reserved_next_call_USD")
    _integer(completed_calls, "completed_calls")
    if config["enabled"] is not True:
        return {"configuration_ready": False, "network_calls_available": False, "reason": "Paid calls are disabled; establish exact study routing and budget with the user.", "configuration_sha256": sha256(config)}
    if config["schema_version"] != "khu_paid_campaign_v1":
        raise ValueError("Unknown paid campaign schema")
    if config["fallback_enabled"] is not False or config["automatic_reembedding_enabled"] is not False or config["network_execution_implemented"] is not False:
        raise ValueError("No hidden fallback, re-embedding or network implementation is supported")
    _text(config["campaign_id"], "campaign_id")
    auth = config["authorization"]
    _keys(auth, {"authorized", "reference", "established_at", "study_id"}, label="authorization")
    if auth["authorized"] is not True or auth["study_id"] != "khu_sulfoxidation_v1":
        raise ValueError("Specific KHU study authorization is required; prior campaign authorization does not transfer")
    _text(auth["reference"], "authorization reference")
    _text(auth["established_at"], "authorization established_at")
    routes = config["routing"]
    if not isinstance(routes, list) or not routes:
        raise ValueError("Exact per-role routing is required")
    seen = set()
    for route in routes:
        _keys(route, {"role", "provider", "model"}, label="route")
        if route["role"] not in ROLES or route["role"] in seen:
            raise ValueError("Routes must name unique supported reviewer roles")
        seen.add(route["role"])
        for field in ("provider", "model"):
            _text(route[field], field)
        if route["model"].lower() in {"default", "auto", "latest"}:
            raise ValueError("A concrete model identifier is required")
    _number(config["maximum_cost_USD"], "maximum_cost_USD", positive=True)
    for field in ("max_calls", "max_output_tokens_per_call"):
        _integer(config[field], field, 1)
    _integer(config["revision_budget"], "revision_budget")
    _keys(config["decoding"], {"temperature", "seed"}, label="decoding")
    _number(config["decoding"]["temperature"], "temperature")
    _integer(config["decoding"]["seed"], "seed")
    blockers = []
    if completed_calls >= config["max_calls"]:
        blockers.append("Call budget exhausted")
    if spent_USD + reserved_next_call_USD > config["maximum_cost_USD"] or spent_USD >= config["maximum_cost_USD"]:
        blockers.append("Spending budget exhausted or next-call reservation exceeds it")
    return {"configuration_ready": not blockers, "network_calls_available": False, "blockers": blockers, "configuration_sha256": sha256(config), "remaining_cost_USD": max(0.0, config["maximum_cost_USD"] - spent_USD), "remaining_calls": max(0, config["max_calls"] - completed_calls)}


def _validate_visible_input(visible: dict[str, Any]) -> None:
    _keys(visible, {"schema_version", "comparison_arm", "common", "batch_optimization_evidence", "access", "information_boundary"}, label="visible input")
    if visible["schema_version"] != VERSION or visible["comparison_arm"] not in ARMS:
        raise ValueError("Unknown matched input schema or arm")
    _validate_common(visible["common"])
    if visible["access"] != {"retrieval_enabled": False, "shared_rag_enabled": False, "cache_context_enabled": False, "automatic_reembedding_enabled": False} or visible["information_boundary"] != BOUNDARY_NOTE:
        raise ValueError("Retrieval/cache/re-embedding must remain disabled and boundary declared")
    evidence = visible["batch_optimization_evidence"]
    if not isinstance(evidence, list) or (visible["comparison_arm"] == "history_omitted" and evidence):
        raise ValueError("History evidence is forbidden in the history-omitted input")
    if visible["comparison_arm"] == "history_informed" and not evidence:
        raise ValueError("History-informed input requires batch evidence")
    seen = {source["source_id"] for source in visible["common"]["shared_sources"]}
    seen.update((visible["common"]["shared_recipe"]["protocol_id"], visible["common"]["apparatus"]["profile_id"]))
    for item in evidence:
        _keys(item, {"source_id", "source_sha256", "kind", "payload"}, label="history evidence")
        _text(item["source_id"], "source_id")
        _hash(item["source_sha256"], "source_sha256")
        if item["kind"] not in ("batch_record", "slide_evidence", "history_summary") or item["source_id"] in seen:
            raise ValueError("Invalid or duplicate history evidence source")
        seen.add(item["source_id"])
    canonical_json(visible)


def build_review_request(visible_input: dict[str, Any], role: str) -> dict[str, Any]:
    _validate_visible_input(visible_input)
    if role not in ROLES:
        raise ValueError(f"Unsupported role: {role}")
    body = {
        "role": role,
        "visible_input": visible_input,
        "response_contract": {
            "role": role,
            "candidate_reviews": [{"candidate_id": "every visible candidate exactly once", "verdict": "eligible|reject|needs_confirmation", "statements": [{"kind": "batch_observation|calculated_quantity|hypothesis|unresolved", "text": "source-supported observation or explicitly uncertain rationale", "citations": ["visible source ID"]}], "rejection_reasons": ["required for reject verdict"]}],
            "nominated_candidate_ids": "Chief: exactly the configured nomination_count eligible IDs, ordered; others: []",
            "limitations": ["required unresolved limitations"],
        },
    }
    request = {"system_prompt": SYSTEM_INSTRUCTIONS, "user_prompt": canonical_json(body)}
    return {**request, "prompt_sha256": sha256(request), "visible_input_sha256": sha256(visible_input)}


def validate_review_response(response: dict[str, Any], visible_input: dict[str, Any], role: str) -> dict[str, Any]:
    _validate_visible_input(visible_input)
    if role not in ROLES:
        raise ValueError("Unsupported reviewer role")
    _keys(response, {"role", "candidate_reviews", "nominated_candidate_ids", "limitations"}, label="review response")
    if response["role"] != role:
        raise ValueError("Response role does not match request")
    candidates = {c["candidate_id"] for c in visible_input["common"]["physical_candidates"]}
    sources = visible_source_ids(visible_input)
    history_sources = {s["source_id"] for s in visible_input["batch_optimization_evidence"] if s["kind"] in ("batch_record", "slide_evidence")}
    seen, eligible = set(), set()
    rows = response["candidate_reviews"]
    if not isinstance(rows, list):
        raise ValueError("candidate_reviews must be a list")
    for row in rows:
        _keys(row, {"candidate_id", "verdict", "statements", "rejection_reasons"}, label="candidate review")
        candidate = row["candidate_id"]
        if candidate not in candidates or candidate in seen:
            raise ValueError("Review must reference each visible candidate exactly once")
        seen.add(candidate)
        if row["verdict"] not in {"eligible", "reject", "needs_confirmation"}:
            raise ValueError("Invalid review verdict")
        if row["verdict"] == "eligible":
            eligible.add(candidate)
        if not isinstance(row["rejection_reasons"], list) or (row["verdict"] == "reject" and not row["rejection_reasons"]):
            raise ValueError("Rejected proposals need recorded reasons")
        for reason in row["rejection_reasons"]:
            _text(reason, "rejection reason")
        if not isinstance(row["statements"], list) or not row["statements"]:
            raise ValueError("Every candidate needs evidence-labelled statements")
        for statement in row["statements"]:
            _keys(statement, {"kind", "text", "citations"}, label="statement")
            if statement["kind"] not in {"batch_observation", "calculated_quantity", "hypothesis", "unresolved"}:
                raise ValueError("Measured flow outcomes and unsupported claim types are unavailable before the screen")
            _text(statement["text"], "statement text")
            citations = statement["citations"]
            if not isinstance(citations, list) or not citations or not set(citations) <= sources:
                raise ValueError("Every statement must cite evidence visible to its arm")
            if statement["kind"] == "batch_observation" and not set(citations) <= history_sources:
                raise ValueError("Optimization observations require visible source batch records or slide evidence")
    if seen != candidates:
        raise ValueError("Review must retain all candidates including rejections")
    selected = response["nominated_candidate_ids"]
    if not isinstance(selected, list) or len(selected) != len(set(selected)) or not set(selected) <= eligible:
        raise ValueError("Nominations must be unique eligible reviewed candidates")
    expected = visible_input["common"]["nomination_count"] if role == "Chief" else 0
    if len(selected) != expected:
        raise ValueError(f"{role} requires {expected} nominations")
    if not isinstance(response["limitations"], list) or not response["limitations"]:
        raise ValueError("Review must preserve limitations")
    for limitation in response["limitations"]:
        _text(limitation, "limitation")
    return {"valid": True, "candidate_count": len(seen), "nominated_candidate_ids": selected, "scope": "Structural and citation validation only; chemistry truth and release remain human review tasks."}


def archive_review_attempt(directory: str | Path, *, request: dict[str, Any], raw_response: str, configuration: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Retain an exact offline import even when it fails parsing or validation.

    The immutable directory includes exact UTF-8 prompts/response and canonical
    JSON config/metadata. Invalid attempts remain visible rather than repaired,
    retried, or dropped. Missing cost stays null, never silently zero.
    """
    # Only serialization/type failures prevent an import; semantic validation
    # happens after the original payload has been persisted.
    _keys(request, {"system_prompt", "user_prompt", "prompt_sha256", "visible_input_sha256"}, label="request")
    for field in ("system_prompt", "user_prompt"):
        if not isinstance(request[field], str):
            raise ValueError("Prompts must be exact strings")
    if not isinstance(raw_response, str):
        raise ValueError("raw_response must be the exact string, including invalid JSON")
    blobs = {
        "system_prompt.txt": request["system_prompt"],
        "user_prompt.txt": request["user_prompt"],
        "response.txt": raw_response,
        "configuration.json": canonical_json(configuration),
        "metadata.json": canonical_json(metadata),
        "request.json": canonical_json(request),
    }
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    attempt = root / ("attempt-" + uuid4().hex)
    attempt.mkdir(exist_ok=False)
    hashes = {}
    for name, text in blobs.items():
        contents = text.encode("utf-8")
        with (attempt / name).open("xb") as handle:
            handle.write(contents)
        hashes[name] = hashlib.sha256(contents).hexdigest()
    errors = []
    validation = None
    try:
        _keys(metadata, {"attempt_id", "started_at", "completed_at", "role", "provider", "model", "status", "input_tokens", "output_tokens", "cost_USD", "source_hashes"}, label="attempt metadata")
        for field in ("attempt_id", "started_at", "completed_at", "role", "provider", "model", "status"):
            _text(metadata[field], field)
        if metadata["status"] not in ("completed", "failed", "cancelled"):
            raise ValueError("Attempt status must preserve completed/failed/cancelled")
        if metadata["cost_USD"] is not None:
            _number(metadata["cost_USD"], "cost_USD")
        for field in ("input_tokens", "output_tokens"):
            if metadata[field] is not None:
                _integer(metadata[field], field)
        if not isinstance(metadata["source_hashes"], dict) or not metadata["source_hashes"]:
            raise ValueError("Record exact source hashes")
        for digest in metadata["source_hashes"].values():
            _hash(digest, "source hash")
        prompt = {key: request[key] for key in ("system_prompt", "user_prompt")}
        if sha256(prompt) != request["prompt_sha256"]:
            raise ValueError("Prompt hash mismatch")
        body = json.loads(request["user_prompt"])
        visible = body["visible_input"]
        if sha256(visible) != request["visible_input_sha256"]:
            raise ValueError("Visible-input hash mismatch")
        if metadata["role"] != body["role"]:
            raise ValueError("Metadata role differs from exact prompt role")
        if request != build_review_request(visible, body["role"]):
            raise ValueError("Exact prompt differs from the recorded review contract")
        common = visible["common"]
        required_sources = {
            common["shared_recipe"]["protocol_id"]: common["shared_recipe"]["protocol_sha256"],
            common["apparatus"]["profile_id"]: common["apparatus"]["profile_sha256"],
        }
        required_sources.update({s["source_id"]: s["source_sha256"] for s in common["shared_sources"] + visible["batch_optimization_evidence"]})
        if any(metadata["source_hashes"].get(source_id) != digest for source_id, digest in required_sources.items()):
            raise ValueError("Source hashes do not cover every visible source exactly")
        if sha256(configuration) != common["generation"]["routing_config_sha256"]:
            raise ValueError("Configuration differs from the input's frozen routing configuration")
        validate_paid_campaign(configuration)
        if configuration["enabled"]:
            route = next((r for r in configuration["routing"] if r["role"] == body["role"]), None)
            if route is None or (route["provider"], route["model"]) != (metadata["provider"], metadata["model"]):
                raise ValueError("Imported attempt route differs from recorded campaign routing")
        elif metadata["provider"] != "offline" or (metadata["cost_USD"] is not None and metadata["cost_USD"] > 0):
            raise ValueError("Model attempt lacks an enabled, authorized paid campaign record")
        if metadata["status"] != "completed":
            raise ValueError(f"Attempt status is {metadata['status']}; raw response retained")
        response = json.loads(raw_response)
        validation = validate_review_response(response, visible, body["role"])
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(str(exc))
    manifest = {"schema_version": "khu_offline_review_attempt_v1", "imported_at": datetime.now(timezone.utc).isoformat(), "import_only_no_model_call": True, "files_sha256": hashes, "validation": validation, "errors": errors, "cost_USD": metadata.get("cost_USD") if isinstance(metadata, dict) else None, "missing_cost_is_unknown": True}
    with (attempt / "manifest.json").open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(manifest))
    return {"path": str(attempt), "manifest_sha256": sha256(manifest), "valid": not errors, "errors": errors}
