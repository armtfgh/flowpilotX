"""Offline preflight and explicitly authorized Astra/Sol FlowPilotX review.

This runner never updates the production campaign or GUI. Its output remains a
scientifically unreviewed proposal even when structural checks pass. Transport,
spending reservations and immutable API logs belong to CampaignGateway.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

from flowpilotx.campaign import validate_spec
from flowpilotx.gateway import digest as context_digest, planning_context, validate_plan
from flowpilotx.selection import select
from .model_gateway import (CampaignGateway, ROUTING, build_payload, canonical,
    conservative_input_bound, digest, reservation_USD, validate_campaign)

ROLES = tuple(ROUTING)
SPECIALISTS = ROLES[1:5]
ARM = "history_informed"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frozen(frozen_inputs):
    f = deepcopy(frozen_inputs)
    canonical(f)  # Validate JSON safety without changing archived prompt order.
    for key in ["campaign_id", "campaign_version", "campaign_revision_sha256", "campaign_spec",
                "context", "context_sha256", "planner_system", "review_system", "request"]:
        if key not in f:
            raise ValueError("Missing frozen input: " + key)
    if not isinstance(f["campaign_id"], str) or not f["campaign_id"]:
        raise ValueError("Frozen campaign ID required")
    if type(f["campaign_version"]) is not int or f["campaign_version"] < 1:
        raise ValueError("Frozen campaign version required")
    if not isinstance(f["campaign_revision_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", f["campaign_revision_sha256"]):
        raise ValueError("Frozen campaign revision hash required")
    if context_digest(f["context"]) != f["context_sha256"]:
        raise ValueError("Frozen context hash mismatch")
    for key in ["planner_system", "review_system"]:
        if not isinstance(f[key], str) or not f[key].strip():
            raise ValueError("Frozen system prompt required: " + key)
    validate_spec(f["campaign_spec"])
    if f["request"].get("requested_count") != 6 or f["request"].get("repeat_count") != 1:
        raise ValueError("This comparison freezes six unique conditions plus one separate repeat")
    context = f["context"]
    expected = planning_context({"spec": f["campaign_spec"], "results": []}, f["request"])
    # The archived Qwen input predates this derived empty-feedback ledger. It
    # can be omitted for an exactly matched initial-planner information boundary.
    if "feedback_eligibility" not in context:
        expected.pop("feedback_eligibility", None)
    if expected != context:
        raise ValueError("Context differs from the frozen evidence/pool/request; no prior model output or unrecorded additions allowed")
    if context.get("measured_results") != [] or len(context["candidates"]) != 12:
        raise ValueError("Expected the frozen12-candidate pool without flow outcomes")
    raw = context["evidence"]["homogeneous_batch_evidence"]["raw_records"]
    entries = [r[0] for r in raw["rows"]]
    if len(entries) != 116 or len(set(entries)) != 116 or set(map(int, entries)) != set(range(1, 117)):
        raise ValueError("All116 homogeneous canonical records must remain present once")
    return f


def _config(config):
    c = json.loads(canonical(config))
    if type(c.get("authorized")) is not bool:
        raise ValueError("Authorization must be explicit")
    # Schema check only: this does not authorize, construct a gateway, inspect
    # credentials, count provider tokens or send a request.
    schema_copy = {**c, "authorized": True}
    validate_campaign(schema_copy)
    for key, expected in {"max_attempts": 7, "max_attempts_per_arm": 7,
            "maximum_cost_USD": 5, "max_input_tokens": 64000, "max_output_tokens": 16000}.items():
        if c[key] != expected:
            raise ValueError("This proposed campaign fixes " + key + "=" + str(expected))
    return c


def _source_check(config, project_root):
    root = Path(project_root).resolve()
    checked = []
    for name, expected in config["source_hashes"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Source is missing or outside the project: " + name)
        actual = _file_hash(path)
        if actual != expected:
            raise ValueError("Source hash changed: " + name)
        checked.append({"path": name, "sha256": actual})
    return checked


def prepare_round(frozen_inputs, config, project_root):
    """Read-only preflight. No gateway, credential read or network operation."""
    f, c = _frozen(frozen_inputs), _config(config)
    sources = _source_check(c, project_root)
    bound_inputs = []
    for source in sources:
        path = Path(project_root) / source["path"]
        if path.name == "frozen_inputs.json":
            recorded = _read(path)
            if "campaign_spec" in recorded:
                bound_inputs.append(recorded)
    if not bound_inputs or any(recorded != f for recorded in bound_inputs):
        raise ValueError("Frozen input object differs from the explicitly hashed frozen_inputs.json source")
    request = request_for(f, "Upstream", {}, c["source_hashes"])
    bound = conservative_input_bound(build_payload(request, ROUTING["Upstream"], c["max_output_tokens"]))
    return {"schema_version": "flowpilotx-astra-preflight-v1", "status": "prepared_not_executed",
        "authorized": c["authorized"], "campaign_id": c["campaign_id"], "input_campaign_id": f["campaign_id"],
        "frozen_inputs_sha256": digest(f), "context_sha256": f["context_sha256"],
        "configuration_sha256": digest(c), "source_checks": sources,
        "routing": deepcopy(ROUTING), "roles": list(ROLES), "arm": ARM,
        "maximum_attempts": 7, "maximum_cost_USD": 5,
        "full_seven_call_reservation_USD": float(sum((Decimal(str(reservation_USD(ROUTING[r], 64000, 16000))) for r in ROLES), Decimal(0))),
        "upstream_conservative_input_bound": bound,
        "provider_token_count_may_be_required_after_authorization": bound > 64000,
        "homogeneous_records": 116, "physical_candidates": 12, "requested_unique_conditions": 6,
        "additional_independent_repeats": 1, "measured_flow_results": 0,
        "paid_or_token_count_calls_made": 0, "scientific_review_status": "pending",
        "production_mutation": False, "laboratory_released": False,
        "comparison_limits": f.get("comparison_limits", f.get("comparisonlimits", []))}


def _authorization(config, project_root):
    if config["authorized"] is not True:
        raise ValueError("Paid execution requires fresh explicit user authorization; proposal is disabled")
    path = (Path(project_root) / config["authorization_reference"]).resolve()
    if not path.is_relative_to(Path(project_root).resolve()) or not path.is_file():
        raise ValueError("Recorded authorization file is missing")
    record = _read(path)
    if not isinstance(record.get("user_instruction"), str) or not record["user_instruction"].strip():
        raise ValueError("Authorization needs the user's explicit instruction")
    if record.get("campaign_id") != config["campaign_id"]:
        raise ValueError("Authorization belongs to another campaign")
    if record.get("maximum_cost_USD") != 5 or record.get("maximum_attempts") != 7:
        raise ValueError("Authorization must establish the7-attempt/$5 spending cap")
    route = record.get("authorized_route")
    if route not in [ROUTING, {"Upstream": "gpt-6-astra", "Council": "gpt-6.1-sol"}]:
        raise ValueError("Authorization does not establish the exact Astra/Sol routing")
    return {"path": str(path), "sha256": _file_hash(path)}


def request_for(frozen_inputs, role, prior, source_hashes):
    """Each stage sees shared input and only this campaign's preceding outputs."""
    if role not in ROLES:
        raise ValueError("Unknown Astra/Sol stage")
    context = deepcopy(frozen_inputs["context"])
    if role == "Upstream":
        system = frozen_inputs["planner_system"]
    elif role in SPECIALISTS:
        system = frozen_inputs["review_system"]
        context.update(assigned_role=role, upstream_proposal=prior["Upstream"])
    elif role == "Skeptic":
        system = frozen_inputs["review_system"]
        context.update(assigned_role=role, upstream_proposal=prior["Upstream"],
            specialist_reviews=[{"role": r, "response": prior[r]} for r in SPECIALISTS])
    else:
        system = frozen_inputs["planner_system"] + "\nIntegrate this campaign's specialist and skeptic reviews. Check their claims against the original input. Return the required JSON with exactly six candidate IDs; hypotheses remain unreviewed."
        context.update(upstream_proposal=prior["Upstream"],
            specialist_reviews=[{"role": r, "response": prior[r]} for r in SPECIALISTS], skeptic=prior["Skeptic"])
    return {"system_prompt": system, "user_prompt": json.dumps(context, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        "source_hashes": {**source_hashes, "frozen_inputs_semantic": digest(frozen_inputs)}}


def _visible_ids(context):
    ids = set()
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"id", "record_id", "source_id", "summary_source_id"} and isinstance(item, str):
                    ids.add(item)
                if key == "evidence_ids" and isinstance(item, list):
                    ids.update(x for x in item if isinstance(x, str))
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(context)
    rows = context["evidence"]["homogeneous_batch_evidence"]["raw_records"]["rows"]
    ids.update(f"khu:homogeneous:raw-entry:{int(row[0]):04d}" for row in rows)
    return ids


def _validate_response(role, response, frozen):
    if not isinstance(response, dict):
        raise ValueError("Model response must be an object")
    errors = []
    if role in {"Upstream", "Chief"}:
        validation = validate_plan(response, {"spec": frozen["campaign_spec"], "results": []}, 6)
        errors.extend(validation["errors"])
        by_id = {c["id"]: c for c in frozen["campaign_spec"]["candidates"]}
        selected = [by_id[i] for i in response.get("selected_candidate_ids", []) if i in by_id]
        pairs = {(c["parameters"]["target_time_min"], c["parameters"]["target_oxygen_eq"]) for c in selected}
        times = {p[0] for p in pairs}
        if len(times) != 3 or pairs != {(t, e) for t in times for e in [.5, 1.]}:
            errors.append("Requested screen requires three time coordinates with both oxygen levels")
    else:
        if response.get("role") != role or response.get("verdict") not in {"accept_conditional", "revise", "reject"}:
            errors.append("Reviewer role/verdict invalid")
        for key in ["issues", "required_changes", "evidence_ids", "unresolved"]:
            if not isinstance(response.get(key), list) or not all(isinstance(x, str) for x in response[key]):
                errors.append("Reviewer requires text array: " + key)
    visible = _visible_ids(frozen["context"])
    def citations(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "evidence_ids" and isinstance(item, list):
                    errors.extend("Citation outside visible input: " + str(i) for i in item if not isinstance(i, str) or i not in visible)
                citations(item)
        elif isinstance(value, list):
            for item in value:
                citations(item)
    citations(response)
    return {"passed": not errors, "errors": errors, "definition": "Structural/source checks only; scientific review remains pending"}


def _cached(gateway, role, request):
    matches = [e for e in gateway.audit()["entries"] if e["role"] == role and e["arm"] == ARM]
    if not matches:
        return None
    if len(matches) != 1 or matches[0]["status"] != "completed":
        raise ValueError("Previously failed/incomplete role cannot be retried: " + role)
    path = Path(gateway.root) / "attempts" / matches[0]["attempt_id"]
    if _read(path / "input.json") != request:
        raise ValueError("Archived role request differs from frozen request: " + role)
    answer = _read(path / "result.json")
    if answer.get("ok") is not True:
        raise ValueError("Archived role was not successful: " + role)
    return answer


def run_round(frozen_inputs, gateway, output_dir, *, project_root):
    """Run/resume seven bounded stages using an already explicit gateway.

    Fake gateways can test this API offline. Production callers should use
    execute_prepared, which checks authorization before constructing transport.
    """
    f, c = _frozen(frozen_inputs), _config(gateway.config)
    preflight = prepare_round(f, c, project_root)
    authorization = _authorization(c, project_root)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    snapshot = {"frozen_inputs": f, "configuration": c, "authorization": authorization,
                "frozen_inputs_sha256": digest(f), "configuration_sha256": digest(c)}
    path = out / "input_freeze.json"
    if path.exists():
        if _read(path) != snapshot:
            raise ValueError("Cannot resume changed inputs, authorization or configuration")
    else:
        _write(path, snapshot)
    prior = {}
    for role in ROLES:
        _source_check(c, project_root)
        if _authorization(c, project_root) != authorization:
            raise ValueError("Authorization changed after the round was frozen")
        request = request_for(f, role, prior, c["source_hashes"])
        stage_path = out / (role + ".json")
        failure_path = out / (role + "_failure.json")
        if failure_path.exists():
            raise ValueError("Recorded failed stage is not retried: " + role)
        try:
            answer = _cached(gateway, role, request)
            if stage_path.exists() and answer is None:
                raise ValueError("Stage artifact has no matching successful gateway attempt")
            if answer is None:
                answer = gateway.execute_request(request, role=role, arm=ARM)
            if answer.get("ok") is not True:
                raise ValueError("API attempt failed; no retry: " + role)
            validation = _validate_response(role, answer.get("parsed_json"), f)
            stage = {"role": role, "request_sha256": digest(request), "response": answer["parsed_json"],
                "validation": validation, "scientific_review_status": "pending",
                "api_metadata": {k: v for k, v in answer.items() if k not in {"text", "parsed_json"}}}
            if stage_path.exists():
                if _read(stage_path) != stage:
                    raise ValueError("Saved stage differs from its audited API response")
            else:
                _write(stage_path, stage)
            if not validation["passed"]:
                raise ValueError("Structural/source validation failed: " + "; ".join(validation["errors"]))
            prior[role] = answer["parsed_json"]
        except Exception as exc:
            if not failure_path.exists():
                _write(failure_path, {"role": role, "request_sha256": digest(request), "error": str(exc),
                    "scientific_review_status": "pending", "retry_allowed": False})
            raise
    chosen = prior["Chief"]["selected_candidate_ids"]
    pool = {c["id"]: c for c in f["campaign_spec"]["candidates"]}
    high = sorted((pool[i] for i in chosen if pool[i]["parameters"]["target_oxygen_eq"] == 1),
                  key=lambda x: x["parameters"]["target_time_min"])
    repeat_of = high[len(high) // 2]["id"]
    selection = select(f["campaign_spec"], [], {"requested_count": 6, "repeat_count": 1,
        "repeat_of": repeat_of, "allow_conditional": True, "selected_candidate_ids": chosen}, "no_flow_observations")
    if selection["status"] == "rejected":
        _write(out / "selection_rejection.json", selection)
        raise ValueError("Deterministic physical selection gate rejected final proposal")
    report = {"schema_version": "flowpilotx-astra-review-v1", "status": "structurally_checked_scientific_review_pending",
        "frozen_inputs_sha256": digest(f), "configuration_sha256": digest(c),
        "input_campaign_id": f["campaign_id"], "input_campaign_version": f["campaign_version"],
        "input_campaign_revision_sha256": f["campaign_revision_sha256"], "source_context_sha256": f["context_sha256"],
        "routing": deepcopy(ROUTING), "roles": list(ROLES), "arm": ARM,
        "selected_candidate_ids": chosen, "repeat_of": repeat_of, "final_plan": prior["Chief"],
        "role_outputs": prior, "selection_audit": selection, "api_audit": gateway.audit(),
        "scientific_review_status": "pending", "human_corrections_supplied": False,
        "Qwen_outputs_supplied": False, "measured_flow_outcomes_seen": False,
        "production_mutation": False, "laboratory_released": False,
        "comparison_limits": preflight["comparison_limits"]}
    result_path = out / "campaign_result.json"
    if result_path.exists():
        if _read(result_path) != report:
            raise ValueError("Completed review report differs from audited stages")
    else:
        _write(result_path, report)
    freeze_path = out / "nominations_freeze.json"
    if freeze_path.exists():
        if _read(freeze_path).get("report_sha256") != digest(report):
            raise ValueError("Nomination freeze no longer matches completed review")
    else:
        _write(freeze_path, {"frozen_at": datetime.now(timezone.utc).isoformat(),
            "report_sha256": digest(report), "scientific_review_status": "pending", "laboratory_released": False,
            "selected_candidate_ids": chosen, "repeat_of": repeat_of, "production_mutation": False})
    return report


def execute_prepared(frozen_inputs, config, project_root, output_dir):
    """Only explicit execution constructs a gateway, after all local guards."""
    prepare_round(frozen_inputs, config, project_root)
    _authorization(config, project_root)
    gateway = CampaignGateway(Path(output_dir) / "api", config)
    return run_round(frozen_inputs, gateway, output_dir, project_root=project_root)
