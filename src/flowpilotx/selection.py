"""Transparent set selection over adapter-supplied candidates.

Coverage is deterministic. Optimization feedback uses observed objective order
and proximity to observed conditions, not fitted kinetics, BO or yield prediction.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .evidence import analytical_method_issue, digest, json_copy, number, strings, text
from .inventory import candidate_checks, compare, lookup, validate_constraint


def validate_observation_contract(contract, candidates):
    """Declare measured controls/context explicitly, separately from targets.

    This is an adapter-owned observation schema, not a declaration that every
    proposed/calculated field is measurable. Without it, old fixtures retain the
    conservative exact-key check; no field meanings are guessed.
    """
    c = json_copy(contract)
    if not isinstance(c, dict) or c.get("schema_version") != "flowpilotx-observation-contract-v1":
        raise ValueError("Unsupported observation_contract schema")
    required = strings(c.get("required_actual_conditions"), "required_actual_conditions", nonempty=True)
    if len(set(required)) != len(required):
        raise ValueError("Required actual-condition paths must be unique")
    mapping = c.get("parameter_mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("observation_contract needs a parameter_mapping for measured comparison controls")
    for parameter, actual in mapping.items():
        text(parameter, "candidate comparison parameter")
        text(actual, "actual observation path")
        if actual not in required:
            raise ValueError("Mapped observation must appear in required_actual_conditions: " + actual)
        for candidate in candidates:
            present, value = lookup(candidate["parameters"], parameter)
            if candidate["feasible"] and (not present or value is None):
                raise ValueError("Observation mapping references a missing candidate control: " + parameter)
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Do not count one measured control twice through duplicate parameter aliases")
    for key in ["units", "types"]:
        if not isinstance(c.get(key, {}), dict):
            raise ValueError("observation_contract." + key + " must be an object")
        if set(c.get(key, {})) - set(required):
            raise ValueError("Observation units/types reference an undeclared actual-condition path")
    for unit in c.get("units", {}).values():
        text(unit, "observation unit")
    for parameter, actual in mapping.items():
        for candidate in candidates:
            _, value = lookup(candidate["parameters"], parameter)
            if candidate["feasible"] and isinstance(value, dict) and value.get("unit"):
                if value["unit"] != c.get("units", {}).get(actual):
                    raise ValueError("Mapped candidate/actual units differ; no implicit conversion: " + parameter)
    if any(kind not in {"number", "string", "boolean"} for kind in c.get("types", {}).values()):
        raise ValueError("Observation field types must be number, string or boolean")
    c["comparability_requirements"] = [validate_constraint(rule) for rule in c.get("comparability_requirements", [])]
    for rule in c["comparability_requirements"]:
        if rule["parameter"] not in required:
            raise ValueError("Comparability gate references an undeclared required observation")
    if not isinstance(c.get("actual_conditions_template", {}), dict):
        raise ValueError("Actual-condition template must be an object")
    return c


def _actual_condition_issue(record, contract):
    actual = record["actual_conditions"]
    if contract is None:
        if any(_features(actual).get(key) is None for key in _features(record["proposed_conditions"])):
            return "Actual operating conditions do not cover the legacy exact-key contract; provide an explicit measured-control observation_contract"
        return None
    for path in contract["required_actual_conditions"]:
        present, value = lookup(actual, path)
        if not present or value is None:
            return "Missing required measured control/context: " + path
        if isinstance(value, dict):
            if value.get("status") in {"assumed", "calculated", "derived", "estimated", "nominal", "proposed", "unknown", "unmeasured"}:
                return "Required observation is not reported actual data: " + path
            expected_unit = contract.get("units", {}).get(path)
            if expected_unit is not None and value.get("unit") != expected_unit:
                return "Actual observation unit mismatch; no implicit conversion: " + path
            value = value.get("value")
        if value is None or isinstance(value, str) and value.strip().lower() in {"", "unknown", "n/a", "na", "-", "pending"}:
            return "Required measured control/context remains unknown: " + path
        kind = contract.get("types", {}).get(path)
        if kind == "number" and not _numeric(value):
            return "Required measured control must be numeric: " + path
        if kind == "string" and not isinstance(value, str):
            return "Required recorded context must be text: " + path
        if kind == "boolean" and type(value) is not bool:
            return "Required observation must be an explicit boolean: " + path
        if isinstance(value, (list, dict)):
            return "Required observation needs an explicit scalar value: " + path
    for rule in contract.get("comparability_requirements", []):
        # Bare template scalars have the units declared by this versioned
        # contract; quantity objects were checked above. Never guess a unit.
        comparison = dict(actual)
        path = rule["parameter"]
        _, value = lookup(actual, path)
        if not isinstance(value, dict) and path in contract.get("units", {}):
            comparison[path] = {"value": value, "unit": contract["units"][path]}
        status, reason = compare(comparison, rule)
        if status != "pass":
            return "Observation is not comparable without an explicit adapter conversion/review: " + rule["parameter"] + ": " + reason
    return None


def _comparison_features(parameters, contract, *, actual=False):
    if contract is None:
        return _features(parameters)
    result = {}
    for proposed_path, actual_path in contract["parameter_mapping"].items():
        present, value = lookup(parameters, actual_path if actual else proposed_path)
        if isinstance(value, dict):
            value = value.get("value")
        # Both sides use the same explicit unit; comparability gates handle gas
        # reference identity. Targets and derived indices never enter distances.
        key = proposed_path + "|" + contract.get("units", {}).get(actual_path, "")
        result[key] = value if present else None
    return result


def validate_candidates(candidates, question_ids):
    if not isinstance(candidates, list):
        raise ValueError("candidates must be an adapter-supplied list")
    result, ids = [], set()
    for original in candidates:
        c = json_copy(original)
        if not isinstance(c, dict):
            raise ValueError("Candidate must be an object")
        c["id"] = text(c.get("id", c.get("candidate_id")), "candidate ID")
        if c["id"] in ids:
            raise ValueError("Candidate IDs must be unique")
        ids.add(c["id"])
        if not isinstance(c.get("parameters"), dict) or not c["parameters"]:
            raise ValueError("Candidate parameters must be a nonempty object")
        if type(c.get("feasible")) is not bool:
            raise ValueError("Candidate feasible must be an explicit adapter boolean")
        strings(c.get("rejection_reasons"), "candidate rejection_reasons")
        strings(c.get("question_ids"), "candidate question_ids")
        strings(c.get("evidence_ids"), "candidate evidence_ids")
        if any(q not in question_ids for q in c["question_ids"]):
            raise ValueError("Candidate references an unknown question ID")
        if c["feasible"] and not c["question_ids"]:
            raise ValueError("Each feasible candidate needs an explicit prospective question")
        if c.get("engineering_status") not in {"conditional", "validated", "rejected", "unsupported"}:
            raise ValueError("Explicit candidate engineering_status is required")
        if c["feasible"] and (c["engineering_status"] in {"rejected", "unsupported"} or c["rejection_reasons"]):
            raise ValueError("Candidate feasible flag contradicts its engineering rejection")
        if not c["feasible"] and not c["rejection_reasons"]:
            raise ValueError("Infeasible candidates require rejection reasons")
        costs = c.get("costs", {})
        if not isinstance(costs, dict):
            raise ValueError("Candidate costs must be an explicit numeric map")
        for key, value in costs.items():
            text(key, "cost name (include its unit)")
            if value is not None:
                number(value, "candidate cost " + key, minimum=0)
        result.append(c)
    return result


def identity(c):
    return digest({"parameters": c["parameters"], "inventory_ids": sorted(c.get("inventory_ids", [])),
                   "substrate_id": c.get("substrate_id"), "topology_id": c.get("topology_id")})


def _features(parameters, prefix=""):
    result = {}
    for key, value in parameters.items():
        name = prefix + key
        if isinstance(value, dict) and "value" in value:
            result[name + "|" + str(value.get("unit", ""))] = value["value"]
        elif isinstance(value, dict):
            result.update(_features(value, name + "."))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[name] = value
    return result


def _numeric(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool)


def _distance(a, b, scales):
    keys = set(a) | set(b)
    if not keys:
        return 1.
    values = []
    for key in keys:
        x, y = a.get(key), b.get(key)
        if x is None or y is None:
            values.append(1.)
        elif _numeric(x) and _numeric(y):
            span = scales.get(key, 0.)
            values.append(min(1., abs(x - y) / span) if span else float(x != y))
        else:
            values.append(float(x != y))
    return sum(values) / len(values)


def qualified_feedback(results, objective, domain_hash, observation_contract=None):
    qualified, excluded = [], []
    for record in results:
        reason = None
        measurement = record["analytes"].get(objective.get("metric"))
        if record.get("source_observation_domain_sha256") != domain_hash:
            reason = "Different chemistry/observation domain; no automatic transfer"
        elif record["experiment_mode"] != "flow":
            reason = "Only actual qualified flow observations drive numerical feedback"
        elif record["outcome_status"] != "qualified":
            reason = "Unqualified, assay or operational failure; not assigned zero objective"
        elif analytical_method_issue(record.get("analytical_method")):
            reason = analytical_method_issue(record.get("analytical_method"))
        elif not measurement or measurement["value"] is None:
            reason = "Objective analyte was not measured"
        elif measurement.get("unit") != objective.get("unit"):
            reason = "Objective measurement unit differs; no undeclared conversion"
        elif measurement.get("qualifier", "exact") != "exact":
            reason = "Censored/estimated objective retained but not treated as an exact ranking value"
        else:
            reason = _actual_condition_issue(record, observation_contract)
        if reason:
            excluded.append({"result_id": record["id"], "reason": reason})
        else:
            qualified.append(record)
    reverse = objective.get("direction", "maximize") == "maximize"
    qualified.sort(key=lambda r: ((-1 if reverse else 1) * r["analytes"][objective["metric"]]["value"], r["id"]))
    return qualified, excluded


def feedback_summary(campaign):
    """Model-context companion: all records remain visible with reuse boundaries."""
    spec, results = campaign["spec"], campaign.get("results", [])
    superseded = {r["supersedes"] for r in results if r.get("supersedes")}
    active = [r for r in results if r["id"] not in superseded]
    domain = campaign.get("observation_domain_sha256", digest({"chemistry": spec.get("chemistry"), "observation_domain": spec.get("observation_domain")}))
    eligible, excluded = qualified_feedback(active, spec["objective"], domain, spec.get("observation_contract"))
    excluded += [{"result_id": ident, "reason": "Superseded by a documented correction; original retained"} for ident in sorted(superseded)]
    contract = spec.get("observation_contract")
    return {"numerical_optimization_feedback_eligible_ids": [r["id"] for r in eligible],
            "excluded_from_numerical_feedback": excluded,
            "observation_contract": deepcopy(contract),
            "measured_comparison_parameters": deepcopy(contract["parameter_mapping"]) if contract else None,
            "recorded_context_not_distance_dimensions": [p for p in contract["required_actual_conditions"]
                if p not in contract["parameter_mapping"].values()] if contract else [],
            "context_policy": "Required context is retained; comparability uses only explicitly declared gates. Unmapped actual context is not a numerical distance dimension. Assumed/proposed values are never copied into actual observations.",
            "policy": "Original records remain available for qualitative review; failed, superseded, non-flow, incompatible-domain, censored or incompletely observed conditions must not be treated as qualified objective values. No zero imputation."}


def select(spec, results, request, domain_hash):
    """Return a complete frozen selection or a rejected audit (never partial N)."""
    request = json_copy(request)
    if not isinstance(request, dict):
        raise ValueError("Generation request must be an object")
    allowed = {"requested_count", "count", "repeat_count", "include_repeat", "repeat_of", "allow_conditional", "budget",
               "selected_candidate_ids", "model_review", "model_route", "mode", "selection_method", "objective_mode", "request_note",
               "researcher_instructions", "expected_campaign_version", "expected_revision_sha256"}
    if set(request) - allowed:
        raise ValueError("Unsupported generation fields; revise the campaign instead: " + ", ".join(sorted(set(request) - allowed)))
    if "researcher_instructions" in request and not isinstance(request["researcher_instructions"], str):
        raise ValueError("researcher_instructions must be text; numerical/constraint changes require a campaign revision")
    if request.get("objective_mode", spec["objective"]["mode"]) != spec["objective"]["mode"]:
        raise ValueError("Requested objective differs from frozen campaign objective; revise it before generation")
    n = request.get("requested_count")
    repeats = request.get("repeat_count", int(request.get("include_repeat") is True))
    if type(n) is not int or not 1 <= n <= 100:
        raise ValueError("requested_count must be an integer from 1 to 100 unique conditions")
    if type(repeats) is not int or not 0 <= repeats <= 100:
        raise ValueError("repeat_count must be an integer from 0 to 100, additional to unique conditions")
    allow = request.get("allow_conditional", True)
    if type(allow) is not bool:
        raise ValueError("allow_conditional must be a boolean")
    budget = request.get("budget", {})
    if not isinstance(budget, dict):
        raise ValueError("budget must be a map of unit-labelled cost names to caps")
    for key, value in budget.items():
        text(key, "budget key")
        number(value, "budget cap", minimum=0)
    pool, eligible, seen = deepcopy(spec["candidates"]), [], set()
    for c in pool:
        reasons = list(c["rejection_reasons"])
        c["requirement_checks"] = candidate_checks(c, spec["inventory"], spec["requirements"])
        reasons.extend(check["id"] + ": " + check["reason"] for check in c["requirement_checks"] if check["blocking"])
        if not c["feasible"] or c["engineering_status"] in {"rejected", "unsupported"}:
            reasons.append("Adapter did not establish candidate feasibility")
        if not allow and c["engineering_status"] != "validated":
            reasons.append("Request excludes conditional engineering scenarios")
        if not allow and any(check["status"] in {"unknown", "release_pending"} or check.get("basis_status") == "assumed" for check in c["requirement_checks"]):
            reasons.append("Request excludes unresolved inventory/requirement assumptions")
        for cost in budget:
            if c.get("costs", {}).get(cost) is None:
                reasons.append("Budget cost is unknown: " + cost)
        key = identity(c)
        if key in seen:
            reasons.append("Duplicate realizable condition in declared candidate pool")
        if not reasons:
            seen.add(key)
            eligible.append(c)
        c["selection_disposition"] = "rejected" if reasons else "not_selected"
        c["selection_reasons"] = reasons or ["Eligible candidate; no outcome prediction assigned"]
    objective = spec["objective"]
    observation_contract = spec.get("observation_contract")
    feedback, excluded = qualified_feedback(results, objective, domain_hash, observation_contract)
    audit = {"requested_count": n, "repeat_count": repeats, "candidate_pool": pool, "selected": [], "repeats": [],
             "feedback_used": [r["id"] for r in feedback] if objective["mode"] == "optimize" else [],
             "feedback_exclusions": excluded, "budget": budget, "cost_totals": {}, "rejection_reasons": [],
             "selection_method": "deterministic_coverage", "predicted_outcomes": None,
             "observation_contract": deepcopy(observation_contract),
             "feedback_dimension_policy": "Explicit measured-control mapping; targets and derived/assumed quantities are not actual measurements" if observation_contract else "Legacy exact-key actual/proposed check; no automatic interpretation of dimensions",
             "researcher_instructions_role": "Model/researcher context; no implicit structured constraint or candidate override",
             "policy": "Only adapter-supplied candidates; exact requested unique count; repeats are separate. No BO, fitted batch kinetics or fabricated predictions.",
             "laboratory_released": False, "experimental_validation": False, "model_review": request.get("model_review"),
             "model_route": request.get("model_route")}
    if not pool:
        audit["rejection_reasons"].append("No candidate pool supplied; a chemistry/engineering adapter or reviewed JSON candidates are required")
    fixed = request.get("selected_candidate_ids")
    if fixed is not None:
        strings(fixed, "selected_candidate_ids")
        if len(fixed) != n or len(set(fixed)) != n:
            audit["rejection_reasons"].append("Model/request selection must contain exactly N unique candidate IDs")
        by_id = {c["id"]: c for c in eligible}
        if any(cid not in by_id for cid in fixed):
            audit["rejection_reasons"].append("Model/request selected an unknown or ineligible candidate")
        if not audit["rejection_reasons"]:
            audit["selected"] = [by_id[cid] for cid in fixed]
        audit["selection_method"] = "externally_proposed_ids_validated_by_deterministic_gates"
    else:
        if request.get("mode") in {"model", "llm"} or request.get("model_review") is not None or request.get("model_route") not in {None, "deterministic", "offline"}:
            audit["rejection_reasons"].append("Model mode requires explicit selected_candidate_ids; no silent deterministic fallback")
        if objective["mode"] == "generalize" and any(not c.get("substrate_id") for c in eligible):
            audit["rejection_reasons"].append("Generalize mode requires explicit substrate_id on every eligible candidate")
        features = {c["id"]: _comparison_features(c["parameters"], observation_contract) for c in eligible}
        keys = set().union(*(f.keys() for f in features.values())) if features else set()
        scales = {}
        for key in keys:
            vals = [f[key] for f in features.values() if key in f and _numeric(f[key])]
            scales[key] = max(vals) - min(vals) if vals else 0.
        prior = [_comparison_features(r["actual_conditions"], observation_contract, actual=True) for r in results if r["experiment_mode"] == "flow"
                 and r["outcome_status"] == "qualified" and r.get("source_observation_domain_sha256") == domain_hash
                 and analytical_method_issue(r.get("analytical_method")) is None
                 and _actual_condition_issue(r, observation_contract) is None]
        best = _comparison_features(feedback[0]["actual_conditions"], observation_contract, actual=True) if feedback else None
        if objective["mode"] == "optimize" and best is not None:
            audit["selection_method"] = "observed_best_neighborhood_then_diversity_not_predictive_model"
            audit["policy"] += " The first choice is closest to the best qualified measured flow observation; subsequent choices balance that proximity and geometric diversity equally. This is a disclosed heuristic, not a calibrated prediction."
        selected, spent = [], {key: 0. for key in budget}
        while len(selected) < n and not audit["rejection_reasons"]:
            choices = [c for c in eligible if c not in selected and all(spent[k] + c["costs"][k] <= budget[k] + 1e-12 for k in budget)]
            if not choices:
                break
            occupied = [features[c["id"]] for c in selected] + prior
            groups = {c.get("substrate_id") for c in selected}
            observed_groups = {r.get("substrate_id", r["actual_conditions"].get("substrate_id")) for r in results
                               if r["outcome_status"] == "qualified" and r["experiment_mode"] == "flow"
                               and analytical_method_issue(r.get("analytical_method")) is None
                               and r.get("source_observation_domain_sha256") == domain_hash}
            def score(c):
                f = features[c["id"]]
                spread = min((_distance(f, o, scales) for o in occupied), default=1.)
                if objective["mode"] == "optimize" and best is not None:
                    value = -_distance(f, best, scales) if not selected else (spread - _distance(f, best, scales)) / 2
                    return (-value, c["id"])
                if objective["mode"] == "generalize":
                    return (c["substrate_id"] in groups, c["substrate_id"] in observed_groups, -spread, c["id"])
                return (-spread, c["id"])
            choice = min(choices, key=score)
            selected.append(choice)
            for key in budget:
                spent[key] += choice["costs"][key]
        audit["selected"] = selected
    if len(audit["selected"]) != n:
        audit["rejection_reasons"].append(f"Declared selection policy cannot supply {n} eligible unique conditions within the stated budget; no partial screen returned")
    selected = audit["selected"]
    if repeats and selected:
        ref = request.get("repeat_of", selected[0]["id"])
        matches = [c for c in selected if c["id"] == ref]
        if len(matches) != 1:
            audit["rejection_reasons"].append("repeat_of must name a condition selected in this round")
        else:
            for i in range(repeats):
                r = deepcopy(matches[0])
                r.update(id=f"independent-repeat-{i + 1}", repeat_of=ref,
                         repeat_reason="Explicit independent re-preparation and operation proposed; not yet performed")
                if r["id"] in {c["id"] for c in selected}:
                    audit["rejection_reasons"].append("Generated repeat ID conflicts with a candidate ID")
                audit["repeats"].append(r)
    for key in budget:
        total = sum(c.get("costs", {}).get(key, 0) for c in selected + audit["repeats"])
        audit["cost_totals"][key] = total
        if total > budget[key] + 1e-12:
            audit["rejection_reasons"].append("Total including independent repeats exceeds budget: " + key)
    if audit["rejection_reasons"]:
        audit["attempted_selected_ids"] = [c["id"] for c in selected]
        audit.update(status="rejected", selected=[], repeats=[])
        return audit
    for c in selected:
        c["selection_disposition"] = "selected"
        c["selection_reasons"] = [audit["selection_method"], "Prospective question IDs: " + ", ".join(c["question_ids"])]
        c["selection_reason"] = "; ".join(c["selection_reasons"])
    conditional = any(c["engineering_status"] != "validated" or
                      any(x["status"] in {"unknown", "release_pending"} or x.get("basis_status") == "assumed" for x in c["requirement_checks"]) for c in selected)
    audit.update(status="conditional" if conditional else "engineering_checked",
                 unique_design_count=n, total_proposed_runs=n + repeats,
                 execution_policy="Proposed sequential runs; no parallel resource allocation or equipment execution implied")
    return audit
