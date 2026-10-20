"""Explicit generic inventory and sourced requirement contracts.

This validates declared data and comparisons. It is not an automatic safety or
chemistry compatibility certificate; adapter engineering checks remain separate.
"""
from __future__ import annotations

from .evidence import json_copy, number, text, strings

OPERATORS = {"<", "<=", ">", ">=", "==", "!=", "in", "not_in"}


def validate_constraint(constraint):
    c = json_copy(constraint)
    if not isinstance(c, dict):
        raise ValueError("Constraint must be an object")
    text(c.get("parameter"), "constraint parameter")
    if c.get("operator") not in OPERATORS:
        raise ValueError("Constraint operator must be explicit: < <= > >= == != in not_in")
    if "value" not in c:
        raise ValueError("Constraint value must be explicit (null means unresolved)")
    if c["operator"] in {"in", "not_in"} and c["value"] is not None and not isinstance(c["value"], list):
        raise ValueError("Membership constraints need a list of allowed/excluded values")
    if c["operator"] in {"<", "<=", ">", ">="} and c["value"] is not None:
        number(c["value"], "numeric constraint value")
    if c.get("unit") is not None:
        text(c["unit"], "constraint unit")
    return c


def validate_inventory(inventory):
    inv = json_copy(inventory)
    if not isinstance(inv, dict) or not isinstance(inv.get("items"), list):
        raise ValueError("Inventory requires an explicit items list; an empty inventory is allowed for abstract candidates")
    inv.setdefault("schema_version", "flowpilotx-inventory-v1")
    if inv["schema_version"] != "flowpilotx-inventory-v1":
        raise ValueError("Unsupported inventory schema; adapt the source without overwriting it")
    ids = []
    for item in inv["items"]:
        if not isinstance(item, dict):
            raise ValueError("Inventory item must be an object")
        ids.append(text(item.get("id"), "inventory item ID"))
        text(item.get("kind"), "inventory item kind")
        if "quantity" not in item or "availability" not in item:
            raise ValueError("Inventory quantity and availability must be explicit; use null/unknown when unconfirmed")
        if item["quantity"] is not None:
            number(item["quantity"], "inventory quantity", minimum=0)
            if isinstance(item["quantity"], float) and not item["quantity"].is_integer():
                raise ValueError("Inventory item quantity counts physical items; use resources for material amounts")
        if item["availability"] not in {"available", "unavailable", "unknown"}:
            raise ValueError("Inventory availability must be available, unavailable or unknown")
        if not isinstance(item.get("capabilities", {}), dict):
            raise ValueError("Inventory capabilities must be an object")
        if not isinstance(item.get("limits", []), list):
            raise ValueError("Inventory limits must be a list with explicit comparison operators")
        item["limits"] = [validate_constraint(c) for c in item.get("limits", [])]
        if not isinstance(item.get("source_references", []), list):
            raise ValueError("Inventory source_references must be a list")
    if len(set(ids)) != len(ids):
        raise ValueError("Inventory IDs must be unique")
    return inv


def validate_requirements(requirements):
    if not isinstance(requirements, list):
        raise ValueError("requirements must be a list")
    result, ids = [], set()
    for original in requirements:
        r = json_copy(original)
        rid = text(r.get("id"), "requirement ID")
        if rid in ids:
            raise ValueError("Requirement IDs must be unique")
        ids.add(rid)
        text(r.get("description"), "requirement description")
        if r.get("scope") not in {"candidate", "release", "context"}:
            raise ValueError("Requirement scope must be candidate, release or context")
        if r.get("status") not in {"confirmed", "assumed", "unresolved", "conflicting"}:
            raise ValueError("Requirement status must be explicit")
        if r.get("severity", "hard") not in {"hard", "soft"}:
            raise ValueError("Requirement severity must be hard or soft")
        r.setdefault("severity", "hard")
        if not r.get("source_reference"):
            raise ValueError("Requirements need a source_reference")
        if r["scope"] == "candidate":
            r = validate_constraint(r)
        result.append(r)
    return result


def lookup(parameters, path):
    if path in parameters:
        return True, parameters[path]
    value = parameters
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False, None
        value = value[part]
    return True, value


def compare(parameters, rule):
    present, actual = lookup(parameters, rule["parameter"])
    if not present or actual is None or rule["value"] is None:
        return "unknown", "Required value or source limit is unknown"
    expected = rule["value"]
    if isinstance(actual, dict) and "value" in actual:
        if rule.get("unit") and actual.get("unit") != rule["unit"]:
            return "unknown", "Units differ or are missing; automatic conversion is not declared"
        actual = actual["value"]
    elif rule.get("unit"):
        return "unknown", "Dimensional requirement needs a value/unit parameter object"
    if actual is None:
        return "unknown", "Parameter value is unknown"
    op = rule["operator"]
    try:
        if op in {"<", "<=", ">", ">="}:
            number(actual, "parameter")
        passed = {"<": lambda: actual < expected, "<=": lambda: actual <= expected,
                  ">": lambda: actual > expected, ">=": lambda: actual >= expected,
                  "==": lambda: ((isinstance(actual, bool) == isinstance(expected, bool)) and actual == expected),
                  "!=": lambda: actual != expected, "in": lambda: actual in expected,
                  "not_in": lambda: actual not in expected}[op]()
    except (TypeError, ValueError):
        return "unknown", "Parameter type is incompatible with the declared comparison"
    return ("pass", "Matches declared requirement") if passed else ("fail", "Violates declared requirement")


def candidate_checks(candidate, inventory, requirements):
    checks = []
    for r in requirements:
        if r["scope"] != "candidate":
            checks.append({"id": r["id"], "status": "context" if r["scope"] == "context" else "release_pending",
                           "blocking": False, "reason": r["description"]})
            continue
        status, reason = compare(candidate["parameters"], r)
        if r["status"] in {"unresolved", "conflicting"}:
            status, reason = "unknown", "Requirement is unresolved or conflicting"
        checks.append({"id": r["id"], "status": status,
                       "basis_status": r["status"], "blocking": r["severity"] == "hard" and status != "pass", "reason": reason})
    assets = {i["id"]: i for i in inventory["items"]}
    assigned = candidate.get("inventory_ids", [])
    strings(assigned, "candidate inventory_ids")
    for ident in assigned:
        item = assets.get(ident)
        if item is None or item["availability"] == "unavailable" or item["quantity"] == 0:
            checks.append({"id": "inventory:" + ident, "status": "fail", "blocking": True,
                           "reason": "Assigned inventory does not exist or is unavailable"})
            continue
        if item["availability"] == "unknown" or item["quantity"] is None:
            checks.append({"id": "inventory:" + ident, "status": "unknown", "blocking": False,
                           "reason": "Inventory availability is unknown; conditional scenario only"})
        for i, limit in enumerate(item.get("limits", [])):
            status, reason = compare(candidate["parameters"], limit)
            checks.append({"id": f"inventory:{ident}:limit:{i + 1}", "status": status,
                           "blocking": status != "pass", "reason": reason})
    return checks
