"""Lossless JSON evidence and qualitative question/result contracts.

No interpretation here converts batch duration into flow kinetics, normalizes
analytical values, or assigns numerical probabilities to verbal hypotheses.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import io
import json
import math
from pathlib import Path


def json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def number(value, label, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def strings(value, label, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{label} must be {'a nonempty' if nonempty else 'a'} list")
    for item in value:
        text(item, label)
    return value


def validate_evidence(value):
    """Preserve adapter evidence verbatim; identity/curation remains source-owned.

    A row list or a structured archive is allowed. Repeated record IDs within an
    explicit records list are rejected, rather than deduplicated by conditions.
    """
    result = json_copy(value)
    if not isinstance(result, (dict, list)):
        raise ValueError("evidence must be a record list or structured source archive")
    rows = result if isinstance(result, list) else result.get("records", [])
    if not isinstance(rows, list):
        raise ValueError("evidence.records must be a list")
    ids = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each evidence record must be an object")
        ids.append(text(row.get("id", row.get("record_id")), "evidence record ID"))
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evidence record IDs; reconcile source representations explicitly")
    return result


def citation_ids(evidence, source_references=(), results=()):
    """IDs resolvable to archived evidence/results, not arbitrary plausible labels."""
    found = set()
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"id", "record_id", "source_id", "source_record_id"} and isinstance(child, str):
                    found.add(child)
                if key == "visible_citation_ids" and isinstance(child, list):
                    found.update(item for item in child if isinstance(item, str))
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(evidence)
    visit(list(results))
    visit(list(source_references))
    for source in source_references:
        if isinstance(source, str):
            found.add(source)
        elif isinstance(source, dict):
            if isinstance(source.get("path"), str):
                found.add(source["path"])
            if isinstance(source.get("sha256"), str):
                found.add("sha256:" + source["sha256"])
    return found


def import_evidence(name, raw):
    """Import chemistry-agnostic CSV/JSON without interpreting unknown columns.

    Explicit IDs and source pointers survive. When no shared dataset identity is
    supplied, the file hash forms its namespace. Identical operating settings
    never cause deduplication. Proposed or generic conditions are not promoted to
    actual measurements, and historical rows are not automatic feedback results.
    """
    text(name, "source filename")
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 8 * 1024 * 1024:
        raise ValueError("Evidence source must contain 1 byte to 8 MiB")
    source_hash = hashlib.sha256(raw).hexdigest()
    source_id = "sha256:" + source_hash
    suffix = Path(name).suffix.lower()
    metadata = {}
    if suffix == ".json":
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate JSON object field; clarify source before import: " + key)
                result[key] = value
            return result
        data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs,
                          parse_constant=lambda v: (_ for _ in ()).throw(ValueError("Nonfinite JSON number: " + v)))
        if isinstance(data, dict):
            metadata = {k: deepcopy(v) for k, v in data.items() if k not in {"records", "rows"}}
            data = data.get("records", data.get("rows"))
        if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
            raise ValueError("JSON evidence requires a record list or an object containing records")
        original_rows = [(i, r) for i, r in enumerate(data, 1)]
    elif suffix == ".csv":
        try:
            decoded = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            decoded = raw.decode("cp949")
        reader = csv.DictReader(io.StringIO(decoded, newline=""))
        if not reader.fieldnames or any(not h or not h.strip() for h in reader.fieldnames) or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV needs unique, nonempty column names")
        original_rows = []
        for i, r in enumerate(reader, 2):
            if None in r:
                raise ValueError("CSV row exceeds header width at row " + str(i))
            if any(v not in {None, ""} for v in r.values()):
                original_rows.append((i, r))
                if len(original_rows) > 10000:
                    raise ValueError("Evidence source exceeds 10000 rows")
    else:
        raise ValueError("Generic evidence supports CSV and JSON; other formats require an explicit source adapter")
    if not 1 <= len(original_rows) <= 10000:
        raise ValueError("Evidence source requires 1 to 10000 records")
    rows, seen = [], set()
    for index, original in original_rows:
        r = deepcopy(original)
        dataset = r.get("dataset_id", metadata.get("dataset_id")) or source_id
        entry = r.get("source_entry", r.get("record_id", r.get("id")))
        if entry is None or str(entry).strip() == "":
            # Source-local pointers are allowed, but never imply shared exports.
            entry = "row:" + str(index)
            identity_dataset = source_id
        else:
            identity_dataset = dataset
        entry = str(entry)  # No float coercion: exact large/string identities survive.
        mode = r.get("experiment_mode", metadata.get("experiment_mode", "unknown"))
        system = r.get("catalyst_system", metadata.get("catalyst_system", "unspecified"))
        identity = [identity_dataset, system, mode, entry]
        rid = "evidence:" + digest(identity)[:24]
        if rid in seen:
            raise ValueError("Repeated source identity within one import; reconcile explicitly")
        seen.add(rid)
        if "id" in r or "record_id" in r:
            r["provided_record_id"] = r.get("record_id", r.get("id"))
        r.update(id=rid, record_id=rid, dataset_id=dataset, source_entry=entry,
                 experiment_mode=mode, catalyst_system=system)
        r["source_representations"] = [*deepcopy(original.get("source_representations", [])), {
            "source_id": source_id, "source_sha256": source_hash, "source_name": name,
            "source_row": index, "source_record_id": source_id + "#row:" + str(index),
            "raw_fields": deepcopy(original), "source_metadata": deepcopy(metadata)}]
        r["interpretation_policy"] = "Uninterpreted source record; explicit adapter/curation required for numerical reuse"
        r["feedback_eligible"] = False
        rows.append(r)
    return {"schema_version": "flowpilotx-generic-evidence-import-v1", "records": rows,
            "sources": [{"name": name, "source_id": source_id, "sha256": source_hash, "format": suffix[1:], "size_bytes": len(raw)}],
            "summary": {"record_count": len(rows), "canonical_count": len(rows), "source_count": 1,
                        "mode_counts": {mode: sum(r["experiment_mode"] == mode for r in rows) for mode in sorted({r["experiment_mode"] for r in rows}, key=str)},
                        "normalization_applied": False, "chemistry_mapping_inferred": False,
                        "policy": "Unknown columns preserved; explicit dataset/entry identity only; imported history is not qualified measured feedback"}}


def validate_question(question, previous=None):
    q = json_copy(question)
    if not isinstance(q, dict):
        raise ValueError("Question must be an object")
    q["id"] = text(q.get("id", q.get("question_id")), "question ID")
    q["question"] = text(q.get("question", q.get("title")), "scientific question")
    for field in ["competing_explanations", "expected_observations", "measurements", "decision_rules"]:
        strings(q.get(field), field, nonempty=True)
    strings(q.get("confounders", []), "confounders")
    strings(q.get("evidence_ids", []), "question evidence_ids")
    q.setdefault("confounders", [])
    q.setdefault("evidence_ids", [])
    status = q.setdefault("status", "unresolved")
    if status not in {"unresolved", "supported", "contradicted", "revised", "retired"}:
        raise ValueError("Question status must be a qualitative documented state")
    if any(k in q for k in ["posterior", "posterior_probability", "probability", "confidence_score"]):
        raise ValueError("Question registry is qualitative; numerical belief requires a separate calibrated model")
    if previous is None and status != "unresolved":
        raise ValueError("New prospective questions start unresolved; document later amendments")
    if previous is not None:
        amendment = q.get("amendment")
        if not isinstance(amendment, dict):
            raise ValueError("Question revisions require a documented amendment")
        text(amendment.get("reason"), "amendment reason")
        text(amendment.get("author"), "amendment author")
        strings(amendment.get("evidence_ids"), "amendment evidence_ids", nonempty=True)
    q["belief_semantics"] = "qualitative_documented_assessment_not_bayesian_posterior"
    return q


def analytical_method_issue(method):
    """A qualified assay needs a recorded method, not a completed status flag.

    This checks explicit reporting only; it does not certify calibration or
    chemical suitability. Unqualified records may retain unknown methods.
    """
    unknown = {"", "unknown", "unknown method", "n/a", "na", "-", "pending", "tbd",
               "not recorded", "not specified", "not assessed", "to be determined"}
    if isinstance(method, dict):
        status = str(method.get("status", "")).strip().lower().replace(" ", "_")
        if status in {"unknown", "missing", "pending", "unconfirmed", "proposed", "unmeasured", "not_assessed"}:
            return "Analytical method remains unknown or unconfirmed"
        name = method.get("name")
    else:
        name = method
    if not isinstance(name, str) or name.strip().lower() in unknown:
        return "Qualified analytical result requires an explicit known method name"
    return None


def validate_result(record, rounds):
    """Only explicit observations can become feedback; failure is never zero yield."""
    r = json_copy(record)
    if not isinstance(r, dict):
        raise ValueError("Each result must be an object")
    for key in ["id", "round_id", "design_id"]:
        text(r.get(key), "result " + key)
    matching = [item for item in rounds if item["id"] == r["round_id"]]
    if len(matching) != 1 or matching[0].get("status") == "rejected":
        raise ValueError("Result must reference an existing successfully frozen round")
    available = matching[0]["selected"] + matching[0]["repeats"]
    candidates = [item for item in available if item["id"] == r["design_id"]]
    if len(candidates) != 1:
        raise ValueError("Result design_id is not a selected condition or repeat in its round")
    if r.get("experiment_mode") not in {"flow", "batch", "other"}:
        raise ValueError("Results require an explicit experiment_mode")
    if not isinstance(r.get("actual_conditions"), dict) or not r["actual_conditions"]:
        raise ValueError("Results require explicit actual_conditions; unknown values may be null")
    if not isinstance(r.get("analytical_method"), (str, dict)) or not r["analytical_method"]:
        raise ValueError("Results require an analytical_method or explicit unknown-method record")
    if r.get("outcome_status") not in {"qualified", "unqualified", "assay_failed", "operational_failure"}:
        raise ValueError("Result outcome_status is required")
    analytes = r.get("analytes")
    if not isinstance(analytes, dict):
        raise ValueError("Results require an analytes map, including nulls for unknown measurements")
    for name, measurement in analytes.items():
        text(name, "analyte name")
        if not isinstance(measurement, dict) or "value" not in measurement:
            raise ValueError("Each analyte needs value and unit, not a bare inferred yield")
        text(measurement.get("unit"), "analyte unit")
        if measurement["value"] is not None:
            number(measurement["value"], "analyte value")
        if "uncertainty" in measurement and measurement["uncertainty"] is not None:
            number(measurement["uncertainty"], "analytical uncertainty", minimum=0)
        if measurement.get("qualifier", "exact") not in {"exact", "estimated", "<", ">", "<=", ">=", "not_detected", "unknown"}:
            raise ValueError("Unsupported analyte qualifier")
    if r["outcome_status"] == "qualified":
        method_issue = analytical_method_issue(r["analytical_method"])
        if method_issue:
            raise ValueError(method_issue)
        if not analytes or not any(v["value"] is not None for v in analytes.values()):
            raise ValueError("Qualified results require a reported analytical value")
        quality = r.get("quality")
        if not isinstance(quality, dict) or quality.get("accepted") is not True:
            raise ValueError("Qualified results require quality.accepted=true and a recorded basis")
        text(quality.get("basis"), "quality basis")
    r["proposed_conditions"] = deepcopy(candidates[0]["parameters"])
    r["source_candidate_id"] = candidates[0].get("repeat_of", candidates[0]["id"])
    r["source_contract_sha256"] = matching[0]["contract_sha256"]
    r["normalization_applied"] = False
    r["result_sha256"] = digest({k: v for k, v in r.items() if k != "result_sha256"})
    return r
