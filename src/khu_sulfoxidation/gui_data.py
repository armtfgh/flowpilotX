"""Local history import with source preservation and explicit identity boundaries."""
from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import posixpath
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile, BadZipFile

from .history import CONDITION_FIELDS, OUTCOME_FIELDS, ingest_history

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 48 * 1024 * 1024
MAX_ROWS = 10000
TEMPLATE_COLUMNS = ["dataset_id", "catalyst_system", "source_entry", "experiment_mode",
    "substrate", "solvent_1", "concentration_mol_L", "photocatalyst",
    "catalyst_loading_mol_pct", "batch_duration_h", "temperature_C", "oxidant_reported",
    "light_source_reported", "residual_sulfide_pct", "sulfoxide_pct", "sulfone_pct",
    "other_products_pct", "unknown_components_pct", "analytical_method"]
NUMERIC_CONDITIONS = {"concentration_mol_L", "catalyst_loading_mol_pct", "batch_duration_h",
    "temperature_C", "solvent_1_percentage", "liquid_flow_mL_min", "gas_reference_flow_mL_min",
    "reactor_pressure_bar_abs", "measured_residence_time_min"}
EXTRA_CONDITIONS = ["substrate", "liquid_flow_mL_min", "gas_reference_flow_mL_min",
    "reactor_pressure_bar_abs", "measured_residence_time_min"]


def _key(value):
    return re.sub(r"\s+", " ", str(value).strip()).lower()


ALIASES = {_key(k): k for k in TEMPLATE_COLUMNS + list(CONDITION_FIELDS) + list(OUTCOME_FIELDS) + EXTRA_CONDITIONS}
ALIASES.update({_key(v): k for k, v in {**CONDITION_FIELDS, **OUTCOME_FIELDS}.items()})
ALIASES.update({"entry": "source_entry", "record_id": "provided_record_id", "system": "catalyst_system",
    "mode": "experiment_mode", "solvent": "solvent_1", "concentration_m": "concentration_mol_L",
    "catalyst": "photocatalyst", "catalyst_mol_percent": "catalyst_loading_mol_pct",
    "residual_sulfide_percent": "residual_sulfide_pct", "sulfoxide_percent": "sulfoxide_pct",
    "sulfone_percent": "sulfone_pct"})


def template_csv():
    stream = io.StringIO(newline="")
    csv.writer(stream).writerow(TEMPLATE_COLUMNS)
    return stream.getvalue()


def _number(value, field):
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "-", "na", "n/a", "null", "unknown"}):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} requires a number or an empty cell")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} contains a nonnumeric reported value; retain the original file and clarify its mapping") from None
    if not math.isfinite(result):
        raise ValueError(f"{field} must not contain NaN or infinity")
    return result


def _text(value):
    return None if value is None or str(value).strip() in {"", "-"} else str(value).strip()


def _csv_rows(raw):
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp949")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames or len(set(map(_key, reader.fieldnames))) != len(reader.fieldnames):
        raise ValueError("CSV needs unique named columns")
    rows = []
    for i, row in enumerate(reader, 2):
        if None in row:
            raise ValueError(f"CSV row {i} has more values than its header")
        if any(v not in (None, "") for v in row.values()):
            rows.append((dict(row), {"source_row": i, "sheet": None, "formulas": {}}))
            if len(rows) > MAX_ROWS:
                raise ValueError("History exceeds 10000 records")
    return rows


def _xlsx_rows(raw):
    """Read cached cell values; never evaluate formulas or external links."""
    try:
        archive = ZipFile(io.BytesIO(raw))
    except BadZipFile:
        raise ValueError("XLSX is not a readable workbook") from None
    with archive:
        if sum(i.file_size for i in archive.infolist()) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Uncompressed workbook exceeds the local import limit")
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(t.text or "" for t in s.findall(".//{*}t"))
                       for s in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        sheets = ET.fromstring(archive.read("xl/workbook.xml")).findall("{*}sheets/{*}sheet")
        rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        rows = []
        for sheet in sheets:
            rid = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            target = rels[rid]
            member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            header = None
            for row in ET.fromstring(archive.read(member)).findall("{*}sheetData/{*}row"):
                cells, formulas = {}, {}
                for cell in row.findall("{*}c"):
                    address = cell.get("r", "")
                    col = re.sub(r"\d", "", address)
                    val = cell.find("{*}v")
                    value = None if val is None else val.text
                    if cell.get("t") == "s" and value is not None:
                        value = strings[int(value)]
                    elif cell.get("t") == "inlineStr":
                        value = "".join(t.text or "" for t in cell.findall(".//{*}t"))
                    cells[col] = value
                    formula = cell.find("{*}f")
                    if formula is not None:
                        formulas[address] = {"formula": formula.text, "cached_value": value}
                if header is None:
                    if any(ALIASES.get(_key(v)) == "source_entry" for v in cells.values() if v):
                        header = {k: str(v) for k, v in cells.items() if v is not None}
                        if len(set(map(_key, header.values()))) != len(header):
                            raise ValueError("Workbook header has duplicate columns")
                    continue
                if any(v is not None for v in cells.values()):
                    rows.append(({name: cells.get(col) for col, name in header.items()},
                        {"source_row": int(row.get("r")), "sheet": sheet.get("name"), "formulas": formulas}))
                    if len(rows) > MAX_ROWS:
                        raise ValueError("History exceeds 10000 records")
            if header is None:
                raise ValueError(f"Sheet {sheet.get('name')} needs a source_entry or Entry header; no sheet was silently ignored")
        return rows


def _json_rows(raw):
    value = json.loads(raw.decode("utf-8-sig"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("JSON contains nonfinite numbers")))
    metadata = {}
    if isinstance(value, dict):
        metadata = {k: value[k] for k in ["dataset_id", "catalyst_system", "experiment_mode"] if k in value}
        value = value.get("records", value.get("rows"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("JSON must be a row list or an object with records and optional dataset_id/system/mode")
    if len(value) > MAX_ROWS:
        raise ValueError("History exceeds 10000 records")
    rows = []
    for i, row in enumerate(value, 1):
        rows.append(({**metadata, **row}, {"source_row": i, "sheet": None, "formulas": {}, "raw_json_record": row}))
    return rows


def _entry(value):
    if value is None or str(value).strip() == "":
        raise ValueError("Every row needs a stable source_entry; row identity must not be inferred from conditions")
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        text = str(int(text.partition(".")[0]))
    if len(text) > 200:
        raise ValueError("source_entry is too long")
    return text


def _mapped(fields):
    result = {}
    for k, v in fields.items():
        key = ALIASES.get(_key(k), str(k))
        if key in result and result[key] != v:
            raise ValueError(f"Multiple conflicting columns map to {key}")
        result[key] = v
    return result


def _reported_fields(raw_fields):
    """Explicit actual groups, including null, outrank generic/proposed fields."""
    top = _mapped(raw_fields)
    status = _text(top.get("status") or top.get("source_status") or top.get("record_status") or top.get("condition_status") or top.get("record_kind"))
    proposal = bool(status and re.search(r"propos|plan|predict|simulat|conditional", status, re.I))
    explicit_proposed = any(k in top for k in ("proposed_conditions", "proposed_setpoints"))
    actual = top.get("actual")
    if "actual" in top and actual is not None and not isinstance(actual, dict):
        raise ValueError("actual must be an object or null")
    result = dict(top)
    group_fields = {"conditions": [*CONDITION_FIELDS, *EXTRA_CONDITIONS], "outcomes": list(OUTCOME_FIELDS)}
    explicit_groups = {}
    for group, fields in group_fields.items():
        generic = top.get(group)
        if generic is not None and not isinstance(generic, dict):
            raise ValueError(f"{group} must be an object or null")
        values = {k: v for k, v in _mapped(generic or {}).items() if k in fields}
        actual_name = "actual_conditions" if group == "conditions" else "actual_outcomes"
        reported_name = "reported_actual_conditions" if group == "conditions" else "reported_outcomes"
        explicit = False
        for present, source in [
            ("actual" in top, actual.get(group, actual) if isinstance(actual, dict) else None),
            (actual_name in top, top.get(actual_name)),
            (reported_name in top, top.get(reported_name)),
        ]:
            if present:
                if source is not None and not isinstance(source, dict):
                    raise ValueError(f"Explicit {group} must be an object or null")
                values = {k: None for k in fields}
                values.update({k: v for k, v in _mapped(source or {}).items() if k in fields})
                explicit = True
        explicit_groups[group] = explicit
        if explicit or generic is not None:
            result.update(values)
    # Null actual placeholders do not turn an otherwise explicit proposal into
    # an observation. Preserve any nonmissing reported actuals independently.
    has_explicit_actual = any(explicit_groups[group] and any(
        result.get(field) is not None and str(result[field]).strip().lower() not in {"", "-", "na", "n/a", "null", "unknown"}
        for field in fields) for group, fields in group_fields.items())
    if explicit_proposed and not has_explicit_actual:
        proposal = True
    proposed = top.get("proposed_conditions", top.get("proposed_setpoints"))
    if proposed is not None and not isinstance(proposed, dict):
        raise ValueError("proposed_conditions must be an object or null")
    if proposed is None and proposal:
        proposed = top.get("conditions") or {k: top[k] for k in group_fields["conditions"] if k in top}
    if proposal:
        for group, fields in group_fields.items():
            if not explicit_groups[group]:
                result.update({k: None for k in fields})
    return result, status, proposal, deepcopy(proposed)


def canonical_record_id(dataset, system, mode, entry):
    if dataset == "khu" and system in {"homogeneous", "heterogeneous"} and mode == "batch" and entry.isdigit():
        return f"khu:{system}:raw-entry:{int(entry):04d}"
    identity = json.dumps([dataset, system, mode, entry], ensure_ascii=False, separators=(",", ":"))
    return "local-history:" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def parse_history_upload(name, raw, *, known_sources=None):
    """Parse without writing; use declared dataset IDs to reconcile exports safely."""
    if not isinstance(name, str) or not name or len(name) > 240:
        raise ValueError("An upload filename is required")
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload must contain 1 byte to 8 MiB")
    suffix = Path(name).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".json"}:
        raise ValueError("Supported history formats: CSV, XLSX and JSON")
    source_hash = hashlib.sha256(raw).hexdigest()
    known = (known_sources or {}).get(source_hash)
    source = {"name": name, "sha256": source_hash, "source_id": "sha256:" + source_hash,
              "format": suffix[1:], "size_bytes": len(raw)}
    defaults = known or {}
    parsed = {".csv": _csv_rows, ".xlsx": _xlsx_rows, ".json": _json_rows}[suffix](raw)
    if not parsed or len(parsed) > MAX_ROWS:
        raise ValueError("History needs 1 to 10000 records")
    records, warnings, seen = [], [], set()
    for raw_fields, location in parsed:
        mapped, source_status, is_proposal, proposed_conditions = _reported_fields(raw_fields)
        entry = _entry(mapped.get("source_entry"))
        dataset = _text(mapped.get("dataset_id")) or defaults.get("dataset_id") or "source:" + source_hash
        system = (_text(mapped.get("catalyst_system")) or defaults.get("catalyst_system") or "unspecified").lower()
        if system not in {"homogeneous", "heterogeneous", "unspecified"}:
            raise ValueError("catalyst_system must be homogeneous, heterogeneous or unspecified")
        mode = (_text(mapped.get("experiment_mode")) or defaults.get("experiment_mode") or "unknown").lower()
        if mode not in {"batch", "flow", "unknown"}:
            raise ValueError("experiment_mode must be batch, flow or unknown")
        rid = canonical_record_id(dataset, system, mode, entry)
        if is_proposal:
            rid += ":proposal"
        if rid in seen:
            raise ValueError(f"Repeated source_entry {entry} within one dataset/system/mode: clarify its identity before import")
        seen.add(rid)
        conditions = {k: _number(mapped.get(k), k) if k in NUMERIC_CONDITIONS else _text(mapped.get(k))
                      for k in [*CONDITION_FIELDS, *EXTRA_CONDITIONS]}
        outcomes = {k: _number(mapped.get(k), k) for k in OUTCOME_FIELDS}
        reasons = []
        if is_proposal:
            reasons.append("Source labels this record as a proposal; it is not measured-history evidence")
        if not any(outcomes[k] is not None for k in ["residual_sulfide_pct", "sulfoxide_pct", "sulfone_pct"]):
            reasons.append("No reported sulfide/sulfoxide/sulfone outcome is available")
        loc = f"{location['sheet']}!row:{location['source_row']}" if location['sheet'] else f"row:{location['source_row']}"
        representation = {**source, **location, "source_record_id": source["source_id"] + "#" + loc,
                          "raw_fields": raw_fields, "reported_actual_conditions": conditions,
                          "reported_outcomes": outcomes, "analytical_method": _text(mapped.get("analytical_method")),
                          "source_status": source_status, "proposed_conditions": proposed_conditions}
        records.append({"record_id": rid, "dataset_id": dataset, "source_entry": entry,
            "catalyst_system": system, "experiment_mode": mode, "reported_actual_conditions": conditions,
            "reported_outcomes": outcomes, "analytical_method": _text(mapped.get("analytical_method")),
            "source_representations": [representation], "disputed_fields": [], "interpretation_eligible": not reasons,
            "source_status": source_status, "record_kind": "proposal" if is_proposal else "reported_observation",
            "proposed_conditions": proposed_conditions, "interpretation_exclusion_reasons": reasons,
            "independent_experiment_identity": None})
    if any(r["dataset_id"].startswith("source:") for r in records):
        warnings.append("No shared dataset_id: this file remains a separate dataset. Provide the same dataset_id to reconcile overlapping exports; identical conditions are never deduplicated.")
    if any(r["catalyst_system"] == "unspecified" for r in records):
        warnings.append("Catalyst system is unspecified; these rows remain separate and are not supplied as homogeneous evidence.")
    if any(r["experiment_mode"] == "unknown" for r in records):
        warnings.append("Experiment mode is unknown; durations are not interpreted as batch or flow kinetics.")
    return {"schema_version": "flowpilotx-history-import-v1", "sources": [source], "rows": records,
            "warnings": warnings, "summary": summarize_history(records, [source], warnings)}


def _equal(a, b, field):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= (0.0051 if field == "reported_two_product_selectivity_pct" else 1e-8)
    return a == b


def merge_history(datasets):
    records, sources, warnings = {}, {}, []
    for dataset in datasets:
        warnings.extend(dataset.get("warnings", []))
        sources.update({s["sha256"]: deepcopy(s) for s in dataset["sources"]})
        for original in dataset["rows"]:
            row = deepcopy(original)
            rid = row["record_id"]
            if rid not in records:
                records[rid] = row
                continue
            target = records[rid]
            existing = {r["source_record_id"] for r in target["source_representations"]}
            for representation in row["source_representations"]:
                if representation["source_record_id"] not in existing:
                    target["source_representations"].append(representation)
                    existing.add(representation["source_record_id"])
            for group in ["reported_actual_conditions", "reported_outcomes"]:
                for field in set(target[group]) | set(row[group]):
                    if not _equal(target[group].get(field), row[group].get(field), field):
                        target["disputed_fields"].append(group + "." + field)
            if target["analytical_method"] != row["analytical_method"]:
                target["disputed_fields"].append("analytical_method")
            target["disputed_fields"] = sorted(set(target["disputed_fields"]))
            target["interpretation_exclusion_reasons"] = list(dict.fromkeys(
                target.get("interpretation_exclusion_reasons", []) + row.get("interpretation_exclusion_reasons", [])
                + (["Conflicting exports require source adjudication"] if target["disputed_fields"] else [])))
            target["interpretation_eligible"] = target["interpretation_eligible"] and row["interpretation_eligible"] and not bool(target["disputed_fields"])
    rows = list(records.values())
    warnings = list(dict.fromkeys(warnings))
    return {"rows": rows, "sources": list(sources.values()), "warnings": warnings,
            "summary": summarize_history(rows, list(sources.values()), warnings)}


def summarize_history(rows, sources, warnings=()):
    above, missing = 0, 0
    for row in rows:
        out = row["reported_outcomes"]
        components = [out.get(k) for k in ["residual_sulfide_pct", "sulfoxide_pct", "sulfone_pct", "other_products_pct", "unknown_components_pct"]]
        above += any(v is not None and v > 100 for v in out.values()) or (all(v is not None for v in components) and sum(components) > 100)
        missing += any(out.get(k) is None for k in ["residual_sulfide_pct", "sulfoxide_pct", "sulfone_pct"])
    representations = sum(len(r["source_representations"]) for r in rows)
    conflicts = sum(bool(r["disputed_fields"]) for r in rows)
    issues = list(warnings)
    if conflicts:
        issues.append("Conflicting exports remain visible with all alternatives; affected rows are excluded from interpreted evidence.")
    return {"canonical_count": len(rows), "homogeneous_count": sum(r["catalyst_system"] == "homogeneous" for r in rows),
        "heterogeneous_count": sum(r["catalyst_system"] == "heterogeneous" for r in rows),
        "unspecified_count": sum(r["catalyst_system"] == "unspecified" for r in rows),
        "source_count": len({s["sha256"] for s in sources}), "source_representation_count": representations,
        "overlap_count": representations - len(rows), "above_100_count": above, "missing_outcome_count": missing,
        "conflict_count": conflicts, "issues": issues,
        "proposal_count": sum(r.get("record_kind") == "proposal" for r in rows),
        "interpretation_eligible_count": sum(r["interpretation_eligible"] for r in rows),
        "evidence_ids": [r["record_id"] for r in rows if r["catalyst_system"] == "homogeneous" and r["interpretation_eligible"]],
        "policy": "Same dataset/system/mode/source entry across exports is one entry; repeated settings alone never establish shared identity. No clipping, normalization or kinetic transfer."}


def builtin_history(project_root):
    history = ingest_history(project_root)
    rows, sources, known = [], [], {}
    for system, group in history["systems"].items():
        for row in group["records"]:
            conditions = {**row["reported_actual_conditions"], **{k: None for k in EXTRA_CONDITIONS}}
            representations = []
            for rep in row["source_representations"]:
                representations.append({**deepcopy(rep), "reported_actual_conditions": conditions,
                    "reported_outcomes": {k: row["outcomes"].get(k) for k in OUTCOME_FIELDS}, "analytical_method": None})
                known[rep["source_sha256"]] = {"dataset_id": "khu", "catalyst_system": system, "experiment_mode": "batch"}
            rows.append({"record_id": row["record_id"], "dataset_id": "khu", "source_entry": row["source_entry"],
                "catalyst_system": system, "experiment_mode": "batch", "reported_actual_conditions": conditions,
                "reported_outcomes": {k: row["outcomes"].get(k) for k in OUTCOME_FIELDS}, "analytical_method": None,
                "source_representations": representations, "disputed_fields": list(row["disputed_source_fields"]),
                "interpretation_eligible": not row["disputed_source_fields"], "independent_experiment_identity": None})
    sources = [s for s in history["sources"] if s.get("format") in {"csv", "xlsx"}]
    warnings = [issue["reason"] for issue in history["issues"]]
    return {"id": "khu_builtin", "rows": rows, "sources": sources, "known_sources": known,
            "warnings": warnings, "summary": summarize_history(rows, sources, warnings)}
