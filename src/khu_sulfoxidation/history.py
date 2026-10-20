"""Read-only, standard-library ingestion of the supplied KHU evidence.

Rows are reconciled by source-system entry, never by chemical conditions. Each
source representation, including rounded CSV values and Excel cached formulas,
is retained. The logical IDs identify table entries, not proven independent runs.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from copy import deepcopy
from pathlib import Path
import posixpath
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from .observations import AnalyticalOutcome, reported_number

SCHEMA_VERSION = "khu-history-v1"
REVIEWED_CONTEXT_SHA256 = "25d908cf026870bcc58121450b60ca55ab22dc98b4f6f0533ea9abb0fe5854c7"
PRIVATE_CONTEXT_RELATIVE_PATH = "data/private/khu_source_context_v1.json"
CONDITION_FIELDS = {
    "solvent_1": "solvent 1", "solvent_2": "solvent 2",
    "solvent_1_percentage": "solvent ratio (% of solvent 1) ",
    "concentration_mol_L": "concentration (M)",
    "photocatalyst": "type of photocatalyst", "catalyst_loading_mol_pct": "catalyst loading (mol%)",
    "batch_duration_h": "time (h)", "temperature_C": "temperature (℃)",
    "oxidant_reported": "oxidant ", "light_source_reported": "light source",
}
NUMERIC_CONDITIONS = {"solvent_1_percentage", "concentration_mol_L", "catalyst_loading_mol_pct",
                      "batch_duration_h", "temperature_C"}
OUTCOME_FIELDS = {
    "residual_sulfide_pct": "yield 1 (%)", "sulfoxide_pct": "yield 2 (%)",
    "sulfone_pct": "yield 3 (%)", "other_products_pct": "side (%)",
    "unknown_components_pct": "unknown (%)", "reported_component_sum_pct": "mass balance",
    "reported_conversion_pct": "conversion (100 - yield 1)",
    "reported_two_product_selectivity_pct": "selectivity [{yield 2 / (yield 2 + yield 3)} x 100]",
}


def _load_private_context(root: Path | str | None = None) -> dict:
    """Load reviewed local study facts; never replace missing inputs with examples.

    Source code contains no slide measurements or private report correspondence.
    Set FLOWPILOTX_PRIVATE_STUDY_CONTEXT to the reviewed JSON when using an
    installed package outside the original study workspace.
    """
    configured = os.environ.get("FLOWPILOTX_PRIVATE_STUDY_CONTEXT")
    project_root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    path = Path(configured).expanduser() if configured else project_root / PRIVATE_CONTEXT_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Private KHU study context is required: {path}. Restore the reviewed local input "
            "or set FLOWPILOTX_PRIVATE_STUDY_CONTEXT; no private data are supplied with public source."
        )
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REVIEWED_CONTEXT_SHA256:
        raise ValueError("Private study context changed; source review and an explicit version update are required")
    context = json.loads(raw)
    if context.get("schema_version") != "khu-private-source-context-v1":
        raise ValueError("Unsupported private study context schema")
    return context


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_id(system: str, entry: str | int) -> str:
    return f"khu:{system}:raw-entry:{int(entry):04d}"


def _source(path: Path, root: Path) -> dict:
    digest = sha256_file(path)
    return {"source_id": f"sha256:{digest}", "path": str(path.relative_to(root)),
            "sha256": digest, "size_bytes": path.stat().st_size}


def _entry(value: str) -> str:
    parsed = reported_number(value)
    if parsed is None or parsed < 1 or not parsed.is_integer():
        raise ValueError(f"Invalid source entry: {value!r}")
    return str(int(parsed))


def read_csv_source(path: Path, root: Path) -> dict:
    raw = path.read_bytes()
    try:
        text, encoding = raw.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        text, encoding = raw.decode("cp949"), "cp949"
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("CSV has missing or duplicate headers; explicit source review required")
    rows = []
    for ordinal, values in enumerate(reader, 2):
        if str(values.get("Entry", "")).strip():
            rows.append({"entry": _entry(values["Entry"]), "source_row": ordinal,
                         "physical_line_end": reader.line_num, "values": values,
                         "cells": None, "formulas": {}})
    return {**_source(path, root), "format": "csv", "encoding": encoding,
            "sheet": None, "header": reader.fieldnames, "rows": rows}


def read_xlsx_source(path: Path, root: Path) -> dict:
    """Retain serialized XML cell values and cached formulas without recalculation."""
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(t.text or "" for t in s.findall(".//{*}t"))
                       for s in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        sheets = ET.fromstring(archive.read("xl/workbook.xml")).findall("{*}sheets/{*}sheet")
        if len(sheets) != 1:
            raise ValueError("Additional sheets require an explicit source mapping")
        relationships = {r.get("Id"): r.get("Target") for r in
                         ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        rid = sheets[0].get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        target = relationships[rid]
        member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        matrix = []
        for row in ET.fromstring(archive.read(member)).findall("{*}sheetData/{*}row"):
            cells = {}
            for cell in row.findall("{*}c"):
                address = cell.get("r")
                column = re.sub(r"\d", "", address)
                v, f = cell.find("{*}v"), cell.find("{*}f")
                raw_value = None if v is None else v.text
                value = raw_value or ""
                if cell.get("t") == "s" and value:
                    value = strings[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//{*}t"))
                cells[column] = {"address": address, "type": cell.get("t", "n"),
                                 "serialized_value": raw_value, "value": value,
                                 "formula": None if f is None else f.text,
                                 "formula_attributes": None if f is None else dict(f.attrib),
                                 "style_index": cell.get("s")}
            matrix.append((int(row.get("r")), cells))
        header_row, header_cells = next((r, c) for r, c in matrix if c.get("A", {}).get("value") == "Entry")
        header = {col: cell["value"] for col, cell in header_cells.items() if cell["value"]}
        if len(header.values()) != len(set(header.values())):
            raise ValueError("Workbook has duplicate column labels; explicit source review required")
        rows = []
        for row_num, cells in matrix:
            if row_num <= header_row or reported_number(cells.get("A", {}).get("value")) is None:
                continue
            values = {label: cells.get(col, {}).get("value", "") for col, label in header.items()}
            formulas = {c["address"]: {"formula": c["formula"], "cached_value": c["value"],
                                       "attributes": c["formula_attributes"]}
                        for c in cells.values() if c["formula_attributes"] is not None}
            rows.append({"entry": _entry(values["Entry"]), "source_row": row_num,
                         "values": values, "cells": cells, "formulas": formulas})
    return {**_source(path, root), "format": "xlsx", "sheet": sheets[0].get("name"),
            "header_row": header_row, "header": list(header.values()), "rows": rows,
            "formula_policy": "Read cached values and retained formulas; no recalculation"}


def _equivalent(a, b, key):
    x, y = reported_number(a), reported_number(b)
    if x is not None and y is not None:
        return abs(x - y) <= (0.0051 if key.startswith("selectivity") else 1e-8)
    return str(a).strip() == str(b).strip()


def _representation(source: dict, row: dict) -> dict:
    location = f"{source['sheet']}!row:{row['source_row']}" if source["sheet"] else f"row:{row['source_row']}"
    return {"source_record_id": f"{source['source_id']}#{location}",
            "source_id": source["source_id"], "source_path": source["path"],
            "source_sha256": source["sha256"], "format": source["format"],
            "sheet": source["sheet"], **row}


def reconcile_system(system: str, csv_source: dict, xlsx_source: dict, slide_source_id: str,
                     *, private_context_root: Path | str | None = None) -> dict:
    """Keep all entries and alternative exports; differing fields remain disputed."""
    context = _load_private_context(private_context_root)
    keyed = {}
    for source in (csv_source, xlsx_source):
        ids = [row["entry"] for row in source["rows"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate Entry inside source {source['path']}; no silent overwrite permitted")
        keyed[source["format"]] = {row["entry"]: row for row in source["rows"]}
    records, differences = [], []
    common = keyed["csv"].keys() & keyed["xlsx"].keys()
    for entry in sorted(keyed["csv"].keys() | keyed["xlsx"].keys(), key=int):
        representations = [_representation(source, keyed[source["format"]][entry])
                           for source in (xlsx_source, csv_source) if entry in keyed[source["format"]]]
        selected = representations[0]
        values, conflicts = selected["values"], []
        if entry in common:
            for key in sorted(set(keyed["csv"][entry]["values"]) | set(keyed["xlsx"][entry]["values"])):
                a, b = keyed["csv"][entry]["values"].get(key), keyed["xlsx"][entry]["values"].get(key)
                if not _equivalent(a, b, key):
                    conflicts.append(key)
                    differences.append({"record_id": record_id(system, entry), "field": key, "csv": a, "xlsx": b})
        conditions = {name: (reported_number(values.get(key)) if name in NUMERIC_CONDITIONS
                             else (values.get(key) if values.get(key) not in {"", "-", None} else None))
                      for name, key in CONDITION_FIELDS.items()}
        conditions["solvent_ratio_basis"] = None
        outcomes = AnalyticalOutcome(**{name: reported_number(values.get(key)) for name, key in OUTCOME_FIELDS.items()},
                                     raw_reported={key: values.get(key) for key in OUTCOME_FIELDS.values()})
        membership = "common_csv_xlsx" if entry in common else "xlsx_only" if selected["format"] == "xlsx" else "csv_only"
        records.append({"record_id": record_id(system, entry), "source_entry": entry,
            "catalyst_system": system, "experiment_mode": "batch",
            "version_membership": membership, "independent_experiment_identity": None,
            "proposed_conditions": None, "reported_actual_conditions": conditions,
            "condition_status": "reported batch settings; actual instrument measurements unavailable",
            "outcomes": outcomes.to_dict(), "source_representations": representations,
            "interpretation_source_record_id": selected["source_record_id"],
            "interpretation_policy": "Workbook caches for display, preserving both exports; disputed fields excluded from evidence summaries",
            "disputed_source_fields": conflicts,
            "identity_mapping": {"substrate_identity": None, **deepcopy(context["row_identity"]),
                "source_id": slide_source_id, "slide": 2 if system == "homogeneous" else 3},
            "analytical_method": {"row_reported": None,
                "candidate_method": context["row_analytical_method"]["candidate_method"],
                "candidate_internal_standard": context["row_analytical_method"]["candidate_internal_standard"],
                "source_id": slide_source_id, "slide": 2 if system == "homogeneous" else 3,
                "scope_status": context["row_analytical_method"]["scope_status"], "uncertainty": None},
        })
    return {"records": records, "record_count": len(records),
            "common_subset_ids": [record_id(system, e) for e in sorted(common, key=int)],
            "reconciliation": {"csv_count": len(keyed["csv"]), "xlsx_count": len(keyed["xlsx"]),
                "csv_only_entries": sorted(keyed["csv"].keys() - keyed["xlsx"].keys(), key=int),
                "xlsx_only_entries": sorted(keyed["xlsx"].keys() - keyed["csv"].keys(), key=int),
                "differences": differences, "numeric_tolerance": 1e-8, "selectivity_tolerance_pct_points": 0.0051,
                "authoritative_version_confirmed": False,
                "deduplication_basis": "Same system and source Entry across exports, not identical conditions"}}


def _slide_evidence(ppt_source: dict, root: Path) -> dict:
    """Read the reviewed local transcription; keep image and raw evidence separate."""
    context = _load_private_context(root)
    if ppt_source["sha256"] != context["reviewed_slide_sha256"]:
        raise ValueError("Slide source changed; manual transcription and identity mapping require renewed visual review")
    expected = context["reviewed_render_sha256"]
    images = [_source(root / f"review/slides/slide-{i}.png", root) for i in (2, 3)]
    if any(source["sha256"] != expected[str(i)] for i, source in zip((2, 3), images)):
        raise ValueError("Reviewed slide render changed; manual transcription requires renewed visual review")
    evidence = deepcopy(context["slide_evidence"])
    if any(record["source_id"] != ppt_source["source_id"] for record in evidence["records"]):
        raise ValueError("Private slide transcription does not match the supplied source identity")
    evidence["render_sources"] = images
    return evidence


def ingest_history(root: Path | str) -> dict:
    root = Path(root).resolve()
    raw_dir = root / "data/raw/khu"
    ppt_paths = sorted(raw_dir.glob("*.pptx"))
    if len(ppt_paths) != 1:
        raise ValueError("Exactly one identified KHU slide source is required")
    ppt_source = _source(ppt_paths[0], root)
    sources, systems = [ppt_source], {}
    for system in ("homogeneous", "heterogeneous"):
        pair = {}
        for extension, reader in [("csv", read_csv_source), ("xlsx", read_xlsx_source)]:
            paths = sorted(raw_dir.glob(f"*raw data_{system}.{extension}"))
            if len(paths) != 1:
                raise ValueError(f"Expected exactly one {system} {extension} source")
            pair[extension] = reader(paths[0], root)
            sources.append({k: v for k, v in pair[extension].items() if k != "rows"})
        systems[system] = reconcile_system(system, pair["csv"], pair["xlsx"], ppt_source["source_id"], private_context_root=root)
    issues = [
        {"issue_id": "source-version", "status": "unresolved", "reason": "Homogeneous Excel-only entries 112–116 retained; KHU authoritative version unconfirmed"},
        {"issue_id": "identity-scope", "status": "unresolved", "reason": "Raw tables lack substrate identities; slide structure/component/analytical-method mapping across all raw rows requires confirmation"},
        {"issue_id": "repeat-relationships", "status": "unresolved", "reason": "Identical recorded conditions do not establish independent repeats or chronology"},
        {"issue_id": "analytical-values", "status": "unresolved", "reason": "Reported values above 100 and component-sum anomalies retained for analytical review without normalization or outcome-based exclusion"},
        {"issue_id": "slide-raw-discrepancies", "status": "unresolved", "reason": "Slide image-table differences remain separate evidence, with candidate links only"},
        {"issue_id": "solvent-basis", "status": "unresolved", "reason": "Percentage basis and buffer formulations unavailable"},
        {"issue_id": "transfer-metadata", "status": "unresolved", "reason": "Photon flux, gas delivery, geometry, collection handling and analytical uncertainty lack raw-row metadata; batch times are not flow kinetics"},
    ]
    slides = _slide_evidence(ppt_source, root)
    for system in systems.values():
        if system["reconciliation"]["differences"]:
            issues.append({"issue_id": "export-value-conflicts", "status": "unresolved",
                           "reason": "Disputed source fields are retained and excluded from summary interpretation"})
    source_to_record = [{"source_record_id": representation["source_record_id"], "record_id": record["record_id"]}
                        for system in systems.values() for record in system["records"]
                        for representation in record["source_representations"]]
    return {"schema_version": SCHEMA_VERSION, "sources": sources, "systems": systems,
            "source_to_record": source_to_record, "slide_evidence": slides, "issues": issues,
            "curation_policy": {"exclude_by_outcome": False, "normalize_or_clip": False,
                "source_selection_reason": "Workbook has greater coverage; all alternative source representations retained",
                "merge_homogeneous_heterogeneous": False, "fit_batch_as_flow": False,
                "amendments": [], "independence_and_chronology": "Unconfirmed; Entry is a source identifier only"}}
