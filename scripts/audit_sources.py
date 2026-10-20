#!/usr/bin/env python3
"""Read the supplied files without editing them; no third-party dependencies or APIs.

Writes source extracts and descriptive checks, not a cleaned training dataset.
XLSX values are stored caches; formulas are retained, not recalculated.
"""
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/khu"
OUT = ROOT / "data/derived"


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def read_csv(path):
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        text = raw.decode("cp949")
        encoding = "cp949"
    reader = csv.DictReader(text.splitlines())
    rows = [{"source_row": i, "values": row} for i, row in enumerate(reader, 2)
            if str(row.get("Entry", "")).strip()]
    return {"encoding": encoding, "header": reader.fieldnames, "rows": rows}


def read_xlsx(path):
    with ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = ["".join(t.text or "" for t in s.findall(".//{*}t"))
                       for s in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        sheets = ET.fromstring(z.read("xl/workbook.xml")).findall("{*}sheets/{*}sheet")
        assert len(sheets) == 1, "Review additional sheets before changing this importer"
        root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        matrix, formulas = [], {}
        for row in root.findall("{*}sheetData/{*}row"):
            cells = {}
            for c in row.findall("{*}c"):
                address = c.get("r")
                col = re.sub(r"\d", "", address)
                v = c.find("{*}v")
                value = v.text if v is not None else ""
                if c.get("t") == "s" and value:
                    value = strings[int(value)]
                elif c.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in c.findall(".//{*}t"))
                cells[col] = value
                f = c.find("{*}f")
                if f is not None:
                    formulas[address] = {"formula": f.text, "cached_value": value}
            matrix.append((int(row.get("r")), cells))
        header_row, header = next((r, c) for r, c in matrix if c.get("A") == "Entry")
        rows = [{"source_row": r, "values": {label: c.get(col, "") for col, label in header.items()}}
                for r, c in matrix if r > header_row and number(c.get("A")) is not None]
    return {"sheet": sheets[0].get("name"), "header_row": header_row,
            "header": list(header.values()), "rows": rows, "formulas": formulas,
            "formula_policy": "Cached values read without recalculation; independently checked derived columns."}


def equivalent(a, b, key):
    x, y = number(a), number(b)
    if x is not None and y is not None:
        return abs(x - y) <= (0.0051 if key.startswith("selectivity") else 1e-8)
    return str(a).strip() == str(b).strip()


def checks(extracted):
    rows = extracted["rows"]
    flags, condition_groups = [], defaultdict(list)
    header = extracted["header"]
    for item in rows:
        r, entry = item["values"], item["values"]["Entry"]
        problems = []
        vals = [number(r.get(k)) for k in header[11:16]]
        if any(v is not None and not 0 <= v <= 100 for v in vals):
            problems.append("component_percentage_outside_0_100")
        if all(v is not None for v in vals):
            total = sum(vals)
            reported = number(r.get("mass balance"))
            if reported is not None and abs(total - reported) > 1e-8:
                problems.append("mass_balance_arithmetic_mismatch")
            if total > 100:
                problems.append("component_sum_above_100")
            if total < 90 or total > 110:
                problems.append("component_sum_outside_90_110_review_only")
            conv = 100 - vals[0]
            if number(r.get(header[17])) is not None and abs(conv - number(r[header[17]])) > 1e-8:
                problems.append("conversion_arithmetic_mismatch")
            if vals[1] + vals[2] > 0:
                selectivity = 100 * vals[1] / (vals[1] + vals[2])
                if number(r.get(header[18])) is not None and abs(selectivity - number(r[header[18]])) > .0051:
                    problems.append("selectivity_arithmetic_mismatch")
            if vals[1] > conv:
                problems.append("sulfoxide_percentage_exceeds_apparent_conversion")
        if problems:
            flags.append({"entry": entry, "source_row": item["source_row"], "flags": problems})
        condition_groups[tuple(r.get(k, "") for k in header[1:11])].append(entry)
    return {"record_count": len(rows), "flag_counts": dict(Counter(p for f in flags for p in f["flags"])),
            "flags": flags, "repeated_condition_groups": [v for v in condition_groups.values() if len(v) > 1],
            "note": "Repeated conditions are not confirmed independent repeats. 90–110 is an audit flag, not an exclusion rule.",
            "field_values": {k: dict(Counter(r["values"].get(k, "") for r in rows)) for k in header[1:11]}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"files": {}, "pairs": {}, "scope": "Descriptive preparation only; no model fitting or condition generation."}
    for kind in ["homogeneous", "heterogeneous"]:
        paths = {ext: next(RAW.glob(f"*raw data_{kind}.{ext}")) for ext in ["csv", "xlsx"]}
        extracts = {}
        for ext, path in paths.items():
            data = read_csv(path) if ext == "csv" else read_xlsx(path)
            data.update(source=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            extracts[ext] = data
            (OUT / f"{kind}_{ext}_extract.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            summary["files"][f"{kind}_{ext}"] = checks(data)
        keyed = {ext: {str(int(float(r["values"]["Entry"]))): r for r in data["rows"]} for ext, data in extracts.items()}
        a, b = keyed["csv"], keyed["xlsx"]
        differences = []
        for entry in sorted(a.keys() & b.keys(), key=int):
            for key in extracts["csv"]["header"]:
                av, bv = a[entry]["values"].get(key), b[entry]["values"].get(key)
                if not equivalent(av, bv, key):
                    differences.append({"entry": entry, "field": key, "csv": av, "xlsx": bv,
                                        "csv_row": a[entry]["source_row"], "xlsx_row": b[entry]["source_row"]})
        summary["pairs"][kind] = {"csv_only_entries": sorted(a.keys()-b.keys(), key=int),
            "xlsx_only_entries": sorted(b.keys()-a.keys(), key=int), "differences": differences,
            "comparison_tolerance": "0.0051 percentage points for rounded selectivity, 1e-8 for other numeric cells"}
    ppt = next(RAW.glob("*.pptx"))
    with ZipFile(ppt) as z:
        paths = sorted([n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)],
                       key=lambda n: int(re.search(r"slide(\d+)\.xml", n)[1]))
        ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        slides = [{"slide": i, "text": "\n".join(" ".join(t.text or "" for t in p.findall(".//a:t", ns))
                    for p in ET.fromstring(z.read(n)).findall(".//a:p", ns))} for i, n in enumerate(paths, 1)]
    (OUT / "slide_text.json").write_text(json.dumps(slides, indent=2, ensure_ascii=False) + "\n")
    summary["slide_count"] = len(slides)
    summary["slide_extraction_note"] = "Tables and structures are images; also inspect review/slides/slide-2.png through slide-7.png."
    (OUT / "source_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"records": {k: v["record_count"] for k,v in summary["files"].items()},
                      "pairs": summary["pairs"]}, indent=2))


if __name__ == "__main__":
    main()
