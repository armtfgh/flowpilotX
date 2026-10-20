"""Six standard FlowPilot figures and concise tables from frozen KHU settings.

This presentation adapter makes no design changes, model calls or laboratory
assignments. Rendered figures must already exist and match the six core IDs.
"""
from __future__ import annotations

import csv
import html
import importlib.util
import json
from pathlib import Path

from .provenance import file_hash

STATUS = "Conditional proposal - not released for laboratory execution"
CONNECTIVITY = "The six conditions share one proposed apparatus topology; only the liquid and air delivery settings differ."
TIME_NOTE = "V/Q_L is calculated liquid-only space time, not measured liquid contact time or elapsed run duration."
GAS_NOTE = "Reference air flow assumes 273.15 K, 1.01325 bar absolute and O2 mole fraction 0.2095; actual gas calibration/reference remain unconfirmed."
UNKNOWN_NOTE = "BPR/pressure, actual light setting, installed assignments, feed mode and collection procedure remain unresolved."
CORE_COLUMNS = ["condition_id", "candidate_id", "liquid_flow_mL_min", "reference_air_flow_mL_min", "nominal_oxygen_equivalents", "calculated_supplied_oxygen_equivalents", "liquid_only_space_time_min", "role"]
CORE_HEADERS = ["Run", "Candidate", "Liquid\nmL/min", "Reference air\nmL/min", "Nominal O2\nequivalents", "Calculated O2\nequivalents", "Liquid-only\nV/Q_L min", "Role"]


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_path(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe rendered artifact: {relative}")
    return path


def _load(source_dir, output_dir):
    source_path = source_dir / "authoritative_conditions.json"
    authoritative = _read(source_path)
    index = _read(output_dir / "render_index.json")
    all_rows = authoritative["conditions"]
    core = [r for r in all_rows if r["comparison_arm"] == "history_informed" and r["repeat_of_condition_id"] is None]
    expected = [f"HI{i:02d}" for i in range(1, 7)]
    if [r["condition_id"] for r in core] != expected:
        raise ValueError("Exactly the frozen six informed core rows HI01-HI06 are required")
    if len({r["candidate_id"] for r in core}) != 6:
        raise ValueError("Core candidates must be distinct")
    cases = index["cases"]
    if len(cases) != 6 or {c["condition_id"] for c in cases} != set(expected):
        raise ValueError("Render index must contain exactly one figure for each core condition")
    by_condition = {c["condition_id"]: c for c in cases}
    ordered = []
    for row in core:
        case = by_condition[row["condition_id"]]
        if case["candidate_id"] != row["candidate_id"]:
            raise ValueError("Figure condition/candidate pair differs from authoritative data")
        for kind in ("png", "svg", "topology_json"):
            _safe_path(output_dir, case[kind])
        if row["actual_conditions"] is not None or row["actual_outcomes"] is not None:
            raise ValueError("This conditional report requires unknown actual conditions/outcomes")
        ordered.append(case)
    scenario = authoritative["scenario"]
    recipe = scenario["shared_recipe"]
    # The current labels and gas footnote intentionally refer to the frozen v2 scenario.
    assumptions = {a["id"]: a for a in scenario["assumptions"]}
    gas, air = assumptions["A_GAS_REFERENCE"], assumptions["A_AIR_COMPOSITION"]
    if (recipe["substrate"], recipe["catalyst"], recipe["solvent"], recipe["substrate_concentration_M"],
        recipe["catalyst_loading_mol_percent"], gas["temperature_K"], gas["pressure_bar_abs"], air["oxygen_mole_fraction"]) != ("thioanisole", "acridine", "EtOH", .25, 3., 273.15, 1.01325, .2095):
        raise ValueError("Changed chemistry/gas assumptions require an explicit new presentation version")
    if any(r["scenario_id"] != scenario["scenario_id"] for r in all_rows):
        raise ValueError("Rows do not share the frozen apparatus scenario")
    return authoritative, index, core, ordered


def _roles(authoritative):
    roles = {r["condition_id"]: [] for r in authoritative["conditions"]}
    for arm, items in authoritative["comparison"]["arms"].items():
        label = "informed" if arm == "history_informed" else "omitted"
        for item in items:
            roles[item["condition_id"]].append(f"{label} nomination {item['rank']}")
    for row in authoritative["conditions"]:
        if row["repeat_of_condition_id"]:
            roles[row["repeat_of_condition_id"]].append("independent-repeat reference")
    return {key: "; ".join(value) if value else "core screen" for key, value in roles.items()}


def _summary_row(row, role):
    return dict(zip(CORE_COLUMNS, [row["condition_id"], row["candidate_id"], row["proposed_setpoints"]["liquid_flow_mL_min"],
        row["proposed_setpoints"]["gas_reference_flow_mL_min"], row["nominal_oxygen_equivalents"],
        row["calculated"]["oxygen_equivalents_supplied"], row["calculated"]["liquid_only_space_time_min"], role]))


def _shared(authoritative):
    scenario = authoritative["scenario"]
    recipe, hardware = scenario["shared_recipe"], scenario["hardware_choices"]
    rows = [
        ("Substrate / solvent", f"{recipe['substrate']}, {recipe['substrate_concentration_M']:g} M in {recipe['solvent']}", "proposed feed"),
        ("Catalyst", f"{recipe['catalyst']}, {recipe['catalyst_loading_mol_percent']:g} mol%; {scenario['feed_calculations']['catalyst_concentration_mM']:g} mM", "proposed; concentration calculated"),
        ("Liquid delivery", hardware["pump"]["source_reported_inventory_record"]["name"], "proposed inventory option; assignment unconfirmed"),
        ("Gas path", "Air / regulator / FF-C00 MFC / CV-3301 check valve / P-713 T-mixer", "proposed; gas calibration/reference unconfirmed"),
        ("Reactor / illumination", f"{scenario['geometry']['volume_mL']:g} mL PFA coil, {scenario['geometry']['ID_mm']:g} mm ID; UV-150 / 450 nm", "source-reported geometry; proposed apparatus"),
        ("Temperature", f"{scenario['proposed_temperature_C']:g} C", "proposed; not measured"),
        ("Pressure / BPR", "Unassigned", "unknown; 2/4 bar calculations are illustrations only"),
        ("Light power / feed mode / collection", "Unconfirmed", "unknown; no run duration or total consumption inferred"),
    ]
    return [{"setting": name, "value": value, "basis_or_status": status} for name, value, status in rows]


def _tables(authoritative, core):
    roles = _roles(authoritative)
    summaries = [_summary_row(row, roles[row["condition_id"]]) for row in core]
    core_ids = {r["condition_id"] for r in core}
    supplementary = []
    for row in authoritative["conditions"]:
        if row["condition_id"] not in core_ids:
            purpose = f"Independent preparation and operation repeating {row['repeat_of_condition_id']}" if row["repeat_of_condition_id"] else "Additional history-omitted nomination"
            supplementary.append({**_summary_row(row, purpose), "repeat_of_condition_id": row["repeat_of_condition_id"]})
    comparison = [{"arm": arm, "frozen_rank": item["rank"], "candidate_id": item["candidate_id"], "measurement_condition_id": item["condition_id"]}
                  for arm, items in authoritative["comparison"]["arms"].items() for item in items]
    return summaries, _shared(authoritative), supplementary, comparison


def _write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _display(value):
    if value is None:
        return "Unconfirmed"
    return f"{value:.4f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def _html_table(headers, rows):
    return '<table><thead><tr>' + ''.join('<th>' + html.escape(h).replace('\n', '<br>') + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + html.escape(_display(c)) + '</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table>'


def _supplement_note(authoritative):
    comparison = authoritative["comparison"]
    overlaps = comparison["omitted_nominations_already_in_informed_core"]
    named = ", ".join(overlaps) if overlaps else "none"
    return (f"{len(authoritative['conditions'])} proposed runs, including one independent repeat; "
            f"{len({r['candidate_id'] for r in authoritative['conditions']})} unique conditions. "
            f"Omitted nominations already present in the core: {named}. Shared conditions use one primary measurement. "
            "The independent repeat is reported separately, not used as an extra chance to select the better result. Table order is not execution order.")


def _html(authoritative, core, cases, tables):
    summaries, shared, supplementary, comparison = tables
    body = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><title>KHU FlowPilot topology proposals</title>',
        '<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1250px;margin:32px auto;padding:0 24px;color:#183741}h1{font-size:30px}h2{margin-top:32px;color:#17645f}table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}th,td{border:1px solid #c6d5d8;padding:8px;text-align:left;vertical-align:top}th{background:#eaf3f1}.status{background:#fff3df;padding:14px;border-left:5px solid #b16a28}.figure{width:100%;height:auto;border:1px solid #d8e1e2}.small{font-size:13px;color:#48616a}a{color:#16645f}</style></head><body>',
        '<h1>KHU: six FlowPilot flow proposals</h1>', '<p class="status"><strong>' + html.escape(STATUS) + '</strong></p>',
        '<p>' + html.escape(CONNECTIVITY) + '</p>', '<h2>Six core conditions</h2>',
        _html_table(CORE_HEADERS, [[r[k] for k in CORE_COLUMNS] for r in summaries]),
        '<p>' + html.escape(TIME_NOTE + " " + GAS_NOTE) + '</p>',
        '<h2>Shared chemistry and apparatus</h2>', _html_table(["Setting", "Value", "Basis / status"], [list(r.values()) for r in shared]),
        '<h2>Independent repeat and additional omitted-history run</h2>',
        _html_table(CORE_HEADERS, [[r[k] for k in CORE_COLUMNS] for r in supplementary]),
        '<p>' + html.escape(_supplement_note(authoritative)) + '</p>',
        '<h2>Frozen nomination comparison</h2>', _html_table(["Arm", "Rank", "Candidate", "Primary measurement"], [list(r.values()) for r in comparison]),
        '<p>Compare the two frozen nominees per arm, not best-of-six against two. Both arms share the recipe and physical pool; only the informed arm received optimization history. No superiority or flow yield has been established.</p>']
    for row, case, summary in zip(core, cases, summaries):
        body += ['<h2>' + html.escape(case["title"]) + '</h2>',
            _html_table(CORE_HEADERS, [[summary[k] for k in CORE_COLUMNS]]),
            '<img class="figure" src="' + html.escape(case["svg"], quote=True) + '" alt="' + html.escape(case["title"], quote=True) + '">',
            '<p>' + html.escape(row["review"]["rationale"]) + '</p>',
            '<p class="small">Frozen Chief citations: ' + html.escape('; '.join(row["review"]["citations"])) + '</p>',
            '<p class="small"><a href="' + html.escape(case["png"], quote=True) + '">PNG</a> | <a href="' + html.escape(case["svg"], quote=True) + '">SVG</a> | <a href="' + html.escape(case["topology_json"], quote=True) + '">Topology JSON</a></p>']
    body += ['<h2>Remaining laboratory confirmations</h2><ul>' + ''.join('<li>' + html.escape(item) + '</li>' for item in authoritative["scenario"]["unresolved_release_items"]) + '</ul>',
        '<p>' + html.escape(UNKNOWN_NOTE) + ' Track residual sulfide, sulfoxide and sulfone separately. No actual analytical outcomes are available.</p>',
        '<p class="small">CSV tables retain authoritative numeric precision; displayed tables are rounded. Source conditions and all model nominations remain unchanged.</p></body></html>']
    return '\n'.join(body)


def _pdf(path, output_dir, authoritative, core, cases, tables):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import Paragraph, Table, TableStyle

    width, height = landscape(A4)
    margin, usable = 28, width - 56
    font = "Helvetica"
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"):
        if Path(candidate).exists():
            pdfmetrics.registerFont(TTFont("KHUTopologySans", candidate))
            font = "KHUTopologySans"
            break
    styles = {"normal": ParagraphStyle("TopologyBody", fontName=font, fontSize=9, leading=12, textColor=colors.HexColor("#183741")),
              "small": ParagraphStyle("TopologySmall", fontName=font, fontSize=7.9, leading=10.2, textColor=colors.HexColor("#183741"))}
    with path.open("xb") as stream:
        canvas = Canvas(stream, pagesize=(width, height), pageCompression=1)
        canvas.setTitle("KHU - six FlowPilot topology proposals")
        canvas.setAuthor("KHU study-specific FlowPilot adapter")

        def paragraph(text, top, size="normal", x=margin, area_width=usable):
            p = Paragraph(html.escape(str(text)).replace('\n', '<br/>'), styles[size])
            _, h = p.wrap(area_width, height)
            p.drawOn(canvas, x, top - h)
            return top - h

        def heading(text, top, size=13):
            canvas.setFillColor(colors.HexColor("#17645f"))
            canvas.setFont(font, size)
            canvas.drawString(margin, top - size, text)
            return top - size - 9

        def table(headers, data, top, widths):
            cells = [[Paragraph(html.escape(_display(value)).replace('\n', '<br/>'), styles["small"]) for value in row] for row in [headers, *data]]
            t = Table(cells, colWidths=widths)
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7f1f0")),
                ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#bfd0d3")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
            _, h = t.wrap(usable, height)
            if top - h < 35:
                raise ValueError("Topology report table exceeds its page; review layout before export")
            t.drawOn(canvas, margin, top - h)
            return top - h

        def footer(page):
            canvas.setStrokeColor(colors.HexColor("#c9d8db"))
            canvas.line(margin, 24, width - margin, 24)
            canvas.setFont(font, 7)
            canvas.setFillColor(colors.HexColor("#52686e"))
            canvas.drawString(margin, 13, "KHU selective sulfoxidation | Conditional, pre-outcome proposals | Actual operating conditions pending")
            canvas.drawRightString(width - margin, 13, f"{page} / 8")

        summaries, shared, supplementary, comparison = tables
        widths = [47, 56, 66, 82, 82, 89, 80, usable - 502]
        top = heading("KHU: six FlowPilot flow proposals", height - margin, 20)
        top = paragraph(STATUS + ". " + CONNECTIVITY, top) - 10
        top = heading("Core condition summary", top)
        top = table(CORE_HEADERS, [[r[k] for k in CORE_COLUMNS] for r in summaries], top, widths) - 8
        top = paragraph(TIME_NOTE + " " + GAS_NOTE, top, "small") - 10
        top = heading("Shared chemistry and apparatus", top)
        top = table(["Setting", "Value", "Basis / status"], [list(r.values()) for r in shared], top, [154, 320, usable - 474])
        if top < 35:
            raise ValueError("Shared-settings summary does not fit its page")
        footer(1)
        canvas.showPage()

        top = heading("Repeat, comparison and laboratory confirmations", height - margin, 18)
        top = heading("Supplementary runs", top)
        top = table(CORE_HEADERS, [[r[k] for k in CORE_COLUMNS] for r in supplementary], top, widths) - 9
        top = paragraph(_supplement_note(authoritative), top) - 10
        top = heading("Two frozen nominations per arm", top)
        top = table(["Arm", "Frozen rank", "Candidate", "Primary measurement"], [list(r.values()) for r in comparison], top, [220, 100, 145, usable - 465]) - 8
        top = paragraph("Compare the two frozen nominees per arm, not best-of-six against two. Both arms share the recipe and physical pool; only the informed arm received optimization history. No flow outcomes or experimental validation are reported.", top) - 9
        top = heading("Confirm before laboratory release", top)
        for item in authoritative["scenario"]["unresolved_release_items"]:
            top = paragraph("- " + item, top, "small") - 4
        top = paragraph("Changing oxygen delivery also changes gas flow and hydrodynamics. Supplied oxygen is not measured uptake. Record residual sulfide, sulfoxide and sulfone separately; no yield predictions are supplied.", top - 3, "small")
        if top < 35:
            raise ValueError("Comparison/confirmation summary does not fit its page")
        footer(2)
        canvas.showPage()

        for page, (row, case, summary) in enumerate(zip(core, cases, summaries), 3):
            top = heading(case["title"], height - margin, 18)
            top = table(CORE_HEADERS, [[summary[k] for k in CORE_COLUMNS]], top, widths)
            image = ImageReader(str(_safe_path(output_dir, case["png"])))
            iw, ih = image.getSize()
            image_bottom, image_top = 136, top - 9
            scale = min(usable / iw, (image_top - image_bottom) / ih)
            dw, dh = iw * scale, ih * scale
            canvas.drawImage(image, margin + (usable - dw) / 2, image_bottom + (image_top - image_bottom - dh) / 2,
                             width=dw, height=dh, preserveAspectRatio=True, mask="auto")
            paragraph(row["review"]["rationale"], 121)
            paragraph("Frozen Chief citations: " + "; ".join(row["review"]["citations"]), 83, "small")
            paragraph(UNKNOWN_NOTE + " " + TIME_NOTE, 61, "small")
            footer(page)
            canvas.showPage()
        canvas.save()


def export_topology_report(source_dir, output_dir, pdf=True):
    """Add exclusive report/table artifacts beside an existing render_index.json.

    The caller prepares the six rendered PNG/SVG/JSON figures first. No files in
    the original campaign/package are modified. Return hashes for the caller's
    combined package manifest. A new destination is needed for any revision.
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    authoritative, index, core, cases = _load(source_dir, output_dir)
    names = ["summary.html", "core_conditions_summary.csv", "shared_settings.csv", "supplementary_runs.csv", "nomination_comparison.csv"] + (["flow_proposals.pdf"] if pdf else [])
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("Topology reports are immutable; choose a fresh report destination")
    if pdf and importlib.util.find_spec("reportlab") is None:
        raise RuntimeError("PDF generation requires reportlab")
    guarded = [source_dir / "authoritative_conditions.json", output_dir / "render_index.json"]
    guarded += [_safe_path(output_dir, case[k]) for case in cases for k in ("png", "svg", "topology_json")]
    hashes_before = {str(path): file_hash(path) for path in guarded}
    tables = _tables(authoritative, core)
    for name, rows in zip(names[1:5], tables):
        _write_csv(output_dir / name, rows)
    with (output_dir / "summary.html").open("x", encoding="utf-8") as stream:
        stream.write(_html(authoritative, core, cases, tables))
    if pdf:
        _pdf(output_dir / "flow_proposals.pdf", output_dir, authoritative, core, cases, tables)
    if hashes_before != {str(path): file_hash(path) for path in guarded}:
        raise ValueError("Source or rendered figures changed during export; preserve and review this incomplete version")
    return {"status": STATUS, "core_condition_count": 6, "figure_count": 6,
        "planned_run_count_including_supplementary": len(authoritative["conditions"]),
        "pdf_pages_expected": 8 if pdf else None, "output_dir": str(output_dir.resolve()),
        "source_authoritative_sha256": file_hash(source_dir / "authoritative_conditions.json"),
        "render_index_sha256": file_hash(output_dir / "render_index.json"),
        "files": {name: {"sha256": file_hash(output_dir / name), "size_bytes": (output_dir / name).stat().st_size} for name in names},
        "rendered_source_hashes": hashes_before,
        "pdf_visual_QA": "required_before_delivery" if pdf else "not_requested"}
