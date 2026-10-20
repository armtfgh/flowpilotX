"""Presentation of the offline corrected gas-liquid KHU design.

No optimization or model calls occur here. Source values and hypothesis/evidence
boundaries are retained; figures are supplied by the isolated FlowPilot renderer.
"""
from __future__ import annotations

import csv
import html
import importlib.util
import json
from pathlib import Path

from .provenance import digest, file_hash

STATUS = "Conditional engineering proposal - laboratory confirmation required"
ORIGIN = ("This is an offline engineering correction of the earlier time-basis design. "
          "No new Astra or council review was run. Earlier model nominations remain archived; "
          "this replacement is not a new history-informed versus history-omitted comparison.")
TIME_NOTE = ("The design axis is estimated gas-liquid transit at the declared pressure. "
             "The ideal-gas, no-slip calculation is not measured liquid residence time. "
             "Liquid and gas delivery are solved together and recomputed after controller rounding.")
GAS_NOTE = ("The O2 volumetric flow is the calculated oxygen component within air, not a second gas stream. "
            "Inlet delivery is not absorbed oxygen or measured chemical conversion.")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _asset(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe figure path: {relative}")
    return path


def _text(value):
    if value is None:
        return "Unconfirmed"
    if isinstance(value, list):
        return "; ".join(_text(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_text(v)}" for k, v in value.items())
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def _condition_row(row):
    p, c, target = row["proposed_setpoints"], row["calculated"], row["target"]
    return {"condition_id": row["condition_id"], "candidate_id": row["candidate_id"],
        "target_gas_liquid_transit_min": target["nominal_mixture_transit_min"],
        "estimated_gas_liquid_transit_min": c["no_slip_mixture_transit_min"],
        "liquid_flow_mL_min": p["liquid_flow_mL_min"],
        "inlet_air_STP_mL_min": p["gas_reference_flow_mL_min"],
        "target_inlet_oxygen_equivalents": target["inlet_oxygen_equivalents"],
        "calculated_inlet_oxygen_equivalents": c["oxygen_equivalents_supplied"],
        "time_relative_quantization_error": row["realized_time_relative_error"],
        "oxygen_relative_quantization_error": row["realized_oxygen_relative_error"],
        "BPR_equipment_id": p["bpr_equipment_id"], "BPR_label_bar": p["bpr_label_value_bar"],
        "BPR_label_basis_assumption": p["bpr_label_basis_assumption"],
        "reactor_pressure_bar_abs_assumed": c["reactor_pressure_bar_abs_assumed"],
        "actual_measured_residence_time_min": None}


def _oxygen_row(row):
    c = row["calculated"]
    return {"condition_id": row["condition_id"],
        "inlet_air_STP_mL_min": c["gas_reference_flow_mL_min"],
        "O2_within_air_STP_mL_min_calculated": c["oxygen_reference_volume_flow_mL_min"],
        "total_gas_mmol_min": c["total_gas_mmol_min"],
        "oxygen_mmol_min_supplied": c["oxygen_mmol_min_supplied"],
        "substrate_mmol_min": c["substrate_mmol_min"], "catalyst_mmol_min": c["catalyst_mmol_min"],
        "oxygen_equivalents_supplied": c["oxygen_equivalents_supplied"],
        "gas_reference_temperature_K_assumed": c["gas_reference_temperature_K_assumed"],
        "gas_reference_pressure_bar_abs_assumed": c["gas_reference_pressure_bar_abs_assumed"],
        "gas_oxygen_mole_fraction_assumed": c["gas_oxygen_mole_fraction_assumed"]}


MATERIAL_NOTE = ("Rates apply only while feed is delivered; they are not a one-hour run protocol. "
                 "Durations and campaign totals are unassigned. Total feed = Q_L x (equilibration + collection time) "
                 "+ additional dead-volume, flushing and loss allowances; avoid counting the same volume twice.")


def _material_rows(design):
    rows=[]
    for source in design["conditions"]:
        c=source["calculated"]
        rows.append({"condition_id":source["condition_id"],"repeat_of":None,
            "feed_mL_per_elapsed_hour":c["feed_mL_per_elapsed_hour"],
            "substrate_mmol_per_elapsed_hour":c["substrate_mmol_min"]*60,
            "acridine_mmol_per_elapsed_hour":c["catalyst_mmol_min"]*60,
            "actual_feed_duration_min":None,"total_feed_volume_mL":None,
            "total_substrate_mmol":None,"total_acridine_mmol":None,
            "basis":"Scalable rate during feed delivery; no one-hour run or total consumption assigned"})
    repeat=design["independent_repeat"]
    reference=next(row for row in rows if row["condition_id"]==repeat["repeat_of"])
    rows.append({**reference,"condition_id":repeat["condition_id"],"repeat_of":repeat["repeat_of"]})
    return rows


def _material_table(design):
    return [[r[k] for k in ("condition_id","feed_mL_per_elapsed_hour","substrate_mmol_per_elapsed_hour","acridine_mmol_per_elapsed_hour")] for r in _material_rows(design)]


def _shared_rows(design):
    s = design["scenario"]
    pressure, gas = s["pressure"], s["gas_reference"]
    return [
        {"item": "Chemistry", "value": f"{s['chemistry']['substrate']}, {s['chemistry']['substrate_concentration_M']:g} M in {s['chemistry']['solvent']}; {s['chemistry']['catalyst']} {s['chemistry']['catalyst_loading_mol_percent']:g} mol%", "basis": "shared proposed feed; batch evidence does not establish flow kinetics"},
        {"item": "Reactor", "value": f"{s['reactor_volume_mL']:g} mL PFA, {s['reactor_ID_mm']:g} mm ID", "basis": "proposed inventory selection; nominal geometry"},
        {"item": "Temperature", "value": f"{s['temperature_C']:g} C", "basis": "proposed setting; not measured"},
        {"item": "BPR", "value": f"{pressure['bpr_equipment_id']}: {pressure['label_value_bar']:g} bar", "basis": "proposed fixed inventory option"},
        {"item": "BPR pressure basis", "value": pressure["label_basis_assumption"], "basis": "assumed interpretation; source basis unconfirmed" if pressure["label_basis_source"] is None else _text(pressure["label_basis_source"])},
        {"item": "Pressure in calculation", "value": f"{pressure['reactor_pressure_bar_abs_assumed']:g} bar absolute", "basis": f"{pressure['label_value_bar']:g} bar assumed gauge + {pressure['ambient_pressure_bar_abs_assumed']:g} bar assumed ambient; uniform BPR-setpoint approximation"},
        {"item": "Gas reference", "value": f"{gas['temperature_K']:g} K / {gas['pressure_bar_abs']:g} bar absolute", "basis": "declared STP assumption; active MFC reference/calibration needs confirmation"},
        {"item": "Air composition", "value": f"O2 mole fraction {gas['oxygen_mole_fraction']:g}", "basis": "assumed dry-air composition"},
        {"item": "Actual operating data", "value": "Unknown", "basis": "pressure/holdup/RTD, light setting, feed mode and collection are not measured or assigned"},
    ]


def _load(design_path, output_dir, history_path):
    design = _read(design_path)
    if design["laboratory_released"] is not False or design["experimental_validation"] is not False:
        raise ValueError("Corrected report requires an unreleased, unvalidated engineering proposal")
    rows = design["conditions"]
    if [r["condition_id"] for r in rows] != [f"KD{i:02d}" for i in range(1, 7)]:
        raise ValueError("Six ordered corrected conditions KD01-KD06 are required")
    if len({r["candidate_id"] for r in rows}) != 6:
        raise ValueError("Corrected candidates must be distinct")
    index = _read(output_dir / "render_index.json")
    cases = index["cases"]
    if len(cases) != 6 or {c["condition_id"] for c in cases} != {r["condition_id"] for r in rows}:
        raise ValueError("Render index must contain exactly the six corrected figures")
    by_id = {c["condition_id"]: c for c in cases}
    ordered = []
    for row in rows:
        case = by_id[row["condition_id"]]
        if case.get("candidate_id", case["condition_id"]) != row["candidate_id"]:
            raise ValueError("Figure candidate does not match corrected source condition")
        for key in ("png", "svg", "topology_json"):
            _asset(output_dir, case[key])
        if any(value is not None for value in row["actual"].values()):
            raise ValueError("Unknown actual conditions/outcomes must remain null")
        rationale = row["rationale"]
        for key in ("hypothesis", "evidence_ids", "evidence_summary", "why_this_condition", "readout", "limitation"):
            if not rationale.get(key):
                raise ValueError(f"Missing per-design rationale field: {key}")
        ordered.append(case)
    repeat = design["independent_repeat"]
    if repeat["condition_id"] != "KD-R01" or repeat["repeat_of"] not in by_id or repeat["execution_order"] is not None:
        raise ValueError("Independent repeat must reference a core condition with unassigned execution order")
    history = _read(history_path) if history_path else None
    if history:
        report = history["report_summary"]
        allowed = {r["evidence_id"] for r in report["reference_rows"]}
        rationales=[r["rationale"] for r in rows]
        rationales += [r for r in (design.get("held_fixed_recipe_rationale"),repeat.get("rationale")) if r]
        for rationale in rationales:
            if not set(rationale["evidence_ids"]) <= allowed:
                raise ValueError("Per-design citation is absent from the evidence ledger")
    for row,case in zip(rows,ordered):
        topology=_read(_asset(output_dir,case["topology_json"]))
        if topology.get("condition_id")!=row["condition_id"] or topology.get("source_condition_sha256")!=digest(row):
            raise ValueError("Rendered topology source-condition hash differs from corrected proposal")
    return design, ordered, history


def _csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _table(headers, rows):
    return '<table><thead><tr>' + ''.join('<th>' + html.escape(h).replace('\n', '<br>') + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(_text(c)) + '</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table>'


def _display_tables(design):
    rows = design["conditions"]
    settings = [[r["condition_id"], r["target"]["nominal_mixture_transit_min"], r["calculated"]["no_slip_mixture_transit_min"],
        r["proposed_setpoints"]["liquid_flow_mL_min"], r["proposed_setpoints"]["gas_reference_flow_mL_min"],
        r["target"]["inlet_oxygen_equivalents"], r["calculated"]["oxygen_equivalents_supplied"]] for r in rows]
    delivery = [[r["condition_id"], r["calculated"]["gas_reference_flow_mL_min"], r["calculated"]["oxygen_reference_volume_flow_mL_min"],
        r["calculated"]["total_gas_mmol_min"], r["calculated"]["oxygen_mmol_min_supplied"], r["calculated"]["substrate_mmol_min"],
        r["calculated"]["oxygen_equivalents_supplied"]] for r in rows]
    return settings, delivery


SETTINGS_HEADERS = ["Design", "Target transit\nmin", "Estimated transit\nmin", "Liquid\nmL/min", "Inlet air\nSTP mL/min", "Target O2\neq", "Calculated O2\neq"]
DELIVERY_HEADERS = ["Design", "Inlet air\nSTP mL/min", "O2 within air\nSTP mL/min", "Total gas\nmmol/min", "O2 supplied\nmmol/min", "Substrate\nmmol/min", "Supplied O2\neq"]


def _pressure_note(design):
    p, ref = design["scenario"]["pressure"], design["scenario"]["gas_reference"]
    return (f"Proposed {p['bpr_equipment_id']}: {p['label_value_bar']:g} bar, treated as {p['label_basis_assumption']} for planning. "
            f"The calculation uses {p['reactor_pressure_bar_abs_assumed']:g} bar absolute; the source pressure basis remains unconfirmed. "
            f"STP here means {ref['temperature_K']:g} K / {ref['pressure_bar_abs']:g} bar absolute, with air O2 fraction {ref['oxygen_mole_fraction']:g}.")


def _engineering_blocks(design):
    rationale=design["configuration"].get("rationale",{})
    names={"time_window":"Practical time window", "coil_choice":"Selected 5 mL coil", "bpr_choice":"Selected BPR and pressure model", "oxygen_levels":"Two inlet oxygen levels"}
    return [(label,rationale[key]) for key,label in names.items() if key in rationale]


def _stp_reference(design):
    refs=design["configuration"].get("reference_sources",[])
    if not refs:return "The gas reference state remains an explicit engineering assumption."
    ref=refs[0]
    return ("STP reference: Bronkhorst manual 917132, section 5.5.4, printed page 35, specifies SCCM at 0 C / 101.325 kPa absolute. "
            "This supports the reporting convention; installed MFC units and calibration were not read. " + ref["url"])


def _evidence_blocks(row):
    r = row["rationale"]
    return [("Hypothesis", r["hypothesis"]), ("Evidence", r["evidence_summary"]),
            ("Why this condition", r["why_this_condition"]), ("What the result tests", r["readout"]),
            ("Limit", r["limitation"]), ("Evidence IDs", r["evidence_ids"])]


def _feed_note(preparation):
    return (f"Scalable feed example: make to {preparation['final_volume_example_mL']:g} mL with EtOH using "
            f"{preparation['substrate_mmol_example']:g} mmol thioanisole and {preparation['catalyst_mmol_example']:g} mmol acridine "
            f"({preparation['pure_equivalent_mass_mg']['thioanisole']:g} and {preparation['pure_equivalent_mass_mg']['acridine']:g} mg pure-equivalent, respectively). "
            "Actual bottle purity and solubility are unknown; weighed mass requires the actual purity correction. "
            "This preserved preparation example does not establish total campaign material demand.")


def _html(design, cases, history, preparation):
    settings, delivery = _display_tables(design)
    body = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><title>KHU corrected gas-liquid screen</title>',
        '<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1280px;margin:32px auto;padding:0 24px;color:#173640}h1{font-size:32px}h2{color:#17675f;margin-top:34px}table{border-collapse:collapse;width:100%;font-size:14px;margin:14px 0}th,td{border:1px solid #c3d3d6;padding:9px;text-align:left;vertical-align:top}th{background:#e6f2ef}.status{padding:14px;background:#fff2dc;border-left:5px solid #b36a24}.figure{width:100%;height:auto;border:1px solid #d6e1e2}.small{font-size:13px;color:#4c646c}.evidence{display:grid;grid-template-columns:1fr 1fr;gap:10px 24px}.evidence p{margin:5px 0}a{color:#166b60}</style></head><body>',
        '<h1>KHU corrected gas-liquid first screen</h1><p class="status"><strong>' + STATUS + '</strong></p>',
        '<p>' + html.escape(ORIGIN) + '</p><p>Six operating conditions share one proposed FlowPilot apparatus topology. ' + html.escape(TIME_NOTE) + '</p>',
        '<h2>Paired time and oxygen targets</h2>', _table(SETTINGS_HEADERS, settings), '<p>' + html.escape(_pressure_note(design)) + '</p>',
        '<h2>Inlet oxygen delivery</h2>', _table(DELIVERY_HEADERS, delivery), '<p>' + html.escape(GAS_NOTE) + '</p>',
        '<h2>Shared settings and assumptions</h2>', _table(["Item", "Value", "Basis"], [list(r.values()) for r in _shared_rows(design)]),
        '<p>Independently prepare and operate ' + html.escape(design["independent_repeat"]["condition_id"]) + ' as a repeat of ' + html.escape(design["independent_repeat"]["repeat_of"]) + '. Execution order, collection duration and total material requirements remain unassigned. No replacement history-omitted nominees or ranks are asserted.</p>']
    body += ['<h2>Scalable feed preparation</h2><p>' + html.escape(_feed_note(preparation)) + '</p>']
    body += ['<h2>Scalable material rates</h2>',_table(["Design","Feed mL/h","Substrate mmol/h","Acridine mmol/h"],_material_table(design)), '<p>'+html.escape(MATERIAL_NOTE)+'</p>']
    body += ['<p class="small">' + html.escape(_stp_reference(design)) + '</p>', '<h2>Engineering basis for the corrected window</h2>']
    body.extend('<p><strong>' + html.escape(label) + ':</strong> ' + html.escape(_text(value)) + '</p>' for label,value in _engineering_blocks(design))
    for row, case in zip(design["conditions"], cases):
        body += ['<h2>' + html.escape(case["title"]) + '</h2>', '<img class="figure" src="' + html.escape(case["svg"], quote=True) + '" alt="' + html.escape(case["title"], quote=True) + '">', '<div class="evidence">']
        body.extend('<p><strong>' + label + ':</strong> ' + html.escape(_text(value)) + '</p>' for label, value in _evidence_blocks(row))
        body += ['</div><p class="small"><a href="' + html.escape(case["png"], quote=True) + '">PNG</a> | <a href="' + html.escape(case["svg"], quote=True) + '">SVG</a> | <a href="' + html.escape(case["topology_json"], quote=True) + '">Topology JSON</a></p>']
    for label,rationale in [("Shared recipe evidence",design.get("held_fixed_recipe_rationale")),("Independent-repeat evidence",design["independent_repeat"].get("rationale"))]:
        if rationale:
            body.append('<h2>'+html.escape(label)+'</h2>')
            body.extend('<p><strong>'+html.escape(k.replace('_',' ').title())+':</strong> '+html.escape(_text(v))+'</p>' for k,v in rationale.items())
    if history:
        report = history["report_summary"]
        body += ['<h2>Historical evidence coverage and model use</h2>', _table(["Item", "Coverage", "Interpretation"], [[r[k] for k in ("item", "value", "interpretation")] for r in report["coverage_rows"]]),
            '<h2>All-factor descriptive evidence review</h2>', _table(["Factor", "Evidence summary", "Flow implication", "Limitations", "Evidence IDs"], [[r[k] for k in ("factor", "evidence_summary", "flow_implication", "limitations", "evidence_ids")] for r in report["factor_rows"]]),
            '<h2>Evidence references</h2>', _table(["Evidence ID", "Label", "Source location"], [[r[k] for k in ("evidence_id", "label", "source_location")] for r in report["reference_rows"]])]
    body += ['<h2>Limitations</h2><ul>' + ''.join('<li>' + html.escape(_text(x)) + '</li>' for x in design["limitations"]) + '</ul>',
        '<p class="small">CSV files retain source precision; tables display rounded values. Evidence is descriptive and batch-specific. No measured flow yield or superiority is claimed.</p></body></html>']
    return '\n'.join(body)


def _pdf(path, output_dir, design, cases, history, preparation):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import Paragraph, Table, TableStyle

    width, height = landscape(A4)
    margin, usable = 30, width - 60
    font = "Helvetica"
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"):
        if Path(candidate).exists():
            pdfmetrics.registerFont(TTFont("KHUCorrectedSans", candidate)); font = "KHUCorrectedSans"; break
    styles = {"body": ParagraphStyle("CorrectedBody", fontName=font, fontSize=9.3, leading=12.5, textColor=colors.HexColor("#183840")),
              "small": ParagraphStyle("CorrectedSmall", fontName=font, fontSize=8, leading=10.5, textColor=colors.HexColor("#183840"))}
    page = 1
    with path.open("xb") as stream:
        canvas = Canvas(stream, pagesize=(width, height), pageCompression=1)
        canvas.setTitle("KHU corrected gas-liquid flow proposals")
        canvas.setAuthor("KHU study-specific FlowPilot engineering adapter")
        def paragraph(text, top, style="body", x=margin, w=usable, allow_low=False):
            p = Paragraph(html.escape(_text(text)).replace('\n', '<br/>'), styles[style])
            _, h = p.wrap(w, height)
            if top - h < 38 and not allow_low:
                raise ValueError("Corrected report paragraph exceeds page; shorten text or review layout")
            p.drawOn(canvas, x, top-h)
            return top-h
        def heading(text, top=height-margin, size=18):
            canvas.setFillColor(colors.HexColor("#17695e")); canvas.setFont(font, size)
            canvas.drawString(margin, top-size, str(text)); return top-size-10
        def footer():
            canvas.setStrokeColor(colors.HexColor("#cad9da")); canvas.line(margin, 26, width-margin, 26)
            canvas.setFillColor(colors.HexColor("#506971")); canvas.setFont(font, 7.2)
            canvas.drawString(margin, 14, "KHU | Offline engineering correction | Calculated transit, not measured residence | Laboratory release pending")
            canvas.drawRightString(width-margin, 14, str(page))
        def next_page():
            nonlocal page
            footer(); canvas.showPage(); page += 1
        def table(headers, rows, top, widths, compact=False):
            cells = [[Paragraph(html.escape(_text(c)).replace('\n', '<br/>'), styles["small"]) for c in r] for r in [headers,*rows]]
            t=Table(cells,colWidths=widths)
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e7f3ef")),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#bbced1")),("VALIGN",(0,0),(-1,-1),"TOP"),
                ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),2 if compact else 6),("BOTTOMPADDING",(0,0),(-1,-1),2 if compact else 6)]))
            _,h=t.wrap(usable,height)
            if top-h<38:raise ValueError("Corrected report table exceeds page; review layout")
            t.drawOn(canvas,margin,top-h);return top-h
        settings,delivery=_display_tables(design)
        seven=[55,105,120,100,130,120,usable-630]
        top=heading("KHU corrected gas-liquid first screen",size=21)
        top=paragraph(STATUS + ". " + ORIGIN,top)-12
        top=heading("Three transit targets x two inlet oxygen levels",top,14)
        top=table(SETTINGS_HEADERS,settings,top,seven)-10
        top=paragraph(TIME_NOTE,top)-10
        top=heading("Inlet air and oxygen delivery",top,14)
        top=table(DELIVERY_HEADERS,delivery,top,[55,125,130,115,120,110,usable-655])-9
        paragraph(GAS_NOTE,top,"small")
        next_page()
        top=heading("Apparatus, pressure basis and calculation limits")
        top=table(["Item","Proposed value","Basis / remaining confirmation"],[list(r.values()) for r in _shared_rows(design)],top,[150,270,usable-420])-12
        top=paragraph(f"Independent repeat: {design['independent_repeat']['condition_id']} repeats {design['independent_repeat']['repeat_of']} with independent preparation and operation. Seven planned runs; execution order and collection windows are unassigned. No new history-omitted comparison nominees are assigned.",top)-10
        top=paragraph(_feed_note(preparation),top,"small")-10
        top=paragraph("Core equations: Q_gas,reactor = Q_air,STP (P_STP/P_abs)(T_reactor/T_STP); transit = V/(Q_L + Q_gas,reactor). O2 molar supply = x_O2 P_STP Q_air,STP/(R T_STP); O2 equivalents = O2 molar supply/(C Q_L). Controller rounding is followed by recomputation.",top,"small")
        top=heading("Scalable material rates",top-7,12)
        top=table(["Design","Feed mL/h","Substrate mmol/h","Acridine mmol/h"],_material_table(design),top,[100,210,235,usable-545],compact=True)-6
        paragraph(MATERIAL_NOTE,top,'small')
        next_page()
        top=heading("Engineering basis for the corrected window")
        for label,value in _engineering_blocks(design):
            top=heading(label,top,13)
            top=paragraph(value,top)-13
        paragraph(_stp_reference(design),top,'small')
        next_page()
        for row,case,setting in zip(design['conditions'],cases,settings):
            top=heading(case['title'])
            top=table(SETTINGS_HEADERS,[setting],top,seven)-8
            image=ImageReader(str(_asset(output_dir,case['png'])));iw,ih=image.getSize()
            low=137;area_h=top-low;scale=min(usable/iw,area_h/ih);dw,dh=iw*scale,ih*scale
            canvas.drawImage(image,margin+(usable-dw)/2,low+(area_h-dh)/2,width=dw,height=dh,mask='auto')
            bottom=paragraph("Hypothesis: "+_text(row['rationale']['hypothesis']),122)
            bottom=paragraph(_pressure_note(design),bottom-7,'small')
            paragraph("See the per-design evidence section for cited observations, discriminating readouts and limitations.",bottom-5,'small')
            next_page()
        def paragraph_height(text, style="small"):
            p=Paragraph(html.escape(_text(text)).replace('\n','<br/>'),styles[style])
            return p.wrap(usable,height)[1]
        top=heading("Per-design evidence and experimental questions")
        for row in design['conditions']:
            blocks=_evidence_blocks(row)
            needed=33+sum(paragraph_height(label+": "+_text(value))+5 for label,value in blocks)
            if top-needed<38:
                next_page();top=heading("Per-design evidence - continued")
            identity=row["condition_id"] if row["condition_id"]==row["candidate_id"] else row["condition_id"]+" / "+row["candidate_id"]
            top=heading(identity,top,13)
            for label,value in blocks:top=paragraph(label+": "+_text(value),top,'small')-5
            top-=10
        for label,key in (("Shared recipe basis","held_fixed_recipe_rationale"),("Independent repeat","independent_repeat")):
            rationale=design.get(key,{})
            if key=="independent_repeat":rationale=rationale.get("rationale",{})
            if rationale:
                blocks=[(k.replace('_',' ').title(),v) for k,v in rationale.items()]
                needed=33+sum(paragraph_height(k+": "+_text(v))+5 for k,v in blocks)
                if top-needed<38:next_page();top=heading("Shared recipe and repeat evidence")
                top=heading(label,top,13)
                for key,value in blocks:top=paragraph(key+": "+_text(value),top,'small')-5
                top-=10
        next_page()
        if history:
            report=history['report_summary'];top=heading("Historical coverage and original model use")
            # Coverage rows are individually paginated to preserve readable type.
            for row in report['coverage_rows']:
                line=f"{row['item']}: {_text(row['value'])}. {_text(row['interpretation'])}"
                if top-paragraph_height(line,"body")<38:next_page();top=heading("Historical coverage - continued")
                top=paragraph(line,top)-10
            next_page();top=heading("All-factor historical review")
            for row in report['factor_rows']:
                lines=[("Evidence",row['evidence_summary']),("Flow implication",row['flow_implication']),("Limit",row['limitations']),("Evidence IDs",row['evidence_ids'])]
                estimated=32+sum(paragraph_height(label+": "+_text(v))+4 for label,v in lines)
                if top-estimated<45:next_page();top=heading("All-factor historical review - continued")
                top=heading(row['factor'],top,13)
                for label,value in lines:top=paragraph(label+": "+_text(value),top,'small')-4
                top-=9
            next_page();top=heading("Evidence reference ledger")
            cited={eid for r in design['conditions'] for eid in r['rationale']['evidence_ids']}
            cited.update(design['independent_repeat'].get('rationale',{}).get('evidence_ids',[]))
            cited.update(design.get('held_fixed_recipe_rationale',{}).get('evidence_ids',[]))
            for factor in report['factor_rows']:cited.update(factor['evidence_ids'])
            top=paragraph('References cited in this report are listed below. The complete record/group reference registry is retained in the accompanying history ledger.',top,'small')-10
            for row in report['reference_rows']:
                if row['evidence_id'] not in cited:continue
                line=f"{row['evidence_id']} | {row['label']} | {row['source_location']}"
                if top-paragraph_height(line)<38:next_page();top=heading("Evidence reference ledger - continued")
                top=paragraph(line,top,'small')-9
        else:
            top=heading("Historical evidence appendix not supplied")
            paragraph("This draft omits the requested coverage appendix and is incomplete for final delivery.",top)
        footer();canvas.save()
    return page


def export_corrected_report(design_path, output_dir, *, history_path=None, feed_path=None, pdf=True):
    """Create exclusive report files beside the six corrected rendered figures."""
    design_path,output_dir=Path(design_path),Path(output_dir)
    history_path=Path(history_path) if history_path else None
    design,cases,history=_load(design_path,output_dir,history_path)
    feed_path=Path(feed_path) if feed_path else Path(__file__).resolve().parents[2]/'outputs/khu_first_screen_astra_sol_2026-10-13_v2/feed_preparation.json'
    preparation=_read(feed_path)
    chemistry=design['scenario']['chemistry']
    if any(preparation[k]!=chemistry[k] for k in ('substrate','catalyst','solvent','substrate_concentration_M','catalyst_loading_mol_percent')):
        raise ValueError('Preserved feed preparation differs from corrected chemistry')
    names=['summary.html','conditions.csv','oxygen_delivery.csv','shared_settings.csv','independent_repeat.csv','design_evidence.csv','material_rates.csv']+(['corrected_flow_proposals.pdf'] if pdf else [])
    if any((output_dir/name).exists() for name in names):raise FileExistsError("Corrected reports are immutable; choose a new output version")
    if pdf and importlib.util.find_spec('reportlab') is None:raise RuntimeError('PDF generation requires reportlab')
    paths=[design_path,output_dir/'render_index.json',feed_path]+([history_path] if history_path else [])
    paths += [_asset(output_dir,c[k]) for c in cases for k in ('png','svg','topology_json')]
    before={str(p):file_hash(p) for p in paths}
    conditions=[_condition_row(r) for r in design['conditions']]
    repeat=design['independent_repeat'];source=next(r for r in design['conditions'] if r['condition_id']==repeat['repeat_of'])
    repeat_row={**_condition_row(source),'condition_id':repeat['condition_id'],'repeat_of':repeat['repeat_of'],'preparation':'independent preparation and operation','execution_order':None}
    evidence=[{'condition_id':r['condition_id'],**{k:_text(v) for k,v in r['rationale'].items()}} for r in design['conditions']]
    for name,rows in [('conditions.csv',conditions),('oxygen_delivery.csv',[_oxygen_row(r) for r in design['conditions']]),('shared_settings.csv',_shared_rows(design)),('independent_repeat.csv',[repeat_row]),('design_evidence.csv',evidence),('material_rates.csv',_material_rows(design))]:_csv(output_dir/name,rows)
    with (output_dir/'summary.html').open('x',encoding='utf-8') as stream:stream.write(_html(design,cases,history,preparation))
    pages=_pdf(output_dir/'corrected_flow_proposals.pdf',output_dir,design,cases,history,preparation) if pdf else None
    if before!={str(p):file_hash(p) for p in paths}:raise ValueError('Source or figure changed during export; retain and review incomplete version')
    return {'status':STATUS,'condition_count':6,'planned_run_count':7,'pdf_page_count':pages,'history_appendix_included':history is not None,
        'output_dir':str(output_dir.resolve()),'source_sha256':before,'files':{n:{'sha256':file_hash(output_dir/n),'size_bytes':(output_dir/n).stat().st_size} for n in names},
        'pdf_visual_QA':'required_before_delivery' if pdf else 'not_requested'}
