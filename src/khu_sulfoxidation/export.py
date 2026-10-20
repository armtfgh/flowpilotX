"""One campaign object feeds every local review artifact. No network access."""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import platform
import zipfile

from .comparison import build_comparison_plan
from .evidence import build_evidence_packet
from .history import ingest_history
from .inventory import reconcile_inventory, validate_apparatus
from .materials import feed_preparation, run_consumption, validate_budget_inputs
from .observations import results_template
from .provenance import digest, file_hash, source_snapshot, write_json
from .screen_selector import generate_screen


def build_campaign(root, config, apparatus=None):
    root = Path(root)
    config = deepcopy(config)
    validate_budget_inputs(config["materials"])
    if config.get("live_model_calls_enabled") is not False:
        raise ValueError("This offline implementation never enables live model calls")
    history = ingest_history(root)
    apparatus = deepcopy(apparatus) if apparatus is not None else reconcile_inventory(root)
    screen = generate_screen(apparatus, config["chemistry"], config["screen"])
    evidence = build_evidence_packet(history, match=config["batch_evidence_match"])
    comparison = build_comparison_plan(config["optional_history_omitted_nominations"])
    candidate_map = {c["candidate_id"]: c for c in screen["candidates"]}
    rows, templates = [], []
    slots = screen.get("selected_runs") or screen["planned_slots"]
    execution_order = config["order"].get("execution_order")
    slot_ids = [s["run_id"] for s in slots]
    if execution_order is not None and (not isinstance(execution_order, list) or len(execution_order) != len(slot_ids) or set(execution_order) != set(slot_ids)):
        raise ValueError("Execution order must contain every core run exactly once; optional nominations freeze separately")
    for i, slot in enumerate(slots, 1):
        candidate = candidate_map.get(slot.get("candidate_id"), {})
        target = slot.get("target") or candidate.get("target", {})
        calc = slot.get("calculated") or candidate.get("calculated") or {}
        cid = slot.get("run_id") or slot.get("slot_id") or slot.get("candidate_id") or f"R{i:02d}"
        row = {"condition_id": cid, "design_version": config["configuration_version"],
               "study_configuration_sha256": digest(config),
               "screen_configuration_sha256": screen["configuration_sha256"],
               "comparison_arm": config["screen"]["comparison_arm"],
               "table_order_only": i, "proposed_execution_order": execution_order.index(cid)+1 if execution_order is not None else None,
               "repeat_of": slot.get("repeat_of"), "candidate_id": slot.get("candidate_id"),
               "preparation_type": slot.get("preparation_type"),
               "target": target, "calculated": calc, "chemistry": config["chemistry"],
               "apparatus_profile_id": apparatus["profile_id"],
               "proposed_apparatus": {k:deepcopy(apparatus[k]) for k in ("pump","mfc","gas","reactor","light","pressure","connections","operating")},
               "time_coordinate": candidate.get("time_coordinate"), "oxygen_coordinate": candidate.get("oxygen_coordinate"),
               "status": candidate.get("status", "blocked"),
               "supporting_history_ids": config["chemistry"]["source_record_ids"],
               "rationale": config["screen"]["selection_rationale"],
               "experimental_validation": False}
        row["material_estimate"] = run_consumption(
            liquid_flow_mL_min=calc.get("liquid_flow_mL_min"),
            gas_reference_flow_mL_min=calc.get("gas_reference_flow_mL_min"),
            substrate_concentration_M=config["chemistry"]["substrate_concentration_M"],
            catalyst_loading_mol_percent=config["chemistry"]["catalyst_loading_mol_percent"],
            feed_mode=apparatus.get("operating", {}).get("feed_mode"), sampling=config["materials"]["sampling"])
        rows.append(row)
        template = results_template(cid, row, config["configuration_version"])
        template["repeat_of"] = row["repeat_of"]
        templates.append(template)
    # These are budget reservations, not fabricated arm recommendations.
    optional = [{"slot_id": f"HO{i+1:02d}", "status": "unfilled_nomination_slot", "candidate_id": None,
                 "comparison_arm": "history_omitted", "proposed_conditions": None,
                 "rationale": "Fill through the isolated matched-input workflow before flow outcomes."}
                for i in range(config["optional_history_omitted_nominations"])]
    preparation = feed_preparation(config["materials"]["final_stock_volume_mL"],
        config["chemistry"]["substrate_concentration_M"], config["chemistry"]["catalyst_loading_mol_percent"],
        config["materials"]["properties"])
    missing = validate_apparatus(apparatus)
    for rejection in screen.get("rejection_reasons", []):
        missing.extend(rejection["reasons"])
    for key in ("sulfone_limit_pct", "analytical_method", "sample_handling", "sample_capacity_per_day"):
        if config["endpoints"].get(key) is None: missing.append(f"endpoints.{key}: pending KHU")
    for key in ("identity_scope_confirmed", "solubility_confirmed"):
        if config["chemistry"].get(key) is not True: missing.append(f"chemistry.{key}: pending KHU")
    if config["endpoints"].get("analytical_basis_confirmed") is not True:
        missing.append("analytical_basis_confirmed: pending KHU")
    for key in ("final_stock_volume_mL", "extra_stock_reserve_mL", "available_substrate_mmol", "available_catalyst_mmol"):
        if config["materials"].get(key) is None: missing.append(f"materials.{key}: pending")
    if config["order"].get("execution_order") is None: missing.append("execution order: pending pre-outcome freeze")
    if config.get("laboratory_review_complete") is not True: missing.append("KHU laboratory review: pending")
    estimates = [r["material_estimate"] for r in rows]
    complete_materials = bool(estimates) and all(r["status"] != "pending" for r in estimates)
    totals = {k: sum(r[k] for r in estimates) if complete_materials else None for k in
              ("feed_volume_mL", "substrate_mmol", "catalyst_mmol", "flush_solvent_mL", "additional_carrier_solvent_mL", "elapsed_min", "gas_reference_volume_mL")}
    totals.update(status="core_runs_only_calculated" if complete_materials else "pending_windows_and_setpoints",
                  optional_nominations_included=False, blanks_and_calibration_included=False,
                  reserve_mL=config["materials"]["extra_stock_reserve_mL"])
    if not complete_materials:
        missing.append("Material/time totals: collection and consumption windows incomplete")
        for row in rows:
            missing.extend(f"{row['condition_id']} sampling: {field}" for field in row["material_estimate"].get("missing", []))
    missing += [f"Feed preparation: {field}" for field in preparation["missing"]]
    for component in ("substrate", "catalyst"):
        available = config["materials"].get(f"available_{component}_mmol")
        quantities = [q for q in [preparation.get(f"{component}_mmol"), totals.get(f"{component}_mmol")] if q is not None]
        reserve = config["materials"].get("extra_stock_reserve_mL")
        if complete_materials and reserve is not None:
            concentration = preparation["substrate_concentration_M" if component == "substrate" else "catalyst_concentration_M"]
            quantities.append(totals[f"{component}_mmol"] + reserve*concentration)
        required = max(quantities) if quantities else None
        totals[f"minimum_{component}_required_mmol_including_known_preparation_and_reserve"] = required
        if available is not None and required is not None and available < required:
            missing.append(f"Insufficient {component} inventory for known preparation or core consumption plus declared reserve; optional conditions are additional")
    if complete_materials:
        prepared = config["materials"].get("final_stock_volume_mL")
        reserve = config["materials"].get("extra_stock_reserve_mL")
        if prepared is not None and reserve is not None and prepared < totals["feed_volume_mL"] + reserve:
            missing.append("Prepared stock volume is below core feed consumption plus reserve")
    return {"schema_version": "khu_review_package_v1", "status": "conditional_review_not_released_for_experiment",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "as_of_date": config["as_of_date"],
            "target_delivery_date": config["target_delivery_date"], "configuration": config,
            "configuration_sha256": digest(config), "history": history, "evidence": evidence,
            "apparatus": apparatus, "screen": screen, "condition_rows": rows, "optional_slots": optional,
            "results_template": templates, "feed_preparation": preparation, "core_material_totals": totals,
            "comparison_plan": comparison, "release_blockers": sorted(set(missing)),
            "methods": {"model_calls": 0, "paid_cost_USD": 0, "legacy_yield_heuristic_used": False,
                        "batch_as_flow_kinetics": False, "external_delivery": False,
                        "time_oxygen_coordinate_source": "caller_supplied_screen_configuration",
                        "history_automatically_selects_time_oxygen_coordinates": False,
                        "recommendations_frozen": False, "runtime": platform.python_version()},
            "source_snapshot": source_snapshot(root)}


def _flatten(value, prefix=""):
    result = {}
    for k, v in value.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict): result.update(_flatten(v, name))
        elif isinstance(v, list): result[name] = json.dumps(v, ensure_ascii=False)
        else: result[name] = v
    return result


def _write_csv(path, rows):
    rows = [_flatten(row) for row in rows]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def results_csv_rows(campaign):
    rows = []
    actual_fields = ("liquid_flow_mL_min", "gas_reference_flow_mL_min", "gas_identity", "gas_o2_mole_fraction",
        "mfc_calibration_id", "gas_reference_temperature_K", "gas_reference_pressure_bar_abs",
        "reactor_id", "reactor_volume_mL", "reactor_ID_mm", "pressure_value", "pressure_unit", "pressure_basis",
        "pressure_sensor_id", "temperature_C", "light_id", "wavelength_nm", "light_power_setting",
        "substrate_concentration_M", "catalyst_loading_mol_percent", "feed_preparation_id", "solvent",
        "feed_mode", "loop_volume_mL", "equilibration_min", "collection_start", "collection_end",
        "elapsed_min", "liquid_holdup", "tracer_contact_time_min", "collected_amount_mmol")
    for condition in campaign["condition_rows"]:
        row = {"condition_id": condition["condition_id"], "design_version": condition["design_version"],
               "study_configuration_sha256": condition["study_configuration_sha256"],
               "screen_configuration_sha256": condition["screen_configuration_sha256"], "repeat_of": condition["repeat_of"],
               "sample_id": None, "actual_execution_order": None, "operator": None, "run_datetime": None,
               "proposed": condition["calculated"], "actual": {k: None for k in actual_fields},
               "residual_sulfide_pct": None, "sulfoxide_pct": None, "sulfone_pct": None,
               "other_products_pct": None, "unknown_components_pct": None,
               "analytical_method": None, "internal_standard": None, "calibration_basis": None,
               "analytical_uncertainty": None, "repeat_type": None, "sample_handling": None,
               "analysis_delay_min": None, "post_collection_oxidation_observed": None,
               "precipitation_observed": None, "clogging_observed": None, "gas_behavior": None,
               "pressure_observations": None, "temperature_observations": None, "deviations": None}
        rows.append(row)
    return rows


# Shared layout coordinates for PDF and SVG. Dashed components require assignment.
NODES = [
    ("gas", 10, 25, 110, 44, ["Gas supply", "identity pending"]),
    ("regulator", 145, 25, 110, 44, ["Regulator / MFC", "calibration pending"]),
    ("valve", 280, 25, 110, 44, ["Check valve", "assignment pending"]),
    ("feed", 10, 125, 110, 44, ["Premixed feed", "mode pending"]),
    ("pump", 145, 125, 110, 44, ["R-series pump", "channel pending"]),
    ("mixer", 280, 125, 110, 44, ["Gas-liquid mixer", "assignment pending"]),
    ("coil", 415, 125, 110, 44, ["UV-150 / coil", "450 nm proposed"]),
    ("bpr", 550, 125, 110, 44, ["BPR", "basis pending"]),
    ("collect", 685, 125, 110, 44, ["Collection", "handling pending"]),
    ("sensor", 490, 25, 160, 44, ["Pressure / temperature", "sensing & control pending"]),
]
EDGES = [(120,47,145,47),(255,47,280,47),(335,69,335,125),(120,147,145,147),
         (255,147,280,147),(390,147,415,147),(525,147,550,147),(660,147,685,147)]


def diagram_nodes(campaign=None):
    nodes = deepcopy(NODES)
    if campaign is None: return nodes
    a = campaign["apparatus"]
    labels = {
        "gas": ["Gas supply", a["gas"].get("identity") or "identity pending"],
        "regulator": ["Regulator / MFC", "assigned" if a["mfc"].get("status") == "confirmed" else "calibration pending"],
        "valve": ["Check valve", "assigned" if a["connections"].get("check_valve_id") else "assignment pending"],
        "feed": ["Premixed feed", a["operating"].get("feed_mode") or "mode pending"],
        "pump": ["R-series pump", "assigned" if a["pump"].get("status") == "confirmed" else "channel pending"],
        "mixer": ["Gas-liquid mixer", "assigned" if a["connections"].get("mixer_id") else "assignment pending"],
        "coil": ["UV-150 / coil", f"{a['reactor']['volume_mL']:g} mL" if a["reactor"].get("volume_mL") else "450 nm proposed"],
        "bpr": ["BPR", f"{a['pressure']['setpoint_value']:g} {a['pressure'].get('unit')}" if a["pressure"].get("setpoint_value") is not None else "basis pending"],
        "collect": ["Collection", "defined" if a["connections"].get("collection_arrangement") else "handling pending"],
    }
    return [(key,x,y,w,h,labels.get(key,lines)) for key,x,y,w,h,lines in nodes]


def apparatus_svg(campaign=None):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="810" height="215" viewBox="0 0 810 215">',
             '<title>Proposed homogeneous apparatus topology; component assignments pending</title>',
             '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6" fill="#176b78"/></marker></defs>',
             '<rect width="810" height="215" fill="white"/>']
    for _, x,y,w,h,lines in diagram_nodes(campaign):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="#eff6f7" stroke="#176b78" stroke-dasharray="4 3"/>')
        for i,line in enumerate(lines): parts.append(f'<text x="{x+w/2}" y="{y+18+i*14}" text-anchor="middle" font-family="sans-serif" font-size="10">{html.escape(line)}</text>')
    for x1,y1,x2,y2 in EDGES: parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#176b78" marker-end="url(#arrow)"/>')
    parts.append('<path d="M570 69 L570 95 L470 95 L470 125" fill="none" stroke="#777" stroke-dasharray="3 3"/>')
    parts.append('<text x="10" y="200" font-family="sans-serif" font-size="11">Conceptual connection order. Sensor location, cooling, gas supply and collection assembly require KHU confirmation.</text></svg>')
    return "".join(parts)


def document_blocks(c):
    cfg = c["configuration"]
    chem, prep = cfg["chemistry"], c["feed_preparation"]
    unique_count = len(c["screen"]["candidates"])
    ntime, noxygen = len(cfg["screen"]["times_min"]), len(cfg["screen"]["oxygen_equivalents"])
    show = lambda v: "Pending" if v is None else f"{v:.6g}" if isinstance(v,(int,float)) else str(v)
    blocks = []
    def title(t): blocks.append(("title", t))
    def p(t): blocks.append(("p", t))
    def table(headers, rows, widths=None): blocks.append(("table", (headers, rows, widths)))
    def page(): blocks.append(("page", None))
    title("KHU selective sulfoxidation")
    p(f"First homogeneous acridine screen | {c['as_of_date']} | Target delivery: {c['target_delivery_date']}")
    p("CONDITIONAL REVIEW PACKAGE. Engineering screen status: " + c["screen"]["status"].replace("_"," ") + ". This exploratory design has not been experimentally validated or released for laboratory execution.")
    table(["Delivery", "Prepared content"], [["Core budget", f"{ntime} time coordinates x {noxygen} O2 delivery levels ({unique_count} conditions), plus one independent repeat ({len(c['condition_rows'])} runs)."],
         ["Optional comparison", f"Up to {len(c['optional_slots'])} history-omitted nominations. Slots are empty pending a fair, frozen comparison."],
         ["Offline implementation", "Traceable ingestion, coupled calculations, feasibility checks, screen selection, comparison contracts and results templates."],
         ["Current limitations", "Resolve apparatus/gas, sampling, analytical endpoints and material capacity before laboratory release. Detailed pending fields are in campaign.json. No paid model calls or external sharing."]], [110,390])
    blocks.append(("diagram", None))
    p("Diagram basis: KHU slides 5-6, reconciled with the supplied inventory. Dashed boxes denote proposed components; labels show assignment status. The figure gives connection order; it does not assert that pressure protection, calibration or collection arrangements have been verified.")
    p(f"Proposed composition: {chem['substrate']} {chem['substrate_concentration_M']:g} M, {chem['catalyst']} {chem['catalyst_loading_mol_percent']:g} mol% in {chem['solvent']}. Calculated catalyst concentration: {prep['catalyst_concentration_M']*1000:g} mM. Composition status: {chem['status'].replace('_',' ')}. The 24 C batch anchor and air preference remain source evidence; use separately assigned flow temperature and gas.")
    page(); title("Condition table and rationale")
    compact = []
    for i,row in enumerate(c["condition_rows"]):
        target, calc = row["target"], row["calculated"]
        time_label = show(target.get("time_min")) if target.get("time_min") is not None else f"{row['time_coordinate']} pending"
        oxygen_label = show(target.get("oxygen_equivalents")) if target.get("oxygen_equivalents") is not None else f"{row['oxygen_coordinate']} pending"
        compact.append([row["condition_id"], time_label, show(calc.get(target.get("time_basis"))),
                        oxygen_label,show(calc.get("oxygen_equivalents_supplied")),
                        show(calc.get("liquid_flow_mL_min")), show(calc.get("gas_reference_flow_mL_min")), row["repeat_of"] or "-"])
    table(["ID", "Target time min", "Calc. time min", "Target O2 eq", "Supplied O2 eq", "Liquid mL/min", "Gas ref. mL/min", "Repeat of"], compact, [40,55,55,50,60,90,90,60])
    p("Time coordinate basis: " + (cfg["screen"]["time_basis"] or "pending choice of liquid-only space time or no-slip mixture transit estimate") + ". Neither is a measured liquid contact time. Controller settings remain unset until the requested grid is feasible within declared tolerances.")
    order = cfg["order"].get("execution_order")
    p(("Prespecified execution order: " + ", ".join(order) + ". " if order else "The table order is an identifier order; execution order remains pending. ") + f"Candidates enumerate time first and oxygen level within each time. R01 repeats the prespecified {cfg['screen']['repeat_candidate_id']} reference using independent preparation and operation where feasible. Calculated values use quantized controller settings and can differ from requested targets.")
    p(cfg["screen"]["selection_rationale"])
    p("Batch evidence motivates a selectivity-aware screen: the 0.5 M air/EtOH group contains 20% sulfoxide at 2 h, 92% at 6 h and 91% with 4% sulfone at 12 h. These observations vary non-monotonically. Their concentration and batch geometry must be distinguished from the separately proposed flow composition/apparatus; they do not specify flow times. Source entries: 110, 32 and 35, homogeneous workbook.")
    if c["optional_slots"]:
        p("Optional " + "/".join(s["slot_id"] for s in c["optional_slots"]) + " slots reserve comparison capacity; no nominations have been generated. Compare equal first nominations from both arms under the same physical candidate space and settings. Both arms may know the selected standard recipe, with that disclosure recorded. Freeze both before any flow outcomes; convergence is allowed and an overlapping measurement is shared evidence.")
    else:
        p("Optional history comparison is disabled in this configuration; the package reserves core runs only.")
    p("A 3-5 time / one-oxygen-level fallback is configurable if needed. It does not test oxygen sensitivity. Blanks, calibration samples and analytical repeat injections need a separate budget.")
    page(); title("Calculation basis and feed preparation")
    table(["Quantity", "Calculation / meaning"], [
        ["Substrate feed", "nS (mmol/min) = C (mol/L) x QL (mL/min), for one premixed liquid stream."],
        ["Catalyst feed", "nCat = nS x loading(mol%)/100. Additional liquid feeds require an explicit dilution extension."],
        ["Gas supply", "nGas = P_ref x QG_ref / (R x T_ref); R = 0.08314462618 mL bar/(mmol K). nO2 = xO2 x nGas."],
        ["Oxygen equivalents", "nO2 / nS. This is molecular O2 supplied, not uptake or oxygen-atom equivalents."],
        ["Gas at reactor", "QG_est = QG_ref x (P_ref/P_abs) x (T_reactor/T_ref). Ideal gas estimate; all temperatures in K."],
        ["Time indices", "Liquid space time = V/QL. No-slip transit = V/(QL+QG_est). Reference-volume index = V/(QL+QG_ref)."],
        ["Pressure", "1 psi = 0.06894757293 bar. Gauge conversion requires declared ambient pressure. A BPR setpoint approximation is labeled separately from measured pressure."],
        ["Flow realization", "Solve liquid/gas jointly, apply controller grids and recompute time and O2 delivery. Preserve every candidate and rejection."],
    ], [115,385])
    p("Gas dissolution, reaction consumption, phase slip and pressure gradients can change contact time. Liquid holdup/tracer data are absent. No fitted flow kinetics, photon-power scaling or predicted yield is used.")
    p(f"For a final stock volume V mL at the proposed composition: substrate = {prep['substrate_concentration_M']:g} x V mmol; catalyst = {prep['catalyst_concentration_M']:g} x V mmol. Final stock volume (mL): {show(prep['final_stock_volume_mL'])}. Molecular mass, assay/purity and density must come from confirmed chemical identity/supplier records before converting these amounts to weighings or liquid aliquots.")
    if prep["final_stock_volume_mL"] is not None:
        table(["Component", "mmol", "Mass mg", "Liquid aliquot mL"], [[x,show(prep[x+"_mmol"]),show(prep.get(x+"_mass_mg")),show(prep.get(x+"_aliquot_mL"))] for x in ("substrate","catalyst")],[130,100,130,140])
    p(prep["solvent_instruction"])
    p("Material budget = priming feed + liquid flow x (equilibration + collection), plus separately recorded reserve. Flushing uses a separate solvent budget. Repeat preparation is included as its own run. The module also accepts explicit injected feed/carrier volumes for sample loops; it does not infer a steady collection plateau from a finite injection.")
    totals = c["core_material_totals"]
    p(f"Core total feed (mL): {show(totals['feed_volume_mL'])}; substrate (mmol): {show(totals['substrate_mmol'])}; catalyst (mmol): {show(totals['catalyst_mmol'])}; flush solvent (mL): {show(totals['flush_solvent_mL'])}; carrier solvent (mL): {show(totals['additional_carrier_solvent_mL'])}; elapsed time (min): {show(totals['elapsed_min'])}. Reserve and optional nominations are separate. Confirm material availability and daily sample capacity. Prepare the independent repeat stock separately and record its preparation ID.")
    page(); title("Evidence and apparatus reconciliation")
    table(["Evidence", "Treatment"], [["Homogeneous exports", "116 Excel entries and 111 CSV entries become 116 logical entries. The common 111 retain both source representations; entries 112-116 remain version-unconfirmed."],
        ["Heterogeneous exports", "42 entries, kept in a separate system. No pooling with homogeneous outcomes."],
        ["Analytical values", "Preserve 101% and 102% heterogeneous sulfoxide reports and all component sums above 100. Flag, never clip or normalize."],
        ["Slides", "Image tables have separate slide-entry IDs. Homogeneous slide entry 13 matches raw entry 5 as a candidate link, not a raw ID substitution."],
        ["Outcome definitions", "Residual sulfide, sulfoxide, sulfone, other and unknown components stay separate. Product-ratio selectivity excludes other products; entry 112 has 100% selectivity but only 1% sulfoxide."],
        ["Raw mapping scope", "Slides depict thioanisole and crude 1H NMR with 1,3-benzodioxole. Their application to every raw row still needs confirmation."],
    ], [115,385])
    p("The newly supplied khu_inventory.json has August 2026 metadata. Its internal 'validated' label is not confirmation of this study's assembly. The source is preserved in apparatus.json with its hash.")
    for text in [
        "R-series and UV150 450 nm have inventory IDs. The original source documents do not assign this study's channel, installed coil or light power setting. Current assignments are stored separately in apparatus.json.",
        "The inventory MFC is O2-tagged and lists 0.01-10 sccm with 0.01 increments. Those source statements do not establish air calibration, exact reference T/P, pressure basis or a current usable range.",
        "No mixer is catalogued in the supplied JSON, while the newer slides show a T-mixer. Coil diameter descriptions conflict. Bind actual hardware before numerical release.",
        "The JSON lists 1.5/2.5/7/8/16 bar BPRs, while the slides label 20/40/100 psi (1.379/2.758/6.895 bar). They are not interchangeable labels.",
        "UV150 temperature sources differ: the September profile lists 25-80 C; the new JSON lists 35-80 C standard and 10-35 C with additional cooling. The 24 C batch condition needs cooling/temperature confirmation."]:
        p(text)
    page(); title("KHU confirmations and results return")
    table(["Needed", "Details to return"], [["Gas and pressure", "Gas identity/O2 fraction; active MFC calibration and reference T/P; usable range/increment; source pressure; BPR label/basis; sensor location and pressure envelope."],
        ["Installed assembly", "Pump channel; coil ID, material, volume, internal diameter and illuminated length; mixer, check valve, regulator and connections; cooling and actual temperature."],
        ["Feed and sampling", "Continuous feed or loop volume/carrier; equilibration and collection windows; flush/prime/reserve; post-collection handling and analysis timing."],
        ["Analysis and endpoints", "Confirm 1/2/3 identity and analytical basis; sulfone acceptance limit; uncertainties; separate independent repeats from repeat analyses."],
        ["Data version", "Confirm Excel-only 112-116, repeated-condition relationships, and authoritative values for slide/raw discrepancies. Originals remain preserved."],
    ], [115,385])
    p("Return results_template.csv or results_template.json. Proposed fields are prefilled only where known. Enter actual settings separately, then residual sulfide, sulfoxide, sulfone, other/unknown outcomes, method/internal standard, uncertainty, sample ID and operating observations. Blank cells / JSON null mean unknown, not zero or 'no event'. Preserve reported percentages above 100 and explain corrections as amendments.")
    p("Archive contents include canonical history with source hashes and row mappings; the exact configuration and apparatus snapshot; all candidates and rejection reasons; comparison information-boundary plan; feed/material calculations; conditions and results forms. manifest.json verifies every delivered file. CSV files are editable snapshots; JSON is the authoritative structured record.")
    p("Paid model routing remains disabled. A future run needs exact provider/model IDs by role, decoding settings, call/token/revision limits and a total USD budget agreed with the user. No automatic fallback or corpus re-embedding is available in this adapter.")
    p("Source basis: KHU optimization/scope slides 2-7, homogeneous and heterogeneous Excel/CSV records, September inventory workbook/profile, khu_inventory.json, and the labeled correspondence summary. Exact paths, hashes and interpretation issues are in the archive. The inherited baseline and original source files remain unchanged.")
    return blocks


def render_html(campaign, path):
    parts = ['<!doctype html><meta charset="utf-8"><title>KHU first-screen review</title><style>body{font:16px/1.5 system-ui;max-width:1000px;margin:40px auto;color:#153340}h1{font-size:28px}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border-bottom:1px solid #cad8dc;text-align:left;padding:10px;vertical-align:top}th{background:#edf4f6}svg{width:100%;height:auto}.page{page-break-before:always;margin-top:50px}@media print{body{font-size:11px;margin:0}.page{margin:0}}</style>']
    for kind, value in document_blocks(campaign):
        if kind == "title": parts.append(f"<h1>{html.escape(value)}</h1>")
        elif kind == "p": parts.append(f"<p>{html.escape(value)}</p>")
        elif kind == "page": parts.append('<div class="page"></div>')
        elif kind == "diagram": parts.append(apparatus_svg(campaign))
        elif kind == "table":
            headers, rows, _ = value
            parts.append("<table><thead><tr>"+"".join(f"<th>{html.escape(str(v))}</th>" for v in headers)+"</tr></thead><tbody>")
            parts.extend("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in row)+"</tr>" for row in rows)
            parts.append("</tbody></table>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def render_pdf(campaign, path):
    """Optional reportlab renderer; core adapter has no third-party dependency."""
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Flowable
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyKHU", fontName="Helvetica", fontSize=9.1, leading=12.6, spaceAfter=9, textColor=colors.HexColor("#18323e")))
    styles.add(ParagraphStyle(name="CellKHU", fontName="Helvetica", fontSize=8.2, leading=11))
    styles.add(ParagraphStyle(name="TitleKHU", fontName="Helvetica-Bold", fontSize=21, leading=25, spaceAfter=16, textColor=colors.HexColor("#145e6d")))
    def para(text, style="BodyKHU"): return Paragraph(html.escape(str(text)), styles[style])
    class Diagram(Flowable):
        def __init__(self):
            super().__init__()
            self.width, self.height = 500, 140
        def draw(self):
            cv = self.canv
            cv.saveState(); cv.scale(500/810,500/810)
            for _,x,y,w,h,lines in diagram_nodes(campaign):
                cv.setStrokeColor(colors.HexColor("#176b78")); cv.setFillColor(colors.HexColor("#eff6f7")); cv.setDash(4,3)
                cv.roundRect(x,215-y-h,w,h,4,stroke=1,fill=1)
                cv.setFillColor(colors.HexColor("#18323e")); cv.setFont("Helvetica",10)
                for i,line in enumerate(lines): cv.drawCentredString(x+w/2,215-y-18-i*14,line)
            cv.setDash(); cv.setStrokeColor(colors.HexColor("#176b78"))
            for x1,y1,x2,y2 in EDGES:
                cv.line(x1,215-y1,x2,215-y2)
                if x2 != x1:
                    cv.line(x2,215-y2,x2-5,215-y2+3); cv.line(x2,215-y2,x2-5,215-y2-3)
                else:
                    cv.line(x2,215-y2,x2-3,215-y2+5); cv.line(x2,215-y2,x2+3,215-y2+5)
            cv.setDash(3,3); cv.setStrokeColor(colors.grey)
            cv.line(570,146,570,120); cv.line(570,120,470,120); cv.line(470,120,470,90)
            cv.restoreState()
    story = []
    for kind,value in document_blocks(campaign):
        if kind == "title": story.append(para(value,"TitleKHU"))
        elif kind == "p": story.append(para(value))
        elif kind == "page": story.append(PageBreak())
        elif kind == "diagram": story.extend([Spacer(1,6),Diagram(),Spacer(1,6)])
        elif kind == "table":
            headers, rows, widths = value
            t = Table([[para(v,"CellKHU") for v in headers]]+[[para(v,"CellKHU") for v in row] for row in rows], colWidths=widths, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e6f0f3")),("VALIGN",(0,0),(-1,-1),"TOP"),
                ("LINEBELOW",(0,0),(-1,-1),.35,colors.HexColor("#c7d4d8")),("LEFTPADDING",(0,0),(-1,-1),7),
                ("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
            story.extend([t,Spacer(1,12)])
    def footer(canvas, doc):
        canvas.setFont("Helvetica",8); canvas.setFillColor(colors.HexColor("#516d78"))
        canvas.drawString(47,27,"KHU | Conditional review | " + campaign["as_of_date"])
        canvas.drawRightString(A4[0]-47,27,str(doc.page))
    SimpleDocTemplate(str(path), pagesize=A4, rightMargin=47,leftMargin=47,topMargin=38,bottomMargin=43,
                      title="KHU first homogeneous flow-screen review",author="KHU study workspace").build(story,onFirstPage=footer,onLaterPages=footer)


def export_package(root, config, output, apparatus=None, *, pdf=False, verification=None):
    """Exclusive directory creation preserves every version and failed attempt."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        c = build_campaign(root, config, apparatus)
        if verification is not None and (verification.get("status") != "passed" or verification.get("source_snapshot",{}).get("tree_sha256") != c["source_snapshot"]["tree_sha256"]):
            raise ValueError("Verification report failed or describes a different source/configuration snapshot")
        if pdf:
            from importlib.metadata import version
            c["methods"]["pdf_renderer"] = {name:version(name) for name in ("reportlab","Pillow")}
        for name,key in [("campaign.json",None),("history.json","history"),("evidence.json","evidence"),
                         ("apparatus.json","apparatus"),("screen.json","screen"),("configuration.json","configuration"),
                         ("comparison_plan.json","comparison_plan"),("results_template.json","results_template"),
                         ("feed_preparation.json","feed_preparation"),("material_requirements.json","core_material_totals")]:
            write_json(output/name,c if key is None else c[key])
        if verification is not None: write_json(output/"verification.json",verification)
        _write_csv(output/"conditions.csv",c["condition_rows"])
        _write_csv(output/"optional_nomination_slots.csv",c["optional_slots"])
        _write_csv(output/"results_template.csv",results_csv_rows(c))
        (output/"apparatus.svg").write_text(apparatus_svg(c),encoding="utf-8")
        render_html(c,output/"KHU_review.html")
        if pdf: render_pdf(c,output/"KHU_review.pdf")
        write_json(output/"manifest.json",{"schema_version":"khu_package_manifest_v1", "configuration_sha256":c["configuration_sha256"],
            "source_snapshot":c["source_snapshot"], "files":[{"path":p.name,"sha256":file_hash(p),"bytes":p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()],
            "status":c["status"],"experimental_validation":False})
        return c
    except Exception as exc:
        write_json(output/"build_error.json",{"exception":type(exc).__name__,"message":str(exc),"configuration":config})
        raise


def verify_package(output):
    output = Path(output)
    manifest = json.loads((output/"manifest.json").read_text())
    failures = [v["path"] for v in manifest["files"] if not (output/v["path"]).is_file() or file_hash(output/v["path"]) != v["sha256"]]
    expected = {v["path"] for v in manifest["files"]} | {"manifest.json"}
    failures += [f"unmanifested:{p.name}" for p in output.iterdir() if p.name not in expected]
    if failures: raise ValueError("Package integrity failure: " + ", ".join(failures))
    return {"status":"passed","files_verified":len(manifest["files"])}


def zip_package(output):
    output = Path(output)
    verify_package(output)
    target = output.with_suffix(".zip")
    with zipfile.ZipFile(target,"x",compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.iterdir()): archive.write(p, arcname=output.name+"/"+p.name)
    return target
