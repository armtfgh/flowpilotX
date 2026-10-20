"""KHU-specific, question-led pure-O2/sample-loop FlowPilotX adapter.

This offline adapter freezes six engineering proposals, not laboratory release.
Original history, inventory and former air packages are read, never amended.
The existing coupled calculator handles the undiluted part of an injected plug;
it does not simulate dispersion, depletion, uptake or collection fractions.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .corrected_design import SOURCES, _scenario, _solve, _inventory_review
from .candidate_space import R_ML_BAR_PER_MMOL_K
from .history import ingest_history, record_id, _load_private_context
from .campaign_evidence import build_campaign_evidence
from .provenance import digest, file_hash

CONFIG = "configs/khu_flowpilotx_2026-10-15_v1.json"
REPLY = "provenance/khu_lab_reply_reconciliation_2026-10-15_v1.json"
REPLY_ID = "khu:lab-reply:reconciled-2026-10-15"
LITERATURE = "review/literature_2026-10-13_v1/assessment.json"
EDGES = [("carrier", "injector", "liquid"), ("injector", "sample_loop", "liquid"),
         ("sample_loop", "mixer", "liquid"), ("oxygen", "check", "gas"),
         ("check", "mixer", "gas"), ("mixer", "reactor", "gas_liquid"),
         ("reactor", "bpr", "gas_liquid"), ("bpr", "collector", "gas_liquid")]


def _question(index, time, eq, prefix="KX"):
    roles = [
        ("Short exposure / lower oxygen", "Does the short, lower-supply condition retain sulfide or already form appreciable sulfoxide?"),
        ("Short exposure / oxygen contrast", "At the short target transit, how do sulfide, sulfoxide and sulfone change when inlet O2 supply is raised?"),
        ("Middle exposure / lower oxygen", "Relative to the shorter target at lower O2 supply, does more nominal exposure change the three-component distribution?"),
        ("Middle exposure / repeat reference", "What distribution is obtained at the middle target with higher O2 supply, and is it reproducible in a fresh injection?"),
        ("Long exposure / lower oxygen", "At lower O2 supply, does the longest target continue oxidation or leave substantial starting material?"),
        ("Long exposure / upper comparison", "Does combining the longest target with higher O2 supply increase sulfoxide or favor further oxidation?"),
    ]
    role, question = roles[index]
    partner = f"{prefix}{index + (2 if index % 2 == 0 else 0):02d}"
    if prefix != "KX":
        role = f"Alternative {time:g}-minute coordinate / {'lower' if eq == .5 else 'higher'} oxygen"
        question = f"Does the alternative {time:g}-minute target at {eq:g} inlet O2 equivalents help locate the sulfide/sulfoxide/sulfone response between or beyond preferred coordinates?"
    evidence_ids = [record_id("homogeneous", i) for i in [22, 28, 35, 40, 99, 110, 111]] + [REPLY_ID]
    return {"id": f"{prefix}-Q{index+1:02d}", "question": question, "design_role": role,
        "companion_candidate_id": partner,
        "competing_explanations": ["Insufficient oxidation under this exposure and oxygen delivery",
            "Desired sulfoxide accumulates within the sampled exposure window",
            "Further oxidation consumes sulfoxide", "Transport, illumination or plug dilution dominates the comparison"],
        "expected_observations": ["Residual sulfide may remain high, decrease, or vary between collected fractions",
            "Sulfoxide and sulfone may respond differently; report both without normalizing component sums"],
        "measurements": ["Separate residual sulfide, sulfoxide, sulfone and other-product amounts using a confirmed analytical method",
            "Actual liquid/gas flows, gas calibration/reference, pressure, temperature and light setting",
            "Loop volume, injection timestamp, fraction times/volumes and sample dilution/handling"],
        "confounders": ["Matched nominal transit does not match gas fraction, photon dose or oxygen mass transfer",
            "Finite-loop dispersion and front/tail dilution can make effective equivalents differ from central-plug inlet ratios",
            "Actual holdup, gas uptake, dissolved oxygen, vapor and RTD are unmeasured"],
        "decision_rules": [f"Compare all three chemical components with {partner}; treat changes as associations, not isolated oxygen causality",
            "If sulfide persists across the grid, review transport and consider a longer measured follow-up rather than declaring chemistry failure",
            "If sulfone rises, consider a lower-supply/shorter-exposure follow-up after checking analytical uncertainty",
            "Resolve fraction reproducibility and analytical ambiguity before claiming optimization"],
        "evidence_ids": evidence_ids, "evidence_boundary": "Batch records motivate competing hypotheses only; the numerical time/oxygen coordinates are configured engineering choices.",
        "status": "unresolved", "target_time_min": time, "target_oxygen_eq": eq}


def _requirements():
    """Reviewed collaborator requirements are private source inputs."""
    return deepcopy(_load_private_context()["profile_requirements"])


def _inventory(source, scenario):
    result = []
    # Retain the complete catalog, not only the chosen hardware. Source quantities
    # may themselves be assumptions (notably FF-C00); preserve those notes.
    for category, rows in source.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or not row.get("equipment_id"):
                continue
            status = row.get("service_status", "unknown")
            item = {"id": row["equipment_id"], "kind": category, "name": row.get("name"),
                "quantity": row.get("quantity"), "availability": status if status in {"available", "unavailable"} else "unknown",
                "capabilities": deepcopy(row), "limits": [], "source_references": [SOURCES[0]],
                "value_status": "source_reported_not_installation_verified", "actual_assignment": None}
            if row["equipment_id"] == "mfc_ffc00":
                item["quantity_status"] = "source_assumption_one_listed_instrument; laboratory_confirmation_required"
            result.append(item)
    by_id = {x["id"]: x for x in result}
    for role, parameter, unit, lowkey, highkey in [
        ("pump", "liquid_setting", "mL/min", "min_flow_rate_mL_min", "max_flow_rate_mL_min"),
        ("mfc", "oxygen_setting", "STP mL/min", "min_flow_sccm", "max_flow_sccm")]:
        record = scenario["selected_equipment"][role]
        for op, key in [(">=", lowkey), ("<=", highkey)]:
            by_id[record["equipment_id"]]["limits"].append({"parameter": parameter, "operator": op,
                "value": record[key], "unit": unit, "status": "reported", "source_reference": SOURCES[0]})
    return {"schema_version": "flowpilotx-inventory-v1", "items": result,
        "assignment_status": "proposed_subset; installed identities unconfirmed",
        "unassigned_items": ["sample_loop", "injection_valve", "oxygen_regulator", "pressure_sensor", "collector"],
        "valve_note": "V-101L/V-100T are listed manual switching valves, not evidence of an installed six-port injection valve or calibrated sample loop."}


def _diagram(candidate, scenario, config):
    p, c = candidate["parameters"], candidate["calculated"]
    e = config["selected_equipment"]
    def node(identity, kind, label, params, eid=None):
        return {"op_id": identity, "op_type": kind, "label": label, "parameters": params,
            "inventory_item_id": eid, "actual_assignment_confirmed": None,
            "assignment_status": "proposed_from_available_inventory" if eid else "unresolved_equipment_assignment"}
    nodes = [
        node("carrier", "pump", "EtOH carrier", {"instrument_name": "R-series HPLC", "contents": ["EtOH carrier"],
            "solvent": "EtOH", "flow_rate_mL_min": p["liquid_mL_min"]}, e["pump_id"]),
        node("injector", "injection_valve", "Injection valve\nAssignment / port map pending", {"mode": "INJECT schematic only", "actual_port_map": None}),
        node("sample_loop", "sample_loop", "Sample loop: volume unknown\n0.25 M thioanisole\nAcridine 3 mol% in EtOH", {"volume_mL": None,
            "actual_injected_volume_mL": None, "sample_concentration_M_proposed": .25}),
        node("oxygen", "mfc", "Pure O2 / FF-C00", {"gas_flow_sccm": p["oxygen_STP_mL_min"], "oxygen_STP_mL_min": p["oxygen_STP_mL_min"],
            "inlet_oxygen_equivalents": p["oxygen_eq"], "reference_temperature_K": scenario["gas_reference"]["temperature_K"],
            "reference_pressure_bar_abs": scenario["gas_reference"]["pressure_bar_abs"], "oxygen_mole_fraction_assumed": 1.}, e["mfc_id"]),
        node("check", "check_valve", "Gas check valve\n1 bar cracking differential", {"instrument_name": "CV-3301"}, e["check_valve_id"]),
        node("mixer", "t_mixer", "Gas-liquid T-mixer", {"instrument_name": "P-713"}, e["mixer_id"]),
        node("reactor", "photoreactor", "Photoreactor", {"instrument_name": "PFA coil", "light_instrument_name": "UV-150",
            "temperature_C": scenario["temperature_C"], "wavelength_nm": 450., "volume_mL": scenario["reactor_volume_mL"], "ID_mm": scenario["reactor_ID_mm"],
            "target_mixture_transit_min": p["target_time_min"], "estimated_mixture_transit_min": p["transit_min"],
            "actual_liquid_contact_time_min": None, "light_inventory_id": e["light_id"]}, e["coil_id"]),
        node("bpr", "bpr", "Fixed BPR", {"pressure_bar": p["pressure_bar_gauge_assumed"], "proposed_label_basis": "gauge",
            "modeled_reactor_pressure_bar_abs": p["pressure_bar_abs_assumed"], "actual_pressure_bar_abs": None}, e["bpr_id"]),
        node("collector", "collector", "Fraction collection / vent\nAssignment + timing pending", {"actual_collection_window": None})]
    return {"schema_version": "khu_flowpilotx_display_topology_v1", "topology_id": candidate["id"],
        "unit_operations": nodes, "streams": [{"stream_id": f"s{i:02d}", "from_op": a, "to_op": b,
            "stream_type": phase, "connection_type": "process"} for i,(a,b,phase) in enumerate(EDGES,1)],
        "source_parameters_sha256": digest(p), "display_quantities": deepcopy(p),
        "scope": "INJECT-state process schematic, not a complete P&ID or executable valve sequence. LOAD/waste routing, gas regulator/sensor, loop size and fraction collection remain unresolved.",
        "gas_source": {"identity": "O2", "status": "laboratory_reported_requirement", "regulator_inventory_id": None, "actual_purity": None},
        "actual_conditions": None, "actual_outcomes": None}


def _materials(config):
    chem = config["chemistry"]
    concentration = chem["substrate_concentration_M"]
    catalyst_M = concentration * chem["catalyst_loading_mol_percent"] / 100
    return {"status": "scalable_preparation_proposal; actual_purities_and_loop_volume_unknown",
        "sample_stock": {**chem, "catalyst_concentration_M": catalyst_M,
            "formula_substrate_mmol": "C_M * final_stock_volume_mL",
            "formula_catalyst_mmol": "substrate_mmol * catalyst_mol_percent / 100",
            "formula_weighed_mass_mg": "amount_mmol * molecular_mass_g_mol / actual_purity_fraction",
            "illustrative_final_volume_mL": 10., "illustrative_substrate_mmol": concentration * 10,
            "illustrative_catalyst_mmol": catalyst_M * 10,
            "illustrative_substrate_pure_equivalent_mg": concentration * 10 * 124.20,
            "illustrative_catalyst_pure_equivalent_mg": catalyst_M * 10 * 179.22,
            "preparation": "Dissolve confirmed weighed materials in EtOH and make to the selected final solution volume. Confirm solubility and actual purity. The 10mL example is stock preparation, not installed loop volume or campaign requirement.",
            "molecular_masses_g_mol": {"thioanisole": 124.20, "acridine": 179.22},
            "mass_sources": ["https://www.sigmaaldrich.com/SG/en/product/sial/t28002", "https://www.sigmaaldrich.com/US/en/product/sigma/01640"],
            "mass_source_check_date": "2026-10-13", "actual_purity_fractions": None},
        "loop_accounting": {"actual_loop_volume_mL": None, "actual_delivered_sample_mL": None,
            "nominal_substrate_mmol_if_loop_completely_filled": f"{concentration:g} * V_loop_mL",
            "nominal_catalyst_mmol_if_loop_completely_filled": f"{catalyst_M:g} * V_loop_mL",
            "nominal_undispersed_injection_duration_min": "V_loop_mL / Q_L_mL_min",
            "actual_fraction_collection_times": None, "actual_sample_recovery": None,
            "warning": "Finite plug fronts/tails dilute. Nominal holdup and V_loop/Q_L do not establish a constant-composition plateau or collection window."},
        "gas_total_formula": "P_ref * Q_ref * gas_on_time_min / (R * T_ref); gas_on_time unknown",
        "carrier_and_flush_volumes_mL": None, "campaign_total_material_requirement": None,
        "available_thioanisole": deepcopy(_load_private_context()["available_thioanisole"])}


def observation_contract():
    """Actual controls and sample context, never requested/derived coordinates.

    A different gas reference is preserved but excluded from this adapter's
    numerical comparison until an explicit reference conversion is supplied.
    Unknown observations remain null in every template.
    """
    units = {"carrier_flow_mL_min": "mL/min", "oxygen_flow_reference_mL_min": "reference mL/min",
        "gas_reference_temperature_K": "K", "gas_reference_pressure_bar_abs": "bar absolute",
        "reactor_pressure_bar_abs": "bar absolute", "temperature_C": "degC", "concentration_M": "mol/L",
        "catalyst_loading_mol_percent": "mol%", "loop_volume_mL": "mL", "injected_sample_volume_mL": "mL",
        "fraction_start_min": "min after injection", "fraction_end_min": "min after injection",
        "fraction_volume_mL": "mL", "oxygen_purity": "mole fraction"}
    types = {**{k:"number" for k in units}, "MFC_calibration":"string", "gas_identity":"string", "feed_mode":"string"}
    return {"schema_version":"flowpilotx-observation-contract-v1", "required_actual_conditions":list(types),
        "parameter_mapping":{"liquid_mL_min":"carrier_flow_mL_min", "oxygen_STP_mL_min":"oxygen_flow_reference_mL_min",
            "temperature_C_proposed":"temperature_C"}, "units":units, "types":types,
        "comparability_requirements":[{"parameter":"gas_reference_temperature_K","operator":"==","value":273.15},
            {"parameter":"gas_reference_pressure_bar_abs","operator":"==","value":1.01325},
            {"parameter":"concentration_M","operator":"==","value":0.25},
            {"parameter":"catalyst_loading_mol_percent","operator":"==","value":3.0},
            {"parameter":"gas_identity","operator":"==","value":"O2"},
            {"parameter":"feed_mode","operator":"==","value":"sample_loop"}],
        "actual_conditions_template":{k:None for k in types},
        "mapping_basis":"Compare measured liquid/MFC settings at the stated confirmed reference and measured temperature. Shared concentration/loading/pressure and injection/fraction context must be recorded but are not target coordinates.",
        "excluded_from_measurement_requirements":["target_time_min","target_oxygen_eq","transit_min","oxygen_eq","liquid_only_space_time_min","pressure_bar_gauge_assumed","pressure_bar_abs_assumed"],
        "mismatched_reference_policy":"Preserve the observation, exclude from numerical comparison; no implicit gas reference conversion or assumption copied into actual measurements."}


def build_profile(project_root: Path | str, config_path: Path | str | None = None) -> dict:
    root = Path(project_root).resolve()
    path = Path(config_path) if config_path is not None else root / CONFIG
    if not path.is_absolute():
        path = root / path
    config = json.loads(path.read_text())
    digest(config)
    if config["feed"]["mode"] != "sample_loop" or config["assumptions"]["gas_identity"] != "O2":
        raise ValueError("This version requires the KHU pure-O2/sample-loop constraints")
    if config["chemistry"] != {"substrate": "thioanisole", "solvent": "EtOH", "substrate_concentration_M": .25,
        "catalyst": "acridine", "catalyst_loading_mol_percent": 3.}:
        raise ValueError("This evidence-linked version fixes the shared KHU recipe; make an explicit adapter revision for other chemistry")
    times, eqs = config["target_mixture_transit_min"], config["target_inlet_oxygen_equivalents"]
    if times != [5.,10.,20.] or eqs != [.5,1.]:
        raise ValueError("This frozen six-question version requires its documented 3x2 grid")
    source = json.loads((root / SOURCES[0]).read_text())
    scenario = _scenario(source["lab_inventory"], config)
    history = ingest_history(root)
    reply = json.loads((root / REPLY).read_text())
    candidates, questions, attempts = [], [], []
    targets = [("KX",i,t,e) for i,(t,e) in enumerate((t,e) for t in times for e in eqs)]
    alternate_times = config.get("alternative_pool_mixture_transit_min", [])
    if alternate_times not in ([], [3.,7.5,15.]):
        raise ValueError("Alternative time pool requires an explicit question/configuration revision")
    targets += [("KXA",i,t,e) for i,(t,e) in enumerate((t,e) for t in alternate_times for e in eqs)]
    for prefix, i, time, eq in targets:
        row = _solve(scenario, config, time, eq, f"{prefix}{i+1:02d}")
        attempts.append(deepcopy(row))
        if row["calculated"] is None:
            raise ValueError(f"Configured KHU grid infeasible: {row['condition_id']} {row['rejection_reasons']}")
        q = _question(i, time, eq, prefix)
        questions.append(q)
        c = row["calculated"]
        parameters = {"target_time_min": time, "target_oxygen_eq": eq,
            "liquid_mL_min": c["liquid_flow_mL_min"], "oxygen_STP_mL_min": c["oxygen_reference_volume_flow_mL_min"],
            "oxygen_eq": c["oxygen_equivalents_supplied"], "transit_min": c["no_slip_mixture_transit_min"],
            "liquid_only_space_time_min": c["liquid_only_space_time_min"], "coil_mL": scenario["reactor_volume_mL"],
            "pressure_bar_gauge_assumed": scenario["pressure"]["label_value_bar"],
            "pressure_bar_abs_assumed": scenario["pressure"]["reactor_pressure_bar_abs_assumed"],
            "temperature_C_proposed": scenario["temperature_C"], "gas_identity": "O2", "feed_mode": "sample_loop",
            "liquid_setting": {"value": c["liquid_flow_mL_min"], "unit": "mL/min"},
            "oxygen_setting": {"value": c["oxygen_reference_volume_flow_mL_min"], "unit": "STP mL/min"},
            "loop_volume_mL": None, "actual_pressure_bar_abs": None}
        candidate = {"id": row["condition_id"], "candidate_id": row["condition_id"], "parameters": parameters,
            "feasible": True, "rejection_reasons": [], "question_ids": [q["id"]], "evidence_ids": q["evidence_ids"],
            "curiosity": "Compare competing oxidation and transport explanations; no outcome probability assigned",
            "engineering_status": "conditional", "substrate_id": "thioanisole", "question": q["question"],
            "design_role": q["design_role"], "companion_candidate_id": q["companion_candidate_id"],
            "rationale": f"{q['design_role']}: target {time:g}min and {eq:g} inlet O2 equivalents; paired comparison with {q['companion_candidate_id']}. {q['evidence_boundary']}",
            "calculated": c, "target": row["target"], "proposed_setpoints": row["proposed_setpoints"],
            "actual": row["actual"], "calculation_audit": {k: row[k] for k in ["ideal_unquantized", "quantization_search", "realized_time_relative_error", "realized_oxygen_relative_error"]},
            "inventory_ids": list(config["selected_equipment"].values()),
            "value_status": {"parameters": "proposed and calculated; explicit assumed pressure/STP", "actual": "unknown", "outcomes": "unmeasured"},
            "diagram_url": f"diagrams/{row['condition_id']}.svg"}
        candidate["diagram"] = _diagram(candidate, scenario, config)
        candidates.append(candidate)
    alternatives = []
    for coil in ["coil_row3", "coil_row5"]:
        altconfig = deepcopy(config)
        altconfig["selected_equipment"]["coil_id"] = coil
        altscenario = _scenario(source["lab_inventory"], altconfig)
        altrows = [_solve(altscenario, altconfig, t,e,f"ALT-{coil}-{i+1:02d}") for i,(t,e) in enumerate((t,e) for t in times for e in eqs)]
        alternatives.append({"equipment_id": coil, "configuration": altconfig, "configuration_sha256": digest(altconfig),
            "scenario": altscenario, "attempts": altrows, "selected": False,
            "disposition": "feasible_not_selected" if all(x["calculated"] for x in altrows) else "not_all_targets_grid_feasible",
            "reason": config["rationale"]["coil_choice"], "chemical_rejection": False})
    records = [deepcopy(r) for system in ["homogeneous", "heterogeneous"] for r in history["systems"][system]["records"]]
    evidence = [{"id": r["record_id"], **r, "use_scope": "homogeneous_batch_hypotheses_only" if r["record_id"].startswith("khu:homogeneous:") else "heterogeneous_later_phase_separate"} for r in records]
    evidence.append({"id": REPLY_ID, "kind": "laboratory_reported_constraints", "source": REPLY, "record": reply["reported_requirements"], "status": "reported_not_measured_by_software"})
    literature = json.loads((root / LITERATURE).read_text())
    literature_evidence = {"id": "khu:literature:review-2026-10-13", "source": LITERATURE,
        "status": "primary_literature_context_not_acridine_flow_kinetics", "record": literature,
        "interpretation": "Related minute-scale photooxidations and the reported estimated21min PDI-bead thioanisole example motivate a longer diagnostic point. Different catalysts, geometry and time bases prevent quantitative transfer. The prior review's air/2-5-10min current-study section is historical, not this profile's configuration."}
    evidence.append(literature_evidence)
    compact_history = build_campaign_evidence(root)["informed_history_payload"]
    compact_history.pop("recorded_factor_comparisons")  # full ledger remains in historical archive/source; no raw row removed
    compact_raw = compact_history["raw_records"]
    derived_index = compact_raw["row_columns"].index("calculated_outcomes")
    compact_raw["row_columns"].pop(derived_index)
    for compact_row in compact_raw["rows"]:
        compact_row.pop(derived_index)
    compact_raw.pop("calculated_outcome_columns")
    compact_raw["model_projection_reason"] = (
        "All116 reported condition/outcome rows, source pointers, codebooks and flags remain. "
        "Redundant derived conversion/selectivity/component sums are omitted only from model context "
        "to fit the explicit local context limit; full-precision derived and original values remain in history_archive.")
    model_literature = {"id": literature_evidence["id"], "source": LITERATURE,
        "status": literature_evidence["status"], "interpretation": literature_evidence["interpretation"],
        "sources": [{k:s[k] for k in ["id","doi","findings","time_basis","relevance"]} for s in literature["sources"]]}
    model_evidence = {"homogeneous_batch_evidence": compact_history,
        "records": [deepcopy(evidence[-2]), model_literature],
        "coverage": "All116 canonical homogeneous rows occur once in the column-described raw_records arrays, with citation ID format and source pointers. The42 heterogeneous laboratory rows are archived separately and excluded here."}
    source_hashes = {p: file_hash(root / p) for p in [*SOURCES, REPLY, LITERATURE, "src/khu_sulfoxidation/corrected_design.py", "src/khu_sulfoxidation/flowpilotx_profile.py"]}
    source_hashes[str(path.relative_to(root)) if path.is_relative_to(root) else str(path)] = file_hash(path)
    source_hashes.update({s["path"]: s["sha256"] for s in history["sources"]})
    return {"schema_version": "khu-flowpilotx-profile-v1", "name": "KHU homogeneous acridine / pure oxygen / sample-loop exploration",
        "protocol_text": _load_private_context(root)["profile_protocol_text"],
        "operating_context": {"laboratory_reply": deepcopy(reply["reported_requirements"]),
            "context_status": "source_reported; not newly measured", "batch_to_flow_boundary": _load_private_context(root)["batch_to_flow_boundary"],
            "shared_confounders": questions[0]["confounders"], "required_measurements": questions[0]["measurements"],
            "possible_observations": questions[0]["expected_observations"], "feedback_rules": questions[0]["decision_rules"]},
        "profile_id": config["design_id"], "configuration": config, "configuration_sha256": digest(config),
        "status": "conditional_review_package", "laboratory_released": False, "experimental_validation": False, "paid_model_calls": 0,
        "chemistry": deepcopy(config["chemistry"]), "objective": {"mode": "explore", "metric": "sulfoxide_pct", "direction": "maximize", "unit": "%",
            "separate_components": ["residual_sulfide_pct", "sulfoxide_pct", "sulfone_pct", "other_products_pct"], "sulfone_limit_pct": None},
        "inventory": _inventory(source["lab_inventory"], scenario), "source_inventory": source,
        "model_inventory": {"selected": {role:{"id":r["equipment_id"],"name":r.get("name"),"reported_quantity":r.get("quantity"),
                "selection_status":"proposed; actual assignment unknown"} for role,r in scenario["selected_equipment"].items()},
            "controllers": scenario["controllers"], "reactor_volume_mL":scenario["reactor_volume_mL"],"reactor_ID_mm":scenario["reactor_ID_mm"],
            "temperature_C_proposed":scenario["temperature_C"],"light_nm_reported":450.,"module_capacity_mL_reported":10.,
            "inventory_projection_reason":"Selected assembly IDs and decisive controller/geometry limits only in model context; full source catalog, capabilities and all alternatives remain in source_inventory/inventory_review.",
            "unresolved": ["loop", "injection valve/port map", "sensor/regulator", "collection", "oxygen calibration/pressure basis"],
            "coil_comparison": [{"coil_id": a["equipment_id"], "disposition": a["disposition"], "reason": a["reason"]} for a in alternatives]},
        "model_candidates": [{k:c[k] for k in ["id","parameters","feasible","rejection_reasons","question_ids","evidence_ids","engineering_status","companion_candidate_id","inventory_ids"]} for c in candidates],
        "model_questions": [{k:q[k] for k in ["id","question","design_role","companion_candidate_id","evidence_ids","evidence_boundary"]} for q in questions],
        "inventory_review": _inventory_review(source["lab_inventory"], scenario), "scenario": scenario,
        "constraints": {"gas": "O2", "feed": "sample_loop", "fixed_recipe": config["chemistry"], "grid": config["tolerances"], "maximum_unique_frozen_candidates": len(candidates)},
        "requirements": _requirements(), "candidates": candidates, "questions": questions,
        "observation_contract": observation_contract(),
        "default_selected_candidate_ids": [c["id"] for c in candidates if c["id"].startswith("KX") and not c["id"].startswith("KXA")],
        "selection_policy": "Default six is the complete5/10/20-minute by0.5/1.0-equivalent grid. Six additional3/7.5/15-minute alternatives permit genuine choice from the frozen pool. Prefer complementary matched oxygen pairs when selecting six, but require rationale rather than forcing different choices. One topology, sequential conditions; repeated proposals do not count as unique designs.",
        "optional_repeat": {"id": "KX-R01", "repeat_of": "KX04", "counts_as_unique_design": False, "status": "proposed_optional",
            "preparation": "Fresh independently prepared/re-established stock, loading/injection and operation as agreed with laboratory", "execution_order": None},
        "candidate_attempts": attempts, "engineering_alternatives": alternatives,
        "evidence": evidence, "model_evidence": model_evidence, "history_archive": history,
        "evidence_coverage": {"homogeneous_canonical": 116, "homogeneous_xlsx": 116, "homogeneous_csv": 111,
            "heterogeneous_canonical": 42, "heterogeneous_xlsx": 42, "heterogeneous_csv": 42,
            "systems_pooled": False, "overlapping_exports_double_counted": False,
            "values_above_100_preserved": True, "batch_duration_fit_as_flow_kinetics": False,
            "source_representations": len(history["source_to_record"])},
        "information_boundary": "All 116 homogeneous batch records are available as hypothesis evidence, including workbook-only entries112–116 and anomalies. All42 heterogeneous records remain separately archived for later phase. Numerical grid follows explicit engineering choices and the lab reply, not inferred flow kinetics. No flow outcomes exist.",
        "history_omitted_request_fixture": {"status": "not_run; no independent nominations fabricated", "requested_count": 2,
            "common_inputs": ["shared recipe (disclosed history-selected starting composition)", "same candidate pool", "same inventory/constraints/lab reply", "same objective/model settings/budget when authorized"],
            "withheld": ["batch optimization rows/outcomes/rankings", "history-derived question explanations", "informed nominations"],
            "requires": "Freeze independent nominations and informed nominations before flow outcomes; allow convergence; compare two versus two under matched evaluation opportunities"},
        "materials": _materials(config),
        "calculation_basis": {"R": R_ML_BAR_PER_MMOL_K, "R_units": "mL bar / (mmol K)",
            "substrate_mmol_min": "C_M * Q_L (central undiluted plug only)",
            "oxygen_mmol_min": "xO2 * P_ref * Q_ref / (R*T_ref)",
            "inlet_oxygen_eq": "oxygen_mmol_min / (C_M*Q_L); independent of reactor pressure",
            "gas_reactor_mL_min": "Q_ref*(P_ref/P_abs)*(T_reactor/T_ref)",
            "nominal_transit_min": "V_coil/(Q_L+Q_gas_reactor); ideal gas/no slip/uniform pressure/constant inlet-gas approximation",
            "liquid_only_space_time_min": "V_coil/Q_L; bookkeeping index, not residence time",
            "nominal_liquid_holdup_mL": "V_coil*Q_L/(Q_L+Q_gas_reactor); not a measured holdup or sufficient plug volume",
            "quantization": "Existing coupled solver enumerates every grid pair within analytic time/equivalent tolerance bounds; all post-quantization quantities are recalculated. Each per-liquid grid interval/count/best pair is archived.",
            "finite_plug_limit": "Central undiluted segment hypothetical; actual plug shape, liquid RTD, O2 uptake, collection, front/tail dilution and gas-on duration are unknown."},
        "source_references": [{"path": p, "sha256": h} for p,h in source_hashes.items()], "source_hashes": source_hashes,
        "limitations": ["Reported inventory values and laboratory statements are not installed-assignment measurements.",
            "Installed O2 calibration, gas purity/reference basis and usable accuracy are unconfirmed; pure O2 x=1 is the calculation idealization.",
            "1.5bar gauge and uniform2.51325bar absolute are assumptions, not measured pressure or confirmed BPR label basis.",
            "5/10/20min are nominal mixture-transit estimates; actual liquid contact time, elapsed experimental duration and collection windows are unknown.",
            "No-slip modeled holdup does not establish a usable finite-plug plateau. Confirm loop geometry and fraction/dispersion behavior.",
            "6 conditions share one topology; not six simultaneously available assemblies. Optional repeat is separate.",
            "No yield predictions, fitted flow kinetics, laboratory execution or release is claimed."]}
