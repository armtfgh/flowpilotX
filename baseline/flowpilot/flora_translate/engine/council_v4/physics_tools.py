"""Bounded tool dispatcher for existing council reviewers, independent of provider.

The model requests a named tool in JSON. Python validates the request, executes
the simulator, archives the trace, and returns the result to the same reviewer.
No model-supplied pressures, hardware or physical parameters are accepted.
"""
from copy import deepcopy
import json
from pathlib import Path

from flora_translate.engine.council_v4.backflow import candidate_assessment, oxygen_option
from flora_translate.gas_settings import gas_setting_supported
from flora_translate.engine.council_v4.transient_flow import (
    HydraulicNetwork, LIMITATIONS, SCENARIOS, TransientProfile, VERSION,
    fingerprint, simulate,
)

TOOL_NAME = "simulate_flow_transients"
REVIEWERS = {"DrFluidics", "DrSafety"}


def network_from_candidate(row, assessment, inventory, profile):
    """Reject unsupported plumbing instead of silently projecting it to two pipes."""
    prop = row["proposal"]
    stages = prop.get("stage_parameters", [])
    liquid = [s for s in prop.get("streams", []) if s["phase"] == "liquid"]
    gas = [s for s in prop.get("streams", []) if s["phase"] == "gas"]
    if (len(stages) != 2 or [s.get("stage_number") for s in stages] != [1, 2]
            or len(liquid) != 1 or len(gas) != 1 or liquid[0].get("introduction_stage") != 1
            or gas[0].get("introduction_stage") != 2):
        raise ValueError("Prototype supports exactly two serial coil stages, one liquid feed at Stage 1 and one gas feed at Stage 2.")
    branches = assessment.get("branches", [])
    lb = [b for b in branches if b["branch_phase"] == "liquid"]
    gb = [b for b in branches if b["branch_phase"] == "gas"]
    if len(lb) != 1 or len(gb) != 1 or lb[0]["junction_id"] != gb[0]["junction_id"]:
        raise ValueError("A single resolved gas/liquid mixing junction is required")
    if lb[0]["valves_before_exposed_operation"]:
        raise ValueError("Installed liquid valve dynamics/leakage are unspecified; do not replace them with an ideal valve.")
    valves = gb[0]["valves_before_exposed_operation"]
    if len(valves) != 1 or not valves[0]["direction_declared_forward"] or valves[0]["cracking_pressure_bar"] is None:
        raise ValueError("This prototype requires a declared forward gas check valve with a known cracking pressure")
    if any(s.get("reactor_segments") or s.get("serial_reactor_train") for s in stages):
        raise ValueError("Serial sub-trains need an explicit hydraulic network; not supported by the reduced tool")
    for stage in stages:
        reactor = next((r for r in inventory.reactors if r.equipment_id == stage.get("reactor_equipment_id")), None)
        if reactor is None or reactor.type.lower() != "coil":
            raise ValueError("Hydraulic geometry must be an assigned circular coil, not a packed bed, chip or unspecified reactor")
    pump = next((p for p in inventory.pumps if p.equipment_id == liquid[0].get("pump_equipment_id")), None)
    mfc = next((g for g in inventory.gas_hardware if g.equipment_id == gas[0].get("pump_equipment_id")), None)
    if pump is None or mfc is None:
        raise ValueError("Liquid pump and MFC must be assigned to the inventory")
    if pump.max_pressure_bar is None or profile.pump_pressure_limit_bar_g > pump.max_pressure_bar:
        raise ValueError("Explicit assumed pump cutoff exceeds its inventory rating or the rating is unknown")
    if stages[0]["temperature_C"] != stages[1]["temperature_C"]:
        raise ValueError("Prototype requires equal stage temperatures; temperature-dependent branch properties are not available")
    network = HydraulicNetwork(liquid_feed_mL_min=liquid[0]["flow_rate_mL_min"],
        gas_feed_STP_mL_min=gas[0]["gas_flow_sccm"],
        upstream_volume_mL=stages[0]["reactor_volume_mL"], upstream_id_mm=stages[0]["d_mm"],
        downstream_volume_mL=stages[1]["reactor_volume_mL"], downstream_id_mm=stages[1]["d_mm"],
        temperature_C=stages[1]["temperature_C"], BPR_bar_g=prop["BPR_bar"],
        gas_valve_cracking_bar=valves[0]["cracking_pressure_bar"])
    roles = {"liquid_branch": {"equipment_id": pump.equipment_id, "type": "liquid_pump",
                               "upstream_reactor_operation": lb[0]["exposed_operation_id"]},
             "gas_branch": {"equipment_id": mfc.equipment_id, "type": "MFC",
                            "check_valve_id": valves[0]["equipment_id"]}}
    return network, gas[0], mfc, roles


class PhysicsToolSession:
    def __init__(self, pool, batch, plan, inventory, intake, profile, archive):
        self.pool = deepcopy(pool)
        self.profile = TransientProfile.model_validate(profile)
        self.inventory = inventory
        self.option = oxygen_option(plan, inventory, intake)
        self.archive = Path(archive) / "physics_tools"
        self.archive.mkdir(parents=True, exist_ok=False)
        self.cache = {}
        self.networks = {}
        self.inputs = []
        self.assessments = []
        self.calls = []
        for row in self.pool:
            assessment = row.get("backflow_assessment") or candidate_assessment(row, batch, plan, inventory)
            self.assessments.append(assessment)
            try:
                n, gas, mfc, roles = network_from_candidate(row, assessment, inventory, self.profile)
                self.networks[row["candidate_id"]] = n, gas, mfc
                self.inputs.append({"candidate_id": row["candidate_id"], "network": n.model_dump(),
                                    "equipment_roles": roles, "status": "supported_unvalidated"})
            except (ValueError, KeyError) as exc:
                self.inputs.append({"candidate_id": row["candidate_id"], "status": "not_assessable", "reason": str(exc)})
        self._write("inputs.json", {"schema_version": VERSION, "profile": self.profile.model_dump(),
                                   "candidates": self.inputs, "oxygen_option": self.option,
                                   "limitations": LIMITATIONS})

    def _write(self, name, value):
        path = self.archive / name
        path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        return str(path)

    def manifest(self):
        return {"enabled": True, "schema_version": VERSION, "archive_path": str(self.archive),
                "profile": self.profile.model_dump(), "inputs": self.inputs,
                "tool_calls": self.calls, "results": list(self.cache.values()),
                "backflow_probability": None, "laboratory_execution_status": "review_required",
                "design_modification_applied": False, "limitations": LIMITATIONS}

    def execute(self, role, request):
        if role not in REVIEWERS:
            raise ValueError("Transient tool is assigned only to DrFluidics and DrSafety")
        if not isinstance(request, dict) or set(request) != {"tool_name", "candidate_ids"} or request["tool_name"] != TOOL_NAME:
            raise ValueError("Unknown tool or unapproved tool arguments")
        ids = request["candidate_ids"]
        allowed = {r["candidate_id"] for r in self.pool}
        if (not isinstance(ids, list) or not ids or len(ids) > 12 or any(type(i) is not int or i not in allowed for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("Tool candidate IDs must be unique integers from the frozen council pool")
        if role == "DrFluidics" and set(ids) != allowed:
            raise ValueError("DrFluidics must screen all 12 candidates, not selectively favorable ones")
        if any(c["role"] == role for c in self.calls):
            raise ValueError("One bounded batch tool call per reviewer is allowed")
        results = [self._candidate(i) for i in sorted(ids)]
        result_id = fingerprint({"profile": self.profile.model_dump(), "results": results})
        response = {"tool_name": TOOL_NAME, "result_id": result_id, "candidate_results": results,
                    "assumption_profile": self.profile.model_dump(),
                    "profile_provenance": self.profile.provenance, "limitations": LIMITATIONS,
                    "backflow_probability": None, "laboratory_execution_status": "review_required"}
        path = self._write(f"{role}_tool_result.json", response)
        self.calls.append({"role": role, "request": deepcopy(request), "result_id": result_id, "response_path": path})
        return response

    def _candidate(self, candidate_id):
        if candidate_id in self.cache:
            return self.cache[candidate_id]
        if candidate_id not in self.networks:
            result = next(deepcopy(r) for r in self.inputs if r["candidate_id"] == candidate_id)
            self.cache[candidate_id] = result
            return result
        n, gas, mfc = self.networks[candidate_id]
        variants = {"as_designed": n}
        availability = {"as_designed": "unchanged inventory; dynamics unverified"}
        oxygen_flow = n.gas_feed_STP_mL_min * gas["gas_reagent_mole_fraction"]
        if (self.option["eligible"] and mfc.min_flow_sccm <= oxygen_flow <= mfc.max_flow_sccm
                and gas_setting_supported(mfc, oxygen_flow)):
            variants["pure_oxygen_equal_molar"] = n.model_copy(update={"gas_feed_STP_mL_min": oxygen_flow})
            availability["pure_oxygen_equal_molar"] = "same delivered O2 molar feed; oxygen supply/service and chemistry unapproved"
        else:
            availability["pure_oxygen_equal_molar"] = "not simulated: gas identity constraint, capability, MFC range or setting grid prevents this exact equal-dose alternative"
        variants["hypothetical_liquid_check"] = n.model_copy(update={
            "liquid_valve_cracking_bar": self.profile.hypothetical_liquid_valve_cracking_bar,
            "liquid_valve_reverse_leak_mL_min_bar": self.profile.hypothetical_liquid_valve_reverse_leak_mL_min_bar})
        availability["hypothetical_liquid_check"] = "hypothetical liquid-branch valve, not assigned or service-verified; zero leakage is an idealization if specified"
        rows = []
        # An explicit deterministic sensitivity bracket, not a probability distribution.
        profiles = {"entered_profile": self.profile,
                    "one_tenth_gas_plenum": TransientProfile.model_validate({**self.profile.model_dump(),
                        "gas_plenum_mL": self.profile.gas_plenum_mL / 10})}
        for variant, network in variants.items():
            for sensitivity, profile in profiles.items():
                for scenario in SCENARIOS:
                    simulation = simulate(network, profile, scenario)
                    evidence_id = f"C{candidate_id}/{variant}/{sensitivity}/{scenario}"
                    path = self._write(evidence_id.replace("/", "__") + ".json", {
                        "evidence_id": evidence_id, "network": network.model_dump(), "profile": profile.model_dump(), **simulation})
                    summary = {k: v for k, v in simulation.items() if k not in {"trace", "limitations", "resistances_bar_min_mL"}}
                    summary["MFC_outlet_rating_screen"] = {
                        "declared_max_pressure_bar": mfc.max_pressure_bar,
                        "assumed_gauge_basis_requires_confirmation": True,
                        "exceeded_under_assumptions": simulation.get("peak_gas_plenum_bar_g", 0) > mfc.max_pressure_bar
                            if mfc.max_pressure_bar is not None and "peak_gas_plenum_bar_g" in simulation else None}
                    summary.update(evidence_id=evidence_id, variant=variant, sensitivity=sensitivity, artifact_path=path)
                    rows.append(summary)
        result = {"candidate_id": candidate_id, "status": "conditional_model_results", "alternatives": availability,
                  "gas_inlet_STP_mL_min": {k: v.gas_feed_STP_mL_min for k, v in variants.items()},
                  "alternative_parameters": {k: {"liquid_check_cracking_bar": v.liquid_valve_cracking_bar,
                      "liquid_check_reverse_leak_mL_min_bar": v.liquid_valve_reverse_leak_mL_min_bar}
                      for k, v in variants.items()},
                  "hypothetical_valve_inventory_item_id": None,
                  "hypothetical_valve_service_verified": False,
                  "sensitivity_basis": "Entered gas plenum and one tenth that value; illustrative bracket, not calibrated uncertainty or occurrence frequency.",
                  "simulations": rows}
        self.cache[candidate_id] = result
        return result

    def call_reviewer_tool(self, role, call_review):
        request = {"role": role, "phase": "tool_request", "task":
            "Call simulate_flow_transients before your review. DrFluidics must request every candidate; DrSafety may request any candidates to cross-check. "
            "No other tool or numerical arguments are permitted. Parameters are explicit assumptions, not measurements.",
            "available_tool": {"name": TOOL_NAME, "description": "Conditional transient pressure and reverse-liquid-flow screen with fixed scenarios and alternatives"},
            "candidate_inputs": self.inputs, "assumption_profile": self.profile.model_dump(),
            "response_schema": {"tool_name": TOOL_NAME, "candidate_ids": [r["candidate_id"] for r in self.pool]}}
        selected = call_review("Request the named engineering tool. Return only the requested JSON object, not a design or a safety verdict.", request)
        return self.execute(role, selected)


def compact_tool_result(response):
    """Keep evidence IDs/numbers without sending full time traces to the LLM."""
    candidates = []
    for result in response["candidate_results"]:
        item = {k: v for k, v in result.items() if k != "simulations"}
        representative = []
        for variant in sorted({r["variant"] for r in result.get("simulations", [])}):
            rows = [r for r in result["simulations"] if r["variant"] == variant]
            valid = [r for r in rows if r["status"] == "simulated_unvalidated"]
            invalid = [r for r in rows if r["status"] != "simulated_unvalidated"]
            if valid:
                worst = max(valid, key=lambda r: r["reverse_displacement_uL"])
                representative.append({**worst, "summary_scope": "largest reverse displacement across the eight fixed scenario/sensitivity combinations; NOT a probability",
                    "valid_simulations": len(valid), "invalid_simulations": len(invalid),
                    "MFC_rating_exceeded_in_any_scenario": any(r["MFC_outlet_rating_screen"]["exceeded_under_assumptions"] is True for r in valid),
                    "steady_reversal_in_any_sensitivity": any(r["scenario"] == "steady" and r["reverse_flow_predicted"] for r in valid)})
            representative.extend(invalid[:1])
        item["representative_simulations"] = [{k: r.get(k) for k in (
            "evidence_id", "status", "reason", "reverse_flow_predicted", "peak_reverse_liquid_mL_min",
            "reverse_displacement_uL", "peak_junction_bar_g", "min_MFC_margin_above_required_bar", "summary_scope",
            "valid_simulations", "invalid_simulations", "steady_reversal_in_any_sensitivity",
            "MFC_rating_exceeded_in_any_scenario", "peak_gas_plenum_bar_g", "peak_liquid_upstream_bar_g",
            "BPR_opening_pressure_reached", "window_note")}
            for r in representative]
        candidates.append(item)
    return {**response, "candidate_results": candidates}


def validate_physics_assessment(parsed, response):
    assessment = parsed.get("physics_assessment", {})
    ids = {r["evidence_id"] for c in response["candidate_results"] for r in c.get("simulations", [])}
    references = assessment.get("evidence_ids", [])
    if (assessment.get("tool_result_id") != response["result_id"] or not isinstance(references, list)
            or (ids and not references) or any(not isinstance(r, str) or r not in ids for r in references)
            or not assessment.get("limitations") or not isinstance(assessment.get("proposed_alternatives"), list)):
        raise ValueError("Physics reviewer must cite its actual tool result and evidence IDs, with limitations and proposed alternatives")
    return assessment


def selected_physics_findings(screen):
    """Publish conditional findings even when the Chief omits them in prose."""
    if not screen:
        return []
    rows = [r for r in screen.get("simulations", []) if r["variant"] == "as_designed"]
    common = {"kind": "conditional_physics_screen", "severity": "requires_engineering_review",
              "resolved": False, "measured_evidence": False, "backflow_probability": None}
    prefix = f"BF-PHYSICS:C{screen['candidate_id']}"
    findings = []
    valid = [r for r in rows if r["status"] == "simulated_unvalidated"]
    invalid = [r for r in rows if r["status"] != "simulated_unvalidated"]
    if invalid or not rows:
        findings.append({**common, "finding_id": prefix + ":INCOMPLETE",
            "message": "The conditional transient screen is incomplete or unsupported; this is not evidence of absence of backflow.",
            "evidence_ids": [r["evidence_id"] for r in invalid]})
    reversal = [r for r in valid if r["reverse_flow_predicted"]]
    if reversal:
        worst = max(reversal, key=lambda r: r["reverse_displacement_uL"])
        findings.append({**common, "finding_id": prefix + ":LIQUID-REVERSE",
            "message": f"Under illustrative/unvalidated dynamics, the as-designed model predicts up to {worst['reverse_displacement_uL']:.4g} uL reverse liquid displacement in the tested windows. This is not observed KHU backflow or proof that oxygen reaches Stage 1.",
            "evidence_ids": [r["evidence_id"] for r in reversal],
            "scenarios": sorted({r["scenario"] for r in reversal})})
    rating = [r for r in valid if r.get("MFC_outlet_rating_screen", {}).get("exceeded_under_assumptions") is True]
    if rating:
        worst = max(rating, key=lambda r: r["peak_gas_plenum_bar_g"])
        findings.append({**common, "finding_id": prefix + ":MFC-PRESSURE",
            "message": f"An as-designed stress scenario predicts MFC-outlet pressure up to {worst['peak_gas_plenum_bar_g']:.4g} bar(g), above the declared {worst['MFC_outlet_rating_screen']['declared_max_pressure_bar']:g} bar rating when treated as gauge. These are assumed dynamics; pressure basis and actual operating pressures require confirmation. This warning is not limited to the hypothetical valve alternative.",
            "evidence_ids": [r["evidence_id"] for r in rating],
            "scenarios": sorted({r["scenario"] for r in rating})})
    return findings
