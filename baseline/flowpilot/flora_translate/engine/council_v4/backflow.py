"""Council-side reverse-path screening and bounded gas-source alternatives.

This is a topology hazard screen, not a transient multiphase flow simulator.
An equipment pressure rating and a drawn arrow do not prove forward flow.
"""
from collections import defaultdict
from copy import deepcopy
import math
import re

from flora_translate.schemas import FlowProposal


VERSION = "flowpilot_council_backflow_v1"
SCENARIOS = ["gas introduction/startup", "liquid delivery interrupted",
             "shutdown with residual pressure", "downstream restriction or pressure change"]
REACTORS = {"reactor", "coil_reactor", "photoreactor", "heated_coil", "packed_bed", "chip_reactor"}
AVAILABLE = {"available", "ready", "in_service", "in service"}


def assess_topology(topology, proposal, inventory):
    """Trace physical reverse paths, including through nominally forward edges."""
    data = topology.model_dump() if hasattr(topology, "model_dump") else topology
    ops = {op["op_id"]: op for op in data.get("unit_operations", [])}
    incoming = defaultdict(list)
    for edge in data.get("streams", []):
        if edge.get("connection_type", "process") == "process":
            incoming[edge["to_op"]].append(edge)
    accessories = {a.equipment_id: a for a in inventory.safety_accessories}
    findings, branches = [], []
    for junction, edges in sorted(incoming.items()):
        if len(edges) < 2 or not any(e.get("stream_type") in {"gas", "gas_liquid"} for e in edges):
            continue
        for edge in sorted(edges, key=lambda e: e["stream_id"]):
            if edge.get("stream_type") not in {"liquid", "gas", "gas_liquid"}:
                continue
            stack = [(edge["from_op"], [junction], [])]
            while stack:
                node, path, valves = stack.pop()
                if node in path or node not in ops:
                    continue
                path = [*path, node]
                op = ops[node]
                typ = op.get("op_type", "").lower()
                if typ == "check_valve":
                    params = op.get("parameters", {})
                    item = accessories.get(op.get("inventory_item_id") or params.get("inventory_item_id"))
                    declared_direction = params.get("direction") in {"toward reactor", "forward"}
                    valves = [*valves, {"operation_id": node, "direction_declared_forward": declared_direction,
                        "equipment_id": item.equipment_id if item else None,
                        "max_pressure_bar": item.max_pressure_bar if item else None,
                        "cracking_pressure_bar": item.cracking_pressure_bar if item else None,
                        "service_verified": False}]
                terminal = typ in REACTORS | {"pump", "mfc"} or not incoming[node]
                if terminal:
                    protected = any(v["direction_declared_forward"] for v in valves)
                    branch = {"junction_id": junction, "stream_id": edge["stream_id"],
                        "branch_phase": edge.get("stream_type"), "reverse_path": path,
                        "exposed_operation_id": node, "valves_before_exposed_operation": valves,
                        "status": "protection_declared_requires_verification" if protected else "unprotected_reverse_path"}
                    branches.append(branch)
                    if not protected:
                        findings.append({"finding_id": f"BF-REVERSE-PATH:{junction}:{edge['stream_id']}:{node}",
                            "kind": "credible_risk", "severity": "requires_engineering_review",
                            "message": "Gas at the junction has a connected reverse path to " + node +
                                "; no forward-oriented non-return protection is declared before this operation.",
                            "reverse_path": path, "scenarios": SCENARIOS,
                            "consequence": "Possible upstream gas intrusion and displacement; upstream chemistry and stage residence may be disrupted.",
                            "required_action": "Review branch-specific non-return protection and pressure control, or a pressure-decoupled arrangement. Confirm equipment and service suitability before use.",
                            "resolved": False})
                    else:
                        findings.append({"finding_id": f"BF-VERIFY-PROTECTION:{junction}:{edge['stream_id']}:{node}",
                            "kind": "missing_confirmation", "severity": "requires_engineering_review",
                            "message": "Declared branch valve requires verification of direction, leakage, service compatibility, pressure rating, and low-flow operation.",
                            "reverse_path": path, "scenarios": SCENARIOS, "resolved": False})
                else:
                    for upstream in incoming[node]:
                        stack.append((upstream["from_op"], path, valves))
    branches.sort(key=lambda b: (b["junction_id"], b["stream_id"], b["reverse_path"]))
    findings.sort(key=lambda f: f["finding_id"])
    return {"schema_version": VERSION, "applicable": bool(branches), "branches": branches,
        "findings": findings, "laboratory_execution_status": "review_required" if branches else "not_assessed_by_this_screen",
        "BPR_setpoint_bar": proposal.BPR_bar,
        "limitations": ["Reverse connectivity is not a prediction that backflow will occur.",
            "A gas/liquid flow ratio alone cannot establish backflow or safety.",
            "Actual supply pressures, MFC differential-pressure requirements, valve characteristics and startup behaviour require verification.",
            "A valve on the MFC branch does not protect a different inlet branch.",
            "Lower gas volume does not close an unprotected reverse path."]}


def candidate_assessment(candidate, batch, plan, inventory):
    from flora_translate.main import _build_translate_topology
    from flora_translate.topology_compiler import compile_inventory_topology
    prop = FlowProposal.model_validate(candidate["proposal"])
    topology = _build_translate_topology(prop, plan, batch, inventory)
    topology, _ = compile_inventory_topology(topology, proposal=prop, inventory=inventory)
    report = assess_topology(topology, prop, inventory)
    report["candidate_id"] = candidate["candidate_id"]
    report["pressure_rating_check"] = deepcopy(candidate.get("pressure_headroom", {}))
    report["pressure_rating_check_is_operating_pressure_confirmation"] = False
    return report


def oxygen_option(plan, inventory, intake=None):
    """Do not override a chemist-confirmed gas or an explicit prohibition."""
    gas = plan.scientific_context.get("gas_delivery", {})
    if gas.get("species", "").lower() != "air" or gas.get("identity_source") != "protocol_fact":
        return {"eligible": False, "reason": "Gas is not batch-derived air; no automatic source substitution is authorized."}
    pkg = intake.model_dump() if hasattr(intake, "model_dump") else (intake or {})
    if any(a.get("question_id") == "Q-GAS-001" and a.get("status") == "answered"
           for a in pkg.get("answers", [])):
        return {"eligible": False, "reason": "An explicit gas-identity answer requires a separate chemist-approved revision."}
    gas_feeds = [f for f in plan.stream_logic if f.phase == "gas" and f.delivery_mode != "carried_from_previous"]
    if len(gas_feeds) != 1 or not re.search(r"\bair\b", " ".join(gas_feeds[0].reagents), re.I):
        return {"eligible": False, "reason": "Bounded source revision supports one physical air feed only."}
    explicit = [str(pkg.get("operating_limits", "")), str(plan.scientific_context.get("hard_constraints", "")),
                *[str(a.get("answer", "")) for a in pkg.get("answers", []) if a.get("status", "answered") == "answered"]]
    if re.search(r"only\s+air|air\s+only|(?:no|avoid|forbid\w*|do not use)\s+(?:pure\s+)?(?:oxygen|o2)|must\s+use\s+air", " ".join(explicit), re.I):
        return {"eligible": False, "reason": "An explicit input restricts the gas source."}
    devices = [g.equipment_id for g in inventory.gas_hardware if g.service_status in AVAILABLE
               and re.search(r"\b(?:o2|oxygen)\b", g.gas, re.I)]
    return {"eligible": bool(devices), "reason": "Declared MFC gas capability only; oxygen supply and oxygen-service compatibility are not confirmed.",
        "equipment_ids": devices, "action": "propose_pure_oxygen", "reactive_species": "O2",
        "preserve": ["delivered oxygen molar feed", "liquid flows and concentrations", "reactors", "temperature", "BPR", "gas addition stage"],
        "requires_chemist_confirmation": True}


def revised_oxygen_pool(pool, batch, plan, inventory):
    from flora_translate.design_realizer import realize_executable_design
    from flora_translate.final_engineering import calculate_final_stages
    from flora_translate.engine.council_v4.scientific import pressure_headroom, signature
    updated = plan.model_copy(deep=True)
    gas = deepcopy(updated.scientific_context["gas_delivery"])
    gas.update(species="O2", reagent_mole_fraction=1.0, identity_source="council_screening_proposal",
               requires_chemist_confirmation=True, previous_species="air", batch_species="air",
               decision_source="council_backflow_review", backflow_resolution_claimed=False)
    gas["rationale"] = "Council proposed lower total gas volume at unchanged delivered oxygen molar feed; source change is unapproved and does not resolve reverse paths."
    updated.scientific_context["gas_delivery"] = gas
    for feed in [*updated.stream_logic, *[f for s in updated.stages for f in s.feed_streams]]:
        if feed.phase == "gas":
            feed.reagents = ["O2"]
            feed.gas_reagent_mole_fraction = 1.0
            feed.requirement_authority = "model_inference"
            feed.reasoning = "Council-proposed pure oxygen; source change and oxygen service require chemist confirmation."
            feed.source_evidence = ["Council flow-operability source proposal, not a confirmed protocol requirement."]
            feed.feed_group = f"ST{feed.introduction_stage}-GAS-O2-COUNCIL-PROPOSAL"
            feed.molar_equiv_basis = "preserved_delivered_O2_inlet_stp"
    revised, checks = [], []
    for row in pool:
        old = FlowProposal.model_validate(row["proposal"])
        prop = old.model_copy(deep=True)
        prop.scientific_design["gas_delivery_policy"] = deepcopy(gas)
        for feed in prop.streams:
            if feed.phase == "gas":
                feed.gas_flow_sccm *= feed.gas_reagent_mole_fraction
                feed.gas_reagent_mole_fraction = 1.0
                feed.contents = ["O2 (pure; council proposal, confirmation required)"]
                feed.molar_equiv_basis = "proposed_delivered_O2_inlet_stp"
        prop, realization, validation = realize_executable_design(prop, batch_record=batch,
            chemistry_plan=updated.model_copy(deep=True), inventory=inventory,
            operating_limits=updated.scientific_context.get("operating_limits"),
            hard_constraints=updated.scientific_context.get("hard_constraints"))
        if not validation.get("checks") or not all(validation["checks"].values()):
            raise ValueError("Oxygen alternative failed deterministic realization: " + str(validation.get("unresolved_reasons")))
        before, after = signature(old), signature(prop)
        if len(old.streams) != len(prop.streams) or len(old.stage_parameters) != len(prop.stage_parameters):
            raise ValueError("Gas-source alternative changed the feed or stage count")
        for key in ("BPR_bar",):
            if before[key] != after[key]:
                raise ValueError("Gas-source alternative changed pressure")
        for a, b in zip(old.streams, prop.streams):
            if a.phase == "liquid" and a.model_dump() != b.model_dump():
                # Realization may refresh explanatory metadata, not setpoints.
                for key in ("flow_rate_mL_min", "concentration_M", "molar_equiv", "introduction_stage", "pump_equipment_id", "contents"):
                    if getattr(a, key) != getattr(b, key):
                        raise ValueError("Gas-source alternative changed a liquid feed: " + key)
            if a.phase == "gas":
                if b.introduction_stage != a.introduction_stage or not math.isclose(
                    a.gas_flow_sccm * a.gas_reagent_mole_fraction, b.gas_flow_sccm * b.gas_reagent_mole_fraction,
                    rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError("Gas-source alternative did not preserve stage and delivered oxygen")
        for a, b in zip(old.stage_parameters, prop.stage_parameters):
            for key in ("reactor_equipment_id", "reactor_volume_mL", "d_mm", "temperature_C", "light_equipment_id"):
                if a.get(key) != b.get(key):
                    raise ValueError("Gas-source alternative changed stage equipment/settings: " + key)
        engineering = calculate_final_stages(prop, batch, updated, inventory)
        if not engineering["complete"]:
            raise ValueError("Oxygen alternative has incomplete stage engineering")
        pressure = pressure_headroom(prop, engineering, inventory)
        if not engineering["complete"] or not pressure["passed"]:
            raise ValueError("Oxygen alternative failed engineering or pressure-rating checks")
        from flora_translate.scientific_objective import objective_fit
        item = {**deepcopy(row), "proposal": prop.model_dump(), "validation": validation,
                "objective_fit": objective_fit(prop, updated, row["target_stage_screen_min"]),
                "realization": realization, "engineering": engineering, "pressure_headroom": pressure}
        from flora_translate.main import _build_translate_topology, _topology_matches_serialized_proposal
        from flora_translate.topology_compiler import compile_inventory_topology
        topology = _build_translate_topology(prop, updated, batch, inventory)
        topology, allocation = compile_inventory_topology(topology, proposal=prop, inventory=inventory)
        if inventory.strict_assignment and (not allocation.get("checks", {}).get("all_required_operations_assigned")
                or not _topology_matches_serialized_proposal(topology, prop)):
            raise ValueError("Oxygen alternative failed topology inventory assignment")
        item["inventory_allocation"] = allocation
        item["backflow_assessment"] = candidate_assessment(item, batch, updated, inventory)
        checks.append({"candidate_id": row["candidate_id"], "oxygen_molar_feed_preserved": True,
                       "gas_flow_serialization_tolerance_mL_min": 1e-6,
                       "liquid_feeds_and_stage_equipment_preserved": True,
                       "backflow_resolved": False})
        revised.append(item)
    return revised, updated, checks


def review_and_revise(pool, batch, plan, inventory, intake, call_review, audit, save):
    assessments = [candidate_assessment(row, batch, plan, inventory) for row in pool]
    record = {"schema_version": VERSION, "enabled": True, "before": assessments,
              "application_status": "not_applicable", "revision_attempts": 0}
    audit["backflow_review"] = record
    save()
    if not any(a["applicable"] for a in assessments):
        return pool, plan
    option = oxygen_option(plan, inventory, intake)
    alternatives = []
    for row in pool:
        gases = [f for f in row["proposal"]["streams"] if f["phase"] == "gas"]
        if len(gases) == 1:
            g = gases[0]
            alternatives.append({"candidate_id": row["candidate_id"], "current_gas": g["contents"],
                "gas_inlet_STP_mL_min": g["gas_flow_sccm"], "gas_reagent_fraction": g["gas_reagent_mole_fraction"],
                "same_oxygen_pure_O2_STP_mL_min": g["gas_flow_sccm"] * g["gas_reagent_mole_fraction"] if option["eligible"] else None,
                "BPR_bar": row["proposal"]["BPR_bar"],
                "liquid_feeds": [{k: f[k] for k in ("stream_label", "flow_rate_mL_min", "concentration_M", "introduction_stage")}
                                 for f in row["proposal"]["streams"] if f["phase"] == "liquid"]})
    system = ("You are the FlowPilot council flow-operability reviewer. Return JSON only. "
        "Assess branch-to-branch reverse paths in a connected flow process, including startup, interrupted liquid delivery and residual pressure. "
        "Do not claim backflow is certain, assign probabilities, or use the inlet gas/liquid ratio as a backflow threshold. "
        "A gas-branch valve does not protect the liquid branch. Equipment maximum pressure is not actual supply pressure. "
        "Choose retain or propose_pure_oxygen only from allowed_actions, with a scientific explanation of tradeoffs. "
        "Neither choice is preferred by default. Pure oxygen can reduce delivered gas volume at equal oxygen molar feed, "
        "but changes oxygen partial pressure and fire risk; it requires chemist and oxygen-service confirmation and does NOT resolve reverse connectivity. "
        "Identify branch-specific controls and missing operating data. Do not invent installed valves or claim laboratory validation. "
        "The scope is one bounded source revision; other arrangements must be labelled recommendations requiring separate engineering. "
        "Use only the supplied candidate IDs and finding IDs. Do not output hidden reasoning; provide concise review conclusions.")
    request = {"role": "FlowOperability", "allowed_actions": ["retain", "propose_pure_oxygen"] if option["eligible"] else ["retain"],
        "source_gas": plan.scientific_context.get("gas_delivery", {}), "oxygen_option": option,
        "candidate_flows": alternatives,
        "assessment": assessments[0], "other_candidate_pressure_checks": [
            {"candidate_id": a["candidate_id"], "BPR_bar": a["BPR_setpoint_bar"], "paths": a["pressure_rating_check"].get("paths", [])} for a in assessments],
        "response_schema": {"action": "one allowed action", "justification": "brief conclusion and tradeoffs",
            "addressed_finding_ids": ["existing finding ID"], "required_controls": ["specific control; availability unconfirmed unless established"],
            "required_confirmations": ["specific missing information"], "limitations": ["specific uncertainty"]}}
    decision = call_review(system, request)
    record["decision"] = decision
    save()
    ids = {f["finding_id"] for a in assessments for f in a["findings"]}
    if (decision.get("action") not in request["allowed_actions"] or not isinstance(decision.get("justification"), str)
            or not decision["justification"].strip() or any(
                not isinstance(decision.get(k), list) or not all(isinstance(v, str) and v.strip() for v in decision[k])
                for k in ("addressed_finding_ids", "required_controls", "required_confirmations", "limitations"))
            or not decision["addressed_finding_ids"] or not set(decision["addressed_finding_ids"]).issubset(ids)):
        raise ValueError("Invalid council flow-operability decision; raw response preserved")
    record["application_status"] = "retained"
    if decision["action"] == "propose_pure_oxygen":
        record["revision_attempts"] = 1
        try:
            revised, updated, checks = revised_oxygen_pool(pool, batch, plan, inventory)
        except ValueError as exc:
            record.update(application_status="revision_rejected_by_engineering", revision_error=str(exc))
        else:
            record.update(application_status="proposed_gas_source_revision", revision_checks=checks,
                original_candidates=deepcopy(pool), after=[r["backflow_assessment"] for r in revised],
                requires_chemist_confirmation=True, backflow_resolved=False)
            save()
            return revised, updated
    for row, assessment in zip(pool, assessments):
        row["backflow_assessment"] = assessment
    record["after"] = deepcopy(assessments)
    save()
    return pool, plan
