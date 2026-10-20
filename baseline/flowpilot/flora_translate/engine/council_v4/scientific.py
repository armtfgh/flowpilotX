"""Joint stage screens and evidence-aware review within the existing council.

Scope: one or two timed liquid stages, optionally irradiated, with at most one
metered gas feed. Screening coordinates do not imply achieved conversion.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from flora_translate.design_realizer import realize_executable_design, AVAILABLE
from flora_translate.final_engineering import calculate_final_stages
from flora_translate.design_calculator import DesignCalculator
from flora_translate.inventory_constraints import available_pressure_settings
from flora_translate.schemas import DesignCandidate, DeliberationLog, AgentDeliberation
from flora_translate.scientific_evidence import assess_analogies


def signature(proposal):
    return {"stages": [{k: s.get(k) for k in ("stage_number", "reactor_equipment_id", "reactor_volume_mL", "temperature_C",
                  "Q_liquid_mL_min", "Q_gas_sccm", "residence_time_min", "residence_time_basis", "d_mm", "light_equipment_id", "wavelength_nm")} for s in proposal.stage_parameters],
            "streams": [{"label": s.stream_label, "stage": s.introduction_stage,
                         "flow": s.flow_rate_mL_min, "concentration": s.concentration_M,
                         "contents": s.contents, "phase": s.phase, "gas_inlet_mL_min": s.gas_flow_sccm,
                         "gas_fraction": s.gas_reagent_mole_fraction, "equiv": s.molar_equiv} for s in proposal.streams], "BPR_bar": proposal.BPR_bar}


def candidate_summary(row):
    p = row["proposal"]
    return {"candidate_id": row["candidate_id"], "target_stage_screen_min": row["target_stage_screen_min"],
            "objective_fit": row.get("objective_fit"),
            "stage_parameters": [{k: s.get(k) for k in ("stage_number", "reactor_equipment_id", "reactor_volume_mL", "d_mm", "material", "temperature_C", "Q_liquid_mL_min", "Q_gas_sccm", "residence_time_min", "residence_time_basis", "light_equipment_id", "wavelength_nm")} for s in p["stage_parameters"]],
            "streams": [{k: s.get(k) for k in ("stream_label", "contents", "solvent", "concentration_M", "flow_rate_mL_min", "phase", "gas_flow_sccm", "gas_reagent_mole_fraction", "molar_equiv", "introduction_stage", "pump_equipment_id")} for s in p["streams"]], "BPR_bar": p["BPR_bar"],
            "temperature_deviations": row["temperature_deviations"],
            "engineering": row["engineering"], "validation": row["validation"],
            "inventory_assumptions": row.get("inventory_allocation", {}).get("warnings", []),
            "predicted_yield_pct": None, "kinetics_status": "uncharacterized"}


def comparison_facts(pool, source_context):
    """Expose computed comparisons separately from the council's preferences."""
    rows = []
    holds = {h["stage_number"]: h["batch_time_min"] for h in source_context.get("timed_holds", [])}
    for candidate in pool:
        p = candidate["proposal"]
        reference = next((s for s in p["streams"] if s.get("phase") == "liquid"
                          and s.get("introduction_stage") == 1 and s.get("molar_equiv") == 1
                          and s.get("concentration_M")), None)
        rows.append({"candidate_id": candidate["candidate_id"],
            "limiting_feed_mmol_min": reference["flow_rate_mL_min"] * reference["concentration_M"] if reference else None,
            "total_stage_time_min": sum(s["residence_time_min"] for s in p["stage_parameters"]),
            "actual_stage_hold_ratios": [{"stage_number": s["stage_number"],
                "basis": "inlet/STP apparent index" if s.get("Q_gas_sccm", 0) > 0 else "liquid V/Q",
                "physical_exposure_equivalence_established": False,
                "flow_min": s["residence_time_min"], "batch_min": holds.get(s["stage_number"]),
                "ratio": s["residence_time_min"] / holds[s["stage_number"]] if holds.get(s["stage_number"]) else None}
                for s in p["stage_parameters"]],
            "mixers": [{"stage_number": s["stage_number"], "name": s.get("mixer_name"),
                "assignment_status": s.get("mixer_assignment_status")} for s in p["stage_parameters"] if s.get("mixer_name")]})
    return {"candidates": rows, "batch_reaction_hold_total_min": sum(holds.values()),
            "batch_total_excludes": "Cooling, charging, workup and other handling time are excluded from the summed reaction holds.",
            "scope": "These twelve screens vary stage-time coordinates and compatible reactor geometry within the upstream topology. They do not compare cooled versus direct interstage transfer and do not optimize gas equivalents. A gas-stage inlet/STP time ratio is not a ratio of liquid exposure to batch exposure.",
            "equipment_boundary": "A two-port union is not a three-port T-mixer. An assumed_standard_accessory mixer is a separate unconfirmed item, not a confirmed union capability.",
            "evidence_boundary": "Computed feed throughput is not measured product throughput. No measured time-temperature compensation, yield or comparative degradation ranking is available."}


def build_screen_pool(proposal, batch, plan, inventory, budget=12, objective_policy=None):
    from flora_translate.scientific_objective import screen_patterns, objective_fit, source_stage_temperature
    context = plan.scientific_context
    constraints = {"operating_limits": context.get("operating_limits"), "hard_constraints": context.get("hard_constraints")}
    realization_options = {"hard_constraints": constraints, "operating_limits": context.get("operating_limits")}
    if context.get("issues"):
        raise ValueError("Scientific source confirmation required: " + "; ".join(context["issues"]))
    stages = plan.stages
    if not 1 <= len(stages) <= 2:
        raise ValueError("Scientific screens support one or two connected stages")
    base, _, validation = realize_executable_design(proposal, batch_record=batch,
                           chemistry_plan=plan.model_copy(deep=True), inventory=inventory, **realization_options)
    base.heat_transfer_metrics = {}
    liquids = [s for s in base.streams if s.phase == "liquid"]
    gases = [s for s in base.streams if s.phase == "gas"]
    if len(gases) > 1 or any(s.phase not in {"liquid", "gas"} for s in base.streams):
        raise ValueError("Scientific screens support liquid feeds and at most one metered gas, not solids or mixed-gas feeds")
    if not liquids or any(not s.concentration_M or not s.molar_equiv for s in liquids):
        raise ValueError("Every scientific screen feed needs an explicit concentration and equivalent basis.")
    from flora_translate.residence_time_basis import stp_gas_flow_for_equiv
    weights = {s.stream_label: s.molar_equiv / s.concentration_M for s in liquids}
    for gas in gases:
        weights[gas.stream_label] = stp_gas_flow_for_equiv(1.0, 1.0, gas.molar_equiv, gas.gas_reagent_mole_fraction or 1.0)
    cumulative = [sum(weights[s.stream_label] for s in base.streams if s.introduction_stage <= st.stage_number) for st in stages]
    reactors = sorted([r for r in inventory.reactors if r.service_status in AVAILABLE and r.volume_mL > 0
                       and r.max_temperature_C is not None], key=lambda r: r.equipment_id)
    combos = []
    pressures = available_pressure_settings(inventory) or [base.BPR_bar]
    for rs in itertools.product(reactors, repeat=len(stages)):
        if any(n > next(r.quantity for r in rs if r.equipment_id == key) for key, n in Counter(r.equipment_id for r in rs).items()):
            continue
        temps = []
        for st, r in zip(stages, rs):
            desired = st.temperature_C or batch.temperature_C
            t = min(desired, r.max_temperature_C)
            if r.allowed_temperatures_C:
                allowed = [v for v in r.allowed_temperatures_C if v <= t]
                if not allowed:
                    break
                t = max(allowed)
            temps.append(t)
        if len(temps) == len(stages):
            combos.append((rs, temps))
    patterns = screen_patterns((objective_policy or {}).get("priority", "balanced"), len(stages))
    if budget != 12:
        raise ValueError("Scientific preview retains exactly 12 candidates; use legacy mode for budget ablations.")
    pool, rejected, seen = [], [], set()
    for pattern in patterns:
        targets = [s.batch_time_h * 60 * m for s, m in zip(stages, pattern)]
        ranked = []
        for rs, temps in combos:
            scales = [r.volume_mL / (t * q) for r, t, q in zip(rs, targets, cumulative)]
            k = math.exp(sum(math.log(v) for v in scales) / len(scales))
            error = sum(abs(math.log((r.volume_mL / (q * k)) / t)) for r, q, t in zip(rs, cumulative, targets))
            deviation = sum(abs((source_stage_temperature(plan, st.stage_number) if source_stage_temperature(plan, st.stage_number) is not None else st.temperature_C or batch.temperature_C) - t) for st, t in zip(stages, temps))
            for pressure in pressures:
                ranked.append(((deviation, error, sum(r.volume_mL for r in rs), abs(pressure - base.BPR_bar), tuple(r.equipment_id for r in rs)), rs, temps, k, pressure))
        found = False
        for _, rs, temps, k, pressure in sorted(ranked):
            p = base.model_copy(deep=True)
            p.scientific_design = {"mode": "scientific_v2", "lock_stage_inventory": True}
            if gases:
                p.scientific_design["gas_delivery_policy"] = deepcopy(context.get("gas_delivery", {}))
            p.flow_rate_mL_min = k * sum(weights[s.stream_label] for s in liquids)
            for s in p.streams:
                if s.phase == "liquid":
                    s.flow_rate_mL_min = k * weights[s.stream_label]
                else:
                    s.gas_flow_sccm = k * weights[s.stream_label]
            p.stage_parameters = [{"stage_number": st.stage_number, "reactor_equipment_id": r.equipment_id,
                "reactor_volume_mL": r.volume_mL, "V_R_mL": r.volume_mL, "d_mm": r.ID_mm,
                "material": r.material, "temperature_C": t} for st, r, t in zip(stages, rs, temps)]
            p.reactor_volume_mL = sum(r.volume_mL for r in rs)
            p.temperature_C = temps[0]
            p.BPR_bar = pressure
            p, report, val = realize_executable_design(p, batch_record=batch,
                           chemistry_plan=plan.model_copy(deep=True), inventory=inventory, **realization_options)
            key = json.dumps(signature(p), sort_keys=True)
            failed = [name for name, ok in val.get("checks", {}).items() if not ok]
            if failed:
                rejected.append({"target": targets, "reasons": failed})
                continue
            if key in seen:
                continue
            engineering = calculate_final_stages(p, batch, plan, inventory)
            if not engineering["complete"]:
                rejected.append({"target": targets, "reasons": ["stage engineering incomplete"], "engineering": engineering})
                continue
            pressure_check = pressure_headroom(p, engineering, inventory)
            if not pressure_check["passed"]:
                rejected.append({"target": targets, "BPR_bar": p.BPR_bar, "reasons": ["pump/reactor pressure headroom"], "details": pressure_check})
                continue
            val["checks"]["pump_and_reactor_pressure_headroom"] = True
            allocation = {}
            if inventory.strict_assignment:
                from flora_translate.main import _build_translate_topology, _topology_matches_serialized_proposal
                from flora_translate.topology_compiler import compile_inventory_topology
                topology = _build_translate_topology(p, plan, batch, inventory)
                _, allocation = compile_inventory_topology(topology, proposal=p, inventory=inventory)
                if not allocation.get("checks", {}).get("all_required_operations_assigned") or not _topology_matches_serialized_proposal(topology, p):
                    rejected.append({"target": targets, "reasons": ["pre-council topology allocation incomplete"], "allocation": allocation})
                    continue
                val["checks"]["pre_council_topology_assignment_complete"] = True
            deviations = [{"stage_number": st.stage_number, "batch_temperature_C": source_stage_temperature(plan, st.stage_number),
                           "screen_temperature_C": t, "consequence": "Temperature differs from the source batch hold or is not stage-resolved; no validated rate correction. Yield and required time remain unknown."}
                          for st, t in zip(stages, temps) if t != source_stage_temperature(plan, st.stage_number)]
            pool.append({"candidate_id": len(pool) + 1, "target_stage_screen_min": targets,
                         "objective_fit": objective_fit(p, plan, targets),
                         "proposal": p.model_dump(), "validation": val, "realization": report,
                         "engineering": engineering, "temperature_deviations": deviations, "inventory_allocation": allocation,
                         "pressure_headroom": pressure_check})
            seen.add(key)
            found = True
            break
        if not found:
            error = ValueError(f"Cannot construct 12 distinct feasible screens: target {targets}; failures {rejected[-3:]}")
            error.rejections = rejected
            raise error
    return pool, rejected


def pressure_headroom(proposal, engineering, inventory):
    """Conservative path pressure: downstream BPR plus cumulative reactor losses.

    Additional connector/mixer losses and instrument calibration still require
    verification. A pump's maximum rating is not a usable BPR setpoint itself.
    """
    losses = {s["stage_number"]: s["calculations"]["pressure_drop_bar"] for s in engineering["stages"]}
    pumps = {p.equipment_id: p for p in inventory.pumps}
    gas_devices = {p.equipment_id: p for p in inventory.gas_hardware}
    reactors = {r.equipment_id: r for r in inventory.reactors}
    rows = []
    for stream in proposal.streams:
        required = proposal.BPR_bar + sum(v for n, v in losses.items() if n >= stream.introduction_stage)
        pump = (gas_devices if stream.phase == "gas" else pumps).get(stream.pump_equipment_id)
        if stream.phase == "gas" and pump and pump.required_outlet_accessory_type:
            accessories = [a for a in inventory.safety_accessories if a.type == pump.required_outlet_accessory_type]
            required += min((a.cracking_pressure_bar or 0 for a in accessories), default=0)
        limit = pump.max_pressure_bar if pump else None
        rows.append({"item": stream.pump_equipment_id, "required_bar": required, "maximum_bar": limit,
                     "passed": limit is not None and required < limit})
    for s in proposal.stage_parameters:
        reactor = reactors.get(s.get("reactor_equipment_id"))
        required = proposal.BPR_bar + sum(v for n, v in losses.items() if n >= s["stage_number"])
        limit = reactor.max_pressure_bar if reactor else None
        rows.append({"item": s.get("reactor_equipment_id"), "required_bar": required, "maximum_bar": limit,
                     "passed": limit is not None and required < limit})
    return {"passed": bool(rows) and all(r["passed"] for r in rows), "paths": rows,
            "basis": "BPR plus calculated downstream coil losses; minor losses and rated operating margins require laboratory verification."}


def _json_response(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Council response must be a JSON object")
    return result


def _call_json_review(call_llm, system, request, token_budget, audit, save):
    """Retry invalid JSON once; preserve both responses and never repair verdicts."""
    for attempt in (1, 2):
        prompt = system if attempt == 1 else system + (
            " Your previous response was not complete valid JSON. Return the entire requested JSON object again, "
            "with concise text and no extra keys or commentary. Do not omit required fields.")
        budget = min(token_budget * attempt, 12000)
        raw = call_llm(prompt, json.dumps(request), budget)
        entry = {"role": request["role"], "attempt": attempt, "max_tokens": budget,
                 "system": prompt, "request": request, "raw_response": raw}
        audit["calls"].append(entry)
        save()
        try:
            return _json_response(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            entry["parse_error"] = str(exc)
            save()
            if attempt == 2:
                raise ValueError(f'{request["role"]} returned invalid JSON after two recorded attempts') from exc


def run_scientific_council(proposal, batch, plan, inventory, analogies, objectives, budget=12, intake=None, backflow_review=True, physics_profile=None):
    from flora_translate.engine.llm_agents import call_llm
    from flora_translate.engine import llm_agents
    from flora_translate.intake_agent import intake_context_block
    from flora_translate.schemas import FlowProposal
    from flora_translate.scientific_objective import resolve_objective, answer_effects, pool_fingerprint
    intent = resolve_objective(objectives, intake)
    pool, rejected = build_screen_pool(proposal, batch, plan, inventory, budget, intent)
    audit = {"schema_version": "flowpilot_scientific_council_v1", "status": "screening_hypothesis",
             "candidate_count": len(pool), "candidates": pool, "rejected_generation_attempts": rejected,
             "source_context": deepcopy(plan.scientific_context), "analogy_audit": assess_analogies(analogies, plan),
             "calls": [], "reviews": {}, "predicted_yield_pct": None,
             "screening_policy": "Fixed exploratory stage-time multipliers relative to explicit batch holds; these are not kinetic predictions.",
             "objective": objectives, "objective_policy": intent, "pool_sha256": pool_fingerprint(pool)}
    audit["provider"] = llm_agents.ENGINE_PROVIDER
    audit["model"] = getattr(llm_agents, "ENGINE_MODEL_" + llm_agents.ENGINE_PROVIDER.upper(), "not recorded")
    archive = Path("outputs/scientific_council") / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    archive.mkdir(parents=True, exist_ok=False)
    audit["archive_path"] = str(archive)
    def save():
        (archive / "audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    save()
    physics = None
    if physics_profile is not None:
        if not backflow_review:
            raise ValueError("Physics tools require topology review")
        from flora_translate.engine.council_v4.physics_tools import PhysicsToolSession
        physics = PhysicsToolSession(pool, batch, plan, inventory, intake, physics_profile, archive)
        for row, assessment in zip(pool, physics.assessments):
            row["backflow_assessment"] = assessment
        audit["backflow_review"] = {"enabled": True, "application_status": "existing_reviewers_with_physics_tools",
                                    "before": physics.assessments, "backflow_resolved": False}
        audit["physics_review"] = physics.manifest()
    elif backflow_review:
        from flora_translate.engine.council_v4.backflow import review_and_revise
        pool, plan = review_and_revise(pool, batch, plan, inventory, intake,
            lambda system, request: _call_json_review(call_llm, system, request, 3000, audit, save), audit, save)
        audit.update(candidates=pool, source_context=deepcopy(plan.scientific_context), pool_sha256=pool_fingerprint(pool))
    else:
        audit["backflow_review"] = {"enabled": False, "application_status": "disabled"}
    save()
    context = {"objective": objectives, "objective_policy": intent, "authority_labeled_intake": intake_context_block(intake),
               "source_context": audit["source_context"], "analogy_audit": audit["analogy_audit"],
               "candidates": [candidate_summary(c) for c in pool],
               "computed_comparisons": comparison_facts(pool, audit["source_context"])}
    if backflow_review and audit["backflow_review"].get("application_status") != "not_applicable":
        flow_review = audit["backflow_review"]
        context["flow_operability_review"] = {k: flow_review[k] for k in (
            "decision", "application_status", "revision_checks", "revision_error",
            "requires_chemist_confirmation", "backflow_resolved") if k in flow_review}
        context["candidate_backflow_assessments"] = [row.get("backflow_assessment", {}) for row in pool]
    # Send only physical annotations, not entire repeated calculator prose.
    for row in context["candidates"]:
        row["engineering"] = [{"stage_number": s["stage_number"], "status": s["status"],
            "hydraulics": {k: s["calculations"].get(k) for k in ("reynolds_number", "pressure_drop_bar", "peclet_number", "kinetics_status")}}
            for s in row["engineering"]["stages"]]
    system = ("You are a FlowPilot council reviewer. Return JSON only with concise evidence-based justification, not hidden reasoning. "
              "Authority: measured evidence > hard constraints > protocol facts > hypotheses > inference. "
              "These are unvalidated experimental SCREENS, not proven processes. Do not invent conversion, yield, kinetics, or residence-time superiority. "
              "No global shortest-time objective, 30% conversion floor, or class intensification factor. "
              "Batch residence times are screening anchors, not guaranteed flow optima. Review the COMPLETE connected process. "
              "Apply the explicit objective_policy to preferences. Near-batch times are not inherently the best choice. "
              "For yield priority compare adequate exposure in ALL stages against longer-time degradation risks, especially at a lower temperature. "
              "For throughput priority distinguish feed throughput from unmeasured product throughput. "
              "For gas stages, gas_flow_sccm and Q_gas_sccm mean mL/min at inlet/STP (273.15 K, 1.01325 bar). "
              "Their apparent residence-time index is V/(Q_liquid+Q_gas_STP), not measured liquid residence or photon exposure. "
              "Gas equivalents are proposed delivered molar ratios, not established stoichiometric consumption or uptake. "
              "If source_context.gas_delivery.requires_chemist_confirmation is true, explicitly report the gas-source change as unapproved and requiring laboratory review before use. "
              "Irradiation sources and wavelengths must be stage-assigned; no optical intensification or light-dose adequacy is proven. "
              "Use computed_comparisons for quantitative statements: target multipliers are not actual achieved hold ratios. "
              "Do not call unequal achieved ratios symmetric or claim highest throughput without checking all candidates. "
              "Honor equipment_boundary and scope: a union is not a T-mixer; an untested cooling hypothesis does not remove a hard constraint. "
              "State reduced-temperature compensation and comparative degradation risk as untested hypotheses, never established facts. "
              "Account for temperature deviations, later feed dilution and addition order, equipment and unanswered safety/solubility issues. "
              "Do not rewrite candidate numbers. A hard violation vetoes a candidate; an unresolved experimental outcome is an uncertainty, not proof of failure. "
              "Keep each candidate justification under 45 words and each uncertainty under 20 words. Use a shared assessment for issues common to all candidates.")
    if backflow_review and audit["backflow_review"].get("application_status") != "not_applicable":
        system += (" Address supplied flow_operability_review and reverse-path findings explicitly. "
                   "A reduced gas flow does not resolve reverse connectivity. A proposal requiring chemist confirmation is not an approved gas supply. "
                   "State branch-specific protection requirements; do not infer protection from a valve on a different branch. "
                   "Missing operating data must remain unconfirmed, not silently replaced by equipment maximum ratings.")
    if physics:
        system += (" DrFluidics and DrSafety use an experimental deterministic pressure-network tool. "
            "Tool results are conditional on unmeasured parameters, not measured evidence, probabilities, or proof of safe operation. "
            "Distinguish as_designed from hypothetical alternatives and entered_profile from gas-volume sensitivity. "
            "No reversal in these tests does not establish safety. Report predicted liquid displacement, not oxygen reaching Reactor 1. "
            "A gas check valve's cracking pressure concerns pressure DIFFERENCE across that valve, not BPR minus cracking pressure. "
            "Respect equipment_roles: a gas MFC is not the liquid pump. Do not rewrite designs or automatically approve pure oxygen. "
            "Recommend mitigations only as unapproved proposals with chemistry, inventory and oxygen-service review. "
            "Assumed transient results alone are not confirmed hard violations of a candidate. Retain unresolved limitations.")
    entries = []
    physics_reviewers = {}
    for role, focus in [("DrChemistry", "identity, order of addition, chemical compatibility, solubility and no intermediate isolation"),
                        ("DrKinetics", "stage coupling, evidence sufficiency, screening informativeness, uncertainty"),
                        ("DrFluidics", "cumulative stream flows, inventory assignments, geometry and hydraulic closure"),
                        ("DrSafety", "pressure, temperature, volatility, materials and controls requiring chemist confirmation")]:
        tool_response = None
        if physics and role in {"DrFluidics", "DrSafety"}:
            from flora_translate.engine.council_v4.physics_tools import compact_tool_result, validate_physics_assessment
            tool_response = physics.call_reviewer_tool(role,
                lambda sys, req: _call_json_review(call_llm, sys, req, 1200, audit, save))
            context.setdefault("physics_tool_results", {})[role] = compact_tool_result(tool_response)
            audit["physics_review"] = physics.manifest()
            audit["physics_review"]["reviewer_assessments"] = deepcopy(physics_reviewers)
            save()
        request = {"role": role, "focus": focus, "context": context,
            "response_schema": {"reviews": [{"candidate_id": "integer, exactly one per candidate 1..12",
                "recommendation": "prefer | acceptable | reject", "hard_violation": "boolean",
                "justification": "brief evidence-based assessment", "uncertainties": ["specific uncertainty"]}]}}
        if tool_response:
            request["response_schema"]["physics_assessment"] = {
                "tool_result_id": tool_response["result_id"], "evidence_ids": ["actual simulation evidence ID(s) used"],
                "proposed_alternatives": ["conditional mitigation and what remains unconfirmed; empty allowed"],
                "limitations": ["explicit limitations of the simulation and missing measurements"]}
        parsed = _call_json_review(call_llm, system, request, 6000, audit, save)
        if tool_response:
            physics_reviewers[role] = validate_physics_assessment(parsed, tool_response)
            audit["physics_review"]["reviewer_assessments"] = deepcopy(physics_reviewers)
            save()
        rows = parsed.get("reviews", [])
        ids = [r.get("candidate_id") for r in rows]
        if sorted(ids) != list(range(1, 13)) or any(type(r.get("hard_violation")) is not bool or r.get("recommendation") not in {"prefer", "acceptable", "reject"} or not r.get("justification") for r in rows):
            raise ValueError(f"{role} did not provide 12 complete candidate reviews; no silent council skip.")
        audit["reviews"][role] = rows
        entries.append(AgentDeliberation(agent=role, agent_display_name=role, round=1,
            chain_of_thought=json.dumps(parsed, indent=2), findings=[f"Reviewed all {len(rows)} complete stage designs."], status="WARNING"))
    # A soft preference against an experiment is not a physical prohibition.
    veto = {r["candidate_id"] for rows in audit["reviews"].values() for r in rows if r["hard_violation"]}
    if physics:
        audit["physics_review"]["reviewer_assessments"] = physics_reviewers
        context["physics_reviewer_assessments"] = physics_reviewers
    request = {"role": "Skeptic", "context": context, "reviews": audit["reviews"], "already_vetoed": sorted(veto),
               "task": "Audit cross-domain conflicts and unsupported scientific claims. Return extra vetoes only for identified hard violations.",
               "response_schema": {"vetoes": [{"candidate_id": "integer", "reason": "hard violation and evidence"}], "assessment": "brief", "required_measurements": ["measurement"]}}
    skeptic = _call_json_review(call_llm, system, request, 4500, audit, save)
    for v in skeptic.get("vetoes", []):
        if v.get("candidate_id") not in range(1, 13) or not v.get("reason"):
            raise ValueError("Invalid scientific skeptic veto")
        veto.add(v["candidate_id"])
    allowed = sorted(set(range(1, 13)) - veto)
    if not allowed:
        raise ValueError("All scientific screens vetoed; retain council transcript for diagnosis")
    request = {"role": "Chief", "context": context, "reviews": audit["reviews"], "skeptic": skeptic, "eligible_ids": allowed,
               "task": "Select an experiment using objective_policy, evidence and constraints. Do not substitute baseline similarity for the stated priority. Do not invent yield. Compare the selected candidate against at least two other eligible candidates with different stage exposures. Explain why the trade-off meets the priority, what cannot be inferred, and what measurement would change the choice. No forced time increase or decrease. Never choose outside eligible_ids.",
               "response_schema": {"candidate_id": "integer", "justification": "brief", "objective_alignment": "specific trade-off for the resolved priority", "alternatives": [{"candidate_id": "another eligible integer", "reason_not_selected": "evidence and objective based comparison"}], "answer_impacts": [{"question_id": "an actual intake question ID", "effect": "specific influence or explain no change; do not claim causation"}], "limitations": ["limitation"], "next_measurements": ["measurement"]}}
    chief = _call_json_review(call_llm, system, request, 4500, audit, save)
    if type(chief.get("candidate_id")) is not int or chief.get("candidate_id") not in allowed or not chief.get("justification"):
        raise ValueError("Chief selected an ineligible or unidentified scientific candidate")
    alternatives = chief.get("alternatives", [])
    if not isinstance(alternatives, list) or any(not isinstance(v, dict) or type(v.get("candidate_id")) is not int for v in alternatives):
        raise ValueError("Chief alternatives must be a list of candidate comparisons")
    alternative_ids = [v.get("candidate_id") for v in alternatives]
    if (not chief.get("objective_alignment") or len(set(alternative_ids)) < min(2, len(allowed) - 1)
            or any(n not in allowed or n == chief["candidate_id"] for n in alternative_ids)
            or any(not v.get("reason_not_selected") for v in alternatives)):
        raise ValueError("Chief must explain objective alignment and compare eligible alternatives; raw response saved")
    data = intake.model_dump() if hasattr(intake, "model_dump") else (intake or {})
    question_ids = {v["question_id"] for v in data.get("answers", [])}
    impacts = chief.get("answer_impacts", [])
    if not isinstance(impacts, list) or any(not isinstance(v, dict) or v.get("question_id") not in question_ids or not v.get("effect") for v in impacts):
        raise ValueError("Chief answer impacts contain invented or invalid intake references")
    audit["answer_effects"] = answer_effects(intake, intent, {
        "objective_alignment": chief["objective_alignment"], "alternatives": alternatives,
        "council_answer_assessments": impacts})
    audit.update({"selected_candidate_id": chief["candidate_id"], "chief": chief, "skeptic": skeptic, "vetoed_ids": sorted(veto)})
    selected = next(c for c in pool if c["candidate_id"] == chief["candidate_id"])
    chosen = FlowProposal.model_validate(selected["proposal"])
    audit["selected_signature"] = signature(chosen)
    save()
    chosen.scientific_design.update(audit)
    later = [AgentDeliberation(agent=role, agent_display_name=role, round=2,
              chain_of_thought=json.dumps(data, indent=2), status="WARNING") for role, data in [("Skeptic", skeptic), ("Chief", chief)]]
    log = DeliberationLog(rounds=[entries, later], total_rounds=2, consensus_reached=False,
                         summary="12 hardware-bound candidates reviewed; first experiment selected, chemical performance unvalidated.")
    flow = audit["backflow_review"]
    if flow.get("decision"):
        log.rounds.insert(0, [AgentDeliberation(agent="FlowOperability", agent_display_name="Flow operability",
            round=0, chain_of_thought=json.dumps(flow["decision"], indent=2),
            findings=[flow["application_status"]], status="WARNING")])
        log.total_rounds = 3
    calculations = DesignCalculator().run(batch, chemistry_plan=plan, proposal=chosen, inventory=inventory)
    return DesignCandidate(proposal=chosen, chemistry_plan=plan, council_rounds=log.total_rounds, deliberation_log=log,
            human_explanation=chief["justification"], safety_report={"status": "screening_hypothesis", "limitations": chief.get("limitations", [])}), calculations
