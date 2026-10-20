"""Read-only, stage-aware reporting from the frozen final design.

Legacy gas_flow_sccm keys hold mL/min at STP. Never substitute a gas stream's
generic flow_rate_mL_min: older records use that key for compressed gas.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from flora_translate.residence_time_basis import normalize_residence_time_basis


REPORT_VERSION = "flowpilot_result_report_v1"


def _number(*values):
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
            if math.isfinite(number):
                return number
        except (TypeError, ValueError):
            pass
    return None


def build_result_report(result: dict[str, Any]) -> dict[str, Any]:
    final = result.get("final_design") or {}
    accepted = final.get("status") == "executable"
    params = final.get("parameters") or {}
    streams = []
    manifest = {x.get("equipment_id"): x for x in final.get("instrument_manifest", [])}
    for stream in final.get("streams", []) if accepted else []:
        gas = stream.get("phase") == "gas"
        equipment = stream.get("pump_equipment_id")
        streams.append({
            "label": stream.get("stream_label"), "phase": stream.get("phase"),
            "contents": deepcopy(stream.get("contents") or []),
            "introduction_stage": stream.get("introduction_stage"),
            "flow_mL_min": _number(stream.get("gas_flow_stp_mL_min"), stream.get("gas_flow_sccm")) if gas else _number(stream.get("flow_rate_mL_min")),
            "flow_basis": "inlet/STP" if gas else "liquid",
            "concentration_M": stream.get("concentration_M"),
            "equiv": stream.get("molar_equiv"),
            "reagent_fraction": stream.get("gas_reagent_mole_fraction"),
            "equipment_id": equipment,
            "equipment_name": (manifest.get(equipment) or {}).get("name") or equipment,
            "source": "final_design.streams",
        })
    stages = []
    names = {s.get("stage_number"): s.get("stage_name") for s in (result.get("chemistry_plan") or {}).get("stages", [])}
    source_stages = final.get("stages") or []
    if accepted and not source_stages and params:
        gas_flows = [s["flow_mL_min"] for s in streams if s["phase"] == "gas"]
        source_stages = [{**params, "stage_number": 1,
                          "Q_gas_sccm": sum(gas_flows) if all(q is not None for q in gas_flows) else None}]
    issues = []
    for index, stage in enumerate(source_stages if accepted else []):
        number = stage.get("stage_number") or index + 1
        ql = _number(stage.get("Q_liquid_mL_min"), stage.get("flow_rate_mL_min"))
        qg = _number(stage.get("gas_flow_stp_mL_min"), stage.get("Q_gas_sccm"), stage.get("gas_flow_sccm"))
        if qg is None and not any(s["phase"] == "gas" for s in streams):
            qg = 0.0
        volume = _number(stage.get("reactor_volume_mL"), stage.get("V_R_mL"))
        tau = _number(stage.get("residence_time_inlet_min"))
        if tau is None and (qg == 0 or normalize_residence_time_basis(stage.get("residence_time_basis")) == "inlet_stp"):
            tau = _number(stage.get("residence_time_min"))
        total = ql + qg if ql is not None and qg is not None else None
        calculated = volume / total if volume is not None and total and total > 0 else None
        closed = None if tau is None or calculated is None else math.isclose(tau, calculated, rel_tol=0.002, abs_tol=0.001)
        if closed is not True:
            issues.append(f"Stage {number}: inlet/STP V/Q closure {'is inconsistent' if closed is False else 'cannot be verified from saved fields'}.")
        stages.append({
            "number": number, "name": stage.get("stage_name") or names.get(number) or f"Reaction stage {number}",
            "reactor": stage.get("reactor_name") or stage.get("reactor_equipment_id") or params.get("reactor_type"),
            "equipment_id": stage.get("reactor_equipment_id"),
            "volume_mL": volume, "liquid_flow_mL_min": ql, "gas_flow_stp_mL_min": qg,
            "total_inlet_flow_mL_min": total, "residence_time_min": tau,
            "residence_basis": "not verified" if qg is None else "inlet/STP apparent" if qg > 0 else "liquid",
            "temperature_C": stage.get("temperature_C"),
            "pressure_bar": stage.get("BPR_bar", params.get("BPR_bar")),
            "pressure_basis": params.get("BPR_basis") or "not recorded",
            "wavelength_nm": stage.get("wavelength_nm"),
            "tubing_ID_mm": _number(stage.get("d_mm"), stage.get("tubing_ID_mm")),
            "material": stage.get("material") or stage.get("tubing_material"),
            "closure": closed, "calculated_residence_min": calculated,
            "source": f"final_design.stages[{index}]" if final.get("stages") else "final_design.parameters",
        })
    return {
        "schema_version": REPORT_VERSION, "source": "final_design",
        "canonical_sha256": final.get("canonical_sha256"), "status": final.get("status"),
        "gas_reference": {"flow_unit": "mL/min at STP", "temperature_K": 273.15, "pressure_bar": 1.01325},
        "stages": stages, "streams": streams, "issues": issues,
        "flow_operability": deepcopy(final.get("flow_operability")),
        "responses": _response_audit(result, stages, streams),
    }


def _response_audit(result, stages, streams):
    package = result.get("intake_package") or {}
    questions = {q["question_id"]: q for q in package.get("question_log", [])}
    history: dict[str, list] = {}
    for answer in package.get("answers", []):
        history.setdefault(answer["question_id"], []).append(answer)
    paths = {
        "Q-BATCH-001": "raw_protocol", "Q-OBJ-001": "objective",
        "Q-CHEM-001": "chemistry_identity_confirmation", "Q-HIST-001": "historical_data",
        "Q-INV-001": "inventory_constraints", "Q-CONSTR-001": "operating_limits",
        "Q-HYP-001": "hypotheses", "Q-PREF-001": "output_preferences",
    }
    rows = []
    for qid in dict.fromkeys([*questions, *history]):
        question = questions.get(qid) or {"question_id": qid, "question": "Question text not stored"}
        answer_history = history.get(qid) or []
        answer = answer_history[-1] if answer_history else None
        path = question.get("target_path") or paths.get(qid) or ""
        bound = package
        for part in path.split(".") if path else []:
            bound = bound.get(part) if isinstance(bound, dict) else None
        if not path:
            bound = None
        status = answer.get("status") if answer else "stored in package" if bound not in (None, "", [], {}) else "not recorded"
        evidence = []
        assessment = "context only"
        if status == "unavailable":
            assessment = "explicitly unavailable"
            evidence.append("No supplied evidence or constraint is claimed for this answer.")
        elif qid == "Q-INV-001":
            allocation = result.get("inventory_allocation") or {}
            complete = (allocation.get("checks") or {}).get("all_required_operations_assigned") is True
            assessment = "assignment checked" if complete else "review allocation"
            evidence.append(f"Allocation status: {allocation.get('status') or 'not recorded'}. Assignment does not establish equipment safety or confirm assumed accessories.")
            evidence.extend(f"Stage {s['number']}: {s['reactor']} ({s['equipment_id']})" for s in stages)
            evidence.extend(f"Stream {s['label']}: {s['equipment_name']}" for s in streams)
        elif qid == "Q-CONSTR-001":
            checks = (result.get("final_validation") or {}).get("checks") or {}
            evidence.extend(f"{k}: {'passed' if v is True else 'failed' if v is False else v}" for k, v in checks.items() if any(t in k for t in ("inventory", "temperature", "pressure", "flow", "constraint")))
            if isinstance(bound, dict) and bound.get("inline_degasser_available") is False:
                ops = (result.get("process_topology") or {}).get("unit_operations", [])
                found = any("degas" in str(o.get("op_type", "")).lower() or "deoxygenation" in str(o.get("op_type", "")).lower() for o in ops)
                evidence.append("No inline degasser: " + ("not verified; no final topology" if not ops else "violated" if found else "confirmed in final topology"))
            assessment = "recorded checks; review unstructured limits"
        elif qid.startswith("Q-GAS-"):
            evidence.extend(f"Stream {s['label']}: {', '.join(map(str, s['contents']))}, {s['flow_mL_min']} mL/min at STP, {s['equiv']} equiv, introduced at stage {s['introduction_stage']}" for s in streams if s["phase"] == "gas")
            assessment = "final setpoints available" if evidence else "no gas setpoints published"
        elif qid == "Q-BATCH-001":
            evidence.append("Protocol retained in intake_package.raw_protocol; parsed facts in batch_record; chemistry analysis in chemistry_plan.")
        elif qid == "Q-OBJ-001":
            evidence.append("Objective included in the frozen council context; this is a screening design, not proof that the objective was achieved.")
        elif qid == "Q-HYP-001":
            evidence.append("Hypotheses are supplied as ideas to test, not measured facts. No per-hypothesis causal attribution is stored.")
        elif qid == "Q-CHEM-001":
            plan = result.get("chemistry_plan") or {}
            for key in ("reaction_class", "reaction_type", "transformation", "substrates", "products"):
                if plan.get(key):
                    evidence.append(f"chemistry_plan.{key}: {plan[key]}")
            evidence.append("Identity confirmation is retained as input; chemical correctness is not proven by storing the answer.")
        elif qid.startswith("Q-PHOTO-"):
            evidence.extend(f"Stage {s['number']}: final wavelength {s['wavelength_nm']} nm." for s in stages if s["wavelength_nm"] is not None)
            assessment = "final light settings available" if evidence else "no light settings published"
            evidence.append("Compare these settings with the normalized request; inventory substitutions require review.")
        elif qid.startswith("Q-MULTI-"):
            evidence.extend(f"Stage {s['number']}: {s['name']}; {s['reactor']}; {s['temperature_C']} C; {s['residence_time_min']} min ({s['residence_basis']})." for s in stages)
            evidence.extend(f"Stream {s['label']} enters stage {s['introduction_stage']}." for s in streams)
            assessment = "final stage sequence available" if stages else "no stage sequence published"
        elif qid == "Q-HIST-001":
            calibration = result.get("evidence_calibration") or (result.get("proposal") or {}).get("evidence_calibration")
            evidence.append("Historical measurements are retained in intake_package.historical_data, separate from hypotheses.")
            evidence.append("Evidence calibration is recorded in the result." if calibration else "No independent evidence-calibration record is stored; a causal effect on the design is not verified.")
        else:
            evidence.append("Stored input binding shown below. No independent per-answer outcome verification is recorded.")
        relevant = []
        for decision in (result.get("design_realization") or {}).get("decisions", []):
            name = str(decision.get("decision") or "")
            if (qid.startswith("Q-GAS-") and "gas" in name) or qid in {"Q-INV-001", "Q-CONSTR-001"}:
                relevant.append(deepcopy(decision))
        if any(d.get("confirmation_required") for d in relevant):
            assessment = "chemist confirmation required"
        rows.append({"question_id": qid, "question": question.get("question"),
                     "status": status, "answer": answer.get("answer") if answer else bound,
                     "answer_history": deepcopy(answer_history), "target_path": f"intake_package.{path}" if path else "not recorded",
                     "bound_value": deepcopy(bound), "assessment": assessment, "evidence": evidence,
                     "decisions": relevant})
    return rows


def attach_result_report(result):
    """Return a presentation-enriched copy without rewriting archived evidence."""
    return {**result, "result_report": build_result_report(result)}
