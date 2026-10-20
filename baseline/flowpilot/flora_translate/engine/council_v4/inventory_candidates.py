"""Generate single-stage council candidates on the hardware actually available.

This uses the production realization solver before review, and retains its full
input/output so a chosen candidate can be delivered without changing apparatus.
It contains no benchmark chemistry names, reference yields, or judge feedback.
"""
from copy import deepcopy
import math

from flora_translate.design_realizer import realize_executable_design
from flora_translate.engine.sampling import compute_metrics, format_candidate_table
from flora_translate.engine.flow_value import attach_flow_sense_reports
from flora_translate.engine.tools import calculate_bpr_required
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory
from flora_translate.component_identity import component_key, has_stationary_catalytic_role


def _apply_component_placements(proposal, plan, inventory, placements):
    """Accept only exact, evidenced placement repairs on available packed beds."""
    if not placements:
        return
    if not isinstance(placements, list):
        raise ValueError("component_placements must be a list")
    records = []
    for request in placements:
        if not isinstance(request, dict) or request.get("placement") != "stationary_reactor_phase":
            raise ValueError("Unsupported component placement repair")
        name = str(request.get("component") or "")
        reagent = next((r for r in (plan.reagents if plan else []) if component_key(r.name) == component_key(name)), None)
        reactor = next((r for r in inventory.reactors if r.equipment_id == request.get("equipment_id")), None)
        if not reagent or not reactor or "packed" not in reactor.type.lower():
            raise ValueError("Placement repair requires an existing chemical and available packed reactor")
        if not has_stationary_catalytic_role(reagent.role + " " + reagent.notes, declared_loading=reagent.equiv_or_loading):
            raise ValueError("Stationary placement is not supported by the authoritative chemistry plan")
        selected = proposal.inventory_selection.get("equipment_id")
        if selected and selected != reactor.equipment_id:
            raise ValueError("Placement repair changes the reviewed reactor assignment")
        if not str(request.get("reason") or "").strip():
            raise ValueError("Placement repair requires a reason")
        records.append({**request, "component": reagent.name, "source": "chemistry_review_checked_against_plan_and_inventory"})
    proposal.inventory_constraints["council_component_placements"] = records


def materialize(context, *, flow=None, patch=None, candidate_id=1):
    context = deepcopy(context)
    proposal = FlowProposal.model_validate(context["proposal"])
    patch = patch or {}
    geometry = {"d_mm": "tubing_ID_mm", "V_R_mL": "reactor_volume_mL", "tubing_material": "tubing_material"}
    for key, field in geometry.items():
        if key in patch and str(patch[key]) != str(getattr(proposal, field)):
            try:
                equal = math.isclose(float(patch[key]), float(getattr(proposal, field)), rel_tol=1e-5)
            except (TypeError, ValueError):
                equal = False
            if not equal:
                raise ValueError(f"Revision changes fixed inventory {key} from {getattr(proposal, field)} to {patch[key]}")
    if "tau_min" in patch and "Q_mL_min" not in patch:
        flow = proposal.reactor_volume_mL / float(patch["tau_min"])
    proposal.flow_rate_mL_min = float(patch.get("Q_mL_min", flow or proposal.flow_rate_mL_min))
    for key in ["temperature_C", "concentration_M", "BPR_bar"]:
        if key in patch:
            setattr(proposal, key, float(patch[key]))
    batch = BatchRecord.model_validate(context["batch"])
    plan = ChemistryPlan.model_validate(context["plan"]) if context.get("plan") else None
    inventory = LabInventory.model_validate(context["inventory"])
    _apply_component_placements(proposal, plan, inventory, patch.get("component_placements"))
    realized, audit, validation = realize_executable_design(
        proposal, batch_record=batch, chemistry_plan=plan, inventory=inventory,
        hard_constraints=context.get("constraints"),
    )
    if audit["status"] != "complete" or not all(validation["checks"].values()):
        raise ValueError(f"Inventory realization failed: {audit.get('unresolved_requirements', audit.get('issues', []))}")
    for key in ["temperature_C", "concentration_M", "BPR_bar"]:
        if key in patch and not math.isclose(float(getattr(realized, key)), float(patch[key]), rel_tol=1e-5):
            raise ValueError(f"Revision {key}={patch[key]} is not supported by the inventory")
    if patch.get("tau_min") and not math.isclose(realized.reactor_volume_mL / realized.flow_rate_mL_min, float(patch["tau_min"]), rel_tol=.02):
        raise ValueError("Revised time cannot be achieved with the inventory-constrained feed settings")

    # Council geometry uses the nominal liquid-volume index. Gas STP/channel
    # indices are separate fields, never inferred liquid holdup measurements.
    tau = realized.reactor_volume_mL / realized.flow_rate_mL_min
    row = compute_metrics(tau, realized.tubing_ID_mm, realized.flow_rate_mL_min,
        batch.solvent or "", realized.temperature_C, realized.concentration_M,
        context["assumed_MW"], context["IF_used"], context["tau_kinetics_min"],
        context["pump_max_bar"], context["is_photochem"], tau_source="inventory_realization",
        is_gas_liquid=False, BPR_bar=realized.BPR_bar)
    packed = "packed" in realized.reactor_type.lower()
    gases = [s for s in realized.streams if s.phase == "gas"]
    pressure = calculate_bpr_required(realized.temperature_C, batch.solvent or "", row["delta_P_bar"], bool(gases))
    warnings = ["Conversion, liquid holdup and pressure drop require measurement; computed conversion is an unvalidated screening proxy."]
    if packed:
        warnings.append("Empty-tube pressure/mixing calculations do not model this packed bed; catalyst loading, voidage and particle size remain required commissioning data.")
    elif gases:
        warnings.append("Single-phase pressure-drop proxy excludes multiphase slip and slug losses; verify gas-liquid contacting and pressure stability.")
    violations = []
    # Preserve the existing minimum/margin distinction, rather than the
    # sampler's inconsistent blanket 5-bar gas-liquid cutoff.
    minimum_bpr = max(0., pressure["P_min_bar"] - (2. if gases else .5))
    if realized.BPR_bar + .1 < minimum_bpr:
        violations.append(f"BPR {realized.BPR_bar:g} bar is below calculated minimum {minimum_bpr:g} bar")
    if realized.BPR_bar < pressure["P_min_bar"]:
        warnings.append(f"BPR below the heuristic {pressure['P_min_bar']:g} bar recommended margin; assess actual pressure stability without inventing an unavailable setpoint.")
    if violations:
        raise ValueError("; ".join(violations))
    context["proposal"] = realized.model_dump()
    row.update(id=candidate_id, tau_min=tau, Q_mL_min=realized.flow_rate_mL_min,
        V_R_mL=realized.reactor_volume_mL, d_mm=realized.tubing_ID_mm,
        reactor_type=realized.reactor_type, equipment_id=realized.inventory_selection.get("equipment_id"),
        temperature_C=realized.temperature_C, concentration_M=realized.concentration_M,
        BPR_bar=realized.BPR_bar, BPR_min_bar=minimum_bpr, required_bpr_bar=pressure["P_min_bar"],
        tubing_material=realized.tubing_material, batch_time_min=context["batch_time_min"],
        residence_time_basis="nominal liquid-volume space time; not measured phase contact time",
        residence_time_inlet_min=realized.residence_time_inlet_min,
        residence_time_in_channel_min=realized.residence_time_in_channel_min,
        gas_flow_sccm=sum(s.gas_flow_sccm or 0 for s in gases),
        gas_flow_actual_mL_min=sum(s.gas_flow_actual_mL_min or 0 for s in gases),
        target_gas_equiv_inlet=realized.multiphase_metrics.get("target_gas_equiv_inlet", 0),
        gas_reagent_fraction=realized.multiphase_metrics.get("gas_reagent_fraction", 1),
        hydraulic_model="empty-tube proxy; packed-bed hydraulics unresolved" if packed else "liquid-only tube proxy; multiphase corrections unresolved",
        streams=[s.model_dump() for s in realized.streams],
        stationary_components=realized.inventory_constraints.get("stationary_components", []),
        warnings=warnings, hard_gate_flags=[], feasible=True, violations=[],
        inventory_realized=True, _realization_context=context,
        realized_proposal=realized.model_dump(), realization_audit=audit, realization_validation=validation)
    row["model_validity"] = {
        "kinetics": {"applicable": True, "validated": False, "evidence": [],
                     "basis": "first-order t90 inferred from batch/class/analogy; not measured flow kinetics"},
        "mixing": {"applicable": not packed and not gases, "validated": False, "evidence": [],
                   "basis": "cross-tube molecular diffusion; packed-bed and segmented-flow transport are not modeled"},
        "hydraulics": {"applicable": not packed and not gases, "validated": False, "evidence": [],
                       "basis": row["hydraulic_model"]},
    }
    # compute_metrics' V_liquid field is a reference volume, not a holdup here.
    row.pop("liquid_holdup_volume_mL", None)
    row["gas_holdup"] = 0.0
    row["gas_holdup_status"] = "unmeasured; zero placeholder is not a holdup estimate"
    return row


def generate_inventory_candidates(proposal, batch, plan, inventory, constraints, *, count, tau_kinetics_min, assumed_MW, IF_used, pump_max_bar, is_photochem, batch_time_min):
    context = {"proposal": proposal.model_dump(), "batch": batch.model_dump(),
        "plan": plan.model_dump() if plan else None, "inventory": inventory.model_dump(),
        "constraints": constraints, "tau_kinetics_min": tau_kinetics_min,
        "assumed_MW": assumed_MW, "IF_used": IF_used, "pump_max_bar": pump_max_bar,
        "is_photochem": is_photochem, "batch_time_min": batch_time_min}
    rows, rejected, seen = [], [], set()
    # Bounded feed exploration on the same fixed hardware. Realization enforces
    # pump increments, shared stoichiometry, gas capability, and temperature.
    for factor in [1., .8, 1.25, .5, 1.5, .25, 2., .1, 3., 4., 5., 10., 20.]:
        try:
            row = materialize(context, flow=max(proposal.flow_rate_mL_min, 1e-6) * factor, candidate_id=len(rows)+1)
        except ValueError as exc:
            rejected.append({"flow_factor": factor, "reason": str(exc)})
            continue
        identity = (row["equipment_id"], round(row["Q_mL_min"], 8), row["BPR_bar"], row["temperature_C"])
        if identity in seen:
            continue
        seen.add(identity); rows.append(row)
        if len(rows) >= count:
            break
    attach_flow_sense_reports(rows, batch_time_min=batch_time_min, batch_concentration_M=proposal.concentration_M, solvent_name=batch.solvent or "")
    return {"survivors": rows, "all_candidates": rows, "disqualified": [],
        "killed_designs": rejected, "table_markdown": format_candidate_table(rows, max_rows=count),
        "strategy_reasoning": "Realize each feed setting on the declared reactor and delivery equipment before specialist review.",
        "pool_metadata": {"pool_quality": "INVENTORY_BOUND", "candidates_generated": len(rows), "rejected": rejected},
        "design_envelope_preliminary": {}}
