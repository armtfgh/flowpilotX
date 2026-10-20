"""Recompute realized reactors independently, not as one fictitious mixed coil."""

from dataclasses import asdict
from copy import deepcopy

from flora_translate.design_calculator import DesignCalculator


def calculate_final_stages(proposal, batch_record, chemistry_plan, inventory, analogies=None):
    stages = proposal.stage_parameters or [{
        "stage_number": 1, "reactor_volume_mL": proposal.reactor_volume_mL,
        "flow_rate_mL_min": proposal.flow_rate_mL_min, "d_mm": proposal.tubing_ID_mm,
        "temperature_C": proposal.temperature_C,
        "residence_time_inlet_min": proposal.residence_time_inlet_min or proposal.residence_time_min,
    }]
    records = []
    for stage in stages:
        number = stage.get("stage_number", 1)
        active = [s.model_copy(deep=True) for s in proposal.streams if (s.introduction_stage or 1) <= number]
        stage_gas = stage.get("Q_gas_sccm", stage.get("gas_flow_sccm"))
        if stage_gas == 0:
            active = [s for s in active if s.phase != "gas"]
        if len([s for s in active if s.phase == "gas"]) > 1:
            records.append({"stage_number": number, "status": "review", "source": "realized stage",
                            "error": "Mixed gas-feed transport annotations require an explicit combined-gas model. Canonical inlet setpoints remain unchanged."})
            continue
        local = proposal.model_copy(deep=True)
        local.streams = active
        local.stage_parameters = []
        local.temperature_C = stage.get("temperature_C", proposal.temperature_C)
        local.reactor_volume_mL = stage.get("reactor_volume_mL", stage.get("V_R_mL"))
        local.flow_rate_mL_min = stage.get("Q_liquid_mL_min", stage.get("flow_rate_mL_min"))
        local.tubing_ID_mm = stage.get("d_mm", stage.get("tubing_ID_mm", proposal.tubing_ID_mm))
        local.residence_time_min = stage.get("residence_time_inlet_min", stage.get("residence_time_min"))
        local.residence_time_inlet_min = local.residence_time_min
        local.residence_time_basis = "inlet/STP apparent residence time" if any(s.phase == "gas" for s in active) else "liquid-only stage residence time"
        # The legacy estimator's substrate basis must not become the sum of
        # concentrations after downstream dilution. Preserve molar throughput.
        limiting = next((s for s in active if s.phase != "gas" and s.molar_equiv == 1 and s.concentration_M), None)
        if limiting and local.flow_rate_mL_min:
            local.concentration_M = limiting.concentration_M * limiting.flow_rate_mL_min / local.flow_rate_mL_min
        assigned = set(stage.get("pump_equipment_ids") or [])
        assigned.update(s.pump_equipment_id for s in active if s.phase != "gas")
        local_inventory = inventory.model_copy(deep=True)
        selected = [p for p in inventory.pumps if p.equipment_id in assigned]
        if selected:
            local_inventory.pumps = selected
        local_batch = batch_record.model_copy(deep=True)
        chemistry_stage = next((s for s in chemistry_plan.stages if s.stage_number == number), None) if chemistry_plan else None
        batch_time_known = len(stages) == 1
        if len(stages) > 1:
            local_batch.reaction_time_h = chemistry_stage.batch_time_h if chemistry_stage else None
            batch_time_known = bool(local_batch.reaction_time_h)
        try:
            values = asdict(DesignCalculator().run(
                local_batch, chemistry_plan=chemistry_plan, proposal=local,
                inventory=local_inventory, analogies=analogies, frozen_geometry=True,
            ))
            if not batch_time_known:
                values["intensification_factor"] = None
            values["batch_time_source"] = "stage batch_time_h" if chemistry_stage else "single-stage batch record" if len(stages) == 1 else "not recorded"
            records.append({"stage_number": number, "status": "calculated", "source": "realized stage; frozen geometry", "calculations": values})
        except (ValueError, TypeError) as exc:
            # Do not publish annotations from the wrong geometry as final.
            records.append({"stage_number": number, "status": "review", "source": "realized stage", "error": str(exc)})
    return {"schema_version": "flowpilot_final_stage_engineering_v1", "stages": deepcopy(records),
            "complete": all(r["status"] == "calculated" for r in records)}
