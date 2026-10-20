"""Atomic post-council engineering and inventory finalization.

The council may select chemistry and operating targets, but its JSON is not a
deployable hardware specification until geometry, gas basis, pumps, tubing,
reactor inventory, and residence time have been reconciled together.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from flora_translate.design_calculator import DesignCalculations, DesignCalculator
from flora_translate.inventory_constraints import (
    enforce_reactor_inventory,
    equipment_system_compatible,
    selected_process_systems,
)
from flora_translate.residence_time_basis import (
    INLET_STP_BASIS,
    normalize_residence_time_basis,
)
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    FlowProposal,
    LabInventory,
)


def finalize_design(
    proposal: FlowProposal,
    *,
    batch_record: BatchRecord,
    chemistry_plan: ChemistryPlan | None,
    analogies: list[dict] | None,
    inventory: LabInventory | None,
) -> tuple[FlowProposal, DesignCalculations, dict[str, Any]]:
    """Return one inventory-closed proposal and its matching calculations."""

    calculator = DesignCalculator()
    current = proposal.model_copy(deep=True)

    # First pass discovers a missing gas phase from protocol/chemistry context.
    discovery_calc = _calculate(
        calculator,
        current,
        batch_record=batch_record,
        chemistry_plan=chemistry_plan,
        analogies=analogies,
        inventory=inventory,
    )
    current = calculator.annotate_proposal_with_calculations(
        current, discovery_calc
    )

    current, first_inventory_report = enforce_reactor_inventory(
        current, inventory
    )
    reconciled_calc = _calculate(
        calculator,
        current,
        batch_record=batch_record,
        chemistry_plan=chemistry_plan,
        analogies=analogies,
        inventory=inventory,
    )
    current = calculator.annotate_proposal_with_calculations(
        current, reconciled_calc
    )

    # Annotation can update geometry after pressure checks. Snap once more so
    # the serialized proposal is guaranteed to name one physical reactor.
    current, final_inventory_report = enforce_reactor_inventory(
        current, inventory
    )
    final_calc = _calculate(
        calculator,
        current,
        batch_record=batch_record,
        chemistry_plan=chemistry_plan,
        analogies=analogies,
        inventory=inventory,
    )

    report = _validation_report(
        current,
        final_calc,
        inventory,
        first_inventory_report=first_inventory_report,
        final_inventory_report=final_inventory_report,
    )
    constraints = dict(current.inventory_constraints or {})
    constraints["final_validation"] = report
    current.inventory_constraints = constraints
    return current, final_calc, report


def _calculate(
    calculator: DesignCalculator,
    proposal: FlowProposal,
    *,
    batch_record: BatchRecord,
    chemistry_plan: ChemistryPlan | None,
    analogies: list[dict] | None,
    inventory: LabInventory | None,
) -> DesignCalculations:
    return calculator.run(
        batch_record,
        chemistry_plan=chemistry_plan,
        proposal=proposal,
        inventory=inventory,
        analogies=analogies or [],
        target_flow_rate_mL_min=proposal.flow_rate_mL_min or None,
        target_tubing_ID_mm=proposal.tubing_ID_mm or None,
        target_residence_time_min=proposal.residence_time_min or None,
    )


def _validation_report(
    proposal: FlowProposal,
    calculations: DesignCalculations,
    inventory: LabInventory | None,
    *,
    first_inventory_report: dict[str, Any],
    final_inventory_report: dict[str, Any],
) -> dict[str, Any]:
    reactor_match = _reactor_match(proposal, inventory)
    pump_feasible = _pump_feasible(proposal, inventory)
    tubing_feasible = _tubing_feasible(proposal, inventory)
    closure_error = _geometry_closure_error(proposal)
    geometry_closure = closure_error is not None and closure_error <= 0.02
    calculation_match = (
        abs(calculations.reactor_volume_mL - proposal.reactor_volume_mL)
        <= max(0.02, 0.01 * max(proposal.reactor_volume_mL, 1.0))
        and abs(calculations.tubing_ID_mm - proposal.tubing_ID_mm) <= 1e-6
    )
    gas_required = bool(calculations.is_gas_liquid)
    gas_complete = (not gas_required) or _gas_bookkeeping_complete(proposal)

    checks = {
        "reactor_inventory_match": reactor_match,
        "pump_flow_feasible": pump_feasible,
        "tubing_feasible": tubing_feasible,
        "geometry_closure": geometry_closure,
        "calculation_matches_serialized_design": calculation_match,
        "gas_bookkeeping_complete": gas_complete,
        "gas_time_basis_consistent": _gas_primary_basis_is_inlet_stp(
            proposal
        ),
    }
    unresolved = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "flowpilot_final_validation_v1.0",
        "status": "ready" if not unresolved else "screen_required",
        "checks": checks,
        "unresolved_reasons": unresolved,
        "geometry_relative_error": closure_error,
        "gas_required": gas_required,
        "inventory_enforcement": {
            "first_pass": first_inventory_report,
            "final_pass": final_inventory_report,
        },
        "calculation_snapshot": {
            "reactor_volume_mL": calculations.reactor_volume_mL,
            "tubing_ID_mm": calculations.tubing_ID_mm,
            "flow_rate_mL_min": calculations.flow_rate_mL_min,
            "residence_time_min": calculations.residence_time_min,
            "residence_time_inlet_min": calculations.residence_time_inlet_min,
            "residence_time_in_channel_min": (
                calculations.residence_time_in_channel_min
            ),
            "gas_flow_sccm": calculations.gas_flow_sccm,
            "gas_flow_actual_mL_min": calculations.gas_flow_actual_mL_min,
            "target_gas_equiv_inlet": calculations.target_gas_equiv_inlet,
            "gas_equiv_supplied": calculations.gas_equiv_supplied,
            "consistent": calculations.consistent,
            "consistency_notes": calculations.consistency_notes,
        },
    }


def _reactor_match(
    proposal: FlowProposal,
    inventory: LabInventory | None,
) -> bool:
    if inventory is None or not inventory.reactors:
        return True
    return any(
        abs(float(reactor.volume_mL) - proposal.reactor_volume_mL) <= 1e-6
        and abs(float(reactor.ID_mm) - proposal.tubing_ID_mm) <= 1e-6
        and str(reactor.material).lower()
        == str(proposal.tubing_material).lower()
        for reactor in inventory.reactors
    )


def _pump_feasible(
    proposal: FlowProposal,
    inventory: LabInventory | None,
) -> bool:
    if inventory is None or not inventory.pumps:
        return True
    selected_systems = selected_process_systems(proposal, inventory)
    compatible = [
        pump
        for pump in inventory.pumps
        if equipment_system_compatible(pump, selected_systems)
    ]
    return any(
        pump.min_flow_rate_mL_min - 1e-12
        <= proposal.flow_rate_mL_min
        <= pump.max_flow_rate_mL_min + 1e-12
        and proposal.BPR_bar <= pump.max_pressure_bar + 1e-12
        for pump in compatible
    )


def _tubing_feasible(
    proposal: FlowProposal,
    inventory: LabInventory | None,
) -> bool:
    if inventory is None or not inventory.tubing:
        return True
    return any(
        tubing.material.lower() == proposal.tubing_material.lower()
        and abs(tubing.ID_mm - proposal.tubing_ID_mm) <= 1e-6
        and proposal.BPR_bar <= tubing.max_pressure_bar + 1e-12
        and proposal.temperature_C <= tubing.max_temperature_C + 1e-12
        for tubing in inventory.tubing
    )


def _geometry_closure_error(proposal: FlowProposal) -> float | None:
    volume = float(proposal.reactor_volume_mL or 0.0)
    liquid_flow = float(proposal.flow_rate_mL_min or 0.0)
    if volume <= 0 or liquid_flow <= 0:
        return None

    gas_sccm = 0.0
    gas_actual = 0.0
    for stream in proposal.streams or []:
        if str(stream.phase or "").lower() != "gas":
            continue
        gas_sccm += float(stream.gas_flow_sccm or 0.0)
        gas_actual += float(stream.gas_flow_actual_mL_min or 0.0)

    basis = normalize_residence_time_basis(proposal.residence_time_basis)
    if basis == INLET_STP_BASIS:
        tau = float(
            proposal.residence_time_inlet_min
            or proposal.residence_time_min
            or 0.0
        )
        expected_volume = tau * (liquid_flow + gas_sccm)
    elif gas_actual > 0:
        tau = float(
            proposal.residence_time_in_channel_min
            or proposal.residence_time_min
            or 0.0
        )
        expected_volume = tau * (liquid_flow + gas_actual)
    else:
        tau = float(proposal.residence_time_min or 0.0)
        expected_volume = tau * liquid_flow

    if tau <= 0 or expected_volume <= 0:
        return None
    return abs(expected_volume - volume) / volume


def _gas_bookkeeping_complete(proposal: FlowProposal) -> bool:
    gas_streams = [
        stream
        for stream in proposal.streams or []
        if str(stream.phase or "").lower() == "gas"
    ]
    if not gas_streams:
        return False
    metrics = proposal.multiphase_metrics or {}
    return (
        all(
            float(stream.gas_flow_sccm or 0.0) > 0
            and float(stream.gas_flow_actual_mL_min or 0.0) > 0
            and float(stream.molar_equiv or 0.0) > 0
            for stream in gas_streams
        )
        and float(proposal.residence_time_inlet_min or 0.0) > 0
        and float(proposal.residence_time_in_channel_min or 0.0) > 0
        and float(metrics.get("target_gas_equiv_inlet") or 0.0) > 0
        and float(metrics.get("gas_equiv_supplied") or 0.0) > 0
    )


def _gas_primary_basis_is_inlet_stp(proposal: FlowProposal) -> bool:
    # Compatibility name; time basis is independent of STP gas metering.
    from flora_translate.residence_time_basis import gas_time_basis_consistent
    return gas_time_basis_consistent(proposal)


def calculations_payload(calculations: DesignCalculations) -> dict[str, Any]:
    """Public serializer kept here so every caller stores the same payload."""

    return asdict(calculations)
