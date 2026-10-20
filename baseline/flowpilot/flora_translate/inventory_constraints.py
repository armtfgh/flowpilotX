"""Deterministic enforcement of discrete lab inventory choices.

The LLM and council can reason about inventory, but final hardware selection
must be enforced deterministically. This module snaps flow proposals to the
available reactor options and recalculates flow rates so volume/tau closure is
preserved.
"""

from __future__ import annotations

import re
from typing import Any

from flora_translate.residence_time_basis import (
    INLET_STP_BASIS,
    UNKNOWN_BASIS,
    actual_gas_flow_from_stp,
    gas_equiv_from_stp_flow,
    normalize_residence_time_basis,
    residence_time_basis_label,
    stp_gas_flow_from_actual,
    stp_gas_flow_for_equiv,
)
from flora_translate.schemas import FlowProposal, LabInventory, PumpSpec, ReactorSpec


P_STP_BAR = 1.01325
T_STP_K = 273.15


def available_pressure_settings(inventory: LabInventory | None) -> list[float]:
    """Return all discrete pressure settings declared by legacy or v3 inventory."""

    if inventory is None:
        return []
    values = [float(value) for value in inventory.BPR_available]
    for controller in inventory.pressure_controllers:
        if str(controller.service_status).lower() not in {"available", "ready", "in_service", "in service"}:
            continue
        values.extend(float(value) for value in controller.setpoints_bar)
    return sorted(set(values))


def inventory_prompt_block(inventory: LabInventory | None) -> str:
    """Return a compact prompt block describing hard inventory constraints."""

    if inventory is None:
        return "## Lab Inventory\nNo lab inventory was provided."

    lines = ["## Lab Inventory - hard constraints"]
    lines.append(
        f"Inventory schema: {inventory.schema_version}; deterministic physical "
        f"assignment: {'required' if inventory.strict_assignment else 'legacy compatibility'}"
    )
    if inventory.reactors:
        lines.append("Available reactors. Final reactor design must use one of these exactly:")
        for r in inventory.reactors:
            label = _reactor_label(r)
            constraints = []
            if r.allowed_temperatures_C:
                constraints.append(f"T in {r.allowed_temperatures_C} deg C")
            elif r.min_temperature_C is not None or r.max_temperature_C is not None:
                constraints.append(f"T {r.min_temperature_C}-{r.max_temperature_C} deg C")
            if r.min_concentration_M is not None or r.max_concentration_M is not None:
                constraints.append(f"C {r.min_concentration_M}-{r.max_concentration_M} M")
            if r.min_pressure_bar is not None or r.max_pressure_bar is not None:
                constraints.append(f"P {r.min_pressure_bar}-{r.max_pressure_bar} bar")
            if r.wavelength_nm:
                constraints.append(f"lambda {r.wavelength_nm:g} nm")
            if r.intensity_mW_cm2:
                constraints.append(f"{r.intensity_mW_cm2:g} mW/cm2")
            lines.append(
                f"- [{r.equipment_id}] {label}; quantity={r.quantity}; "
                f"service={r.service_status}"
                + (f" ({'; '.join(constraints)})" if constraints else "")
            )
    pressure_settings = available_pressure_settings(inventory)
    if pressure_settings:
        lines.append(f"Available BPR/pressure settings: {pressure_settings} bar")
    if inventory.pumps:
        lines.append("Available pumps. Final liquid flow must remain within the selected pump range:")
        for pump in inventory.pumps:
            systems = (
                f"; systems: {', '.join(pump.compatible_systems)}"
                if pump.compatible_systems
                else ""
            )
            lines.append(
                f"- [{pump.equipment_id}] {pump.name}: {pump.min_flow_rate_mL_min:g}-"
                f"{pump.max_flow_rate_mL_min:g} mL/min, max "
                f"{pump.max_pressure_bar:g} bar; quantity={pump.quantity}; "
                f"service={pump.service_status}{systems}"
                + (f"; setting increment={pump.flow_rate_increment_mL_min:g} mL/min (not a lower flow limit)"
                   if pump.flow_rate_increment_mL_min is not None else "")
            )
    if inventory.light_sources:
        lines.append("Available light sources:")
        for src in inventory.light_sources:
            detail = f"{src.name}: {src.wavelength_nm:g} nm"
            if src.intensity_mW_cm2:
                detail += f", {src.intensity_mW_cm2:g} mW/cm2"
            lines.append(
                f"- [{src.equipment_id}] {detail}; quantity={src.quantity}; "
                f"service={src.service_status}"
            )
    if inventory.gas_hardware:
        lines.append("Available gas hardware:")
        for item in inventory.gas_hardware:
            details = []
            if item.gas:
                details.append(f"gas: {item.gas}")
            if item.min_flow_sccm is not None or item.max_flow_sccm is not None:
                details.append(
                    f"flow: {item.min_flow_sccm or 0:g}-"
                    f"{item.max_flow_sccm or 0:g} sccm"
                )
            if item.max_pressure_bar is not None:
                details.append(f"max pressure: {item.max_pressure_bar:g} bar")
            if item.flow_rate_increment_sccm is not None:
                details.append(f"inlet/STP setting increment: {item.flow_rate_increment_sccm:g} mL/min")
            details.append(f"service: {item.service_status}")
            label = item.name + (f" ({item.type})" if item.type else "")
            lines.append(
                f"- [{item.equipment_id}] {label}; quantity={item.quantity}: "
                + "; ".join(details)
            )
    process_categories = (
        ("Mixers", inventory.mixers),
        ("Pressure controllers", inventory.pressure_controllers),
        ("Degassers", inventory.degassers),
        ("Filters", inventory.filters),
        ("Separators", inventory.separators),
        ("Connectors", inventory.connectors),
        ("Temperature controllers", inventory.temperature_controllers),
        ("Collectors", inventory.collectors),
        ("Reactor trains", inventory.reactor_trains),
        ("Safety accessories", inventory.safety_accessories),
    )
    for label, items in process_categories:
        if not items:
            continue
        lines.append(f"{label}:")
        for item in items:
            details = []
            for key in (
                "type", "material", "max_inputs", "setpoints_bar",
                "max_pressure_bar", "component_reactor_ids", "connector_ids",
            ):
                value = getattr(item, key, None)
                if value not in (None, "", []):
                    details.append(f"{key}={value}")
            lines.append(
                f"- [{item.equipment_id}] {item.name or item.__class__.__name__}; "
                f"quantity={item.quantity}; service={item.service_status}"
                + (f"; {', '.join(details)}" if details else "")
            )
    if inventory.allows_standard_reactor_connections:
        lines.append(
            "Standard interstage reactor connectors are available. Directly connecting "
            "separate declared reactors does not require a predeclared reactor train. "
            "List the standard connectors in the equipment manifest and verify their "
            "fittings and operating ratings before execution. This does not authorize "
            "new reactor volumes or bypass reactor quantity and operating limits."
        )
    return "\n".join(lines)


def enforce_reactor_inventory(
    proposal: FlowProposal,
    inventory: LabInventory | None,
) -> tuple[FlowProposal, dict[str, Any]]:
    """Snap a proposal to the closest available reactor and recalculate flow.

    The selected reactor dictates physical volume, tubing ID, material, and
    available light metadata. Residence time remains the chemistry/campaign
    decision; flow rates are recalculated to preserve that residence time.
    """

    if inventory is None or not inventory.reactors:
        return proposal, {"applied": False, "reason": "no reactor inventory"}

    selected = select_reactor_for_proposal(proposal, inventory)
    if selected is None:
        return proposal, {"applied": False, "reason": "no usable reactor options"}

    data = proposal.model_dump()
    old = {
        "reactor_volume_mL": data.get("reactor_volume_mL"),
        "tubing_ID_mm": data.get("tubing_ID_mm"),
        "tubing_material": data.get("tubing_material"),
        "flow_rate_mL_min": data.get("flow_rate_mL_min"),
        "BPR_bar": data.get("BPR_bar"),
        "temperature_C": data.get("temperature_C"),
        "concentration_M": data.get("concentration_M"),
        "wavelength_nm": data.get("wavelength_nm"),
    }

    basis = normalize_residence_time_basis(data.get("residence_time_basis"))
    if basis == UNKNOWN_BASIS:
        if _num(data.get("residence_time_inlet_min"), 0.0) > 0:
            basis = INLET_STP_BASIS
        elif _num(data.get("residence_time_in_channel_min"), 0.0) > 0:
            basis = "in_channel"
    has_reagent_gas = _proposal_has_gas(data)
    if has_reagent_gas:
        basis = INLET_STP_BASIS
    if basis == INLET_STP_BASIS:
        tau = _first_positive(
            data.get("residence_time_inlet_min"),
            data.get("residence_time_min"),
            10.0,
        )
    else:
        tau = _first_positive(
            data.get("residence_time_in_channel_min"),
            data.get("residence_time_min"),
            10.0,
        )
    volume = float(selected.volume_mL)
    selected_pump = _pump_for_reactor(selected, inventory)
    temperature = _snap_temperature(_num(data.get("temperature_C"), 25.0), selected)
    concentration = _clamp(
        _num(data.get("concentration_M"), 0.1),
        selected.min_concentration_M,
        selected.max_concentration_M,
    )
    bpr = _snap_pressure(
        _num(data.get("BPR_bar"), 0.0),
        selected,
        inventory,
        is_gas_liquid=_proposal_has_gas(data),
    )

    current_liquid_q, current_gas_actual, current_gas_sccm = _current_flow_basis(data)
    gas_stp_ratio = (
        current_gas_sccm / current_liquid_q
        if current_liquid_q > 0 and current_gas_sccm > 0
        else 0.0
    )
    gas_actual_ratio = (
        current_gas_actual / current_liquid_q
        if current_liquid_q > 0 and current_gas_actual > 0
        else _num((data.get("multiphase_metrics") or {}).get("gas_liquid_ratio"), 0.0)
    )
    # The deterministic calculator caps holdup at 0.85. Keep the scaled
    # gas/liquid ratio inside that physical model so the selected inventory
    # volume remains closed after recalculation.
    gas_actual_ratio = min(gas_actual_ratio, 0.85 / 0.15)
    target_gas_equiv = _target_gas_equiv_inlet(data)
    gas_species, gas_reagent_fraction = _gas_species_and_fraction(data)
    if target_gas_equiv > 0 and has_reagent_gas:
        gas_stp_ratio = stp_gas_flow_for_equiv(
            1.0,
            max(concentration, 1e-9),
            target_gas_equiv,
            gas_reagent_fraction,
        )
        gas_actual_ratio = actual_gas_flow_from_stp(gas_stp_ratio, temperature, bpr)

    if basis == INLET_STP_BASIS and gas_stp_ratio > 0:
        total_stp_q = volume / max(tau, 1e-9)
        liquid_q = total_stp_q / (1.0 + gas_stp_ratio)
        gas_sccm = total_stp_q - liquid_q
        gas_actual = actual_gas_flow_from_stp(gas_sccm, temperature, bpr)
        residence_time_in_channel = volume / max(liquid_q + gas_actual, 1e-9)
        data["residence_time_basis"] = residence_time_basis_label(INLET_STP_BASIS)
        data["residence_time_inlet_min"] = round(tau, 3)
        data["residence_time_in_channel_min"] = round(residence_time_in_channel, 3)
    elif gas_actual_ratio > 0:
        total_actual_q = volume / max(tau, 1e-9)
        liquid_q = total_actual_q / (1.0 + gas_actual_ratio)
        gas_actual = total_actual_q - liquid_q
        gas_sccm = stp_gas_flow_from_actual(gas_actual, temperature, bpr)
        residence_time_inlet = volume / max(liquid_q + gas_sccm, 1e-9)
        data["residence_time_basis"] = residence_time_basis_label("in_channel")
        data["residence_time_in_channel_min"] = round(tau, 3)
        data["residence_time_inlet_min"] = round(residence_time_inlet, 3)
    else:
        liquid_q = volume / max(tau, 1e-9)
        gas_actual = 0.0
        gas_sccm = 0.0
        data["residence_time_basis"] = residence_time_basis_label("liquid_only")
        data["residence_time_in_channel_min"] = round(tau, 3)
        data["residence_time_inlet_min"] = round(tau, 3)

    requested_tau = tau
    pump_flow_clamped = False
    if selected_pump is not None:
        clamped_liquid_q = _clamp(
            liquid_q,
            selected_pump.min_flow_rate_mL_min,
            selected_pump.max_flow_rate_mL_min,
        )
        pump_flow_clamped = abs(clamped_liquid_q - liquid_q) > 1e-12
        if pump_flow_clamped:
            liquid_q = clamped_liquid_q
            if basis == INLET_STP_BASIS and gas_stp_ratio > 0:
                gas_sccm = liquid_q * gas_stp_ratio
                gas_actual = actual_gas_flow_from_stp(gas_sccm, temperature, bpr)
                tau = volume / max(liquid_q + gas_sccm, 1e-9)
                residence_time_in_channel = volume / max(liquid_q + gas_actual, 1e-9)
                data["residence_time_inlet_min"] = round(tau, 3)
                data["residence_time_in_channel_min"] = round(residence_time_in_channel, 3)
            elif gas_actual_ratio > 0:
                gas_actual = liquid_q * gas_actual_ratio
                gas_sccm = stp_gas_flow_from_actual(gas_actual, temperature, bpr)
                tau = volume / max(liquid_q + gas_actual, 1e-9)
                residence_time_inlet = volume / max(liquid_q + gas_sccm, 1e-9)
                data["residence_time_in_channel_min"] = round(tau, 3)
                data["residence_time_inlet_min"] = round(residence_time_inlet, 3)
            else:
                tau = volume / max(liquid_q, 1e-9)
                data["residence_time_in_channel_min"] = round(tau, 3)
                data["residence_time_inlet_min"] = round(tau, 3)

    data["residence_time_min"] = round(tau, 3)
    data["flow_rate_mL_min"] = round(liquid_q, 5)
    data["reactor_volume_mL"] = round(volume, 4)
    data["reactor_type"] = selected.type or data.get("reactor_type") or "coil"
    data["tubing_material"] = selected.material or data.get("tubing_material") or "FEP"
    data["tubing_ID_mm"] = round(float(selected.ID_mm), 3)
    data["temperature_C"] = round(temperature, 2)
    data["concentration_M"] = round(concentration, 4)
    data["BPR_bar"] = round(bpr, 2)
    if selected.wavelength_nm:
        data["wavelength_nm"] = float(selected.wavelength_nm)
    if selected.light_source:
        setup = selected.light_source
        if selected.intensity_mW_cm2:
            setup += f" ({selected.intensity_mW_cm2:g} mW/cm2)"
        if selected.irradiation:
            setup += f", {selected.irradiation}"
        data["light_setup"] = setup

    _scale_liquid_streams(data, liquid_q, concentration)
    if gas_actual_ratio > 0:
        _sync_gas_streams(
            data,
            gas_actual,
            gas_sccm,
            species=gas_species,
        )
        if target_gas_equiv > 0:
            for stream in data.get("streams") or []:
                if _is_gas_stream(stream):
                    stream["molar_equiv"] = round(target_gas_equiv, 4)
                    break

    selected_payload = _reactor_payload(selected)
    data["inventory_selection"] = selected_payload
    data["inventory_constraints"] = {
        # Hardware rebinding must not erase an independent review decision or
        # convert stationary reactor components back into feed ingredients.
        **{key: value for key, value in (data.get("inventory_constraints") or {}).items()
           if key.startswith("council_") or key == "stationary_components"},
        "forced_reactor_inventory": True,
        "allowed_reactor_volumes_mL": sorted({float(r.volume_mL) for r in inventory.reactors}),
        "allowed_reactor_IDs_mm": sorted({float(r.ID_mm) for r in inventory.reactors}),
        "selected_reactor": selected_payload,
        "selected_pump": _pump_payload(selected_pump),
        "flow_recalculation": {
            "residence_time_basis": data["residence_time_basis"],
            "requested_residence_time_min": round(requested_tau, 3),
            "pump_flow_clamped": pump_flow_clamped,
            "pump_min_flow_rate_mL_min": (
                selected_pump.min_flow_rate_mL_min if selected_pump else None
            ),
            "pump_max_flow_rate_mL_min": (
                selected_pump.max_flow_rate_mL_min if selected_pump else None
            ),
            "liquid_flow_rate_mL_min": data["flow_rate_mL_min"],
            "gas_flow_actual_mL_min": round(gas_actual, 5) if gas_actual_ratio > 0 else None,
            "gas_flow_sccm": round(gas_sccm, 5) if gas_actual_ratio > 0 else None,
            "target_gas_equiv_inlet": round(target_gas_equiv, 4) if target_gas_equiv > 0 else None,
            "gas_equiv_supplied": (
                round(
                    gas_equiv_from_stp_flow(
                        gas_sccm,
                        liquid_q,
                        concentration,
                        gas_reagent_fraction,
                    ),
                    4,
                )
                if target_gas_equiv > 0 and gas_sccm > 0 and liquid_q > 0
                else None
            ),
            "o2_equiv_supplied": (
                round(
                    gas_equiv_from_stp_flow(
                        gas_sccm,
                        liquid_q,
                        concentration,
                        gas_reagent_fraction,
                    ),
                    4,
                )
                if (
                    gas_species in {"O2", "air"}
                    and target_gas_equiv > 0
                    and gas_sccm > 0
                    and liquid_q > 0
                )
                else None
            ),
        },
    }
    reasoning = data.setdefault("reasoning_per_field", {})
    reasoning["inventory_selection"] = (
        f"Forced to available inventory reactor: {_reactor_label(selected)}. "
        "Flow rates recalculated from selected volume and target residence time. "
        + (
            f"Liquid flow was clamped to the operating range of {selected_pump.name}."
            if pump_flow_clamped and selected_pump
            else "Selected pump flow range was satisfied."
        )
    )
    data["chemistry_notes"] = _append_note(
        data.get("chemistry_notes", ""),
        f"Inventory enforced: {_reactor_label(selected)}."
    )

    new = {
        "reactor_volume_mL": data.get("reactor_volume_mL"),
        "tubing_ID_mm": data.get("tubing_ID_mm"),
        "tubing_material": data.get("tubing_material"),
        "flow_rate_mL_min": data.get("flow_rate_mL_min"),
        "BPR_bar": data.get("BPR_bar"),
        "temperature_C": data.get("temperature_C"),
        "concentration_M": data.get("concentration_M"),
        "wavelength_nm": data.get("wavelength_nm"),
    }

    return FlowProposal(**data), {
        "applied": True,
        "old": old,
        "new": new,
        "selected_reactor": selected_payload,
    }


def select_reactor_for_proposal(
    proposal: FlowProposal,
    inventory: LabInventory,
) -> ReactorSpec | None:
    """Choose the closest inventory reactor for a proposal."""

    available_reactors = [
        reactor for reactor in inventory.reactors
        if str(reactor.service_status).lower() in {
            "available", "ready", "in_service", "in service"
        }
    ]
    if not available_reactors:
        return None
    desired_volume = _first_positive(
        proposal.reactor_volume_mL,
        (proposal.residence_time_min or 0.0) * (proposal.flow_rate_mL_min or 0.0),
        available_reactors[0].volume_mL,
    )
    desired_id = _first_positive(proposal.tubing_ID_mm, available_reactors[0].ID_mm)
    desired_temp = _num(proposal.temperature_C, 25.0)
    desired_conc = _num(proposal.concentration_M, 0.1)
    desired_pressure = _num(proposal.BPR_bar, 0.0)
    desired_wavelength = _num(proposal.wavelength_nm, 0.0)
    desired_tau = _first_positive(
        proposal.residence_time_min,
        proposal.residence_time_inlet_min,
        proposal.residence_time_in_channel_min,
    )
    basis = normalize_residence_time_basis(proposal.residence_time_basis)
    proposal_data = proposal.model_dump()
    target_gas_equiv = _target_gas_equiv_inlet(proposal_data)
    desired_system = _desired_reactor_system(proposal, inventory)

    def score(r: ReactorSpec) -> tuple[float, float]:
        s = abs(float(r.volume_mL) - desired_volume) / max(desired_volume, 1.0)
        s += 0.1 * abs(float(r.ID_mm) - desired_id)
        s += _range_penalty(desired_temp, r.min_temperature_C, r.max_temperature_C, r.allowed_temperatures_C)
        s += _range_penalty(desired_conc, r.min_concentration_M, r.max_concentration_M, [])
        s += _range_penalty(desired_pressure, r.min_pressure_bar, r.max_pressure_bar, [])
        if desired_wavelength > 0 and r.wavelength_nm:
            s += abs(float(r.wavelength_nm) - desired_wavelength) / 1000.0
        if desired_system and r.system and r.system != desired_system:
            s += 2.0

        pump = _pump_for_reactor(r, inventory)
        required_liquid_q = _required_liquid_flow_for_reactor(
            r,
            tau_min=desired_tau,
            residence_time_basis=basis,
            concentration_M=desired_conc,
            temperature_C=desired_temp,
            pressure_bar=desired_pressure,
            target_gas_equiv=target_gas_equiv,
            proposal_data=proposal_data,
        )
        if pump is not None and required_liquid_q > 0:
            if required_liquid_q < pump.min_flow_rate_mL_min - 1e-12:
                s += 100.0 + (
                    pump.min_flow_rate_mL_min - required_liquid_q
                ) / max(pump.min_flow_rate_mL_min, 1e-9)
            elif required_liquid_q > pump.max_flow_rate_mL_min + 1e-12:
                s += 100.0 + (
                    required_liquid_q - pump.max_flow_rate_mL_min
                ) / max(pump.max_flow_rate_mL_min, 1e-9)
        # Tie-break toward higher irradiance for photochemical reactors.
        intensity = float(r.intensity_mW_cm2 or 0.0)
        return (s, -intensity)

    return min(available_reactors, key=score)


def _desired_reactor_system(
    proposal: FlowProposal,
    inventory: LabInventory,
) -> str:
    for payload in (
        proposal.inventory_selection or {},
        (proposal.inventory_constraints or {}).get("selected_reactor") or {},
    ):
        system = str(payload.get("system") or "").strip()
        if system:
            return system

    calibration = proposal.evidence_calibration or {}
    anchor = calibration.get("anchor_conditions") or {}
    anchor_volume = _num(anchor.get("reactor_volume_mL"), 0.0)
    if anchor_volume > 0:
        matching = [
            reactor
            for reactor in inventory.reactors
            if abs(float(reactor.volume_mL) - anchor_volume) < 1e-6
        ]
        systems = {reactor.system for reactor in matching if reactor.system}
        if len(systems) == 1:
            return next(iter(systems))
    return ""


def _required_liquid_flow_for_reactor(
    reactor: ReactorSpec,
    *,
    tau_min: float,
    residence_time_basis: str,
    concentration_M: float,
    temperature_C: float,
    pressure_bar: float,
    target_gas_equiv: float,
    proposal_data: dict[str, Any],
) -> float:
    if tau_min <= 0:
        return 0.0

    if (
        residence_time_basis == INLET_STP_BASIS
        and target_gas_equiv > 0
        and _proposal_has_gas(proposal_data)
    ):
        _, reagent_fraction = _gas_species_and_fraction(proposal_data)
        gas_stp_ratio = stp_gas_flow_for_equiv(
            1.0,
            max(concentration_M, 1e-9),
            target_gas_equiv,
            reagent_fraction,
        )
        return float(reactor.volume_mL) / tau_min / (1.0 + gas_stp_ratio)

    liquid_q, gas_actual, _ = _current_flow_basis(proposal_data)
    gas_actual_ratio = gas_actual / liquid_q if liquid_q > 0 and gas_actual > 0 else 0.0
    if (
        gas_actual_ratio <= 0
        and target_gas_equiv > 0
        and _proposal_has_gas(proposal_data)
    ):
        _, reagent_fraction = _gas_species_and_fraction(proposal_data)
        gas_stp_ratio = stp_gas_flow_for_equiv(
            1.0,
            max(concentration_M, 1e-9),
            target_gas_equiv,
            reagent_fraction,
        )
        gas_actual_ratio = actual_gas_flow_from_stp(
            gas_stp_ratio,
            temperature_C,
            pressure_bar,
        )
    return float(reactor.volume_mL) / tau_min / (1.0 + gas_actual_ratio)


def reactor_is_compatible(
    reactor: ReactorSpec,
    *,
    temperature_C: float,
    concentration_M: float,
    pressure_bar: float,
) -> bool:
    """Check hard reactor operating ranges where provided."""

    if reactor.allowed_temperatures_C:
        if not any(abs(float(t) - temperature_C) < 1e-6 for t in reactor.allowed_temperatures_C):
            return False
    if reactor.min_temperature_C is not None and temperature_C < reactor.min_temperature_C:
        return False
    if reactor.max_temperature_C is not None and temperature_C > reactor.max_temperature_C:
        return False
    if reactor.min_concentration_M is not None and concentration_M < reactor.min_concentration_M:
        return False
    if reactor.max_concentration_M is not None and concentration_M > reactor.max_concentration_M:
        return False
    if reactor.min_pressure_bar is not None and pressure_bar < reactor.min_pressure_bar:
        return False
    if reactor.max_pressure_bar is not None and pressure_bar > reactor.max_pressure_bar:
        return False
    return True


def _target_gas_equiv_inlet(data: dict[str, Any]) -> float:
    calibration = data.get("evidence_calibration") or {}
    for source in (
        calibration.get("recommended_conditions") if isinstance(calibration, dict) else None,
        calibration.get("anchor_conditions") if isinstance(calibration, dict) else None,
        data.get("multiphase_metrics") or {},
    ):
        if not isinstance(source, dict):
            continue
        for key in ("gas_equiv_inlet", "target_gas_equiv_inlet", "o2_target_equiv"):
            value = _num(source.get(key), 0.0)
            if value > 0:
                return value
    for stream in data.get("streams") or []:
        if not _is_gas_stream(stream):
            continue
        value = _num(stream.get("molar_equiv"), 0.0)
        # StreamAssignment defaults to 1.0, so only treat a non-default stream
        # value as explicit. Calculator-produced targets are carried above in
        # multiphase_metrics and may legitimately equal 1.0.
        if value > 0 and abs(value - 1.0) > 1e-9:
            return value
    return 0.0


def _gas_species_and_fraction(data: dict[str, Any]) -> tuple[str, float]:
    metrics = data.get("multiphase_metrics") or {}
    species = str(metrics.get("gas_species") or "").strip()
    fraction = _num(metrics.get("gas_reagent_fraction"), 0.0)
    if species and fraction > 0:
        return species, fraction

    text_parts = [species]
    for stream in data.get("streams") or []:
        if not _is_gas_stream(stream):
            continue
        explicit_fraction = _num(stream.get("gas_reagent_mole_fraction"), 0.0)
        if explicit_fraction > 0:
            fraction = explicit_fraction
        text_parts.append(str(stream.get("pump_role") or ""))
        text_parts.extend(str(c) for c in (stream.get("contents") or []))
    text = " ".join(text_parts).lower()
    if re.search(r"\bair\b", text):
        return "air", fraction or 0.21
    species_patterns = (
        ("O2", r"\b(o2|oxygen)\b|o₂"),
        ("H2", r"\b(h2|hydrogen)\b|h₂"),
        ("CO2", r"\b(co2|carbon dioxide)\b|co₂"),
        ("CO", r"\bcarbon monoxide\b"),
        ("O3", r"\b(o3|ozone)\b|o₃"),
        ("Cl2", r"\b(cl2|chlorine)\b|cl₂"),
        ("NH3", r"\b(nh3|ammonia)\b|nh₃"),
        ("HCl", r"\b(hcl gas|hydrogen chloride)\b"),
        ("SO2", r"\b(so2|sulfur dioxide)\b|so₂"),
    )
    for label, pattern in species_patterns:
        if re.search(pattern, text):
            return label, fraction or 1.0
    if "syngas" in text:
        return "syngas", 0.5
    return species or "gas", fraction or 1.0


def _current_flow_basis(data: dict[str, Any]) -> tuple[float, float, float]:
    calibration = data.get("evidence_calibration") or {}
    recommended = calibration.get("recommended_conditions") or {}
    if recommended:
        liquid_q = _num(recommended.get("substrate_flow_mL_min"), 0.0)
        gas_actual = _num(recommended.get("gas_flow_in_channel_mL_min"), 0.0)
        gas_sccm = _num(recommended.get("gas_flow_stp_mL_min"), 0.0)
        if liquid_q > 0 and (gas_actual > 0 or gas_sccm > 0):
            return liquid_q, gas_actual, gas_sccm

    liquid_q = _num(data.get("flow_rate_mL_min"), 0.0)
    gas_actual = 0.0
    gas_sccm = 0.0
    for stream in data.get("streams") or []:
        if not _is_gas_stream(stream):
            continue
        gas_actual = _num(stream.get("gas_flow_actual_mL_min"), stream.get("flow_rate_mL_min"), gas_actual)
        gas_sccm = _num(stream.get("gas_flow_sccm"), gas_sccm)
        break
    mp = data.get("multiphase_metrics") or {}
    gas_actual = _num(gas_actual, mp.get("gas_flow_actual_mL_min"), 0.0)
    gas_sccm = _num(gas_sccm, mp.get("gas_flow_sccm"), 0.0)
    return liquid_q, gas_actual, gas_sccm


def _scale_liquid_streams(data: dict[str, Any], liquid_q: float, concentration: float) -> None:
    streams = data.get("streams") or []
    liquid_streams = [
        s for s in streams
        if not _is_gas_stream(s) and "quench" not in str(s.get("pump_role", "")).lower()
    ]
    if not liquid_streams:
        return
    current_sum = sum(_num(s.get("flow_rate_mL_min"), 0.0) for s in liquid_streams)
    for stream in liquid_streams:
        if current_sum > 0:
            frac = _num(stream.get("flow_rate_mL_min"), 0.0) / current_sum
            stream["flow_rate_mL_min"] = round(liquid_q * frac, 5)
        elif len(liquid_streams) == 1:
            stream["flow_rate_mL_min"] = round(liquid_q, 5)
        else:
            stream["flow_rate_mL_min"] = round(liquid_q / len(liquid_streams), 5)
        if stream.get("concentration_M") is not None:
            stream["concentration_M"] = round(concentration, 4)


def _sync_gas_streams(
    data: dict[str, Any],
    gas_actual: float,
    gas_sccm: float,
    *,
    species: str = "gas",
) -> None:
    streams = data.setdefault("streams", [])
    for stream in streams:
        if _is_gas_stream(stream):
            stream["phase"] = "gas"
            stream["gas_flow_actual_mL_min"] = round(gas_actual, 5)
            stream["gas_flow_sccm"] = round(gas_sccm, 5)
            stream["flow_rate_mL_min"] = round(gas_actual, 5)
            return
    streams.append({
        "stream_label": "G",
        "pump_role": f"{species} gas feed",
        "contents": [species],
        "phase": "gas",
        "gas_flow_actual_mL_min": round(gas_actual, 5),
        "gas_flow_sccm": round(gas_sccm, 5),
        "flow_rate_mL_min": round(gas_actual, 5),
        "reasoning": "Inventory enforcement added gas MFC stream scaled to the selected reactor volume.",
    })


def _proposal_has_gas(data: dict[str, Any]) -> bool:
    if (data.get("multiphase_metrics") or {}).get("gas_flow_actual_mL_min"):
        return True
    return any(_is_gas_stream(s) for s in data.get("streams") or [])


def _is_gas_stream(stream: dict[str, Any]) -> bool:
    phase = str(stream.get("phase") or "").lower()
    if phase in {"liquid", "solution"}:
        return False
    if phase == "gas":
        return True
    if stream.get("gas_flow_sccm") is not None or stream.get("gas_flow_actual_mL_min") is not None:
        return True
    text = " ".join([
        str(stream.get("pump_role") or ""),
        " ".join(str(c) for c in (stream.get("contents") or [])),
    ]).lower()
    if any(w in text for w in ("degassed", "deoxygenated", "solution")):
        return False
    return bool(
        re.search(r"\b(o2|oxygen|air|h2|hydrogen|co2|cl2|mfc)\b", text)
        or re.search(r"\bgas[\s-]+(feed|injection|inlet)\b", text)
    )


def _snap_temperature(value: float, reactor: ReactorSpec) -> float:
    if reactor.allowed_temperatures_C:
        return min((float(t) for t in reactor.allowed_temperatures_C), key=lambda t: abs(t - value))
    return _clamp(value, reactor.min_temperature_C, reactor.max_temperature_C)


def _snap_pressure(
    value: float,
    reactor: ReactorSpec,
    inventory: LabInventory,
    *,
    is_gas_liquid: bool,
) -> float:
    min_pressure = reactor.min_pressure_bar if reactor.min_pressure_bar is not None else 0.0
    if is_gas_liquid:
        positive_bprs = [p for p in available_pressure_settings(inventory) if p > 0]
        if positive_bprs:
            min_pressure = max(min_pressure, min(positive_bprs))
    max_pressure = reactor.max_pressure_bar
    value = _clamp(max(value, min_pressure), min_pressure, max_pressure)
    choices = [
        p for p in available_pressure_settings(inventory)
        if float(p) >= min_pressure and (max_pressure is None or float(p) <= max_pressure)
    ]
    if choices:
        value = min(choices, key=lambda p: abs(p - value))
    return value


def _pump_for_reactor(reactor: ReactorSpec, inventory: LabInventory) -> PumpSpec | None:
    if not inventory.pumps:
        return None

    available_pumps = [
        pump for pump in inventory.pumps
        if str(pump.service_status).lower() in {
            "available", "ready", "in_service", "in service"
        }
    ]
    if not available_pumps:
        return None

    compatible_systems = {
        value.strip().lower()
        for value in reactor.compatible_systems
        if value.strip()
    }
    if compatible_systems:
        for pump in available_pumps:
            pump_systems = {
                value.strip().lower()
                for value in pump.compatible_systems
                if value.strip()
            }
            if not pump_systems or compatible_systems.intersection(pump_systems):
                return pump

    system = (reactor.system or "").strip().lower()
    if system:
        for pump in available_pumps:
            if any(system == candidate.strip().lower() for candidate in pump.compatible_systems):
                return pump

        system_token = system.split()[0]
        for pump in available_pumps:
            if system_token and system_token in pump.name.lower():
                return pump

    return available_pumps[0]


def _pump_payload(pump: PumpSpec | None) -> dict[str, Any] | None:
    return pump.model_dump() if pump is not None else None


def _actual_to_sccm(gas_actual_mL_min: float, temperature_C: float, pressure_bar: float) -> float:
    if gas_actual_mL_min <= 0:
        return 0.0
    t_k = temperature_C + 273.15
    p_abs = max(float(pressure_bar or 0.0) + P_STP_BAR, 6.0)
    return gas_actual_mL_min * (T_STP_K / t_k) * (p_abs / P_STP_BAR)


def _reactor_label(r: ReactorSpec) -> str:
    prefix = f"{r.system} " if r.system else ""
    name = r.name or f"{r.type} reactor"
    configuration = f", {r.configuration}" if r.configuration else ""
    return (
        f"{prefix}{name}: {r.volume_mL:g} mL, {r.ID_mm:g} mm ID, "
        f"{r.material}{configuration}"
    )


def _reactor_payload(r: ReactorSpec) -> dict[str, Any]:
    return {
        "equipment_id": r.equipment_id,
        "name": r.name,
        "quantity": r.quantity,
        "service_status": r.service_status,
        "compatible_systems": r.compatible_systems,
        "system": r.system,
        "type": r.type,
        "material": r.material,
        "volume_mL": r.volume_mL,
        "ID_mm": r.ID_mm,
        "light_source": r.light_source,
        "wavelength_nm": r.wavelength_nm,
        "intensity_mW_cm2": r.intensity_mW_cm2,
        "irradiation": r.irradiation,
        "temperature_range_C": [r.min_temperature_C, r.max_temperature_C],
        "allowed_temperatures_C": r.allowed_temperatures_C,
        "concentration_range_M": [r.min_concentration_M, r.max_concentration_M],
        "pressure_range_bar": [r.min_pressure_bar, r.max_pressure_bar],
        "configuration": r.configuration,
        "component_volumes_mL": r.component_volumes_mL,
        "notes": r.notes,
    }


def selected_process_systems(
    proposal: FlowProposal,
    inventory: LabInventory,
) -> set[str]:
    """Return process-system choices supported by the selected reactor.

    ``ReactorSpec.system`` is often a component family (for example, "KHU
    tubing reactor"), while ``compatible_systems`` names the installations in
    which that component can run. Equipment matching must use the latter when
    it is available.
    """

    selection = proposal.inventory_selection or {}
    explicit = selection.get("process_system") or selection.get("selected_system")
    if explicit:
        values = explicit if isinstance(explicit, list) else [explicit]
        return {_normalize_system_name(value) for value in values if str(value).strip()}

    selected_id = str(selection.get("equipment_id") or "")
    selected = next(
        (item for item in inventory.reactors if item.equipment_id == selected_id),
        None,
    )
    if selected is not None and selected.compatible_systems:
        return {
            _normalize_system_name(value)
            for value in selected.compatible_systems
            if value.strip()
        }

    fallback = str(selection.get("system") or "").strip()
    return {_normalize_system_name(fallback)} if fallback else set()


def equipment_system_compatible(item: Any, selected_systems: set[str]) -> bool:
    """Check whether an inventory item can run in any selected installation."""

    item_systems = {
        _normalize_system_name(value)
        for value in getattr(item, "compatible_systems", [])
        if str(value).strip()
    }
    return not item_systems or not selected_systems or bool(item_systems & selected_systems)


def _normalize_system_name(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _range_penalty(
    value: float,
    lower: float | None,
    upper: float | None,
    allowed: list[float],
) -> float:
    if allowed:
        return min(abs(float(v) - value) for v in allowed) / 100.0
    penalty = 0.0
    if lower is not None and value < lower:
        penalty += abs(lower - value)
    if upper is not None and value > upper:
        penalty += abs(value - upper)
    return penalty


def _clamp(value: float, lower: float | None, upper: float | None) -> float:
    if lower is not None:
        value = max(value, float(lower))
    if upper is not None:
        value = min(value, float(upper))
    return value


def _first_positive(*values: Any) -> float:
    for value in values:
        try:
            f = float(value)
            if f > 0:
                return f
        except (TypeError, ValueError):
            continue
    return 0.0


def _num(*values: Any) -> float:
    for value in values:
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _append_note(existing: str, note: str) -> str:
    existing = str(existing or "").strip()
    if not existing:
        return note
    if note in existing:
        return existing
    return existing + "\n" + note
