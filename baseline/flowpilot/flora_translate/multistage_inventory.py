"""Deterministic closure of multistage proposals against physical inventory."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from flora_translate.schemas import (
    ChemistryPlan,
    FlowProposal,
    LabInventory,
    normalized_stream_phase,
)


def reconcile_multistage_inventory(
    proposal: FlowProposal,
    chemistry_plan: ChemistryPlan | None,
    inventory: LabInventory | None,
    operating_limits: dict[str, Any] | None = None,
) -> tuple[FlowProposal, dict[str, Any]]:
    """Attach exact reactor/light IDs and close each stage's residence time.

    LLM prose may name stage hardware without filling ``stage_parameters``.
    This pass resolves only exact IDs that exist in inventory, applies quantity
    limits, and derives stage times from those physical volumes and the final
    liquid/STP/in-channel flow rates. Unknown hardware remains unresolved.
    """

    if (
        chemistry_plan is None
        or len(chemistry_plan.stages or []) <= 1
        or inventory is None
    ):
        return proposal, {"applied": False, "reason": "single-stage or no inventory"}

    current = proposal.model_copy(deep=True)
    existing = {
        int(item.get("stage_number")): dict(item)
        for item in current.stage_parameters or []
        if isinstance(item, dict) and _positive_int(item.get("stage_number"))
    }
    reactor_by_id = {item.equipment_id: item for item in inventory.reactors}
    light_by_id = {item.equipment_id: item for item in inventory.light_sources}
    reactor_refs = _ordered_references(current, reactor_by_id)
    light_refs = _ordered_references(current, light_by_id)
    reactor_remaining = Counter(
        {item.equipment_id: item.quantity for item in inventory.reactors}
    )
    light_remaining = Counter(
        {item.equipment_id: item.quantity for item in inventory.light_sources}
    )
    proposal_gas = {
        str(stream.stream_label or "").upper(): stream
        for stream in current.streams or []
        if _proposal_stream_is_gas(stream)
    }
    proposal_gases = list(proposal_gas.values())
    proposal_streams = {
        str(stream.stream_label or "").upper(): stream
        for stream in current.streams or []
        if str(stream.stream_label or "").strip()
    }
    label_introduction_stage: dict[str, int] = {}
    for stage in chemistry_plan.stages:
        for feed in stage.feed_streams or []:
            label = str(feed.stream_label or "").upper()
            if label and label not in label_introduction_stage:
                label_introduction_stage[label] = int(stage.stage_number)

    reconciled: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    used_reactors: list[str] = []
    used_lights: list[str] = []
    total_volume = 0.0
    total_tau_liquid = 0.0
    total_tau_inlet = 0.0
    total_tau_channel = 0.0
    liquid_flow_from_previous = 0.0
    introduced_gases: dict[str, Any] = {}
    gas_introduction_temperatures: dict[str, float] = {}

    for index, stage in enumerate(chemistry_plan.stages):
        stage_number = int(stage.stage_number or index + 1)
        parameters = dict(existing.get(stage_number, {}))
        parameters["stage_number"] = stage_number
        # Model-authored stage dictionaries commonly contain duplicate flow
        # and hardware aliases plus prose calculated from an earlier proposal.
        # They are diagnostic inputs, not finalized run instructions.
        for key in list(parameters):
            normalized = key.lower()
            if normalized == "bpr_equipment_id" or (
                normalized.startswith("pump_")
                and normalized.endswith("_equipment_id")
            ):
                parameters.pop(key, None)

        requested_reactor_id = str(
            parameters.get("reactor_equipment_id")
            or parameters.get("equipment_id")
            or ""
        )
        reactor_id = _select_available_id(
            requested_reactor_id,
            reactor_refs,
            index,
            reactor_by_id,
            reactor_remaining,
        )
        reactor = reactor_by_id.get(reactor_id)
        if reactor is None:
            unresolved.append(
                {
                    "stage_number": stage_number,
                    "category": "reactors",
                    "reason": (
                        "No exact, quantity-available reactor equipment ID was "
                        "provided for this stage."
                    ),
                    "requested_equipment_id": requested_reactor_id,
                }
            )
            _clear_unresolved_stage(parameters, category="reactor")
            parameters["requested_reactor_equipment_id"] = requested_reactor_id or None
            reconciled.append(parameters)
            continue
        reactor_remaining[reactor_id] -= 1
        used_reactors.append(reactor_id)

        light = None
        if stage.requires_light:
            from flora_translate.equipment_resources import light_fits_stage, resources_fit
            stage_remaining = light_remaining.copy()
            selected_lights = [light_by_id[key] for key in used_lights]
            for item in inventory.light_sources:
                if not light_fits_stage(item, reactor, current, inventory) or not resources_fit(
                    selected_lights + [item], inventory.resource_capacities
                ):
                    stage_remaining[item.equipment_id] = 0
            requested_light_id = str(parameters.get("light_equipment_id") or "")
            light_id = _select_available_id(
                requested_light_id,
                light_refs,
                index,
                light_by_id,
                stage_remaining,
            )
            if not light_id:
                light_id = _select_compatible_light_id(
                    stage,
                    reactor,
                    current,
                    inventory,
                    stage_remaining,
                    operating_limits,
                )
            light = light_by_id.get(light_id)
            if light is None:
                unresolved.append(
                    {
                        "stage_number": stage_number,
                        "category": "light_sources",
                        "reason": (
                            "No exact, quantity-available light equipment ID was "
                            "provided for this photochemical stage."
                        ),
                        "requested_equipment_id": requested_light_id,
                    }
                )
                light = None
            else:
                light_remaining[light_id] -= 1
                used_lights.append(light_id)
                parameters["photoreactor_module_id"] = light.module_id
                parameters["photoreactor_module_name"] = light.module_name

        desired_temperature = _number(
            parameters.get("temperature_C"),
            _number(stage.temperature_C, current.temperature_C),
        )
        temperature = _temperature_for_light(
            desired_temperature,
            light,
            operating_limits,
        )
        temperature = _clamp(
            temperature,
            reactor.min_temperature_C,
            reactor.max_temperature_C,
        )
        temperature_valid = abs(_temperature_for_light(temperature, light, operating_limits) - temperature) < 1e-9
        if not temperature_valid:
            unresolved.append({"stage_number": stage.stage_number, "category": "temperature_controllers",
                               "reason": "Selected reactor and light-source temperature limits do not overlap."})

        active_feeds = [
            feed
            for feed in stage.feed_streams or []
            if feed.delivery_mode != "carried_from_previous"
            and int(
                feed.introduction_stage
                or label_introduction_stage.get(
                    str(feed.stream_label or "").upper(), stage_number
                )
            ) == stage_number
        ]
        new_liquid_flow = sum(
            max(
                float(
                    getattr(
                        proposal_streams.get(str(feed.stream_label or "").upper()),
                        "flow_rate_mL_min",
                        0.0,
                    )
                    or 0.0
                ),
                0.0,
            )
            for feed in active_feeds
            if not _stage_feed_is_gas(feed)
        )
        if stage_number == 1:
            liquid_flow = new_liquid_flow or max(
                float(current.flow_rate_mL_min or 0.0), 0.0
            )
        else:
            liquid_flow = liquid_flow_from_previous + new_liquid_flow
        liquid_flow_from_previous = liquid_flow
        gas_sccm = 0.0
        gas_actual = 0.0
        for feed in active_feeds:
            label = str(feed.stream_label or "").upper()
            gas_stream = proposal_gas.get(label)
            if (
                gas_stream is None
                and _stage_feed_is_gas(feed)
                and len(proposal_gases) == 1
            ):
                # Chemistry plans and deterministic engineering may use
                # different labels (for example B versus G) for the same sole
                # reagent-gas feed. Identity and phase are authoritative here.
                gas_stream = proposal_gases[0]
            if gas_stream is None and not _stage_feed_is_gas(feed):
                continue
            if gas_stream is not None:
                gas_label = str(gas_stream.stream_label).upper()
                introduced_gases[gas_label] = gas_stream
                gas_introduction_temperatures.setdefault(gas_label, temperature)

        # Cumulative inlet bookkeeping, not a prediction of residual gas after
        # reaction or dissolution. A direct connection does not erase a feed.
        gas_sccm = sum(float(stream.gas_flow_sccm or 0.0) for stream in introduced_gases.values())
        gas_actual = sum(
            float(stream.gas_flow_actual_mL_min or 0.0)
            * (temperature + 273.15) / (gas_introduction_temperatures[label] + 273.15)
            for label, stream in introduced_gases.items()
        )
        if gas_sccm:
            parameters["gas_accounting_basis"] = "cumulative inlet/STP feed; uptake and removal not modeled"

        volume = float(reactor.volume_mL)
        if liquid_flow <= 0:
            tau_liquid = 0.0
            tau_inlet = 0.0
            tau_channel = 0.0
        else:
            tau_liquid = volume / max(liquid_flow, 1e-12)
            tau_inlet = volume / max(liquid_flow + gas_sccm, 1e-12)
            tau_channel = volume / max(liquid_flow + gas_actual, 1e-12)
        if gas_sccm <= 0 and gas_actual <= 0:
            tau_liquid = volume / max(liquid_flow, 1e-12)
            tau_inlet = tau_liquid
            tau_channel = tau_inlet
            basis = "liquid-only stage residence time"
        else:
            basis = "inlet/STP apparent residence time"

        parameters.update(
            {
                "reactor_equipment_id": reactor.equipment_id,
                "reactor_name": reactor.name,
                "reactor_volume_mL": volume,
                "V_R_mL": volume,
                "d_mm": float(reactor.ID_mm),
                "material": reactor.material,
                "temperature_C": round(temperature, 3),
                "Q_liquid_mL_min": round(liquid_flow, 6),
                "flow_rate_mL_min": round(liquid_flow, 6),
                "Q_gas_sccm": round(gas_sccm, 6),
                "Q_gas_actual_mL_min": round(gas_actual, 6),
                "gas_flow_sccm": round(gas_sccm, 6),
                "gas_flow_actual_mL_min": round(gas_actual, 6),
                "residence_time_min": round(
                    tau_inlet if gas_sccm > 0 else tau_liquid, 4
                ),
                "residence_time_inlet_min": round(tau_inlet, 4),
                "residence_time_in_channel_min": round(tau_channel, 4),
                "residence_time_basis": basis,
                "inventory_resolved": temperature_valid and (not stage.requires_light or light is not None),
            }
        )
        parameters["notes"] = (
            "Deterministic stage closure: "
            f"V_R={volume:.4g} mL; Q_liquid={liquid_flow:.6g} mL/min; "
            f"tau_liquid={tau_liquid:.4g} min; "
            f"tau_inlet={tau_inlet:.4g} min; "
            f"tau_in_channel={tau_channel:.4g} min; basis={basis}."
        )
        if gas_sccm > 0 or gas_actual > 0:
            gas_names = [
                str(reagent)
                for feed in active_feeds
                if _stage_feed_is_gas(feed)
                for reagent in feed.reagents
            ]
            matched_stage_gases = list(introduced_gases.values())
            if matched_stage_gases:
                gas_names = [
                    str(content)
                    for gas_stream in matched_stage_gases
                    for content in gas_stream.contents
                ]
            gas_name = ", ".join(gas_names) or "reagent gas"
            parameters["atmosphere"] = (
                f"{gas_name}: {gas_sccm:.6g} mL/min at inlet/STP; "
                f"{gas_actual:.6g} mL/min pressure-corrected in channel"
            )
        if light is not None:
            parameters.update(
                {
                    "light_equipment_id": light.equipment_id,
                    "light_name": light.name,
                    "target_wavelength_nm": float(
                        stage.wavelength_nm or current.wavelength_nm or light.wavelength_nm
                    ),
                    "wavelength_nm": float(light.wavelength_nm),
                }
            )
        reconciled.append(parameters)
        total_volume += volume
        total_tau_liquid += tau_liquid
        total_tau_inlet += tau_inlet
        total_tau_channel += tau_channel

    complete = len(reconciled) == len(chemistry_plan.stages) and not unresolved
    current.stage_parameters = reconciled
    if used_reactors:
        component_volumes = [reactor_by_id[item_id].volume_mL for item_id in used_reactors]
        first = reactor_by_id[used_reactors[0]]
        current.inventory_selection = {
            **dict(current.inventory_selection or {}),
            "equipment_id": first.equipment_id,
            "name": first.name,
            "system": first.system,
            "compatible_systems": first.compatible_systems,
            "configuration": "multistage",
            "component_reactor_ids": used_reactors,
            "component_volumes_mL": component_volumes,
            "total_volume_mL": round(sum(component_volumes), 4),
        }
    if complete:
        current.reactor_volume_mL = round(total_volume, 4)
        current.residence_time_inlet_min = round(total_tau_inlet, 4)
        current.residence_time_in_channel_min = round(total_tau_channel, 4)
        has_reagent_gas = any(
            float(item.get("Q_gas_sccm") or 0.0) > 0 for item in reconciled
        )
        if has_reagent_gas:
            current.residence_time_min = round(total_tau_inlet, 4)
            current.residence_time_basis = (
                "sum of per-stage inlet/STP apparent residence times"
            )
        else:
            current.residence_time_min = round(total_tau_liquid, 4)
            current.residence_time_basis = (
                "sum of per-stage liquid-only reactor-volume times"
            )
        metrics = dict(current.multiphase_metrics or {})
        metrics["total_reactor_volume_mL"] = round(total_volume, 4)
        metrics["multistage_residence_time_liquid_min"] = round(total_tau_liquid, 4)
        metrics["multistage_residence_time_inlet_min"] = round(total_tau_inlet, 4)
        metrics["multistage_residence_time_in_channel_min"] = round(total_tau_channel, 4)
        current.multiphase_metrics = metrics

    report = {
        "schema_version": "flowpilot_multistage_inventory_v1.0",
        "applied": True,
        "status": "complete" if complete else "incomplete",
        "stage_count": len(chemistry_plan.stages),
        "used_reactor_ids": used_reactors,
        "used_light_ids": used_lights,
        "total_reactor_volume_mL": round(total_volume, 4),
        "total_residence_time_liquid_min": round(total_tau_liquid, 4),
        "total_residence_time_inlet_min": round(total_tau_inlet, 4),
        "total_residence_time_in_channel_min": round(total_tau_channel, 4),
        "stage_parameters": reconciled,
        "unresolved_requirements": unresolved,
        "checks": {
            "all_stage_reactors_resolved": not any(
                item["category"] == "reactors" for item in unresolved
            ) and len(reconciled) == len(chemistry_plan.stages),
            "all_required_lights_resolved": not any(
                item["category"] == "light_sources" for item in unresolved
            ),
            "equipment_quantities_respected": all(
                value >= 0 for value in [*reactor_remaining.values(), *light_remaining.values()]
            ),
            "stage_geometry_closed": complete and all(
                item.get("reactor_volume_mL", 0) > 0
                and item.get("residence_time_inlet_min", 0) > 0
                for item in reconciled
            ),
        },
    }
    constraints = dict(current.inventory_constraints or {})
    constraints["multistage_inventory_plan"] = report
    current.inventory_constraints = constraints
    return current, report


def _ordered_references(proposal: FlowProposal, equipment: dict[str, Any]) -> list[str]:
    fields = [
        proposal.light_setup,
        proposal.chemistry_notes,
        json.dumps(proposal.reasoning_per_field or {}, sort_keys=True, default=str),
        json.dumps(proposal.stage_parameters or [], sort_keys=True, default=str),
    ]
    text = "\n".join(str(value or "") for value in fields)
    positions = []
    for equipment_id in equipment:
        position = text.find(equipment_id)
        if position >= 0:
            positions.append((position, equipment_id))
    return [equipment_id for _, equipment_id in sorted(positions)]


def _reference_at(values: list[str], index: int) -> str:
    return values[index] if index < len(values) else ""


def _select_available_id(
    preferred_id: str,
    ordered_refs: list[str],
    stage_index: int,
    equipment: dict[str, Any],
    remaining: Counter,
) -> str:
    """Select an explicitly referenced item without reusing exhausted stock."""

    candidates = [preferred_id, _reference_at(ordered_refs, stage_index), *ordered_refs]
    seen = set()
    for equipment_id in candidates:
        equipment_id = str(equipment_id or "")
        if not equipment_id or equipment_id in seen:
            continue
        seen.add(equipment_id)
        if equipment_id in equipment and remaining[equipment_id] > 0:
            return equipment_id
    return ""


def _select_compatible_light_id(
    stage: Any,
    reactor: Any,
    proposal: FlowProposal,
    inventory: LabInventory,
    remaining: Counter,
    operating_limits: dict[str, Any] | None,
) -> str:
    """Select unused light hardware from capabilities when no ID was seeded.

    Model prose is not required to repeat inventory IDs. The deterministic
    allocator owns wavelength, temperature, reactor-type, compatible-system,
    and quantity matching.
    """

    target_wavelength = _number(
        getattr(stage, "wavelength_nm", None),
        proposal.wavelength_nm,
    )
    desired_temperature = _number(
        getattr(stage, "temperature_C", None),
        proposal.temperature_C,
    )
    reactor_type = str(getattr(reactor, "type", "coil") or "coil").lower()
    reactor_systems = {
        str(item).strip().lower()
        for item in getattr(reactor, "compatible_systems", []) or []
        if str(item).strip()
    }
    candidates: list[tuple[tuple[float, float, str], str]] = []
    for light in inventory.light_sources:
        if remaining[light.equipment_id] <= 0:
            continue
        if str(light.service_status).lower() not in {
            "available", "ready", "in_service", "in service"
        }:
            continue
        compatible_reactor = str(light.compatible_reactor or "").lower()
        if compatible_reactor and compatible_reactor not in {
            reactor_type,
            "flow",
            "flow reactor",
        }:
            continue
        light_systems = {
            str(item).strip().lower()
            for item in light.compatible_systems or []
            if str(item).strip()
        }
        if reactor_systems and light_systems and reactor_systems.isdisjoint(light_systems):
            continue
        wavelength_delta = abs(float(light.wavelength_nm) - target_wavelength)
        if target_wavelength > 0 and wavelength_delta > 25.0:
            continue
        supported_temperature = _temperature_for_light(
            desired_temperature,
            light,
            operating_limits,
        )
        temperature_delta = abs(supported_temperature - desired_temperature)
        score = (
            wavelength_delta + 0.2 * temperature_delta,
            temperature_delta,
            light.equipment_id,
        )
        candidates.append((score, light.equipment_id))
    return min(candidates)[1] if candidates else ""


def _clear_unresolved_stage(parameters: dict[str, Any], *, category: str) -> None:
    """Remove inherited run values when physical stage closure has failed."""

    stale_fields = {
        "reactor_name",
        "reactor_volume_mL",
        "V_R_mL",
        "d_mm",
        "material",
        "Q_liquid_mL_min",
        "Q_gas_sccm",
        "Q_gas_actual_mL_min",
        "residence_time_min",
        "residence_time_inlet_min",
        "residence_time_in_channel_min",
        "residence_time_basis",
    }
    if category == "reactor":
        stale_fields.update({"reactor_equipment_id", "equipment_id"})
    for field in stale_fields:
        parameters.pop(field, None)
    parameters["inventory_resolved"] = False


def _temperature_for_light(
    value: float,
    light: Any,
    operating_limits: dict[str, Any] | None = None,
) -> float:
    if light is None:
        return value
    matched_limits = _photoreactor_limits_for_light(light, operating_limits)
    standard = matched_limits.get("standard_temperature_C") or {}
    explicit = matched_limits.get("allowed_temperature_C") or {}
    additional_cooling = matched_limits.get("additional_cooling_temperature_C") or {}
    lower_values = [float(item) for item in (light.min_temperature_C, explicit.get("minimum", additional_cooling.get("minimum", standard.get("minimum")))) if item is not None]
    upper_values = [float(item) for item in (light.max_temperature_C, explicit.get("maximum", standard.get("maximum", additional_cooling.get("maximum")))) if item is not None]
    lower = max(lower_values) if lower_values else None
    upper = min(upper_values) if upper_values else None
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(f"Conflicting temperature ranges for light source {light.name}.")
    equipment_settings = set(light.allowed_temperatures_C or [])
    profile_settings = set(matched_limits.get("allowed_temperatures_C") or [])
    allowed = equipment_settings & profile_settings if equipment_settings and profile_settings else equipment_settings or profile_settings
    if equipment_settings or profile_settings:
        feasible = [float(item) for item in allowed if (lower is None or float(item) >= lower) and (upper is None or float(item) <= upper)]
        if not feasible:
            raise ValueError(f"No available temperature setting satisfies all limits for light source {light.name}.")
        return min(feasible, key=lambda item: (abs(item - value), item))
    return _clamp(value, lower, upper)


def _photoreactor_limits_for_light(
    light: Any,
    operating_limits: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(operating_limits, dict):
        return {}
    limits = operating_limits.get("photoreactor_limits") or {}
    if not isinstance(limits, dict):
        return {}

    identity = " ".join(
        [
            str(getattr(light, "equipment_id", "")),
            str(getattr(light, "name", "")),
        ]
    ).lower()
    for label, values in limits.items():
        normalized_label = str(label).lower().replace("-", " ")
        tokens = [token for token in normalized_label.split() if len(token) >= 3]
        same_family = (
            ("manual" in normalized_label and "manual" in identity)
            or (
                "uv 150" in normalized_label
                and ("uv 150" in identity or "uv150" in identity)
            )
        )
        if same_family or (tokens and all(token in identity for token in tokens)):
            return values if isinstance(values, dict) else {}
    return {}


def _clamp(value: float, lower: Any, upper: Any) -> float:
    if lower is not None:
        value = max(value, float(lower))
    if upper is not None:
        value = min(value, float(upper))
    return value


def _proposal_stream_is_gas(stream: Any) -> bool:
    return normalized_stream_phase(
        getattr(stream, "phase", ""),
        getattr(stream, "contents", []),
        gas_flow_sccm=getattr(stream, "gas_flow_sccm", None),
        gas_flow_actual_mL_min=getattr(stream, "gas_flow_actual_mL_min", None),
    ) == "gas"


def _stage_feed_is_gas(feed: Any) -> bool:
    return normalized_stream_phase(
        getattr(feed, "phase", ""),
        getattr(feed, "reagents", []),
        gas_flow_sccm=getattr(feed, "gas_flow_sccm", None),
        gas_flow_actual_mL_min=getattr(feed, "gas_flow_actual_mL_min", None),
    ) == "gas"


def _positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _number(value: Any, default: Any = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(default)
        except (TypeError, ValueError):
            return 0.0
