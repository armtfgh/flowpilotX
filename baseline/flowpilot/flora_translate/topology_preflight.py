"""Inventory capability preflight for chemistry-derived process topology."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Callable

from flora_translate.schemas import (
    ChemistryPlan,
    LabInventory,
    ProcessStage,
    ProcessTopology,
    StreamConnection,
    UnitOperation,
    normalized_stream_phase,
)


AVAILABLE_STATUSES = {"available", "ready", "in_service", "in service"}


def analyze_topology_requirements(
    chemistry_plan: ChemistryPlan,
    inventory: LabInventory,
) -> tuple[ProcessTopology, dict[str, Any]]:
    """Build the chemistry-required graph and check inventory categories.

    This pass deliberately has no flow rate, volume, pressure, or residence-time
    setpoints. It answers the earlier question: can the required process graph
    be built from explicitly declared capability categories?
    """

    topology = build_requirements_topology(chemistry_plan)
    required = _required_counts(topology)
    requirements: list[dict[str, Any]] = []

    _check_category(
        requirements,
        requirement_id="REQ-LIQUID-PUMPS",
        operation_id="liquid_feeds",
        category="pumps",
        required_count=required["pumps"],
        items=inventory.pumps,
    )
    _check_category(
        requirements,
        requirement_id="REQ-GAS-MFC",
        operation_id="gas_feeds",
        category="gas_hardware",
        required_count=required["gas_hardware"],
        items=inventory.gas_hardware,
        predicate=lambda item: "mfc" in str(item.type or "").lower(),
    )
    _check_category(
        requirements,
        requirement_id="REQ-MIXERS",
        operation_id="interstage_mixing",
        category="mixers",
        required_count=required["mixers"],
        items=inventory.mixers,
        allow_undeclared_standard_accessory=True,
    )
    _check_category(
        requirements,
        requirement_id="REQ-REACTORS",
        operation_id="reaction_stages",
        category="reactors",
        required_count=required["reactors"],
        items=inventory.reactors,
    )
    _check_category(
        requirements,
        requirement_id="REQ-LIGHT-SOURCES",
        operation_id="photochemical_stages",
        category="light_sources",
        required_count=required["light_sources"],
        items=inventory.light_sources,
    )
    _check_category(
        requirements,
        requirement_id="REQ-PRESSURE-CONTROL",
        operation_id="outlet_pressure_control",
        category="pressure_controllers",
        required_count=required["pressure_controllers"],
        items=inventory.pressure_controllers,
        legacy_available_count=len(inventory.BPR_available),
    )
    if required["reactor_connections"]:
        combined = [*inventory.connectors, *inventory.reactor_trains]
        _check_category(
            requirements,
            requirement_id="REQ-DIRECT-REACTOR-CONNECTIONS",
            operation_id="direct_reactor_connections",
            category="connectors_or_reactor_trains",
            required_count=required["reactor_connections"],
            items=combined,
            allow_undeclared_standard_accessory=inventory.allows_standard_reactor_connections,
        )
        if inventory.allows_standard_reactor_connections and requirements[-1]["status"] != "available":
            requirements[-1].update(
                status="assumed_standard_accessory",
                reason="Standard interstage connectors are available under the inventory policy; verify fitting and operating ratings before execution.",
            )
        elif inventory.capability_status.get("connectors") == "unavailable":
            requirements[-1].update(status="unavailable", reason="Connectors are explicitly unavailable.")

    for item in requirements:
        if inventory.capability_status.get(item["category"]) == "unavailable":
            item.update(
                status="unavailable",
                available_count=0,
                reason=f"The chemist explicitly marked {item['category'].replace('_', ' ')} unavailable. This process still requires that capability.",
            )

    unresolved = [
        item
        for item in requirements
        if item["status"] in {"confirmation_required", "unavailable"}
    ]
    assumptions = [
        item
        for item in requirements
        if item["status"] == "assumed_standard_accessory"
    ]
    unavailable = [item for item in unresolved if item["status"] == "unavailable"]
    status = (
        "infeasible"
        if unavailable
        else "needs_confirmation"
        if unresolved
        else "ready_with_assumptions"
        if assumptions
        else "ready"
    )
    for operation in topology.unit_operations:
        category = operation.inventory_category
        matching = next(
            (
                item
                for item in [*unresolved, *assumptions]
                if item["category"] == category
                or (
                    category == "connectors"
                    and item["category"] == "connectors_or_reactor_trains"
                )
            ),
            None,
        )
        if matching:
            operation.assignment_status = matching["status"]
            operation.parameters["inventory_status"] = matching["status"]
            operation.parameters["requirement_id"] = matching["requirement_id"]
            operation.label = f"{operation.label} [VERIFY]"
        elif category in required:
            operation.assignment_status = "capability_available"

    topology.compilation_status = f"preflight_{status}"
    topology.pid_description = " -> ".join(
        operation.label
        for operation in topology.unit_operations
        if operation.op_type != "led_module"
    )
    report = {
        "schema_version": "flowpilot_topology_preflight_v1.0",
        "status": status,
        "strict_assignment": bool(inventory.strict_assignment),
        "required_counts": dict(required),
        "requirements": requirements,
        "unresolved_requirements": unresolved,
        "assumed_standard_accessories": assumptions,
        "checks": {
            "all_blocking_capability_categories_resolved": not unresolved,
            "all_required_capability_categories_declared": not unresolved and not assumptions,
            "no_explicitly_unavailable_required_capability": not unavailable,
        },
        "confirmation_template": {
            "schema_version": "flowpilot_inventory_confirmation_v1.0",
            "items": [
                {
                    "requirement_id": item["requirement_id"],
                    "category": item["category"],
                    "status": "available | unavailable",
                    "equipment_id": "",
                    "name": "",
                    "quantity": item["required_count"],
                    "capability_details": {},
                }
                for item in [*unresolved, *assumptions]
            ],
        },
    }
    return topology, report


def build_requirements_topology(chemistry_plan: ChemistryPlan) -> ProcessTopology:
    stages = list(chemistry_plan.stages or [])
    if not stages:
        stages = [
            ProcessStage(
                stage_number=1,
                stage_name=chemistry_plan.reaction_name or "Reaction stage",
                requires_light=bool(chemistry_plan.light_sensitive_reagents),
                feed_streams=list(chemistry_plan.stream_logic or []),
                oxygen_sensitive=chemistry_plan.oxygen_sensitive,
                deoxygenation_required=chemistry_plan.deoxygenation_required,
            )
        ]

    operations: list[UnitOperation] = []
    streams: list[StreamConnection] = []
    previous_reactor = ""
    stream_index = 0
    has_gas = False
    label_introduction_stage: dict[str, int] = {}
    for stage in stages:
        for feed in stage.feed_streams or []:
            label = str(feed.stream_label or "").upper()
            if label and label not in label_introduction_stage:
                label_introduction_stage[label] = int(stage.stage_number)

    for index, stage in enumerate(stages, 1):
        stage_number = int(stage.stage_number or index)
        prefix = f"st{stage_number}"
        feed_ids: list[str] = []
        for feed_index, feed in enumerate(stage.feed_streams or [], 1):
            if not feed.accepted_requirement:
                continue
            if _is_auxiliary_purge_feed(feed):
                continue
            label = str(feed.stream_label or "").upper()
            introduction_stage = int(
                feed.introduction_stage
                or label_introduction_stage.get(label, stage_number)
            )
            if (
                feed.delivery_mode == "carried_from_previous"
                or introduction_stage != stage_number
            ):
                continue
            phase = normalized_stream_phase(
                feed.phase,
                feed.reagents,
                gas_flow_sccm=feed.gas_flow_sccm,
                gas_flow_actual_mL_min=feed.gas_flow_actual_mL_min,
            )
            is_gas = phase == "gas"
            has_gas = has_gas or is_gas
            op_id = f"{prefix}_{'gas' if is_gas else 'feed'}_{feed_index}"
            operations.append(
                UnitOperation(
                    op_id=op_id,
                    op_type="mfc" if is_gas else "pump",
                    label=(
                        f"Stage {stage_number} gas feed"
                        if is_gas
                        else f"Stage {stage_number} liquid feed {feed_index}"
                    ),
                    parameters={
                        "phase": phase,
                        "contents": list(feed.reagents or []),
                        "requirement_only": True,
                    },
                    inventory_category="gas_hardware" if is_gas else "pumps",
                    requirement_authority=feed.requirement_authority,
                    source_evidence=list(feed.source_evidence),
                    accepted_requirement=feed.accepted_requirement,
                )
            )
            feed_ids.append(op_id)

        inlets = ([previous_reactor] if previous_reactor else []) + feed_ids
        reactor_id = f"{prefix}_reactor"
        if len(inlets) > 1:
            mixer_id = f"{prefix}_mixer"
            operations.append(
                UnitOperation(
                    op_id=mixer_id,
                    op_type="mixer",
                    label=f"Stage {stage_number} mixer",
                    parameters={"required_inputs": len(inlets), "requirement_only": True},
                    inventory_category="mixers",
                    requirement_authority="deterministic_derivation",
                    source_evidence=[f"Stage {stage_number} has {len(inlets)} accepted physical inlets."],
                )
            )
            for inlet in inlets:
                stream_index += 1
                streams.append(
                    StreamConnection(
                        stream_id=f"req_s{stream_index}",
                        from_op=inlet,
                        to_op=mixer_id,
                        stream_type="gas" if "_gas_" in inlet else "process",
                    )
                )
            reactor_inlet = mixer_id
        elif inlets:
            reactor_inlet = inlets[0]
        else:
            reactor_inlet = previous_reactor

        operations.append(
            UnitOperation(
                op_id=reactor_id,
                op_type="photoreactor" if stage.requires_light else "coil_reactor",
                label=f"Stage {stage_number}: {stage.stage_name or 'reaction'}",
                parameters={
                    "stage_number": stage_number,
                    "requires_light": stage.requires_light,
                    "target_wavelength_nm": stage.wavelength_nm,
                    "requirement_only": True,
                },
                inventory_category="reactors",
                requirement_authority="protocol_fact",
                source_evidence=[f"Accepted reaction stage {stage_number}."],
            )
        )
        if reactor_inlet:
            stream_index += 1
            streams.append(
                StreamConnection(
                    stream_id=f"req_s{stream_index}",
                    from_op=reactor_inlet,
                    to_op=reactor_id,
                )
            )
        if stage.requires_light:
            operations.append(
                UnitOperation(
                    op_id=f"{prefix}_light",
                    op_type="led_module",
                    label=f"Stage {stage_number} light source",
                    parameters={
                        "target_wavelength_nm": stage.wavelength_nm,
                        "requirement_only": True,
                    },
                    inventory_category="light_sources",
                    requirement_authority="protocol_fact",
                    source_evidence=["The reconciled protocol contract requires irradiation."],
                )
            )
        previous_reactor = reactor_id

    outlet_source = previous_reactor
    if has_gas:
        operations.append(
            UnitOperation(
                op_id="outlet_bpr",
                op_type="bpr",
                label="Pressure control",
                parameters={"requirement_only": True},
                inventory_category="pressure_controllers",
                requirement_authority="deterministic_derivation",
                source_evidence=["An accepted continuously metered reagent-gas feed is present."],
            )
        )
        stream_index += 1
        streams.append(
            StreamConnection(
                stream_id=f"req_s{stream_index}",
                from_op=outlet_source,
                to_op="outlet_bpr",
            )
        )
        outlet_source = "outlet_bpr"
    operations.append(
        UnitOperation(
            op_id="collector",
            op_type="collector",
            label="Collection",
            parameters={"requirement_only": True},
            inventory_category="process_endpoint",
            assignment_status="process_endpoint",
        )
    )
    stream_index += 1
    streams.append(
        StreamConnection(
            stream_id=f"req_s{stream_index}",
            from_op=outlet_source,
            to_op="collector",
        )
    )
    return ProcessTopology(
        topology_id="chemistry_inventory_requirements",
        unit_operations=operations,
        streams=streams,
        reactor_volume_mL=0.0,
        topology_confidence="HIGH",
        compilation_status="preflight",
    )


def _is_auxiliary_purge_feed(feed: Any) -> bool:
    """Exclude purge-only gas supplies from the reactive process train.

    Nitrogen used for leak testing, startup purge, or shutdown displacement is
    safety support, not a continuously metered reaction feed. Its availability
    is checked later by the safety contract and must not consume an MFC or a
    mixer inlet during topology capability preflight.
    """

    reagents = " ".join(str(item) for item in (feed.reagents or [])).lower()
    reasoning = str(feed.reasoning or "").lower()
    is_inert = bool(
        re.search(r"(?:^|\W)(?:n2|nitrogen|argon|ar)(?:$|\W)", reagents)
    )
    is_purge = any(
        token in reasoning
        for token in ("purge", "leak check", "shutdown", "displacement", "inerting")
    )
    return is_inert and is_purge


def _required_counts(topology: ProcessTopology) -> Counter:
    mapping = {
        "pump": "pumps",
        "mfc": "gas_hardware",
        "mixer": "mixers",
        "t_mixer": "mixers",
        "coil_reactor": "reactors",
        "photoreactor": "reactors",
        "led_module": "light_sources",
        "bpr": "pressure_controllers",
    }
    counts = Counter(
        mapping[operation.op_type]
        for operation in topology.unit_operations
        if operation.op_type in mapping
        and operation.required
        and operation.accepted_requirement
    )
    reactor_ids = {
        operation.op_id
        for operation in topology.unit_operations
        if operation.op_type in {"coil_reactor", "photoreactor"}
        and operation.required
        and operation.accepted_requirement
    }
    counts["reactor_connections"] = sum(
        1
        for stream in topology.streams
        if stream.from_op in reactor_ids and stream.to_op in reactor_ids
    )
    return counts


def _check_category(
    output: list[dict[str, Any]],
    *,
    requirement_id: str,
    operation_id: str,
    category: str,
    required_count: int,
    items: list[Any],
    predicate: Callable[[Any], bool] | None = None,
    legacy_available_count: int = 0,
    allow_undeclared_standard_accessory: bool = False,
) -> None:
    if required_count <= 0:
        return
    predicate = predicate or (lambda item: True)
    matching = [item for item in items if predicate(item)]
    available_count = legacy_available_count + sum(
        int(item.quantity)
        for item in matching
        if str(item.service_status or "").strip().lower() in AVAILABLE_STATUSES
    )
    if available_count >= required_count:
        status = "available"
        reason = "Required capability category is explicitly represented."
    elif (
        not matching
        and not legacy_available_count
        and allow_undeclared_standard_accessory
    ):
        status = "assumed_standard_accessory"
        reason = (
            f"Inventory does not list {category.replace('_', ' ')}. FlowPilot "
            "will use a generic compatible passive fitting for design and mark "
            "it for pre-run verification."
        )
    elif not matching and not legacy_available_count:
        status = "confirmation_required"
        reason = (
            f"Inventory does not declare {category.replace('_', ' ')}. Confirm "
            "an available item with its capability limits or mark it unavailable."
        )
    else:
        status = "unavailable"
        reason = (
            f"Only {available_count} available {category.replace('_', ' ')} "
            f"item(s) are declared; {required_count} are required."
        )
    output.append(
        {
            "requirement_id": requirement_id,
            "operation_id": operation_id,
            "category": category,
            "required_count": required_count,
            "available_count": available_count,
            "status": status,
            "reason": reason,
            "criteria": {"minimum_quantity": required_count},
        }
    )
