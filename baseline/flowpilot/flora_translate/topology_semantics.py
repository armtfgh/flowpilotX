"""Deterministic phase and edge semantics for executable process graphs."""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from typing import Iterable

from flora_translate.schemas import ProcessTopology


CONTROL_OPERATION_TYPES = {
    "heater",
    "temperature_controller",
    "led_module",
    "chiller",
    "sensor",
}
MIXER_OPERATION_TYPES = {"mixer", "t_mixer", "y_mixer", "quench_mixer"}
SEPARATOR_OPERATION_TYPES = {
    "phase_separator",
    "gas_liquid_separator",
    "liquid_liquid_separator",
}


def normalize_topology_semantics(topology: ProcessTopology) -> ProcessTopology:
    """Return a copy with phase-aware process edges and graph-derived PID text."""

    normalized = deepcopy(topology)
    operations = {item.op_id: item for item in normalized.unit_operations}
    incoming: dict[str, list] = defaultdict(list)
    outgoing: dict[str, list] = defaultdict(list)
    for edge in normalized.streams:
        incoming[edge.to_op].append(edge)
        outgoing[edge.from_op].append(edge)

    operation_phase: dict[str, str] = {}
    for operation in normalized.unit_operations:
        phase = _source_phase(operation.op_type, operation.parameters or {})
        if phase:
            operation_phase[operation.op_id] = phase

    indegree = {
        op_id: sum(
            1
            for edge in incoming.get(op_id, [])
            if edge.connection_type == "process"
        )
        for op_id in operations
    }
    queue = deque(op_id for op_id, degree in indegree.items() if degree == 0)
    visited: set[str] = set()

    while queue:
        op_id = queue.popleft()
        visited.add(op_id)
        operation = operations[op_id]
        input_phases = [
            _edge_phase(edge, operation_phase)
            for edge in incoming.get(op_id, [])
            if edge.connection_type == "process"
        ]
        input_phases = [phase for phase in input_phases if phase]
        if op_id not in operation_phase:
            operation_phase[op_id] = _operation_output_phase(
                operation.op_type,
                operation.parameters or {},
                input_phases,
            )
        for edge in outgoing.get(op_id, []):
            if edge.connection_type == "process":
                edge.stream_type = _separator_edge_phase(
                    operation.op_type,
                    edge.label,
                    operation_phase.get(op_id) or "liquid",
                )
            indegree[edge.to_op] = max(0, indegree.get(edge.to_op, 0) - 1)
            if indegree[edge.to_op] == 0 and edge.to_op not in visited:
                queue.append(edge.to_op)

    # Cycles are invalid for the executable graph, but preserving a deterministic
    # phase on their edges gives the validator an intelligible diagnostic.
    for edge in normalized.streams:
        source = operations.get(edge.from_op)
        target = operations.get(edge.to_op)
        if source and source.op_type in CONTROL_OPERATION_TYPES:
            edge.connection_type = "control"
            edge.stream_type = "signal"
        elif target and target.op_type in CONTROL_OPERATION_TYPES:
            edge.connection_type = "control"
            edge.stream_type = "signal"
        elif edge.connection_type == "process":
            edge.stream_type = _separator_edge_phase(
                source.op_type if source else "",
                edge.label,
                operation_phase.get(edge.from_op) or edge.stream_type or "liquid",
            )

    normalized.pid_description = _pid_from_process_graph(normalized)
    return normalized


def topology_semantic_issues(topology: ProcessTopology) -> list[dict[str, str]]:
    """Return deterministic phase/control defects in a normalized topology."""

    expected = normalize_topology_semantics(topology)
    expected_edges = {edge.stream_id: edge for edge in expected.streams}
    operations = {item.op_id: item for item in topology.unit_operations}
    issues: list[dict[str, str]] = []
    for edge in topology.streams:
        reference = expected_edges.get(edge.stream_id)
        if reference and (
            edge.stream_type != reference.stream_type
            or edge.connection_type != reference.connection_type
        ):
            issues.append(
                {
                    "code": "FINAL-TOPOLOGY-PHASE-INCONSISTENT",
                    "message": (
                        f"Edge {edge.stream_id} ({edge.from_op} -> {edge.to_op}) is "
                        f"{edge.connection_type}/{edge.stream_type}; expected "
                        f"{reference.connection_type}/{reference.stream_type}."
                    ),
                }
            )
        if edge.connection_type == "process" and (
            operations.get(edge.from_op)
            and operations[edge.from_op].op_type in CONTROL_OPERATION_TYPES
            or operations.get(edge.to_op)
            and operations[edge.to_op].op_type in CONTROL_OPERATION_TYPES
        ):
            issues.append(
                {
                    "code": "FINAL-CONTROL-IN-PROCESS-PATH",
                    "message": (
                        f"Control operation is encoded as a process edge on {edge.stream_id}."
                    ),
                }
            )
    return _unique_issues(issues)


def merged_phase(phases: Iterable[str]) -> str:
    normalized = {str(item or "").lower() for item in phases if item}
    expanded: set[str] = set()
    for phase in normalized:
        if phase in {"gas_liquid", "liquid_gas", "multiphase"}:
            expanded.update({"gas", "liquid"})
        elif phase == "solid_liquid":
            expanded.update({"solid", "liquid"})
        elif phase == "liquid_liquid":
            expanded.add("liquid")
        else:
            expanded.add(phase)
    if "gas" in expanded and "liquid" in expanded:
        return "gas_liquid"
    if "solid" in expanded and "liquid" in expanded:
        return "solid_liquid"
    if expanded == {"gas"}:
        return "gas"
    if expanded == {"solid"}:
        return "solid"
    return "liquid"


def _source_phase(op_type: str, parameters: dict) -> str:
    op_type = str(op_type or "").lower()
    declared = str(parameters.get("phase") or "").strip().lower()
    if op_type == "mfc" or parameters.get("gas_flow_sccm") is not None:
        return "gas"
    if op_type == "pump":
        return declared or "liquid"
    return declared if declared in {
        "gas", "liquid", "solid", "gas_liquid", "solid_liquid", "liquid_liquid"
    } else ""


def _edge_phase(edge, operation_phase: dict[str, str]) -> str:
    return operation_phase.get(edge.from_op) or str(edge.stream_type or "")


def _operation_output_phase(op_type: str, parameters: dict, inputs: list[str]) -> str:
    op_type = str(op_type or "").lower()
    if op_type in CONTROL_OPERATION_TYPES:
        return "signal"
    declared = str(parameters.get("phase") or "").strip().lower()
    if declared:
        return declared
    if op_type in MIXER_OPERATION_TYPES:
        return merged_phase(inputs)
    if inputs:
        return merged_phase(inputs)
    return "liquid"


def _separator_edge_phase(op_type: str, label: str | None, inherited: str) -> str:
    if str(op_type or "").lower() not in SEPARATOR_OPERATION_TYPES:
        return inherited
    text = str(label or "").lower()
    if any(token in text for token in ("vent", "off-gas", "offgas", "gas outlet")):
        return "gas"
    if any(token in text for token in ("aqueous", "organic", "liquid", "product")):
        return "liquid"
    # A single represented separator outlet in the current graph is the product
    # liquid path; venting remains a declared utility/safety capability.
    return "liquid"


def _pid_from_process_graph(topology: ProcessTopology) -> str:
    operations = {item.op_id: item for item in topology.unit_operations}
    parts = []
    for edge in topology.streams:
        if edge.connection_type != "process":
            continue
        source = operations.get(edge.from_op)
        target = operations.get(edge.to_op)
        if not source or not target:
            continue
        parts.append(
            f"{source.label or source.op_id} -[{edge.stream_type}]-> "
            f"{target.label or target.op_id}"
        )
    return " | ".join(parts)


def _unique_issues(issues: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    output = []
    for issue in issues:
        key = (issue["code"], issue["message"])
        if key not in seen:
            seen.add(key)
            output.append(issue)
    return output
