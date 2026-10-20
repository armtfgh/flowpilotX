"""Canonical finalized-design contract shared by every output surface."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from pydantic import ValidationError
from flora_translate.component_identity import has_stationary_catalytic_role

from flora_translate.executable_artifacts import (
    artifact_semantic_issues,
    canonical_sha256,
    compile_executable_artifacts,
)
from flora_translate.residence_time_basis import (
    INLET_STP_BASIS,
    IN_CHANNEL_BASIS,
    LIQUID_ONLY_BASIS,
    UNKNOWN_BASIS,
    gas_equiv_from_stp_flow,
    normalize_residence_time_basis,
    residence_time_basis_label,
)
from flora_translate.schemas import ExecutableProcessGraph, ProcessTopology


SCHEMA_VERSION = "flowpilot_final_design_v2.0"
REACTOR_OPERATION_TYPES = {
    "coil_reactor",
    "reactor",
    "heated_coil",
    "photoreactor",
    "chip_reactor",
    "packed_bed",
    "packed_bed_reactor",
}
SEMANTIC_GATES = {
    "chemistry_identity_preserved": {
        "FINAL-CHEMISTRY-IDENTITY-DRIFT",
        "FINAL-CHEMISTRY-IDENTITY-UNCONFIRMED",
    },
    "component_stoichiometry_complete": {
        "FINAL-COMPONENT-STOICHIOMETRY-INCOMPLETE",
        "FINAL-COMPONENT-DESCRIPTIONS-CONFLICT",
    },
    "topology_phase_consistent": {"FINAL-TOPOLOGY-PHASE-INCONSISTENT"},
    "control_edges_not_in_process_path": {"FINAL-CONTROL-IN-PROCESS-PATH"},
    "safety_controls_complete": {"FINAL-SAFETY-CONTROLS-INCOMPLETE"},
    "procedure_complete": {"FINAL-PROCEDURE-INCOMPLETE"},
    "no_unsupported_operations": {"FINAL-UNSUPPORTED-EXECUTABLE-OPERATION"},
    "diagram_matches_canonical_topology": {
        "FINAL-DIAGRAM-TOPOLOGY-HASH-MISMATCH"
    },
    "residence_time_basis_unambiguous": {
        "FINAL-RESIDENCE-TIME-BASIS-UNKNOWN"
    },
    "gas_time_basis_consistent": {
        "FINAL-GAS-TIME-BASIS-INCONSISTENT"
    },
    "mixing_regime_matches_topology": {
        "FINAL-MIXING-REGIME-TOPOLOGY-CONFLICT"
    },
    "stationary_components_not_in_feeds": {
        "FINAL-STATIONARY-COMPONENT-IN-FEED"
    },
    "reagent_gas_delivery_unique": {
        "FINAL-CONFLICTING-OXIDANT-DELIVERY"
    },
    "gas_composition_and_equivalent_closure": {
        "FINAL-GAS-COMPOSITION-INCONSISTENT",
        "FINAL-GAS-EQUIVALENT-CLOSURE",
    },
}


def build_final_design_contract(result: dict[str, Any]) -> dict[str, Any]:
    """Build one authoritative design view after every pipeline refinement.

    Preliminary council proposals and deterministic inventory realizations are
    intentionally kept separate. A blocked or inconsistent result has no
    executable ``parameters`` or ``stages``; diagnostic values remain available
    under ``diagnostic`` without being presented as run instructions.
    """

    proposal = dict(result.get("proposal") or {})
    calculations = dict(result.get("design_calculations") or {})
    validation = dict(result.get("final_validation") or {})
    validation_checks = dict(validation.get("checks") or {})
    allocation = dict(result.get("inventory_allocation") or {})
    multistage = dict(result.get("multistage_inventory_plan") or {})
    chemistry_plan = dict(result.get("chemistry_plan") or {})
    stationary_components = _resolved_stationary_components(
        proposal,
        chemistry_plan,
    )

    topology_payload = dict(result.get("process_topology") or {})
    issues = _consistency_issues(
        proposal=proposal,
        calculations=calculations,
        validation=validation,
        allocation=allocation,
        multistage=multistage,
        topology=topology_payload,
    )
    reported_block = str(
        result.get("recommended_disposition")
        or (result.get("design_disposition") or {}).get("recommended_disposition")
        or ""
    ).upper() == "BLOCK"
    process_graph = None
    if not reported_block:
        try:
            process_graph = _build_executable_process_graph(topology_payload)
        except (ValidationError, ValueError, TypeError) as exc:
            issues.append(
                {
                    "code": "FINAL-EXECUTABLE-PROCESS-GRAPH-INVALID",
                    "message": str(exc),
                }
            )
    issues = _deduplicate_issues(issues)

    diagnostic_stages = _diagnostic_stages(proposal, multistage)
    preliminary = {
        "residence_time_min": _number(proposal.get("residence_time_min")),
        "residence_time_inlet_min": _number(
            proposal.get("residence_time_inlet_min")
        ),
        "residence_time_in_channel_min": _number(
            proposal.get("residence_time_in_channel_min")
        ),
        "flow_rate_mL_min": _number(proposal.get("flow_rate_mL_min")),
        "reactor_volume_mL": _number(proposal.get("reactor_volume_mL")),
        "source": "pre-final council/inventory candidate",
    }

    intensification = _intensification_summary(
        calculations=calculations,
        chemistry_plan=chemistry_plan,
        final_residence_time_min=None,
    )
    contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "parameters": None,
        "stages": [],
        "streams": [],
        "instrument_manifest": [],
        "process_graph": None,
        "chemistry_identity": None,
        "stream_components": [],
        "stationary_components": stationary_components,
        "decision_rationale": deepcopy(proposal.get("decision_rationale") or {}),
        "safety": None,
        "operating_procedure": [],
        "validation_experiments": [],
        "canonical_sha256": None,
        "artifact_provenance": {
            "authority": "deterministic_post_realization_compiler",
            "intermediate_agent_text_is_executable": False,
        },
        "intensification": intensification,
        "consistency": {
            "passed": not issues,
            "issues": issues,
            "validation_checks": validation_checks,
            "semantic_checks": _semantic_checks(issues),
        },
        "diagnostic": {
            "preliminary_candidate": preliminary,
            "stage_requirements": diagnostic_stages,
            "unresolved_inventory": allocation.get("unresolved_requirements") or [],
            "message": (
                "No executable final design exists. Values in this diagnostic "
                "section are unapproved intermediate calculations."
            ),
        },
    }
    if reported_block or issues or process_graph is None:
        contract["status"] = "blocked"
        return contract

    stages = _executable_stages(proposal, multistage)
    parameters = _canonical_parameters(proposal, stages)
    streams = list(proposal.get("streams") or [])
    issues.extend(
        _stationary_component_placement_issues(
            streams,
            stationary_components,
        )
    )
    instrument_manifest = list(
        result.get("instrument_manifest")
        or allocation.get("instrument_manifest")
        or []
    )
    artifacts = compile_executable_artifacts(
        result,
        process_graph=process_graph,
        parameters=parameters,
        streams=streams,
    )
    issues.extend(
        artifact_semantic_issues(
            result,
            process_graph=process_graph,
            bundle=artifacts,
            parameters=parameters,
        )
    )
    issues = _deduplicate_issues(issues)
    contract["consistency"]["issues"] = issues
    contract["consistency"]["passed"] = not issues
    contract["consistency"]["semantic_checks"] = _semantic_checks(issues)
    if issues:
        contract["status"] = "blocked"
        contract["diagnostic"]["semantic_artifacts"] = artifacts.model_dump(mode="json")
        contract["diagnostic"]["message"] = (
            "The numerical design closed, but the executable chemistry, topology, "
            "safety, or procedure contract did not. Diagnostic artifacts are not "
            "run instructions."
        )
        return contract

    process_graph_payload = process_graph.model_dump(mode="json")
    artifact_payload = artifacts.model_dump(mode="json")
    instrument_manifest = _augment_instrument_manifest(
        instrument_manifest,
        result.get("inventory_snapshot") or {},
        artifact_payload["safety"].get("controls") or [],
    )
    contract.update(
        {
            "status": "executable",
            "parameters": parameters,
            "stages": stages,
            "streams": streams,
            "instrument_manifest": instrument_manifest,
            "process_graph": process_graph_payload,
            "chemistry_identity": artifact_payload["chemistry_identity"],
            "stream_components": artifact_payload["stream_components"],
            "stationary_components": stationary_components,
            "safety": artifact_payload["safety"],
            "operating_procedure": artifact_payload["operating_procedure"],
            "validation_experiments": artifact_payload["validation_experiments"],
        }
    )
    contract["intensification"] = _intensification_summary(
        calculations=calculations,
        chemistry_plan=chemistry_plan,
        final_residence_time_min=parameters.get("residence_time_min"),
    )
    contract["canonical_sha256"] = canonical_sha256(
        {
            "parameters": contract["parameters"],
            "stages": contract["stages"],
            "streams": contract["streams"],
            "instrument_manifest": contract["instrument_manifest"],
            "process_graph": contract["process_graph"],
            "chemistry_identity": contract["chemistry_identity"],
            "stream_components": contract["stream_components"],
            "stationary_components": contract["stationary_components"],
            "safety": contract["safety"],
            "operating_procedure": contract["operating_procedure"],
            "validation_experiments": contract["validation_experiments"],
        }
    )
    contract["artifact_provenance"]["canonical_sha256"] = contract[
        "canonical_sha256"
    ]
    return contract


def publish_final_design_artifacts(
    result: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    """Publish only canonical executable artifacts to legacy top-level keys."""

    result["final_design"] = contract
    from flora_translate.result_reporting import build_result_report

    result["result_report"] = build_result_report(result)
    result["canonical_explanation"] = canonical_summary_markdown(contract)
    result["explanation_status"] = "pre_realization_audit_only"
    validation = result.setdefault("final_validation", {})
    checks = validation.setdefault("checks", {})
    semantic_checks = dict((contract.get("consistency") or {}).get("semantic_checks") or {})
    checks.update(semantic_checks)
    checks["final_design_contract_consistency"] = contract.get("status") == "executable"
    validation["unresolved_reasons"] = [
        name for name, passed in checks.items() if passed is False
    ]
    validation["status"] = (
        "ready" if contract.get("status") == "executable" else "blocked"
    )

    previous_safety = result.get("safety_report")
    if previous_safety and "council_safety_report_diagnostic" not in result:
        result["council_safety_report_diagnostic"] = previous_safety

    if contract.get("status") != "executable":
        semantic_artifacts = (
            (contract.get("diagnostic") or {}).get("semantic_artifacts") or {}
        )
        diagnostic_safety = dict(semantic_artifacts.get("safety") or {})
        if diagnostic_safety:
            result["safety_report"] = {
                "schema_version": "flowpilot_canonical_safety_v1.0",
                "source": "blocked_final_design_diagnostic",
                **diagnostic_safety,
                "total_checks": len(diagnostic_safety.get("controls") or []),
                "critical": len(diagnostic_safety.get("missing_control_ids") or []),
                "screen_required": True,
            }
        else:
            result.pop("safety_report", None)
        result.pop("operating_procedure", None)
        result.pop("validation_experiments", None)
        result.pop("canonical_design_sha256", None)
        return

    safety = dict(contract.get("safety") or {})
    controls = list(safety.get("controls") or [])
    experiments = list(contract.get("validation_experiments") or [])
    result["safety_report"] = {
        "schema_version": "flowpilot_canonical_safety_v1.0",
        "source": "final_design",
        "canonical_sha256": contract.get("canonical_sha256"),
        "hazards": list(safety.get("hazards") or []),
        "controls": controls,
        "total_checks": len(controls),
        "critical": sum(
            1 for item in controls if item.get("required") and not item.get("satisfied")
        ),
        "complete": bool(safety.get("complete")),
        "validation_experiments": [item.get("procedure") for item in experiments],
        "screen_required": True,
    }
    result["operating_procedure"] = list(contract.get("operating_procedure") or [])
    result["validation_experiments"] = experiments
    result["instrument_manifest"] = list(contract.get("instrument_manifest") or [])
    result["canonical_design_sha256"] = contract.get("canonical_sha256")
    result["artifact_provenance"] = dict(contract.get("artifact_provenance") or {})
    canonical_topology = ((contract.get("process_graph") or {}).get("topology") or {})
    if canonical_topology:
        result["process_topology"] = canonical_topology


def canonical_summary_markdown(contract: dict[str, Any]) -> str:
    """Render a concise summary using only the frozen final-design contract."""

    if contract.get("status") != "executable":
        issues = list((contract.get("consistency") or {}).get("issues") or [])
        lines = [
            "### No Executable Final Design",
            "The deterministic final-design contract is blocked. Do not execute intermediate candidate values.",
        ]
        for issue in issues:
            lines.append(
                f"- **{issue.get('code', 'FINAL-CHECK')}**: "
                f"{issue.get('message', 'Final consistency check failed.')}"
            )
        return "\n\n".join(lines[:2]) + (
            "\n" + "\n".join(lines[2:]) if len(lines) > 2 else ""
        )

    parameters = dict(contract.get("parameters") or {})
    fields = (
        ("residence_time_min", "Residence time", "min"),
        ("residence_time_inlet_min", "Inlet/STP apparent residence time", "min"),
        ("residence_time_in_channel_min", "In-channel apparent residence time", "min"),
        ("flow_rate_mL_min", "Total liquid flow", "mL/min"),
        ("reactor_volume_mL", "Reactor volume", "mL"),
        ("temperature_C", "Temperature", "deg C"),
        ("BPR_bar", "BPR", "bar"),
        ("concentration_M", "Concentration", "M"),
        ("wavelength_nm", "Wavelength", "nm"),
    )
    lines = ["### Final Executable Design"]
    for key, label, unit in fields:
        value = parameters.get(key)
        if value is None or value == "":
            continue
        lines.append(f"- **{label}:** {_display_value(value)} {unit}")
    basis = str(parameters.get("residence_time_basis") or "").strip()
    if basis:
        lines.append(f"- **Residence-time basis:** {basis}")
    equipment = [
        str(item.get("name") or item.get("equipment_id") or "").strip()
        for item in contract.get("instrument_manifest") or []
        if isinstance(item, dict)
    ]
    equipment = [item for item in equipment if item]
    if equipment:
        lines.append(f"- **Assigned equipment:** {', '.join(equipment)}")
    return "\n".join(lines)


def _display_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _build_executable_process_graph(
    topology_payload: dict[str, Any],
) -> ExecutableProcessGraph:
    if not topology_payload.get("unit_operations"):
        raise ValueError("an executable final design requires a compiled process topology")
    topology = ProcessTopology.model_validate(topology_payload)
    operations = {item.op_id: item for item in topology.unit_operations}

    feeds = []
    for operation in topology.unit_operations:
        if operation.op_type not in {"pump", "mfc"}:
            continue
        parameters = operation.parameters or {}
        equipment_id = str(
            operation.inventory_item_id
            or parameters.get("inventory_equipment_id")
            or ""
        ).strip()
        if not equipment_id:
            raise ValueError(
                f"feed operation {operation.op_id} has no inventory equipment assignment"
            )
        phase = str(parameters.get("phase") or "").strip().lower()
        if operation.op_type == "mfc":
            phase = "gas"
        elif not phase:
            phase = "liquid"
        feeds.append(
            {
                "operation_id": operation.op_id,
                "equipment_id": equipment_id,
                "stream_label": str(
                    parameters.get("stream") or operation.label or operation.op_id
                ),
                "phase": phase,
                "contents": list(parameters.get("contents") or []),
                "solvent": str(parameters.get("solvent") or ""),
                "flow_rate_mL_min": (
                    _positive_or_none(parameters.get("flow_rate_mL_min"))
                    if phase != "gas"
                    else None
                ),
                "gas_flow_sccm": _positive_or_none(
                    parameters.get("gas_flow_sccm")
                ),
                "gas_flow_actual_mL_min": _positive_or_none(
                    parameters.get("gas_flow_actual_mL_min")
                ),
            }
        )

    reactors = [
        item
        for item in topology.unit_operations
        if item.op_type in REACTOR_OPERATION_TYPES
    ]
    stages = []
    for index, reactor in enumerate(reactors, start=1):
        parameters = reactor.parameters or {}
        equipment_id = str(
            reactor.inventory_item_id
            or parameters.get("inventory_equipment_id")
            or ""
        ).strip()
        if not equipment_id:
            raise ValueError(
                f"reactor operation {reactor.op_id} has no inventory equipment assignment"
            )
        basis = str(parameters.get("residence_time_basis") or "")
        normalized_basis = normalize_residence_time_basis(basis)
        if normalized_basis == "unknown":
            if any(feed["phase"] == "gas" for feed in feeds):
                raise ValueError(f"reactor {reactor.op_id} requires an explicit residence-time basis in a gas process")
            basis = "liquid-only"
            normalized_basis = normalize_residence_time_basis(basis)
        liquid_flow = _number(
            parameters.get("Q_liquid_mL_min"),
            parameters.get("Q_inlet_mL_min"),
        )
        if normalized_basis == INLET_STP_BASIS:
            gas_flow = _number(parameters.get("Q_gas_sccm"))
            residence_time = _number(
                parameters.get("residence_time_inlet_min"),
                parameters.get("residence_time_min"),
            )
        elif normalized_basis == IN_CHANNEL_BASIS:
            gas_flow = _number(parameters.get("Q_gas_actual_mL_min"))
            residence_time = _number(
                parameters.get("residence_time_in_channel_min"),
                parameters.get("residence_time_min"),
            )
        else:
            gas_flow = 0.0
            residence_time = _number(parameters.get("residence_time_min"))
        mixer = _nearest_upstream_mixer(topology, reactor.op_id)
        stages.append(
            {
                "stage_number": index,
                "reactor_operation_id": reactor.op_id,
                "reactor_equipment_id": equipment_id,
                "mixer_operation_id": mixer.op_id if mixer else None,
                "mixer_equipment_id": (
                    str(mixer.inventory_item_id or "") or None if mixer else None
                ),
                "temperature_C": _number(parameters.get("temperature_C")),
                "reactor_volume_mL": _number(parameters.get("volume_mL")),
                "residence_flow_mL_min": liquid_flow + gas_flow,
                "residence_time_min": residence_time,
                "residence_time_basis": basis,
            }
        )

    return ExecutableProcessGraph.model_validate(
        {
            "topology": topology,
            "feeds": feeds,
            "stages": stages,
            "total_liquid_flow_mL_min": topology.total_flow_rate_mL_min,
            "total_reactor_volume_mL": topology.reactor_volume_mL,
            "total_residence_time_min": topology.residence_time_min,
        }
    )


def _nearest_upstream_mixer(
    topology: ProcessTopology,
    operation_id: str,
):
    operations = {item.op_id: item for item in topology.unit_operations}
    incoming: dict[str, list[str]] = {}
    for connection in topology.streams:
        incoming.setdefault(connection.to_op, []).append(connection.from_op)
    queue = list(incoming.get(operation_id, []))
    visited = set()
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        operation = operations.get(current)
        if operation is None:
            continue
        if operation.op_type in {"mixer", "quench_mixer"}:
            return operation
        queue.extend(incoming.get(current, []))
    return None


def _positive_or_none(value: Any) -> float | None:
    number = _number(value)
    return number if number > 0 else None


def canonical_proposal(result: dict[str, Any]) -> dict[str, Any]:
    """Return proposal-shaped executable data for legacy renderers."""

    contract = result.get("final_design") or build_final_design_contract(result)
    if contract.get("status") != "executable" or not contract.get("parameters"):
        return {}
    proposal = dict(result.get("proposal") or {})
    proposal.update(contract["parameters"])
    proposal["stage_parameters"] = list(contract.get("stages") or [])
    proposal["streams"] = list(contract.get("streams") or [])
    return proposal


def _consistency_issues(
    *,
    proposal: dict[str, Any],
    calculations: dict[str, Any],
    validation: dict[str, Any],
    allocation: dict[str, Any],
    multistage: dict[str, Any],
    topology: dict[str, Any],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    council_block = (proposal.get("inventory_constraints") or {}).get("council_selection_block")
    if council_block:
        issues.append({"code": "FINAL-COUNCIL-NO-APPROVED-CANDIDATE", "message": str(council_block)})
    checks = dict(validation.get("checks") or {})
    for check, passed in checks.items():
        if passed is False:
            issues.append(
                {
                    "code": f"FINAL-{str(check).upper()}",
                    "message": f"Final validation failed: {check}.",
                }
            )
    if not checks:
        issues.append(
            {
                "code": "FINAL-VALIDATION-MISSING",
                "message": "Final deterministic validation is missing.",
            }
        )

    if multistage.get("applied") and multistage.get("status") != "complete":
        issues.append(
            {
                "code": "FINAL-MULTISTAGE-INCOMPLETE",
                "message": "Not every reaction stage has unique, available inventory hardware.",
            }
        )
    if allocation and allocation.get("status") not in (
        None,
        "complete",
        "complete_with_assumptions",
    ):
        issues.append(
            {
                "code": "FINAL-INVENTORY-ALLOCATION-INCOMPLETE",
                "message": "The process topology is not fully assigned to inventory.",
            }
        )

    proposal_tau = _number(proposal.get("residence_time_min"))
    calculation_tau = _number(calculations.get("residence_time_min"))
    if proposal_tau and calculation_tau and not _close(proposal_tau, calculation_tau):
        issues.append(
            {
                "code": "FINAL-TAU-PROPOSAL-CALCULATION-MISMATCH",
                "message": (
                    f"Proposal residence time {proposal_tau:g} min differs from "
                    f"engineering residence time {calculation_tau:g} min."
                ),
            }
        )

    if multistage.get("status") == "complete":
        proposal_basis = str(proposal.get("residence_time_basis") or "").lower()
        if "liquid" in proposal_basis:
            stage_tau = _number(
                multistage.get("total_residence_time_liquid_min"),
            )
            stage_tau_label = "nominal liquid-contact"
        elif "channel" in proposal_basis:
            stage_tau = _number(multistage.get("total_residence_time_in_channel_min"))
            stage_tau_label = "pressure-corrected in-channel"
        else:
            stage_tau = _number(multistage.get("total_residence_time_inlet_min"))
            stage_tau_label = "inlet/STP"
        stage_volume = _number(multistage.get("total_reactor_volume_mL"))
        proposal_volume = _number(proposal.get("reactor_volume_mL"))
        if proposal_tau and stage_tau and not _close(proposal_tau, stage_tau):
            issues.append(
                {
                    "code": "FINAL-TAU-STAGE-TOTAL-MISMATCH",
                    "message": (
                        f"Global residence time {proposal_tau:g} min differs from "
                        f"the stagewise {stage_tau_label} total {stage_tau:g} min."
                    ),
                }
            )
        if proposal_volume and stage_volume and not _close(proposal_volume, stage_volume):
            issues.append(
                {
                    "code": "FINAL-VOLUME-STAGE-TOTAL-MISMATCH",
                    "message": (
                        f"Global reactor volume {proposal_volume:g} mL differs from "
                        f"the stagewise inventory total {stage_volume:g} mL."
                    ),
                }
            )
    elif not proposal.get("stage_parameters"):
        proposal_flow = _number(proposal.get("flow_rate_mL_min"))
        proposal_volume = _number(proposal.get("reactor_volume_mL"))
        proposal_basis = normalize_residence_time_basis(
            proposal.get("residence_time_basis")
        )
        gas_streams = [
            stream for stream in proposal.get("streams") or []
            if str(stream.get("phase") or "").lower() == "gas"
        ]
        if proposal_basis == INLET_STP_BASIS:
            proposal_flow += sum(
                _number(stream.get("gas_flow_sccm")) for stream in gas_streams
            )
        elif proposal_basis == IN_CHANNEL_BASIS:
            proposal_flow += sum(
                _number(stream.get("gas_flow_actual_mL_min")) for stream in gas_streams
            )
        if proposal_flow and proposal_tau and proposal_volume and not _close(
            proposal_volume,
            proposal_flow * proposal_tau,
            tolerance=0.01,
        ):
            issues.append(
                {
                    "code": "FINAL-VOLUME-FLOW-TIME-MISMATCH",
                    "message": (
                        "Single-stage reactor volume does not equal the flow on "
                        "the authoritative residence-time basis times residence time."
                    ),
                }
            )

    bpr = _number(proposal.get("BPR_bar"))
    absolute = _number(proposal.get("pressure_absolute_bar"))
    if absolute and not _close(absolute, bpr + 1.01325, tolerance=0.002):
        issues.append(
            {
                "code": "FINAL-PRESSURE-BASIS-MISMATCH",
                "message": "Absolute pressure is inconsistent with the gauge BPR setpoint.",
            }
        )
    issues.extend(_stream_annotation_issues(proposal))
    issues.extend(_gas_delivery_issues(proposal))
    issues.extend(_gas_composition_and_equivalent_issues(proposal))
    issues.extend(_topology_issues(proposal, topology))
    return _deduplicate_issues(issues)


def _stream_annotation_issues(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    """Reject stale operational numbers embedded in final stream labels."""

    issues: list[dict[str, Any]] = []
    for stream in proposal.get("streams") or []:
        if str(stream.get("phase") or "").lower() != "gas":
            continue
        text = " ".join(str(item) for item in stream.get("contents") or [])
        declared_sccm = re.findall(r"(\d+(?:\.\d+)?)\s*sccm\b", text, re.I)
        canonical_sccm = _number(stream.get("gas_flow_sccm"))
        if declared_sccm and canonical_sccm and any(
            not _close(float(value), canonical_sccm, tolerance=0.01)
            for value in declared_sccm
        ):
            issues.append(
                {
                    "code": "FINAL-STALE-GAS-FLOW-ANNOTATION",
                    "message": "Gas stream text conflicts with canonical inlet/STP flow.",
                }
            )
        declared_equiv = re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:equiv|eq)\b", text, re.I
        )
        canonical_equiv = _number(stream.get("molar_equiv"))
        if declared_equiv and canonical_equiv and any(
            not _close(float(value), canonical_equiv, tolerance=0.01)
            for value in declared_equiv
        ):
            issues.append(
                {
                    "code": "FINAL-STALE-GAS-EQUIV-ANNOTATION",
                    "message": "Gas stream text conflicts with canonical supplied equivalents.",
                }
            )
    return issues


def _stationary_component_placement_issues(
    streams: list[dict[str, Any]],
    stationary_components: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reject a fixed-bed component that is also serialized as pump contents."""

    stationary_names = {
        _component_key(item.get("name") or item.get("source_text"))
        for item in stationary_components
    }
    stationary_names.discard("")
    if not stationary_names:
        return []
    for stream in streams:
        for content in stream.get("contents") or []:
            content_key = _component_key(content)
            if any(
                name == content_key or name in content_key or content_key in name
                for name in stationary_names
            ):
                return [
                    {
                        "code": "FINAL-STATIONARY-COMPONENT-IN-FEED",
                        "message": (
                            "A component assigned to the stationary reactor phase "
                            "also appears in a pumped stream."
                        ),
                    }
                ]
    return []


def _resolved_stationary_components(
    proposal: dict[str, Any],
    chemistry_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    """Recover packed-bed catalyst placement independently of model wording."""

    resolved = list(
        (proposal.get("inventory_constraints") or {}).get(
            "stationary_components", []
        )
    )
    reactor_text = " ".join(
        [
            str(proposal.get("reactor_type") or ""),
            str((proposal.get("inventory_selection") or {}).get("name") or ""),
            str((proposal.get("inventory_selection") or {}).get("type") or ""),
        ]
    ).lower()
    if not any(marker in reactor_text for marker in ("packed", "catcart", "cartridge")):
        return resolved

    known = {_component_key(item.get("name")) for item in resolved}
    for reagent in chemistry_plan.get("reagents") or []:
        name = str(reagent.get("name") or "").strip()
        context = " ".join(
            [
                name,
                str(reagent.get("role") or ""),
                str(reagent.get("notes") or ""),
            ]
        ).lower()
        key = _component_key(name)
        if name and key not in known and has_stationary_catalytic_role(
            context, declared_loading=str(reagent.get("equiv_or_loading") or ""),
        ):
            resolved.append(
                {
                    "name": name,
                    "source_text": context,
                    "placement": "stationary_reactor_phase",
                    "reactor_type": proposal.get("reactor_type"),
                    "basis": "packed-bed catalyst role inferred from frozen chemistry plan",
                }
            )
            known.add(key)
    return resolved


def _gas_delivery_issues(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    """Reject air and pure O2 as simultaneous feeds to the same stage."""

    oxidants_by_stage: dict[int, set[str]] = {}
    for stream in proposal.get("streams") or []:
        if str(stream.get("phase") or "").lower() != "gas":
            continue
        text = " ".join(
            [
                *[str(item) for item in stream.get("contents") or []],
                str(stream.get("pump_role") or ""),
            ]
        ).lower().replace("₂", "2")
        identity = None
        if re.search(r"\bair\b", text):
            identity = "air"
        elif re.search(r"\b(?:oxygen|o2)\b", text):
            identity = "O2"
        if identity:
            stage = int(_number(stream.get("introduction_stage")) or 1)
            oxidants_by_stage.setdefault(stage, set()).add(identity)
    if any({"air", "O2"} <= identities for identities in oxidants_by_stage.values()):
        return [
            {
                "code": "FINAL-CONFLICTING-OXIDANT-DELIVERY",
                "message": (
                    "Air and pure O2 are serialized as simultaneous physical "
                    "feeds for one reaction stage. Select one delivery implementation."
                ),
            }
        ]
    return []


def _gas_composition_and_equivalent_issues(
    proposal: dict[str, Any],
) -> list[dict[str, Any]]:
    """Recompute active-gas equivalents from the final serialized stream."""

    issues: list[dict[str, Any]] = []
    metrics = proposal.get("multiphase_metrics") or {}
    limiting_flow = _number(metrics.get("limiting_liquid_flow_mL_min"))
    concentration = _number(metrics.get("limiting_reagent_concentration_M"))
    if not limiting_flow:
        liquid_streams = [
            stream
            for stream in proposal.get("streams") or []
            if str(stream.get("phase") or "").lower() != "gas"
        ]
        limiting_flow = min(
            (_number(stream.get("flow_rate_mL_min")) for stream in liquid_streams),
            default=0.0,
        )
    if not concentration:
        concentration = _number(proposal.get("concentration_M"))

    for stream in proposal.get("streams") or []:
        if str(stream.get("phase") or "").lower() != "gas":
            continue
        fraction = _number(stream.get("gas_reagent_mole_fraction"))
        identities = [str(item).strip().lower().replace("₂", "2") for item in stream.get("contents") or []]
        named_air = len(identities) == 1 and bool(re.fullmatch(r"(?:(?:clean|dry|compressed|filtered)\s+)*air", identities[0]))
        if named_air and not _close(fraction, 0.21, tolerance=0.001):
            issues.append({
                "code": "FINAL-GAS-COMPOSITION-INCONSISTENT",
                "message": "A named air stream must use 0.21 as the oxygen mole fraction; gas identity and composition disagree.",
            })
            # Independently check the claimed oxygen equivalents, not the
            # same erroneous fraction that produced them.
            fraction = 0.21
        pure_oxygen = len(identities) == 1 and bool(re.fullmatch(r"(?:pure\s+)?(?:o2|oxygen)(?:\s*\((?:gas|o2|oxygen)\))?", identities[0]))
        if pure_oxygen and not _close(
            fraction, 1.0, tolerance=0.001
        ):
            issues.append(
                {
                    "code": "FINAL-GAS-COMPOSITION-INCONSISTENT",
                    "message": (
                        "A pure O2 stream does not have active-reagent mole "
                        "fraction 1.0. Recompute the inlet/STP MFC setpoint."
                    ),
                }
            )
        sccm = _number(stream.get("gas_flow_sccm"))
        stated_equiv = _number(stream.get("molar_equiv"))
        if sccm <= 0 or fraction <= 0 or limiting_flow <= 0 or concentration <= 0:
            continue
        recomputed = gas_equiv_from_stp_flow(
            sccm,
            limiting_flow,
            concentration,
            fraction,
        )
        if stated_equiv > 0 and not _close(
            recomputed, stated_equiv, tolerance=0.02
        ):
            issues.append(
                {
                    "code": "FINAL-GAS-EQUIVALENT-CLOSURE",
                    "message": (
                        "The reported gas equivalents do not match the final "
                        "inlet/STP flow, active-gas fraction, and limiting-liquid feed."
                    ),
                }
            )
    return issues


def _component_key(value: Any) -> str:
    text = re.sub(r"\([^)]*\)", " ", str(value or "")).lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _topology_issues(
    proposal: dict[str, Any],
    topology: dict[str, Any],
) -> list[dict[str, Any]]:
    operations = list(topology.get("unit_operations") or [])
    if not operations:
        return []
    issues = []
    expected_volume = _number(proposal.get("reactor_volume_mL"))
    expected_tau = _number(proposal.get("residence_time_min"))
    expected_flow = _number(proposal.get("flow_rate_mL_min"))
    if expected_flow and not _close(
        expected_flow,
        _number(topology.get("total_flow_rate_mL_min")),
        tolerance=0.01,
    ):
        issues.append(
            {
                "code": "FINAL-TOPOLOGY-GLOBAL-FLOW-MISMATCH",
                "message": "Topology outlet flow differs from the final proposal.",
            }
        )
    if expected_volume and not _close(
        expected_volume, _number(topology.get("reactor_volume_mL")), tolerance=0.01
    ):
        issues.append(
            {
                "code": "FINAL-TOPOLOGY-GLOBAL-VOLUME-MISMATCH",
                "message": "Topology total reactor volume differs from the final proposal.",
            }
        )
    if expected_tau and not _close(
        expected_tau, _number(topology.get("residence_time_min")), tolerance=0.01
    ):
        issues.append(
            {
                "code": "FINAL-TOPOLOGY-GLOBAL-TAU-MISMATCH",
                "message": "Topology global residence time differs from the final proposal.",
            }
        )

    reactor_types = {
        "coil_reactor", "reactor", "heated_coil", "photoreactor",
        "chip_reactor", "packed_bed", "packed_bed_reactor",
    }
    reactors = [
        item for item in operations if item.get("op_type") in reactor_types
    ]
    node_volume = sum(
        _number((item.get("parameters") or {}).get("volume_mL"))
        for item in reactors
    )
    if expected_volume and reactors and not _close(
        expected_volume, node_volume, tolerance=0.01
    ):
        issues.append(
            {
                "code": "FINAL-TOPOLOGY-REACTOR-VOLUME-MISMATCH",
                "message": "Summed reactor-node volume differs from the final proposal.",
            }
        )

    stage_parameters = list(proposal.get("stage_parameters") or [])
    if stage_parameters:
        if len(stage_parameters) != len(reactors):
            issues.append(
                {
                    "code": "FINAL-TOPOLOGY-STAGE-COUNT-MISMATCH",
                    "message": "Topology reactor count differs from the finalized stage count.",
                }
            )
        else:
            ordered_stages = sorted(
                stage_parameters,
                key=lambda item: _number(item.get("stage_number")),
            )
            for stage, reactor in zip(ordered_stages, reactors):
                parameters = reactor.get("parameters") or {}
                expected_stage_volume = _number(
                    stage.get("reactor_volume_mL"), stage.get("V_R_mL")
                )
                if expected_stage_volume and not _close(
                    expected_stage_volume,
                    _number(parameters.get("volume_mL")),
                    tolerance=0.01,
                ):
                    issues.append(
                        {
                            "code": "FINAL-TOPOLOGY-STAGE-VOLUME-MISMATCH",
                            "message": "A topology reactor volume differs from its finalized stage.",
                        }
                    )
                expected_stage_flow = _number(
                    stage.get("Q_liquid_mL_min"),
                    stage.get("cumulative_flow_mL_min"),
                )
                topology_stage_flow = _number(
                    parameters.get("Q_liquid_mL_min"),
                    parameters.get("Q_inlet_mL_min"),
                )
                if expected_stage_flow and not _close(
                    expected_stage_flow,
                    topology_stage_flow,
                    tolerance=0.01,
                ):
                    issues.append(
                        {
                            "code": "FINAL-TOPOLOGY-STAGE-FLOW-MISMATCH",
                            "message": "A topology reactor flow differs from its finalized stage.",
                        }
                    )
                for field in (
                    "residence_time_inlet_min",
                    "residence_time_in_channel_min",
                ):
                    expected_stage_tau = _number(stage.get(field))
                    if expected_stage_tau and not _close(
                        expected_stage_tau,
                        _number(parameters.get(field)),
                        tolerance=0.01,
                    ):
                        issues.append(
                            {
                                "code": "FINAL-TOPOLOGY-STAGE-TAU-MISMATCH",
                                "message": (
                                    f"Topology {field} differs from its finalized stage."
                                ),
                            }
                        )
    elif len(reactors) > 1:
        node_tau = sum(
            _number((item.get("parameters") or {}).get("residence_time_min"))
            for item in reactors
        )
        if expected_tau and not _close(expected_tau, node_tau, tolerance=0.01):
            issues.append(
                {
                    "code": "FINAL-TOPOLOGY-REACTOR-TAU-MISMATCH",
                    "message": (
                        f"Summed reactor-node residence time {node_tau:g} min differs "
                        f"from final residence time {expected_tau:g} min."
                    ),
                }
            )
    return issues


def _diagnostic_stages(
    proposal: dict[str, Any],
    multistage: dict[str, Any],
) -> list[dict[str, Any]]:
    stages = list(multistage.get("stage_parameters") or proposal.get("stage_parameters") or [])
    unresolved_numbers = {
        int(item.get("stage_number"))
        for item in multistage.get("unresolved_requirements") or []
        if _number(item.get("stage_number"))
    }
    output = []
    for stage in stages:
        item = dict(stage)
        number = int(_number(item.get("stage_number")) or len(output) + 1)
        resolved = bool(item.get("inventory_resolved")) and number not in unresolved_numbers
        output.append(
            {
                "stage_number": number,
                "stage_name": item.get("stage_name") or f"Stage {number}",
                "inventory_resolved": resolved,
                "reactor_equipment_id": item.get("reactor_equipment_id"),
                "light_equipment_id": item.get("light_equipment_id"),
                "reactor_volume_mL": _number(item.get("reactor_volume_mL")),
                "flow_rate_mL_min": _number(item.get("Q_liquid_mL_min")),
                "residence_time_inlet_min": (
                    _number(item.get("residence_time_inlet_min")) if resolved else None
                ),
                "residence_time_in_channel_min": (
                    _number(item.get("residence_time_in_channel_min")) if resolved else None
                ),
                "status": "resolved diagnostic" if resolved else "unresolved",
            }
        )
    return output


def _executable_stages(
    proposal: dict[str, Any],
    multistage: dict[str, Any],
) -> list[dict[str, Any]]:
    raw = list(multistage.get("stage_parameters") or proposal.get("stage_parameters") or [])
    basis = normalize_residence_time_basis(proposal.get("residence_time_basis"))
    output = []
    for index, stage in enumerate(raw):
        item = dict(stage)
        if basis == INLET_STP_BASIS:
            primary_tau = _number(
                item.get("residence_time_inlet_min"),
                item.get("residence_time_min"),
            )
        elif basis == IN_CHANNEL_BASIS:
            primary_tau = _number(
                item.get("residence_time_in_channel_min"),
                item.get("residence_time_min"),
            )
        else:
            primary_tau = _number(
                item.get("residence_time_min"),
                item.get("residence_time_inlet_min"),
            )
        output.append(
            {
                **item,
                "stage_number": int(_number(item.get("stage_number")) or index + 1),
                "residence_time_min": primary_tau,
            }
        )
    return output


def _canonical_parameters(
    proposal: dict[str, Any],
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_basis = normalize_residence_time_basis(
        proposal.get("residence_time_basis")
    )
    has_gas_stream = any(
        str(item.get("phase") or "").lower() == "gas"
        for item in proposal.get("streams") or []
    )
    if normalized_basis == UNKNOWN_BASIS and not has_gas_stream:
        normalized_basis = LIQUID_ONLY_BASIS
    parameters = {
        key: proposal.get(key)
        for key in (
            "flow_rate_mL_min",
            "temperature_C",
            "concentration_M",
            "BPR_bar",
            "BPR_basis",
            "pressure_absolute_bar",
            "reactor_type",
            "tubing_material",
            "tubing_ID_mm",
            "wavelength_nm",
        )
    }
    parameters["residence_time_basis_code"] = normalized_basis
    parameters["residence_time_basis"] = residence_time_basis_label(normalized_basis)
    parameters["residence_time_basis_detail"] = str(
        proposal.get("residence_time_basis")
        or parameters["residence_time_basis"]
    )
    if stages:
        parameters.update(
            {
                # For multistage systems this is the final liquid outlet flow;
                # each stage retains its own reactor-inlet flow below.
                "flow_rate_mL_min": _number(proposal.get("flow_rate_mL_min")),
                "flow_rate_basis": "final liquid outlet flow",
                "stage_specific_flow_rates": len(
                    {
                        round(
                            _number(
                                item.get("Q_liquid_mL_min"),
                                item.get("flow_rate_mL_min"),
                            ),
                            8,
                        )
                        for item in stages
                    }
                ) > 1,
                "temperature_C": _uniform_stage_value(stages, "temperature_C"),
                "tubing_material": _uniform_stage_value(stages, "material"),
                "tubing_ID_mm": _uniform_stage_value(
                    stages, "d_mm", "tubing_ID_mm"
                ),
                "wavelength_nm": _uniform_stage_value(
                    stages, "wavelength_nm"
                ),
                "stage_specific_conditions": len(stages) > 1,
                "reactor_volume_mL": round(
                    sum(_number(item.get("reactor_volume_mL")) for item in stages), 4
                ),
                "residence_time_min": round(
                    sum(_number(item.get("residence_time_min")) for item in stages),
                    4,
                ),
                "residence_time_inlet_min": round(
                    sum(_number(item.get("residence_time_inlet_min"), item.get("residence_time_min")) for item in stages),
                    4,
                ),
                "residence_time_in_channel_min": round(
                    sum(_number(item.get("residence_time_in_channel_min"), item.get("residence_time_min")) for item in stages),
                    4,
                ),
                "residence_time_basis": (
                    "sum of per-stage " + residence_time_basis_label(normalized_basis)
                ),
            }
        )
    else:
        parameters.update(
            {
                "reactor_volume_mL": _number(proposal.get("reactor_volume_mL")),
                "residence_time_min": _number(proposal.get("residence_time_min")),
                "residence_time_inlet_min": _number(
                    proposal.get("residence_time_inlet_min"),
                    proposal.get("residence_time_min"),
                ),
                "residence_time_in_channel_min": _number(
                    proposal.get("residence_time_in_channel_min"),
                    proposal.get("residence_time_min"),
                ),
            }
        )
    return parameters


def _uniform_stage_value(
    stages: list[dict[str, Any]],
    *keys: str,
) -> Any:
    """Return a shared stage value, or None when the final stages differ."""

    values: list[Any] = []
    for stage in stages:
        value = next(
            (stage.get(key) for key in keys if stage.get(key) is not None),
            None,
        )
        if value is not None:
            values.append(value)
    if not values:
        return None
    first = values[0]
    if all(isinstance(value, (int, float)) for value in values):
        first_number = float(first)
        if all(_close(first_number, float(value)) for value in values[1:]):
            return first_number
        return None
    normalized = {str(value).strip().lower() for value in values}
    return first if len(normalized) == 1 else None


def _intensification_summary(
    *,
    calculations: dict[str, Any],
    chemistry_plan: dict[str, Any],
    final_residence_time_min: float | None,
) -> dict[str, Any]:
    mandate = dict(chemistry_plan.get("intensification_mandate") or {})
    if (chemistry_plan.get("scientific_context") or {}).get("mode") == "scientific_v2":
        return {"policy": "scientific_screening", "target_factor": None, "hypothesis_factor": None,
                "target_role": "no mandated intensification", "realized_factor": None,
                "applied_as_hard_constraint": False, "basis": "Stage-specific batch anchors; flow kinetics uncharacterized."}
    batch_time_s = _number(calculations.get("batch_time_s"))
    realized = None
    if final_residence_time_min and batch_time_s > 0:
        realized = round(batch_time_s / (60.0 * final_residence_time_min), 3)
    try:
        from flora_translate.config import FLOW_TRANSLATION_POLICY

        policy = FLOW_TRANSLATION_POLICY
    except Exception:
        policy = "evidence_first"
    return {
        "policy": policy,
        "target_factor": (
            _number(mandate.get("tau_reduction_target")) or None
            if policy == "intensify"
            else None
        ),
        "hypothesis_factor": _number(mandate.get("tau_reduction_target")) or None,
        "target_role": (
            "hard design objective"
            if policy == "intensify"
            else "not applied under evidence-first policy"
        ),
        "realized_factor": realized,
        "applied_as_hard_constraint": policy == "intensify",
        "basis": mandate.get("flow_justification_basis") or "",
    }


def _number(value: Any, fallback: Any = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(fallback)
        except (TypeError, ValueError):
            return 0.0


def _close(left: float, right: float, tolerance: float = 1e-3) -> bool:
    scale = max(abs(left), abs(right), 1.0)
    return abs(left - right) / scale <= tolerance


def _deduplicate_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for issue in issues:
        code = str(issue.get("code") or issue.get("message"))
        if code in seen:
            continue
        seen.add(code)
        output.append(issue)
    return output


def _semantic_checks(issues: list[dict[str, Any]]) -> dict[str, bool]:
    codes = {str(item.get("code") or "") for item in issues}
    return {
        gate: not bool(codes & failure_codes)
        for gate, failure_codes in SEMANTIC_GATES.items()
    }


def _augment_instrument_manifest(
    manifest: list[dict[str, Any]],
    inventory: dict[str, Any],
    safety_controls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Include inventory-bound safety accessories in the canonical manifest."""

    output = [dict(item) for item in manifest]
    present = {str(item.get("equipment_id") or "") for item in output}
    inventory_map: dict[str, tuple[str, dict[str, Any]]] = {}
    for category, items in inventory.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and item.get("equipment_id"):
                inventory_map[str(item["equipment_id"])] = (category, item)
    roles: dict[str, list[str]] = {}
    for control in safety_controls:
        for equipment_id in control.get("equipment_ids") or []:
            roles.setdefault(str(equipment_id), []).append(
                f"Safety control {control.get('control_id')}"
            )
    for equipment_id, equipment_roles in roles.items():
        if equipment_id in present:
            continue
        category, item = inventory_map.get(equipment_id, ("safety_accessories", {}))
        output.append(
            {
                "equipment_id": equipment_id,
                "name": item.get("name") or equipment_id,
                "category": category,
                "quantity_used": 1,
                "roles": sorted(set(equipment_roles)),
                "settings": [],
            }
        )
        present.add(equipment_id)
    return output
