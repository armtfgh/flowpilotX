"""Deterministic final disposition for FlowPilot designs.

The LLM council may propose, revise, or screen candidates, but it is not the
authority for hard inventory compatibility. This module converts the final
validated state into one explicit machine-readable decision:

- BLOCK: at least one hard requirement or deterministic validation gate fails.
- SCREEN: a feasible first experiment still requiring wet-lab validation.
- EXECUTE: reserved for externally validated designs (not emitted by default).
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    FlowProposal,
    GasHardwareSpec,
    LabInventory,
    ReactorSpec,
)
from flora_translate.inventory_constraints import available_pressure_settings


BLOCKING_VALIDATION_CHECKS = {
    "reactor_inventory_match",
    "pump_flow_feasible",
    "tubing_feasible",
    "geometry_closure",
    "calculation_matches_serialized_design",
    "topology_matches_serialized_design",
    "inventory_topology_assignment_complete",
    "gas_bookkeeping_complete",
}
UNAVAILABLE_STATUS_MARKERS = {
    "prohibited",
    "unavailable",
    "out_of_service",
    "out of service",
    "not_available",
    "not available",
    "disabled",
}


@dataclass(frozen=True)
class ProcessRequirements:
    temperature_exact_C: float | None = None
    temperature_min_C: float | None = None
    temperature_max_C: float | None = None
    wavelength_min_nm: float | None = None
    wavelength_max_nm: float | None = None
    pressure_min_bar: float | None = None
    pressure_max_bar: float | None = None
    required_gases: tuple[str, ...] = ()
    source_text: str = ""


@dataclass(frozen=True)
class GateFinding:
    finding_id: str
    category: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DesignDispositionDecision:
    schema_version: str
    recommended_disposition: str
    rationale: str
    hard_failures: tuple[GateFinding, ...]
    screen_reasons: tuple[str, ...]
    requirements: ProcessRequirements

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_design_disposition(
    proposal: FlowProposal,
    *,
    final_validation: dict[str, Any] | None,
    inventory: LabInventory | None,
    batch_record: BatchRecord | None,
    chemistry_plan: ChemistryPlan | None,
    objective: str = "",
    hard_constraints: Any = None,
    council_safety_report: dict[str, Any] | None = None,
) -> DesignDispositionDecision:
    """Evaluate hard feasibility after all council and calculator revisions."""

    requirements = normalize_process_requirements(
        objective=objective,
        hard_constraints=hard_constraints,
        batch_record=batch_record,
        chemistry_plan=chemistry_plan,
    )
    failures: list[GateFinding] = []
    screen_reasons: list[str] = []

    failures.extend(_validation_failures(final_validation))
    failures.extend(_proposal_block_flags(proposal))
    failures.extend(
        _temperature_failures(proposal, inventory, requirements)
    )
    failures.extend(
        _wavelength_failures(proposal, inventory, requirements)
    )
    failures.extend(_pressure_failures(proposal, inventory, requirements))
    failures.extend(
        _unconfirmed_pressure_hardware_failures(
            proposal,
            inventory,
            hard_constraints=hard_constraints,
        )
    )
    failures.extend(
        _gas_hardware_failures(
            proposal,
            inventory,
            requirements,
            chemistry_plan=chemistry_plan,
        )
    )
    failures.extend(
        _operation_prohibition_failures(
            inventory,
            batch_record=batch_record,
            chemistry_plan=chemistry_plan,
        )
    )
    failures.extend(
        _multistage_failures(inventory, chemistry_plan=chemistry_plan)
    )

    safety_report = council_safety_report or {}
    fallback_reason = str(safety_report.get("fallback_reason") or "").strip()
    if fallback_reason:
        screen_reasons.append(f"Council fallback: {fallback_reason}")
    if safety_report.get("screen_required"):
        screen_reasons.append(
            str(safety_report.get("screen_reason") or "Council requires screening")
        )
    if final_validation and final_validation.get("status") != "ready":
        screen_reasons.append(
            "Final deterministic validation is not ready"
        )
    if not proposal.engine_validated:
        screen_reasons.append("Proposal is not engine-validated")
    for flag in proposal.safety_flags or []:
        text = str(flag)
        if "screen_required" in text.lower() or "engine_fallback" in text.lower():
            screen_reasons.append(text)

    failures = _deduplicate_findings(failures)
    screen_reasons = _deduplicate_strings(screen_reasons)
    if failures:
        disposition = "BLOCK"
        rationale = (
            f"Blocked by {len(failures)} deterministic hard-feasibility "
            "failure(s); do not execute the proposed experiment."
        )
    else:
        disposition = "SCREEN"
        rationale = (
            "No deterministic hard-feasibility conflict was found. Treat the "
            "design as a first experimental screen until wet-lab evidence validates it."
        )

    return DesignDispositionDecision(
        schema_version="flowpilot_design_disposition_v1.0",
        recommended_disposition=disposition,
        rationale=rationale,
        hard_failures=tuple(failures),
        screen_reasons=tuple(screen_reasons),
        requirements=requirements,
    )


def apply_design_disposition_gate(
    result: dict[str, Any],
    *,
    proposal: FlowProposal,
    final_validation: dict[str, Any] | None,
    inventory: LabInventory | None,
    batch_record: BatchRecord | None,
    chemistry_plan: ChemistryPlan | None,
    objective: str = "",
    hard_constraints: Any = None,
    council_safety_report: dict[str, Any] | None = None,
) -> DesignDispositionDecision:
    """Attach one authoritative disposition to a serialized pipeline result."""

    decision = evaluate_design_disposition(
        proposal,
        final_validation=final_validation,
        inventory=inventory,
        batch_record=batch_record,
        chemistry_plan=chemistry_plan,
        objective=objective,
        hard_constraints=hard_constraints,
        council_safety_report=council_safety_report,
    )
    payload = decision.to_dict()
    result["design_disposition"] = payload
    result["recommended_disposition"] = decision.recommended_disposition
    result["reported_disposition"] = decision.recommended_disposition
    result["disposition_rationale"] = decision.rationale

    if final_validation is not None:
        final_validation["design_disposition"] = payload
        if decision.recommended_disposition == "BLOCK":
            final_validation["status"] = "blocked"
            final_validation["blocking_reasons"] = [
                finding.message for finding in decision.hard_failures
            ]
        # Callers may have serialized a detached validation copy before running
        # this final gate. Replace it so the persisted result cannot disagree
        # with the authoritative top-level disposition.
        result["final_validation"] = copy.deepcopy(final_validation)

    proposal_data = result.get("proposal")
    if not isinstance(proposal_data, dict):
        proposal_data = proposal.model_dump()
        result["proposal"] = proposal_data
    proposal_data["recommended_disposition"] = decision.recommended_disposition
    proposal_data["disposition_rationale"] = decision.rationale
    if decision.recommended_disposition == "BLOCK":
        proposal_data["engine_validated"] = False
        flags = proposal_data.setdefault("safety_flags", [])
        if not isinstance(flags, list):
            flags = [str(flags)]
            proposal_data["safety_flags"] = flags
        for finding in decision.hard_failures:
            marker = f"BLOCKED [{finding.finding_id}]: {finding.message}"
            if marker not in flags:
                flags.append(marker)
        proposal.engine_validated = False
        proposal.safety_flags = list(flags)
    return decision


def normalize_process_requirements(
    *,
    objective: str,
    hard_constraints: Any,
    batch_record: BatchRecord | None,
    chemistry_plan: ChemistryPlan | None,
) -> ProcessRequirements:
    """Normalize structured or textual process constraints without an LLM."""

    raw_text = str(getattr(batch_record, "raw_text", "") or "")
    extracted_objective, extracted_constraints = _canonical_sections(raw_text)
    text = "\n".join(
        part
        for part in (
            str(objective or ""),
            extracted_objective,
            _flatten_constraint_text(hard_constraints),
            extracted_constraints,
        )
        if part.strip()
    )
    lower = text.lower()

    exact_temperature = _first_number(
        text,
        (
            r"temperature\s+must\s+be\s+exactly\s+(-?\d+(?:\.\d+)?)\s*(?:°?\s*c|deg\s*c)",
            r"operate\s+at\s+(-?\d+(?:\.\d+)?)\s*(?:°?\s*c|deg\s*c)",
            r"preserv(?:e|ing)\b[\s\S]{0,80}?temperature\b[\s\S]{0,25}?(-?\d+(?:\.\d+)?)\s*(?:°?\s*c|deg\s*c)",
        ),
    )
    if (
        exact_temperature is None
        and "changing that temperature is not allowed" in lower
        and batch_record is not None
    ):
        exact_temperature = batch_record.temperature_C

    temperature_range = _first_range(
        text,
        (
            r"temperature\s+(?:must\s+be\s+)?between\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)\s*(?:°?\s*c|deg\s*c)",
            r"temperature\s+(?:must\s+be\s+)?(?:in|within)\s+(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s*(?:°?\s*c|deg\s*c)",
        ),
    )
    wavelength_range = _first_range(
        text,
        (
            r"(?:light\s+source|wavelength|excitation(?:\s+window)?)\s+(?:must\s+be\s+)?between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*nm",
            r"(?:validated\s+)?(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*nm\s+(?:excitation\s+)?window",
        ),
    )
    pressure_range = _first_range(
        text,
        (
            r"operate\s+between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*bar",
            r"pressure\s+(?:must\s+be\s+)?(?:between|within)\s+(\d+(?:\.\d+)?)\s+(?:and|[-–])\s+(\d+(?:\.\d+)?)\s*bar",
        ),
    )

    gases: list[str] = []
    if re.search(r"\bhydrogen\b[\s\S]{0,35}\b(required|cannot be substituted)\b", lower):
        gases.append("H2")
    if re.search(r"\b(oxygen|o2|o₂|air)\b[\s\S]{0,35}\b(required|reagent)\b", lower):
        gases.append("O2")
    if chemistry_plan is not None and chemistry_plan.o2_is_reagent:
        gases.append("O2")

    structured = hard_constraints if isinstance(hard_constraints, dict) else {}
    exact_temperature = _structured_number(
        structured,
        ("temperature_C", "required_temperature_C", "temperature_exact_C"),
        exact_temperature,
    )
    wavelength_range = _structured_range(
        structured,
        ("wavelength_range_nm", "wavelength_nm"),
        wavelength_range,
    )
    pressure_range = _structured_range(
        structured,
        ("pressure_range_bar", "pressure_bar"),
        pressure_range,
    )

    return ProcessRequirements(
        temperature_exact_C=exact_temperature,
        temperature_min_C=temperature_range[0] if temperature_range else None,
        temperature_max_C=temperature_range[1] if temperature_range else None,
        wavelength_min_nm=wavelength_range[0] if wavelength_range else None,
        wavelength_max_nm=wavelength_range[1] if wavelength_range else None,
        pressure_min_bar=pressure_range[0] if pressure_range else None,
        pressure_max_bar=pressure_range[1] if pressure_range else None,
        required_gases=tuple(sorted(set(gases))),
        source_text=text,
    )


def _validation_failures(
    final_validation: dict[str, Any] | None,
) -> list[GateFinding]:
    if not final_validation:
        return []
    checks = final_validation.get("checks") or {}
    return [
        GateFinding(
            finding_id=f"FINAL-{name.upper()}",
            category="final_validation",
            message=f"Final deterministic validation failed: {name}.",
            evidence={"check": name, "passed": False},
        )
        for name, passed in checks.items()
        if name in BLOCKING_VALIDATION_CHECKS and not bool(passed)
    ]


def _proposal_block_flags(proposal: FlowProposal) -> list[GateFinding]:
    findings = []
    for index, flag in enumerate(proposal.safety_flags or [], start=1):
        text = str(flag).strip()
        if text.lower().startswith("blocked:") or text.lower().startswith("block:"):
            findings.append(
                GateFinding(
                    finding_id=f"PROPOSAL-BLOCK-{index}",
                    category="proposal_guard",
                    message=text,
                    evidence={"safety_flag": text},
                )
            )
    return findings


def _temperature_failures(
    proposal: FlowProposal,
    inventory: LabInventory | None,
    requirements: ProcessRequirements,
) -> list[GateFinding]:
    failures = []
    required_min = requirements.temperature_min_C
    required_max = requirements.temperature_max_C
    exact = requirements.temperature_exact_C
    if exact is not None:
        required_min = required_max = exact
        if abs(float(proposal.temperature_C) - exact) > 0.5:
            failures.append(
                GateFinding(
                    finding_id="HARD-TEMPERATURE-PROPOSAL",
                    category="hard_constraint",
                    message=(
                        f"Required temperature is {exact:g} C, but the final "
                        f"proposal uses {proposal.temperature_C:g} C."
                    ),
                    evidence={
                        "required_temperature_C": exact,
                        "proposal_temperature_C": proposal.temperature_C,
                    },
                )
            )
    if required_min is None and required_max is None:
        return failures
    if inventory is None or (not inventory.reactors and not inventory.tubing):
        return failures

    target_low = required_min if required_min is not None else -math.inf
    target_high = required_max if required_max is not None else math.inf
    reactor_ok = not inventory.reactors or any(
        _reactor_supports_temperature(reactor, target_low, target_high)
        for reactor in inventory.reactors
    )
    tubing_ok = not inventory.tubing or any(
        tubing.max_temperature_C >= target_low for tubing in inventory.tubing
    )
    if not reactor_ok or not tubing_ok:
        failures.append(
            GateFinding(
                finding_id="INVENTORY-TEMPERATURE-CAPABILITY",
                category="inventory",
                message=(
                    "No listed reactor/tubing combination can satisfy the "
                    "required temperature."
                ),
                evidence={
                    "required_range_C": [required_min, required_max],
                    "reactor_capability": reactor_ok,
                    "tubing_capability": tubing_ok,
                },
            )
        )
    return failures


def _wavelength_failures(
    proposal: FlowProposal,
    inventory: LabInventory | None,
    requirements: ProcessRequirements,
) -> list[GateFinding]:
    low = requirements.wavelength_min_nm
    high = requirements.wavelength_max_nm
    if low is None and high is None:
        return []
    low = low if low is not None else -math.inf
    high = high if high is not None else math.inf
    failures = []
    wavelength = proposal.wavelength_nm
    if wavelength is None or not low <= float(wavelength) <= high:
        failures.append(
            GateFinding(
                finding_id="HARD-WAVELENGTH-PROPOSAL",
                category="hard_constraint",
                message=(
                    f"Final wavelength {wavelength!r} nm is outside the "
                    f"required {low:g}-{high:g} nm window."
                ),
                evidence={
                    "required_range_nm": [low, high],
                    "proposal_wavelength_nm": wavelength,
                },
            )
        )
    if inventory is None:
        return failures
    available = [
        float(source.wavelength_nm) for source in inventory.light_sources
    ]
    available.extend(
        float(reactor.wavelength_nm)
        for reactor in inventory.reactors
        if reactor.wavelength_nm is not None
    )
    if available and not any(low <= value <= high for value in available):
        failures.append(
            GateFinding(
                finding_id="INVENTORY-WAVELENGTH-CAPABILITY",
                category="inventory",
                message=(
                    "No listed light source or photoreactor satisfies the "
                    "required excitation window."
                ),
                evidence={
                    "required_range_nm": [low, high],
                    "available_wavelengths_nm": sorted(set(available)),
                },
            )
        )
    return failures


def _pressure_failures(
    proposal: FlowProposal,
    inventory: LabInventory | None,
    requirements: ProcessRequirements,
) -> list[GateFinding]:
    low = requirements.pressure_min_bar
    high = requirements.pressure_max_bar
    if low is None and high is None:
        return []
    low_value = low if low is not None else -math.inf
    high_value = high if high is not None else math.inf
    failures = []
    if not low_value <= float(proposal.BPR_bar) <= high_value:
        failures.append(
            GateFinding(
                finding_id="HARD-PRESSURE-PROPOSAL",
                category="hard_constraint",
                message=(
                    f"Final pressure {proposal.BPR_bar:g} bar is outside the "
                    f"required {low_value:g}-{high_value:g} bar range."
                ),
                evidence={
                    "required_range_bar": [low, high],
                    "proposal_BPR_bar": proposal.BPR_bar,
                },
            )
        )
    pressure_settings = available_pressure_settings(inventory)
    continuous_match = bool(
        inventory is not None
        and any(
            (item.min_pressure_bar is None or high_value >= item.min_pressure_bar)
            and (item.max_pressure_bar is None or low_value <= item.max_pressure_bar)
            for item in inventory.pressure_controllers
        )
    )
    if pressure_settings and not continuous_match and not any(
        low_value <= value <= high_value for value in pressure_settings
    ):
        failures.append(
            GateFinding(
                finding_id="INVENTORY-PRESSURE-CAPABILITY",
                category="inventory",
                message="No listed BPR setting satisfies the required pressure range.",
                evidence={
                    "required_range_bar": [low, high],
                    "available_BPR_bar": pressure_settings,
                },
            )
        )
    return failures


def _gas_hardware_failures(
    proposal: FlowProposal,
    inventory: LabInventory | None,
    requirements: ProcessRequirements,
    *,
    chemistry_plan: ChemistryPlan | None,
) -> list[GateFinding]:
    required = list(requirements.required_gases)
    if not required and chemistry_plan is not None and chemistry_plan.o2_is_reagent:
        required = ["O2"]
    required.extend(_proposal_required_gases(proposal))
    required = sorted(set(required))
    if not required or inventory is None:
        return []

    if not inventory.gas_hardware:
        return [
            GateFinding(
                finding_id="INVENTORY-GAS-HARDWARE-MISSING",
                category="inventory",
                message="A gas reagent is required, but no gas-delivery hardware is listed.",
                evidence={"required_gases": required},
            )
        ]

    failures = []
    available = [
        item for item in inventory.gas_hardware if _service_available(item)
    ]
    for gas in required:
        controllers = [
            item
            for item in available
            if _is_gas_controller(item) and _gas_matches(item, gas)
        ]
        if not controllers:
            failures.append(
                GateFinding(
                    finding_id=f"INVENTORY-GAS-{gas}",
                    category="inventory",
                    message=f"No available certified gas controller supports required {gas}.",
                    evidence={
                        "required_gas": gas,
                        "gas_hardware": [
                            item.model_dump() for item in inventory.gas_hardware
                        ],
                    },
                )
            )
            continue
        gas_flows = [
            float(stream.gas_flow_sccm or 0.0)
            for stream in proposal.streams or []
            if str(stream.phase or "").lower() == "gas"
        ]
        total_flow = sum(gas_flows)
        if total_flow <= 0:
            failures.append(
                GateFinding(
                    finding_id=f"PROPOSAL-GAS-FLOW-{gas}",
                    category="hard_constraint",
                    message=f"The required {gas} stream has no positive inlet/STP flow.",
                    evidence={"gas_flow_sccm": total_flow},
                )
            )
        elif not any(_controller_supports(item, total_flow, proposal.BPR_bar) for item in controllers):
            failures.append(
                GateFinding(
                    finding_id=f"INVENTORY-GAS-RANGE-{gas}",
                    category="inventory",
                    message=(
                        f"Required {gas} flow/pressure is outside every listed "
                        "controller operating range."
                    ),
                    evidence={
                        "gas_flow_sccm": total_flow,
                        "pressure_bar": proposal.BPR_bar,
                    },
                )
            )

    integrated_mixers = [item for item in available if _is_gas_liquid_mixer(item)]
    declared_mixers = list(inventory.mixers or [])
    available_mixers = [
        item for item in declared_mixers if _service_available(item)
    ]
    mixers = [*integrated_mixers, *available_mixers]
    if declared_mixers and not mixers:
        failures.append(
            GateFinding(
                finding_id="INVENTORY-GAS-LIQUID-MIXER",
                category="inventory",
                message="Every declared gas-liquid mixer is unavailable.",
                evidence={
                    "mixers": [
                        item.model_dump() for item in declared_mixers
                    ]
                },
            )
        )
    elif mixers and not any(
        item.max_pressure_bar is None
        or proposal.BPR_bar <= item.max_pressure_bar + 1e-9
        for item in mixers
    ):
        failures.append(
            GateFinding(
                finding_id="INVENTORY-GAS-LIQUID-MIXER-PRESSURE",
                category="inventory",
                message="Final pressure exceeds every listed gas-liquid mixer rating.",
                evidence={"pressure_bar": proposal.BPR_bar},
            )
        )
    # If no mixer item is declared, the topology allocator represents a
    # generic passive T-mixer as VERIFY-before-run. Absence from an inventory
    # document is not evidence that this standard fitting is unavailable.
    return failures


def _proposal_required_gases(proposal: FlowProposal) -> list[str]:
    required: list[str] = []
    for stream in proposal.streams or []:
        if not (
            str(stream.phase or "").lower() == "gas"
            or float(stream.gas_flow_sccm or 0.0) > 0
        ):
            continue
        text = " ".join(
            [str(stream.pump_role or ""), *(str(item) for item in stream.contents or [])]
        ).lower()
        if re.search(r"\b(o2|oxygen)\b|o₂", text):
            required.append("O2")
        elif re.search(r"\b(h2|hydrogen)\b|h₂", text):
            required.append("H2")
        elif "air" in text:
            required.append("O2")
    return required


def _unconfirmed_pressure_hardware_failures(
    proposal: FlowProposal,
    inventory: LabInventory | None,
    *,
    hard_constraints: Any,
) -> list[GateFinding]:
    if proposal.BPR_bar <= 0 or inventory is None:
        return []
    structured = hard_constraints if isinstance(hard_constraints, dict) else {}
    bpr_unconfirmed = bool(
        structured.get("BPR_settings_not_specified_in_source")
        or structured.get("bpr_settings_not_specified_in_source")
    )
    if inventory.BPR_available or inventory.pressure_controllers or not bpr_unconfirmed:
        return []
    return [
        GateFinding(
            finding_id="INVENTORY-BPR-UNCONFIRMED",
            category="inventory",
            message=(
                f"The candidate requires a {proposal.BPR_bar:g} bar BPR, but the "
                "inventory source does not confirm any available BPR or pressure setting."
            ),
            evidence={
                "proposal_BPR_bar": proposal.BPR_bar,
                "available_BPR_bar": inventory.BPR_available,
                "pressure_controller_ids": [
                    item.equipment_id for item in inventory.pressure_controllers
                ],
            },
        )
    ]


def _operation_prohibition_failures(
    inventory: LabInventory | None,
    *,
    batch_record: BatchRecord | None,
    chemistry_plan: ChemistryPlan | None,
) -> list[GateFinding]:
    if inventory is None or not inventory.reactors:
        return []
    reaction_text = " ".join(
        (
            str(getattr(batch_record, "reaction_description", "") or ""),
            str(getattr(batch_record, "raw_text", "") or ""),
            str(getattr(chemistry_plan, "reaction_name", "") or ""),
            str(getattr(chemistry_plan, "reaction_class", "") or ""),
        )
    ).lower()
    blocked_reactors = [
        reactor
        for reactor in inventory.reactors
        if _reactor_note_prohibits(reactor.notes, reaction_text)
    ]
    if blocked_reactors and len(blocked_reactors) == len(inventory.reactors):
        return [
            GateFinding(
                finding_id="INVENTORY-OPERATION-PROHIBITED",
                category="inventory",
                message=(
                    "Every listed reactor explicitly prohibits or is not approved "
                    "for the requested chemistry."
                ),
                evidence={
                    "reactor_notes": [reactor.notes for reactor in blocked_reactors]
                },
            )
        ]
    return []


def _multistage_failures(
    inventory: LabInventory | None,
    *,
    chemistry_plan: ChemistryPlan | None,
) -> list[GateFinding]:
    if inventory is None or chemistry_plan is None:
        return []
    stage_count = max(
        int(chemistry_plan.n_stages or 1),
        len(chemistry_plan.stages or []),
    )
    if stage_count <= 1 or not inventory.reactors:
        return []
    available_reactor_count = sum(
        int(reactor.quantity)
        for reactor in inventory.reactors
        if str(reactor.service_status or "").strip().lower()
        not in UNAVAILABLE_STATUS_MARKERS
    )
    if available_reactor_count >= stage_count:
        # Distinct reactors connected through a mixer form a valid multistage
        # topology. The topology allocator separately validates direct serial
        # coil trains and exact equipment assignments.
        return []
    compatible = [
        reactor
        for reactor in inventory.reactors
        if _reactor_supports_stages(reactor, stage_count)
    ]
    if compatible:
        return []
    return [
        GateFinding(
            finding_id="INVENTORY-MULTISTAGE-TOPOLOGY",
            category="inventory",
            message=(
                f"The chemistry requires {stage_count} sequential reaction stages, "
                "but no listed reactor train provides the required interstage topology."
            ),
            evidence={
                "required_stages": stage_count,
                "reactors": [reactor.model_dump() for reactor in inventory.reactors],
            },
        )
    ]


def _canonical_sections(raw_text: str) -> tuple[str, str]:
    objective_match = re.search(
        r"\bOBJECTIVE:\s*(.*?)(?=\n\s*HARD CONSTRAINTS:|\Z)",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    constraints_match = re.search(
        r"\bHARD CONSTRAINTS:\s*(.*?)(?=\n\s*AVAILABLE INVENTORY|\Z)",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return (
        objective_match.group(1).strip() if objective_match else "",
        constraints_match.group(1).strip() if constraints_match else "",
    )


def _flatten_constraint_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, Iterable):
        return "\n".join(str(item) for item in value)
    return str(value)


def _first_number(text: str, patterns: tuple[str, ...]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _first_range(
    text: str,
    patterns: tuple[str, ...],
) -> tuple[float, float] | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            values = sorted((float(match.group(1)), float(match.group(2))))
            return values[0], values[1]
    return None


def _structured_number(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    fallback: float | None,
) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return fallback


def _structured_range(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    fallback: tuple[float, float] | None,
) -> tuple[float, float] | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            numeric = float(value)
            return numeric, numeric
        if isinstance(value, (list, tuple)) and len(value) == 2:
            values = sorted((float(value[0]), float(value[1])))
            return values[0], values[1]
        if isinstance(value, dict):
            low = value.get("min")
            high = value.get("max")
            if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                values = sorted((float(low), float(high)))
                return values[0], values[1]
    return fallback


def _reactor_supports_temperature(
    reactor: ReactorSpec,
    target_low: float,
    target_high: float,
) -> bool:
    if reactor.allowed_temperatures_C:
        return any(
            target_low <= float(value) <= target_high
            for value in reactor.allowed_temperatures_C
        )
    minimum = (
        float(reactor.min_temperature_C)
        if reactor.min_temperature_C is not None
        else -math.inf
    )
    maximum = (
        float(reactor.max_temperature_C)
        if reactor.max_temperature_C is not None
        else math.inf
    )
    return minimum <= target_low and maximum >= target_high


def _service_available(item: GasHardwareSpec) -> bool:
    status = str(item.service_status or "available").strip().lower()
    return not any(marker in status for marker in UNAVAILABLE_STATUS_MARKERS)


def _is_gas_controller(item: GasHardwareSpec) -> bool:
    text = f"{item.name} {item.type}".lower()
    return any(marker in text for marker in ("mass-flow", "mass flow", "mfc", "controller"))


def _is_gas_liquid_mixer(item: GasHardwareSpec) -> bool:
    text = f"{item.name} {item.type}".lower()
    return "mixer" in text and ("gas" in text or "liquid" in text)


def _gas_matches(item: GasHardwareSpec, required: str) -> bool:
    text = f"{item.gas} {item.name} {item.notes}".lower()
    if required == "H2":
        return bool(re.search(r"\b(h2|hydrogen)\b", text))
    if required == "O2":
        return bool(re.search(r"\b(o2|oxygen|air)\b", text))
    return required.lower() in text


def _controller_supports(
    item: GasHardwareSpec,
    flow_sccm: float,
    pressure_bar: float,
) -> bool:
    if item.min_flow_sccm is not None and flow_sccm < item.min_flow_sccm - 1e-9:
        return False
    if item.max_flow_sccm is not None and flow_sccm > item.max_flow_sccm + 1e-9:
        return False
    if item.max_pressure_bar is not None and pressure_bar > item.max_pressure_bar + 1e-9:
        return False
    return True


def _reactor_note_prohibits(notes: str, reaction_text: str) -> bool:
    note = str(notes or "").lower()
    if not any(marker in note for marker in ("prohibit", "not approved", "unavailable")):
        return False
    keyword_groups = (
        ("nitration", ("nitration", "nitric acid", "nitrating")),
        ("hydrogen", ("hydrogen", "h2", "hydrogenolysis", "hydrogenation")),
        ("flammable-gas", ("hydrogen", "h2", "flammable gas")),
        ("photochemical", ("photochemical", "photoredox", "irradiat")),
    )
    for note_keyword, reaction_keywords in keyword_groups:
        if note_keyword in note and any(keyword in reaction_text for keyword in reaction_keywords):
            return True
    return "laboratory safety review prohibits" in note


def _reactor_supports_stages(reactor: ReactorSpec, stage_count: int) -> bool:
    text = f"{reactor.type} {reactor.configuration} {reactor.notes}".lower()
    if re.search(
        r"\b(no|without)\s+(?:an?\s+)?interstage\b|"
        r"\binterstage\b[\s\S]{0,35}\b(unavailable|not available|prohibited)\b",
        text,
    ):
        return False
    if len(reactor.component_volumes_mL) >= stage_count:
        return True
    if reactor.configuration.lower() in {"serial", "series"}:
        return True
    if any(marker in text for marker in ("two_stage", "two-stage", "reactor train")):
        return True
    return "interstage" in text and "addition port" in text


def _deduplicate_findings(findings: list[GateFinding]) -> list[GateFinding]:
    output = []
    seen = set()
    for finding in findings:
        if finding.finding_id in seen:
            continue
        seen.add(finding.finding_id)
        output.append(finding)
    return output


def _deduplicate_strings(values: list[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output
