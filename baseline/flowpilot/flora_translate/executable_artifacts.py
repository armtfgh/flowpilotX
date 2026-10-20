"""Compile and validate executable artifacts from the realized final design."""

from __future__ import annotations

from flora_translate.component_identity import (
    component_key, component_name, is_solvent_component, unique_components,
)

import hashlib
import json
import math
import re
from typing import Any

from flora_translate.diagram_artifacts import topology_sha256
from flora_translate.schemas import (
    CanonicalStreamComponent,
    ExecutableArtifactBundle,
    ExecutableChemistryIdentity,
    ExecutableProcessGraph,
    ExecutableProcedureStep,
    ExecutableSafetyContract,
    ExecutableSafetyControl,
    ExecutableValidationExperiment,
)
from flora_translate.residence_time_basis import (
    INLET_STP_BASIS,
    UNKNOWN_BASIS,
    normalize_residence_time_basis,
    gas_time_basis_consistent,
)
from flora_translate.topology_semantics import topology_semantic_issues


QUANTIFIED_ROLES = {
    "substrate", "reactant", "reagent", "oxidant", "reductant", "base",
    "catalyst", "photocatalyst", "co-catalyst", "additive", "quencher",
}


def compile_executable_artifacts(
    result: dict[str, Any],
    *,
    process_graph: ExecutableProcessGraph,
    parameters: dict[str, Any],
    streams: list[dict[str, Any]],
) -> ExecutableArtifactBundle:
    """Build safety, procedure, and validation only from the final design."""

    chemistry_identity = _chemistry_identity(result)
    components = _stream_components(result, streams)
    hazards = _hazards(result, parameters, components)
    requirements = _safety_requirements(hazards, parameters)
    procedure = _operating_procedure(
        result=result,
        process_graph=process_graph,
        parameters=parameters,
        streams=streams,
        components=components,
        hazards=hazards,
        requirements=requirements,
    )
    safety = _safety_contract(
        result=result,
        process_graph=process_graph,
        requirements=requirements,
        procedure=procedure,
        hazards=hazards,
    )
    experiments = _validation_experiments(
        process_graph=process_graph,
        parameters=parameters,
        hazards=hazards,
    )
    return ExecutableArtifactBundle(
        chemistry_identity=chemistry_identity,
        stream_components=components,
        safety=safety,
        operating_procedure=procedure,
        validation_experiments=experiments,
    )


def artifact_semantic_issues(
    result: dict[str, Any],
    *,
    process_graph: ExecutableProcessGraph,
    bundle: ExecutableArtifactBundle,
    parameters: dict[str, Any],
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    identity = bundle.chemistry_identity
    if (
        identity.protocol_transformation_family != "unknown"
        and identity.transformation_family != "unknown"
        and not _compatible_transformation_families(
            identity.protocol_transformation_family,
            identity.transformation_family,
        )
    ):
        issues.append(
            {
                "code": "FINAL-CHEMISTRY-IDENTITY-DRIFT",
                "message": (
                    "Frozen protocol transformation "
                    f"{identity.protocol_transformation_family!r} conflicts with final "
                    f"chemistry-plan transformation {identity.transformation_family!r}."
                ),
            }
        )
    if not identity.confirmed:
        issues.append(
            {
                "code": "FINAL-CHEMISTRY-IDENTITY-UNCONFIRMED",
                "message": (
                    "The protocol does not state a recognized transformation family "
                    "and no chemist-confirmed chemistry identity was supplied. Model "
                    "inference alone cannot authorize an executable design."
                ),
            }
        )

    if parameters.get("residence_time_basis_code") == UNKNOWN_BASIS:
        issues.append(
            {
                "code": "FINAL-RESIDENCE-TIME-BASIS-UNKNOWN",
                "message": (
                    "The authoritative residence-time basis is unknown. Select "
                    "liquid-only, inlet/STP apparent, or in-channel pressure-corrected."
                ),
            }
        )

    gas_streams = [
        stream
        for stream in (result.get("proposal") or {}).get("streams") or []
        if str(stream.get("phase") or "").lower() == "gas"
    ]
    if gas_streams and not gas_time_basis_consistent(result.get("proposal") or {}):
        issues.append(
            {
                "code": "FINAL-GAS-TIME-BASIS-INCONSISTENT",
                "message": (
                    "The declared reactor-time basis must close with its corresponding "
                    "flow and volume. STP gas metering does not require an STP time basis."
                ),
            }
        )

    missing_components = [
        f"{item.stream_label}:{item.name}"
        for item in bundle.stream_components
        if item.quantification_required and not item.quantified
    ]
    if missing_components:
        issues.append(
            {
                "code": "FINAL-COMPONENT-STOICHIOMETRY-INCOMPLETE",
                "message": (
                    "Reactive stream components lack concentration, equivalents, or "
                    "catalyst loading: " + ", ".join(missing_components)
                ),
            }
        )

    by_component: dict[tuple[str, str], list[CanonicalStreamComponent]] = {}
    for item in bundle.stream_components:
        by_component.setdefault((item.stream_label, component_key(item.name)), []).append(item)
    for (label, _), items in by_component.items():
        if len(items) > 1:
            issues.append({
                "code": "FINAL-COMPONENT-DESCRIPTIONS-CONFLICT",
                "message": (
                    f"Stream {label} contains multiple annotated entries for {items[0].name}. "
                    "Resolve to one component quantity before release."
                ),
            })

    issues.extend(topology_semantic_issues(process_graph.topology))
    regime = str(
        ((result.get("chemistry_plan") or {}).get("intensification_mandate") or {}).get(
            "required_mixing_regime"
        )
        or ""
    ).lower()
    has_gas_liquid_edge = any(
        edge.connection_type == "process" and edge.stream_type == "gas_liquid"
        for edge in process_graph.topology.streams
    )
    if (
        _intensification_is_binding(result)
        and any(token in regime for token in ("slug", "segmented", "taylor"))
        and not has_gas_liquid_edge
    ):
        issues.append(
            {
                "code": "FINAL-MIXING-REGIME-TOPOLOGY-CONFLICT",
                "message": (
                    f"Chemistry-plan mixing regime {regime!r} requires a gas-liquid "
                    "or otherwise segmented topology, but the final graph has none."
                ),
            }
        )

    if not bundle.safety.complete:
        issues.append(
            {
                "code": "FINAL-SAFETY-CONTROLS-INCOMPLETE",
                "message": "Missing required safety controls: "
                + ", ".join(bundle.safety.missing_control_ids),
            }
        )

    required_sections = {
        "preparation", "setup", "startup", "steady_state", "collection",
        "shutdown", "emergency", "waste",
    }
    present_sections = {item.section for item in bundle.operating_procedure}
    missing_sections = sorted(required_sections - present_sections)
    if missing_sections:
        issues.append(
            {
                "code": "FINAL-PROCEDURE-INCOMPLETE",
                "message": "Missing operating-procedure sections: "
                + ", ".join(missing_sections),
            }
        )

    inventory_snapshot = result.get("inventory_snapshot") or {}
    inventory_ids = _inventory_ids(inventory_snapshot)
    referenced_ids = {
        equipment_id
        for step in bundle.operating_procedure
        for equipment_id in step.equipment_ids
    } | {
        equipment_id
        for experiment in bundle.validation_experiments
        for equipment_id in experiment.equipment_ids
    } | {
        equipment_id
        for control in bundle.safety.controls
        for equipment_id in control.equipment_ids
    }
    unknown_ids = sorted(referenced_ids - inventory_ids)
    if inventory_snapshot and unknown_ids:
        issues.append(
            {
                "code": "FINAL-UNSUPPORTED-EXECUTABLE-OPERATION",
                "message": "Executable artifacts reference undeclared equipment: "
                + ", ".join(unknown_ids),
            }
        )

    manifest = result.get("diagram_render_manifest") or {}
    rendered_hash = str(manifest.get("topology_sha256") or "")
    if rendered_hash:
        expected_hash = topology_sha256(process_graph.topology)
        if rendered_hash != expected_hash:
            issues.append(
                {
                    "code": "FINAL-DIAGRAM-TOPOLOGY-HASH-MISMATCH",
                    "message": "Rendered diagram does not match the canonical process graph.",
                }
            )
    return _deduplicate(issues)


def _compatible_transformation_families(left: str, right: str) -> bool:
    if left == right:
        return True
    compatible = {
        frozenset({"amidation", "oxidative_amidation"}),
    }
    return frozenset({left, right}) in compatible


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _chemistry_identity(result: dict[str, Any]) -> ExecutableChemistryIdentity:
    batch = result.get("batch_record") or {}
    plan = result.get("chemistry_plan") or {}
    raw_protocol = str(batch.get("raw_text") or "")
    protocol = raw_protocol or str(batch.get("reaction_description") or "")
    intake = result.get("intake_package") or {}
    confirmation = dict(intake.get("chemistry_identity_confirmation") or {})
    chemist_description = str(confirmation.get("chemist_description") or "").strip()
    confirmed_family = transformation_family(
        " ".join(
            part
            for part in (
                str(
                    confirmation.get("transformation_family")
                    or confirmation.get("reaction_class")
                    or ""
                ),
                chemist_description,
            )
            if part
        )
    )
    protocol_family = transformation_family(raw_protocol)
    plan_text = " ".join(
        str(plan.get(key) or "")
        for key in ("reaction_name", "reaction_class", "mechanism_type")
    )
    plan_family = transformation_family(plan_text)
    multistep_protocol = bool(
        re.search(
            r"(?i)\b(two[- ]stage|multi[- ]step|sequential(?:ly)?|followed by)\b",
            raw_protocol,
        )
    )
    if protocol_family != "unknown" and (
        not multistep_protocol
        or plan_family == protocol_family
        or _compatible_transformation_families(protocol_family, plan_family)
    ):
        authority_source = "protocol_fact"
        authoritative_family = protocol_family
        confirmed = True
    elif confirmation.get("confirmed") and (
        confirmed_family != "unknown" or chemist_description
    ):
        authority_source = "chemist_confirmed"
        authoritative_family = confirmed_family
        confirmed = True
    elif not batch:
        authority_source = "legacy_unavailable"
        authoritative_family = "unknown"
        confirmed = True
    else:
        authority_source = "model_inference"
        authoritative_family = "unknown"
        confirmed = False
    bond_formed = str(plan.get("bond_formed") or "")
    if authoritative_family == "nitration" and bond_formed.strip().lower() in {
        "c-n",
        "c–n",
        "carbon-nitrogen",
    }:
        bond_formed = "aromatic C-NO2 bond"
    return ExecutableChemistryIdentity(
        reaction_name=str(plan.get("reaction_name") or ""),
        reaction_class=str(plan.get("reaction_class") or ""),
        transformation_family=plan_family,
        protocol_transformation_family=authoritative_family,
        bond_formed=bond_formed,
        bond_broken=str(plan.get("bond_broken") or ""),
        protocol_sha256=hashlib.sha256(protocol.encode("utf-8")).hexdigest(),
        chemistry_plan_sha256=canonical_sha256(plan),
        authority_source=authority_source,
        confirmed=confirmed,
    )


def transformation_family(text: str) -> str:
    lower = str(text or "").lower()
    families = (
        ("hydrogenolysis", (r"\bhydrogenolysis\b", r"\bdebenzylation\b", r"\bdeprotection\b.*\bh2\b")),
        ("hydrogenation", (r"\bhydrogenation\b",)),
        ("cuaac", (r"\bcuaac\b", r"azide[- ]alkyne cycloaddition", r"\bclick reaction\b")),
        ("nitration", (r"\bnitration\b", r"\bdinitration\b")),
        ("oxidative_amidation", (r"oxidative amidation",)),
        ("amidation", (r"\bamidation\b",)),
        (
            "oxidation",
            (
                r"\boxidation\b",
                r"\boxidative\b",
                r"\bdehydrogenation\b",
                r"\baromati[sz]ation\b",
                r"tetrahydroquinoline.*\bquinoline\b",
            ),
        ),
        ("reduction", (r"\breduction\b",)),
        ("coupling", (r"\bcoupling\b",)),
        ("substitution", (r"\bsubstitution\b",)),
    )
    for family, patterns in families:
        if any(re.search(pattern, lower) for pattern in patterns):
            return family
    return "unknown"


def _stream_components(
    result: dict[str, Any],
    streams: list[dict[str, Any]],
) -> list[CanonicalStreamComponent]:
    plan = result.get("chemistry_plan") or {}
    roles = {
        _normalize_name(item.get("name")): item
        for item in plan.get("reagents") or []
        if item.get("name")
    }
    output: list[CanonicalStreamComponent] = []
    for stream in streams:
        component_start = len(output)
        label = str(stream.get("stream_label") or "?")
        solvent = str(stream.get("solvent") or "")
        contents = unique_components(stream.get("contents") or [])
        reactive = []
        for item in contents:
            item_name = _component_name(item)
            role_entry = _matching_role(_normalize_name(item_name), roles)
            planned_role = str((role_entry or {}).get("role") or "unknown").lower()
            if not _is_explicit_solvent(item, item_name, solvent, planned_role):
                reactive.append(item)
        for source in contents:
            name = _component_name(source)
            normalized = _normalize_name(name)
            role_entry = _matching_role(normalized, roles)
            role = str((role_entry or {}).get("role") or "unknown").lower()
            if (role == "unknown" and str(stream.get("phase") or "").lower() == "gas"
                    and len(contents) == 1 and normalized == "air"
                    and stream.get("gas_active_species")
                    and 0 < float(stream.get("gas_reagent_mole_fraction") or 0) < 1):
                active_name = _normalize_name(stream["gas_active_species"])
                active_entry = _matching_role(active_name, roles)
                if active_entry and active_entry.get("role"):
                    role = f"gas mixture delivering {stream['gas_active_species']}; declared active-species role: {active_entry['role']}".lower()
            if _is_explicit_solvent(source, name, solvent, role):
                role = "solvent"
            concentration = _parse_concentration(source)
            equivalents, loading = _parse_equiv_or_loading(
                str((role_entry or {}).get("equiv_or_loading") or "") + " " + source
            )
            concentration = concentration if concentration is not None and concentration > 0 else None
            equivalents = equivalents if equivalents is not None and equivalents > 0 else None
            loading = loading if loading is not None and loading > 0 else None
            provenance = []
            if "screening assumption" in source.lower():
                provenance.append("screening_assumption_requires_confirmation")
            if concentration is not None:
                provenance.append("component_text")
            if equivalents is not None or loading is not None:
                provenance.append("chemistry_plan_or_component_text")
            if len(reactive) == 1 and role != "solvent":
                if concentration is None and stream.get("concentration_M"):
                    concentration = float(stream["concentration_M"])
                    provenance.append("single_reactive_component_stream_concentration")
                if equivalents is None and stream.get("molar_equiv"):
                    equivalents = float(stream["molar_equiv"])
                    provenance.append("single_reactive_component_stream_equivalents")
            if str(stream.get("phase") or "").lower() == "gas" and role != "solvent":
                equivalents = float(stream.get("molar_equiv") or equivalents or 0) or None
                if equivalents is not None:
                    provenance.append("gas_stream_equivalents")
            required = role != "solvent"
            quantified = not required or bool(
                loading is not None or equivalents is not None or concentration is not None
            )
            output.append(
                CanonicalStreamComponent(
                    stream_label=label,
                    name=name,
                    role=role,
                    source_text=source,
                    concentration_M=concentration,
                    molar_equiv=equivalents,
                    loading_mol_pct=loading,
                    quantification_required=required,
                    quantified=quantified,
                    provenance=provenance,
                )
            )
        _infer_component_concentrations(output[component_start:], stream)
    return output


def _infer_component_concentrations(
    components: list[CanonicalStreamComponent],
    stream: dict[str, Any],
) -> None:
    """Close a stock recipe from the stream's limiting-component basis.

    FlowProposal.concentration_M is defined on the limiting reactive component.
    Equivalents and catalyst loading can therefore be converted into component
    concentrations without asking an LLM to invent a preparation volume.
    """

    try:
        stream_concentration = float(stream.get("concentration_M") or 0.0)
        stream_equiv = float(stream.get("molar_equiv") or 1.0)
    except (TypeError, ValueError):
        stream_concentration = 0.0
        stream_equiv = 1.0
    if stream_concentration <= 0:
        return
    explicit_reference = next(
        (
            item
            for item in components
            if item.concentration_M is not None and item.molar_equiv is not None
        ),
        None,
    )
    if explicit_reference is not None:
        equivalent_basis = (
            float(explicit_reference.concentration_M)
            / float(explicit_reference.molar_equiv)
        )
    else:
        equivalent_basis = stream_concentration / max(stream_equiv, 1e-12)
    for component in components:
        if not component.quantification_required or component.concentration_M is not None:
            continue
        if component.loading_mol_pct is not None:
            component.concentration_M = (
                equivalent_basis * float(component.loading_mol_pct) / 100.0
            )
            component.provenance.append("derived_from_stream_basis_and_mol_pct")
        elif component.molar_equiv is not None:
            component.concentration_M = equivalent_basis * float(component.molar_equiv)
            component.provenance.append("derived_from_stream_basis_and_equivalents")
        component.quantified = bool(
            component.concentration_M is not None
            or component.molar_equiv is not None
            or component.loading_mol_pct is not None
        )


def _component_name(source: str) -> str:
    return component_name(source)


def _is_explicit_solvent(
    source: str,
    component_name: str,
    declared_solvent: str,
    planned_role: str,
) -> bool:
    """Prefer final component annotations over a fallible upstream role label."""

    return is_solvent_component(source, declared_solvent, planned_role)


def _intensification_is_binding(result: dict[str, Any]) -> bool:
    final_summary = (result.get("final_design") or {}).get("intensification") or {}
    if "applied_as_hard_constraint" in final_summary:
        return bool(final_summary.get("applied_as_hard_constraint"))
    try:
        from flora_translate.config import FLOW_TRANSLATION_POLICY

        return FLOW_TRANSLATION_POLICY == "intensify"
    except Exception:
        return False


def _hazards(
    result: dict[str, Any],
    parameters: dict[str, Any],
    components: list[CanonicalStreamComponent],
) -> list[str]:
    text = " ".join(
        [item.name for item in components]
        + [
            str((result.get("chemistry_plan") or {}).get("reaction_name") or ""),
            str((result.get("batch_record") or {}).get("reaction_description") or ""),
            str((result.get("batch_record") or {}).get("raw_text") or ""),
        ]
    ).lower()
    hazards = []
    if re.search(r"(?:^|\W)(?:h2|hydrogen)(?:$|\W)", text) and "peroxide" not in text:
        hazards.append("flammable_hydrogen")
    if "azide" in text:
        hazards.append("energetic_azide")
    if any(token in text for token in ("h2o2", "hydrogen peroxide", "tbhp", "peroxide")):
        hazards.append("peroxide_oxidizer")
    if any(token in text for token in ("nitric acid", "nitration", "dinitration")):
        hazards.append("strong_oxidizing_acid")
        hazards.append("nitration_runaway_decomposition")
    if (
        any(token in text for token in ("photoredox", "irradiat", " led "))
        or re.search(r"\b\d{3,4}(?:\.\d+)?\s*nm\b", text)
    ):
        hazards.append("high_intensity_light")
    has_reagent_gas = any(
        str(item.get("phase") or "").lower() == "gas"
        and any(
            token in " ".join(str(value) for value in item.get("contents") or []).lower()
            for token in ("air", "oxygen", "o2")
        )
        for item in (result.get("proposal") or {}).get("streams") or []
    )
    if has_reagent_gas:
        hazards.append("oxygen_organic_service")
    if float(parameters.get("BPR_bar") or 0) > 0:
        hazards.append("pressurized_operation")
    return list(dict.fromkeys(hazards))


def _safety_requirements(hazards: list[str], parameters: dict[str, Any]) -> list[dict[str, Any]]:
    requirements = [
        _requirement("GENERAL-PPE-CONTAINMENT", "general", "Use the laboratory-approved PPE and containment for every declared reagent."),
        _requirement("GENERAL-EMERGENCY-STOP", "emergency", "Define and rehearse the emergency feed-stop sequence."),
        _requirement("GENERAL-WASTE", "waste", "Segregate and label all process and flush waste by hazard class."),
    ]
    if "pressurized_operation" in hazards:
        requirements.extend(
            [
                _requirement("PRESSURE-LEAK-TEST", "pressure", "Leak- and pressure-test the assembled system before reagent introduction."),
                _requirement("PRESSURE-OVERPRESSURE-RESPONSE", "pressure", "Define blockage and overpressure response without opening a pressurized line."),
                _requirement("PRESSURE-DEPRESSURIZE", "pressure", "Cool and depressurize through the controlled outlet before opening the system."),
            ]
        )
    if "flammable_hydrogen" in hazards:
        requirements.extend(
            [
                _requirement("H2-INERT-PURGE", "hydrogen", "Purge and verify oxygen removal before admitting hydrogen.", capability="nitrogen_purge"),
                _requirement("H2-BACKFLOW-PREVENTION", "hydrogen", "Use declared non-return or check-valve protection on the hydrogen feed.", capability="backflow_prevention"),
                _requirement("H2-VENT-ROUTING", "hydrogen", "Route separator off-gas to a declared safe vent.", capability="vented_separator"),
                _requirement("H2-IGNITION-CONTROL", "hydrogen", "Use grounded ventilation and remove ignition sources."),
                _requirement("H2-SHUTDOWN", "hydrogen", "Stop hydrogen, purge residual gas, and depressurize in a defined sequence."),
            ]
        )
    if "energetic_azide" in hazards:
        requirements.extend(
            [
                _requirement("AZIDE-HOLDUP-LIMIT", "azide", "Minimize azide feed inventory and reactive holdup."),
                _requirement("AZIDE-BLOCKAGE-RESPONSE", "azide", "Stop feeds and cool before responding to blockage; never open under pressure."),
                _requirement("AZIDE-METAL-COMPATIBILITY", "azide", "Prevent unassessed metal-azide accumulation and stagnant zones."),
                _requirement("AZIDE-SHIELD-CONTAINMENT", "azide", "Operate behind declared shielding or enclosed containment.", capability="shield_or_containment"),
            ]
        )
    if "peroxide_oxidizer" in hazards:
        requirements.extend(
            [
                _requirement("PEROXIDE-SEGREGATION", "peroxide", "Keep peroxide and incompatible acid/base feeds segregated until the assigned mixer."),
                _requirement("PEROXIDE-TEMPERATURE-MONITOR", "peroxide", "Continuously monitor reactor temperature and stop feeds on excursion."),
                _requirement("PEROXIDE-QUENCH", "peroxide", "Define a compatible quench and endpoint before collection or work-up."),
                _requirement("PEROXIDE-CONTAINMENT", "peroxide", "Use declared shielding or compatible secondary containment.", capability="shield_or_containment"),
            ]
        )
    if "strong_oxidizing_acid" in hazards:
        requirements.extend(
            [
                _requirement("NITRATION-CONTAINMENT", "nitration", "Operate the nitration process in the declared shielded, ventilated containment.", capability="shield_or_containment"),
                _requirement("NITRATION-NOX-VENT", "nitration", "Route NOx-capable effluent and collection vents through the declared compatible ventilation path.", capability="hazard_ventilation"),
                _requirement("NITRATION-TEMPERATURE-TRIP", "nitration", "Define a maximum allowable temperature and stop both reactive feeds automatically or immediately on excursion."),
                _requirement("NITRATION-COOLED-QUENCH", "nitration", "Verify the declared cooled quench collector is connected and ready before starting nitric-acid feed.", capability="cooled_quench_collection"),
            ]
        )
    if "high_intensity_light" in hazards:
        requirements.append(
            _requirement(
                "PHOTO-LIGHT-ENCLOSURE",
                "photochemical",
                "Use the declared interlocked light enclosure when one is available; otherwise verify equivalent laboratory-approved light shielding before irradiation.",
                capability="shield_or_containment",
                hardware_required=False,
            )
        )
    if "oxygen_organic_service" in hazards:
        requirements.extend(
            [
                _requirement("OXYGEN-ORGANIC-IGNITION-CONTROL", "oxygen", "Exclude ignition sources and minimize oxygen-containing organic holdup."),
                _requirement(
                    "OXYGEN-ORGANIC-VENT",
                    "oxygen",
                    "Route oxygen-containing off-gas and solvent vapor to the declared compatible vent when one is available.",
                    capability="vented_separator",
                    hardware_required=False,
                ),
            ]
        )
    return requirements


def _operating_procedure(
    *,
    result: dict[str, Any],
    process_graph: ExecutableProcessGraph,
    parameters: dict[str, Any],
    streams: list[dict[str, Any]],
    components: list[CanonicalStreamComponent],
    hazards: list[str],
    requirements: list[dict[str, Any]],
) -> list[ExecutableProcedureStep]:
    steps: list[ExecutableProcedureStep] = []
    inventory = result.get("inventory_snapshot") or {}
    feed_by_label = {item.stream_label: item for item in process_graph.feeds}
    for stream in streams:
        label = str(stream.get("stream_label") or "?")
        feed = feed_by_label.get(label)
        stream_components = [item for item in components if item.stream_label == label]
        recipe = _stream_preparation_instruction(
            stream,
            stream_components,
            parameters=parameters,
        )
        steps.append(
            _step(
                f"PREP-{_slug(label)}",
                "preparation",
                f"Prepare Stream {label}: {recipe}",
                equipment_ids=[feed.equipment_id] if feed else [],
                parameter_bindings={
                    "stream_label": label,
                    "flow_rate_mL_min": stream.get("flow_rate_mL_min"),
                    "gas_flow_sccm": stream.get("gas_flow_sccm"),
                },
            )
        )
    process_equipment = _process_equipment_ids(process_graph)
    steps.append(_step("SETUP-VERIFY", "setup", "Assemble only the inventory-assigned process graph and verify every equipment ID and connection against the final diagram.", equipment_ids=process_equipment))
    steps.append(_step("SETUP-PRIME", "startup", "Prime each liquid line with its declared solvent using its assigned pump until the outlet is bubble-free; do not introduce reactive feeds during priming.", equipment_ids=[item.equipment_id for item in process_graph.feeds if item.phase != "gas"]))
    if "pressurized_operation" in hazards:
        steps.append(_step("PRESSURE-LEAK-TEST", "startup", f"Pressure-test with compatible inert fluid to the approved operating setpoint of {float(parameters.get('BPR_bar') or 0):g} bar and verify stable pressure before reagents.", equipment_ids=process_equipment, parameter_bindings={"BPR_bar": parameters.get("BPR_bar")}))
    if "flammable_hydrogen" in hazards:
        steps.append(_step("H2-INERT-PURGE", "startup", "Purge the gas path with the inventory-declared nitrogen source, verify oxygen removal by the laboratory-approved method, then isolate nitrogen before hydrogen admission.", equipment_ids=_capability_equipment_ids("nitrogen_purge", inventory, process_graph)))
        steps.append(_step("H2-BACKFLOW-PREVENTION", "setup", "Verify the declared non-return/check valve is installed in the hydrogen feed in the correct flow direction.", equipment_ids=_capability_equipment_ids("backflow_prevention", inventory, process_graph)))
        steps.append(_step("H2-VENT-ROUTING", "setup", "Verify the gas-liquid separator off-gas is connected to the declared safe vent before hydrogen admission.", equipment_ids=_capability_equipment_ids("vented_separator", inventory, process_graph)))
        steps.append(_step("H2-IGNITION-CONTROL", "setup", "Confirm grounded ventilation is active and ignition sources are excluded."))
    for requirement in requirements:
        control_id = requirement["control_id"]
        if any(item.step_id == control_id for item in steps):
            continue
        section = _control_section(control_id)
        steps.append(
            _step(
                control_id,
                section,
                requirement["description"],
                equipment_ids=_capability_equipment_ids(
                    requirement.get("capability"), inventory, process_graph
                ),
            )
        )
    feed_settings = {
        item.stream_label: {
            "equipment_id": item.equipment_id,
            "flow_rate_mL_min": item.flow_rate_mL_min,
            "gas_flow_sccm": item.gas_flow_sccm,
            "gas_flow_actual_mL_min": item.gas_flow_actual_mL_min,
        }
        for item in process_graph.feeds
    }
    has_gas_feed = any(item.phase == "gas" for item in process_graph.feeds)
    start_instruction = "Start liquid feeds at the canonical setpoints."
    if has_gas_feed:
        start_instruction += (
            " Admit gas only after liquid flow, temperature, pressure, purge, "
            "and vent checks are stable."
        )
    steps.append(_step("START-FEEDS", "startup", start_instruction, equipment_ids=[item.equipment_id for item in process_graph.feeds], parameter_bindings=feed_settings))
    tau = float(parameters.get("residence_time_min") or process_graph.total_residence_time_min)
    timing_reference = max(tau, sum(s.reactor_volume_mL for s in process_graph.stages) / process_graph.total_liquid_flow_mL_min)
    steps.append(_step("STEADY-STATE", "steady_state", f"Allow an initial commissioning interval of at least {3*timing_reference:.3g} min, based on three nominal liquid-volume turnovers. Confirm steady outlet composition and stable flow, pressure and temperature before collection; elapsed time alone does not establish steady state. Measure phase holdup or tracer response when gas or a packed bed is present.", equipment_ids=process_equipment, parameter_bindings={"residence_time_min": tau, "minimum_initial_stabilization_time_min": 3*timing_reference, "timing_reference_min": timing_reference, "steady_state_requires_measurement": True, "residence_time_basis": parameters.get("residence_time_basis_detail") or parameters.get("residence_time_basis")}))
    collection_ids = [
        str(item.inventory_item_id)
        for item in process_graph.topology.unit_operations
        if item.inventory_item_id
        and item.op_type.lower() in {"collector", "collection_vessel", "phase_separator", "separator"}
    ]
    steps.append(_step("COLLECT-PRODUCT", "collection", f"After measured steady state, collect a timed fraction over at least {timing_reference:.3g} min as an initial sampling interval; record exact start/end times and collected mass or volume. Extend collection if needed for the analytical method.", equipment_ids=collection_ids, parameter_bindings={"minimum_initial_collection_time_min": timing_reference}))
    steps.append(
        _step(
            "COLLECT-ANALYZE-WORKUP",
            "collection",
            _workup_instruction(result, hazards),
            equipment_ids=collection_ids,
        )
    )
    if "flammable_hydrogen" in hazards:
        steps.append(_step("H2-SHUTDOWN", "shutdown", "Stop hydrogen first, continue the declared inert purge to displace residual hydrogen, then stop liquid feeds and flush with compatible solvent."))
    steps.append(_step("NORMAL-SHUTDOWN", "shutdown", "Stop reactive feeds, flush each compatible line, cool the reactor to a safe handling temperature, and isolate energy sources."))
    if "pressurized_operation" in hazards:
        steps.append(_step("PRESSURE-DEPRESSURIZE", "shutdown", "Depressurize only through the controlled outlet while monitoring the pressure indicator; verify zero pressure before opening any fitting.", parameter_bindings={"BPR_bar": parameters.get("BPR_bar")}))
    steps.append(_step("GENERAL-EMERGENCY-STOP", "emergency", "On leak, blockage, pressure excursion, or temperature excursion: stop reactive feeds and energy input, maintain only the hazard-appropriate purge/cooling path, isolate sources, and follow the laboratory emergency plan."))
    steps.append(_step("GENERAL-WASTE", "waste", "Collect product, flush, quench, and vent-associated waste in separately labeled containers compatible with the declared hazards; do not combine incompatible waste streams."))
    return _deduplicate_steps(steps)


def _safety_contract(
    *,
    result: dict[str, Any],
    process_graph: ExecutableProcessGraph,
    requirements: list[dict[str, Any]],
    procedure: list[ExecutableProcedureStep],
    hazards: list[str],
) -> ExecutableSafetyContract:
    inventory = result.get("inventory_snapshot") or {}
    procedure_ids = {item.step_id for item in procedure}
    controls = []
    for requirement in requirements:
        equipment_ids = _capability_equipment_ids(
            requirement.get("capability"), inventory, process_graph
        )
        procedural = requirement["control_id"] in procedure_ids
        capability_required = bool(requirement.get("capability"))
        hardware_required = bool(requirement.get("hardware_required", True))
        satisfied = procedural and (
            not capability_required or not hardware_required or bool(equipment_ids)
        )
        evidence = [f"procedure:{requirement['control_id']}"] if procedural else []
        evidence.extend(f"inventory:{item}" for item in equipment_ids)
        controls.append(
            ExecutableSafetyControl(
                control_id=requirement["control_id"],
                category=requirement["category"],
                description=requirement["description"],
                satisfied=satisfied,
                evidence=evidence,
                equipment_ids=equipment_ids,
            )
        )
    missing = [item.control_id for item in controls if item.required and not item.satisfied]
    return ExecutableSafetyContract(
        hazards=hazards,
        controls=controls,
        complete=not missing,
        missing_control_ids=missing,
    )


def _validation_experiments(
    *,
    process_graph: ExecutableProcessGraph,
    parameters: dict[str, Any],
    hazards: list[str],
) -> list[ExecutableValidationExperiment]:
    equipment = _process_equipment_ids(process_graph)
    q = float(parameters.get("flow_rate_mL_min") or process_graph.total_liquid_flow_mL_min)
    tau = float(parameters.get("residence_time_min") or process_graph.total_residence_time_min)
    pressure = float(parameters.get("BPR_bar") or 0)
    temperature = parameters.get("temperature_C")
    output = [
        ExecutableValidationExperiment(
            experiment_id="VAL-HYDRAULIC-BASELINE",
            purpose="Verify hydraulic stability at the final operating point before chemistry collection.",
            procedure=(
                f"Run compatible blank solvent at Q={q:g} mL/min, T={temperature} C, "
                f"and BPR={pressure:g} bar for at least {max(tau, 10):g} min; record "
                "flow, temperature, pressure, leaks, and blockage indicators."
            ),
            equipment_ids=equipment,
            parameter_bindings={"flow_rate_mL_min": q, "temperature_C": temperature, "BPR_bar": pressure, "duration_min": max(tau, 10)},
        ),
        ExecutableValidationExperiment(
            experiment_id="VAL-RTD",
            purpose="Measure the residence-time distribution and phase holdup at the operating point.",
            procedure=(
                f"Introduce a compatible nonreactive tracer at Q={q:g} mL/min and "
                "measure its outlet response after correcting for external tubing delay. "
                "Report the measured mean, dispersion and liquid holdup separately from "
                f"the calculated {tau:g} min index ({parameters.get('residence_time_basis')}); "
                "do not require agreement with an STP turnover index."
            ),
            equipment_ids=equipment,
            parameter_bindings={"flow_rate_mL_min": q, "calculated_residence_time_min": tau, "residence_time_basis": parameters.get("residence_time_basis_detail") or parameters.get("residence_time_basis")},
        ),
        ExecutableValidationExperiment(
            experiment_id="VAL-FIRST-SCREEN",
            purpose="Generate the first wet-lab response at the exact canonical design point.",
            procedure="Execute the canonical operating procedure once and report conversion, isolated yield, selectivity, pressure trace, temperature trace, and any phase or blockage observations.",
            equipment_ids=equipment,
            parameter_bindings={key: parameters.get(key) for key in ("flow_rate_mL_min", "temperature_C", "BPR_bar", "residence_time_min", "reactor_volume_mL")},
        ),
    ]
    if "flammable_hydrogen" in hazards:
        output.append(
            ExecutableValidationExperiment(
                experiment_id="VAL-H2-LEAK-PURGE",
                purpose="Verify hydrogen leak integrity, purge effectiveness, backflow prevention, and vent routing before reagent service.",
                procedure="Perform the laboratory-approved inert leak and purge qualification using the exact final gas path; do not introduce hydrogen until every safety control is signed off.",
                equipment_ids=equipment,
                parameter_bindings={"BPR_bar": pressure},
            )
        )
    return output


def _requirement(
    control_id: str,
    category: str,
    description: str,
    capability: str = "",
    *,
    hardware_required: bool = True,
) -> dict[str, Any]:
    return {
        "control_id": control_id,
        "category": category,
        "description": description,
        "capability": capability,
        "hardware_required": hardware_required,
    }


def _step(step_id: str, section: str, instruction: str, *, equipment_ids: list[str] | None = None, parameter_bindings: dict[str, Any] | None = None) -> ExecutableProcedureStep:
    return ExecutableProcedureStep(step_id=step_id, section=section, instruction=instruction, equipment_ids=equipment_ids or [], parameter_bindings=parameter_bindings or {})


def _control_section(control_id: str) -> str:
    if "WASTE" in control_id:
        return "waste"
    if any(token in control_id for token in ("EMERGENCY", "OVERPRESSURE", "BLOCKAGE")):
        return "emergency"
    if any(token in control_id for token in ("DEPRESSURIZE", "SHUTDOWN")):
        return "shutdown"
    return "setup"


def _component_recipe(component: CanonicalStreamComponent) -> str:
    values = []
    if component.concentration_M is not None:
        values.append(f"{component.concentration_M:g} M")
    if component.molar_equiv is not None:
        values.append(f"{component.molar_equiv:g} equiv")
    if component.loading_mol_pct is not None:
        values.append(f"{component.loading_mol_pct:g} mol%")
    return component.name + (f" ({', '.join(values)})" if values else " (quantity unresolved)")


def _stream_preparation_instruction(
    stream: dict[str, Any],
    components: list[CanonicalStreamComponent],
    *,
    parameters: dict[str, Any],
) -> str:
    """Create an exact-volume stock recipe without inventing MW or density."""

    if str(stream.get("phase") or "").lower() == "gas":
        gas = ", ".join(item.name for item in components) or ", ".join(
            str(item) for item in stream.get("contents") or []
        )
        return (
            f"connect {gas or 'the declared gas'} to the assigned MFC at "
            f"{float(stream.get('gas_flow_sccm') or 0):g} mL/min at STP on the declared "
            "STP basis (273.15 K, 1.01325 bar; 22.414 L/mol); no liquid stock is prepared."
        )

    tau = float(parameters.get("residence_time_min") or 0.0)
    flow = float(stream.get("flow_rate_mL_min") or 0.0)
    needed = max(10.0, 4.5 * tau * flow)
    basis_mL = math.ceil(needed / 5.0) * 5.0
    quantities = []
    pending = []
    for component in components:
        if not component.quantification_required:
            continue
        if "screening_assumption_requires_confirmation" in component.provenance:
            pending.append(component.name)
            continue
        if component.concentration_M is None:
            quantities.append(f"{component.name}: quantity unresolved")
            continue
        mmol = float(component.concentration_M) * basis_mL
        quantities.append(
            f"{component.name}: {mmol:g} mmol ({float(component.concentration_M):g} M final)"
        )
    solvent = str(stream.get("solvent") or "the declared solvent")
    recipe = "; ".join(quantities) or "declared reactive contents"
    instruction = (
        f"on a {basis_mL:g} mL final-volume basis, charge {recipe}; add {solvent} "
        f"to {basis_mL:g} mL total. Convert mmol to weighed/dispensed amount using "
        "the verified reagent MW, assay, and density recorded at preparation."
    )
    if pending:
        instruction = (
            "DRAFT ONLY: confirm the identity, assay, and quantity of "
            + ", ".join(pending)
            + " before preparing this feed; the screening assumptions are not dispensing instructions. "
            + instruction
        )
    return instruction


def _workup_instruction(result: dict[str, Any], hazards: list[str]) -> str:
    raw = str((result.get("batch_record") or {}).get("raw_text") or "")
    source_steps = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", raw)
        if re.search(
            r"(?i)\b(?:was|were)\s+(?:then\s+)?(?:quenched|extracted|washed|separated|purified)|"
            r"\b(?:quenched|extracted|washed|purified)\s+(?:with|by|using)\b",
            sentence,
        )
    ]
    if source_steps:
        return (
            "Retain an aliquot for conversion/selectivity analysis, then apply the "
            "source-protocol downstream operation after confirming compatibility: "
            + " ".join(source_steps[:3])
        )
    if "peroxide_oxidizer" in hazards:
        return (
            "Collect cooled effluent in the assigned compatible vessel, quantify residual "
            "peroxide with the laboratory-approved assay, and do not concentrate or isolate "
            "material until a chemist-approved peroxide quench and endpoint are documented."
        )
    if "strong_oxidizing_acid" in hazards:
        return (
            "Collect into the assigned cooled quench vessel, keep aqueous acid and organic "
            "fractions segregated through the assigned separator, and analyze an aliquot "
            "before any concentration or isolation."
        )
    return (
        "Collect the crude outlet in the assigned vessel and retain a labeled aliquot "
        "for conversion, selectivity, and yield analysis. This first screen ends at "
        "crude collection; no isolation work-up is authorized unless the chemist adds "
        "a confirmed downstream procedure to the intake package."
    )


def _parse_concentration(text: str) -> float | None:
    mm = re.search(r"(?i)(\d+(?:\.\d+)?)\s*mM\b", text)
    if mm:
        return float(mm.group(1)) / 1000.0
    match = re.search(r"(?i)(\d+(?:\.\d+)?)\s*M\b", text)
    return float(match.group(1)) if match else None


def _parse_equiv_or_loading(text: str) -> tuple[float | None, float | None]:
    equiv = re.search(r"(?i)(\d+(?:\.\d+)?)\s*(?:equiv|eq)\b", text)
    loading = re.search(r"(?i)(\d+(?:\.\d+)?)\s*mol\s*%", text)
    return (
        float(equiv.group(1)) if equiv else None,
        float(loading.group(1)) if loading else None,
    )


def _matching_role(name: str, roles: dict[str, dict]) -> dict | None:
    if name in roles:
        return roles[name]
    # A formula and an explicit molecular name identify the same pure gas.
    # Recover only a role actually declared in the chemistry plan; do not
    # infer a reaction role from the gas name or equate air with pure oxygen.
    gas_aliases = (
        {"h2", "hydrogen", "molecular hydrogen", "dihydrogen"},
        {"o2", "oxygen", "molecular oxygen", "dioxygen"},
        {"n2", "nitrogen", "molecular nitrogen", "dinitrogen"},
        {"co2", "carbon dioxide"},
        {"co", "carbon monoxide"},
        {"d2", "deuterium", "molecular deuterium"},
    )
    for group in gas_aliases:
        if name in group:
            matches = [value for key, value in roles.items() if key in group]
            if len(matches) == 1:
                return matches[0]
            # Ambiguous declarations require review. Substring matching must
            # not mistake H2O2 for H2 or CO2 for CO.
            return None
    pure_gas_names = set().union(*gas_aliases)
    return next((value for key, value in roles.items()
                 if key not in pure_gas_names and (key in name or name in key)), None)


def _normalize_name(value: Any) -> str:
    text = component_name(str(value or "")).lower().translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", value.upper()).strip("-") or "STREAM"


def _inventory_ids(inventory: dict[str, Any]) -> set[str]:
    output = set()
    for value in inventory.values():
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict) and item.get("equipment_id"):
                output.add(str(item["equipment_id"]))
    return output


def _inventory_items(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for category, value in inventory.items():
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                output.append({**item, "_category": category})
    return output


def _capability_equipment_ids(capability: str, inventory: dict[str, Any], process_graph: ExecutableProcessGraph) -> list[str]:
    if not capability:
        return []
    items = _inventory_items(inventory)
    operation_text = " ".join(
        f"{item.label} {item.instrument_name} {item.op_type}"
        for item in process_graph.topology.unit_operations
    ).lower()
    patterns = {
        "nitrogen_purge": ("n2", "nitrogen", "nitrogen_purge"),
        "backflow_prevention": ("check valve", "non-return", "non return", "backflow", "backflow_prevention"),
        "vented_separator": ("vented", "safe vent", "safe_vent", "off-gas", "offgas"),
        "shield_or_containment": ("shield", "containment", "shield_or_containment", "enclosure", "blast"),
        "hazard_ventilation": ("vent", "ventilat", "nox", "off-gas", "offgas"),
        "cooled_quench_collection": ("cooled", "quench", "temperature-controlled collector"),
    }.get(capability, ())
    matched = []
    for item in items:
        text = " ".join(str(value) for value in item.values()).lower()
        if any(pattern in text for pattern in patterns):
            if item.get("equipment_id"):
                matched.append(str(item["equipment_id"]))
    if capability == "vented_separator" and any(pattern in operation_text for pattern in patterns):
        matched.extend(
            str(item.inventory_item_id)
            for item in process_graph.topology.unit_operations
            if item.inventory_item_id and "separator" in item.op_type
        )
    return sorted(set(matched))


def _process_equipment_ids(process_graph: ExecutableProcessGraph) -> list[str]:
    return sorted(
        {
            str(item.inventory_item_id)
            for item in process_graph.topology.unit_operations
            if item.inventory_item_id
        }
    )


def _deduplicate_steps(steps: list[ExecutableProcedureStep]) -> list[ExecutableProcedureStep]:
    output = []
    seen = set()
    for step in steps:
        if step.step_id not in seen:
            seen.add(step.step_id)
            output.append(step)
    return output


def _deduplicate(issues: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    seen = set()
    for issue in issues:
        key = (issue.get("code"), issue.get("message"))
        if key not in seen:
            seen.add(key)
            output.append(issue)
    return output
