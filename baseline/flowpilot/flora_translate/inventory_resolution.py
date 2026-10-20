"""Reproducible inventory questions and explicit, versioned confirmations."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any

from flora_translate.schemas import (
    ChemistryPlan, DesignInputPackage, LabInventory, ProcessStage, StreamLogic,
)
from flora_translate.topology_preflight import analyze_topology_requirements


# Fields are specifications supplied by the chemist, never generated equipment.
CAPABILITY_QUESTIONS = {
    "pumps": ("INV-PUMP-001", "Liquid pump", [
        ("type", "Pump type", "text"),
        ("min_flow_rate_mL_min", "Minimum flow (mL/min)", "number"),
        ("max_flow_rate_mL_min", "Maximum flow (mL/min)", "number"),
        ("max_pressure_bar", "Maximum pressure (bar)", "number"),
    ]),
    "reactors": ("INV-REACTOR-001", "Reactor", [
        ("type", "Reactor type", "text"), ("material", "Material", "text"),
        ("volume_mL", "Volume (mL)", "number"), ("ID_mm", "Internal diameter (mm)", "number"),
        ("min_temperature_C", "Minimum temperature (C)", "number"),
        ("max_temperature_C", "Maximum temperature (C)", "number"),
        ("max_pressure_bar", "Maximum pressure (bar)", "number"),
    ]),
    "gas_hardware": ("INV-GAS-001", "Gas mass-flow controller", [
        ("gas", "Calibrated gas", "text"),
        ("min_flow_sccm", "Minimum inlet/STP flow (sccm)", "number"),
        ("max_flow_sccm", "Maximum inlet/STP flow (sccm)", "number"),
        ("max_pressure_bar", "Maximum pressure (bar)", "number"),
    ]),
    "light_sources": ("INV-LIGHT-001", "Light source", [
        ("wavelength_nm", "Wavelength (nm)", "number"),
        ("power_W", "Optical power (W)", "number"),
        ("compatible_reactor", "Compatible reactor type", "text"),
    ]),
    "pressure_controllers": ("INV-PRESSURE-001", "Back-pressure regulator", [
        ("setpoints_bar", "Available setpoints (bar gauge, comma-separated)", "numbers"),
        ("max_pressure_bar", "Maximum pressure (bar)", "number"),
    ]),
    "mixers": ("INV-MIXER-001", "Mixing junction", [
        ("material", "Material", "text"), ("max_inputs", "Number of inlets", "integer"),
        ("max_pressure_bar", "Maximum pressure (bar)", "number"),
        ("supported_ID_mm", "Compatible tubing IDs (mm, comma-separated)", "numbers"),
    ]),
    "connectors_or_reactor_trains": ("INV-CONNECTOR-001", "Reactor connection", [
        ("material", "Material", "text"),
        ("supported_ID_mm", "Compatible tubing IDs (mm, comma-separated)", "numbers"),
        ("max_pressure_bar", "Maximum pressure (bar)", "number"),
    ]),
}


def intake_requirements_plan(package: DesignInputPackage) -> ChemistryPlan:
    """Conservative structural lower bound from frozen protocol requirements."""
    requirements = package.engineering_requirements
    stage_text = str((requirements.get("multistep") or {}).get("stage_definition") or package.raw_protocol)
    stage_ids = [int(value) for value in re.findall(r"\b(?:step|stage)\s*(\d+)\b", stage_text, re.I)]
    n_stages = max(stage_ids, default=2 if "multistep" in package.active_domains else 1)
    gas = requirements.get("gas") or {}
    gas_stage = int(gas.get("introduction_stage") or n_stages)
    n_stages = max(n_stages, gas_stage)
    if not 1 <= gas_stage <= 32 or n_stages > 32:
        raise ValueError("Inventory precheck supports 1 to 32 reaction stages")
    photo = "photochemistry" in package.active_domains or bool(requirements.get("photochemistry"))
    stages = []
    for number in range(1, n_stages + 1):
        feeds = [StreamLogic(stream_label="A", reagents=["Batch reaction mixture"], phase="liquid")] if number == 1 else []
        if gas and number == gas_stage:
            feeds.append(StreamLogic(stream_label="G", reagents=[gas.get("species") or "reagent gas"], phase="gas"))
        # Only one light source is certain before stage-specific chemistry analysis.
        stages.append(ProcessStage(stage_number=number, stage_name=f"Reaction stage {number}", requires_light=photo and number == 1, feed_streams=feeds))
    return ChemistryPlan(reaction_name="Protocol equipment requirements", n_stages=n_stages, stages=stages)


def review_inventory(
    package: DesignInputPackage,
    chemistry_plan: ChemistryPlan | None = None,
) -> dict[str, Any]:
    raw_inventory = package.inventory_constraints
    if not isinstance(raw_inventory, dict):
        return {"status": "not_bound", "ready": True, "questions": [], "requirements": []}
    inventory = LabInventory.model_validate(raw_inventory.get("lab_inventory", raw_inventory))
    if not inventory.strict_assignment:
        return {"status": "not_strict", "ready": True, "questions": [], "requirements": []}
    topology, report = analyze_topology_requirements(chemistry_plan or intake_requirements_plan(package), inventory)
    questions = []
    confirmations = []
    for item in report["unresolved_requirements"]:
        category = item["category"]
        if category not in CAPABILITY_QUESTIONS:
            continue
        question_id, title, fields = CAPABILITY_QUESTIONS[category]
        # Explicit absence remains visible as a requirement, but is not asked again.
        question = {
            **item, "question_id": question_id, "title": title,
            "question": f"Required capability: {title.lower()}. The selected inventory does not establish sufficient available equipment. Confirm its specifications or mark the capability unavailable.",
            "fields": [{"key": key, "label": label, "type": kind} for key, label, kind in fields],
        }
        if inventory.capability_status.get(category) == "unavailable":
            confirmations.append(question)
        else:
            questions.append(question)
    snapshot = package.inventory_profile_snapshot or {}
    fingerprint = hashlib.sha256(json.dumps({
        "protocol": package.raw_protocol, "inventory": inventory.model_dump(),
        "requirements": package.engineering_requirements,
        "chemistry_plan": chemistry_plan.model_dump() if chemistry_plan else None,
    }, sort_keys=True).encode()).hexdigest()
    return {
        **report,
        "schema_version": "flowpilot_inventory_review_v1.0",
        "ready": not report["unresolved_requirements"],
        "status": "ready" if not report["unresolved_requirements"] else "conceptual_only" if not questions else "needs_information",
        "assessment_basis": "chemistry_plan" if chemistry_plan else "protocol_structure_precheck",
        "profile_id": snapshot.get("profile_id"), "profile_name": snapshot.get("name"),
        "profile_version": snapshot.get("version"), "input_sha256": fingerprint,
        "questions": questions, "confirmations": confirmations, "topology": topology.model_dump(),
    }


def bind_inventory(package: DesignInputPackage | dict, profile) -> DesignInputPackage:
    """Bind one explicit profile while retaining separately entered chemist limits."""
    from flora_translate.intake_agent import IntakeAgent
    from flora_translate.schemas import IntakeAnswer

    package = DesignInputPackage.model_validate(package)
    package = package.model_copy(deep=True)
    package.inventory_profile_snapshot = profile.model_dump()
    limits = profile.design_operating_limits()
    user_limits = [answer for answer in package.answers
                   if answer.question_id == "Q-CONSTR-001" and answer.source != "inventory_profile"]
    if user_limits and user_limits[-1].status == "answered":
        original = user_limits[-1].answer
        # Retain free text as authority-labeled input; do not reinterpret it as equipment.
        if isinstance(original, str):
            try:
                original = json.loads(original)
            except ValueError:
                pass
        if isinstance(original, dict):
            for key, value in original.items():
                if key in limits and isinstance(value, (int, float)) and isinstance(limits[key], (int, float)):
                    if key.startswith("max_"):
                        limits[key] = min(value, limits[key])
                    elif key.startswith("min_"):
                        limits[key] = max(value, limits[key])
                    elif value != limits[key]:
                        raise ValueError(f"Chemist constraint conflicts with inventory: {key}")
                elif key in limits and isinstance(value, list) and isinstance(limits[key], list):
                    if key.startswith("allowed_"):
                        limits[key] = [item for item in limits[key] if item in value]
                    else:
                        limits[key] = list(dict.fromkeys([*limits[key], *value]))
                elif key in limits and value != limits[key]:
                    raise ValueError(f"Chemist constraint conflicts with inventory: {key}")
                else:
                    limits[key] = value
        elif original:
            limits["chemist_additional_constraints"] = original
    return IntakeAgent().analyze(
        package.raw_protocol, existing_package=package, use_llm=False,
        answers=[
            IntakeAnswer(question_id="Q-INV-001", answer=profile.lab_inventory.model_dump(), source="inventory_profile"),
            IntakeAnswer(question_id="Q-CONSTR-001", answer=limits, source="inventory_profile"),
        ],
    )


def confirm_inventory(profile, *, category: str, status: str, equipment: dict, note: str = ""):
    """Validate a human confirmation without mutating any stored profile."""
    if category not in CAPABILITY_QUESTIONS:
        raise ValueError("Unknown inventory capability")
    if status not in {"available", "unavailable"}:
        raise ValueError("Choose available or unavailable")
    from flora_translate.inventory_profiles import inventory_profile_from_payload

    payload = profile.model_dump()
    inventory = payload["lab_inventory"]
    target = "connectors" if category == "connectors_or_reactor_trains" else category
    if status == "unavailable":
        inventory.setdefault("capability_status", {})[category] = "unavailable"
        affected = ["connectors", "reactor_trains"] if target == "connectors" else [target]
        for group in affected:
            for item in inventory.get(group) or []:
                item["service_status"] = "unavailable"
        if target == "pressure_controllers":
            inventory["BPR_available"] = []
        confirmed = None
    else:
        confirmed = dict(equipment)
        for key in ("equipment_id", "name"):
            if not str(confirmed.get(key) or "").strip():
                raise ValueError(f"{key.replace('_', ' ')} is required")
        for key, label, kind in CAPABILITY_QUESTIONS[category][2]:
            value = confirmed.get(key)
            if value is None or value == "":
                raise ValueError(f"{label} is required")
            if kind == "numbers":
                values = value if isinstance(value, list) else str(value).split(",")
                confirmed[key] = [float(v) for v in values]
                if not confirmed[key] or any(not math.isfinite(v) or v <= 0 for v in confirmed[key]):
                    raise ValueError(f"{label} must contain positive finite values")
            elif kind in {"number", "integer"}:
                value = float(value)
                if not math.isfinite(value) or ("temperature" not in key and value < 0):
                    raise ValueError(f"{label} must be a finite valid value")
                if key in {"volume_mL", "ID_mm", "wavelength_nm", "power_W", "max_pressure_bar"} and value <= 0:
                    raise ValueError(f"{label} must be positive")
                if kind == "integer" and not value.is_integer():
                    raise ValueError(f"{label} must be a whole number")
                confirmed[key] = int(value) if kind == "integer" else value
        quantity = float(confirmed.get("quantity", 1))
        if not math.isfinite(quantity) or not quantity.is_integer() or quantity < 1:
            raise ValueError("Quantity must be a positive whole number")
        confirmed["quantity"] = int(quantity)
        for low, high in (("min_flow_sccm", "max_flow_sccm"), ("min_flow_rate_mL_min", "max_flow_rate_mL_min"), ("min_temperature_C", "max_temperature_C")):
            if low in confirmed and high in confirmed and confirmed[low] > confirmed[high]:
                raise ValueError(f"{low} must not exceed {high}")
        if category == "pressure_controllers":
            if max(confirmed["setpoints_bar"]) > confirmed["max_pressure_bar"]:
                raise ValueError("BPR setpoint exceeds its maximum pressure rating")
            confirmed["type"] = "BPR"
        if category == "gas_hardware":
            confirmed["type"] = "MFC"
        confirmed["service_status"] = "available"
        confirmed["notes"] = f"Chemist-confirmed inventory specification. {note}".strip()
        for other_category in CAPABILITY_QUESTIONS:
            other_target = "connectors" if other_category == "connectors_or_reactor_trains" else other_category
            if other_target != target and any(i.get("equipment_id") == confirmed["equipment_id"] for i in inventory.get(other_target) or []):
                raise ValueError("Equipment ID is already used in another category")
        inventory[target] = [item for item in inventory.get(target) or [] if item.get("equipment_id") != confirmed["equipment_id"]] + [confirmed]
        inventory.setdefault("capability_status", {})[category] = "available"
    log = payload.setdefault("extraction_metadata", {}).setdefault("inventory_confirmation_log", [])
    entry = {
        "question_id": CAPABILITY_QUESTIONS[category][0], "category": category,
        "status": status, "equipment": confirmed, "note": note,
        "source": "chemist_confirmation", "parent_profile_version": profile.version,
    }
    if log and all(log[-1].get(key) == value for key, value in entry.items() if key != "parent_profile_version"):
        return profile, False
    entry["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    log.append(entry)
    updated = inventory_profile_from_payload(payload)
    if updated.validation.errors:
        raise ValueError("; ".join(updated.validation.errors))
    return updated, True


def alternative_profiles(package: DesignInputPackage, chemistry_plan: ChemistryPlan | None = None) -> list[dict]:
    """Offer complete saved profiles; never silently combine their inventories."""
    from flora_translate.inventory_profiles import list_inventory_profiles, load_inventory_profile
    current = package.inventory_profile_snapshot or {}
    alternatives = []
    for summary in list_inventory_profiles():
        if summary["profile_id"] == current.get("profile_id") and summary["version"] == current.get("version"):
            continue
        if current.get("laboratory") and summary.get("laboratory") != current["laboratory"]:
            continue
        profile = load_inventory_profile(summary["path"])
        candidate = package.model_copy(deep=True)
        candidate.inventory_constraints = profile.lab_inventory.model_dump()
        candidate.inventory_profile_snapshot = profile.model_dump()
        review = review_inventory(candidate, chemistry_plan)
        if review["ready"]:
            alternatives.append({**summary, "basis": "Passes structural equipment precheck; operating settings are checked during design.", "BPR_available": profile.lab_inventory.BPR_available})
    return alternatives
