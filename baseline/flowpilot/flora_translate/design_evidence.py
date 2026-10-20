"""Preserve decision rationale without publishing superseded operating values."""
from copy import deepcopy
import math
from flora_translate.component_identity import component_key


NUMERIC_FIELDS = ("temperature_C", "BPR_bar", "concentration_M", "flow_rate_mL_min",
                  "reactor_volume_mL", "tubing_ID_mm", "residence_time_min", "wavelength_nm")
REASONING_FIELDS = NUMERIC_FIELDS + ("residence_time_basis", "residence_time_inlet_min",
                                   "residence_time_in_channel_min", "BPR_basis")
DEPENDENCIES = {
    "temperature_C": ("temperature_C", "BPR_bar", "BPR_basis"),
    "BPR_bar": ("BPR_bar", "temperature_C", "BPR_basis"),
    "concentration_M": ("concentration_M",),
    "reactor_volume_mL": ("reactor_volume_mL", "tubing_ID_mm"),
    "tubing_ID_mm": ("tubing_ID_mm", "reactor_volume_mL"),
    "flow_rate_mL_min": ("flow_rate_mL_min", "concentration_M", "reactor_volume_mL"),
}


def execution_state(proposal):
    """The physical settings a reviewer approved, excluding prose/history."""
    return {
        "parameters": {field: getattr(proposal, field) for field in NUMERIC_FIELDS},
        "time_basis": reasoning_snapshot(proposal.model_dump())["residence_time_basis"],
        "pressure_basis": proposal.BPR_basis,
        "reactor": proposal.inventory_selection.get("equipment_id"),
        "reactor_type": proposal.reactor_type,
        "tubing_material": proposal.tubing_material,
        "stationary_components": [
            {key: component_key(value) if key == "name" else deepcopy(value)
             for key, value in item.items() if key not in ("source_text", "reactor_type")}
            for item in proposal.inventory_constraints.get("stationary_components") or []],
        "stages": deepcopy(proposal.stage_parameters),
        "feeds": [{field: getattr(stream, field) for field in (
            "stream_label", "contents", "phase", "solvent", "concentration_M",
            "flow_rate_mL_min", "gas_flow_sccm", "gas_reagent_mole_fraction",
            "molar_equiv", "pump_equipment_id", "introduction_stage")}
            for stream in proposal.streams],
    }


def execution_states_match(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(execution_states_match(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(execution_states_match(a, b) for a, b in zip(left, right))
    return _matches(left, right)


def reasoning_snapshot(data):
    from flora_translate.residence_time_basis import normalize_residence_time_basis, UNKNOWN_BASIS
    defaults = {"BPR_basis": "gauge", "residence_time_basis": ""}
    snapshot = {field: data.get(field, defaults.get(field)) for field in REASONING_FIELDS}
    for field in NUMERIC_FIELDS + ("residence_time_inlet_min", "residence_time_in_channel_min"):
        if snapshot[field] is not None:
            try:
                snapshot[field] = float(snapshot[field])
            except (TypeError, ValueError):
                pass
    basis = normalize_residence_time_basis(snapshot["residence_time_basis"])
    if basis != UNKNOWN_BASIS:
        snapshot["residence_time_basis"] = basis
    return snapshot


def capture_reasoning_data(data):
    """Snapshot model reasoning before numeric normalization changes its context."""
    if data.get("decision_rationale"):
        return
    records = data.setdefault("decision_rationale", {})
    snapshot = reasoning_snapshot(data)
    for field, text in {**(data.get("reasoning_per_field") or {}), "chemistry_notes": data.get("chemistry_notes")}.items():
        if text:
            records[field] = {
                "text": text, "source": "model_proposed_scientific_rationale",
                "parameter_snapshot": deepcopy(snapshot),
                "dependencies": list(DEPENDENCIES.get(field, REASONING_FIELDS)),
                "status": "pending_realization",
            }


def capture_reasoning(proposal):
    # Repeated realization must not recapture its own generated summaries.
    data = proposal.model_dump()
    capture_reasoning_data(data)
    proposal.decision_rationale = data.get("decision_rationale") or {}


def _matches(left, right):
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-8)
    return left == right


def _without_saved_reasoning(value, text):
    """Remove only exact copies of the rationale this module publishes."""
    suffix = "\nScientific rationale: " + text
    while value.endswith(suffix):
        value = value[:-len(suffix)]
    if value.strip() in {text.strip(), ("Scientific rationale: " + text).strip()}:
        return ""
    return value


def reconcile_reasoning(proposal):
    """Keep matching rationale; retain changed-condition text as labeled history."""
    current = reasoning_snapshot(proposal.model_dump())
    for field, record in proposal.decision_rationale.items():
        changed = [name for name in record.get("dependencies", REASONING_FIELDS)
                   if not _matches(record.get("parameter_snapshot", {}).get(name), current.get(name))]
        record["changed_dependencies"] = changed
        record["status"] = "requires_review_after_parameter_change" if changed else "conditions_preserved"
        record["authority"] = "scientific rationale; not an independent validation of chemistry or safety"
        text = record.get("text", "")
        canonical = _without_saved_reasoning(proposal.reasoning_per_field.get(field, ""), text)
        if not changed:
            proposal.reasoning_per_field[field] = (canonical + "\nScientific rationale: " + text).strip()
        elif canonical:
            proposal.reasoning_per_field[field] = canonical
        else:
            proposal.reasoning_per_field.pop(field, None)
        if field == "chemistry_notes":
            notes = _without_saved_reasoning(proposal.chemistry_notes, text)
            proposal.chemistry_notes = (notes + "\nScientific rationale: " + text).strip() if not changed else notes


def refresh_derived_metadata(proposal):
    """Move stale intermediate quantities out of the current parameter record."""
    constraints = proposal.inventory_constraints
    old = constraints.pop("flow_recalculation", None)
    if old:
        history = constraints.setdefault("diagnostic_history", [])
        item = {"source": "pre_realization_flow_recalculation", "authoritative": False, "values": old}
        if item not in history:
            history.append(item)
    constraints["current_flow_summary"] = {
        "authority": "final_realization", "liquid_flow_mL_min": proposal.flow_rate_mL_min,
        "reactor_volume_mL": proposal.reactor_volume_mL,
        "residence_time_min": proposal.residence_time_min,
        "residence_time_basis": proposal.residence_time_basis,
        "gas_streams": [{"stream_label": s.stream_label, "gas_flow_sccm": s.gas_flow_sccm,
                         "gas_flow_actual_mL_min": s.gas_flow_actual_mL_min,
                         "molar_equiv": s.molar_equiv, "molar_equiv_basis": s.molar_equiv_basis}
                        for s in proposal.streams if s.phase == "gas"],
    }
    if "packed" in proposal.reactor_type.lower() or any(s.phase == "gas" for s in proposal.streams):
        unresolved = {}
        for key in ("gas_holdup", "liquid_holdup_volume_mL", "two_phase_delta_P_bar",
                    "two_phase_pressure_drop_bar", "kLa_s_inv", "kLa"):
            if key in proposal.multiphase_metrics:
                unresolved[key] = proposal.multiphase_metrics.pop(key)
        if unresolved:
            constraints.setdefault("diagnostic_history", []).append({
                "source": "unvalidated_transport_proxy", "authoritative": False, "values": unresolved})
        proposal.multiphase_metrics["transport_status"] = "Phase holdup, transfer and packed-bed pressure drop require measurement; no measured values supplied."
