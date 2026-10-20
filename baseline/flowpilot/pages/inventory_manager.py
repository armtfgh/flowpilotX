"""FlowPilot laboratory Inventory Manager page."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from flora_translate.inventory_profiles import (
    EquipmentCapability,
    InventoryProfile,
    empty_inventory_profile,
    extract_inventory_profile,
    extract_uploaded_documents,
    inventory_profile_from_payload,
    list_inventory_profiles,
    load_inventory_profile,
    save_inventory_profile,
    validate_inventory_profile,
)
from flora_translate.schemas import LabInventory


DRAFT_KEY = "inventory_manager_draft_v3"
SOURCE_FILES_KEY = "inventory_manager_source_files_v3"
RESET_WIDGETS_KEY = "inventory_manager_reset_widgets_v3"
WIDGET_EPOCH_KEY = "inventory_manager_widget_epoch_v3"


def render() -> None:
    _reset_profile_widgets_if_requested()
    st.title("Inventory Manager")
    st.caption("Build, validate, version, and export laboratory hardware profiles.")

    profile = _draft()
    tabs = st.tabs(
        ["Import", "Equipment", "Constraints", "Review & Save", "Profiles"]
    )
    with tabs[0]:
        _render_import(profile)
    profile = _draft()
    with tabs[1]:
        _render_equipment(profile)
    profile = _draft()
    with tabs[2]:
        _render_constraints(profile)
    profile = _draft()
    with tabs[3]:
        _render_review(profile)
    with tabs[4]:
        _render_profiles()


def _render_import(profile: InventoryProfile) -> None:
    st.subheader("Create or extract a profile")
    c1, c2 = st.columns(2)
    name = c1.text_input(
        "Profile name",
        value=profile.name,
        key=_profile_widget_key("inventory_profile_name"),
    )
    laboratory = c2.text_input(
        "Laboratory",
        value=profile.laboratory,
        key=_profile_widget_key("inventory_profile_laboratory"),
    )
    if name != profile.name or laboratory != profile.laboratory:
        old_name = profile.name
        profile.name = name
        profile.laboratory = laboratory
        profile.status = "draft"
        if (
            profile.version == 0
            and profile.profile_id == empty_inventory_profile(old_name).profile_id
        ):
            profile.profile_id = empty_inventory_profile(name).profile_id
        _set_draft(profile)
    uploads = st.file_uploader(
        "Inventory documents",
        type=["pdf", "pptx", "docx", "xlsx", "xlsm", "csv", "tsv", "txt", "md", "json"],
        accept_multiple_files=True,
        help="Upload one or more equipment lists, specification documents, or an existing inventory JSON.",
    )
    text = st.text_area(
        "Additional inventory description",
        height=150,
        placeholder=(
            "Example: We have two syringe pumps with a minimum flow of 10 uL/min. "
            "We do not have an inline degasser; offline argon sparging is available."
        ),
    )
    use_llm = st.checkbox(
        "Use LLM extraction",
        value=True,
        help="The LLM extracts fields; deterministic schema validation remains authoritative.",
    )

    c1, c2 = st.columns(2)
    if c1.button("Extract inventory", type="primary", use_container_width=True):
        files = [
            (upload.name, upload.getvalue(), upload.type or "") for upload in uploads or []
        ]
        direct_json = None
        if len(files) == 1 and files[0][0].lower().endswith(".json") and not text.strip():
            try:
                direct_json = json.loads(files[0][1].decode("utf-8"))
            except Exception:
                direct_json = None
        try:
            source_text, sources, errors = extract_uploaded_documents(files)
            if direct_json is not None:
                extracted = inventory_profile_from_payload(direct_json, default_name=name)
            else:
                combined = "\n\n".join(part for part in (source_text, text) if part.strip())
                extracted = extract_inventory_profile(
                    combined,
                    name=name,
                    laboratory=laboratory,
                    use_llm=use_llm,
                )
            extracted.provenance = sources
            extracted.validation = validate_inventory_profile(extracted)
            st.session_state[DRAFT_KEY] = extracted.model_dump()
            st.session_state[SOURCE_FILES_KEY] = files
            st.session_state["inventory_import_errors"] = errors
            st.session_state[RESET_WIDGETS_KEY] = True
            st.rerun()
        except Exception as exc:
            st.error(f"Inventory extraction failed: {exc}")

    if c2.button("Start blank profile", use_container_width=True):
        blank = empty_inventory_profile(name)
        blank.laboratory = laboratory
        st.session_state[DRAFT_KEY] = blank.model_dump()
        st.session_state[SOURCE_FILES_KEY] = []
        st.session_state[RESET_WIDGETS_KEY] = True
        st.rerun()

    for error in st.session_state.get("inventory_import_errors", []):
        st.warning(error)
    if profile.extraction_metadata:
        method = profile.extraction_metadata.get("method")
        st.caption(f"Latest extraction method: {method or 'import'}")
        if profile.extraction_metadata.get("llm_error"):
            st.warning(
                "LLM extraction failed; deterministic extraction was used. "
                + str(profile.extraction_metadata["llm_error"])
            )


def _render_equipment(profile: InventoryProfile) -> None:
    st.subheader("Available equipment")
    st.caption("Only equipment that can actually be used should appear in these tables.")
    inventory = profile.lab_inventory

    with st.form("inventory_equipment_form"):
        st.markdown("#### Core flow hardware")
        pumps = _editor("Pumps", inventory.pumps, _pump_template(), "inventory_pumps")
        tubing = _editor("Tubing", inventory.tubing, _tubing_template(), "inventory_tubing")
        reactors = _editor("Reactors", inventory.reactors, _reactor_template(), "inventory_reactors")
        lights = _editor(
            "Light sources", inventory.light_sources, _light_template(), "inventory_lights"
        )
        gas = _editor(
            "Gas hardware", inventory.gas_hardware, _gas_template(), "inventory_gas"
        )
        st.markdown("#### Process components")
        mixers = _editor("Mixers", inventory.mixers, _mixer_template(), "inventory_mixers")
        pressure_controllers = _editor(
            "Pressure controllers",
            inventory.pressure_controllers,
            _pressure_controller_template(),
            "inventory_pressure_controllers",
        )
        degassers = _editor(
            "Degassers", inventory.degassers, _degasser_template(), "inventory_degassers"
        )
        filters = _editor(
            "Filters", inventory.filters, _filter_template(), "inventory_filters"
        )
        separators = _editor(
            "Separators", inventory.separators, _separator_template(), "inventory_separators"
        )
        connectors = _editor(
            "Connectors", inventory.connectors, _connector_template(), "inventory_connectors"
        )
        temperature_controllers = _editor(
            "Temperature controllers",
            inventory.temperature_controllers,
            _temperature_controller_template(),
            "inventory_temperature_controllers",
        )
        collectors = _editor(
            "Collectors", inventory.collectors, _collector_template(), "inventory_collectors"
        )
        safety_accessories = _editor(
            "Safety accessories",
            inventory.safety_accessories,
            _safety_accessory_template(),
            "inventory_safety_accessories",
        )
        st.markdown("#### Declared assemblies")
        reactor_trains = _editor(
            "Reactor trains",
            inventory.reactor_trains,
            _reactor_train_template(),
            "inventory_reactor_trains",
        )
        bpr_text = st.text_input(
            "Legacy BPR settings (bar, comma-separated)",
            value=", ".join(f"{value:g}" for value in inventory.BPR_available),
            key=_profile_widget_key("inventory_bpr"),
        )
        submitted = st.form_submit_button("Apply equipment changes", type="primary")
    if submitted:
        try:
            payload = {
                "pumps": _clean_rows(pumps),
                "tubing": _clean_rows(tubing),
                "reactors": _clean_rows(reactors),
                "light_sources": _clean_rows(lights),
                "gas_hardware": _clean_rows(gas),
                "mixers": _clean_rows(mixers),
                "pressure_controllers": _clean_rows(pressure_controllers),
                "degassers": _clean_rows(degassers),
                "filters": _clean_rows(filters),
                "separators": _clean_rows(separators),
                "connectors": _clean_rows(connectors),
                "temperature_controllers": _clean_rows(temperature_controllers),
                "collectors": _clean_rows(collectors),
                "reactor_trains": _clean_rows(reactor_trains),
                "safety_accessories": _clean_rows(safety_accessories),
                "BPR_available": _comma_numbers(bpr_text),
                "schema_version": inventory.schema_version,
                "strict_assignment": True,
            }
            profile.lab_inventory = LabInventory.model_validate(payload)
            profile.status = "draft"
            profile.validation = validate_inventory_profile(profile)
            _set_draft(profile)
            st.success("Equipment changes applied.")
            st.rerun()
        except Exception as exc:
            st.error(f"Equipment data is incomplete or invalid: {exc}")


def _render_constraints(profile: InventoryProfile) -> None:
    st.subheader("Capabilities and hard constraints")
    inline = profile.equipment_capabilities.get("inline_degassing")
    current = "Not specified"
    if inline is not None:
        current = "Available" if inline.available else "Unavailable"
    default_alternatives = list(inline.allowed_alternatives) if inline else []
    alternative_options = list(
        dict.fromkeys(
            [
                "offline argon sparging",
                "offline nitrogen sparging",
                "pre-degassed solvent in a sealed inert reservoir",
            ]
            + default_alternatives
        )
    )

    with st.form("inventory_constraints_form"):
        degassing = st.selectbox(
            "Inline degassing capability",
            ["Not specified", "Available", "Unavailable"],
            index=["Not specified", "Available", "Unavailable"].index(current),
            key=_profile_widget_key("inventory_inline_degassing"),
        )
        alternatives = st.multiselect(
            "Allowed alternatives when inline degassing is unavailable",
            alternative_options,
            default=default_alternatives,
            key=_profile_widget_key("inventory_degassing_alternatives"),
        )
        forbidden = st.text_area(
            "Forbidden equipment or operations, one per line",
            value="\n".join(_as_string_list(
                profile.operating_constraints.get("forbidden_equipment", [])
            )),
            height=110,
            key=_profile_widget_key("inventory_forbidden"),
        )
        limits_json = st.text_area(
            "Additional operating constraints (JSON)",
            value=json.dumps(
                {
                    key: value
                    for key, value in profile.operating_constraints.items()
                    if key != "forbidden_equipment"
                },
                indent=2,
            ),
            height=180,
            key=_profile_widget_key("inventory_operating_limits"),
        )
        submitted = st.form_submit_button("Apply constraints", type="primary")
    if submitted:
        try:
            constraints = json.loads(limits_json or "{}")
            constraints["forbidden_equipment"] = _lines(forbidden)
            profile.operating_constraints = constraints
            if degassing == "Not specified":
                profile.equipment_capabilities.pop("inline_degassing", None)
                profile.operating_constraints.pop("inline_degasser_available", None)
            else:
                profile.equipment_capabilities["inline_degassing"] = EquipmentCapability(
                    available=degassing == "Available",
                    service_status="available" if degassing == "Available" else "unavailable",
                    allowed_alternatives=alternatives,
                )
                profile.operating_constraints["inline_degasser_available"] = (
                    degassing == "Available"
                )
            profile.status = "draft"
            profile.validation = validate_inventory_profile(profile)
            _set_draft(profile)
            st.success("Constraints applied.")
            st.rerun()
        except Exception as exc:
            st.error(f"Constraint JSON is invalid: {exc}")


def _render_review(profile: InventoryProfile) -> None:
    st.subheader("Review and freeze")
    profile.validation = validate_inventory_profile(profile)
    report = profile.validation
    if report.valid:
        st.success("Deterministic inventory validation passed.")
    else:
        st.error("Inventory validation has unresolved errors.")
    for error in report.errors:
        st.error(error)
    for item in report.unresolved_fields:
        st.warning(f"Unresolved: {item}")
    for warning in report.warnings:
        st.warning(warning)

    inventory = profile.lab_inventory
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reactors", len(inventory.reactors))
    c2.metric("Pumps", len(inventory.pumps))
    c3.metric("Tubing", len(inventory.tubing))
    c4.metric("Process components", _process_component_count(inventory))
    st.caption(
        f"Schema: {inventory.schema_version} | strict assignment: "
        f"{'enabled' if inventory.strict_assignment else 'legacy compatibility'} | "
        f"sources: {len(profile.provenance)}"
    )

    with st.expander("Profile JSON", expanded=False):
        st.json(profile.model_dump())
    st.download_button(
        "Download inventory profile JSON",
        json.dumps(profile.model_dump(), indent=2, default=str),
        file_name=f"{profile.profile_id}_inventory_profile.json",
        mime="application/json",
        use_container_width=True,
    )
    st.download_button(
        "Download core LabInventory JSON",
        json.dumps(profile.lab_inventory.model_dump(), indent=2),
        file_name=f"{profile.profile_id}_lab_inventory.json",
        mime="application/json",
        use_container_width=True,
    )

    confirmed = st.checkbox(
        "I reviewed the extracted equipment, units, availability, and constraints.",
        key=_profile_widget_key("inventory_review_confirmed"),
    )
    if st.button(
        "Save new profile version",
        type="primary",
        disabled=not (confirmed and report.valid),
        use_container_width=True,
    ):
        try:
            saved, directory = save_inventory_profile(
                profile,
                source_files=st.session_state.get(SOURCE_FILES_KEY, []),
            )
            _set_draft(saved)
            st.success(f"Saved {saved.name} v{saved.version} in {directory}")
        except Exception as exc:
            st.error(f"Unable to save profile: {exc}")


def _render_profiles() -> None:
    st.subheader("Saved profiles")
    profiles = list_inventory_profiles()
    if not profiles:
        st.info("No inventory profiles have been saved yet.")
        return
    st.dataframe(profiles, use_container_width=True, hide_index=True)
    labels = {
        f"{item['name']} - v{item['version']} ({item['status']})": item
        for item in profiles
    }
    selected_label = st.selectbox("Open profile", list(labels))
    selected = labels[selected_label]
    if st.button("Load as editable draft", use_container_width=True):
        profile = load_inventory_profile(selected["path"])
        _set_draft(profile)
        st.session_state[SOURCE_FILES_KEY] = []
        st.session_state[RESET_WIDGETS_KEY] = True
        st.rerun()


def _editor(label: str, values, template: dict[str, Any], key: str):
    st.markdown(f"**{label}**")
    rows = [_editable_row(value.model_dump()) for value in values] or [template]
    return st.data_editor(
        rows,
        num_rows="dynamic",
        use_container_width=True,
        key=_profile_widget_key(key),
    )


def _clean_rows(value) -> list[dict[str, Any]]:
    rows = value.to_dict("records") if hasattr(value, "to_dict") else list(value)
    output = []
    for row in rows:
        cleaned = {
            key: item
            for key, item in row.items()
            if item is not None and str(item).strip() not in {"", "nan", "None"}
        }
        if not cleaned:
            continue
        for key in (
            "compatible_systems", "compatible_materials", "allowed_temperatures_C",
            "component_volumes_mL", "setpoints_bar", "supported_ID_mm",
            "compatible_solvents", "supported_phases", "component_reactor_ids",
            "connector_ids",
            "capabilities", "compatible_hazards",
        ):
            if key in cleaned and isinstance(cleaned[key], str):
                cleaned[key] = _comma_values(cleaned[key])
        output.append(cleaned)
    return output


def _comma_values(text: str) -> list[Any]:
    values = []
    for item in str(text).split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError:
            values.append(stripped)
    return values


def _comma_numbers(text: str) -> list[float]:
    return [float(value) for value in str(text).split(",") if value.strip()]


def _lines(text: str) -> list[str]:
    return [line.strip(" -\t") for line in str(text).splitlines() if line.strip(" -\t")]


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _editable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: ", ".join(str(item) for item in value) if isinstance(value, list) else value
        for key, value in row.items()
    }


def _draft() -> InventoryProfile:
    value = st.session_state.get(DRAFT_KEY)
    if value is None:
        profile = empty_inventory_profile()
        st.session_state[DRAFT_KEY] = profile.model_dump()
        return profile
    return inventory_profile_from_payload(value)


def _set_draft(profile: InventoryProfile) -> None:
    st.session_state[DRAFT_KEY] = profile.model_dump()


def _reset_profile_widgets_if_requested() -> None:
    if not st.session_state.pop(RESET_WIDGETS_KEY, False):
        return
    st.session_state[WIDGET_EPOCH_KEY] = (
        int(st.session_state.get(WIDGET_EPOCH_KEY, 0)) + 1
    )


def _profile_widget_key(base: str) -> str:
    return f"{base}_{int(st.session_state.get(WIDGET_EPOCH_KEY, 0))}"


def _pump_template() -> dict[str, Any]:
    return {"equipment_id": "", "name": "", "quantity": 1, "service_status": "available", "type": "", "max_pressure_bar": None, "min_flow_rate_mL_min": None, "max_flow_rate_mL_min": None, "compatible_systems": ""}


def _tubing_template() -> dict[str, Any]:
    return {"equipment_id": "", "name": "", "quantity": 1, "service_status": "available", "material": "", "ID_mm": None, "max_pressure_bar": None, "max_temperature_C": None, "transparent": True, "length_m": None}


def _reactor_template() -> dict[str, Any]:
    return {"equipment_id": "", "name": "", "quantity": 1, "service_status": "available", "system": "", "type": "coil", "material": "FEP", "volume_mL": None, "ID_mm": None, "wavelength_nm": None, "allowed_temperatures_C": "", "min_concentration_M": None, "max_concentration_M": None, "min_pressure_bar": None, "max_pressure_bar": None, "configuration": "", "component_volumes_mL": "", "notes": ""}


def _light_template() -> dict[str, Any]:
    return {"equipment_id": "", "name": "", "quantity": 1, "service_status": "available", "wavelength_nm": None, "power_W": None, "compatible_reactor": "coil", "intensity_mW_cm2": None, "distance_cm": None}


def _gas_template() -> dict[str, Any]:
    return {"equipment_id": "", "name": "", "quantity": 1, "type": "", "gas": "", "min_flow_sccm": None, "max_flow_sccm": None, "max_pressure_bar": None, "service_status": "available", "notes": ""}


def _mixer_template() -> dict[str, Any]:
    return {"equipment_id": "", "name": "", "quantity": 1, "service_status": "available", "type": "T-mixer", "material": "", "max_inputs": 2, "min_total_flow_mL_min": None, "max_total_flow_mL_min": None, "max_pressure_bar": None, "supported_ID_mm": "", "compatible_systems": ""}


def _pressure_controller_template() -> dict[str, Any]:
    return {"equipment_id": "", "name": "", "quantity": 1, "service_status": "available", "type": "BPR", "setpoints_bar": "", "min_pressure_bar": None, "max_pressure_bar": None, "compatible_systems": ""}


def _degasser_template() -> dict[str, Any]:
    return {"equipment_id": "", "name": "", "quantity": 1, "service_status": "available", "type": "inline_degasser", "method": "", "min_flow_mL_min": None, "max_flow_mL_min": None, "max_pressure_bar": None, "compatible_solvents": ""}


def _filter_template() -> dict[str, Any]:
    return {"equipment_id": "", "name": "", "quantity": 1, "service_status": "available", "type": "inline_filter", "material": "", "pore_size_um": None, "max_flow_mL_min": None, "max_pressure_bar": None}


def _separator_template() -> dict[str, Any]:
    return {"equipment_id": "", "name": "", "quantity": 1, "service_status": "available", "type": "phase_separator", "supported_phases": "", "max_flow_mL_min": None, "max_pressure_bar": None}


def _connector_template() -> dict[str, Any]:
    return {"equipment_id": "", "name": "", "quantity": 1, "service_status": "available", "type": "union", "material": "", "supported_ID_mm": "", "max_pressure_bar": None}


def _temperature_controller_template() -> dict[str, Any]:
    return {"equipment_id": "", "name": "", "quantity": 1, "service_status": "available", "type": "heater_chiller", "min_temperature_C": None, "max_temperature_C": None, "allowed_temperatures_C": ""}


def _collector_template() -> dict[str, Any]:
    return {"equipment_id": "", "name": "", "quantity": 1, "service_status": "available", "type": "collection_vessel", "volume_mL": None, "max_pressure_bar": None}


def _reactor_train_template() -> dict[str, Any]:
    return {"equipment_id": "", "name": "", "quantity": 1, "service_status": "available", "configuration": "serial", "component_reactor_ids": "", "connector_ids": "", "total_volume_mL": None, "max_pressure_bar": None, "compatible_systems": ""}


def _safety_accessory_template() -> dict[str, Any]:
    return {"equipment_id": "", "name": "", "quantity": 1, "service_status": "available", "type": "safety_accessory", "capabilities": "", "max_pressure_bar": None, "compatible_hazards": "", "compatible_systems": "", "notes": ""}


def _process_component_count(inventory: LabInventory) -> int:
    categories = (
        inventory.mixers, inventory.pressure_controllers, inventory.degassers,
        inventory.filters, inventory.separators, inventory.connectors,
        inventory.temperature_controllers, inventory.collectors,
        inventory.reactor_trains, inventory.safety_accessories,
    )
    return sum(sum(item.quantity for item in category) for category in categories)
