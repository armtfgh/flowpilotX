"""Inventory profile selector shared by FlowPilot design pages."""

from __future__ import annotations

import hashlib
import json

import streamlit as st

from flora_translate.inventory_profiles import (
    InventoryProfile,
    inventory_profile_from_payload,
    list_inventory_profiles,
    load_inventory_profile,
)


def render_inventory_selector(
    key_prefix: str = "flowpilot_inventory_selector",
) -> InventoryProfile | None:
    """Render saved-profile and JSON-import controls for one design run."""

    state_key = f"{key_prefix}_profile"
    profiles = list_inventory_profiles()
    profile_by_label = {
        _profile_label(item): item for item in profiles
    }
    pending = st.session_state.pop(f"{key_prefix}_pending_profile", None)
    if pending:
        label = next((label for label, item in profile_by_label.items()
                      if item["profile_id"] == pending), None)
        if label:
            st.session_state[f"{key_prefix}_source"] = "Saved profile"
            st.session_state[f"{key_prefix}_saved_profile"] = label

    st.markdown("#### Laboratory inventory")
    source = st.segmented_control(
        "Inventory source",
        ["Intake answer", "Saved profile", "Upload JSON"],
        default="Intake answer",
        key=f"{key_prefix}_source",
        help=(
            "A selected profile is frozen into this design run and supplies both "
            "available equipment and operating constraints."
        ),
    )

    if source == "Saved profile":
        if not profile_by_label:
            st.info("No saved profiles. Create one in Inventory Manager.")
            st.session_state.pop(state_key, None)
        else:
            label = st.selectbox(
                "Inventory profile",
                list(profile_by_label),
                key=f"{key_prefix}_saved_profile",
            )
            selected = profile_by_label[label]
            try:
                profile = load_inventory_profile(selected["path"])
                st.session_state[state_key] = profile.model_dump()
            except Exception as exc:
                st.error(f"Unable to load inventory profile: {exc}")
                st.session_state.pop(state_key, None)
    elif source == "Upload JSON":
        upload = st.file_uploader(
            "Inventory profile JSON",
            type=["json"],
            key=f"{key_prefix}_upload",
        )
        if upload is not None:
            data = upload.getvalue()
            digest = hashlib.sha256(data).hexdigest()
            if st.session_state.get(f"{state_key}_digest") != digest:
                try:
                    profile = inventory_profile_from_payload(
                        json.loads(data.decode("utf-8")),
                        default_name=upload.name.rsplit(".", 1)[0],
                    )
                    st.session_state[state_key] = profile.model_dump()
                    st.session_state[f"{state_key}_digest"] = digest
                except Exception as exc:
                    st.error(f"Invalid inventory JSON: {exc}")
                    st.session_state.pop(state_key, None)
    else:
        st.session_state.pop(state_key, None)

    profile = _load(st.session_state.get(state_key))
    if profile is not None:
        _render_summary(profile)
        if not profile.validation.valid:
            st.error(
                "This profile has unresolved validation errors and cannot be used "
                "for design. Correct it in Inventory Manager."
            )
            return None
    return profile


def _profile_label(item: dict) -> str:
    return (
        f"{item['name']} - v{item['version']} "
        f"({item['status']}, {item['reactor_count']} reactors)"
    )


def _load(value) -> InventoryProfile | None:
    if value is None:
        return None
    if isinstance(value, InventoryProfile):
        return value
    return inventory_profile_from_payload(value)


def _render_summary(profile: InventoryProfile) -> None:
    inventory = profile.lab_inventory
    inline = profile.equipment_capabilities.get("inline_degassing")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reactors", len(inventory.reactors))
    c2.metric("Pumps", len(inventory.pumps))
    c3.metric("Process hardware", _process_hardware_count(inventory))
    c4.metric(
        "Inline degassing",
        "Unavailable" if inline is not None and not inline.available else "Available/unspecified",
    )
    st.caption(
        f"{profile.name} | version {profile.version or 'unsaved'} | "
        f"schema: {inventory.schema_version} | "
        f"validation: {'passed' if profile.validation.valid else 'requires review'}"
    )
    with st.expander("Inventory equipment summary", expanded=False):
        rows = []
        for category in (
            "pumps", "tubing", "reactors", "light_sources", "gas_hardware",
            "mixers", "pressure_controllers", "degassers", "filters",
            "separators", "connectors", "temperature_controllers",
            "collectors", "reactor_trains", "safety_accessories",
        ):
            for item in getattr(inventory, category):
                rows.append(
                    {
                        "category": category,
                        "equipment_id": item.equipment_id,
                        "name": item.name or item.__class__.__name__,
                        "quantity": item.quantity,
                        "status": item.service_status,
                    }
                )
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No equipment entries are present in this profile.")


def _process_hardware_count(inventory) -> int:
    categories = (
        inventory.mixers, inventory.pressure_controllers, inventory.degassers,
        inventory.filters, inventory.separators, inventory.connectors,
        inventory.temperature_controllers, inventory.collectors,
        inventory.reactor_trains, inventory.safety_accessories,
    )
    return sum(sum(item.quantity for item in category) for category in categories)
