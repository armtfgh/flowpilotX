"""Streamlit rendering for deterministic inventory allocation results."""

from __future__ import annotations

import json

import streamlit as st


def render_inventory_result(result: dict) -> None:
    allocation = result.get("inventory_allocation") or {}
    manifest = result.get("instrument_manifest") or allocation.get("instrument_manifest") or []
    unresolved = allocation.get("unresolved_requirements") or []
    assumptions = allocation.get("assumed_standard_accessories") or []

    if not allocation:
        st.info("No deterministic inventory allocation report is attached to this result.")
        return

    status = str(allocation.get("status") or "unknown")
    strict = bool(allocation.get("strict_assignment"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Allocation", status.title())
    c2.metric("Instruments", len(manifest))
    c3.metric("Assignments", len(allocation.get("assignments") or []))
    c4.metric("Unresolved", len(unresolved))
    st.caption(
        f"Inventory schema: {allocation.get('inventory_schema_version', 'unknown')} | "
        f"mode: {'strict' if strict else 'legacy compatibility'} | "
        f"inventory hash: {str(allocation.get('inventory_sha256') or '')[:12]}"
    )

    if unresolved:
        confirmation_required = status == "confirmation_required"
        if confirmation_required:
            st.warning(
                "One or more required capability categories were not declared in "
                "the inventory. Confirm an item or mark it unavailable before design."
            )
        else:
            st.error("Required process equipment remains unresolved. The design is not executable.")
        st.dataframe(
            [
                {
                    "requirement": item.get("requirement_id"),
                    "operation": item.get("operation_id"),
                    "category": item.get("category"),
                    "status": item.get("status") or "unresolved",
                    "reason": item.get("reason"),
                }
                for item in unresolved
            ],
            use_container_width=True,
            hide_index=True,
        )
        confirmation_template = (
            (result.get("inventory_preflight") or {}).get("confirmation_template")
        )
        if confirmation_template:
            st.download_button(
                "Download inventory confirmation template",
                data=json.dumps(confirmation_template, indent=2),
                file_name="flowpilot_inventory_confirmation.json",
                mime="application/json",
                use_container_width=True,
            )
    elif assumptions:
        st.warning(
            "The numerical design is available, but passive accessories marked "
            "VERIFY must be confirmed before the experiment."
        )
        st.dataframe(
            [
                {
                    "operation": item.get("operation_id"),
                    "accessory": item.get("name"),
                    "status": "verify before run",
                }
                for item in assumptions
            ],
            use_container_width=True,
            hide_index=True,
        )
    elif strict:
        st.success("Every required physical operation was assigned to declared inventory.")

    st.markdown("#### Instruments used")
    if manifest:
        st.dataframe(
            [
                {
                    "equipment ID": item.get("equipment_id"),
                    "instrument": item.get("name"),
                    "category": item.get("category"),
                    "quantity used": item.get("quantity_used"),
                    "roles": ", ".join(item.get("roles") or []),
                }
                for item in manifest
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No physical instruments were assigned.")

    with st.expander("Capability checks and operating settings", expanded=False):
        assignments = allocation.get("assignments") or []
        if assignments:
            for assignment in assignments:
                st.markdown(
                    f"**{assignment.get('role') or assignment.get('operation_id')}**  "
                    f"`{', '.join(assignment.get('equipment_item_ids') or [])}`"
                )
                st.json(
                    {
                        "settings": assignment.get("settings") or {},
                        "capability_checks": assignment.get("capability_checks") or {},
                    }
                )
        else:
            st.info("No assignment details are available.")

    for warning in allocation.get("warnings") or []:
        st.warning(warning)
