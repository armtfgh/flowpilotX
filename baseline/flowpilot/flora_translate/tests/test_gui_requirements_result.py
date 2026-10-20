from streamlit.testing.v1 import AppTest


def _requirements_result_app() -> None:
    from pages.flora_design_unified import _render_result

    topology = {
        "topology_id": "requirements",
        "unit_operations": [
            {
                "op_id": "feed",
                "op_type": "pump",
                "label": "Liquid feed",
                "parameters": {"requirement_only": True},
                "inventory_category": "pumps",
                "assignment_status": "capability_available",
            },
            {
                "op_id": "mixer",
                "op_type": "mixer",
                "label": "Stage 2 mixer",
                "parameters": {
                    "requirement_only": True,
                    "inventory_status": "confirmation_required",
                },
                "inventory_category": "mixers",
                "assignment_status": "confirmation_required",
            },
        ],
        "streams": [
            {
                "stream_id": "s1",
                "from_op": "feed",
                "to_op": "mixer",
                "stream_type": "process",
                "label": "",
                "composition": [],
                "flow_rate_mL_min": None,
            }
        ],
        "reactor_volume_mL": 0,
        "compilation_status": "preflight_needs_confirmation",
    }
    requirement = {
        "requirement_id": "REQ-MIXERS",
        "operation_id": "interstage_mixing",
        "category": "mixers",
        "required_count": 1,
        "available_count": 0,
        "status": "confirmation_required",
        "reason": "Inventory does not declare mixers.",
    }
    result = {
        "proposal": {},
        "confidence": "NOT_ASSESSED",
        "design_status": "inventory_confirmation_required",
        "recommended_disposition": "BLOCK",
        "reported_disposition": "BLOCK",
        "design_disposition": {
            "recommended_disposition": "BLOCK",
            "rationale": "Inventory confirmation is required.",
            "hard_failures": [
                {"finding_id": "REQ-MIXERS", "message": requirement["reason"]}
            ],
        },
        "final_validation": {
            "checks": {"topology_capability_preflight_complete": False}
        },
        "inventory_allocation": {
            "status": "confirmation_required",
            "strict_assignment": True,
            "checks": {"all_required_operations_assigned": False},
            "assignments": [],
            "unresolved_requirements": [requirement],
        },
        "inventory_preflight": {
            "confirmation_template": {"items": [requirement]}
        },
        "process_requirements_topology": topology,
        "diagnostic_topology": topology,
        "process_topology": {},
        "chemistry_plan": {},
    }
    _render_result(result, key_prefix="requirements_test")


def test_blocked_inventory_result_still_renders_requirements_diagram():
    app = AppTest.from_function(
        _requirements_result_app,
        default_timeout=20,
    ).run()

    assert not app.exception
    markdown = "\n".join(str(item.value) for item in app.markdown)
    warnings = "\n".join(str(item.value) for item in app.warning)
    assert "INVENTORY CONFIRMATION REQUIRED" in markdown
    assert "REQUIREMENTS TOPOLOGY - NOT EXECUTABLE" in warnings
    assert "data:image/svg+xml" in markdown
    assert any(
        button.label == "Download inventory confirmation template"
        for button in app.get("download_button")
    )
