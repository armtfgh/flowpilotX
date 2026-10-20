from streamlit.testing.v1 import AppTest


def _inventory_result_app(result: dict) -> None:
    from components.inventory_result import render_inventory_result

    render_inventory_result(result)


def test_inventory_result_component_renders_manifest_and_unresolved_items():
    result = {
        "inventory_allocation": {
            "status": "incomplete",
            "strict_assignment": True,
            "inventory_schema_version": "flowpilot_lab_inventory_v3.0",
            "inventory_sha256": "a" * 64,
            "assignments": [],
            "instrument_manifest": [
                {
                    "equipment_id": "pump_1",
                    "name": "Pump 1",
                    "category": "pumps",
                    "quantity_used": 1,
                    "roles": ["liquid feed"],
                }
            ],
            "unresolved_requirements": [
                {
                    "requirement_id": "INV-MIXERS",
                    "operation_id": "mixer_1",
                    "category": "mixers",
                    "reason": "No mixer available.",
                }
            ],
            "warnings": [],
        }
    }
    app = AppTest.from_function(
        _inventory_result_app,
        args=(result,),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert len(app.metric) == 4
    assert app.metric[0].value == "Incomplete"
    assert app.metric[3].value == "1"
    assert len(app.dataframe) == 2
