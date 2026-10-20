import json
from pathlib import Path
from types import SimpleNamespace

import flora_translate.inventory_profiles as inventory_profiles

from flora_translate.inventory_profiles import (
    PROFILE_SCHEMA_VERSION,
    extract_inventory_profile,
    inventory_profile_from_payload,
    list_inventory_profiles,
    load_inventory_profile,
    save_inventory_profile,
    validate_inventory_profile,
)
from flora_translate.schemas import LAB_INVENTORY_SCHEMA_VERSION


def _legacy_inventory() -> dict:
    return {
        "pumps": [
            {
                "name": "Manual syringe pump",
                "type": "syringe",
                "max_pressure_bar": 8,
                "min_flow_rate_mL_min": 0.01,
                "max_flow_rate_mL_min": 5,
            }
        ],
        "tubing": [
            {
                "material": "FEP",
                "ID_mm": 1.0,
                "max_pressure_bar": 8,
                "max_temperature_C": 80,
                "transparent": True,
            }
        ],
        "BPR_available": [3, 6],
        "light_sources": [],
        "gas_hardware": [],
        "reactors": [
            {
                "name": "Serial coil",
                "system": "Manual Setup",
                "type": "coil",
                "material": "FEP",
                "volume_mL": 30,
                "ID_mm": 1.0,
                "component_volumes_mL": [20, 10],
            }
        ],
    }


def test_plain_no_degasser_statement_becomes_hard_capability():
    profile = extract_inventory_profile(
        "We do not have a degasser in the lab. Offline argon sparging is available.",
        name="No-degasser lab",
        use_llm=False,
    )

    capability = profile.equipment_capabilities["inline_degassing"]
    limits = profile.design_operating_limits()
    assert capability.available is False
    assert "offline argon sparging" in capability.allowed_alternatives
    assert limits["inline_degasser_available"] is False
    assert "inline membrane degasser" in limits["forbidden_equipment"]


def test_no_degasser_constraint_merges_with_existing_forbidden_equipment():
    profile = extract_inventory_profile(
        "We dont have a degasser. Offline nitrogen sparging is available.",
        name="No degasser lab",
        use_llm=False,
    )
    profile.operating_constraints["forbidden_equipment"] = ["glass reactor"]

    limits = profile.design_operating_limits()

    assert limits["inline_degasser_available"] is False
    assert limits["forbidden_equipment"][0] == "glass reactor"
    assert "inline vacuum degasser" in limits["forbidden_equipment"]
    assert limits["allowed_oxygen_exclusion_methods"] == [
        "offline nitrogen sparging"
    ]


def test_llm_list_form_sections_are_normalized_instead_of_crashing(monkeypatch):
    llm_payload = {
        "lab_inventory": {
            "pumps": [],
            "tubing": [],
            "BPR_available": 3,
            "light_sources": [],
            "gas_hardware": [],
            "reactors": {
                "name": "PDF coil",
                "system": "Manual",
                "type": "coil",
                "material": "FEP",
                "volume_mL": 10,
                "ID_mm": 1.0,
            },
        },
        "equipment_capabilities": [
            {
                "capability": "inline_degassing",
                "available": False,
                "service_status": "unavailable",
            }
        ],
        "operating_constraints": [
            {"max_pressure_bar": 8},
            "limit",
        ],
        "unresolved_fields": "pump minimum flow is not stated",
    }
    monkeypatch.setattr(
        inventory_profiles,
        "call_model_text",
        lambda **_: SimpleNamespace(text=json.dumps(llm_payload)),
    )

    profile = extract_inventory_profile(
        "Extracted PDF equipment text",
        name="PDF inventory",
        use_llm=True,
    )

    assert profile.lab_inventory.reactors[0].volume_mL == 10
    assert profile.lab_inventory.BPR_available == [3]
    assert profile.equipment_capabilities["inline_degassing"].available is False
    assert profile.operating_constraints["max_pressure_bar"] == 8
    assert profile.operating_constraints["additional_constraints"] == ["limit"]
    assert profile.validation.unresolved_fields == [
        "pump minimum flow is not stated"
    ]
    assert any(
        "list-form operating_constraints" in warning
        for warning in profile.validation.warnings
    )


def test_legacy_lab_inventory_import_remains_compatible():
    profile = inventory_profile_from_payload(
        _legacy_inventory(), default_name="Legacy inventory"
    )

    assert profile.lab_inventory.reactors[0].volume_mL == 30
    assert profile.lab_inventory.reactors[0].component_volumes_mL == [20, 10]
    assert profile.validation.valid


def test_legacy_inventory_migrates_to_strict_v3_with_stable_ids():
    profile = inventory_profile_from_payload(
        _legacy_inventory(), default_name="Migrated inventory"
    )
    serialized = profile.model_dump()
    reloaded = inventory_profile_from_payload(serialized)

    assert profile.schema_version == PROFILE_SCHEMA_VERSION
    assert profile.lab_inventory.schema_version == LAB_INVENTORY_SCHEMA_VERSION
    assert profile.lab_inventory.strict_assignment is True
    assert profile.lab_inventory.pressure_controllers[0].setpoints_bar == [3, 6]
    assert [item.equipment_id for item in profile.lab_inventory.all_equipment()] == [
        item.equipment_id for item in reloaded.lab_inventory.all_equipment()
    ]


def test_safety_accessories_are_preserved_as_first_class_inventory():
    payload = _legacy_inventory()
    payload["safety_accessories"] = [
        {
            "equipment_id": "h2_check_1",
            "name": "Hydrogen non-return valve",
            "type": "check valve",
            "capabilities": ["backflow_prevention"],
            "compatible_hazards": ["flammable_hydrogen"],
        }
    ]

    profile = inventory_profile_from_payload(payload, default_name="Safety inventory")

    accessory = profile.lab_inventory.safety_accessories[0]
    assert accessory.equipment_id == "h2_check_1"
    assert accessory.capabilities == ["backflow_prevention"]
    assert accessory in profile.lab_inventory.all_equipment()


def test_serial_component_mismatch_blocks_validation():
    payload = _legacy_inventory()
    payload["reactors"][0]["component_volumes_mL"] = [20, 20]
    profile = inventory_profile_from_payload(payload, default_name="Bad serial inventory")

    report = validate_inventory_profile(profile)
    assert not report.valid
    assert any("component volumes" in error for error in report.errors)


def test_profile_storage_creates_immutable_versions(tmp_path: Path):
    profile = inventory_profile_from_payload(
        _legacy_inventory(), default_name="Versioned inventory"
    )
    first, first_dir = save_inventory_profile(profile, root=tmp_path)
    second, second_dir = save_inventory_profile(profile, root=tmp_path)

    assert first.version == 1
    assert second.version == 2
    assert first_dir != second_dir
    assert json.loads((first_dir / "lab_inventory.json").read_text())["reactors"]
    assert list_inventory_profiles(tmp_path)[0]["version"] == 2
    assert load_inventory_profile(profile.profile_id, tmp_path).version == 2
