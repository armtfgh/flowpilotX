from copy import deepcopy

from fastapi.testclient import TestClient

from flowpilot_webapp.backend.app import app
from flora_translate.inventory_profiles import inventory_profile_from_payload, load_inventory_profile
from flora_translate.schemas import LabInventory


client = TestClient(app)


def sample():
    profile = client.get("/api/inventory/editor-schema").json()["empty_profile"]
    profile.update(name="Form test lab", profile_id="form_test_lab")
    profile["lab_inventory"]["pumps"] = [{
        "name": "Standalone pump", "type": "syringe", "quantity": 2,
        "min_flow_rate_mL_min": 0.01, "max_flow_rate_mL_min": 10,
        "flow_rate_increment_mL_min": 0.01, "max_pressure_bar": 10,
    }]
    profile["lab_inventory"]["reactors"] = [{
        "name": "Manual coil", "type": "coil", "material": "PFA",
        "volume_mL": 10, "ID_mm": 1, "max_pressure_bar": 8,
    }]
    profile["equipment_capabilities"] = {
        "inline_degassing": {"available": False, "allowed_alternatives": ["offline pre-degassing"]}
    }
    return profile


def test_schema_exposes_every_equipment_category_without_guessing_ratings():
    response = client.get("/api/inventory/editor-schema")
    assert response.status_code == 200
    meta = response.json()
    schema = LabInventory.model_json_schema()
    keys = {key for key, field in schema["properties"].items() if "$ref" in field.get("items", {})}
    assert {c["key"] for c in meta["categories"]} == keys
    assert len(keys) == 15
    pump = next(c for c in meta["categories"] if c["key"] == "pumps")["schema"]
    assert "min_flow_rate_mL_min" in pump["required"]
    assert "default" not in pump["properties"]["max_pressure_bar"]
    assert meta["empty_profile"]["lab_inventory"]["strict_assignment"] is True


def test_form_profile_roundtrip_keeps_constraints_and_ids():
    first = client.post("/api/inventory/import", json=sample()).json()
    assert first["validation"]["valid"]
    again = client.post("/api/inventory/import", json=first).json()
    assert again["lab_inventory"] == first["lab_inventory"]
    assert again["equipment_capabilities"] == first["equipment_capabilities"]
    assert inventory_profile_from_payload(again).design_operating_limits()["inline_degasser_available"] is False
    assert again["lab_inventory"]["pumps"][0]["equipment_id"]


def test_unknown_values_remain_unknown():
    payload = sample()
    payload["lab_inventory"]["tubing"] = [{"material": "PFA", "ID_mm": 1, "max_pressure_bar": 8, "max_temperature_C": 80, "transparent": None}]
    payload["lab_inventory"]["light_sources"] = [{"wavelength_nm": 450, "compatible_reactor": "coil", "power_W": None}]
    result = client.post("/api/inventory/import", json=payload).json()
    assert result["lab_inventory"]["tubing"][0]["transparent"] is None
    assert result["lab_inventory"]["light_sources"][0]["power_W"] is None


def test_impossible_pump_range_is_not_export_ready():
    payload = sample()
    payload["lab_inventory"]["pumps"][0]["min_flow_rate_mL_min"] = 20
    result = client.post("/api/inventory/import", json=payload).json()
    assert not result["validation"]["valid"]
    assert result["validation"]["errors"]


def test_bad_field_type_returns_actionable_validation_error():
    payload = sample()
    payload["lab_inventory"]["pumps"][0]["max_pressure_bar"] = {"unknown": True}
    response = client.post("/api/inventory/import", json=payload)
    assert response.status_code == 400
    assert "max_pressure_bar" in response.json()["detail"]


def test_khu_edit_keeps_compatibility_resources_provenance_and_constraints():
    original = load_inventory_profile("khu_laboratory_inventory_20260915").model_dump(mode="json")
    changed = deepcopy(original)
    changed["lab_inventory"]["pumps"][0]["notes"] += " Reviewed in editor."
    result = client.post("/api/inventory/import", json=changed).json()
    for key in original["lab_inventory"]:
        assert result["lab_inventory"][key] == changed["lab_inventory"][key]
    for key in ["provenance", "equipment_capabilities", "operating_constraints", "extraction_metadata"]:
        assert result[key] == original[key]


def test_save_versions_and_load_from_isolated_profile_store(tmp_path, monkeypatch):
    import flora_translate.inventory_profiles as profiles
    save, load = profiles.save_inventory_profile, profiles.load_inventory_profile
    monkeypatch.setattr(profiles, "save_inventory_profile", lambda profile: save(profile, root=tmp_path))
    monkeypatch.setattr(profiles, "load_inventory_profile", lambda reference: load(reference, root=tmp_path))
    profile = client.post("/api/inventory/import", json=sample()).json()
    first = client.post("/api/inventory/save", json={"profile": profile}).json()["profile"]
    second = client.post("/api/inventory/save", json={"profile": first}).json()["profile"]
    assert first["version"] == 1 and second["version"] == 2
    latest = client.get("/api/inventory/profiles/form_test_lab").json()
    assert latest["version"] == 2
    assert latest["lab_inventory"] == first["lab_inventory"]


def test_manual_profile_binds_constraints_to_intake():
    profile = client.post("/api/inventory/import", json=sample()).json()
    response = client.post("/api/intake/analyze", json={
        "raw_protocol": "A was oxidized to B using air at room temperature.",
        "use_llm": False, "inventory_profile": profile,
    })
    assert response.status_code == 200
    package = response.json()["package"]
    assert "offline pre-degassing" in str(package)
    assert "inline_degasser_available" in str(package)
