"""Structural intake recovery; equipment in these tests is synthetic, not lab evidence."""

import pytest
from fastapi.testclient import TestClient

from flora_translate.intake_agent import IntakeAgent
from flora_translate.inventory_profiles import inventory_profile_from_payload, save_inventory_profile
from flora_translate.inventory_resolution import bind_inventory, confirm_inventory, review_inventory
from flora_translate.schemas import ChemistryPlan, ProcessStage, StreamLogic
from flowpilot_webapp.backend.app import app, JOBS


PROTOCOL = "Photochemical aerobic oxidation of A to B with oxygen gas (2 equiv) at 450 nm, 40 C for 2 hours."
BPR = {"equipment_id": "test_bpr", "name": "Test BPR", "setpoints_bar": [3], "max_pressure_bar": 8, "quantity": 1}


@pytest.fixture
def profile():
    return inventory_profile_from_payload({
        "profile_id": "resolution_test", "name": "Test laboratory", "laboratory": "test",
        "lab_inventory": {
            "schema_version": "flowpilot_lab_inventory_v3.0", "strict_assignment": True,
            "pumps": [{"equipment_id": "pump", "type": "syringe", "min_flow_rate_mL_min": .01, "max_flow_rate_mL_min": 2, "max_pressure_bar": 8}],
            "reactors": [{"equipment_id": "reactor", "type": "coil", "volume_mL": 10, "ID_mm": 1, "material": "PFA"}],
            "light_sources": [{"equipment_id": "led", "wavelength_nm": 450, "compatible_reactor": "coil"}],
            "gas_hardware": [{"equipment_id": "mfc", "type": "MFC", "gas": "O2"}],
        },
    })


@pytest.fixture
def package(profile):
    pkg = IntakeAgent().analyze(PROTOCOL, answers=[
        {"question_id": "Q-OBJ-001", "answer": "First flow screen"},
        {"question_id": "Q-CHEM-001", "answer": "Oxidation of A to B"},
        *[{"question_id": qid, "status": "unavailable"} for qid in ["Q-HIST-001", "Q-HYP-001"]],
    ], use_llm=False)
    return bind_inventory(pkg, profile)


def test_fixed_question_is_stable_and_no_llm_needed(package):
    reviews = [IntakeAgent().analyze(existing_package=package, use_llm=False).inventory_review for _ in range(5)]
    assert all(review == reviews[0] for review in reviews)
    assert [q["question_id"] for q in reviews[0]["questions"]] == ["INV-PRESSURE-001"]
    assert not package.ready_for_design
    assert not package.missing_question_ids


def test_confirmed_equipment_unblocks_without_mutating_original(profile, package):
    original = profile.model_dump()
    updated, changed = confirm_inventory(profile, category="pressure_controllers", status="available", equipment=BPR)
    assert changed
    assert profile.model_dump() == original
    ready = bind_inventory(package, updated)
    assert ready.ready_for_design
    assert not ready.inventory_review["questions"]
    assert ready.inventory_profile_snapshot["extraction_metadata"]["inventory_confirmation_log"][-1]["question_id"] == "INV-PRESSURE-001"
    duplicate, changed = confirm_inventory(updated, category="pressure_controllers", status="available", equipment=BPR)
    assert not changed
    assert duplicate == updated


def test_explicit_absence_remains_conceptual_without_repeated_question(profile, package):
    absent, _ = confirm_inventory(profile, category="pressure_controllers", status="unavailable", equipment={})
    blocked = bind_inventory(package, absent)
    assert not blocked.ready_for_design
    assert blocked.inventory_review["status"] == "conceptual_only"
    assert blocked.inventory_review["questions"] == []
    assert blocked.inventory_review["confirmations"][0]["question_id"] == "INV-PRESSURE-001"
    assert blocked.inventory_review["topology"]["unit_operations"]
    assert all("flow_rate_mL_min" not in op["parameters"] for op in blocked.inventory_review["topology"]["unit_operations"])
    ready_profile, _ = confirm_inventory(absent, category="pressure_controllers", status="available", equipment=BPR)
    assert bind_inventory(package, ready_profile).ready_for_design


@pytest.mark.parametrize("fields", [
    {"max_pressure_bar": 2}, {"setpoints_bar": []}, {"setpoints_bar": [float("nan")]},
    {"max_pressure_bar": float("inf")}, {"quantity": 0}, {"quantity": 1.5}, {"equipment_id": ""},
])
def test_invalid_equipment_is_rejected(profile, fields):
    with pytest.raises(ValueError):
        confirm_inventory(profile, category="pressure_controllers", status="available", equipment={**BPR, **fields})


def test_unavailable_passive_accessory_overrides_legacy_assumption(profile, package):
    absent, _ = confirm_inventory(profile, category="mixers", status="unavailable", equipment={})
    requirements = bind_inventory(package, absent).inventory_review["unresolved_requirements"]
    assert any(item["category"] == "mixers" and item["status"] == "unavailable" for item in requirements)


def test_profile_switch_preserves_stricter_chemist_limits(profile, package):
    package = IntakeAgent().analyze(existing_package=package, use_llm=False, answers=[{
        "question_id": "Q-CONSTR-001", "answer": {"max_pressure_bar": 2, "forbidden_equipment": ["inline degasser"]}, "source": "user"
    }])
    profile.operating_constraints = {"max_pressure_bar": 8}
    rebound = bind_inventory(package, profile)
    assert rebound.operating_limits["max_pressure_bar"] == 2
    assert "inline degasser" in rebound.operating_limits["forbidden_equipment"]


def test_api_resolution_saves_version_and_rejects_forged_readiness(profile, package, tmp_path, monkeypatch):
    import flora_translate.inventory_profiles as profiles
    monkeypatch.setattr(profiles, "save_inventory_profile", lambda p: save_inventory_profile(p, root=tmp_path))
    monkeypatch.setattr(profiles, "list_inventory_profiles", lambda: [])
    stored, path = save_inventory_profile(profile, root=tmp_path)
    before = (path / "inventory_profile.json").read_bytes()
    client = TestClient(app)
    forged = {**package.model_dump(), "ready_for_design": True}
    rejected = client.post("/api/design/jobs", json={"intake_package": forged})
    assert rejected.status_code == 400
    assert "pressure controllers" in rejected.json()["detail"]
    response = client.post("/api/inventory/resolve", json={
        "intake_package": package.model_dump(), "inventory_profile": stored.model_dump(),
        "category": "pressure_controllers", "status": "available", "equipment": BPR,
    })
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["package"]["ready_for_design"]
    assert data["profile"]["version"] == 2
    assert (path / "inventory_profile.json").read_bytes() == before
    assert (tmp_path / "resolution_test/v002/inventory_profile.json").exists()
    captured = []
    from types import SimpleNamespace
    monkeypatch.setattr(JOBS, "submit", lambda request: (captured.append(request) or SimpleNamespace(public=lambda **_: {"job_id": "test", "status": "queued"})))
    response = client.post("/api/design/jobs", json={"intake_package": data["package"]})
    assert response.status_code == 202
    assert captured[0]["intake_package"]["inventory_profile_snapshot"]["version"] == 2


def test_actual_plan_can_add_new_requirements_after_precheck(profile, package):
    updated, _ = confirm_inventory(profile, category="pressure_controllers", status="available", equipment=BPR)
    ready = bind_inventory(package, updated)
    assert ready.ready_for_design
    plan = ChemistryPlan(stages=[ProcessStage(requires_light=True, feed_streams=[
        StreamLogic(stream_label="A", reagents=["substrate"]),
        StreamLogic(stream_label="B", reagents=["other feed"]),
        StreamLogic(stream_label="G", phase="gas", reagents=["O2"]),
    ])])
    review = review_inventory(ready, plan)
    assert not review["ready"]
    assert [q["question_id"] for q in review["questions"]] == ["INV-PUMP-001"]


def test_gui_rechecks_core_answers_without_a_bound_inventory():
    response = TestClient(app).post("/api/design/jobs", json={"intake_package": {
        "raw_protocol": "Oxidation of A to B", "objective": "First screen", "ready_for_design": True,
    }})
    assert response.status_code == 400
    assert "Q-INV-001" in response.json()["detail"]


def test_streamlit_confirmation_preserves_answers_and_queues_new_version(profile, package, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import flora_translate.inventory_profiles as profiles

    saved, _ = save_inventory_profile(profile, root=tmp_path)
    monkeypatch.setattr(profiles, "save_inventory_profile", lambda p: save_inventory_profile(p, root=tmp_path))
    app_test = AppTest.from_string(
        "import streamlit as st\n"
        "from components.intake_wizard import _render_equipment_review, _load_package\n"
        "_render_equipment_review(_load_package(st.session_state['test_intake_package']), 'test_intake')\n", default_timeout=10,
    )
    app_test.session_state["test_intake_package"] = bind_inventory(package, saved).model_dump()
    app_test.session_state["test_intake_protocol"] = PROTOCOL
    app_test.session_state["test_intake_inventory_source"] = "Saved profile"
    app_test.run()
    assert not app_test.exception
    values = {"Equipment ID": "test_bpr", "Equipment name": "Test BPR", "Quantity available": "1",
              "Available setpoints (bar gauge, comma-separated)": "3", "Maximum pressure (bar)": "8"}
    for field in app_test.text_input:
        if field.label in values:
            field.set_value(values[field.label])
    next(button for button in app_test.button if button.label == "Save inventory confirmation").click().run()
    assert not app_test.exception
    final = app_test.session_state["test_intake_package"]
    assert final["ready_for_design"]
    assert final["inventory_profile_snapshot"]["version"] == 2
    assert final["raw_protocol"] == PROTOCOL
    assert app_test.session_state["test_intake_inventory_pending_profile"] == profile.profile_id
