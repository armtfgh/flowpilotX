from fastapi.testclient import TestClient

from flowpilot_webapp.backend.app import app, JOBS


def test_runtime_reports_backend_identity_and_capabilities():
    response = TestClient(app).get("/api/runtime")
    assert response.status_code == 200
    assert response.json()["instance_id"] == JOBS.instance_id
    assert response.json()["backend_build_id"]
    assert "inventory_resolution_v1" in response.json()["capabilities"]
    assert response.headers["cache-control"] == "no-store"


def test_stale_browser_build_is_rejected_before_a_job_is_created(monkeypatch):
    import importlib
    backend = importlib.import_module("flowpilot_webapp.backend.app")
    monkeypatch.setattr(backend, "runtime_status", lambda *args: {"backend_stale": False, "frontend_build_id": "new-build"})
    def fail(*args):
        raise AssertionError("A stale browser must not start a design")
    monkeypatch.setattr(JOBS, "submit", fail)
    response = TestClient(app).post("/api/design/jobs", json={}, headers={"X-FlowPilot-Client-Build": "old-build"})
    assert response.status_code == 409
    assert "outdated" in response.json()["detail"]


def test_legacy_browser_without_build_identity_is_rejected(monkeypatch):
    import importlib
    backend = importlib.import_module("flowpilot_webapp.backend.app")
    monkeypatch.setattr(backend, "runtime_status", lambda *args: {"backend_stale": False, "frontend_build_id": "new-build"})
    response = TestClient(app).post("/api/design/jobs", json={}, headers={"Sec-Fetch-Mode": "cors"})
    assert response.status_code == 409


def test_changed_backend_pauses_new_jobs_without_interrupting_active_jobs(monkeypatch):
    import importlib
    backend = importlib.import_module("flowpilot_webapp.backend.app")
    monkeypatch.setattr(backend, "runtime_status", lambda *args: {"backend_stale": True, "frontend_build_id": "current"})
    response = TestClient(app).post("/api/design/jobs", json={})
    assert response.status_code == 409
    assert "after active jobs finish" in response.json()["detail"]
    assert TestClient(app).get("/api/design/jobs").status_code == 200


def test_entry_html_is_not_cached():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
