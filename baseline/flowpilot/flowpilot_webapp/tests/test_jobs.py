from __future__ import annotations

import json

from flowpilot_webapp.backend import jobs


def test_completed_job_is_recovered_from_disk(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "JOB_ROOT", tmp_path)
    target = tmp_path / "recover-me"
    target.mkdir()
    (target / "job.json").write_text(
        json.dumps({
            "job_id": "recover-me",
            "status": "completed",
            "progress": 1,
            "phase": "Design complete",
            "created_at": "2026-09-04T00:00:00+00:00",
            "messages": [],
            "autosave_dir": "outputs/gui_runs/example",
        }),
        encoding="utf-8",
    )
    (target / "result.json").write_text(
        json.dumps({"final_design": {"status": "executable"}}),
        encoding="utf-8",
    )

    manager = jobs.JobManager(max_workers=1)
    recovered = manager.get("recover-me")

    assert recovered is not None
    assert recovered.status == "completed"
    assert recovered.result["final_design"]["status"] == "executable"


def test_legacy_inflight_job_is_not_falsely_reported_as_failed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "JOB_ROOT", tmp_path)
    target = tmp_path / "interrupted"
    target.mkdir()
    (target / "job.json").write_text(
        json.dumps({
            "job_id": "interrupted",
            "status": "running",
            "progress": 0.4,
            "phase": "Flow proposal",
            "created_at": "2026-09-04T00:00:00+00:00",
            "messages": [],
        }),
        encoding="utf-8",
    )

    manager = jobs.JobManager(max_workers=1)
    recovered = manager.get("interrupted")

    assert recovered is not None
    assert recovered.status == "unknown"
    assert recovered.error is None


def test_second_server_observes_active_owner_and_refreshes_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_ROOT", tmp_path)
    owner, observer = jobs.JobManager(), jobs.JobManager()
    monkeypatch.setattr(owner._executor, "submit", lambda *args: None)
    job = owner.submit({"batch_input": "test protocol"})
    try:
        owner.update(job.job_id, status="running", phase="Council review")
        assert observer.get(job.job_id).status == "running"
        assert observer.list()[0]["phase"] == "Council review"
        assert json.loads((tmp_path / job.job_id / "request.json").read_text())["batch_input"] == "test protocol"
        result = {"design_status": "inventory_confirmation_required", "final_design": {"status": "blocked"}}
        owner.update(job.job_id, status="completed", phase="Design complete", result=result)
        observed = observer.get(job.job_id)
        assert observed.status == "completed"
        assert observed.public()["phase"] == "Inventory confirmation required"
        assert observed.result == result
        assert observer.list()[0]["status"] == "completed"
    finally:
        owner._leases.pop(job.job_id).release()


def test_stopped_known_owner_is_detected_without_overwriting_saved_state(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_ROOT", tmp_path)
    owner, observer = jobs.JobManager(), jobs.JobManager()
    monkeypatch.setattr(owner._executor, "submit", lambda *args: None)
    job = owner.submit({})
    owner.update(job.job_id, status="running")
    owner._leases.pop(job.job_id).release()
    before = (tmp_path / job.job_id / "job.json").read_bytes()
    assert observer.get(job.job_id).status == "failed"
    assert (tmp_path / job.job_id / "job.json").read_bytes() == before


def test_job_snapshots_are_atomically_replaced(tmp_path):
    path = tmp_path / "job.json"
    jobs._write_snapshot(path, {"value": 1})
    jobs._write_snapshot(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 2}
    assert not list(tmp_path.glob("*.tmp"))
