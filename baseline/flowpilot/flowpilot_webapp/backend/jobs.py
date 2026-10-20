"""Threaded design jobs with progress capture and immutable result snapshots."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
import uuid
from typing import Any

from filelock import FileLock, Timeout


JOB_ROOT = Path("outputs/webapp_jobs")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


@dataclass
class DesignJob:
    job_id: str
    request: dict[str, Any]
    status: str = "queued"
    progress: float = 0.0
    phase: str = "Queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    autosave_dir: str | None = None
    owner_instance_id: str | None = None

    def public(self, *, include_result: bool = True) -> dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "status": self.status,
            "progress": round(self.progress, 4),
            "phase": self.phase,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "messages": self.messages[-80:],
            "error": self.error,
            "autosave_dir": self.autosave_dir,
            "owner_instance_id": self.owner_instance_id,
        }
        if self.status == "completed" and self.result:
            payload["phase"] = _result_phase(self.result)
        if include_result:
            payload["result"] = self.result
        return payload


def _result_phase(result: dict[str, Any]) -> str:
    if result.get("design_status") == "inventory_confirmation_required":
        return "Inventory confirmation required"
    if result.get("design_status") == "inventory_infeasible":
        return "Required equipment unavailable"
    if result.get("final_design", {}).get("status") == "executable":
        return "Design complete"
    return "Design requires review"


def _write_snapshot(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(_json_safe(value), indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class _JobLogHandler(logging.Handler):
    def __init__(self, manager: "JobManager", job_id: str, thread_id: int):
        super().__init__(level=logging.INFO)
        self.manager = manager
        self.job_id = job_id
        self.thread_id = thread_id

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self.thread_id:
            return
        message = self.format(record)
        self.manager.log(self.job_id, message, level=record.levelname.lower())
        progress_map = {
            "Step 1:": (0.08, "Parsing protocol"),
            "Step 2:": (0.16, "Chemistry analysis"),
            "Step 3:": (0.28, "Literature retrieval"),
            "Step 3b:": (0.36, "Engineering calculations"),
            "Step 4:": (0.46, "Flow proposal"),
            "Step 5:": (0.58, "Council review"),
            "Step 6:": (0.72, "Design realization"),
            "Step 7:": (0.84, "Final validation"),
        }
        for marker, (progress, phase) in progress_map.items():
            if marker in message:
                self.manager.update(self.job_id, progress=progress, phase=phase)
                break


class JobManager:
    """Serialize expensive designs while keeping the API responsive."""

    def __init__(self, max_workers: int = 1):
        self.instance_id = uuid.uuid4().hex[:12]
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="flowpilot-design",
        )
        self._jobs: dict[str, DesignJob] = {}
        self._lock = threading.RLock()
        self._leases: dict[str, FileLock] = {}

    def submit(self, request: dict[str, Any]) -> DesignJob:
        job = DesignJob(job_id=uuid.uuid4().hex[:12], request=deepcopy(request), owner_instance_id=self.instance_id)
        target = JOB_ROOT / job.job_id
        target.mkdir(parents=True, exist_ok=False)
        lease = FileLock(target / "owner.lock", thread_local=False)
        lease.acquire(timeout=0)
        with self._lock:
            self._jobs[job.job_id] = job
            self._leases[job.job_id] = lease
        try:
            _write_snapshot(target / "request.json", job.request)
            self._persist(job)
            self._executor.submit(self._run, job.job_id)
        except Exception:
            lease.release()
            with self._lock:
                self._leases.pop(job.job_id, None)
                self._jobs.pop(job.job_id, None)
            raise
        return job

    def get(self, job_id: str) -> DesignJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job
            # Other instances own their jobs. Never cache their transient state.
            return self._load(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            observed = dict(self._jobs)
            if JOB_ROOT.is_dir():
                for directory in JOB_ROOT.iterdir():
                    if directory.is_dir() and directory.name not in self._jobs:
                        job = self._load(directory.name)
                        if job is not None:
                            observed[job.job_id] = job
            jobs = sorted(observed.values(), key=lambda item: item.created_at, reverse=True)
            return [job.public(include_result=False) for job in jobs]

    def update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in values.items():
                setattr(job, key, value)
            self._persist(job)

    def log(self, job_id: str, message: str, *, level: str = "info") -> None:
        text = str(message).strip()
        if not text:
            return
        with self._lock:
            job = self._jobs[job_id]
            if job.messages and job.messages[-1].get("message") == text:
                return
            job.messages.append({"time": _now(), "level": level, "message": text})
            self._persist(job)

    def _run(self, job_id: str) -> None:
        self.update(
            job_id,
            status="running",
            progress=0.03,
            phase="Starting FlowPilot",
            started_at=_now(),
        )
        root_logger = logging.getLogger()
        previous_root_level = root_logger.level
        if previous_root_level > logging.INFO:
            root_logger.setLevel(logging.INFO)
        handler = _JobLogHandler(self, job_id, threading.get_ident())
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)
        try:
            with self._lock:
                request = deepcopy(self._jobs[job_id].request)
            runtime = request.get("runtime_options") or {}
            if runtime.get("upstream_model") or runtime.get("downstream_model"):
                self.log(
                    job_id,
                    "Model routing: upstream=%s; downstream/council=%s"
                    % (
                        runtime.get("upstream_model") or "default",
                        runtime.get("downstream_model") or "default",
                    ),
                )
            from flora_translate.gui_autosave import autosave_gui_result
            from flora_translate.main import translate

            result = translate(
                request.get("batch_input") or "",
                inventory_path=request.get("inventory_path") or "flora_translate/data/lab_inventory.json",
                intake_package=request.get("intake_package"),
                runtime_options=request.get("runtime_options"),
            )
            self.update(job_id, progress=0.93, phase="Saving artifacts")
            autosave_dir = autosave_gui_result(
                result,
                intake_package=request.get("intake_package"),
                source="webapp",
                user_input=request.get("batch_input") or "",
            )
            result["autosave_dir"] = str(autosave_dir)
            self.update(
                job_id,
                status="completed",
                progress=1.0,
                phase=_result_phase(result),
                result=_json_safe(result),
                autosave_dir=str(autosave_dir),
                finished_at=_now(),
            )
        except Exception as exc:
            self.log(job_id, f"Design failed: {exc}", level="error")
            root_logger.removeHandler(handler)
            logging.getLogger("flowpilot.webapp").exception("Design job failed")
            self.update(
                job_id,
                status="failed",
                phase="Design failed",
                error=str(exc),
                finished_at=_now(),
            )
        finally:
            if handler in root_logger.handlers:
                root_logger.removeHandler(handler)
            root_logger.setLevel(previous_root_level)
            with self._lock:
                lease = self._leases.pop(job_id, None)
                if lease is not None:
                    lease.release()

    def _load(self, job_id: str) -> DesignJob | None:
        if not job_id or Path(job_id).name != job_id:
            return None
        target = JOB_ROOT / job_id
        state_path = target / "job.json"
        if not state_path.is_file():
            return None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            result_path = target / "result.json"
            result = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.is_file()
                else None
            )
            status = str(state.get("status") or "failed")
            error = state.get("error")
            phase = str(state.get("phase") or "Recovered job")
            finished_at = state.get("finished_at")
            if status in {"queued", "running"}:
                if state.get("owner_instance_id"):
                    probe = FileLock(target / "owner.lock")
                    try:
                        with probe.acquire(timeout=0):
                            # Re-read after acquiring: the owner may have just completed.
                            latest = json.loads(state_path.read_text(encoding="utf-8"))
                            if latest.get("status") not in {"queued", "running"}:
                                return self._load(job_id)
                        status = "failed"
                        phase = "Interrupted: job owner stopped"
                        error = "The job owner stopped before completion. The saved request is retained."
                        finished_at = _now()
                    except Timeout:
                        pass
                else:
                    # Older servers do not hold a lease. Absence of ownership is
                    # insufficient evidence of a restart or failed design.
                    status = "unknown"
                    phase = "Legacy job: monitor the originating server"
                    error = None
            return DesignJob(
                job_id=job_id,
                request={},
                status=status,
                progress=float(state.get("progress") or 0),
                phase=phase,
                created_at=str(state.get("created_at") or _now()),
                started_at=state.get("started_at"),
                finished_at=finished_at,
                messages=list(state.get("messages") or []),
                result=result,
                error=error,
                autosave_dir=state.get("autosave_dir"),
                owner_instance_id=state.get("owner_instance_id"),
            )
        except Exception:
            logging.getLogger("flowpilot.webapp").exception("Could not recover job %s", job_id)
            return None

    def _persist(self, job: DesignJob) -> None:
        try:
            target = JOB_ROOT / job.job_id
            target.mkdir(parents=True, exist_ok=True)
            if job.result is not None:
                _write_snapshot(target / "result.json", job.result)
            _write_snapshot(target / "job.json", job.public(include_result=False))
        except Exception:
            logging.getLogger("flowpilot.webapp").exception("Could not persist job %s", job.job_id)


JOBS = JobManager(max_workers=1)
