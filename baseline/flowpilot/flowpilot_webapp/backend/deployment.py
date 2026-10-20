"""Identify the loaded backend and the frontend currently served from disk."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = [*ROOT.joinpath("flowpilot_webapp/backend").glob("*.py"),
             *(p for p in ROOT.joinpath("flora_translate").rglob("*.py") if "tests" not in p.relative_to(ROOT).parts),
             *ROOT.joinpath("flora_design/visualizer").glob("*.py")]
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


LOADED_BACKEND_BUILD = source_fingerprint()


def runtime_status(dist: Path, instance_id: str) -> dict:
    try:
        frontend_build = json.loads((dist / "build.json").read_text())["frontend_build_id"]
    except (OSError, ValueError, KeyError):
        frontend_build = None
    return {
        "instance_id": instance_id,
        "backend_build_id": LOADED_BACKEND_BUILD,
        "backend_stale": source_fingerprint() != LOADED_BACKEND_BUILD,
        "frontend_build_id": frontend_build,
        "capabilities": ["inventory_resolution_v1", "shared_job_ownership_v1", "workspace_resume_v1"],
    }
