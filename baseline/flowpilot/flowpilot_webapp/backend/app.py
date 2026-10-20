"""FastAPI adapter for the validated FlowPilot Python pipeline."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import json
import os
import sys
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .jobs import JOBS
from .deployment import runtime_status


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass


def _frontend_dist() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "flowpilot_webapp" / "frontend" / "dist"
    return ROOT / "flowpilot_webapp" / "frontend" / "dist"


DIST = _frontend_dist()


class IntakePayload(BaseModel):
    raw_protocol: str = ""
    existing_package: dict[str, Any] | None = None
    answers: list[dict[str, Any]] = Field(default_factory=list)
    use_llm: bool = True
    inventory_profile: dict[str, Any] | None = None
    upstream_model_id: str | None = None


class DesignPayload(BaseModel):
    batch_input: str = ""
    intake_package: dict[str, Any]
    inventory_profile: dict[str, Any] | None = None
    inventory_path: str = "flora_translate/data/lab_inventory.json"
    runtime_options: dict[str, Any] | None = None
    upstream_model_id: str | None = None
    downstream_model_id: str | None = None


class InventorySavePayload(BaseModel):
    profile: dict[str, Any]


class InventoryReviewPayload(BaseModel):
    intake_package: dict[str, Any]
    inventory_profile: dict[str, Any] | None = None
    chemistry_plan: dict[str, Any] | None = None


class InventoryResolutionPayload(InventoryReviewPayload):
    category: str
    status: str
    equipment: dict[str, Any] = Field(default_factory=dict)
    note: str = ""


class RefinementPayload(BaseModel):
    job_id: str | None = None
    current_result: dict[str, Any] | None = None
    experiments: list[dict[str, Any]] = Field(default_factory=list)
    target_yield_pct: float = 80.0
    target_conversion_pct: float = 90.0
    target_selectivity_pct: float = 85.0


app = FastAPI(title="FlowPilot API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def deployment_guard(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/api/design/jobs":
        runtime = runtime_status(DIST, JOBS.instance_id)
        client_build = request.headers.get("x-flowpilot-client-build")
        browser_request = request.headers.get("sec-fetch-mode") in {"cors", "same-origin"}
        if runtime["backend_stale"]:
            return JSONResponse(status_code=409, content={"detail": "The backend code changed after this server started. Restart this server after active jobs finish; no new design was started."})
        if (client_build or browser_request) and client_build != runtime["frontend_build_id"]:
            return JSONResponse(status_code=409, content={"detail": "This FlowPilot page is outdated. Reload the workspace before submitting; no design was started."})
    response = await call_next(request)
    if not request.url.path.startswith("/assets/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/runtime")
def deployment_status() -> dict[str, Any]:
    return runtime_status(DIST, JOBS.instance_id)


@app.get("/api/health")
def health() -> dict[str, Any]:
    from flora_translate import config as cfg
    from flora_translate.inventory_profiles import list_inventory_profiles
    from flora_translate.model_catalog import (
        available_default_route_ids,
        model_route_statuses,
        model_routes,
    )

    corpus_records = 0
    try:
        import chromadb

        client = chromadb.PersistentClient(path="flora_translate/data/chroma_db")
        corpus_records = client.get_or_create_collection("flora_records").count()
    except Exception:
        pass
    routes = model_routes()
    route_defaults = available_default_route_ids(model_route_statuses())
    upstream = routes[route_defaults["upstream"]]
    downstream = routes[route_defaults["downstream"]]
    return {
        "status": "ok",
        "app": "FlowPilot",
        "version": "2.0.0",
        "models": {
            "intake": cfg.MODEL_INPUT_PARSER,
            "chemistry": upstream.model,
            "translation": downstream.model,
            "council_provider": downstream.provider,
            "council_model": downstream.model,
        },
        "inventory_profiles": len(list_inventory_profiles()),
        "corpus_records": corpus_records,
    }


@app.get("/api/models")
def available_models() -> dict[str, Any]:
    from flora_translate.model_catalog import (
        available_default_route_ids,
        model_route_statuses,
        model_routes,
    )

    routes = model_routes()
    statuses = model_route_statuses()
    return {
        "models": [
            {**route.public(), **statuses[route_id]}
            for route_id, route in routes.items()
        ],
        "defaults": available_default_route_ids(statuses),
    }


def _apply_inventory(package: dict[str, Any], profile_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not profile_payload:
        return package
    from flora_translate.inventory_profiles import inventory_profile_from_payload
    from flora_translate.inventory_resolution import bind_inventory

    profile = inventory_profile_from_payload(profile_payload)
    if not profile.validation.valid:
        raise ValueError("Inventory profile has validation errors: " + "; ".join(profile.validation.errors))
    return bind_inventory(package, profile).model_dump()


def _inventory_response(package: dict, chemistry_plan: dict | None = None) -> dict:
    from flora_translate.intake_agent import IntakeAgent
    from flora_translate.inventory_resolution import alternative_profiles, review_inventory
    from flora_translate.schemas import ChemistryPlan, DesignInputPackage

    pkg = DesignInputPackage.model_validate(package)
    plan = ChemistryPlan.model_validate(chemistry_plan) if chemistry_plan else None
    pkg.inventory_review = review_inventory(pkg, plan)
    pkg.ready_for_design = not pkg.missing_question_ids and pkg.inventory_review["ready"]
    return {
        "package": pkg.model_dump(),
        "pending_questions": [q.model_dump() for q in IntakeAgent().pending_questions(pkg)],
        "alternatives": alternative_profiles(pkg, plan) if not pkg.inventory_review["ready"] else [],
    }


@app.post("/api/inventory/review")
def review_inventory_intake(payload: InventoryReviewPayload) -> dict[str, Any]:
    from flora_translate.intake_agent import IntakeAgent

    try:
        package = IntakeAgent().analyze(existing_package=payload.intake_package, use_llm=False).model_dump()
        package = _apply_inventory(package, payload.inventory_profile)
        return _inventory_response(package, payload.chemistry_plan)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/inventory/resolve")
def resolve_inventory(payload: InventoryResolutionPayload) -> dict[str, Any]:
    from flora_translate.inventory_profiles import inventory_profile_from_payload, save_inventory_profile
    from flora_translate.inventory_resolution import bind_inventory, confirm_inventory

    try:
        raw = payload.inventory_profile or payload.intake_package.get("inventory_profile_snapshot")
        if not raw:
            raise ValueError("Select or import an inventory profile before confirming equipment")
        profile = inventory_profile_from_payload(raw)
        updated, changed = confirm_inventory(profile, category=payload.category,
                                             status=payload.status, equipment=payload.equipment, note=payload.note)
        # Revalidate the package before persisting an immutable new version.
        bind_inventory(payload.intake_package, updated)
        if changed:
            updated, _ = save_inventory_profile(updated)
        package = bind_inventory(payload.intake_package, updated)
        return {**_inventory_response(package.model_dump(), payload.chemistry_plan),
                "profile": updated.model_dump(), "saved_new_version": changed}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/intake/analyze")
def analyze_intake(payload: IntakePayload) -> dict[str, Any]:
    from flora_translate.intake_agent import IntakeAgent
    from flora_translate.pipeline_runtime import (
        PipelineRuntimeOptions,
        runtime_model_routing,
    )

    try:
        runtime = PipelineRuntimeOptions()
        if payload.use_llm and payload.upstream_model_id:
            from flora_translate.model_catalog import (
                model_route_availability,
                route_for_id,
            )

            route = route_for_id(payload.upstream_model_id, role="upstream")
            status = model_route_availability(route)
            if not status["available"]:
                raise ValueError(
                    f"Upstream model {route.label} is unavailable: {status['reason']}"
                )
            runtime = PipelineRuntimeOptions(
                upstream_model=route.model,
                upstream_provider=route.provider,
                model_endpoints={route.model: route.base_url} if route.base_url else {},
            )
        agent = IntakeAgent()
        routing_context = (
            runtime_model_routing(runtime)
            if payload.use_llm and payload.upstream_model_id
            else nullcontext()
        )
        with routing_context:
            package = agent.analyze(
                payload.raw_protocol,
                existing_package=payload.existing_package,
                answers=payload.answers,
                use_llm=payload.use_llm,
            ).model_dump()
        package = _apply_inventory(package, payload.inventory_profile)
        return _inventory_response(package)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/inventory/editor-schema")
def inventory_editor_schema() -> dict[str, Any]:
    from .inventory_editor import editor_schema

    return editor_schema()


@app.get("/api/inventory/profiles")
def inventory_profiles() -> dict[str, Any]:
    from flora_translate.inventory_profiles import list_inventory_profiles

    return {"profiles": list_inventory_profiles()}


@app.get("/api/inventory/profiles/{profile_id}")
def inventory_profile(profile_id: str) -> dict[str, Any]:
    from flora_translate.inventory_profiles import load_inventory_profile

    try:
        return load_inventory_profile(profile_id).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/inventory/import")
def import_inventory(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    from flora_translate.inventory_profiles import inventory_profile_from_payload

    try:
        return inventory_profile_from_payload(payload).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/inventory/extract")
async def extract_inventory(
    name: str = Form("Imported laboratory inventory"),
    laboratory: str = Form(""),
    source_text: str = Form(""),
    use_llm: bool = Form(True),
    files: list[UploadFile] = File(default=[]),
) -> dict[str, Any]:
    from flora_translate.inventory_profiles import (
        extract_inventory_profile,
        extract_uploaded_documents,
    )

    uploaded: list[tuple[str, bytes, str]] = []
    for item in files:
        uploaded.append((item.filename or "upload", await item.read(), item.content_type or ""))
    try:
        document_text, provenance, errors = extract_uploaded_documents(uploaded)
        combined = "\n\n".join(value for value in (source_text, document_text) if value.strip())
        profile = extract_inventory_profile(
            combined,
            name=name,
            laboratory=laboratory,
            use_llm=use_llm,
        )
        profile.provenance = provenance
        response = profile.model_dump()
        response["document_errors"] = errors
        response["source_preview"] = combined[:5000]
        return response
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/inventory/save")
def save_inventory(payload: InventorySavePayload) -> dict[str, Any]:
    from flora_translate.inventory_profiles import (
        inventory_profile_from_payload,
        save_inventory_profile,
    )

    try:
        profile = inventory_profile_from_payload(payload.profile)
        saved, path = save_inventory_profile(profile)
        return {"profile": saved.model_dump(), "path": str(path)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/design/jobs", status_code=202)
def create_design_job(payload: DesignPayload) -> dict[str, Any]:
    try:
        package = _apply_inventory(deepcopy(payload.intake_package), payload.inventory_profile)
        from flora_translate.intake_agent import IntakeAgent
        package = IntakeAgent().analyze(existing_package=package, use_llm=False).model_dump()
        if payload.batch_input and payload.batch_input != package.get("raw_protocol"):
            raise ValueError("Protocol changed after intake. Re-analyze the protocol before design.")
        if not package.get("ready_for_design"):
            missing = ", ".join(package.get("missing_question_ids") or [])
            inventory_reasons = "; ".join(item["reason"] for item in package.get("inventory_review", {}).get("unresolved_requirements", []))
            raise ValueError(f"Intake is incomplete: {missing or inventory_reasons}. Resolve the equipment requirements or select a compatible profile.")
        request = payload.model_dump()
        request["intake_package"] = package
        if payload.upstream_model_id or payload.downstream_model_id:
            from flora_translate.model_catalog import (
                model_route_availability,
                route_for_id,
                runtime_model_options,
            )

            selected = {
                "upstream": route_for_id(payload.upstream_model_id, role="upstream"),
                "downstream": route_for_id(payload.downstream_model_id, role="downstream"),
            }
            for role, route in selected.items():
                status = model_route_availability(route)
                if not status["available"]:
                    raise ValueError(
                        f"{role.capitalize()} model {route.label} is unavailable: "
                        f"{status['reason']}"
                    )

            runtime_options = dict(payload.runtime_options or {})
            runtime_options.update(
                runtime_model_options(
                    payload.upstream_model_id,
                    payload.downstream_model_id,
                )
            )
            request["runtime_options"] = runtime_options
        job = JOBS.submit(request)
        return job.public(include_result=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/design/jobs")
def list_design_jobs() -> dict[str, Any]:
    return {"jobs": JOBS.list()}


@app.get("/api/design/jobs/{job_id}")
def get_design_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Design job not found")
    from flora_translate.result_reporting import attach_result_report

    payload = job.public(include_result=True)
    if payload.get("result"):
        payload["result"] = attach_result_report(payload["result"])
    return payload


def _artifact_for_job(job_id: str, kind: str) -> Path:
    job = JOBS.get(job_id)
    if not job or not job.result:
        raise HTTPException(status_code=404, detail="Completed design job not found")
    from flowpilot_webapp.backend.diagram_view import current_diagram

    current = current_diagram(job.result, kind)
    if current:
        return current
    keys = {
        "process-png": "png_path",
        "process-svg": "svg_path",
        "diagnostic-png": "diagnostic_png_path",
        "diagnostic-svg": "diagnostic_svg_path",
    }
    key = keys.get(kind)
    if not key:
        raise HTTPException(status_code=404, detail="Unknown artifact")
    path = Path(str(job.result.get(key) or ""))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact is unavailable")
    return path


@app.get("/api/design/jobs/{job_id}/artifacts/{kind}")
def design_artifact(job_id: str, kind: str) -> FileResponse:
    path = _artifact_for_job(job_id, kind)
    media_type = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/design/jobs/{job_id}/download")
def download_design(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job or not job.result:
        raise HTTPException(status_code=404, detail="Completed design job not found")
    return JSONResponse(job.result, headers={"Content-Disposition": f'attachment; filename="flowpilot_{job_id}.json"'})


@app.post("/api/refinement")
def refine_design(payload: RefinementPayload) -> dict[str, Any]:
    from flora_translate.experiment_loop import (
        ExperimentResult,
        refine_from_experiment,
        refine_from_experimental_campaign,
    )
    from flora_translate.gui_autosave import autosave_gui_result

    current = payload.current_result
    if payload.job_id:
        job = JOBS.get(payload.job_id)
        current = job.result if job else None
    if not current:
        raise HTTPException(status_code=400, detail="A completed design result is required")
    if (current.get("pipeline_runtime") or {}).get("design_policy") == "scientific_v2":
        raise HTTPException(status_code=409, detail="Scientific preview requires a new intake with measured results and a new 12-candidate council review; legacy scalar refinement cannot modify this frozen design.")
    if not payload.experiments:
        raise HTTPException(status_code=400, detail="At least one experimental result is required")
    try:
        experiments = [ExperimentResult.model_validate(item) for item in payload.experiments]
        if len(experiments) == 1:
            closed = refine_from_experiment(
                current,
                experiments[0],
                target_yield_pct=payload.target_yield_pct,
                target_conversion_pct=payload.target_conversion_pct,
                target_selectivity_pct=payload.target_selectivity_pct,
            )
        else:
            closed = refine_from_experimental_campaign(
                current,
                experiments,
                target_yield_pct=payload.target_yield_pct,
                target_conversion_pct=payload.target_conversion_pct,
                target_selectivity_pct=payload.target_selectivity_pct,
            )
        result = closed.model_dump()
        save_dir = autosave_gui_result(
            result["refined_result"],
            source="webapp_refinement",
            user_input=json.dumps(payload.experiments, default=str),
        )
        result["autosave_dir"] = str(save_dir)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runs")
def saved_runs(limit: int = 50) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    root = Path("outputs/gui_runs")
    if root.is_dir():
        for directory in sorted((item for item in root.iterdir() if item.is_dir()), reverse=True):
            summary_path = directory / "summary.json"
            if not summary_path.is_file():
                continue
            try:
                summary = json.loads(summary_path.read_text())
            except Exception:
                continue
            summary["run_id"] = directory.name
            summary["path"] = str(directory)
            summary["has_process_png"] = (directory / "process.png").is_file()
            rows.append(summary)
            if len(rows) >= max(1, min(limit, 200)):
                break
    return {"runs": rows}


def _saved_run_directory(run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id:
        raise HTTPException(status_code=400, detail="Invalid saved run identifier")
    directory = Path("outputs/gui_runs") / run_id
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="Saved run not found")
    return directory


@app.get("/api/runs/{run_id}")
def saved_run(run_id: str) -> dict[str, Any]:
    directory = _saved_run_directory(run_id)
    result_path = directory / "result.json"
    if not result_path.is_file():
        raise HTTPException(status_code=404, detail="Saved result JSON is unavailable")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Saved result is invalid: {exc}") from exc
    result["autosave_dir"] = str(directory)
    from flora_translate.result_reporting import attach_result_report

    return attach_result_report(result)


@app.get("/api/runs/{run_id}/artifacts/{kind}")
def saved_run_artifact(run_id: str, kind: str) -> FileResponse:
    directory = _saved_run_directory(run_id)
    names = {
        "process-png": "process.png",
        "process-svg": "process.svg",
        "diagnostic-png": "diagnostic_process.png",
        "diagnostic-svg": "diagnostic_process.svg",
    }
    filename = names.get(kind)
    if not filename:
        raise HTTPException(status_code=404, detail="Unknown artifact")
    path = directory / filename
    from flowpilot_webapp.backend.diagram_view import current_diagram

    current = current_diagram(saved_run(run_id), kind)
    if current:
        path = current
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Saved artifact is unavailable")
    media_type = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/runs/{run_id}/download")
def download_saved_run(run_id: str) -> FileResponse:
    path = _saved_run_directory(run_id) / "result.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Saved result JSON is unavailable")
    return FileResponse(path, media_type="application/json", filename=f"flowpilot_{run_id}.json")


if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{full_path:path}")
def frontend(full_path: str) -> FileResponse:
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")
    candidate = (DIST / full_path).resolve()
    if full_path and candidate.is_file() and DIST.resolve() in candidate.parents:
        return FileResponse(candidate)
    index = DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(status_code=503, detail="Frontend has not been built. Run npm run build.")
