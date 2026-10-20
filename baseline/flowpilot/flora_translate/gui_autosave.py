"""Autosave helpers for GUI-generated FlowPilot designs."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GUI_RUNS_DIR = Path("outputs/gui_runs")


def autosave_gui_result(
    result: dict[str, Any],
    *,
    intake_package: Any = None,
    source: str = "gui",
    user_input: str | None = None,
    base_dir: Path = GUI_RUNS_DIR,
) -> Path:
    """Write one completed GUI result and rebind it to immutable assets."""

    run_dir = _unique_run_dir(base_dir, source)
    run_dir.mkdir(parents=True, exist_ok=False)

    safe_result = _json_safe(result)
    if not safe_result.get("final_design"):
        from flora_translate.final_design_contract import build_final_design_contract

        safe_result["final_design"] = build_final_design_contract(safe_result)
        result["final_design"] = _json_safe(safe_result["final_design"])

    svg_path = _copy_if_exists(
        safe_result.get("svg_path"), run_dir / "process.svg"
    )
    png_path = _copy_if_exists(
        safe_result.get("png_path"), run_dir / "process.png"
    )
    safe_result["svg_path"] = svg_path
    safe_result["png_path"] = png_path
    result["svg_path"] = svg_path
    result["png_path"] = png_path

    diagnostic_svg_path = _copy_if_exists(
        safe_result.get("diagnostic_svg_path"), run_dir / "diagnostic_process.svg"
    )
    diagnostic_png_path = _copy_if_exists(
        safe_result.get("diagnostic_png_path"), run_dir / "diagnostic_process.png"
    )
    safe_result["diagnostic_svg_path"] = diagnostic_svg_path
    safe_result["diagnostic_png_path"] = diagnostic_png_path
    result["diagnostic_svg_path"] = diagnostic_svg_path
    result["diagnostic_png_path"] = diagnostic_png_path

    topology = safe_result.get("process_topology")
    if topology is not None:
        (run_dir / "topology.json").write_text(
            json.dumps(topology, indent=2), encoding="utf-8"
        )

    allocation = safe_result.get("inventory_allocation")
    if allocation is not None:
        (run_dir / "inventory_allocation.json").write_text(
            json.dumps(allocation, indent=2), encoding="utf-8"
        )
    manifest_items = safe_result.get("instrument_manifest")
    if manifest_items is not None:
        (run_dir / "instrument_manifest.json").write_text(
            json.dumps(manifest_items, indent=2), encoding="utf-8"
        )
    requirements_topology = safe_result.get("process_requirements_topology")
    if requirements_topology is not None:
        (run_dir / "process_requirements_topology.json").write_text(
            json.dumps(requirements_topology, indent=2), encoding="utf-8"
        )
    diagnostic_topology = safe_result.get("diagnostic_topology")
    if diagnostic_topology is not None:
        (run_dir / "diagnostic_topology.json").write_text(
            json.dumps(diagnostic_topology, indent=2), encoding="utf-8"
        )

    diagnostic_manifest = _json_safe(
        safe_result.get("diagnostic_diagram_render_manifest") or {}
    )
    if diagnostic_manifest or diagnostic_svg_path or diagnostic_png_path:
        diagnostic_manifest.update(
            {
                "autosaved_at": datetime.now(timezone.utc).isoformat(),
                "autosave_run_dir": str(run_dir.resolve()),
                "topology_path": (
                    str((run_dir / "diagnostic_topology.json").resolve())
                    if (run_dir / "diagnostic_topology.json").is_file()
                    else ""
                ),
                "svg_path": diagnostic_svg_path,
                "png_path": diagnostic_png_path,
                "status": (
                    "complete"
                    if diagnostic_svg_path and diagnostic_png_path
                    else "partial"
                    if diagnostic_svg_path
                    else "failed"
                ),
            }
        )
        (run_dir / "diagnostic_render_manifest.json").write_text(
            json.dumps(diagnostic_manifest, indent=2), encoding="utf-8"
        )
        safe_result["diagnostic_diagram_render_manifest"] = diagnostic_manifest
        result["diagnostic_diagram_render_manifest"] = diagnostic_manifest

    diagnostic_artifacts = _json_safe(
        safe_result.get("diagnostic_diagram_artifacts") or {}
    )
    if diagnostic_artifacts or diagnostic_svg_path or diagnostic_png_path:
        diagnostic_artifacts.update(
            {
                "run_dir": str(run_dir.resolve()),
                "svg_path": diagnostic_svg_path,
                "png_path": diagnostic_png_path,
                "topology_path": (
                    str((run_dir / "diagnostic_topology.json").resolve())
                    if (run_dir / "diagnostic_topology.json").is_file()
                    else ""
                ),
                "render_manifest_path": str(
                    (run_dir / "diagnostic_render_manifest.json").resolve()
                ),
                "render_status": diagnostic_manifest.get("status", "failed"),
            }
        )
        safe_result["diagnostic_diagram_artifacts"] = diagnostic_artifacts
        result["diagnostic_diagram_artifacts"] = diagnostic_artifacts

    render_manifest = _autosave_render_manifest(
        safe_result,
        run_dir=run_dir,
        svg_path=svg_path,
        png_path=png_path,
    )
    if render_manifest:
        manifest_path = run_dir / "render_manifest.json"
        manifest_path.write_text(json.dumps(render_manifest, indent=2), encoding="utf-8")
        safe_result["diagram_render_manifest"] = render_manifest
        result["diagram_render_manifest"] = render_manifest

    diagram_artifacts = _rebind_diagram_artifacts(
        safe_result.get("diagram_artifacts"),
        run_dir=run_dir,
        svg_path=svg_path,
        png_path=png_path,
        has_topology=topology is not None,
        has_manifest=bool(render_manifest),
    )
    if diagram_artifacts:
        safe_result["diagram_artifacts"] = diagram_artifacts
        result["diagram_artifacts"] = diagram_artifacts

    (run_dir / "result.json").write_text(
        json.dumps(safe_result, indent=2), encoding="utf-8"
    )
    final_design = safe_result.get("final_design")
    if final_design is not None:
        (run_dir / "final_design.json").write_text(
            json.dumps(final_design, indent=2), encoding="utf-8"
        )
    (run_dir / "summary.json").write_text(
        json.dumps(_summary(safe_result, source), indent=2), encoding="utf-8"
    )

    package_payload = intake_package if intake_package is not None else safe_result.get("intake_package")
    if package_payload is not None:
        if hasattr(package_payload, "model_dump"):
            package_payload = package_payload.model_dump()
        (run_dir / "intake_package.json").write_text(
            json.dumps(_json_safe(package_payload), indent=2)
        )

    if user_input:
        (run_dir / "input.txt").write_text(str(user_input))

    return run_dir


def _unique_run_dir(base_dir: Path, source: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in source)
    candidate = base_dir / f"{stamp}_{safe_source}"
    counter = 2
    while candidate.exists():
        candidate = base_dir / f"{stamp}_{safe_source}_{counter}"
        counter += 1
    return candidate


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _summary(result: dict[str, Any], source: str) -> dict[str, Any]:
    from flora_translate.final_design_contract import build_final_design_contract

    final_design = result.get("final_design") or build_final_design_contract(result)
    proposal = final_design.get("parameters") or {}
    gas = {}
    for stream in final_design.get("streams") or []:
        if str(stream.get("phase") or "").lower() == "gas" or stream.get("gas_flow_sccm") is not None:
            gas = stream
            break
    return {
        "source": source,
        "final_design_status": final_design.get("status"),
        "final_design_schema_version": final_design.get("schema_version"),
        "confidence": result.get("confidence"),
        "recommended_disposition": result.get("recommended_disposition"),
        "disposition_rationale": result.get("disposition_rationale"),
        "hard_failures": (
            (result.get("design_disposition") or {}).get("hard_failures") or []
        ),
        "reaction": (result.get("chemistry_plan") or {}).get("reaction_name"),
        "residence_time_min": proposal.get("residence_time_min"),
        "residence_time_inlet_min": proposal.get("residence_time_inlet_min"),
        "residence_time_in_channel_min": proposal.get("residence_time_in_channel_min"),
        "flow_rate_mL_min": proposal.get("flow_rate_mL_min"),
        "gas_flow_sccm": gas.get("gas_flow_sccm"),
        "gas_flow_actual_mL_min": gas.get("gas_flow_actual_mL_min"),
        "reactor_volume_mL": proposal.get("reactor_volume_mL"),
        "tubing_ID_mm": proposal.get("tubing_ID_mm"),
        "temperature_C": proposal.get("temperature_C"),
        "concentration_M": proposal.get("concentration_M"),
        "BPR_bar": proposal.get("BPR_bar"),
        "wavelength_nm": proposal.get("wavelength_nm"),
        "inventory_selection": proposal.get("inventory_selection"),
        "safety_flags": proposal.get("safety_flags"),
        "consistency_issues": (final_design.get("consistency") or {}).get("issues") or [],
    }


def _copy_if_exists(src: Any, dst: Path) -> str:
    if not src:
        return ""
    path = Path(str(src))
    if path.exists() and path.is_file():
        if path.resolve() != dst.resolve():
            shutil.copy2(path, dst)
        return str(dst.resolve())
    return ""


def _autosave_render_manifest(
    result: dict[str, Any],
    *,
    run_dir: Path,
    svg_path: str,
    png_path: str,
) -> dict[str, Any]:
    manifest = _json_safe(result.get("diagram_render_manifest") or {})
    artifacts = result.get("diagram_artifacts") or {}
    if not manifest and not artifacts and not svg_path and not png_path:
        return {}

    manifest.update(
        {
            "autosaved_at": datetime.now(timezone.utc).isoformat(),
            "autosave_run_dir": str(run_dir.resolve()),
            "topology_path": (
                str((run_dir / "topology.json").resolve())
                if (run_dir / "topology.json").is_file()
                else ""
            ),
            "svg_path": svg_path,
            "png_path": png_path,
            "status": (
                "complete" if svg_path and png_path else "partial" if svg_path else "failed"
            ),
        }
    )
    return manifest


def _rebind_diagram_artifacts(
    value: Any,
    *,
    run_dir: Path,
    svg_path: str,
    png_path: str,
    has_topology: bool,
    has_manifest: bool,
) -> dict[str, Any]:
    artifacts = _json_safe(value or {})
    if not artifacts and not svg_path and not png_path:
        return {}
    original_run_dir = artifacts.get("run_dir")
    if original_run_dir:
        artifacts["source_run_dir"] = original_run_dir
    artifacts.update(
        {
            "run_dir": str(run_dir.resolve()),
            "svg_path": svg_path,
            "png_path": png_path,
            "topology_path": str((run_dir / "topology.json").resolve()) if has_topology else "",
            "render_manifest_path": (
                str((run_dir / "render_manifest.json").resolve()) if has_manifest else ""
            ),
            "render_status": (
                "complete" if svg_path and png_path else "partial" if svg_path else "failed"
            ),
        }
    )
    return artifacts
