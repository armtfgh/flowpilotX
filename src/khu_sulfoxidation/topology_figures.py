"""Six frozen KHU settings projected through FlowPilot's original icon renderer.

The preserved renderer is loaded as an isolated file, never via its eager agent
package. Only study captions change; icons and material edges are reused. All
generated images, caches and source copies live in the new output directory.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import types
import xml.etree.ElementTree as ET

from .provenance import digest, file_hash, write_json

BUILDER = "baseline/flowpilot/flora_design/visualizer/flowsheet_builder.py"
ARTIFACTS = "baseline/flowpilot/flora_translate/diagram_artifacts.py"
ICONS = "baseline/flowpilot/flora_design/visualizer/icons"
CORE_IDS = tuple(f"HI{i:02d}" for i in range(1, 7))
EDGES = (("feed", "mixer", "liquid"), ("air", "check", "gas"),
         ("check", "mixer", "gas"), ("mixer", "reactor", "gas_liquid"),
         ("reactor", "bpr", "gas_liquid"), ("bpr", "collector", "gas_liquid"))


def load_proposals(source_dir):
    source_dir = Path(source_dir)
    manifest = json.loads((source_dir / "manifest.json").read_text())
    for name in ("authoritative_conditions.json", "nominations_freeze.json"):
        if file_hash(source_dir / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Frozen report artifact changed: {name}")
    proposals = json.loads((source_dir / "authoritative_conditions.json").read_text())
    nominations = json.loads((source_dir / "nominations_freeze.json").read_text())
    core = [r for r in proposals["conditions"] if r["condition_id"] in CORE_IDS]
    chief_ids = nominations["arms"]["history_informed"]["Chief"]["recommended_core_candidate_ids"]
    if tuple(r["condition_id"] for r in core) != CORE_IDS or [r["candidate_id"] for r in core] != chief_ids:
        raise ValueError("Six ordered core proposals must match the frozen Chief selection")
    if proposals["status"] != "conditional_review_not_released_for_experiment":
        raise ValueError("This projection requires the conditional review package")
    if nominations["flow_outcomes_seen"] is not False or nominations["laboratory_released"] is not False:
        raise ValueError("Expected an unreleased, pre-outcome nomination freeze")
    return proposals, core


def build_topology(row, scenario):
    """Build display data without assigning unresolved hardware or actual values."""
    p, c = row["proposed_setpoints"], row["calculated"]
    recipe, geometry = scenario["shared_recipe"], scenario["geometry"]
    ref = row["gas_reference_assumptions"]
    def op(identity, kind, label, parameters, equipment_id=None, unresolved=False):
        return {"op_id": identity, "op_type": kind, "label": label,
                "parameters": parameters, "inventory_item_id": equipment_id,
                "assignment_status": "unresolved" if unresolved else "proposed_not_confirmed",
                "inventory_category": parameters.get("unresolved_inventory_category", kind),
                "actual_assignment_confirmed": None}
    operations = [
        op("feed", "pump", "Premixed feed", {
            "stream": "A", "phase": "liquid", "instrument_name": "R-series HPLC (proposed)",
            "contents": ["Thioanisole", "Acridine: 3 mol% (7.5 mM)"],
            "solvent": recipe["solvent"], "concentration_M": recipe["substrate_concentration_M"],
            "flow_rate_mL_min": p["liquid_flow_mL_min"]}, "pump_r_series_hplc"),
        op("air", "mfc", "Air delivery", {
            "stream": "B", "phase": "gas", "instrument_name": "FF-C00 (proposed)",
            "contents": ["Air"], "gas_flow_sccm": p["gas_reference_flow_mL_min"],
            "gas_reference_assumptions": deepcopy(ref),
            "oxygen_equivalents_supplied": c["oxygen_equivalents_supplied"],
            "nominal_oxygen_equivalents": row["nominal_oxygen_equivalents"],
            "unresolved_inventory_category": "gas calibration / reference",
            "regulator_id": None, "supply_pressure_bar": None}, "mfc_ffc00", True),
        op("check", "check_valve", "Gas check valve", {
            "instrument_name": "CV-3301 (proposed)", "method": "Rating / headroom pending"}, "cv-3301"),
        op("mixer", "t_mixer", "Gas-liquid T-mixer", {
            "instrument_name": "P-713 (proposed)"}, "p-713"),
        op("reactor", "photoreactor", "Homogeneous photoreactor", {
            "instrument_name": "PFA coil (proposed)", "light_instrument_name": "UV-150",
            "temperature_C": p["temperature_C"], "wavelength_nm": 450,
            "volume_mL": geometry["volume_mL"], "ID_mm": geometry["ID_mm"],
            "liquid_only_space_time_min": c["liquid_only_space_time_min"],
            "actual_liquid_contact_time_min": None, "light_power_setting": None}, "coil_row5"),
        op("bpr", "bpr", "Back-pressure regulator", {
            "instrument_name": "BPR: selection pending", "pressure_bar": None,
            "unresolved_inventory_category": "pressure / basis"}, unresolved=True),
        op("collector", "collector", "Collection\nProcedure pending", {
            "actual_collection_window": None, "sample_handling": None}, unresolved=True),
    ]
    return {"schema_version": "khu_flowpilot_display_topology_v1", "topology_id": row["condition_id"] + "_" + row["candidate_id"],
        "condition_id": row["condition_id"], "candidate_id": row["candidate_id"],
        "compilation_status": "conditional_display_projection_not_executable",
        "source_condition_sha256": digest(row), "unit_operations": operations,
        "streams": [{"stream_id": f"s{i:02d}", "from_op": source, "to_op": target,
                     "stream_type": phase, "connection_type": "process"}
                    for i, (source, target, phase) in enumerate(EDGES, 1)],
        "unplaced_requirements": [{"component": "pressure sensing", "actual_equipment_id": None,
            "actual_location": None, "display_note": "Pressure-sensor identity/location remain pending; no location is assigned by this rendering."}],
        "gas_upstream_display": "Air supply and regulator are described in the MFC caption; regulator identity and supply pressure remain unknown.",
        "feed_mode": None, "actual_liquid_contact_time_min": None,
        "proposed_setpoints": deepcopy(p), "calculated_quantities": deepcopy(c),
        "gas_reference_assumptions": deepcopy(ref), "nominal_oxygen_equivalents": row["nominal_oxygen_equivalents"],
        "measurement_status": "No actual flow observations or experimental validation",
        "connectivity_note": "All six core proposals share these material connections; operating settings differ."}


def _load_file(path, name):
    # Compile explicitly: no eager package __init__, import hooks or .pyc writes
    # into the preserved baseline.
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def _configure_captions(module):
    original_reactor_label = module._reactor_label
    def reactor_label(op):
        text = original_reactor_label(op)
        value = op.parameters["liquid_only_space_time_min"]
        return text + f"\nLiquid-only V/Q_L: {value:.2f} min\nContact time: unmeasured"
    def mfc_label(op):
        p = op.parameters
        ref = p["gas_reference_assumptions"]
        lines = [module._font_lines("Stream B\n" + p["instrument_name"], bold=True),
            module._font_lines("Air via regulator", 10, "#DC2626"),
            module._font_lines(f"{p['gas_flow_sccm']:.2f} reference mL/min", 10),
            module._font_lines(f"Assumed {ref['temperature_K']:.2f} K\n{ref['pressure_bar_abs']:.5f} bar abs", 9),
            module._font_lines(f"Calculated O2: {p['oxygen_equivalents_supplied']:.4f} eq", 9),
            module._font_lines("Calibration / supply P pending", 8, "#B91C1C")]
        return "<BR/>".join(lines)
    module._reactor_label = reactor_label
    module._mfc_label = mfc_label


def render_figures(root, source_dir, output_dir):
    root, source_dir, output_dir = Path(root).resolve(), Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Choose a new output directory; prior figures are immutable")
    proposals, core = load_proposals(source_dir)
    original_paths = [root / BUILDER, root / ARTIFACTS, *sorted((root / ICONS).glob('*'))]
    original_hashes = {str(p.relative_to(root)): file_hash(p) for p in original_paths if p.is_file()}
    output_dir.mkdir(parents=True)
    asset_dir = output_dir / "assets/icons"
    shutil.copytree(root / ICONS, asset_dir)
    source_copies = output_dir / "renderer_source"
    source_copies.mkdir()
    for path in [root / BUILDER, root / ARTIFACTS]:
        shutil.copy2(path, source_copies / path.name)
    renderer = _load_file(root / BUILDER, "_khu_original_flowsheet_builder")
    artifacts = _load_file(root / ARTIFACTS, "_khu_original_diagram_artifacts")
    renderer.ICONS_DIR = asset_dir
    renderer.ASSETS = {key: asset_dir / p.name for key, p in renderer.ASSETS.items()}
    _configure_captions(renderer)
    previous_cache = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(output_dir / "render_cache")
    before_source = file_hash(source_dir / "authoritative_conditions.json")
    cases = []
    try:
        if not renderer._ensure_graphviz_on_path():
            raise RuntimeError("Graphviz is required; diagnostic fallback is not a topology deliverable")
        class StrictBuilder(renderer.FlowsheetBuilder):
            def build(self, payload, **kwargs):
                view = types.SimpleNamespace(
                    unit_operations=[types.SimpleNamespace(**deepcopy(o)) for o in payload["unit_operations"]],
                    streams=[types.SimpleNamespace(**deepcopy(s)) for s in payload["streams"]])
                return super().build(view, **kwargs)
            def _legacy_fallback(self, *args, **kwargs):
                raise RuntimeError("Study rendering refuses legacy/diagnostic fallback")
        for row in core:
            payload = build_topology(row, proposals["scenario"])
            title = f"{row['condition_id']} / {row['candidate_id']} - proposed flow configuration"
            builder = StrictBuilder()
            result = artifacts.render_topology_artifacts(payload, title=title,
                base_dir=output_dir / "render_runs" / row["condition_id"], builder=builder)
            if result["renderer"] != "graphviz" or result["render_status"] != "complete" or result["warnings"]:
                raise RuntimeError(f"Incomplete original-renderer output preserved: {result}")
            svg_root = ET.parse(result["svg_path"]).getroot()
            images = list(svg_root.iter("{http://www.w3.org/2000/svg}image"))
            if len(images) != 7 or any(not e.get("{http://www.w3.org/1999/xlink}href", e.get("href", "")).startswith("data:image/") for e in images):
                raise ValueError("Expected seven self-contained original equipment icons")
            edge_titles = [e.find("{http://www.w3.org/2000/svg}title").text for e in svg_root.iter("{http://www.w3.org/2000/svg}g") if e.get("class") == "edge"]
            actual_edges = {"->".join(part.split(":", 1)[0] for part in title.split("->")) for title in edge_titles}
            if actual_edges != {source + "->" + target for source, target, _ in EDGES}:
                raise ValueError("Rendered graph does not preserve all six material connections")
            dest = output_dir / "diagrams"
            dest.mkdir(exist_ok=True)
            stem = payload["topology_id"]
            files = {}
            for key, suffix in [("svg_path", "svg"), ("png_path", "png"), ("topology_path", "json")]:
                path = dest / f"{stem}.{suffix}"
                shutil.copyfile(result[key], path)
                files[{"svg": "svg", "png": "png", "json": "topology_json"}[suffix]] = str(path.relative_to(output_dir))
            cases.append({"condition_id": row["condition_id"], "candidate_id": row["candidate_id"],
                "title": title, **files, "topology_sha256": result["topology_sha256"],
                "render_manifest": str(Path(result["render_manifest_path"]).relative_to(output_dir)),
                "renderer": result["renderer"], "main_process_path": builder.last_render_info["main_process_path"]})
        if before_source != file_hash(source_dir / "authoritative_conditions.json"):
            raise ValueError("Authoritative proposals changed during rendering")
        if any(file_hash(root / p) != h for p, h in original_hashes.items()):
            raise ValueError("Preserved FlowPilot renderer or icon changed")
        index = {"schema_version": "khu_six_flowpilot_figures_v1", "cases": cases,
            "source_report_directory": str(source_dir),
            "source_files_sha256": {name: file_hash(source_dir / name) for name in ("manifest.json", "authoritative_conditions.json", "nominations_freeze.json")},
            "renderer_reuse": {"original_files_sha256": original_hashes, "adapter_sha256": file_hash(Path(__file__)),
                "module_loading": "Isolated source-file loading; no FlowPilot agent/runtime imports or baseline writes",
                "caption_overrides": ["MFC: explicit assumed reference, calculated oxygen supply and pending calibration", "Reactor: liquid-only V/Q_L and unmeasured contact time; legacy residence fields omitted"],
                "icons": "Original FlowPilot equipment assets, copied to output-local directory", "fallback_allowed": False},
            "no_model_calls": True, "laboratory_released": False}
        write_json(output_dir / "render_index.json", index)
        return index
    except Exception as exc:
        write_json(output_dir / "render_failure.json", {"error": str(exc), "prior_figures_preserved": True})
        raise
    finally:
        if previous_cache is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = previous_cache
