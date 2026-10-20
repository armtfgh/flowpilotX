"""Versioned icon diagrams shared by both GUIs, without changing archives."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from filelock import FileLock

from flora_translate.diagram_artifacts import render_topology_artifacts, topology_sha256
from flora_translate.schemas import ProcessTopology


CACHE = Path("outputs/diagram_views")


def current_diagram(result: dict, kind: str) -> Path | None:
    if kind not in {"process-svg", "process-png"} or (result.get("final_design") or {}).get("status") != "executable":
        return None
    topology = ((result["final_design"].get("process_graph") or {}).get("topology") or result.get("process_topology"))
    if not topology:
        return None
    from flora_design.visualizer import flowsheet_builder

    source_hash = topology_sha256(topology)
    display = deepcopy(topology)
    streams = {str(s.get("stream_label", "")).casefold(): s for s in result["final_design"].get("streams", [])}
    for operation in display.get("unit_operations", []):
        params = operation.get("parameters") or {}
        stream = streams.get(str(params.get("stream", "")).casefold())
        if stream and stream.get("phase") == "gas":
            params["molar_equiv"] = stream.get("molar_equiv")
    fingerprint = hashlib.sha256(Path(flowsheet_builder.__file__).read_bytes() + Path(__file__).read_bytes()
                                 + json.dumps(display, sort_keys=True).encode()).hexdigest()
    folder = CACHE / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    index = folder / "view.json"
    with FileLock(str(folder / "render.lock"), timeout=90):
        artifacts = None
        if index.is_file():
            try:
                cached = json.loads(index.read_text())
                if isinstance(cached, dict) and all(
                    isinstance(cached.get(key), str) and Path(cached[key]).is_file()
                    for key in ("svg_path", "png_path")
                ):
                    artifacts = cached
            except (OSError, ValueError):
                pass
        if artifacts is None:
            artifacts = render_topology_artifacts(ProcessTopology.model_validate(display), base_dir=folder)
            artifacts["source_topology_sha256"] = source_hash
            artifacts["caption_source"] = "final_design.streams for gas equivalents; original topology for setpoints"
            temporary = index.with_suffix(".tmp")
            temporary.write_text(json.dumps(artifacts, indent=2))
            temporary.replace(index)
    path = Path(artifacts["svg_path" if kind.endswith("svg") else "png_path"])
    return path if path.is_file() else None
