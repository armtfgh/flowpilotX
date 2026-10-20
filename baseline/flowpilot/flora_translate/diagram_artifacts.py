"""Deterministic, run-local process-diagram artifact generation."""

from __future__ import annotations

import hashlib
import base64
import json
import logging
import mimetypes
import re
import shutil
import subprocess
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
from uuid import uuid4


logger = logging.getLogger("flora.diagram_artifacts")

DIAGRAM_RUNS_DIR = Path("outputs/diagram_runs")


def render_topology_artifacts(
    topology: Any,
    *,
    title: str = "",
    base_dir: Path = DIAGRAM_RUNS_DIR,
    builder: Any = None,
) -> dict[str, Any]:
    """Render one topology into an immutable, self-describing run folder."""

    run_id = _run_id()
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    topology_payload = _topology_payload(topology)
    topology_json = _canonical_json(topology_payload)
    topology_hash = hashlib.sha256(topology_json.encode("utf-8")).hexdigest()
    topology_path = run_dir / "topology.json"
    topology_path.write_text(
        json.dumps(topology_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    svg_path = run_dir / "process.svg"
    png_path = run_dir / "process.png"
    warnings: list[str] = []
    renderer = "graphviz"

    try:
        if builder is None:
            from flora_design.visualizer.flowsheet_builder import FlowsheetBuilder

            builder = FlowsheetBuilder()
        rendered_svg, rendered_png = builder.build(
            topology,
            title=title,
            output_svg=str(svg_path),
            output_png=str(png_path),
        )
        renderer_info = getattr(builder, "last_render_info", {}) or {}
        renderer = str(renderer_info.get("renderer") or renderer)
        warnings.extend(str(item) for item in renderer_info.get("warnings", []))
        svg_path = Path(rendered_svg) if rendered_svg else svg_path
        png_path = Path(rendered_png) if rendered_png else png_path
    except Exception as exc:  # final fallback must remain independent of Graphviz
        warnings.append(f"Primary renderer failed: {exc}")
        logger.exception("Process-diagram renderer failed; using minimal SVG")

    if not svg_path.is_file():
        _write_minimal_svg(topology_payload, title, svg_path)
        renderer = "deterministic_svg_fallback"
        warnings.append("Used deterministic SVG fallback.")

    if svg_path.is_file():
        warnings.extend(_inline_svg_images(svg_path))

    if svg_path.is_file() and not png_path.is_file():
        png_warning = _convert_svg_to_png(svg_path, png_path)
        if png_warning:
            warnings.append(png_warning)

    svg_value = str(svg_path.resolve()) if svg_path.is_file() else ""
    png_value = str(png_path.resolve()) if png_path.is_file() else ""
    status = "complete" if svg_value and png_value else "partial" if svg_value else "failed"

    manifest = {
        "schema_version": "flowpilot_diagram_render_v1.0",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "renderer": renderer,
        "topology_sha256": topology_hash,
        "topology_path": str(topology_path.resolve()),
        "svg_path": svg_value,
        "png_path": png_value,
        "warnings": warnings,
    }
    manifest_path = run_dir / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "topology_path": str(topology_path.resolve()),
        "render_manifest_path": str(manifest_path.resolve()),
        "topology_sha256": topology_hash,
        "render_status": status,
        "renderer": renderer,
        "svg_path": svg_value,
        "png_path": png_value,
        "warnings": warnings,
        "manifest": manifest,
    }


def topology_sha256(topology: Any) -> str:
    """Return the stable hash used to bind a diagram to its source topology."""

    payload = _topology_payload(topology)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"{stamp}_{uuid4().hex[:8]}"


def _topology_payload(topology: Any) -> dict[str, Any]:
    if hasattr(topology, "model_dump"):
        return topology.model_dump(mode="json")
    if isinstance(topology, dict):
        return json.loads(json.dumps(topology, default=str))
    raise TypeError("Topology must be a Pydantic model or dictionary")


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_minimal_svg(payload: dict[str, Any], title: str, output_path: Path) -> None:
    """Write a dependency-free diagnostic diagram when richer renderers fail."""

    operations = list(payload.get("unit_operations") or [])
    if not operations:
        operations = [{"op_id": "empty", "op_type": "diagnostic", "label": "No unit operations"}]

    box_width = 170
    box_height = 84
    gap = 54
    margin = 28
    title_height = 52 if title else 20
    width = max(460, margin * 2 + len(operations) * box_width + (len(operations) - 1) * gap)
    height = title_height + box_height + 76
    top = title_height + 12

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="FlowPilot process topology">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" '
        'orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#2457a6"/></marker></defs>',
    ]
    if title:
        elements.append(
            f'<text x="{width / 2}" y="28" text-anchor="middle" '
            f'font-family="Arial,sans-serif" font-size="15" font-weight="700" '
            f'fill="#172033">{escape(title[:100])}</text>'
        )

    for index, operation in enumerate(operations):
        x = margin + index * (box_width + gap)
        label = escape(str(operation.get("label") or operation.get("op_id") or "Operation")[:32])
        op_type = escape(str(operation.get("op_type") or "unit operation")[:28])
        elements.extend(
            [
                f'<rect x="{x}" y="{top}" width="{box_width}" height="{box_height}" '
                'rx="6" fill="#f7f9fc" stroke="#315b8a" stroke-width="2"/>',
                f'<text x="{x + box_width / 2}" y="{top + 34}" text-anchor="middle" '
                f'font-family="Arial,sans-serif" font-size="12" font-weight="700" '
                f'fill="#172033">{label}</text>',
                f'<text x="{x + box_width / 2}" y="{top + 58}" text-anchor="middle" '
                f'font-family="Arial,sans-serif" font-size="10" fill="#526174">{op_type}</text>',
            ]
        )
        if index:
            previous_x = x - gap
            elements.append(
                f'<line x1="{previous_x}" y1="{top + box_height / 2}" x2="{x - 8}" '
                f'y2="{top + box_height / 2}" stroke="#2457a6" stroke-width="2" '
                'marker-end="url(#arrow)"/>'
            )

    elements.append(
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" '
        'font-family="Arial,sans-serif" font-size="9" fill="#7a8797">'
        'Deterministic fallback view</text>'
    )
    elements.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")


def _convert_svg_to_png(svg_path: Path, png_path: Path) -> str:
    """Use an installed ImageMagick executable when the renderer skipped PNG."""

    executable = shutil.which("magick") or shutil.which("convert")
    if not executable:
        return "PNG conversion unavailable; SVG remains the authoritative artifact."
    try:
        subprocess.run(
            [executable, str(svg_path), str(png_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"PNG conversion failed: {exc}"
    return ""


def _inline_svg_images(svg_path: Path) -> list[str]:
    """Embed Graphviz image references so copied/browser SVGs remain complete."""

    try:
        source = svg_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"SVG asset inlining failed: {exc}"]

    icon_dir = Path(__file__).resolve().parents[1] / "flora_design" / "visualizer" / "icons"
    unresolved: list[str] = []

    def replace(match: re.Match[str]) -> str:
        attribute = match.group("attribute")
        reference = match.group("reference")
        if reference.startswith(("data:", "#", "http://", "https://")):
            return match.group(0)
        candidate = Path(reference)
        search = [candidate] if candidate.is_absolute() else [
            svg_path.parent / candidate,
            icon_dir / candidate.name,
        ]
        asset = next((path for path in search if path.is_file()), None)
        if asset is None:
            unresolved.append(reference)
            return match.group(0)
        mime = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(asset.read_bytes()).decode("ascii")
        return f'{attribute}="data:{mime};base64,{encoded}"'

    updated = re.sub(
        r'(?P<attribute>(?:xlink:)?href)="(?P<reference>[^"]+)"',
        replace,
        source,
    )
    if updated != source:
        svg_path.write_text(updated, encoding="utf-8")
    return [
        "SVG contains unresolved external image assets: " + ", ".join(sorted(set(unresolved)))
    ] if unresolved else []
