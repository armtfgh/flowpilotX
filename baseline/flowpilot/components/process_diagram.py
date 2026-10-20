"""FlowPilot process-diagram renderer."""

import base64
import html
import json
import re
import streamlit as st
from pathlib import Path


@st.cache_data(show_spinner=False)
def _render_cached_diagnostic(topology_json: str) -> dict:
    """Render an older stored topology that predates diagnostic artifacts."""

    from flora_translate.diagram_artifacts import render_topology_artifacts
    from flora_translate.schemas import ProcessTopology

    topology = ProcessTopology.model_validate(json.loads(topology_json))
    artifacts = render_topology_artifacts(
        topology,
        title="DIAGNOSTIC - NOT EXECUTABLE",
        base_dir=Path("outputs/diagram_runs/gui_diagnostics"),
    )
    return {
        "svg_path": artifacts.get("svg_path", ""),
        "png_path": artifacts.get("png_path", ""),
        "manifest": artifacts.get("manifest", {}),
    }


def render_process_diagram(
    svg_path: str = "",
    png_path: str = "",
    key_prefix: str = "",
    *,
    topology: dict | None = None,
    render_manifest: dict | None = None,
):
    """Render a responsive process diagram with artifact diagnostics.

    key_prefix : unique string so multiple calls in the same page don't
                 produce duplicate widget IDs (required for chat UI).
    """
    svg_exists = svg_path and Path(svg_path).exists()
    png_exists = png_path and Path(png_path).exists()

    if (
        not svg_exists
        and not png_exists
        and topology
        and topology.get("unit_operations")
    ):
        try:
            rendered = _render_cached_diagnostic(
                json.dumps(topology, sort_keys=True, default=str)
            )
            svg_path = rendered.get("svg_path", "")
            png_path = rendered.get("png_path", "")
            svg_exists = bool(svg_path and Path(svg_path).exists())
            png_exists = bool(png_path and Path(png_path).exists())
            if not render_manifest:
                render_manifest = rendered.get("manifest") or {}
        except Exception:
            # The dependency-free HTML topology below remains the last-resort
            # display path if local artifact generation cannot run.
            pass

    uid = key_prefix or str(abs(hash(str(svg_path) + str(png_path))))

    manifest = render_manifest or _load_adjacent_manifest(svg_path, png_path)
    _render_status(manifest)

    # Older Graphviz SVGs reference local icon filenames. They display as
    # blank space when embedded in a browser or moved by autosave. Prefer the
    # complete PNG for those historical artifacts; newly generated SVGs have
    # their images embedded as data URIs.
    prefer_png = bool(
        svg_exists
        and png_exists
        and _svg_has_external_image_references(Path(svg_path))
    )

    fit_to_width = st.toggle("Fit diagram to width", value=True, key=f"fit_{uid}")
    zoom = 100
    if not fit_to_width:
        zoom = st.slider("Diagram zoom (%)", 50, 200, 100, 10, key=f"zoom_{uid}")

    if svg_exists and not prefer_png:
        b64 = base64.b64encode(Path(svg_path).read_bytes()).decode()
        width_style = "width:100%;" if fit_to_width else f"width:{zoom}%;min-width:600px;"
        st.markdown(
            f'<div style="overflow-x:auto; background:white; padding:12px; '
            f'border-radius:8px; border:1px solid #e2e8f0;">'
            f'<img src="data:image/svg+xml;base64,{b64}" alt="FlowPilot process topology" '
            f'style="{width_style} height:auto; display:block; margin:0 auto;"/>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            with open(svg_path, "rb") as f:
                st.download_button(
                    "Download SVG", f,
                    "flowpilot_process.svg", "image/svg+xml",
                    use_container_width=True,
                    key=f"dl_svg_{uid}",
                )
        with col2:
            if png_exists:
                with open(png_path, "rb") as f:
                    st.download_button(
                        "Download PNG", f,
                        "flowpilot_process.png", "image/png",
                        use_container_width=True,
                        key=f"dl_png_{uid}",
                    )
    elif png_exists:
        b64 = base64.b64encode(Path(png_path).read_bytes()).decode()
        width_style = "width:100%;" if fit_to_width else f"width:{zoom}%;min-width:600px;"
        st.markdown(
            f'<div style="overflow-x:auto;background:white;padding:12px;border-radius:8px;'
            f'border:1px solid #e2e8f0;"><img src="data:image/png;base64,{b64}" '
            f'alt="FlowPilot process topology" style="{width_style}height:auto;display:block;'
            f'margin:0 auto;"/></div>',
            unsafe_allow_html=True,
        )
        with open(png_path, "rb") as f:
            st.download_button(
                "Download PNG", f,
                "flowpilot_process.png", "image/png",
                use_container_width=True,
                key=f"dl_png_{uid}",
            )
    elif topology and topology.get("unit_operations"):
        st.warning("Saved image is unavailable. Showing the stored topology as a diagnostic view.")
        labels = [
            html.escape(str(op.get("label") or op.get("op_id") or "Operation"))
            for op in topology.get("unit_operations", [])
        ]
        st.markdown(
            '<div style="overflow-x:auto;white-space:nowrap;background:white;padding:18px;'
            'border:1px solid #e2e8f0;border-radius:8px;">'
            + '<span style="color:#2457a6;padding:0 10px;">→</span>'.join(
                f'<span style="display:inline-block;padding:12px;border:1px solid #315b8a;'
                f'border-radius:6px;background:#f7f9fc;color:#172033;">{label}</span>'
                for label in labels
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("No process diagram or stored topology is available for this result.")


def _svg_has_external_image_references(path: Path) -> bool:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(
        re.search(
            r'(?:xlink:)?href="(?!data:|#|https?://)[^"]+"',
            source,
        )
    )


def _load_adjacent_manifest(svg_path: str, png_path: str) -> dict:
    for value in (svg_path, png_path):
        if not value:
            continue
        candidate = Path(value).parent / "render_manifest.json"
        if candidate.is_file():
            try:
                import json

                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
    return {}


def _render_status(manifest: dict) -> None:
    if not manifest:
        return
    status = str(manifest.get("status") or "unknown")
    renderer = str(manifest.get("renderer") or "unknown")
    topology_hash = str(manifest.get("topology_sha256") or "")
    suffix = f" · topology {topology_hash[:12]}" if topology_hash else ""
    if status == "failed":
        st.error(f"Diagram render failed · {renderer}{suffix}")
    elif manifest.get("warnings"):
        st.caption(f"Diagram: {status} · {renderer}{suffix}")
    else:
        st.caption(f"Diagram: verified · {renderer}{suffix}")
