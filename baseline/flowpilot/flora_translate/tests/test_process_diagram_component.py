import json

from streamlit.testing.v1 import AppTest


def _diagram_app(svg_path: str, png_path: str, manifest_path: str) -> None:
    import json

    from components.process_diagram import render_process_diagram

    render_process_diagram(
        svg_path,
        png_path,
        key_prefix="component_test",
        topology={},
        render_manifest=json.loads(open(manifest_path).read()),
    )


def _diagnostic_topology_app() -> None:
    from components.process_diagram import render_process_diagram

    render_process_diagram(
        "",
        "",
        key_prefix="diagnostic_component_test",
        topology={
            "topology_id": "blocked_requirements",
            "unit_operations": [
                {
                    "op_id": "reactor_1",
                    "op_type": "coil_reactor",
                    "label": "Unresolved reactor",
                    "parameters": {"volume_mL": 3.0},
                    "assignment_status": "unresolved",
                }
            ],
            "streams": [],
        },
    )


def test_diagram_component_exposes_fit_and_zoom_controls(tmp_path):
    svg = tmp_path / "process.svg"
    png = tmp_path / "process.png"
    manifest = tmp_path / "render_manifest.json"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
    png.write_bytes(b"png")
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "renderer": "test",
                "topology_sha256": "a" * 64,
                "warnings": [],
            }
        )
    )

    app = AppTest.from_function(
        _diagram_app,
        args=(str(svg), str(png), str(manifest)),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert app.toggle[0].label == "Fit diagram to width"
    app.toggle[0].set_value(False).run()
    assert not app.exception
    assert app.slider[0].label == "Diagram zoom (%)"


def test_diagram_component_renders_stored_diagnostic_topology():
    app = AppTest.from_function(
        _diagnostic_topology_app,
        default_timeout=15,
    ).run()

    assert not app.exception
    assert app.toggle[0].label == "Fit diagram to width"
    assert any("data:image/svg+xml" in item.value for item in app.markdown)


def test_diagram_component_prefers_png_for_legacy_external_asset_svg(tmp_path):
    svg = tmp_path / "legacy.svg"
    png = tmp_path / "complete.png"
    manifest = tmp_path / "render_manifest.json"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink">'
        '<image xlink:href="missing-icon.png"/></svg>'
    )
    png.write_bytes(b"complete-png")
    manifest.write_text(
        json.dumps({"status": "complete", "renderer": "graphviz", "warnings": []})
    )

    app = AppTest.from_function(
        _diagram_app,
        args=(str(svg), str(png), str(manifest)),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert any("data:image/png;base64" in item.value for item in app.markdown)
