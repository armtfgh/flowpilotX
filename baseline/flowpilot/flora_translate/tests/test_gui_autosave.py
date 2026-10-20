import json
from pathlib import Path

from flora_translate.gui_autosave import autosave_gui_result


def test_autosave_gui_result_writes_jsons_and_diagrams(tmp_path):
    svg = tmp_path / "source.svg"
    png = tmp_path / "source.png"
    svg.write_text("<svg></svg>")
    png.write_bytes(b"png")
    diagnostic_svg = tmp_path / "diagnostic_source.svg"
    diagnostic_png = tmp_path / "diagnostic_source.png"
    diagnostic_svg.write_text("<svg>diagnostic</svg>")
    diagnostic_png.write_bytes(b"diagnostic-png")

    result = {
        "confidence": "MEDIUM",
        "recommended_disposition": "BLOCK",
        "disposition_rationale": "Inventory conflict.",
        "design_disposition": {
            "hard_failures": [
                {
                    "finding_id": "INVENTORY-TEST",
                    "message": "No compatible reactor.",
                }
            ]
        },
        "svg_path": str(svg),
        "png_path": str(png),
        "diagnostic_svg_path": str(diagnostic_svg),
        "diagnostic_png_path": str(diagnostic_png),
        "diagnostic_topology": {"topology_id": "diagnostic"},
        "diagnostic_diagram_render_manifest": {"renderer": "test"},
        "chemistry_plan": {"reaction_name": "test reaction"},
        "process_requirements_topology": {"topology_id": "requirements"},
        "inventory_allocation": {"status": "incomplete"},
        "instrument_manifest": [{"equipment_id": "pump_1"}],
        "proposal": {
            "residence_time_min": 12.5,
            "flow_rate_mL_min": 0.1,
            "reactor_volume_mL": 1.25,
            "tubing_ID_mm": 1.0,
            "temperature_C": 40,
            "concentration_M": 0.5,
            "BPR_bar": 6,
            "streams": [{"phase": "gas", "gas_flow_sccm": 1.2}],
        },
    }
    intake = {"schema_version": "flowpilot_intake_v1.0", "ready_for_design": True}

    run_dir = autosave_gui_result(
        result,
        intake_package=intake,
        source="test_gui",
        user_input="A to B",
        base_dir=tmp_path / "gui_runs",
    )

    assert (run_dir / "result.json").exists()
    assert (run_dir / "final_design.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "intake_package.json").exists()
    assert (run_dir / "input.txt").read_text() == "A to B"
    assert (run_dir / "process.svg").read_text() == "<svg></svg>"
    assert (run_dir / "process.png").read_bytes() == b"png"
    assert (run_dir / "diagnostic_process.svg").read_text() == "<svg>diagnostic</svg>"
    assert (run_dir / "diagnostic_process.png").read_bytes() == b"diagnostic-png"
    assert json.loads((run_dir / "diagnostic_topology.json").read_text())["topology_id"] == "diagnostic"
    assert (run_dir / "diagnostic_render_manifest.json").exists()
    assert json.loads((run_dir / "inventory_allocation.json").read_text())["status"] == "incomplete"
    assert json.loads((run_dir / "instrument_manifest.json").read_text())[0]["equipment_id"] == "pump_1"
    assert json.loads((run_dir / "process_requirements_topology.json").read_text())["topology_id"] == "requirements"
    assert result["svg_path"] == str((run_dir / "process.svg").resolve())
    assert result["png_path"] == str((run_dir / "process.png").resolve())

    stored = json.loads((run_dir / "result.json").read_text())
    assert stored["svg_path"] == result["svg_path"]
    assert stored["png_path"] == result["png_path"]

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["source"] == "test_gui"
    assert summary["final_design_status"] == "blocked"
    assert summary["residence_time_min"] is None
    assert summary["gas_flow_sccm"] is None
    assert summary["recommended_disposition"] == "BLOCK"
    assert summary["hard_failures"][0]["finding_id"] == "INVENTORY-TEST"


def test_autosave_gui_result_creates_unique_run_dirs(tmp_path):
    result = {"proposal": {}}

    first = autosave_gui_result(result, source="repeat", base_dir=tmp_path)
    second = autosave_gui_result(result, source="repeat", base_dir=tmp_path)

    assert first != second
    assert first.exists()
    assert second.exists()


def test_autosaved_diagram_is_not_changed_when_source_is_overwritten(tmp_path):
    source = tmp_path / "shared.svg"
    source.write_text("<svg>first</svg>")
    result = {"proposal": {}, "svg_path": str(source), "png_path": ""}

    run_dir = autosave_gui_result(result, source="immutable", base_dir=tmp_path / "runs")
    source.write_text("<svg>second</svg>")

    assert Path(result["svg_path"]).read_text() == "<svg>first</svg>"
    assert (run_dir / "process.svg").read_text() == "<svg>first</svg>"
