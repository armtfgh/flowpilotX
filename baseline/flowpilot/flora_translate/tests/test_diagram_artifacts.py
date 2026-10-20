import json
from pathlib import Path

from flora_translate.diagram_artifacts import (
    _inline_svg_images,
    render_topology_artifacts,
    topology_sha256,
)
from flora_translate.schemas import ProcessTopology


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "diagram_stage0_topologies.json"


def _fixture(name: str) -> ProcessTopology:
    fixtures = json.loads(FIXTURE_PATH.read_text())
    payload = next(item["topology"] for item in fixtures if item["name"] == name)
    return ProcessTopology.model_validate(payload)


def test_render_is_run_local_and_does_not_mutate_topology(tmp_path):
    topology = _fixture("gas_liquid_photochemistry")
    before = topology.model_dump_json()

    first = render_topology_artifacts(topology, title="First", base_dir=tmp_path)
    second = render_topology_artifacts(topology, title="Second", base_dir=tmp_path)

    assert first["run_dir"] != second["run_dir"]
    assert first["svg_path"] != second["svg_path"]
    assert Path(first["svg_path"]).is_file()
    assert Path(second["svg_path"]).is_file()
    assert topology.model_dump_json() == before
    assert first["topology_sha256"] == topology_sha256(topology)

    manifest = json.loads(Path(first["render_manifest_path"]).read_text())
    stored_topology = ProcessTopology.model_validate_json(
        Path(first["topology_path"]).read_text()
    )
    assert manifest["topology_sha256"] == topology_sha256(stored_topology)


def test_final_svg_fallback_survives_renderer_failure(tmp_path):
    class BrokenBuilder:
        def build(self, *args, **kwargs):
            raise RuntimeError("renderer unavailable")

    artifacts = render_topology_artifacts(
        _fixture("single_stage_liquid"),
        title="Fallback",
        base_dir=tmp_path,
        builder=BrokenBuilder(),
    )

    assert artifacts["renderer"] == "deterministic_svg_fallback"
    assert artifacts["render_status"] in {"complete", "partial"}
    assert Path(artifacts["svg_path"]).read_text().startswith("<svg")
    assert "Primary renderer failed" in " ".join(artifacts["warnings"])


def test_twenty_repeated_renders_have_unique_verified_artifacts(tmp_path):
    topology = _fixture("two_stage_no_degasser")
    artifacts = [
        render_topology_artifacts(topology, title=f"Repeat {index}", base_dir=tmp_path)
        for index in range(20)
    ]

    assert len({item["run_dir"] for item in artifacts}) == 20
    assert len({item["svg_path"] for item in artifacts}) == 20
    assert all(Path(item["svg_path"]).is_file() for item in artifacts)
    assert all(item["topology_sha256"] == topology_sha256(topology) for item in artifacts)


def test_svg_image_assets_are_embedded_for_browser_and_autosave(tmp_path):
    icon = tmp_path / "icon.png"
    icon.write_bytes(b"png-bytes")
    svg = tmp_path / "process.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink">'
        '<image xlink:href="icon.png"/></svg>'
    )

    warnings = _inline_svg_images(svg)

    assert warnings == []
    source = svg.read_text()
    assert 'xlink:href="data:image/png;base64,' in source
    assert 'xlink:href="icon.png"' not in source
