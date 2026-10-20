from __future__ import annotations

from pathlib import Path

from flora_design.visualizer import flowsheet_builder


def test_portable_windows_graphviz_is_discovered(tmp_path, monkeypatch) -> None:
    executable_dir = tmp_path / "bundle"
    dot = executable_dir / "graphviz" / "bin" / "dot.exe"
    dot.parent.mkdir(parents=True)
    dot.write_bytes(b"placeholder")

    monkeypatch.setattr(flowsheet_builder.shutil, "which", lambda _: None)
    monkeypatch.setattr(flowsheet_builder.sys, "executable", str(executable_dir / "FlowPilot.exe"))
    monkeypatch.setenv("PATH", "")

    assert flowsheet_builder._ensure_graphviz_on_path() == str(dot)
    assert str(dot.parent) in flowsheet_builder.os.environ["PATH"].split(flowsheet_builder.os.pathsep)
