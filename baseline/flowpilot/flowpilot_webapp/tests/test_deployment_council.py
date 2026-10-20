from flowpilot_webapp.backend import deployment


def test_nested_council_changes_invalidate_loaded_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(deployment, "ROOT", tmp_path)
    path = tmp_path / "flora_translate/engine/council_v4/scientific.py"
    path.parent.mkdir(parents=True)
    path.write_text("POLICY = 1\n")
    before = deployment.source_fingerprint()
    path.write_text("POLICY = 2\n")
    assert deployment.source_fingerprint() != before
