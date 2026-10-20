#!/usr/bin/env python3
"""Offline integrity checks for the prepared handoff, not validation of a new pipeline."""
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]


def verify_source_files(root, manifest, patch_manifest=None):
    """Accept only intact originals or exact, backed-up, documented patches."""
    root = Path(root).resolve()
    expected = {item["destination"]: item["sha256"] for item in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("Duplicate source manifest paths")
    patches = {}
    for item in (patch_manifest or {}).get("files", []):
        path = item["path"]
        if path in patches or path not in expected or not path.startswith("baseline/flowpilot/"):
            raise ValueError(f"Invalid baseline patch path: {path}")
        patches[path] = item

    def digest(relative):
        p = (root / relative).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            return None
        return hashlib.sha256(p.read_bytes()).hexdigest()

    failures = []
    unchanged = 0
    patched = 0
    for path, original_sha in expected.items():
        actual_sha = digest(path)
        patch = patches.get(path)
        if patch:
            if (patch["original_sha256"] != original_sha
                    or digest(patch["backup"]) != original_sha
                    or actual_sha != patch["updated_sha256"]
                    or digest(patch["diff"]) is None):
                failures.append(path)
            else:
                patched += 1
        elif actual_sha == original_sha:
            unchanged += 1
        else:
            failures.append(path)
    if failures:
        raise ValueError(f"Source integrity failures: {failures}")
    return {"unchanged_source_files": unchanged, "documented_patched_files": patched,
            "protected_files_accounted_for": unchanged + patched}


def main():
    manifest = json.loads((ROOT / "provenance/source_manifest.json").read_text())
    patch_paths = sorted((ROOT / "provenance").glob("baseline_*patches_*/manifest.json"))
    patch_manifest = {"files": [entry for path in patch_paths
                                for entry in json.loads(path.read_text())["files"]]}
    integrity = verify_source_files(ROOT, manifest, patch_manifest)
    supplemental_path = ROOT / "provenance/supplemental_flowpilot_modules_2026-10-16_v1/manifest.json"
    supplemental = json.loads(supplemental_path.read_text())["files"] if supplemental_path.exists() else []
    for item in supplemental:
        path = (ROOT / item["destination"]).resolve()
        if (not path.is_relative_to(ROOT) or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]):
            raise ValueError("Supplemental source integrity failure: " + item["destination"])
    integrity["supplemental_files_verified"] = len(supplemental)
    audit = json.loads((ROOT / "data/derived/source_audit.json").read_text())
    counts = {k: v["record_count"] for k, v in audit["files"].items()}
    assert counts == {"homogeneous_csv": 111, "homogeneous_xlsx": 116,
                      "heterogeneous_csv": 42, "heterogeneous_xlsx": 42}, counts
    assert audit["pairs"]["homogeneous"]["xlsx_only_entries"] == ["112", "113", "114", "115", "116"]
    assert all(not p["differences"] for p in audit["pairs"].values())
    assert len(list((ROOT / "review/slides").glob("slide-*.png"))) == 7
    # Dependency installs may contain legitimate node_modules/.bin links; audit
    # the protected source tree paths rather than generated dependency products.
    baseline_root = ROOT / "baseline"
    protected_paths = {p for entry in manifest["files"]
                       for p in [ROOT / entry["destination"], *(ROOT / entry["destination"]).parents]
                       if p.is_relative_to(baseline_root)}
    assert not any(p.is_symlink() for p in protected_paths), "Protected baseline must not link to original workspace"
    db = ROOT / "baseline/flowpilot/flora_translate/data/chroma_db/chroma.sqlite3"
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    config = json.loads((ROOT / "configs/study_draft.json").read_text())
    assert not config["live_model_calls_enabled"]
    assert config["flow_planning"]["residence_time_values_min"] is None
    print(json.dumps({"status": "passed", **integrity,
                      "record_counts": counts, "slide_renders": 7,
                      "scope": "Handoff integrity and source reconciliation only; no pipeline execution"}, indent=2))


if __name__ == "__main__":
    main()
