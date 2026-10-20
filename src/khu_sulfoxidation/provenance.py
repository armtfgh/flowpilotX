"""Canonical hashes and immutable output snapshots. No external services."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Exclusive creation: a revision must use a new filename/directory."""
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_snapshot(root):
    root = Path(root)
    files = []
    for folder in ("src/khu_sulfoxidation", "configs", "tests", "scripts", "docs", "data/raw/correspondence"):
        for p in sorted((root / folder).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".json", ".md"}:
                files.append({"path": str(p.relative_to(root)), "sha256": file_hash(p)})
    for name in ("pyproject.toml", "requirements-study.txt", "README.md", "AGENTS.md", "provenance/source_manifest.json", "khu_inventory.json"):
        if (root / name).is_file():
            files.append({"path": name, "sha256": file_hash(root / name)})
    return {"files": files, "tree_sha256": digest(files), "code_version_basis": "content hashes; workspace has no study commit yet"}
