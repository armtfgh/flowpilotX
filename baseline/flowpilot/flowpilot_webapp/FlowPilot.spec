# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir bundle for FlowPilot.

Run this spec on Windows. PyInstaller does not cross-compile Windows binaries
from Linux. The onedir layout is intentional: ChromaDB and scientific Python
dependencies are materially more reliable than in a temporary onefile bundle.
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


ROOT = Path(SPEC).resolve().parents[1]

datas = [
    (str(ROOT / "flowpilot_webapp" / "frontend" / "dist"), "flowpilot_webapp/frontend/dist"),
    (str(ROOT / "flora_translate" / "prompts"), "flora_translate/prompts"),
    (str(ROOT / "flora_translate" / "data"), "flora_translate/data"),
    (str(ROOT / "flora_design" / "visualizer" / "icons"), "flora_design/visualizer/icons"),
]

graphviz_root = os.environ.get("FLOWPILOT_GRAPHVIZ_ROOT")
if graphviz_root and Path(graphviz_root).is_dir():
    datas.append((graphviz_root, "graphviz"))

binaries = []
hiddenimports = collect_submodules("flora_translate")
hiddenimports += collect_submodules("flora_design")
hiddenimports += [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "webview.platforms.edgechromium",
]

for package in ("chromadb", "onnxruntime", "scipy", "sklearn", "pandas"):
    try:
        package_datas, package_binaries, package_hidden = collect_all(package)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hidden
    except Exception:
        pass

a = Analysis(
    [str(ROOT / "flowpilot_webapp" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["streamlit"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FlowPilot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory=".",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FlowPilot",
)
