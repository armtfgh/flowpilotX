#!/usr/bin/env python3
"""Run the synthetic public-clone suite and a dependency-free source smoke.

The explicitly named suites use synthetic inputs and mocked model transports. Local
loopback HTTP is allowed for GUI tests; external sockets are blocked. Backbone
source and icon checks do not import its optional provider/vector-store stack,
and must not be described as validating a live end-to-end FlowPilot run.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from importlib import import_module, resources
import ipaddress
import json
import math
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SUITES = (
    "test_flowpilotx_core",
    "test_flowpilotx_gateway",
    "test_flowpilotx_portable",
    "test_flowpilotx_review_import",
    "test_handoff_integrity",
)
BACKBONE_COMPONENTS = (
    "flora_scout/scout.py",
    "flora_prism/prism.py",
    "paper_knowledge_extractor.py",
    "visualization/benchmark_figure_studio.py",
    "flora_translate/main.py",
    "flora_translate/input_parser.py",
    "flora_translate/embedding_engine.py",
    "flora_translate/vector_store.py",
    "flora_translate/retriever.py",
    "flora_translate/analogy_selector.py",
    "flora_translate/design_calculator.py",
    "flora_translate/design_realizer.py",
    "flora_translate/engine/council_v4/scientific.py",
    "flora_design/visualizer/flowsheet_builder.py",
    "flora_fundamentals/knowledge_store.py",
    "flora_optimize/bo_engine.py",
    "flowpilot_webapp/backend/app.py",
)


@contextmanager
def offline_environment():
    """Disable model routes and external networking without breaking local GUI tests."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo
    original_sendto = socket.socket.sendto

    def check_host(host):
        if isinstance(host, bytes):
            host = host.decode("ascii")
        if host == "localhost":
            return
        try:
            permitted = ipaddress.ip_address(host).is_loopback
        except (ValueError, TypeError):
            permitted = False
        if not permitted:
            raise RuntimeError("Public offline tests cannot contact external hosts")

    def check_address(address):
        if not isinstance(address, tuple) or not address:
            raise RuntimeError("Public offline tests permit only loopback IP sockets")
        check_host(address[0])

    def connect(sock, address):
        check_address(address)
        return original_connect(sock, address)

    def connect_ex(sock, address):
        check_address(address)
        return original_connect_ex(sock, address)

    def getaddrinfo(host, *args, **kwargs):
        check_host(host)
        return original_getaddrinfo(host, *args, **kwargs)

    def sendto(sock, data, *args):
        check_address(args[-1])
        return original_sendto(sock, data, *args)

    environment = {
        key: value for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in
                   ("API_KEY", "ACCESS_TOKEN", "AUTH_TOKEN", "SECRET", "PASSWORD", "PROXY"))
    }
    for key in list(environment):
        if key.startswith("FLOWPILOTX_"):
            environment.pop(key)
    environment.update({"FLOWPILOTX_ENABLE_QWEN": "0", "FLOWPILOTX_QWEN_MAX_CALLS": "0",
                        "PYTHONDONTWRITEBYTECODE": "1", "PYTHON_DOTENV_DISABLED": "1",
                        "ANONYMIZED_TELEMETRY": "False"})
    with patch.dict(os.environ, environment, clear=True), \
            patch.object(socket.socket, "connect", connect), \
            patch.object(socket.socket, "connect_ex", connect_ex), \
            patch.object(socket.socket, "sendto", sendto), \
            patch.object(socket, "getaddrinfo", getaddrinfo):
        yield


def smoke(installed: bool) -> dict:
    """Check shipped packages/resources, a synthetic adapter case, and source completeness."""
    package_locations = {}
    for name in ("flowpilotx", "khu_sulfoxidation"):
        module = import_module(name)
        location = Path(module.__file__).resolve()
        if installed and location.is_relative_to(ROOT / "src"):
            raise RuntimeError(f"--installed resolved {name} to source; install the distribution first")
        package_locations[name] = str(location)
        for asset in ("index.html", "app.js", "style.css"):
            if not resources.files(name).joinpath("gui", asset).read_bytes():
                raise RuntimeError(f"Empty packaged GUI asset: {name}/gui/{asset}")

    from khu_sulfoxidation.cli import main as study_main
    from khu_sulfoxidation.inventory import absolute_pressure
    from khu_sulfoxidation.materials import feed_preparation
    from khu_sulfoxidation.observations import AnalyticalOutcome

    if not callable(study_main):
        raise RuntimeError("Missing khu-screen entry point")
    outcome = AnalyticalOutcome(residual_sulfide_pct=3, sulfoxide_pct=104, sulfone_pct=None)
    if outcome.sulfoxide_pct != 104 or outcome.component_sum_pct is not None:
        raise RuntimeError("Analytical smoke lost a reported value or invented missing data")
    if not math.isclose(absolute_pressure(2, "bar", "gauge", 1), 3):
        raise RuntimeError("Explicit gauge-pressure conversion failed")
    recipe = feed_preparation(10, 0.1, 2)
    if recipe["substrate_mmol"] != 1 or recipe["substrate_mass_mg"] is not None:
        raise RuntimeError("Synthetic feed smoke failed or invented missing physical properties")

    backbone = ROOT / "baseline" / "flowpilot"
    for relative in BACKBONE_COMPONENTS:
        if not (backbone / relative).is_file():
            raise RuntimeError(f"Restored backbone component missing: {relative}")
    source_files = sorted((ROOT / "src").rglob("*.py")) + sorted(backbone.rglob("*.py"))
    for path in source_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path.relative_to(ROOT)))
    icon_dir = backbone / "flora_design" / "visualizer" / "icons"
    for name in ("bpr.png", "check_valve.png", "photoreactor.png", "mfc.png"):
        if not (icon_dir / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(f"Missing or invalid topology icon: {name}")
    return {"status": "passed", "packages": package_locations,
            "gui_assets_checked": 6, "python_sources_parsed": len(source_files),
            "backbone_components_present": len(BACKBONE_COMPONENTS),
            "scope": "Synthetic adapter/component and source-presence checks; no private evidence, live model, embeddings, or baseline runtime execution."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true",
                        help="Check installed packages and assets instead of adding src to the import path")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if not args.installed:
        sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tests"))
    with offline_environment():
        print(json.dumps(smoke(args.installed), indent=2), flush=True)
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name)
                                   for name in PUBLIC_SUITES)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
