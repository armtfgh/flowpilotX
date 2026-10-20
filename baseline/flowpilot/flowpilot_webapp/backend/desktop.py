"""Launch FlowPilot as a local desktop application."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

import uvicorn


def _runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait(url: str, timeout: float = 45.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=1.0):
                return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("FlowPilot local server did not start")


def main() -> None:
    root = _runtime_root()
    os.chdir(root)
    port = _port()
    url = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(
        "flowpilot_webapp.backend.app:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait(url)
    try:
        try:
            import webview

            webview.create_window(
                "FlowPilot",
                url,
                width=1460,
                height=940,
                min_size=(1080, 720),
                text_select=True,
            )
            webview.start()
        except Exception:
            webbrowser.open(url)
            try:
                while thread.is_alive():
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
    finally:
        server.should_exit = True
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
