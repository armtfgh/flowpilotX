"""Loopback-only FlowPilotX GUI and immutable offline run archives."""
from __future__ import annotations

import argparse
import base64
import binascii
from copy import deepcopy
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
from pathlib import Path
import re
import threading
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from .gui_data import MAX_UPLOAD_BYTES, builtin_history, merge_history, parse_history_upload, template_csv
from .provenance import digest, file_hash, write_json

MAX_BODY_BYTES = 12 * 1024 * 1024
SAFE_ID = re.compile(r"[a-z]+-[0-9TZ]+-[a-f0-9]{12}\Z")
ICONS = {"_pump_generic.png", "mfc.png", "check_valve.png", "mixer2.png", "photoreactor.png", "bpr.png", "vial.png"}
DEFAULT_REQUEST = {
    "requested_count": 6, "include_repeat": True,
    "protocol_text": "Selective oxidation of thioanisole to sulfoxide using acridine (3 mol%) in EtOH, 0.25 M, air and blue light. Batch evidence is not flow kinetics.",
    "research_notes": "Compare nominal gas-liquid transit and inlet oxygen supply; measure residual sulfide, sulfoxide and sulfone separately.",
    "chemistry": {"substrate": "thioanisole", "solvent": "EtOH", "concentration_M": 0.25,
                  "catalyst": "acridine", "catalyst_mol_percent": 3.0},
    "constraints": {"time_min": 2, "time_max": 10, "oxygen_min": 0.5, "oxygen_max": 1.0,
                    "coil_mL": 5, "pressure_bar_gauge": 1.5, "temperature_C": 25},
    "history_ids": ["khu_builtin"],
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix):
    return prefix + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    # Spreadsheet exports must not turn uploaded prose into executable formulas.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def result_csv(response):
    columns = ["id", "candidate_id", "role", "repeat_of", "target_time_min", "target_oxygen_eq",
        "liquid_mL_min", "air_STP_mL_min", "oxygen_STP_mL_min", "oxygen_eq", "transit_min",
        "rationale", "evidence_ids", "actual", "status"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    rows = [dict(r, role="core") for r in response.get("conditions", [])]
    repeat = response.get("repeat")
    if isinstance(repeat, dict):
        rows.append(dict(repeat, role="independent_repeat"))
    for row in rows:
        writer.writerow({k: _csv_value(row.get(k, response.get("status") if k == "status" else None)) for k in columns})
    return output.getvalue()


class GuiApplication:
    def __init__(self, project_root, *, storage_root=None, generator=None):
        self.root = Path(project_root).resolve()
        self.storage = Path(storage_root).resolve() if storage_root else self.root / "outputs/flowpilotx_gui_sessions"
        self.static = self.root / "src/khu_sulfoxidation/gui"
        self.icons = self.root / "outputs/khu_corrected_screen_2026-10-13_v3/assets/icons"
        self.generator = generator
        self._builtin = None
        self._lock = threading.RLock()

    def builtin(self):
        with self._lock:
            if self._builtin is None:
                self._builtin = builtin_history(self.root)
            return self._builtin

    def imports(self):
        result = []
        for path in sorted((self.storage / "imports").glob("*/result.json")):
            data = json.loads(path.read_text())
            result.append({"id": data["id"], "name": data["sources"][0]["name"], "summary": data["summary"]})
        return result

    def bootstrap(self):
        data = self.builtin()
        package = self.root / "outputs/khu_corrected_screen_2026-10-13_v3"
        design = json.loads((package / "corrected_design.json").read_text())
        ledger = json.loads((package / "history_hypotheses.json").read_text())
        return {"application": "FlowPilotX", "mode": "local_offline", "default_request": deepcopy(DEFAULT_REQUEST),
            "history": {"id": "khu_builtin", "summary": data["summary"],
                "rows": [r for r in data["rows"] if r["catalyst_system"] == "homogeneous"],
                "heterogeneous_rows": [r for r in data["rows"] if r["catalyst_system"] == "heterogeneous"],
                "sources": data["sources"]},
            "evidence_insights": ledger["report_summary"],
            "inventory": {"scenario": design["scenario"], "review": design["inventory_review"],
                "source_hashes": design["source_hashes"]},
            "model_status": {"new_model_calls": 0, "paid_calls_enabled": False,
                "current_generator": "deterministic coupled-flow screen; history is supplied context, not fitted kinetics",
                "archived_campaign": "14 completed Astra/Sol calls; original CP recommendations withdrawn; no current model ranking",
                "future_feedback": "Qualitative review only; an autonomous adaptive optimizer is not implemented"},
            "imports": self.imports()}

    def resolve_history(self, ids):
        if not isinstance(ids, list) or len(ids) > 30 or not all(isinstance(i, str) for i in ids):
            raise ValueError("history_ids must be a list of at most 30 local dataset IDs")
        datasets = []
        for ident in dict.fromkeys(ids):
            if ident == "khu_builtin":
                datasets.append(self.builtin())
            elif SAFE_ID.fullmatch(ident) and ident.startswith("import-"):
                path = self.storage / "imports" / ident / "result.json"
                if not path.is_file():
                    raise ValueError("Unknown local history ID")
                datasets.append(json.loads(path.read_text()))
            else:
                raise ValueError("Unknown local history ID")
        return merge_history(datasets)

    def import_history(self, request):
        if not isinstance(request, dict) or set(request) != {"name", "content_base64"}:
            raise ValueError("Import requires only name and content_base64")
        if not isinstance(request["content_base64"], str):
            raise ValueError("content_base64 must be a base64 string")
        try:
            raw = base64.b64decode(request["content_base64"], validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("Invalid base64 upload") from None
        if not raw or len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError("Upload must contain 1 byte to 8 MiB")
        ident = _new_id("import")
        directory = self.storage / "imports" / ident
        directory.mkdir(parents=True, exist_ok=False)
        with (directory / "source.bin").open("xb") as stream:
            stream.write(raw)
        write_json(directory / "request.json", {"name": request["name"], "source_sha256": file_hash(directory / "source.bin"), "created_at": _now()})
        try:
            result = parse_history_upload(request["name"], raw, known_sources=self.builtin()["known_sources"])
        except Exception as exc:
            write_json(directory / "error.json", {"error": str(exc), "status": "rejected", "source_preserved": True})
            raise ValueError(str(exc)) from None
        result.update(id=ident, created_at=_now(), archive_path=str(directory.relative_to(self.root)) if directory.is_relative_to(self.root) else str(directory))
        write_json(directory / "result.json", result)
        # All imported exports can be selected together without double counting.
        all_ids = [r["id"] for r in self.imports()]
        return {**result, "combined_summary": self.resolve_history(all_ids)["summary"]}

    def generate(self, user_request):
        if not isinstance(user_request, dict):
            raise ValueError("Design request must be a JSON object")
        ident = _new_id("run")
        directory = self.storage / "runs" / ident
        directory.mkdir(parents=True, exist_ok=False)
        created = _now()
        write_json(directory / "request.json", user_request)
        metadata = {"schema_version": "flowpilotx-gui-run-v1", "run_id": ident, "created_at": created,
                    "user_request_sha256": digest(user_request), "paid_model_calls": 0,
                    "adapter_sources_sha256": {str(p.relative_to(self.root)): file_hash(p)
                        for p in [self.root / "src/khu_sulfoxidation/gui_server.py",
                                  self.root / "src/khu_sulfoxidation/gui_data.py",
                                  self.root / "src/khu_sulfoxidation/gui_selection.py",
                                  self.static / "index.html", self.static / "app.js",
                                  self.static / "style.css"] if p.is_file()}}
        try:
            history = self.resolve_history(user_request.get("history_ids", ["khu_builtin"]))
            request = deepcopy(user_request)
            request["history_records"] = [r for r in history["rows"] if r["catalyst_system"] == "homogeneous"]
            request["history_summary"] = history["summary"]
            write_json(directory / "engine_request.json", request)
            metadata["engine_request_sha256"] = digest(request)
            if self.generator is None:
                from .gui_selection import generate_gui_designs
                generator = generate_gui_designs
            else:
                generator = self.generator
            result = generator(request, self.root)
            if not isinstance(result, dict):
                raise ValueError("Design engine returned an invalid result")
            response = {**result, "run_id": ident, "created_at": created,
                "request": deepcopy(user_request),
                "history_summary": history["summary"], "downloads": {"json": f"/api/export/{ident}.json", "csv": f"/api/export/{ident}.csv"},
                "archive_path": str(directory.relative_to(self.root)) if directory.is_relative_to(self.root) else str(directory)}
            write_json(directory / "response.json", response)
            with (directory / "conditions.csv").open("x", encoding="utf-8", newline="") as stream:
                stream.write(result_csv(response))
            metadata.update(status=response.get("status", "conditional"), requested_count=user_request.get("requested_count"),
                condition_count=len(response.get("conditions", [])), response_sha256=digest(response),
                files_sha256={p.name: file_hash(p) for p in directory.iterdir() if p.is_file()})
            write_json(directory / "manifest.json", metadata)
            return response
        except Exception as exc:
            error = {"error": str(exc), "run_id": ident, "status": "rejected", "details": getattr(exc, "details", None)}
            write_json(directory / "error.json", error)
            metadata.update(status="rejected", requested_count=user_request.get("requested_count"), condition_count=0,
                            files_sha256={p.name: file_hash(p) for p in directory.iterdir() if p.is_file()})
            write_json(directory / "manifest.json", metadata)
            raise ArchivedRequestError(error) from exc

    def runs(self):
        return {"runs": [json.loads(path.read_text()) for path in sorted((self.storage / "runs").glob("*/manifest.json"), reverse=True)]}

    def export(self, run_id, suffix):
        if not SAFE_ID.fullmatch(run_id) or not run_id.startswith("run-") or suffix not in {"json", "csv"}:
            raise ValueError("Invalid export ID")
        directory = self.storage / "runs" / run_id
        path = directory / ("response.json" if suffix == "json" else "conditions.csv")
        if suffix == "json":
            if not path.is_file():
                path = directory / "error.json"
            if not path.is_file():
                raise FileNotFoundError("Export not found")
            # Older successful runs and rejected attempts have separate inputs.
            # Assemble their read-only export without rewriting the archive.
            payload = json.loads(path.read_text())
            if "request" not in payload:
                payload["request"] = json.loads((directory / "request.json").read_text())
            if "created_at" not in payload:
                payload["created_at"] = json.loads((directory / "manifest.json").read_text())["created_at"]
            engine_path = directory / "engine_request.json"
            if engine_path.is_file():
                engine_request = json.loads(engine_path.read_text())
                payload["history_context"] = {key: engine_request[key]
                    for key in ("history_records", "history_summary", "history_ids") if key in engine_request}
            else:
                payload["history_context"] = None
            return json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2).encode()
        if not path.is_file():
            raise FileNotFoundError("Export not found")
        return path.read_bytes()


class ArchivedRequestError(ValueError):
    def __init__(self, payload):
        super().__init__(payload["error"])
        self.payload = payload


class GuiHandler(BaseHTTPRequestHandler):
    server_version = "FlowPilotXLocal/1"

    @property
    def app(self):
        return self.server.application

    def log_message(self, format, *args):
        # Avoid logging protocol text, uploaded content or request headers.
        pass

    def _trusted(self):
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        return host in allowed and (origin is None or origin == "http://" + host) and self.headers.get("Sec-Fetch-Site") != "cross-site"

    def _reply(self, code, body, content_type="application/json; charset=utf-8", filename=None):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        if filename:
            self.send_header("Content-Disposition", 'attachment; filename="' + filename + '"')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._trusted():
            return self._reply(403, {"error": "Only same-origin local requests are accepted"})
        path = unquote(urlsplit(self.path).path)
        try:
            if path == "/api/bootstrap":
                return self._reply(200, self.app.bootstrap())
            if path == "/api/runs":
                return self._reply(200, self.app.runs())
            if path == "/api/history/template.csv":
                return self._reply(200, template_csv().encode(), "text/csv; charset=utf-8", "flowpilotx_history_template.csv")
            if path.startswith("/api/export/"):
                stem, sep, suffix = path.removeprefix("/api/export/").rpartition(".")
                if not sep:
                    raise ValueError("Export needs .json or .csv")
                return self._reply(200, self.app.export(stem, suffix), "application/json" if suffix == "json" else "text/csv; charset=utf-8", stem + "." + suffix)
            if path.startswith("/assets/icons/"):
                name = path.removeprefix("/assets/icons/")
                if name not in ICONS:
                    raise FileNotFoundError("Icon not found")
                file = self.app.icons / name
            elif path in {"/", "/index.html", "/app.js", "/style.css"}:
                file = self.app.static / ("index.html" if path == "/" else path[1:])
            else:
                raise FileNotFoundError("Route not found")
            return self._reply(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
        except FileNotFoundError:
            return self._reply(404, {"error": "Local resource not found"})
        except ValueError as exc:
            return self._reply(400, {"error": str(exc)})

    def do_POST(self):
        if not self._trusted():
            return self._reply(403, {"error": "Only same-origin local requests are accepted"})
        if self.headers.get_content_type() != "application/json":
            return self._reply(415, {"error": "Use application/json"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                return self._reply(413, {"error": "JSON request exceeds the local 12 MiB limit or is empty"})
            request = json.loads(self.rfile.read(length), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))
            path = urlsplit(self.path).path
            if path == "/api/generate":
                return self._reply(200, self.app.generate(request))
            if path == "/api/history/import":
                return self._reply(200, self.app.import_history(request))
            if path == "/api/history/resolve":
                if not isinstance(request, dict) or set(request) != {"history_ids"}:
                    raise ValueError("History resolution requires only history_ids")
                return self._reply(200, self.app.resolve_history(request["history_ids"]))
            return self._reply(404, {"error": "API route not found"})
        except ArchivedRequestError as exc:
            return self._reply(400, exc.payload)
        except (ValueError, UnicodeError) as exc:
            return self._reply(400, {"error": str(exc)})


def create_server(project_root, *, port=8787, storage_root=None, generator=None):
    server = ThreadingHTTPServer(("127.0.0.1", port), GuiHandler)
    server.daemon_threads = True
    server.application = GuiApplication(project_root, storage_root=storage_root, generator=generator)
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run FlowPilotX locally; offline, loopback only")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    server = create_server(args.project_root, port=args.port)
    print(f"FlowPilotX: http://127.0.0.1:{server.server_address[1]} (offline; Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
