import base64
from copy import deepcopy
import csv
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest

from khu_sulfoxidation.gui_server import ArchivedRequestError, DEFAULT_REQUEST, GuiApplication, create_server, result_csv

ROOT = Path(__file__).resolve().parents[1]


def fake_generator(request, root):
    return {"status": "conditional", "conditions": [{"id": "FPX01", "candidate_id": "x", "liquid_mL_min": .2,
        "rationale": "local only", "actual": {"sulfoxide_pct": None}}], "repeat": None,
        "candidate_pool": [{"id": "x", "selection_disposition": "selected"}, {"id": "y", "rejection_reasons": ["outside grid"]}],
        "configuration": request["constraints"], "configuration_sha256": "test", "paid_model_calls": 0}


class GuiServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = GuiApplication(ROOT, storage_root=self.tmp.name, generator=fake_generator)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bootstrap_builtin_history_and_model_status(self):
        b = self.app.bootstrap()
        self.assertEqual(len(b["history"]["rows"]), 116)
        self.assertEqual(len(b["history"]["heterogeneous_rows"]), 42)
        self.assertEqual(b["history"]["summary"]["source_count"], 4)
        self.assertFalse(b["model_status"]["paid_calls_enabled"])

    def test_request_versions_and_export_preserve_rejections(self):
        req = deepcopy(DEFAULT_REQUEST)
        a, b = self.app.generate(req), self.app.generate(req)
        self.assertNotEqual(a["run_id"], b["run_id"])
        exported = json.loads(self.app.export(a["run_id"], "json"))
        self.assertEqual(exported["candidate_pool"][1]["rejection_reasons"], ["outside grid"])
        self.assertEqual(exported["request"], req)
        self.assertEqual(a["request"], req)
        self.assertIsNot(a["request"], req)
        directory = Path(self.tmp.name) / "runs" / a["run_id"]
        self.assertEqual(json.loads((directory / "request.json").read_text()), req)
        engine = json.loads((directory / "engine_request.json").read_text())
        self.assertEqual(len(engine["history_records"]), 116)
        self.assertEqual(exported["history_context"], {k: engine[k] for k in ("history_records", "history_summary", "history_ids")})
        self.assertEqual(sum(len(r["source_representations"]) for r in exported["history_context"]["history_records"]), 227)
        self.assertNotIn("history_context", json.loads((directory / "response.json").read_text()))
        self.assertEqual(len(self.app.runs()["runs"]), 2)
        self.assertIn("response.json", json.loads((directory / "manifest.json").read_text())["files_sha256"])

    def test_failed_request_archived_with_details(self):
        class Failure(ValueError):
            details = {"candidate_pool": [{"rejection_reasons": ["infeasible"]}]}
        self.app.generator = lambda *args: (_ for _ in ()).throw(Failure("outside feasible range"))
        with self.assertRaises(ArchivedRequestError) as ctx:
            self.app.generate(DEFAULT_REQUEST)
        run = ctx.exception.payload["run_id"]
        error = json.loads((Path(self.tmp.name) / "runs" / run / "error.json").read_text())
        self.assertEqual(error["details"], Failure.details)
        self.assertEqual(self.app.runs()["runs"][0]["status"], "rejected")
        directory = Path(self.tmp.name) / "runs" / run
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        exported = json.loads(self.app.export(run, "json"))
        self.assertEqual(exported["request"], DEFAULT_REQUEST)
        self.assertEqual(exported["details"], Failure.details)
        self.assertEqual(len(exported["history_context"]["history_records"]), 116)
        self.assertEqual(exported["created_at"], self.app.runs()["runs"][0]["created_at"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})
        with self.assertRaises(FileNotFoundError):
            self.app.export(run, "csv")

    def test_legacy_saved_run_export_recovers_separate_request_without_writes(self):
        run = "run-20261014T000000Z-aaaaaaaaaaaa"
        directory = Path(self.tmp.name) / "runs" / run
        directory.mkdir(parents=True)
        files = {"response.json": {"run_id": run, "conditions": [], "status": "conditional"},
            "request.json": {**DEFAULT_REQUEST, "research_notes": "original saved note"},
            "manifest.json": {"created_at": "2026-10-14T00:00:00+00:00"}}
        for name, value in files.items():
            (directory / name).write_text(json.dumps(value))
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        exported = json.loads(self.app.export(run, "json"))
        self.assertEqual(exported["request"], files["request.json"])
        self.assertEqual(exported["created_at"], files["manifest.json"]["created_at"])
        self.assertIsNone(exported["history_context"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_upload_persisted_and_lookup_survives_restart(self):
        raw = b"dataset_id,catalyst_system,source_entry,experiment_mode,sulfoxide_pct\ncustom,homogeneous,1,batch,102\n"
        imported = self.app.import_history({"name": "../../private.csv", "content_base64": base64.b64encode(raw).decode()})
        self.assertEqual((Path(self.tmp.name) / "imports" / imported["id"] / "source.bin").read_bytes(), raw)
        restarted = GuiApplication(ROOT, storage_root=self.tmp.name, generator=fake_generator)
        result = restarted.generate({**DEFAULT_REQUEST, "history_ids": [imported["id"], imported["id"]]})
        self.assertEqual(result["history_summary"]["canonical_count"], 1)
        self.assertEqual(result["history_summary"]["above_100_count"], 1)

    def test_failed_import_retains_original_bytes(self):
        with self.assertRaises(ValueError):
            self.app.import_history({"name": "bad.xlsx", "content_base64": base64.b64encode(b"invalid workbook").decode()})
        directories = list((Path(self.tmp.name) / "imports").iterdir())
        self.assertEqual((directories[0] / "source.bin").read_bytes(), b"invalid workbook")
        self.assertTrue((directories[0] / "error.json").is_file())

    def test_csv_missing_actuals_repeat_and_formula_escape(self):
        data = result_csv({"status": "conditional", "conditions": [{"id": "x", "rationale": "=1+1"}],
            "repeat": {"id": "r", "repeat_of": "x"}})
        rows = list(csv.DictReader(io.StringIO(data)))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["transit_min"], "")
        self.assertEqual(rows[0]["rationale"], "'=1+1")
        self.assertEqual(rows[1]["role"], "independent_repeat")

    def test_http_local_origin_routes_and_path_guards(self):
        server = create_server(ROOT, port=0, storage_root=self.tmp.name, generator=fake_generator)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        def call(method, path, data=None, headers=None):
            con = http.client.HTTPConnection("127.0.0.1", port)
            body = json.dumps(data) if data is not None else None
            con.request(method, path, body=body, headers=headers or ({"Content-Type": "application/json"} if data is not None else {}))
            r = con.getresponse(); response = (r.status, r.read(), dict(r.getheaders())); con.close(); return response
        try:
            self.assertEqual(call("GET", "/api/history/template.csv")[0], 200)
            imported_ids = []
            for yield_pct in (99, 98):
                source = f"dataset_id,catalyst_system,source_entry,experiment_mode,sulfoxide_pct\nlocal,homogeneous,1,flow,{yield_pct}\n".encode()
                code, body, _ = call("POST", "/api/history/import", {"name": "history.csv", "content_base64": base64.b64encode(source).decode()})
                self.assertEqual(code, 200)
                imported_ids.append(json.loads(body)["id"])
            before = {str(p): p.read_bytes() for p in Path(self.tmp.name).rglob("*") if p.is_file()}
            code, body, _ = call("POST", "/api/history/resolve", {"history_ids": imported_ids})
            self.assertEqual(code, 200)
            resolved = json.loads(body)
            self.assertEqual(resolved["summary"]["canonical_count"], 1)
            self.assertEqual(resolved["summary"]["conflict_count"], 1)
            self.assertEqual(len(resolved["rows"][0]["source_representations"]), 2)
            self.assertFalse(resolved["rows"][0]["interpretation_eligible"])
            self.assertEqual(before, {str(p): p.read_bytes() for p in Path(self.tmp.name).rglob("*") if p.is_file()})
            self.assertEqual(call("POST", "/api/history/resolve", {"history_ids": "khu_builtin"})[0], 400)
            self.assertEqual(call("GET", "/api/runs", headers={"Host": "evil.example"})[0], 403)
            self.assertEqual(call("POST", "/api/generate", DEFAULT_REQUEST, {"Content-Type": "application/json", "Origin": "https://evil.example"})[0], 403)
            self.assertEqual(call("GET", "/assets/icons/../../../../AGENTS.md")[0], 404)
            self.assertEqual(call("GET", "/%2e%2e/AGENTS.md")[0], 404)
            self.assertEqual(call("POST", "/api/generate", DEFAULT_REQUEST, {"Content-Type": "text/plain"})[0], 415)
            code, body, headers = call("POST", "/api/generate", DEFAULT_REQUEST)
            self.assertEqual(code, 200)
            self.assertNotIn("Access-Control-Allow-Origin", headers)
            self.assertEqual(call("GET", json.loads(body)["downloads"]["csv"])[0], 200)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
