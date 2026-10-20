"""Portable demo and loopback API checks; no study data or model calls."""
from contextlib import contextmanager
import base64
import http.client
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from flowpilotx.server import Application, KHU_REQUIRED_FILES, handler, resolve_profile


@contextmanager
def local_server(app):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


class PortableTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {'FLOWPILOTX_QWEN_MAX_CALLS': '0', 'FLOWPILOTX_ENABLE_QWEN': '0'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_standalone_bootstrap_has_six_synthetic_designs_and_empty_history(self):
        app = Application(self.root)
        bootstrap = app.bootstrap()
        self.assertEqual(bootstrap['profile'], 'demo')
        self.assertTrue(bootstrap['synthetic_demo'])
        self.assertEqual(bootstrap['history']['rows'], [])
        self.assertEqual(bootstrap['history']['sources'], [])
        self.assertTrue(bootstrap['history']['summary']['synthetic_fixture'])
        campaign = app.store.get(bootstrap['default_campaign_id'])
        self.assertEqual(len(campaign['last_round']['selected']), 6)
        self.assertEqual(len(campaign['last_round']['repeats']), 1)
        self.assertEqual(len(campaign['last_round']['candidate_pool']), 12)
        self.assertEqual(campaign['spec']['evidence'], [])
        self.assertEqual(campaign['results'], [])
        self.assertFalse(campaign['laboratory_released'])
        self.assertEqual(app.gateway.status()['max_calls'], 0)
        self.assertEqual(bootstrap['default_model_route'], 'deterministic')
        self.assertEqual(app.storage, self.root / '.flowpilotx')
        self.assertEqual(Application(self.root).ensure_default(), campaign['id'])

    def test_partial_khu_bundle_never_silently_becomes_demo(self):
        with self.assertRaisesRegex(ValueError, 'complete private source bundle'):
            Application(self.root, profile='khu')
        (self.root / 'KHU_inventory_complete_20260918_v7.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'complete private source bundle'):
            Application(self.root)
        self.assertEqual(Application(self.root, profile='demo').profile, 'demo')
        for name in KHU_REQUIRED_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic presence fixture; not valid source contents')
        self.assertEqual(resolve_profile(self.root, 'auto'), 'khu')
        self.assertEqual(resolve_profile(self.root, 'demo'), 'demo')
        with self.assertRaises(ValueError):
            resolve_profile(self.root, 'unknown')

    def test_demo_does_not_read_study_requirements_or_history(self):
        private_config = self.root / 'configs/flowpilotx_requirements_v1.json'
        private_config.parent.mkdir()
        private_config.write_text('invalid private JSON must not be read')
        app = Application(self.root, profile='demo')
        self.assertTrue(app.requirements()['requirements'])
        campaign_id = app.ensure_default()
        source = b'id,experiment_mode,value,unreported\nsynthetic-row,batch,104,\n'
        imported = app.import_history({'campaign_id': campaign_id, 'name': 'synthetic.csv',
            'content_base64': base64.b64encode(source).decode()})
        self.assertEqual(imported['records'][0]['value'], '104')
        self.assertEqual(imported['records'][0]['unreported'], '')
        self.assertEqual(len(app.store.get(campaign_id)['spec']['evidence']), 1)

    def test_model_availability_reflects_configured_gateway(self):
        app = Application(self.root)
        app.gateway.route['available'] = False
        self.assertFalse(app.bootstrap()['capabilities']['model_generation'])
        app.gateway.route['available'] = True
        self.assertTrue(app.bootstrap()['capabilities']['model_generation'])
        self.assertEqual(app.bootstrap()['models'][0], app.gateway.route)
        self.assertEqual(app.gateway.status()['attempted_calls'], 0)

    def test_loopback_api_serves_packaged_assets_and_archives_rejection(self):
        app = Application(self.root)
        self.assertTrue((app.static / 'index.html').is_file())
        with local_server(app) as address:
            connection = http.client.HTTPConnection(*address, timeout=5)
            self.addCleanup(connection.close)
            for route, marker in [('/', b'FlowPilotX'), ('/app.js', b'use strict'), ('/style.css', b'{')]:
                connection.request('GET', route)
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIn(marker, response.read())
            connection.request('GET', '/api/bootstrap')
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            bootstrap = json.loads(response.read())
            cid = bootstrap['default_campaign_id']
            connection.request('POST', f'/api/campaigns/{cid}/generate',
                               json.dumps({'requested_count': 3, 'repeat_count': 0, 'model_route': 'deterministic'}),
                               {'Content-Type': 'application/json'})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(len(json.loads(response.read())['selected']), 3)
            connection.request('POST', f'/api/campaigns/{cid}/generate',
                               json.dumps({'requested_count': 13, 'model_route': 'deterministic'}),
                               {'Content-Type': 'application/json'})
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
            self.assertEqual(app.store.get(cid)['last_round']['status'], 'rejected')
            connection.request('GET', '/api/bootstrap', headers={'Host': 'untrusted.example'})
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()

    def test_launch_import_outside_repository_without_khu_package(self):
        # The demo must remain independent of the shipped study adapter.
        import flowpilotx
        source = str(Path(flowpilotx.__file__).resolve().parent.parent)
        script = """import importlib.abc, json, sys
class NoKHU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('khu_sulfoxidation'):
            raise AssertionError('Portable bootstrap imported private adapter')
sys.meta_path.insert(0, NoKHU())
from flowpilotx.server import Application
app = Application('.')
boot = app.bootstrap()
print(json.dumps({'profile': boot['profile'], 'assets': (app.static / 'index.html').is_file(), 'calls': app.gateway.status()['attempted_calls']}))
"""
        env = {**os.environ, 'PYTHONPATH': source, 'PYTHONDONTWRITEBYTECODE': '1'}
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=self.root, env=env,
                                text=True, capture_output=True, check=True, timeout=20)
        self.assertEqual(json.loads(result.stdout), {'profile': 'demo', 'assets': True, 'calls': 0})


if __name__ == '__main__':
    unittest.main()
