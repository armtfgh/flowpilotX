#!/usr/bin/env python3
"""Resume six authorized council calls after the documented source-registry fix."""
from pathlib import Path
import json, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from khu_sulfoxidation.flowpilotx_astra_reconcile import continue_round, validate_response
read = lambda p: json.loads((ROOT/p).read_text())
f = read('outputs/flowpilotx_astra_preflight_2026-10-15_v1/frozen_inputs.json')
out = ROOT/'outputs/flowpilotx_astra_sol_2026-10-15_v1'
if '--execute' not in sys.argv:
    print(json.dumps(validate_response('Upstream', json.loads((out/'Upstream.json').read_text())['response'], f), indent=2))
else:
    c = read('configs/flowpilotx_astra_sol_authorized_2026-10-15_v1.json')
    a = read('provenance/flowpilotx_astra_registry_amendment_2026-10-15_v1.json')
    print(json.dumps(continue_round(f,c,ROOT,out,a), indent=2))
