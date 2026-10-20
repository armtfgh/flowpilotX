#!/usr/bin/env python3
"""Offline response check by default; --execute continues remaining approved roles."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from khu_sulfoxidation.flowpilotx_astra_council import validate_response,resume
read=lambda p:json.loads((ROOT/p).read_text())
f=read('outputs/flowpilotx_astra_preflight_2026-10-15_v1/frozen_inputs.json')
out=ROOT/'outputs/flowpilotx_astra_sol_2026-10-15_v1'
if '--execute' not in sys.argv:
    print(json.dumps({r:validate_response(r,json.loads((out/p).read_text())['response'],f)
        for r,p in [('Upstream','Upstream.json'),('Chemistry','continuation_v2/Chemistry.json')]},indent=2))
else:
    c=read('configs/flowpilotx_astra_sol_authorized_2026-10-15_v1.json')
    a=read('provenance/flowpilotx_astra_replay_amendment_2026-10-15_v3.json')
    print(json.dumps(resume(f,c,ROOT,out,a),indent=2))
