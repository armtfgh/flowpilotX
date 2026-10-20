#!/usr/bin/env python3
"""Offline by default. Paid execution needs a separate approved configuration."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    from khu_sulfoxidation.flowpilotx_astra import prepare_round,execute_prepared
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=ROOT/'outputs/flowpilotx_astra_preflight_2026-10-15_v1/frozen_inputs.json')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/flowpilotx_astra_sol_proposal_2026-10-15_v1.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/flowpilotx_astra_sol_2026-10-15_v1')
    parser.add_argument('--execute',action='store_true',help='Requires separately recorded new user spending authorization; never enables it itself')
    args=parser.parse_args()
    frozen=json.loads(args.inputs.read_text());config=json.loads(args.config.read_text())
    if args.execute:
        report=execute_prepared(frozen,config,project_root=ROOT,output_dir=args.output)
    else:
        report=prepare_round(frozen,config,project_root=ROOT)
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
