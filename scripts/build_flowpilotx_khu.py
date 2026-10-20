"""Build a versioned offline KHU FlowPilotX profile/review package."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from khu_sulfoxidation.flowpilotx_profile import build_profile
from khu_sulfoxidation.flowpilotx_report import export_review


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--profile',type=Path,help='Use an already frozen profile rather than rebuilding')
    parser.add_argument('--selected-ids',nargs='+',help='Previously frozen candidate IDs to display; default preferred six')
    parser.add_argument('--render',action='store_true',help='Requires original-renderer dependencies/Graphviz')
    args=parser.parse_args()
    profile=json.loads(args.profile.read_text()) if args.profile else build_profile(ROOT)
    manifest=export_review(profile,args.output,args.selected_ids)
    if args.render:
        from khu_sulfoxidation.flowpilotx_figures import render_figures
        render_figures(ROOT,profile,args.output,args.selected_ids)
    print(json.dumps({'output':str(args.output),'conditions':len(manifest['selected_candidate_ids']),
        'candidate_pool':len(profile['candidates']),'status':profile['status']}))
