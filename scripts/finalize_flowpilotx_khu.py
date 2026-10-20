#!/usr/bin/env python3
"""Freeze independently reviewed KHU selection without a new model call."""
from pathlib import Path
from copy import deepcopy
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from flowpilotx.server import Application
from flowpilotx.gateway import write,validate_plan
from khu_sulfoxidation.flowpilotx_profile import build_profile
from khu_sulfoxidation.flowpilotx_report import export_review
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'outputs/flowpilotx_khu_2026-10-15_v1').exists():raise FileExistsError('Final package already exists; create a reviewed new version')
app=Application(ROOT);campaign=app.store.get(app.ensure_default())
folder=app.storage/'benchmark_v2';benchmark=json.loads((folder/'campaign_result.json').read_text())
audit=json.loads((ROOT/'review/flowpilotx_qwen_scientific_audit_2026-10-15_v2.json').read_text())
old_round=next(r for r in campaign['rounds'] if r['id']==benchmark['round_id'])
new=build_profile(ROOT)
assert old_round['configuration']['candidates']==new['candidates'],'Numerical candidates changed: requires new review'
assert campaign['spec']['evidence']==new['evidence'],'Evidence changed: requires new review'
campaign=app.store.revise(campaign['id'],{'observation_contract':new['observation_contract'],'source_hashes':new['source_hashes'],
    'candidates':new['candidates'],'model_candidates':new['model_candidates']})
review=deepcopy(old_round['model_review'])
review['scientific_review']={'status':'failed_hypothesis_review',
 'summary':'Qwen selection IDs pass engineering checks; raw hypotheses and projected observations failed independent scientific review, including assisted Chief.',
 'findings':[x['finding']+': '+x['correction'] for x in audit['findings']],
 'audit_source':'review/flowpilotx_qwen_scientific_audit_2026-10-15_v2.json',
 'display_policy':'Use reviewed adapter question cards; model hypotheses are archived unapproved and not directly adoptable.'}
review['reconciliation']={'model_input_campaign_version':old_round['campaign_version'],
 'current_campaign_version':campaign['version'],
 'change':'Added explicit measured-condition feedback contract and current source hashes. Physical candidates, full historical evidence and curated question semantics unchanged.',
 'new_model_calls':0,'original_frozen_round':old_round['id'],
 'independent_structural_recheck':validate_plan(review['parsed'],campaign,6)}
round_=app.store.generate(campaign['id'],{'requested_count':6,'repeat_count':1,'repeat_of':'KX04',
 'allow_conditional':True,'selected_candidate_ids':benchmark['final_plan']['selected_candidate_ids'],
 'model_route':'qwen3.6-27b','model_review':review,
 'selection_method':'Archived Qwen selection with independent engineering review; scientific model claims unapproved',
 'researcher_instructions':'Six homogeneous KHU settings; explicit oxygen/sample-loop/BPR constraints. Use reviewed adapter question cards, not the rejected model causal claims.',
 'expected_campaign_version':campaign['version'],'expected_revision_sha256':campaign['content_sha256']})
out=ROOT/'outputs/flowpilotx_khu_2026-10-15_v1'
export_review(round_['configuration'],out,benchmark['final_plan']['selected_candidate_ids'])
write(out/'frozen_round.json',round_)
write(out/'model_selected.json',{'selected_candidate_ids':benchmark['final_plan']['selected_candidate_ids'],
 'repeat_of':'KX04','chief_call':benchmark['chief_call'],'round_id':round_['id'],
 'selection_origin':'Qwen chose among12 engineered candidates; preferred six also existed as offline planning anchor',
 'scientific_review':review['scientific_review'],'reconciliation':review['reconciliation'],
 'original_campaign_result':str(folder/'campaign_result.json')})
write(out/'scientific_review.json',audit)
report=json.loads((app.storage/'benchmark_report.json').read_text())
for c in report['checks']:
 if c['name']=='SyntheticFeedback':
  c['original_structural_citation_result']=c['passed'];c['passed']=False
  c['name']='IncompleteFixtureFeedbackQualification';c['explanation']='Test fixture omitted domain provenance. It does not demonstrate qualified-feedback use; corrected explicit follow-up is reported separately.'
report['final_review_round_id']=round_['id']
report['campaign']['structural_status']=report['campaign']['status']
report['campaign']['status']='selection_structure_passed_scientific_review_failed'
report['campaign']['scientific_review']=review['scientific_review']
report['status']='partial_or_failed'
report['gateway']=app.gateway.status()
report['checks'].append({'name':'Invalid candidate cannot bypass hard constraints','passed':True,
 'evidence':'tests/test_flowpilotx_core.py::test_model_cannot_bypass_pool_hard_constraints_or_unique_count',
 'type':'deterministic_software_test'})
(app.storage/'benchmark_report.json').write_text(json.dumps(report,indent=2)+'\n')
write(out/'benchmark_snapshot.json',report)
print(json.dumps({'package':str(out),'round':round_['id'],'campaign_version':campaign['version'],'calls':app.gateway.status()['attempted_calls']},indent=2))
