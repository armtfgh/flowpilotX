#!/usr/bin/env python3
"""Explicit final benchmark fixture repair; one bounded call, no retry/fallback."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from flowpilotx.server import Application
from flowpilotx.gateway import planning_context,PLANNER_SYSTEM,validate_plan,write
from flowpilotx.evidence import digest
from benchmark_flowpilotx import generic_fixture,archived_or_call
ROOT=Path(__file__).resolve().parents[1]
app=Application(ROOT);folder=app.storage/'benchmark_v2'
source=next(p for p in app.gateway.calls() if json.loads(p.read_text())['role']=='SyntheticFeedback')
old_context=json.loads(json.loads(source.read_text())['body']['messages'][1]['content'])
spec=generic_fixture();spec['objective']['mode']='optimize'
fixture={'spec':spec,'results':old_context['measured_results']}
domain=digest({'chemistry':spec['chemistry'],'observation_domain':spec['observation_domain']})
for result in fixture['results']:
    result['source_observation_domain_sha256']=domain
    result['proposed_conditions']=next(c['parameters'] for c in spec['candidates'] if c['id']==result['design_id'])
context=planning_context(fixture,{'requested_count':2,'researcher_instructions':old_context['researcher_instructions']})
write(folder/'qualified_feedback_fixture_revision.json',{
  'reason':'Initial synthetic probe harness omitted observation-domain provenance. That response remains archived as an exclusion-boundary probe, not a valid qualified-feedback demonstration.',
  'original_call_id':source.parent.name,'status':'synthetic_fixture_not_KHU_results','context':context,
  'eligible_ids_before':old_context['feedback_eligibility'],'eligible_ids_after':context['feedback_eligibility']})
call=archived_or_call(app,'SyntheticFeedbackQualified',PLANNER_SYSTEM,context,output_tokens=4000)
validation=validate_plan(call['parsed'],fixture,2)
selected=call['parsed'].get('selected_candidate_ids',[])
cited={e for r in call['parsed'].get('design_rationales',[]) for e in r.get('evidence_ids',[])}
check={'name':'SyntheticFeedbackQualified','passed':validation['passed'] and bool(cited&{'fixture-flow-1','fixture-flow-2'}) and 'T6' not in selected,
  'structural_validation':validation,'call_id':call['call_id'],'selected_candidate_ids':selected,
  'data_status':'fictional_software_fixture','rubric':'Uses actual eligible synthetic flow evidence, cites it, and does not prioritize the observed poor T6 as one of two optimization settings; this alone is not scientific validation',
  'rubric_status':'explicit_fixture_repair_followup_not_original_prospective_score','plan':call['parsed']}
write(folder/'qualified_feedback_result.json',check)
report=json.loads((app.storage/'benchmark_report.json').read_text())
report['checks'].append({k:v for k,v in check.items() if k!='plan'})
report['qualified_feedback_fixture_repair']=check
report['original_synthetic_probe_limitation']='Missing domain provenance correctly excluded both observations from numerical feedback. Do not count initial probe as eligible-feedback success.'
report['gateway']=app.gateway.status();report['status']='partial_or_failed'
(app.storage/'benchmark_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':check['passed'],'selected':selected,'calls':report['gateway']['attempted_calls']},indent=2))
