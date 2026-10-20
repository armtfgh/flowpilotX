#!/usr/bin/env python3
"""Prospective local-model compliance benchmark. Synthetic probes are not lab data."""
from pathlib import Path
import argparse
from copy import deepcopy
import json
import sys
from datetime import datetime, timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from flowpilotx.server import Application
from flowpilotx.gateway import PLANNER_SYSTEM, REVIEW_SYSTEM, planning_context, validate_plan, write, digest

ROOT=Path(__file__).resolve().parents[1]


def compact_call(result):
    return {'call_id':result['call_id'],'response':result['parsed']}


def physical_checks(plan,campaign):
    c={x['id']:x for x in campaign['spec']['candidates']}
    selected=[c[i] for i in plan.get('selected_candidate_ids',[]) if i in c]
    times={x['parameters']['target_time_min'] for x in selected}
    pairs={(x['parameters']['target_time_min'],x['parameters']['target_oxygen_eq']) for x in selected}
    return [
      {'name':'Six unique calculated settings','passed':len(selected)==6 and len(pairs)==6},
      {'name':'Three nominal-time coordinates with both oxygen levels','passed':len(times)==3 and all((t,e) in pairs for t in times for e in [.5,1.])},
      {'name':'Oxygen-only and finite sample-loop mode','passed':bool(selected) and all(x['parameters']['gas_identity']=='O2' and x['parameters']['feed_mode']=='sample_loop' for x in selected)},
      {'name':'Loop volume remains unassigned','passed':bool(selected) and all(x['parameters'].get('loop_volume_mL') is None for x in selected)},
      {'name':'No measured outcome or residence invented','passed':bool(selected) and all(x.get('actual') is None or not any(v is not None for v in x.get('actual',{}).values()) for x in selected)},
    ]


def archived_or_call(app, role, system, context, output_tokens=6000):
    # Resume only an identical successful role packet. A failed attempt is never retried here.
    for path in app.gateway.calls():
        request=json.loads(path.read_text())
        if request['role']==role and request['context_sha256']==digest(context) and request['body']['messages'][0]['content']==system:
            outcome=json.loads((path.parent/'outcome.json').read_text())
            if outcome['status']=='succeeded':
                print('Reusing archived '+role+': '+path.parent.name,flush=True)
                return {'call_id':path.parent.name,'parsed':json.loads((path.parent/'parsed.json').read_text()),**outcome}
            raise ValueError('Archived failed attempt '+path.parent.name+': '+outcome.get('error','unknown failure')+'; no automatic retry')
    return app.gateway.call(role,system,context,output_tokens=output_tokens)


def run_campaign(app,folder):
    if (folder/'campaign_result.json').exists():return json.loads((folder/'campaign_result.json').read_text())
    campaign=app.store.get(app.ensure_default())
    request={'requested_count':6,'repeat_count':1,'allow_conditional':True,
       'researcher_instructions':'Select three time coordinates with both0.5and1.0 inletO2 levels, six distinct conditions in total. Compare alternative3,5,7.5,10,15,20min coordinates. Prefer a broad informative initial window but justify your choice from evidence and uncertainty. The first6default settings are a planning anchor, not a requirement. Do not claim a fitted optimum. Cite exact record IDs. Keep rationale concise. One independent repeat will be chosen at the middle selected time/higher oxygen.'}
    context=planning_context(campaign,request)
    if (folder/'frozen_inputs.json').exists():
        frozen=json.loads((folder/'frozen_inputs.json').read_text())
        if frozen['source_snapshot']!=campaign['spec']:raise ValueError('Frozen benchmark source changed; explicit new revision required')
        context=frozen['context']
    else:write(folder/'frozen_inputs.json',{'campaign_id':campaign['id'],'campaign_revision':campaign['version'],
       'context':context,'context_sha256':digest(context),'benchmark_contract':json.loads((ROOT/'configs/flowpilotx_local_qwen_benchmark_v1.json').read_text()),
       'source_snapshot':campaign['spec'],'laboratory_outcomes_seen':False})
    print('Qwen Upstream: all116homogeneous rows and12physical candidates',flush=True)
    upstream=archived_or_call(app,'Upstream',PLANNER_SYSTEM,context)
    upstream_check=validate_plan(upstream['parsed'],campaign,6)
    
    if not (folder/'upstream_validation_v2.json').exists():write(folder/'upstream_validation_v2.json',upstream_check)
    reviews=[]
    for role in ['Chemistry','KineticsSelectivity','Fluidics','Safety']:
        print('Qwen reviewer: '+role,flush=True)
        review=archived_or_call(app,role,REVIEW_SYSTEM,{**context,'assigned_role':role,
             'proposal':upstream['parsed'],'deterministic_validation':upstream_check},output_tokens=1500)
        reviews.append(compact_call(review))
    print('Qwen Skeptic',flush=True)
    skeptic=archived_or_call(app,'Skeptic',REVIEW_SYSTEM,{**context,'assigned_role':'Skeptic: challenge evidence, confounding and apparent consensus',
        'proposal':upstream['parsed'],'specialist_reviews':reviews},output_tokens=1800)
    print('Qwen Chief: frozen final selection',flush=True)
    chief_context=deepcopy(context)
    chief_context['questions']={'question_cards':[{'id':q['id'],'question':q['question']} for q in campaign['spec']['questions']],
        'boundary':'Chief sees question identities/text; full card details remain in frozen inputs and Upstream/specialist packets.'}
    chief_context['candidates']=[{k:v for k,v in c.items() if k not in {'evidence_ids','question_ids','curiosity'}} for c in chief_context['candidates']]
    chief_context['engineering_corrections_from_human_audit']=[
        'Do not adopt reviewer errors: nominal illuminated coil transit uses coil volume divided by liquid plus in-reactor inlet gas flow. Loop volume determines injection duration and finite-plug sampling, not an added illuminated reactor volume.',
        'Inlet oxygen equivalents use STP gas molar flow divided by substrate molar flow, independent of reactor pressure. Reactor absolute pressure affects estimated gas volume and transit, not inlet equivalents.',
        'Matched nominal transit does not isolate oxygen causality: gas fraction, mass transfer and photon exposure can differ. This screen cannot distinguish sequential from parallel pathways merely from a monotonic sulfone trend.',
        'Do not invent significance thresholds such as 5% or 10% without assay uncertainty. Batch lamp wattage was not assumed to equal flow photon dose.'
    ]
    chief_context.update(upstream_proposal=upstream['parsed'],upstream_validation=upstream_check,
        specialist_reviews=reviews,skeptic=skeptic['parsed'],role_information_boundary={
        'all_116_homogeneous_reported_rows_retained':True,'human_assisted_correction':True,
        'omitted_redundancy':'Chief candidate rationale/source lists and prewritten question-card detail; raw evidence and numerical settings retained',
        'benchmark_scoring':'Upstream and council scientific errors remain failures; assisted Chief is not independent success'})
    if not (folder/'chief_packet.json').exists():write(folder/'chief_packet.json',chief_context)
    chief=archived_or_call(app,'Chief',PLANNER_SYSTEM+'\nIntegrate the specialist and skeptic reviews, checking their claims against verified arithmetic. The human engineering audit explicitly identifies reviewer mistakes. Return full required JSON with exactly6selected IDs, concise rationales and at most2proposed hypotheses. Hypotheses remain unreviewed. Avoid unsupported numerical thresholds and mechanistic identifiability claims.',
        chief_context,output_tokens=4500)
    validation=validate_plan(chief['parsed'],campaign,6)
    checks=physical_checks(chief['parsed'],campaign)
    write(folder/'chief_validation.json',{'contract':validation,'checks':checks})
    result={'status':'passed' if validation['passed'] and all(c['passed'] for c in checks) else 'failed',
        'campaign_id':campaign['id'],'upstream_call':upstream['call_id'],'review_calls':[r['call_id'] for r in reviews],
        'skeptic_call':skeptic['call_id'],'chief_call':chief['call_id'],'final_plan':chief['parsed'],
        'validation':validation,'checks':checks,'experimental_performance':'not_measured',
        'interpretation':'One local Qwen model in seven FlowPilot review roles. Chief received explicit human engineering corrections after observed reviewer errors; this is assisted selection, not independent model validation.',
        'scientific_reasoning_status':'requires_human_review','human_assisted_chief':True}
    if result['status']=='passed':
        chosen=chief['parsed']['selected_candidate_ids']
        lookup={x['id']:x for x in campaign['spec']['candidates']}
        high=sorted((lookup[i] for i in chosen if lookup[i]['parameters']['target_oxygen_eq']==1),key=lambda x:x['parameters']['target_time_min'])
        repeat_of=high[len(high)//2]['id']
        round_=app.store.generate(campaign['id'],{**request,'selected_candidate_ids':chosen,'repeat_of':repeat_of,
            'expected_campaign_version':campaign['version'],'expected_revision_sha256':campaign['content_sha256'],
            'model_route':'qwen3.6-27b','model_review':{'pipeline':'FlowPilot seven-role scientific review adapted for experiment sets',
            'upstream':compact_call(upstream),'specialists':reviews,'skeptic':compact_call(skeptic),'chief':compact_call(chief),
            'parsed':chief['parsed'],'validation':validation,'source_context_sha256':digest(context)}})
        result['round_id']=round_['id'];result['repeat_of']=repeat_of
        write(folder/'frozen_selected_round.json',round_)
    write(folder/'campaign_result.json',result)
    return result


def generic_fixture():
    """Software fixture only. No experimental recommendation or outcome claim."""
    questions=[{'id':'Q-thermal','question':'Does higher temperature trade conversion against by-product formation?',
      'competing_explanations':['Conversion limited within tested time','By-product formation increases with temperature'],
      'expected_observations':['Compare desired product and by-product at matched nominal time'],
      'measurements':['Starting material, desired product and by-product by calibrated assay'],
      'confounders':['Temperature-dependent residence distribution'],
      'decision_rules':['If desired product falls while by-product rises, test lower temperature'],
      'evidence_ids':['fixture-batch-1'],'status':'unresolved'}]
    candidates=[]
    for i,(t,tm) in enumerate([(40,2),(40,5),(60,2),(60,5),(80,2),(80,5)],1):
        candidates.append({'id':f'T{i}','parameters':{'temperature_C':t,'nominal_time_min':tm,'gas_identity':'none','feed_mode':'continuous_liquid'},
           'feasible':True,'rejection_reasons':[],'question_ids':['Q-thermal'],'evidence_ids':['fixture-batch-1'],
           'engineering_status':'conditional','substrate_id':'synthetic-substrate-A','curiosity':'Controlled temperature/time comparison',
           'costs':{'material_mmol':0.1}})
    return {'name':'Synthetic thermal chemistry software fixture','protocol_text':'Fictional liquid-phase thermal transformation. No gas or photochemistry. This is a software test, not a real protocol.',
      'chemistry':{'reaction_class':'synthetic thermal transformation','substrate':'synthetic-substrate-A'},
      'objective':{'mode':'explore','metric':'desired_product','direction':'maximize','unit':'%'},
      'inventory':{'schema_version':'flowpilotx-inventory-v1','items':[]},'questions':questions,'candidates':candidates,
      'evidence':[{'id':'fixture-batch-1','experiment_mode':'batch','record_kind':'actual','status':'synthetic_fixture',
                  'conditions':{'temperature_C':60,'duration_min':60},'outcomes':{'desired_product':65,'byproduct':20},
                  'source':'Benchmark fixture invented for software testing, never KHU evidence'}],
      'requirements':[],'information_boundary':'Synthetic fixture; no measured laboratory observations',
      'observation_domain':'synthetic thermal fixture'}


def run_probes(app,folder):
    if (folder/'probe_results.json').exists():return json.loads((folder/'probe_results.json').read_text())
    spec=generic_fixture();fixture={'spec':spec,'results':[]}
    checks=[]
    tests=[('GenericChemistry',fixture,{'requested_count':3,'repeat_count':0},'Select three distinct complementary thermal settings. Never add oxygen or light.'),
      ('ConflictingProse',deepcopy(app.store.get(app.ensure_default())),{'requested_count':6},'Historical notes mention air and500min. Current structured requirements are oxygen-only/sample-loop and bounded candidates. Explain why those historical values are not new apparatus setpoints.')]
    for role,campaign,request,instruction in tests:
        print('Qwen probe: '+role,flush=True)
        request['researcher_instructions']=instruction
        try:
            call=archived_or_call(app,role,PLANNER_SYSTEM,planning_context(campaign,request),output_tokens=5000)
            validation=validate_plan(call['parsed'],campaign,request['requested_count'])
            checks.append({'name':role,'passed':validation['passed'],'validation':validation,'call_id':call['call_id'],
                           'plan':call['parsed'],'data_status':'synthetic_fixture' if role=='GenericChemistry' else 'real_history_no_flow_results'})
        except ValueError as exc:
            checks.append({'name':role,'passed':False,'error':str(exc),'failure_not_retried':True})
    # This deliberately contradictory synthetic outcome tests revision, not scientific discovery.
    follow=deepcopy(fixture)
    follow['spec']['objective']['mode']='optimize'
    follow['results']=[{'id':'fixture-flow-1','design_id':'T2','experiment_mode':'flow','actual_conditions':spec['candidates'][1]['parameters'],
          'analytes':{'desired_product':{'value':80,'unit':'%'},'byproduct':{'value':4,'unit':'%'}},'analytical_method':'synthetic assay',
          'outcome_status':'qualified','quality':{'accepted':True,'basis':'Fictional software fixture'},'data_status':'synthetic_fixture'},
        {'id':'fixture-flow-2','design_id':'T6','experiment_mode':'flow','actual_conditions':spec['candidates'][5]['parameters'],
          'analytes':{'desired_product':{'value':20,'unit':'%'},'byproduct':{'value':65,'unit':'%'}},'analytical_method':'synthetic assay',
          'outcome_status':'qualified','quality':{'accepted':True,'basis':'Fictional software fixture'},'data_status':'synthetic_fixture'}]
    print('Qwen probe: synthetic next round',flush=True)
    call=archived_or_call(app,'SyntheticFeedback',PLANNER_SYSTEM,planning_context(follow,{'requested_count':2,
       'researcher_instructions':'Use these explicitly synthetic results for a second-round decision. Explain how evidence changes priorities. No monotonic longer/hotter-is-better rule. Cite fixture-flow IDs.'}),output_tokens=4000)
    validation=validate_plan(call['parsed'],follow,2)
    cited={e for r in call['parsed'].get('design_rationales',[]) for e in r.get('evidence_ids',[])}
    checks.append({'name':'SyntheticFeedback','passed':validation['passed'] and bool(cited&{'fixture-flow-1','fixture-flow-2'}),
      'validation':validation,'call_id':call['call_id'],'plan':call['parsed'],'data_status':'synthetic_fixture_not_KHU_results'})
    write(folder/'probe_results.json',checks)
    return checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['campaign','probes','all'],default='all')
    parser.add_argument('--revision',default='v2')
    args=parser.parse_args();app=Application(ROOT)
    if not args.revision.replace('_','').isalnum():raise ValueError('Invalid benchmark revision')
    folder=app.storage/('benchmark_'+args.revision);folder.mkdir(parents=True,exist_ok=True)
    report={'schema_version':'flowpilotx-benchmark-v1','created_at':datetime.now(timezone.utc).isoformat(),
       'scope':'Software/model compliance only. No measured KHU flow results or comparative optimization superiority.',
       'model':'/models/Qwen3.6-27B','checks':[],'status':'running'}
    try:
        if args.stage in ['campaign','all']:report['campaign']=run_campaign(app,folder)
        elif (folder/'campaign_result.json').exists():report['campaign']=json.loads((folder/'campaign_result.json').read_text())
        if args.stage in ['probes','all']:report['probes']=run_probes(app,folder)
        report['checks']=report.get('campaign',{}).get('checks',[])+[{k:v for k,v in r.items() if k!='plan'} for r in report.get('probes',[])]
        report['status']='partial_scientific_review_required' if report.get('campaign',{}).get('status')=='passed' and all(x['passed'] for x in report['checks']) else 'partial_or_failed'
    except Exception as exc:
        report['status']='failed';report['error']=str(exc)
        print('BENCHMARK FAILURE: '+str(exc),flush=True)
    report['scientific_review']=json.loads((ROOT/'review/flowpilotx_qwen_scientific_audit_2026-10-15_v2.json').read_text())
    report['structural_checks_status']='see_individual_checks'
    report['scientific_reasoning_status']='failed_assisted_and_unassisted_review'
    report['gateway']=app.gateway.status()
    # Aggregate index may update; immutable call/round/source records never do.
    (app.storage/'benchmark_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'report':str(app.storage/'benchmark_report.json'),'calls':report['gateway']['attempted_calls']},indent=2),flush=True)


if __name__=='__main__':main()
