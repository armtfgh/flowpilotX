"""Resume unchanged council prompts with prompt-compatible review arrays.

REVIEW_SYSTEM specifies arrays without constraining entries to plain strings.
Structured findings/actions are retained verbatim. Earlier parsing failures and
all paid attempts remain immutable; successful API replies are never retried.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from . import flowpilotx_astra as base
from .flowpilotx_astra_reconcile import validate_response as registry_validation
from .model_gateway import CampaignGateway, ROUTING, digest
from flowpilotx.selection import select


def ordered_response(answer):
    """Restore original output order without changing the decoded response."""
    parsed = json.loads(answer['text'])
    if not isinstance(parsed, dict) or parsed != answer['parsed_json']:
        raise ValueError('Response text/object mismatch')
    return parsed


def validate_response(role, response, frozen):
    check = registry_validation(role, response, frozen)
    if role not in ('Upstream', 'Chief') and isinstance(response, dict):
        def text_present(value):
            if isinstance(value, str): return bool(value.strip())
            if isinstance(value, dict): return any(text_present(x) for x in value.values())
            if isinstance(value, list): return any(text_present(x) for x in value)
            return False
        for key in ('issues', 'required_changes', 'unresolved'):
            values = response.get(key)
            if isinstance(values, list) and all(isinstance(x, (str, dict)) and text_present(x) for x in values):
                check['errors'] = [e for e in check['errors'] if e != 'Reviewer requires text array: ' + key]
        check['passed'] = not check['errors']
    check['review_array_contract'] = 'Arrays of text or structured textual findings/actions; evidence_ids remains an array of source-ID strings'
    return check


def resume(frozen, config, root, output, amendment):
    root, output = Path(root), Path(output)
    base.prepare_round(frozen, config, root)
    authorization = base._authorization(config, root)
    def runtime_check():
        base._source_check(config, root)
        if base._authorization(config, root) != authorization: raise ValueError('Authorization changed')
        for path, expected in amendment['runtime_source_hashes'].items():
            if base._file_hash(root/path) != expected: raise ValueError('Amended source changed: ' + path)
    runtime_check()
    gateway = CampaignGateway(output/'api', config)
    folder = output/amendment.get('artifact_folder', 'continuation_v3'); folder.mkdir(exist_ok=True)
    record = {'amendment': amendment, 'authorization': authorization, 'prompts_changed': False,
        'original_failures_preserved': True, 'generation_retries': 0}
    if (folder/'amendment.json').exists():
        if base._read(folder/'amendment.json') != record: raise ValueError('Changed continuation')
    else: base._write(folder/'amendment.json', record)
    prior = {}
    for role in base.ROLES:
        runtime_check()
        request = base.request_for(frozen, role, prior, config['source_hashes'])
        failure = folder/(role+'_failure.json')
        if failure.exists(): raise ValueError('Failed role is not retried: ' + role)
        try:
            answer = base._cached(gateway, role, request)
            cached = answer is not None
            if answer is None: answer = gateway.execute_request(request, role=role, arm=base.ARM)
            if answer.get('ok') is not True: raise ValueError('API failure, no retry: ' + role)
            # Gateway JSON archives are canonical (sorted keys). Recover the
            # original model JSON key order from its retained output text so
            # subsequent user-prompt bytes match the first execution exactly.
            parsed = ordered_response(answer)
            check = validate_response(role, parsed, frozen)
            stage = {'role': role, 'request_sha256': digest(request), 'response': parsed,
                'validation': check, 'scientific_review_status': 'pending',
                'api_metadata': {k:v for k,v in answer.items() if k not in ('text','parsed_json')}}
            path = folder/(role+'.json')
            if path.exists():
                if base._read(path) != stage: raise ValueError('Changed saved stage')
            else: base._write(path, stage)
            if not check['passed']: raise ValueError('Response validation failed: ' + str(check['errors']))
            prior[role] = parsed
        except Exception as exc:
            if not failure.exists(): base._write(failure, {'role':role,'error':str(exc),'retry_allowed':False})
            raise
    chosen = prior['Chief']['selected_candidate_ids']
    pool = {c['id']:c for c in frozen['campaign_spec']['candidates']}
    high = sorted((pool[i] for i in chosen if pool[i]['parameters']['target_oxygen_eq']==1),
        key=lambda c:c['parameters']['target_time_min'])
    repeat = high[len(high)//2]['id']
    selection = select(frozen['campaign_spec'], [], {'requested_count':6,'repeat_count':1,
        'repeat_of':repeat,'allow_conditional':True,'selected_candidate_ids':chosen},'no_flow_observations')
    if selection['status']=='rejected': raise ValueError('Physical selection failed')
    report = {'schema_version':'flowpilotx-astra-review-v1','status':'structurally_checked_scientific_review_pending',
        'frozen_inputs_sha256':digest(frozen),'configuration_sha256':digest(config),
        'input_campaign_id':frozen['campaign_id'],'input_campaign_version':frozen['campaign_version'],
        'input_campaign_revision_sha256':frozen['campaign_revision_sha256'],'source_context_sha256':frozen['context_sha256'],
        'routing':deepcopy(ROUTING),'roles':list(base.ROLES),'arm':base.ARM,'selected_candidate_ids':chosen,
        'repeat_of':repeat,'final_plan':prior['Chief'],'role_outputs':prior,'selection_audit':selection,'api_audit':gateway.audit(),
        'scientific_review_status':'pending','human_corrections_supplied':False,'Qwen_outputs_supplied':False,
        'measured_flow_outcomes_seen':False,'production_mutation':False,'laboratory_released':False,
        'comparison_limits':frozen['comparison_limits'],'validation_amendment':record}
    base._write(output/'campaign_result.json',report)
    base._write(output/'nominations_freeze.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'report_sha256':digest(report),
        'selected_candidate_ids':chosen,'repeat_of':repeat,'scientific_review_status':'pending','laboratory_released':False,'production_mutation':False})
    return report
