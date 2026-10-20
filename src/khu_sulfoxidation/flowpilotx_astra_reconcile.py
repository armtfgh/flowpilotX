"""Source-derived citation registry correction for an archived successful call.

The first runner missed source IDs nested in evidence and compact slide tables.
This module leaves its source, configuration and failure artifacts unchanged.
It rechecks the same response and continues remaining roles on the same ledger.
No generation retry, new evidence, relaxed criterion or prompt correction occurs.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from flowpilotx.gateway import validate_plan
from flowpilotx.selection import select
from . import flowpilotx_astra as old
from .model_gateway import CampaignGateway, ROUTING, digest


def visible_sources(context):
    """Return only IDs explicitly exposed by the evidence representation."""
    found = {}
    def add(value, pointer):
        if isinstance(value, str) and value:
            found.setdefault(value, []).append(pointer)
    def visit(value, pointer):
        if isinstance(value, dict):
            for key in ('id', 'record_id', 'source_id', 'summary_source_id', 'evidence_id'):
                if key in value:
                    add(value[key], pointer + '/' + key)
            columns = value.get('row_columns', value.get('columns', []))
            if isinstance(columns, list) and isinstance(value.get('rows'), list):
                for col, name in enumerate(columns):
                    if name in ('id', 'record_id', 'source_id', 'evidence_id'):
                        for i, row in enumerate(value['rows']):
                            if isinstance(row, list) and len(row) > col:
                                add(row[col], f'{pointer}/rows/{i}/{col}')
            for key, item in value.items():
                visit(item, pointer + '/' + key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                visit(item, f'{pointer}/{i}')
    visit(context['evidence'], '/context/evidence')
    rows = context['evidence']['homogeneous_batch_evidence']['raw_records']['rows']
    for i, row in enumerate(rows):
        add(f'khu:homogeneous:raw-entry:{int(row[0]):04d}',
            f'/context/evidence/homogeneous_batch_evidence/raw_records/rows/{i}/0')
    return found


def validate_response(role, response, frozen):
    registry = visible_sources(frozen['context'])
    if not isinstance(response, dict):
        return {'passed': False, 'errors': ['Response must be an object']}
    errors = []
    if role in ('Upstream', 'Chief'):
        # Flat registry is for validation only. No model input is modified.
        spec = deepcopy(frozen['campaign_spec'])
        spec['model_evidence'] = [{'id': i} for i in registry]
        spec['evidence'] = []
        errors.extend(validate_plan(response, {'spec': spec, 'results': []}, 6)['errors'])
        pool = {c['id']: c for c in spec['candidates']}
        chosen = response.get('selected_candidate_ids', [])
        if not isinstance(chosen, list): chosen = []
        pairs = {(pool[i]['parameters']['target_time_min'], pool[i]['parameters']['target_oxygen_eq'])
            for i in chosen if isinstance(i, str) and i in pool}
        times = {p[0] for p in pairs}
        if len(times) != 3 or pairs != {(t, e) for t in times for e in (.5, 1.)}:
            errors.append('Requested screen requires three time coordinates with both oxygen levels')
    else:
        if response.get('role') != role or response.get('verdict') not in ('accept_conditional', 'revise', 'reject'):
            errors.append('Reviewer role/verdict invalid')
        for key in ('issues', 'required_changes', 'evidence_ids', 'unresolved'):
            if not isinstance(response.get(key), list) or not all(isinstance(x, str) for x in response[key]):
                errors.append('Reviewer requires text array: ' + key)
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'evidence_ids' and isinstance(item, list):
                    errors.extend('Citation outside visible input: ' + str(i) for i in item
                        if not isinstance(i, str) or i not in registry)
                visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
    visit(response)
    return {'passed': not errors, 'errors': errors, 'visible_source_count': len(registry),
        'definition': 'Same structural/source criteria, now covering source IDs explicitly present in nested and compact evidence; scientific review remains separate'}


def continue_round(frozen, config, root, output, amendment):
    root, output = Path(root), Path(output)
    old.prepare_round(frozen, config, root)
    old._authorization(config, root)
    for path, expected in amendment['runtime_source_hashes'].items():
        if old._file_hash(root / path) != expected: raise ValueError('Amendment source changed: ' + path)
    upstream = old._read(output / 'Upstream.json')
    check = validate_response('Upstream', upstream['response'], frozen)
    if not check['passed']: raise ValueError('Saved Upstream still fails validation: ' + str(check))
    gateway = CampaignGateway(output / 'api', config)
    expected_request = old.request_for(frozen, 'Upstream', {}, config['source_hashes'])
    answer = old._cached(gateway, 'Upstream', expected_request)
    if answer is None or answer['parsed_json'] != upstream['response']:
        raise ValueError('Original successful Upstream attempt does not match')
    folder = output / 'continuation_v2'; folder.mkdir(exist_ok=True)
    record = {'amendment': amendment, 'revalidated_original_response': check,
        'response_sha256': digest(upstream['response']), 'visible_source_registry': visible_sources(frozen['context']),
        'original_failure_preserved': True, 'extra_upstream_calls': 0}
    if (folder/'reconciliation.json').exists():
        if old._read(folder/'reconciliation.json') != record: raise ValueError('Changed reconciliation')
    else: old._write(folder/'reconciliation.json', record)
    prior = {'Upstream': upstream['response']}
    for role in old.ROLES[1:]:
        old._source_check(config, root)
        for path, expected in amendment['runtime_source_hashes'].items():
            if old._file_hash(root/path) != expected: raise ValueError('Amendment source changed')
        request = old.request_for(frozen, role, prior, config['source_hashes'])
        failure = folder / (role + '_failure.json')
        if failure.exists(): raise ValueError('Failed role cannot be retried: ' + role)
        try:
            answer = old._cached(gateway, role, request)
            if answer is None: answer = gateway.execute_request(request, role=role, arm=old.ARM)
            if answer.get('ok') is not True: raise ValueError('API failure; no retry: ' + role)
            validation = validate_response(role, answer['parsed_json'], frozen)
            stage = {'role': role, 'request_sha256': digest(request), 'response': answer['parsed_json'],
                'validation': validation, 'scientific_review_status': 'pending',
                'api_metadata': {k: v for k, v in answer.items() if k not in ('text', 'parsed_json')}}
            path = folder / (role + '.json')
            if path.exists():
                if old._read(path) != stage: raise ValueError('Archived stage differs')
            else: old._write(path, stage)
            if not validation['passed']: raise ValueError('Structural/source validation failed: ' + str(validation['errors']))
            prior[role] = answer['parsed_json']
        except Exception as exc:
            if not failure.exists(): old._write(failure, {'role': role, 'error': str(exc), 'retry_allowed': False})
            raise
    chosen = prior['Chief']['selected_candidate_ids']
    pool = {c['id']: c for c in frozen['campaign_spec']['candidates']}
    high = sorted((pool[i] for i in chosen if pool[i]['parameters']['target_oxygen_eq'] == 1),
        key=lambda c: c['parameters']['target_time_min'])
    repeat = high[len(high)//2]['id']
    selection = select(frozen['campaign_spec'], [], {'requested_count': 6, 'repeat_count': 1,
        'repeat_of': repeat, 'allow_conditional': True, 'selected_candidate_ids': chosen}, 'no_flow_observations')
    if selection['status'] == 'rejected': raise ValueError('Physical selection gate rejected proposal')
    report = {'schema_version': 'flowpilotx-astra-review-v1', 'status': 'structurally_checked_scientific_review_pending',
        'frozen_inputs_sha256': digest(frozen), 'configuration_sha256': digest(config),
        'input_campaign_id': frozen['campaign_id'], 'input_campaign_version': frozen['campaign_version'],
        'input_campaign_revision_sha256': frozen['campaign_revision_sha256'], 'source_context_sha256': frozen['context_sha256'],
        'routing': deepcopy(ROUTING), 'roles': list(old.ROLES), 'arm': old.ARM,
        'selected_candidate_ids': chosen, 'repeat_of': repeat, 'final_plan': prior['Chief'],
        'role_outputs': prior, 'selection_audit': selection, 'api_audit': gateway.audit(),
        'scientific_review_status': 'pending', 'human_corrections_supplied': False, 'Qwen_outputs_supplied': False,
        'measured_flow_outcomes_seen': False, 'production_mutation': False, 'laboratory_released': False,
        'comparison_limits': frozen['comparison_limits'], 'validation_amendment': record}
    old._write(output/'campaign_result.json', report)
    old._write(output/'nominations_freeze.json', {'frozen_at': datetime.now(timezone.utc).isoformat(),
        'report_sha256': digest(report), 'selected_candidate_ids': chosen, 'repeat_of': repeat,
        'scientific_review_status': 'pending', 'laboratory_released': False, 'production_mutation': False})
    return report
