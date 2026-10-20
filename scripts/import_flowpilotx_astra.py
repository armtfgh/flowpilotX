#!/usr/bin/env python3
"""Prepare or explicitly commit a completed Astra/Sol review to the local GUI.

Read-only by default. This script has no model calls and cannot extend a budget.
"""
import argparse
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from flowpilotx.evidence import digest
from flowpilotx.review_import import prepare_review_import, commit_review_import, file_hash, write_import_package_records


def prepare(root, run_dir, package_dir, audit_path, package_id, freeze_path=None):
    run_dir, audit_path = Path(run_dir), Path(audit_path)
    freeze_path = Path(freeze_path or run_dir / 'input_freeze.json')
    result_path = run_dir / 'campaign_result.json'
    result, frozen, audit = [json.loads(p.read_text()) for p in [result_path, freeze_path, audit_path]]
    inputs, config = frozen['frozen_inputs'], frozen['configuration']
    if result['frozen_inputs_sha256'] != digest(inputs) or result['configuration_sha256'] != digest(config):
        raise ValueError('Result does not bind the supplied frozen inputs/configuration')
    if result.get('laboratory_released') is True or result.get('measured_flow_outcomes_seen') is not False:
        raise ValueError('Expected an unexecuted prospective model campaign')
    final = result['final_plan']
    selected = result['selected_candidate_ids']
    if final['selected_candidate_ids'] != selected:
        raise ValueError('Chief and frozen selected IDs differ')
    api = result['api_audit']
    entries = api['entries']
    routing = result['routing']
    expected_routing = {'Upstream': 'gpt-6-astra', **{role: 'gpt-6.1-sol' for role in
        ['Chemistry', 'KineticsSelectivity', 'FluidicsPhotoreactor', 'Safety', 'Skeptic', 'Chief']}}
    if routing != config['routing'] or routing != expected_routing:
        raise ValueError('Result/configuration differ from the authorized Astra/Sol route')
    if len(entries) != 7 or api['attempt_count'] != 7 or len(routing) != 7:
        raise ValueError('This authorization requires the completed seven-stage review')
    if {e['role'] for e in entries} != set(routing) or any(e['status'] != 'completed' or e['model'] != routing[e['role']] for e in entries):
        raise ValueError('Attempt routing/status does not match the frozen review')
    if api.get('unknown_or_reserved_attempts') or api.get('unresolved_inflight') or api['accounted_cost_USD'] > 5:
        raise ValueError('Incomplete accounting or authorization cap exceeded')
    if config['max_attempts'] != 7 or config['maximum_cost_USD'] != 5:
        raise ValueError('Unexpected campaign authorization limits')
    if set(result['role_outputs']) != set(routing):
        raise ValueError('Review role outputs are incomplete')
    # The audit must name the exact final result among its source hashes.
    audit_hashes = audit.get('source_hashes', {})
    result_hash = file_hash(result_path)
    if result_hash not in audit_hashes.values():
        raise ValueError('Scientific audit does not bind the exact final campaign result')
    report_profile_path = Path(package_dir) / 'profile.json'
    report_profile = json.loads(report_profile_path.read_text())
    original_profile = inputs['campaign_spec']
    allowed_display = {'default_selected_candidate_ids', 'optional_repeat', 'limitations', 'selection_policy'}
    changed = {k for k in set(report_profile) | set(original_profile) if report_profile.get(k) != original_profile.get(k)}
    if changed - allowed_display:
        raise ValueError('Reporting profile changed fields outside the allowed display overrides: ' + ', '.join(sorted(changed - allowed_display)))
    display_overrides = {k: report_profile[k] for k in changed}
    review = {'parsed': final, 'role_outputs': result['role_outputs'], 'routing': routing,
              'route_label': 'Astra upstream → GPT-6.1 Sol council', 'call_count': 7,
              'scientific_review': audit, 'api_audit': api, 'selection_audit': result['selection_audit'],
              'generation_authorization': {'attempts_used': 7, 'attempt_limit': 7, 'remaining_attempts': 0,
                  'maximum_cost_USD': 5, 'actual_cost_USD': api['known_token_cost_USD'],
                  'generation_available': False}, 'original_campaign_result': result}
    return prepare_review_import(root, package_dir, package_id, inputs['campaign_spec'], selected,
        result['repeat_of'], review, [result_path, freeze_path, audit_path, report_profile_path, Path(package_dir) / 'render_index.json'],
        name='KHU · Astra / Sol reviewed exploration · 15 October 2026', display_overrides=display_overrides)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--run-dir', required=True)
    p.add_argument('--package-dir', required=True)
    p.add_argument('--scientific-audit', required=True)
    p.add_argument('--package-id', default='khu-astra-sol-2026-10-15-v1')
    p.add_argument('--freeze')
    p.add_argument('--storage', default='outputs/flowpilotx_platform_2026-10-15_v1')
    p.add_argument('--commit', action='store_true', help='Explicitly create the separate local campaign; never change the default campaign')
    args = p.parse_args()
    root = Path(args.root).resolve()
    path = lambda value: Path(value) if Path(value).is_absolute() else root / value
    plan = prepare(root, path(args.run_dir), path(args.package_dir), path(args.scientific_audit), args.package_id,
                   path(args.freeze) if args.freeze else None)
    if args.commit:
        receipt = commit_review_import(root, path(args.storage), plan)
        write_import_package_records(path(args.storage), receipt, path(args.package_dir))
        print(json.dumps(receipt, indent=2))
    else:
        print(json.dumps({'status': 'prepared_not_imported', 'name': plan['spec']['name'],
            'selected_candidate_ids': plan['selected_candidate_ids'], 'repeat_of': plan['repeat_of'],
            'scientific_status': plan['request']['model_review']['scientific_review']['status'],
            'source_files_sha256': plan['source_files_sha256'], 'new_model_calls': 0}, indent=2))

if __name__ == '__main__': main()
