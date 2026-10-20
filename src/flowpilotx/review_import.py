"""Offline import of already completed model reviews. No model/network calls.

Source packages and model outputs remain untouched. An import creates a separate
campaign and frozen round, with a hash-bound allowlist for displayed figures.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .campaign import CampaignStore, validate_spec
from .evidence import digest
from .selection import select

SAFE_PACKAGE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}\Z')
ALLOWED_FIGURES = re.compile(r'diagrams/[A-Za-z0-9_.-]+\.(?:svg|png|json)\Z')
ARCHIVED_ROUTE = {'id': 'astra-sol-archived', 'label': 'Astra → Sol council · archived review',
                  'available': False, 'paid': True,
                  'reason': 'Archived campaign only. No additional generation is authorized for this route.'}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def prepare_asset_package(root, package_dir, package_id, candidates, selected_ids):
    """Read-only validation; figures must match the exact prospective parameters."""
    root, package_dir = Path(root).resolve(), Path(package_dir).resolve()
    if not SAFE_PACKAGE.fullmatch(package_id):
        raise ValueError('Invalid package ID')
    if not package_dir.is_relative_to(root / 'outputs'):
        raise ValueError('Asset packages must be inside this workspace outputs directory')
    index = json.loads((package_dir / 'render_index.json').read_text())
    pool = {c['id']: c for c in candidates}
    cases = {}
    allowed = {}
    for case in index['cases']:
        cid = case['condition_id']
        if cid in cases or cid not in pool:
            raise ValueError('Figure references a duplicate or unknown candidate')
        if case.get('source_parameters_sha256') != digest(pool[cid]['parameters']):
            raise ValueError('Figure parameters differ from the frozen candidate: ' + cid)
        for field in ['png', 'svg', 'topology_json']:
            relative = case[field]
            if not ALLOWED_FIGURES.fullmatch(relative):
                raise ValueError('Invalid or non-display asset path')
            path = (package_dir / relative).resolve()
            if not path.is_relative_to(package_dir) or not path.is_file():
                raise ValueError('Missing or escaping figure asset')
            allowed[relative] = file_hash(path)
        cases[cid] = deepcopy(case)
    if set(selected_ids) - set(cases):
        raise ValueError('A selected condition has no matching versioned figure')
    return {'schema_version': 'flowpilotx-asset-package-v1', 'package_id': package_id,
            'root_relative': str(package_dir.relative_to(root)), 'files_sha256': allowed,
            'render_index_sha256': file_hash(package_dir / 'render_index.json'), 'cases': cases}


def registered_asset(root, storage, package_id, relative):
    """Return only a registered, unchanged figure; never serve arbitrary outputs."""
    if not SAFE_PACKAGE.fullmatch(package_id) or not ALLOWED_FIGURES.fullmatch(relative):
        raise ValueError('Invalid versioned asset path')
    registry = Path(storage) / 'asset_packages' / (package_id + '.json')
    package = json.loads(registry.read_text())
    if package.get('package_id') != package_id or relative not in package['files_sha256']:
        raise ValueError('Unregistered figure asset')
    root = Path(root).resolve()
    base = (root / package['root_relative']).resolve()
    path = (base / relative).resolve()
    if not base.is_relative_to(root / 'outputs') or not path.is_relative_to(base) or not path.is_file():
        raise ValueError('Asset escaped the registered output package')
    if file_hash(path) != package['files_sha256'][relative]:
        raise ValueError('Registered figure integrity check failed')
    return path


def prepare_review_import(root, package_dir, package_id, spec, selected_ids, repeat_of,
                          model_review, source_files, *, name=None, display_overrides=None):
    """Build and check a reviewable offline import plan without storage mutation."""
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError('Selected IDs must be nonempty and unique')
    scientific = model_review.get('scientific_review', {})
    if scientific.get('status') in {None, 'pending'}:
        raise ValueError('Completed scientific review is required before importing a final campaign')
    if scientific.get('laboratory_released') is True or scientific.get('experimental_validation') is True:
        raise ValueError('A reasoning review cannot establish laboratory release or experimental validation')
    profile = deepcopy(spec)
    allowed_display = {'default_selected_candidate_ids', 'optional_repeat', 'limitations', 'selection_policy'}
    display_overrides = deepcopy(display_overrides or {})
    if set(display_overrides) - allowed_display:
        raise ValueError('Only explicitly declared display fields may override the frozen model input')
    if display_overrides.get('default_selected_candidate_ids', selected_ids) != selected_ids:
        raise ValueError('Display selected IDs differ from final review')
    if display_overrides.get('optional_repeat', {}).get('repeat_of', repeat_of) != repeat_of:
        raise ValueError('Display repeat differs from final review')
    profile.update(display_overrides)
    profile['name'] = name or profile['name'] + ' · imported model review'
    source_hashes = {str(Path(p).resolve().relative_to(Path(root).resolve())): file_hash(p) for p in source_files}
    package = prepare_asset_package(root, package_dir, package_id, profile['candidates'], selected_ids)
    for candidate in profile['candidates']:
        # Do not fall back to a previous campaign's image for a new route.
        candidate.pop('diagram_url', None)
        candidate.pop('diagram_png_url', None)
        if isinstance(candidate.get('topology'), dict):
            candidate['topology'].pop('diagram_url', None)
        candidate.pop('diagram', None)
        candidate.pop('topology_url', None)
        if candidate['id'] in package['cases']:
            candidate['diagram_url'] = '/assets/packages/' + package_id + '/' + package['cases'][candidate['id']]['svg']
    profile['imported_review'] = {'source_files_sha256': source_hashes,
        'original_spec_sha256': digest(spec), 'asset_package_id': package_id,
        'display_overrides': display_overrides,
        'policy': 'Separate offline display import; exact selected parameters preserved; no model dispatch or automatic hypothesis adoption.'}
    profile['model_config'] = {'route_id': 'deterministic', 'archived_review_route': ARCHIVED_ROUTE['id'],
                               'archived_review_generation_available': False}
    profile = validate_spec(profile)
    review = deepcopy(model_review)
    review['source_files_sha256'] = source_hashes
    review['route_label'] = review.get('route_label', 'Astra upstream → GPT-6.1 Sol council')
    request = {'requested_count': len(selected_ids), 'repeat_count': int(repeat_of is not None),
               'selected_candidate_ids': selected_ids, 'allow_conditional': True,
               'model_route': ARCHIVED_ROUTE['id'], 'model_review': review,
               'selection_method': 'Imported completed model review; deterministic engineering gates; no new model calls'}
    if repeat_of is not None:
        request['repeat_of'] = repeat_of
    selection = select(profile, [], request, digest({"chemistry": profile["chemistry"], "observation_domain": profile.get("observation_domain")}))
    if selection.get('status') == 'rejected':
        raise ValueError('Import failed selection checks: ' + '; '.join(selection.get('rejection_reasons', [])))
    return {'schema_version': 'flowpilotx-review-import-plan-v1', 'spec': profile,
            'request': request, 'asset_package': package, 'source_files_sha256': source_hashes,
            'selected_candidate_ids': selected_ids, 'repeat_of': repeat_of,
            'new_model_calls': 0, 'laboratory_released': False}


def commit_review_import(root, storage, plan):
    """Explicit local mutation; create exclusive receipt, campaign and round.

    Never changes the default campaign pointer or any existing campaign/round.
    A pre-created reservation remains visible if an interrupted import fails.
    """
    root, storage = Path(root).resolve(), Path(storage).resolve()
    for relative, expected in plan['source_files_sha256'].items():
        if file_hash(root / relative) != expected:
            raise ValueError('Source changed after import preparation: ' + relative)
    package = plan['asset_package']
    base = root / package['root_relative']
    for relative, expected in package['files_sha256'].items():
        if file_hash(base / relative) != expected:
            raise ValueError('Figure changed after import preparation: ' + relative)
    package_id = package['package_id']
    imports = storage / 'review_imports' / package_id
    imports.mkdir(parents=True, exist_ok=False)
    write_new(imports / 'import_plan.json', plan)
    write_new(storage / 'asset_packages' / (package_id + '.json'), package)
    store = CampaignStore(storage / 'campaigns')
    campaign = store.create(plan['spec'])
    round_ = store.generate(campaign['id'], plan['request'])
    receipt = {'schema_version': 'flowpilotx-review-import-receipt-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'campaign_id': campaign['id'], 'round_id': round_['id'], 'package_id': package_id,
        'selected_candidate_ids': plan['selected_candidate_ids'], 'repeat_of': plan['repeat_of'],
        'source_files_sha256': plan['source_files_sha256'], 'import_plan_sha256': digest(plan),
        'round_sha256': round_['content_sha256'], 'default_campaign_changed': False,
        'new_model_calls': 0, 'laboratory_released': False}
    write_new(imports / 'receipt.json', receipt)
    return receipt


def blank_results_template(round_):
    """One empty observation form per unique condition and independent repeat."""
    spec = round_['configuration']
    contract = spec.get('observation_contract', {})
    actual = deepcopy(contract.get('actual_conditions_template', {}))
    def clear(value):
        return {key: clear(item) if isinstance(item, dict) else None for key, item in value.items()}
    actual = clear(actual)
    for path in contract.get('required_actual_conditions', []):
        parts = path.split('.')
        cursor = actual
        for key in parts[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[parts[-1]] = None
    if not actual:
        actual = {'observation': None}
    objective = spec['objective']
    components = list(dict.fromkeys([objective.get('metric', 'product_yield'), *objective.get('separate_components', [])]))
    rows = []
    for candidate in round_['selected'] + round_['repeats']:
        row = {'id': 'observation-' + round_['id'] + '-' + candidate['id'],
            'round_id': round_['id'], 'design_id': candidate['id'], 'experiment_mode': 'flow',
            'actual_conditions': deepcopy(actual),
            'analytes': {key: {'value': None, 'unit': objective.get('unit', '%'), 'qualifier': 'unknown'} for key in components},
            'analytical_method': {'status': 'unknown', 'name': None}, 'outcome_status': 'unqualified',
            'quality': {'accepted': None, 'basis': 'Blank template; no measurements or quality assessment recorded'},
            'template_note': 'Fill only actual observations after execution. These null fields are not measured outcomes.'}
        if candidate.get('repeat_of'):
            row['repeat_of'] = candidate['repeat_of']
        rows.append(row)
    return rows


def write_import_package_records(storage, receipt, package_dir):
    """Add new, exact round/receipt and blank entry form to the new package only."""
    path = Path(storage) / 'campaigns' / receipt['campaign_id'] / 'rounds' / (receipt['round_id'] + '.json')
    round_ = json.loads(path.read_text())
    if round_['content_sha256'] != receipt['round_sha256'] or digest({k:v for k,v in round_.items() if k != 'content_sha256'}) != receipt['round_sha256']:
        raise ValueError('Imported frozen round integrity mismatch')
    package_dir = Path(package_dir)
    write_new(package_dir / 'frozen_round.json', round_)
    write_new(package_dir / 'gui_import_receipt.json', receipt)
    write_new(package_dir / 'results_import_template.json', blank_results_template(round_))
