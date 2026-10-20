"""Build the offline correction package without invoking a paid model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

from khu_sulfoxidation.corrected_design import build_corrected_design
from khu_sulfoxidation.history_hypotheses import build_history_hypothesis_ledger, attach_design_rationales
from khu_sulfoxidation.provenance import digest, file_hash, write_json


def figure_specs(design):
    scenario = design['scenario']
    pressure, gas = scenario['pressure'], scenario['gas_reference']
    result = []
    for row in design['conditions']:
        p, c, target = row['proposed_setpoints'], row['calculated'], row['target']
        result.append({
            'condition_id': row['condition_id'], 'source_condition_sha256': digest(row),
            'chemistry': scenario['chemistry'], 'equipment_ids': design['configuration']['selected_equipment'],
            'liquid_flow_mL_min': p['liquid_flow_mL_min'],
            'air_STP_mL_min': p['gas_reference_flow_mL_min'],
            'oxygen_STP_mL_min': c['oxygen_reference_volume_flow_mL_min'],
            'inlet_oxygen_equivalents': c['oxygen_equivalents_supplied'],
            'target_inlet_oxygen_equivalents': target['inlet_oxygen_equivalents'],
            'target_mixture_transit_min': target['nominal_mixture_transit_min'],
            'estimated_mixture_transit_min': c['no_slip_mixture_transit_min'],
            'reactor_volume_mL': scenario['reactor_volume_mL'], 'reactor_ID_mm': scenario['reactor_ID_mm'],
            'temperature_C': scenario['temperature_C'],
            'wavelength_nm': scenario['selected_equipment']['light']['wavelength_nm'],
            'bpr_label_pressure_bar': pressure['label_value_bar'],
            'bpr_basis': pressure['label_basis_assumption'],
            'modeled_reactor_pressure_bar_abs': pressure['reactor_pressure_bar_abs_assumed'],
            'reference_temperature_K': gas['temperature_K'],
            'reference_pressure_bar_abs': gas['pressure_bar_abs'],
            'oxygen_mole_fraction': gas['oxygen_mole_fraction'],
        })
    return result


def build(root, config_path, output):
    root, output, config_path = Path(root).resolve(), Path(output).resolve(), Path(config_path).resolve()
    if output.exists():
        raise FileExistsError('Choose a fresh output version; previous proposals are preserved')
    config = json.loads(config_path.read_text())
    campaign = root / 'outputs/khu_astra_sol_campaign_2026-10-13_v1'
    original_report = root / 'outputs/khu_first_screen_astra_sol_2026-10-13_v2'
    source_guard = {str(path): file_hash(path) for path in [config_path, campaign/'input_freeze.json', campaign/'nominations_freeze.json']}
    output.mkdir(parents=True)
    try:
        engineering = build_corrected_design(root, config)
        write_json(output/'engineering_selection.json', engineering)
        if any(row.get('status') != 'conditional_proposal' for row in engineering['conditions']):
            raise ValueError('One or more targets could not be realized; review preserved candidate attempts')
        history = build_history_hypothesis_ledger(campaign)
        write_json(output/'history_hypotheses.json', history)
        design = attach_design_rationales(engineering, history)
        design['revision_provenance'] = {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'configuration_file': str(config_path.relative_to(root)), 'configuration_file_sha256': file_hash(config_path),
            'history_ledger_sha256': file_hash(output/'history_hypotheses.json'),
            'original_campaign_input_sha256': file_hash(campaign/'input_freeze.json'),
            'original_campaign_nominations_sha256': file_hash(campaign/'nominations_freeze.json'),
            'new_model_calls': 0, 'fresh_model_selection': False, 'flow_outcomes_seen': False,
            'comparison_status': 'Original matched-arm nominations preserved as withdrawn history. No replacement history-omitted nominations or model ranking are claimed.'}
        write_json(output/'corrected_design.json', design)
        write_json(output/'figure_specs.json', {'schema_version': 'khu_corrected_figure_specs_v1',
            'authoritative_design_sha256': file_hash(output/'corrected_design.json'), 'cases': figure_specs(design)})
        shutil.copyfile(config_path, output/'configuration.json')
        shutil.copyfile(original_report/'feed_preparation.json', output/'feed_preparation.json')
        shutil.copyfile(root/'provenance/design_issue_time_basis_2026-10-13_v1.json', output/'prior_design_issue.json')
        source_copy = output/'source_history'
        source_copy.mkdir()
        for name in ['input_freeze.json','nominations_freeze.json']:
            shutil.copyfile(campaign/name, source_copy/name)
        shutil.copytree(campaign/'history_informed', source_copy/'history_informed')
        with (original_report/'results_template.csv').open(newline='') as stream:
            columns = next(csv.reader(stream))
        with (output/'results_template.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in design['conditions']:
                writer.writerow({'condition_id': row['condition_id'], 'candidate_id': row['condition_id']})
            repeat = design['independent_repeat']
            writer.writerow({'condition_id': repeat['condition_id'], 'candidate_id': repeat['repeat_of'], 'repeat_of_condition_id': repeat['repeat_of']})
        if any(file_hash(path) != expected for path, expected in source_guard.items()):
            raise ValueError('Source changed during offline package construction')
        return design
    except Exception as exc:
        write_json(output/'build_failure.json', {'error': str(exc), 'partial_attempt_preserved': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = build(Path(__file__).resolve().parents[1], args.config, args.output)
    print(f"Prepared {len(result['conditions'])} corrected proposals, independent repeat, complete evidence ledger and results template; no model calls.")
