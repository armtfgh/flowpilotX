"""Original FlowPilot icons with explicit inlet-STP and two-phase time labels.

The input is a display projection of a separately hashed engineering design.
No model calls, equipment operation or modifications to the baseline occur.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import types
import xml.etree.ElementTree as ET

from .provenance import digest, file_hash, write_json
from .topology_figures import BUILDER, ARTIFACTS, ICONS, EDGES, _load_file


def build_display_topology(spec):
    s = deepcopy(spec)
    chemistry = s['chemistry']
    def node(identity, kind, label, parameters, equipment=None):
        assignment = ('proposed_from_available_inventory' if equipment else
                      'unresolved_collection_assignment' if kind == 'collector' else
                      'unresolved_equipment_assignment')
        return {'op_id': identity, 'op_type': kind, 'label': label, 'parameters': parameters,
                'inventory_item_id': equipment, 'assignment_status': assignment,
                'actual_assignment_confirmed': None}
    operations = [
        node('feed', 'pump', 'Premixed liquid feed', {
            'stream': 'A', 'phase': 'liquid', 'instrument_name': 'R-series HPLC',
            'contents': [chemistry['substrate'].capitalize(), f"{chemistry['catalyst'].capitalize()}: {chemistry['catalyst_loading_mol_percent']:g} mol%"],
            'solvent': chemistry['solvent'], 'concentration_M': chemistry['substrate_concentration_M'],
            'flow_rate_mL_min': s['liquid_flow_mL_min']}, s['equipment_ids']['pump_id']),
        node('air', 'mfc', 'Inlet air', {
            'stream': 'B', 'phase': 'gas', 'instrument_name': 'FF-C00', 'contents': ['Air'],
            'gas_flow_sccm': s['air_STP_mL_min'], 'oxygen_STP_mL_min': s['oxygen_STP_mL_min'],
            'inlet_oxygen_equivalents': s['inlet_oxygen_equivalents'],
            'reference_temperature_K': s['reference_temperature_K'],
            'reference_pressure_bar_abs': s['reference_pressure_bar_abs'],
            'oxygen_mole_fraction': s['oxygen_mole_fraction']}, s['equipment_ids']['mfc_id']),
        node('check', 'check_valve', 'Gas check valve', {
            'instrument_name': 'CV-3301', 'method': '1 bar cracking differential'}, s['equipment_ids']['check_valve_id']),
        node('mixer', 't_mixer', 'Gas-liquid T-mixer', {'instrument_name': 'P-713'}, s['equipment_ids']['mixer_id']),
        node('reactor', 'photoreactor', 'Homogeneous photoreactor', {
            'instrument_name': 'PFA coil', 'light_instrument_name': 'UV-150',
            'temperature_C': s['temperature_C'], 'wavelength_nm': s['wavelength_nm'],
            'volume_mL': s['reactor_volume_mL'], 'ID_mm': s['reactor_ID_mm'],
            'target_mixture_transit_min': s['target_mixture_transit_min'],
            'estimated_mixture_transit_min': s['estimated_mixture_transit_min'],
            'actual_liquid_contact_time_min': None}, s['equipment_ids']['coil_id']),
        node('bpr', 'bpr', 'Fixed BPR', {
            'instrument_name': 'Fixed BPR', 'pressure_bar': s['bpr_label_pressure_bar'],
            'proposed_label_basis': s['bpr_basis'],
            'modeled_reactor_pressure_bar_abs': s['modeled_reactor_pressure_bar_abs'],
            'pressure_basis_status': 'explicit_interpretation_not_recorded_in_inventory',
            'actual_pressure_bar_abs': None}, s['equipment_ids']['bpr_id']),
        node('collector', 'collector', 'Sample collection', {
            'actual_collection_window': None, 'sample_handling': None}),
    ]
    return {'schema_version': 'khu_corrected_display_topology_v1',
            'topology_id': s['condition_id'], 'condition_id': s['condition_id'],
            'source_condition_sha256': s['source_condition_sha256'],
            'compilation_status': 'proposed_display_not_equipment_control',
            'unit_operations': operations,
            'streams': [{'stream_id': f's{i:02d}', 'from_op': a, 'to_op': b,
                         'stream_type': phase, 'connection_type': 'process'}
                        for i, (a, b, phase) in enumerate(EDGES, 1)],
            'display_quantities': s,
            'oxygen_equivalent_basis': 'inlet oxygen molar flow / inlet substrate molar flow, using stated STP; no channel oxygen inventory',
            'time_basis': 'nominal reactor volume / combined in-reactor gas and liquid volume rates under explicit ideal-gas/no-slip assumptions',
            'actual_conditions': None, 'actual_outcomes': None,
            'measurement_note': 'Calculated inlet delivery and estimated transit; no measured residence or outcome is claimed.'}


def configure_captions(renderer):
    base_reactor = renderer._reactor_label
    def reactor(op):
        p = op.parameters
        return (base_reactor(op) + f"\nEstimated gas-liquid\ntransit: {p['estimated_mixture_transit_min']:.2f} min"
                + f"\nTarget: {p['target_mixture_transit_min']:g} min")
    def mfc(op):
        p = op.parameters
        return '<BR/>'.join([
            renderer._font_lines('Inlet air / FF-C00', bold=True),
            renderer._font_lines(f"Air: {p['gas_flow_sccm']:.2f} STP mL/min", 10, '#B91C1C'),
            renderer._font_lines(f"O2 in air: {p['oxygen_STP_mL_min']:.4f}\nSTP mL/min (calculated)", 9),
            renderer._font_lines(f"Inlet O2: {p['inlet_oxygen_equivalents']:.3f} equivalents", 10),
            renderer._font_lines(f"STP: {p['reference_temperature_K']-273.15:g} C / {p['reference_pressure_bar_abs']:.5f} bar abs", 8),
        ])
    def bpr(op):
        p = op.parameters
        basis = 'gauge' if p['proposed_label_basis'] == 'gauge' else 'absolute'
        return (f"Fixed BPR: {p['pressure_bar']:g} bar\n{basis.capitalize()} basis assumed"
                + f"\nModel: {p['modeled_reactor_pressure_bar_abs']:.3f} bar abs")
    renderer._reactor_label, renderer._mfc_label, renderer._bpr_label = reactor, mfc, bpr


def render_corrected_figures(root, design_path, specs_path, output_dir):
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    design_path, specs_path = Path(design_path), Path(specs_path)
    bundle = json.loads(specs_path.read_text())
    design = json.loads(design_path.read_text())
    if file_hash(design_path) != bundle['authoritative_design_sha256']:
        raise ValueError('Display specification does not match authoritative corrected design')
    specs = bundle['cases']
    if [s['condition_id'] for s in specs] != [f'KD{i:02d}' for i in range(1, 7)]:
        raise ValueError('Exactly six corrected core figures are required')
    by_id = {row['condition_id']: row for row in design['conditions']}
    for s in specs:
        row = by_id[s['condition_id']]
        if digest(row) != s['source_condition_sha256']:
            raise ValueError('Display condition is not bound to authoritative row')
        scenario = design['scenario']
        expected = {
            'liquid_flow_mL_min': row['proposed_setpoints']['liquid_flow_mL_min'],
            'air_STP_mL_min': row['proposed_setpoints']['gas_reference_flow_mL_min'],
            'oxygen_STP_mL_min': row['calculated']['oxygen_reference_volume_flow_mL_min'],
            'inlet_oxygen_equivalents': row['calculated']['oxygen_equivalents_supplied'],
            'target_inlet_oxygen_equivalents': row['target']['inlet_oxygen_equivalents'],
            'target_mixture_transit_min': row['target']['nominal_mixture_transit_min'],
            'estimated_mixture_transit_min': row['calculated']['no_slip_mixture_transit_min'],
            'reactor_volume_mL': scenario['reactor_volume_mL'], 'reactor_ID_mm': scenario['reactor_ID_mm'],
            'temperature_C': scenario['temperature_C'],
            'wavelength_nm': scenario['selected_equipment']['light']['wavelength_nm'],
            'bpr_label_pressure_bar': scenario['pressure']['label_value_bar'],
            'bpr_basis': scenario['pressure']['label_basis_assumption'],
            'modeled_reactor_pressure_bar_abs': scenario['pressure']['reactor_pressure_bar_abs_assumed'],
            'reference_temperature_K': scenario['gas_reference']['temperature_K'],
            'reference_pressure_bar_abs': scenario['gas_reference']['pressure_bar_abs'],
            'oxygen_mole_fraction': scenario['gas_reference']['oxygen_mole_fraction'],
            'chemistry': scenario['chemistry'], 'equipment_ids': design['configuration']['selected_equipment'],
        }
        for key, value in expected.items():
            if s[key] != value:
                raise ValueError(f'Display value differs from authoritative design: {key}')
    for name in ['diagrams', 'render_runs', 'assets', 'renderer_source', 'render_index.json']:
        if (output_dir / name).exists():
            raise FileExistsError('Use a new package version; prior render artifacts are immutable')
    output_dir.mkdir(parents=True, exist_ok=True)
    before = {str(p.relative_to(root)): file_hash(p) for p in
              [root / BUILDER, root / ARTIFACTS, *sorted((root / ICONS).glob('*'))] if p.is_file()}
    copied_icons = output_dir / 'assets/icons'
    shutil.copytree(root / ICONS, copied_icons)
    (output_dir / 'renderer_source').mkdir()
    for name in [BUILDER, ARTIFACTS]:
        shutil.copy2(root / name, output_dir / 'renderer_source' / Path(name).name)
    renderer = _load_file(root / BUILDER, '_khu_corrected_original_renderer')
    artifacts = _load_file(root / ARTIFACTS, '_khu_corrected_artifacts')
    renderer.ICONS_DIR = copied_icons
    renderer.ASSETS = {k: copied_icons / p.name for k, p in renderer.ASSETS.items()}
    configure_captions(renderer)
    old_cache = os.environ.get('XDG_CACHE_HOME')
    os.environ['XDG_CACHE_HOME'] = str(output_dir / 'render_cache')
    cases = []
    try:
        if not renderer._ensure_graphviz_on_path():
            raise RuntimeError('Graphviz required for the original FlowPilot figure style')
        class Builder(renderer.FlowsheetBuilder):
            def build(self, data, **kwargs):
                obj = types.SimpleNamespace(
                    unit_operations=[types.SimpleNamespace(**deepcopy(o)) for o in data['unit_operations']],
                    streams=[types.SimpleNamespace(**deepcopy(s)) for s in data['streams']])
                return super().build(obj, **kwargs)
            def _legacy_fallback(self, *args, **kwargs):
                raise RuntimeError('Diagnostic fallback is not a topology deliverable')
        for spec in specs:
            topology = build_display_topology(spec)
            title = f"{spec['condition_id']} - {spec['target_mixture_transit_min']:g} min target / {spec['target_inlet_oxygen_equivalents']:g} inlet O2 eq"
            builder = Builder()
            result = artifacts.render_topology_artifacts(topology, title=title,
                base_dir=output_dir / 'render_runs' / spec['condition_id'], builder=builder)
            if result['renderer'] != 'graphviz' or result['render_status'] != 'complete' or result['warnings']:
                raise ValueError(f'Original renderer did not complete cleanly: {result}')
            svg = ET.parse(result['svg_path']).getroot()
            images = list(svg.iter('{http://www.w3.org/2000/svg}image'))
            if len(images) != 7 or any(not im.get('{http://www.w3.org/1999/xlink}href', im.get('href', '')).startswith('data:image/') for im in images):
                raise ValueError('Expected seven embedded original FlowPilot icons')
            edges = [g.find('{http://www.w3.org/2000/svg}title').text for g in svg.iter('{http://www.w3.org/2000/svg}g') if g.get('class') == 'edge']
            actual = {'->'.join(part.split(':', 1)[0] for part in e.split('->')) for e in edges}
            if actual != {a+'->'+b for a, b, _ in EDGES}:
                raise ValueError('Material connections differ from the corrected apparatus')
            (output_dir / 'diagrams').mkdir(exist_ok=True)
            files = {}
            for field, suffix, key in [('svg_path','svg','svg'),('png_path','png','png'),('topology_path','json','topology_json')]:
                dest = output_dir / 'diagrams' / f"{spec['condition_id']}.{suffix}"
                shutil.copyfile(result[field], dest)
                files[key] = str(dest.relative_to(output_dir))
            cases.append({'condition_id': spec['condition_id'], 'title': title, **files,
                'topology_sha256': result['topology_sha256'],
                'render_manifest': str(Path(result['render_manifest_path']).relative_to(output_dir)),
                'renderer': result['renderer'], 'main_process_path': builder.last_render_info['main_process_path']})
        if file_hash(design_path) != bundle['authoritative_design_sha256']:
            raise ValueError('Authoritative design changed during rendering')
        if any(file_hash(root / p) != h for p, h in before.items()):
            raise ValueError('Preserved original renderer or icons changed')
        index = {'schema_version': 'khu_corrected_figures_v1', 'cases': cases,
                 'authoritative_design_sha256': file_hash(design_path), 'figure_specs_sha256': file_hash(specs_path),
                 'original_renderer_and_icon_hashes': before,
                 'adapter_sha256': file_hash(Path(__file__)), 'new_model_calls': 0,
                 'laboratory_released': False, 'experimental_validation': False}
        write_json(output_dir / 'render_index.json', index)
        return index
    except Exception as exc:
        write_json(output_dir / 'render_failure.json', {'error': str(exc), 'attempt_preserved': True})
        raise
    finally:
        if old_cache is None:
            os.environ.pop('XDG_CACHE_HOME', None)
        else:
            os.environ['XDG_CACHE_HOME'] = old_cache
