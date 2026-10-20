"""Regression checks for the corrected figure meanings, values and material path."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import types
import unittest

from khu_sulfoxidation.corrected_design import build_corrected_design
from khu_sulfoxidation.corrected_figures import build_display_topology, configure_captions, render_corrected_figures
from khu_sulfoxidation.provenance import file_hash, write_json
from khu_sulfoxidation.topology_figures import BUILDER, EDGES, _load_file
from scripts.build_corrected_screen import figure_specs

ROOT = Path(__file__).resolve().parents[1]


class CorrectedFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads((ROOT/'configs/corrected_screen_2026-10-13_v1.json').read_text())
        cls.design = build_corrected_design(ROOT, config)
        cls.specs = figure_specs(cls.design)

    def test_primary_time_is_two_phase_estimate_and_bpr_is_selected(self):
        for spec in self.specs:
            graph = build_display_topology(spec)
            ops = {o['op_id']: o for o in graph['unit_operations']}
            p = ops['reactor']['parameters']
            self.assertNotIn('liquid_only_space_time_min', p)
            self.assertNotIn('residence_time_min', p)
            self.assertLess(abs(p['estimated_mixture_transit_min']/p['target_mixture_transit_min']-1), .03)
            self.assertEqual(ops['bpr']['inventory_item_id'], 'bpr_row4')
            self.assertEqual(ops['bpr']['parameters']['pressure_bar'], 1.5)
            self.assertAlmostEqual(ops['bpr']['parameters']['modeled_reactor_pressure_bar_abs'], 2.51325)
            self.assertIsNone(ops['bpr']['parameters']['actual_pressure_bar_abs'])
            self.assertEqual(ops['bpr']['assignment_status'], 'proposed_from_available_inventory')
            self.assertIsNone(ops['collector']['inventory_item_id'])
            self.assertEqual(ops['collector']['assignment_status'], 'unresolved_collection_assignment')
            self.assertIsNone(ops['collector']['actual_assignment_confirmed'])
            self.assertIsNone(graph['actual_outcomes'])
            self.assertEqual(tuple((s['from_op'], s['to_op'], s['stream_type']) for s in graph['streams']), EDGES)

    def test_caption_reports_inlet_oxygen_component_and_explicit_pressure_basis(self):
        before = file_hash(ROOT/BUILDER)
        renderer = _load_file(ROOT/BUILDER, '_test_corrected_caption_renderer')
        configure_captions(renderer)
        graph = build_display_topology(self.specs[0])
        ops = {n['op_id']: types.SimpleNamespace(**n) for n in graph['unit_operations']}
        self.assertIn('O2 in air:', renderer._mfc_label(ops['air']))
        self.assertIn('STP mL/min', renderer._mfc_label(ops['air']))
        self.assertIn('Inlet O2:', renderer._mfc_label(ops['air']))
        self.assertIn('Estimated gas-liquid', renderer._reactor_label(ops['reactor']))
        self.assertNotIn('V/Q', renderer._reactor_label(ops['reactor']))
        self.assertIn('1.5 bar', renderer._bpr_label(ops['bpr']))
        self.assertIn('Gauge basis assumed', renderer._bpr_label(ops['bpr']))
        self.assertEqual(file_hash(ROOT/BUILDER), before)

    def test_tampered_display_flow_rejected_even_with_matching_row_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root/'design.json', self.design)
            specs = deepcopy(self.specs)
            specs[0]['air_STP_mL_min'] += 1
            write_json(root/'specs.json', {'authoritative_design_sha256': file_hash(root/'design.json'), 'cases': specs})
            with self.assertRaisesRegex(ValueError, 'Display value differs'):
                render_corrected_figures(ROOT, root/'design.json', root/'specs.json', root/'out')
            self.assertFalse((root/'out').exists())


if __name__ == '__main__':
    unittest.main()
