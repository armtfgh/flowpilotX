"""Regression for explicit nested citations missed by the first Astra checker."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from khu_sulfoxidation.flowpilotx_astra_reconcile import visible_sources, validate_response

ROOT = Path(__file__).resolve().parents[1]

class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = json.loads((ROOT/'outputs/flowpilotx_astra_preflight_2026-10-15_v1/frozen_inputs.json').read_text())
        cls.response = json.loads((ROOT/'outputs/flowpilotx_astra_sol_2026-10-15_v1/Upstream.json').read_text())['response']

    def test_exact_explicit_sources_are_visible(self):
        registry = visible_sources(self.frozen['context'])
        self.assertEqual(registry['khu:slide:2:table-entry:14'], ['/context/evidence/homogeneous_batch_evidence/slide_table_evidence/rows/13/0'])
        for ident in ['ali_2024', 'casado_sanchez_2016', 'freeburne_pester_2024_counterexample',
                'khu:homogeneous:campaign-evidence:v1', 'khu:slide:2:table-entry:16']:
            self.assertIn(ident, registry)
        self.assertTrue(validate_response('Upstream', self.response, self.frozen)['passed'])

    def test_real_alias_or_external_source_cannot_be_invented(self):
        for ident in ['khu:slide:2:table-entry:99', 'ali_2099', 'khu:heterogeneous:raw-entry:0001']:
            response = deepcopy(self.response)
            response['design_rationales'][0]['evidence_ids'] = [ident]
            self.assertFalse(validate_response('Upstream', response, self.frozen)['passed'])

    def test_mentioned_citations_without_source_records_do_not_authorize_themselves(self):
        frozen = deepcopy(self.frozen)
        frozen['context']['evidence']['unverified_mentions'] = {'evidence_ids': ['fabricated']}
        self.assertNotIn('fabricated', visible_sources(frozen['context']))

    def test_candidate_and_rectangle_validation_stays_in_force(self):
        response = deepcopy(self.response); response['selected_candidate_ids'][0] = 'nonexistent'
        self.assertFalse(validate_response('Upstream', response, self.frozen)['passed'])
        response = deepcopy(self.response); response['selected_candidate_ids'][0] = 'KXA03'
        self.assertFalse(validate_response('Upstream', response, self.frozen)['passed'])

if __name__ == '__main__': unittest.main()
