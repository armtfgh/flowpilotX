"""Synthetic offline import/asset boundaries. No model or network calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from flowpilotx.campaign import CampaignStore
from flowpilotx.evidence import digest
from flowpilotx.review_import import (prepare_review_import, commit_review_import,
    registered_asset, prepare_asset_package, write_new, blank_results_template, write_import_package_records)
from test_flowpilotx_core import synthetic_spec


class ReviewImportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.storage = self.root / 'storage'
        self.package = self.root / 'outputs' / 'synthetic-package-v1'
        self.package.mkdir(parents=True)
        self.spec = synthetic_spec()
        cases = []
        for candidate in self.spec['candidates']:
            cid = candidate['id']
            candidate['diagram_url'] = '/assets/khu/diagrams/old.svg'
            case = {'condition_id': cid, 'source_parameters_sha256': digest(candidate['parameters'])}
            for key, ext in [('svg', 'svg'), ('png', 'png'), ('topology_json', 'json')]:
                rel = 'diagrams/' + cid + '.' + ext
                file = self.package / rel
                file.parent.mkdir(exist_ok=True)
                file.write_text(cid)
                case[key] = rel
            cases.append(case)
        write_new(self.package / 'render_index.json', {'cases': cases})
        self.source = self.root / 'source.json'
        write_new(self.source, {'fixture': True})
        self.review = {'parsed': {'selected_candidate_ids': ['E1', 'E3']}, 'call_count': 7,
            'scientific_review': {'status': 'conditional_scientific_review_pass',
              'laboratory_released': False, 'experimental_validation': False}}

    def plan(self):
        return prepare_review_import(self.root, self.package, 'synthetic-v1', self.spec,
            ['E1', 'E3'], 'E3', self.review, [self.source])

    def test_import_preserves_existing_campaign_and_exact_parameters(self):
        store = CampaignStore(self.storage / 'campaigns')
        prior = store.create(self.spec)
        prior_snapshot = deepcopy(store.get(prior['id']))
        plan = self.plan()
        self.assertEqual(len(store.list()), 1, 'preparation must not mutate storage')
        receipt = commit_review_import(self.root, self.storage, plan)
        imported = store.get(receipt['campaign_id'])
        self.assertEqual(store.get(prior['id']), prior_snapshot)
        self.assertEqual([c['id'] for c in imported['last_round']['selected']], ['E1', 'E3'])
        self.assertEqual(imported['last_round']['repeats'][0]['repeat_of'], 'E3')
        self.assertEqual([q['id'] for q in imported['spec']['questions']], [q['id'] for q in self.spec['questions']])
        self.assertTrue(all(q['status']=='unresolved' for q in imported['spec']['questions']))
        self.assertFalse(imported['laboratory_released'])
        self.assertEqual(receipt['new_model_calls'], 0)
        self.assertFalse((self.storage / 'default_campaign.json').exists())
        for candidate in imported['spec']['candidates']:
            original = next(c for c in self.spec['candidates'] if c['id'] == candidate['id'])
            self.assertEqual(candidate['parameters'], original['parameters'])
            self.assertTrue(candidate['diagram_url'].startswith('/assets/packages/synthetic-v1/'))
        with self.assertRaises(FileExistsError): commit_review_import(self.root, self.storage, plan)

    def test_pending_audit_and_unknown_selected_id_are_rejected(self):
        self.review['scientific_review']['status'] = 'pending'
        with self.assertRaisesRegex(ValueError, 'Completed scientific review'): self.plan()
        self.review['scientific_review']['status'] = 'failed_scientific_review'
        plan = self.plan()
        self.assertEqual(plan['request']['model_review']['scientific_review']['status'], 'failed_scientific_review')
        with self.assertRaisesRegex(ValueError, 'no matching versioned figure'):
            prepare_asset_package(self.root, self.package, 'fixture', self.spec['candidates'], ['absent'])

    def test_parameter_drift_or_source_drift_prevents_import(self):
        self.spec['candidates'][0]['parameters']['feed_mL_min'] = 9
        with self.assertRaisesRegex(ValueError, 'Figure parameters'): self.plan()
        self.spec['candidates'][0]['parameters']['feed_mL_min'] = 1.0
        plan = self.plan()
        self.source.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Source changed'): commit_review_import(self.root, self.storage, plan)
        self.assertFalse((self.storage / 'campaigns').exists())

    def test_result_entry_form_uses_bound_round_and_null_actuals_with_separate_repeat(self):
        receipt = commit_review_import(self.root, self.storage, self.plan())
        store = CampaignStore(self.storage / 'campaigns')
        round_ = store.get(receipt['campaign_id'])['last_round']
        round_['configuration']['observation_contract'] = {
            'actual_conditions_template': {'temperature_C': 50, 'context': {'pressure_bar_abs': 2}},
            'required_actual_conditions': ['temperature_C', 'context.pressure_bar_abs', 'feed_mL_min']}
        rows = blank_results_template(round_)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r['design_id'] for r in rows], ['E1', 'E3', 'independent-repeat-1'])
        self.assertEqual(rows[-1]['repeat_of'], 'E3')
        self.assertTrue(all(r['round_id'] == receipt['round_id'] for r in rows))
        self.assertTrue(all(r['actual_conditions'] == {'temperature_C': None, 'context': {'pressure_bar_abs': None}, 'feed_mL_min': None} for r in rows))
        self.assertTrue(all(m['value'] is None for r in rows for m in r['analytes'].values()))
        self.assertTrue(all(r['outcome_status'] == 'unqualified' and r['quality']['accepted'] is None for r in rows))
        write_import_package_records(self.storage, receipt, self.package)
        frozen = json.loads((self.package / 'frozen_round.json').read_text())
        self.assertEqual(frozen['content_sha256'], receipt['round_sha256'])
        self.assertEqual(len(json.loads((self.package / 'results_import_template.json').read_text())), 3)
        self.assertEqual(store.get(receipt['campaign_id'])['results'], [])

    def test_assets_reject_traversal_unregistered_and_changed_files(self):
        commit_review_import(self.root, self.storage, self.plan())
        file = registered_asset(self.root, self.storage, 'synthetic-v1', 'diagrams/E1.svg')
        self.assertEqual(file.read_text(), 'E1')
        for relative in ['../source.json', 'diagrams/../../source.json', 'source.json', 'diagrams/absent.svg']:
            with self.assertRaises(ValueError): registered_asset(self.root, self.storage, 'synthetic-v1', relative)
        file.write_text('modified')
        with self.assertRaisesRegex(ValueError, 'integrity'): registered_asset(self.root, self.storage, 'synthetic-v1', 'diagrams/E1.svg')

if __name__ == '__main__': unittest.main()
