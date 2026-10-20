"""Focused presentation checks for corrected transit/oxygen proposals."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_corrected_design import fixture_config
from khu_sulfoxidation.corrected_design import build_corrected_design
from khu_sulfoxidation.corrected_report import export_corrected_report
from khu_sulfoxidation.provenance import digest


def rows(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))


class CorrectedReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design=build_corrected_design(ROOT,fixture_config())

    def fixture(self,base):
        design=deepcopy(self.design)
        for row in design['conditions']:
            row['rationale']={'hypothesis':'Synthetic hypothesis; no predicted yield.','evidence_ids':['test:evidence:1'],
                'evidence_summary':'Synthetic reported batch endpoints are not flow kinetics.','why_this_condition':'Synthetic coupled target corner.',
                'readout':'Measure residual sulfide, sulfoxide and sulfone separately.','limitation':'Synthetic unvalidated test only.'}
        design['held_fixed_recipe_rationale']=deepcopy(design['conditions'][0]['rationale'])
        design['independent_repeat']['rationale']=deepcopy(design['conditions'][0]['rationale'])
        (base/'design.json').write_text(json.dumps(design))
        out=base/'report';(out/'diagrams').mkdir(parents=True)
        cases=[]
        for row in design['conditions']:
            stem=row['condition_id'];case={'condition_id':stem,'candidate_id':row['candidate_id'],'title':stem+' <synthetic>',
                'png':f'diagrams/{stem}.png','svg':f'diagrams/{stem}.svg','topology_json':f'diagrams/{stem}.json'}
            for kind in ('png','svg'):(out/case[kind]).write_text('SYNTHETIC FIGURE PLACEHOLDER')
            (out/case['topology_json']).write_text(json.dumps({'condition_id':stem,'source_condition_sha256':digest(row)}))
            cases.append(case)
        (out/'render_index.json').write_text(json.dumps({'cases':cases}))
        history={'report_summary':{'coverage_rows':[{'item':'Canonical records','value':116,'interpretation':'Provided once does not prove model use.'}],
            'factor_rows':[{'factor':'synthetic_time','evidence_summary':'Mixed outcomes.','flow_implication':'Explore without monotonic assumptions.',
                'limitations':'Descriptive batch evidence only.','evidence_ids':['test:evidence:1']}],
            'reference_rows':[{'evidence_id':'test:evidence:1','label':'Synthetic evidence','source_location':'synthetic.json#/record/1'}]}}
        (base/'history.json').write_text(json.dumps(history))
        prep={'final_volume_example_mL':10.,'substrate':'thioanisole','substrate_concentration_M':.25,'substrate_mmol_example':2.5,
            'catalyst':'acridine','catalyst_loading_mol_percent':3.,'catalyst_mmol_example':.075,'solvent':'EtOH',
            'pure_equivalent_mass_mg':{'thioanisole':310.5,'acridine':13.4415}}
        (base/'feed.json').write_text(json.dumps(prep))
        return design,cases

    def export(self,base):
        return export_corrected_report(base/'design.json',base/'report',history_path=base/'history.json',feed_path=base/'feed.json',pdf=False)

    def test_exact_transit_and_oxygen_values_have_separate_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);design,_=self.fixture(base);result=self.export(base)
            self.assertEqual(result['condition_count'],6)
            settings,gas=rows(base/'report/conditions.csv'),rows(base/'report/oxygen_delivery.csv')
            self.assertEqual(len(settings),6);self.assertEqual(len(gas),6)
            for proposed,delivery,source in zip(settings,gas,design['conditions']):
                self.assertEqual(float(proposed['target_gas_liquid_transit_min']),source['target']['nominal_mixture_transit_min'])
                self.assertEqual(float(proposed['estimated_gas_liquid_transit_min']),source['calculated']['no_slip_mixture_transit_min'])
                self.assertEqual(float(delivery['O2_within_air_STP_mL_min_calculated']),source['calculated']['oxygen_reference_volume_flow_mL_min'])
                self.assertEqual(float(delivery['oxygen_mmol_min_supplied']),source['calculated']['oxygen_mmol_min_supplied'])
                self.assertEqual(float(delivery['oxygen_equivalents_supplied']),source['calculated']['oxygen_equivalents_supplied'])
                self.assertNotIn('liquid_only_space_time_min',proposed)
                self.assertEqual(proposed['actual_measured_residence_time_min'],'')

    def test_material_rates_include_repeat_without_assigning_duration_or_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);design,_=self.fixture(base);self.export(base)
            material=rows(base/'report/material_rates.csv')
            self.assertEqual(len(material),7)
            by_id={r['condition_id']:r for r in material}
            for source in design['conditions']:
                actual=by_id[source['condition_id']]
                self.assertAlmostEqual(float(actual['feed_mL_per_elapsed_hour']),source['proposed_setpoints']['liquid_flow_mL_min']*60)
                self.assertAlmostEqual(float(actual['substrate_mmol_per_elapsed_hour']),source['calculated']['substrate_mmol_min']*60)
                self.assertAlmostEqual(float(actual['acridine_mmol_per_elapsed_hour']),source['calculated']['catalyst_mmol_min']*60)
                for key in ('actual_feed_duration_min','total_feed_volume_mL','total_substrate_mmol','total_acridine_mmol'):
                    self.assertEqual(actual[key],'')
            repeat=by_id['KD-R01']
            self.assertEqual(repeat['repeat_of'],'KD04')
            for key in ('feed_mL_per_elapsed_hour','substrate_mmol_per_elapsed_hour','acridine_mmol_per_elapsed_hour'):
                self.assertEqual(repeat[key],by_id['KD04'][key])
            text=(base/'report/summary.html').read_text()
            self.assertIn('not a one-hour run protocol',text)
            self.assertIn('equilibration + collection time',text)
            self.assertIn('dead-volume, flushing and loss allowances',text)

    def test_pressure_basis_feed_and_no_new_model_claims_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);self.fixture(base);self.export(base)
            document=(base/'report/summary.html').read_text()
            for phrase in ['1.5 bar','2.51325 bar absolute','gauge','source pressure basis remains unconfirmed','not a second gas stream','No new Astra or council review','310.5','13.4415','Actual bottle purity']:
                self.assertIn(phrase,document)
            self.assertEqual(document.count('<img class="figure"'),6)
            shared={r['item']:r for r in rows(base/'report/shared_settings.csv')}
            self.assertEqual(shared['Actual operating data']['value'],'Unknown')
            self.assertEqual(shared['BPR pressure basis']['value'],'gauge')

    def test_repeat_reuses_corrected_settings_and_sources_are_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);design,_=self.fixture(base);before=(base/'design.json').read_bytes();self.export(base)
            repeat=rows(base/'report/independent_repeat.csv')[0]
            self.assertEqual(repeat['condition_id'],'KD-R01');self.assertEqual(repeat['repeat_of'],'KD04')
            self.assertEqual(float(repeat['liquid_flow_mL_min']),design['conditions'][3]['proposed_setpoints']['liquid_flow_mL_min'])
            self.assertEqual(repeat['execution_order'],'');self.assertEqual((base/'design.json').read_bytes(),before)
            self.assertEqual(len(rows(base/'report/design_evidence.csv')),6)
            with self.assertRaises(FileExistsError):self.export(base)

    def test_missing_evidence_or_mismatched_figure_fails_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);design,cases=self.fixture(base)
            cases[0]['candidate_id']='WRONG';(base/'report/render_index.json').write_text(json.dumps({'cases':cases}))
            with self.assertRaisesRegex(ValueError,'Figure candidate'):self.export(base)
            cases[0]['candidate_id']=design['conditions'][0]['candidate_id'];(base/'report/render_index.json').write_text(json.dumps({'cases':cases}))
            design['conditions'][0]['rationale']['evidence_ids']=['missing'];(base/'design.json').write_text(json.dumps(design))
            with self.assertRaisesRegex(ValueError,'absent from the evidence ledger'):self.export(base)
            self.assertFalse((base/'report/summary.html').exists())

    def test_no_actuals_or_changed_chemistry_silently_enter_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);design,_=self.fixture(base);field=next(iter(design['conditions'][0]['actual']))
            design['conditions'][0]['actual'][field]=1;(base/'design.json').write_text(json.dumps(design))
            with self.assertRaisesRegex(ValueError,'must remain null'):self.export(base)
            design['conditions'][0]['actual'][field]=None;design['scenario']['chemistry']['substrate_concentration_M']=1
            (base/'design.json').write_text(json.dumps(design))
            with self.assertRaisesRegex(ValueError,'feed preparation differs'):self.export(base)
            self.assertFalse((base/'report/conditions.csv').exists())


if __name__=='__main__':unittest.main()
