"""Focused offline checks; calculation consistency is not laboratory validation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from khu_sulfoxidation.flowpilotx_profile import build_profile, EDGES
from khu_sulfoxidation.flowpilotx_report import export_review, condition_rows
from khu_sulfoxidation.history import ingest_history
from khu_sulfoxidation.campaign_evidence import build_campaign_evidence
from khu_sulfoxidation.corrected_design import calculate_coupled_values
from flowpilotx.inventory import validate_inventory,validate_requirements,candidate_checks
from flowpilotx.evidence import validate_evidence,validate_question


class FlowPilotXKhuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.p=build_profile(ROOT)

    def test_six_preferred_twelve_unique_with_separate_repeat(self):
        p=self.p
        self.assertEqual(p['default_selected_candidate_ids'],[f'KX{i:02d}' for i in range(1,7)])
        self.assertEqual(len(p['candidates']),12)
        self.assertEqual(len({(c['parameters']['liquid_mL_min'],c['parameters']['oxygen_STP_mL_min']) for c in p['candidates']}),12)
        self.assertEqual([(c['parameters']['liquid_mL_min'],c['parameters']['oxygen_STP_mL_min']) for c in p['candidates'][:6]],
            [(.45,1.26),(.29,1.62),(.22,.62),(.14,.8),(.11,.31),(.07,.4)])
        self.assertEqual(p['optional_repeat']['repeat_of'],'KX04')
        self.assertFalse(p['optional_repeat']['counts_as_unique_design'])

    def test_independent_inlet_and_expanded_gas_arithmetic(self):
        for c in self.p['candidates']:
            p=c['parameters']
            oxygen=1.01325*p['oxygen_STP_mL_min']/(.0831446261815324*273.15)
            eq=oxygen/(.25*p['liquid_mL_min'])
            qg=p['oxygen_STP_mL_min']*1.01325/2.51325*298.15/273.15
            time=5/(p['liquid_mL_min']+qg)
            self.assertAlmostEqual(eq,p['oxygen_eq'])
            self.assertAlmostEqual(time,p['transit_min'])
            self.assertLessEqual(abs(time/p['target_time_min']-1),.03)
            self.assertLessEqual(abs(eq/p['target_oxygen_eq']-1),.03)
            for value in [p['liquid_mL_min'],p['oxygen_STP_mL_min']]:
                self.assertAlmostEqual(value*100,round(value*100))
            self.assertTrue(all(x is None for x in c['actual'].values()))
        s=deepcopy(self.p['scenario'])
        a=calculate_coupled_values(s,.29,1.62)
        s['pressure']['reactor_pressure_bar_abs_assumed']=1.5
        b=calculate_coupled_values(s,.29,1.62)
        self.assertEqual(a['oxygen_equivalents_supplied'],b['oxygen_equivalents_supplied'])
        self.assertNotEqual(a['no_slip_mixture_transit_min'],b['no_slip_mixture_transit_min'])

    def test_core_schema_and_release_gates(self):
        p=self.p
        validate_inventory(p['inventory']); validate_requirements(p['requirements']); validate_evidence(p['evidence'])
        for q in p['questions']:validate_question(q)
        for c in p['candidates']:
            self.assertFalse(any(x['blocking'] for x in candidate_checks(c,p['inventory'],p['requirements'])))
        self.assertFalse(p['laboratory_released'])
        self.assertTrue(all(r['status']=='unresolved' for r in p['requirements'] if r['scope']=='release'))

    def test_history_lossless_separate_and_compact_complete(self):
        original=ingest_history(ROOT)
        self.assertEqual(self.p['history_archive'],original)
        compact=self.p['model_evidence']['homogeneous_batch_evidence']['raw_records']
        self.assertEqual([r[0] for r in compact['rows']],list(range(1,117)))
        self.assertEqual(self.p['evidence_coverage']['source_representations'],311)
        self.assertEqual(len(original['systems']['heterogeneous']['records']),42)
        self.assertTrue(any(r['outcomes']['sulfoxide_pct']>100 for r in original['systems']['heterogeneous']['records']))
        self.assertNotIn('heterogeneous_batch_evidence',self.p['model_evidence'])
        reference=build_campaign_evidence(ROOT)['informed_history_payload']['raw_records']
        for field in ['conditions','reported_outcomes','analytical_flag_codes','source_rows']:
            compact_index=compact['row_columns'].index(field)
            original_index=reference['row_columns'].index(field)
            self.assertEqual([r[compact_index] for r in compact['rows']],[r[original_index] for r in reference['rows']])
        self.assertNotIn('calculated_outcomes',compact['row_columns'])

    def test_loop_topology_and_unknowns_are_not_assigned(self):
        for c in self.p['candidates']:
            diagram=c['diagram'];nodes={n['op_id']:n for n in diagram['unit_operations']}
            self.assertEqual({(s['from_op'],s['to_op'],s['stream_type']) for s in diagram['streams']},set(EDGES))
            for name in ['injector','sample_loop','collector']:
                self.assertIsNone(nodes[name]['inventory_item_id'])
                self.assertEqual(nodes[name]['assignment_status'],'unresolved_equipment_assignment')
            self.assertIsNone(nodes['sample_loop']['parameters']['volume_mL'])
            self.assertEqual(nodes['bpr']['inventory_item_id'],'bpr_row4')
            self.assertEqual(nodes['reactor']['inventory_item_id'],'coil_row4')
            self.assertEqual(nodes['oxygen']['parameters']['oxygen_mole_fraction_assumed'],1.)
        self.assertEqual(len({c['rationale'] for c in self.p['candidates']}),12)

    def test_coil_alternatives_and_all_attempts_preserved(self):
        alts=self.p['engineering_alternatives']
        self.assertEqual(len(self.p['candidate_attempts'])+sum(len(a['attempts']) for a in alts),24)
        small=next(a for a in alts if a['equipment_id']=='coil_row3')
        self.assertEqual(len([a for a in small['attempts'] if a['calculated'] is None]),2)
        self.assertTrue(all(a['rejection_reasons'] for a in small['attempts'] if a['calculated'] is None))
        large=next(a for a in alts if a['equipment_id']=='coil_row5')
        self.assertEqual(large['disposition'],'feasible_not_selected')
        self.assertFalse(large['chemical_rejection'])

    def test_export_immutable_and_blank_actuals(self):
        with tempfile.TemporaryDirectory() as td:
            export_review(self.p,td)
            profile=json.loads((Path(td)/'profile.json').read_text())
            self.assertEqual(profile,self.p)
            form=json.loads((Path(td)/'results_template.json').read_text())
            self.assertEqual(len(form['experiments']),7)
            for row in form['experiments']:
                self.assertTrue(all(v is None for v in row['actual_conditions'].values()))
                self.assertTrue(all(v is None for v in row['analytical_results'].values()))
            with self.assertRaises(FileExistsError):export_review(self.p,td)
        with self.assertRaises(ValueError):condition_rows(self.p,['KX01','KX01'])

    def test_stock_scaling_and_no_total_or_loop_invention(self):
        m=self.p['materials']
        self.assertAlmostEqual(m['sample_stock']['illustrative_substrate_pure_equivalent_mg'],310.5)
        self.assertAlmostEqual(m['sample_stock']['illustrative_catalyst_pure_equivalent_mg'],13.4415)
        self.assertIsNone(m['loop_accounting']['actual_loop_volume_mL'])
        self.assertIsNone(m['campaign_total_material_requirement'])
        self.assertIsNone(m['sample_stock']['actual_purity_fractions'])

    def test_observation_contract_requires_actual_controls_not_targets(self):
        contract=self.p['observation_contract']
        required=set(contract['required_actual_conditions'])
        self.assertTrue({'carrier_flow_mL_min','oxygen_flow_reference_mL_min','loop_volume_mL','fraction_start_min','MFC_calibration'}<=required)
        self.assertFalse(required & set(contract['excluded_from_measurement_requirements']))
        self.assertEqual(contract['parameter_mapping']['liquid_mL_min'],'carrier_flow_mL_min')
        self.assertEqual(contract['parameter_mapping']['oxygen_STP_mL_min'],'oxygen_flow_reference_mL_min')
        self.assertTrue(all(v is None for v in contract['actual_conditions_template'].values()))
        rules={r['parameter']:r['value'] for r in contract['comparability_requirements']}
        self.assertEqual(rules['concentration_M'],.25)
        self.assertEqual(rules['catalyst_loading_mol_percent'],3.)
        from khu_sulfoxidation.flowpilotx_report import results_template
        form=results_template(self.p,['KX01'])
        self.assertTrue(required<=set(form['experiments'][0]['actual_conditions']))
        self.assertTrue(all(v is None for v in form['experiments'][0]['actual_conditions'].values()))


if __name__=='__main__':unittest.main()
