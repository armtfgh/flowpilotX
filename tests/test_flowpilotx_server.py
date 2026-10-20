"""Boundary tests using synthetic fixtures; no live model or laboratory calls."""
from copy import deepcopy
import base64
import json
from pathlib import Path
import tempfile
import unittest
from flowpilotx.server import Application
from flowpilotx.gateway import ModelGateway
from flowpilotx.campaign import CampaignError
from test_flowpilotx_core import synthetic_spec

ROOT=Path(__file__).resolve().parents[1]

class ServerTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.app=Application(ROOT,self.temp.name)

    def test_gui_alias_and_objective_are_resolved_before_freeze(self):
        c=self.app.store.create(synthetic_spec())
        r=self.app.generate(c['id'],{'requested_count':2,'repeat_count':0,'instructions':'Observe separately',
            'objective_mode':'generalize','model_route':'deterministic'})
        self.assertEqual(r['request']['researcher_instructions'],'Observe separately')
        self.assertNotIn('objective_mode',r['request'])
        self.assertEqual(r['configuration']['objective']['mode'],'generalize')
        self.assertEqual(r['request']['expected_campaign_version'],2)

    def test_concurrent_change_during_model_wait_rejects_old_context(self):
        c=self.app.store.create(synthetic_spec())
        plan={'selected_candidate_ids':['E1'],'design_rationales':[{'candidate_id':'E1',
          'question':'Compare component distributions','evidence_ids':['batch:one'],
          'expected_observations':['Product and byproduct may differ'],'confounders':['Actual temperature'],
          'next_decision':'Review actual assay before narrowing'}],'hypotheses':[],'set_rationale':'Synthetic check'}
        def response(body):
            self.app.store.revise(c['id'],{'protocol_text':'New researcher information during call'})
            return {'choices':[{'message':{'content':json.dumps(plan)},'finish_reason':'stop'}]}
        self.app.gateway=ModelGateway(Path(self.temp.name)/'model_calls',transport=response)
        with self.assertRaisesRegex(CampaignError,'changed since planning'):
            self.app.generate(c['id'],{'requested_count':1,'model_route':'qwen3.6-27b'})
        self.assertEqual(self.app.store.get(c['id'])['last_round']['status'],'rejected')
        self.assertEqual(self.app.gateway.status()['attempted_calls'],1)

    def test_known_original_exports_reimport_without_new_precision_conflicts(self):
        cid=self.app.ensure_default()
        before=self.app.store.get(cid)['spec']['evidence']
        baseline={r['id']:r.get('interpretation_eligible') for r in before}
        paths=list((ROOT/'data/raw/khu').rglob('*.csv'))+list((ROOT/'data/raw/khu').rglob('*.xlsx'))
        self.assertEqual(len(paths),4)
        for path in paths:
            result=self.app.import_history({'campaign_id':cid,'name':path.name,
              'content_base64':base64.b64encode(path.read_bytes()).decode()})
            self.assertEqual(result['conflicts'],[])
        after=self.app.store.get(cid)['spec']['evidence']
        self.assertEqual(len(after),len(before))
        self.assertEqual({r['id']:r.get('interpretation_eligible') for r in after},baseline)

if __name__=='__main__':unittest.main()
