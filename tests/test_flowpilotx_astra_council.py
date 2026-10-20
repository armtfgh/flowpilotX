from copy import deepcopy
import json
from pathlib import Path
import unittest
from khu_sulfoxidation.flowpilotx_astra_council import validate_response
ROOT=Path(__file__).resolve().parents[1]

class CouncilFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f=json.loads((ROOT/'outputs/flowpilotx_astra_preflight_2026-10-15_v1/frozen_inputs.json').read_text())
        cls.response=json.loads((ROOT/'outputs/flowpilotx_astra_sol_2026-10-15_v1/continuation_v2/Chemistry.json').read_text())['response']
    def test_structured_arrays_allowed_by_actual_prompt(self):
        original=deepcopy(self.response)
        self.assertTrue(validate_response('Chemistry',self.response,self.f)['passed'])
        self.assertEqual(self.response,original)
    def test_missing_evidence_and_empty_structured_findings_still_fail(self):
        for key,value in [('evidence_ids',[{'source':'invented'}]),('evidence_ids',['fabricated']),
                ('issues',[{}]),('required_changes',{'text':'wrong container'}),('unresolved',[False])]:
            changed=deepcopy(self.response);changed[key]=value
            self.assertFalse(validate_response('Chemistry',changed,self.f)['passed'],(key,value))
    def test_plain_array_contract_and_wrong_role(self):
        changed=deepcopy(self.response)
        changed.update(issues=['A concern'],required_changes=['A change'],unresolved=[])
        self.assertTrue(validate_response('Chemistry',changed,self.f)['passed'])
        self.assertFalse(validate_response('Safety',changed,self.f)['passed'])
if __name__=='__main__':unittest.main()
