import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy

from flowpilotx.gateway import ModelGateway, planning_context, validate_plan


class GatewayTests(unittest.TestCase):
    def test_budget_counts_failed_attempt_and_never_falls_back(self):
        calls=[]
        def fail(body):
            calls.append(body)
            return {"choices":[{"message":{"content":"invalid"},"finish_reason":"stop"}]}
        with tempfile.TemporaryDirectory() as d:
            g=ModelGateway(d,max_calls=1,transport=fail)
            with self.assertRaisesRegex(ValueError,"failed"):
                g.call("planner","system",{"records":[1]})
            with self.assertRaisesRegex(ValueError,"exhausted"):
                g.call("planner","system",{})
            self.assertEqual(len(calls),1)
            self.assertEqual(g.status()["attempted_calls"],1)
            self.assertEqual(len(list(Path(d).glob("*/response.json"))),1)
            self.assertEqual(g.status()["attempts"][0]["status"],"failed")

    def test_preserves_exact_request_response_and_usage(self):
        with tempfile.TemporaryDirectory() as d:
            response={"model":"/models/Qwen3.6-27B","usage":{"total_tokens":19},
                      "choices":[{"message":{"content":'{"answer":true}'},"finish_reason":"stop"}]}
            g=ModelGateway(d,transport=lambda b:response)
            result=g.call("test","read values",{"yield":104.2,"missing":None})
            self.assertTrue(result["parsed"]["answer"])
            request=json.loads(next(Path(d).glob("*/request.json")).read_text())
            context=json.loads(request["body"]["messages"][1]["content"])
            self.assertEqual(context,{"yield":104.2,"missing":None})
            self.assertEqual(json.loads(next(Path(d).glob("*/response.json")).read_text()),response)
            self.assertEqual(g.status()["paid_api_cost_usd"],0)

    def test_unconfigured_route_never_contacts_network(self):
        with tempfile.TemporaryDirectory() as d, patch.dict("os.environ", {}, clear=True):
            g = ModelGateway(d)
            self.assertFalse(g.status()["route"]["available"])
            self.assertIsNone(g.route["base_url"])
            with patch("urllib.request.urlopen") as network:
                with self.assertRaisesRegex(ValueError, "disabled"):
                    g.call("test", "system", {})
                network.assert_not_called()
            self.assertEqual(g.calls(), [])

    def test_endpoint_requires_explicit_opt_in_and_no_embedded_credentials(self):
        with tempfile.TemporaryDirectory() as d, patch.dict("os.environ", {
                "FLOWPILOTX_QWEN_BASE_URL": "http://localhost:8000/v1"}, clear=True):
            self.assertFalse(ModelGateway(d).route["available"])
            with patch.dict("os.environ", {"FLOWPILOTX_ENABLE_QWEN": "1"}):
                self.assertTrue(ModelGateway(d).route["available"])
            for endpoint in ["https://user:pass@example.invalid/v1", "file:///v1",
                             "https://example.invalid/v1?token=example"]:
                with patch.dict("os.environ", {"FLOWPILOTX_QWEN_BASE_URL": endpoint}):
                    with self.assertRaisesRegex(ValueError, "without credentials"):
                        ModelGateway(d)

    def test_paid_route_is_blocked_before_transport(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,"Paid routing"):
                ModelGateway(d,route={"paid":True},transport=lambda b:None)

    def test_source_and_candidate_validation(self):
        campaign={"spec":{"name":"thermal example","candidates":[{"id":"A","feasible":True}],
                          "evidence":[{"id":"record-1"}],"protocol_text":"Heat the liquid","objective":{"mode":"explore"}}}
        good={"selected_candidate_ids":["A"],"set_rationale":"Test useful contrast",
              "design_rationales":[{"candidate_id":"A","question":"Does temperature affect selectivity?",
                                    "evidence_ids":["record-1"],"expected_observations":["changed product ratio"],
                                    "confounders":["residence distribution"],"next_decision":"retain or revise temperature range"}]}
        self.assertTrue(validate_plan(good,campaign,1)["passed"])
        good["design_rationales"][0]["evidence_ids"]=["invented-record"]
        self.assertFalse(validate_plan(good,campaign,1)["passed"])
        good["selected_candidate_ids"]=["invented-design"]
        self.assertFalse(validate_plan(good,campaign,1)["passed"])
        context=planning_context(campaign,{"requested_count":1})
        self.assertEqual(context["requested_count"],1)
        self.assertEqual(context["protocol_text"],"Heat the liquid")
        self.assertNotIn("sulfoxide",json.dumps(context))

    def test_boolean_rationales_duplicate_cards_and_bad_hypothesis_sources_fail(self):
        campaign={'spec':{'candidates':[{'id':'A','feasible':True}], 'evidence':[{'id':'real'}]}}
        good={'selected_candidate_ids':['A'],'set_rationale':'Explore', 'design_rationales':[
            {'candidate_id':'A','question':'What changes?','expected_observations':['Separate components'],
             'confounders':['Temperature'],'next_decision':'Compare assay','evidence_ids':['real']}]}
        bad=deepcopy(good);bad['design_rationales'][0]['question']=True
        self.assertFalse(validate_plan(bad,campaign,1)['passed'])
        bad=deepcopy(good);bad['design_rationales']*=2
        self.assertFalse(validate_plan(bad,campaign,1)['passed'])
        bad=deepcopy(good);bad['hypotheses']=[{'id':'H','question':'What changes?',
            **{k:['Observation'] for k in ['competing_explanations','expected_observations','measurements','confounders','decision_rules']},
            'evidence_ids':['fabricated']}]
        self.assertFalse(validate_plan(bad,campaign,1)['passed'])


if __name__ == '__main__': unittest.main()
