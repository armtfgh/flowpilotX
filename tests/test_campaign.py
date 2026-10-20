from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from khu_sulfoxidation.campaign import build_inputs, request_for, run_campaign, validate_proposal, STAGES


def pool_fixture():
    ids = [f"C{i}" for i in range(1, 7)]
    return {"candidates": [{"candidate_id": i} for i in ids],
            "grid_groups": [{"candidate_ids": ids}], "scenario": {"status": "synthetic"}}


def response_fixture(role):
    ids = [f"C{i}" for i in range(1, 7)]
    selects = role in ("Upstream", "Chief")
    return {"role": role, "conclusion": "Synthetic conditional screen", "recommended_core_candidate_ids": ids if selects else [],
            "comparison_nomination_ids": ids[:2] if selects else [], "repeat_candidate_id": "C1" if selects else None,
            "candidate_reviews": [{"candidate_id": i, "verdict": "retain", "rationale": "Synthetic", "citations": ["calculation:pool"]} for i in ids],
            "findings": [{"kind": "hypothesis", "statement": "Synthetic", "citations": ["study:brief"]}],
            "limitations": ["Not an experiment"]}


class CampaignTests(unittest.TestCase):
    def test_omitted_input_has_no_history_or_other_arm_context(self):
        evidence = {"common_brief": {"recipe": "shared"}, "informed_history_payload": {"secret_outcome": 98}}
        inputs = build_inputs(pool_fixture(), evidence)
        self.assertNotIn("secret_outcome", json.dumps(inputs["history_omitted"]))
        self.assertIn("secret_outcome", json.dumps(inputs["history_informed"]))
        inputs["history_informed"]["common"]["study_brief"]["recipe"] = "changed"
        self.assertEqual(inputs["history_omitted"]["common"]["study_brief"]["recipe"], "shared")

    def test_bad_grid_or_inaccessible_evidence_is_rejected(self):
        p = response_fixture("Chief")
        kwargs = dict(role="Chief", pool=pool_fixture(), citation_ids={"calculation:pool", "study:brief"})
        validate_proposal(p, **kwargs)
        p["recommended_core_candidate_ids"][-1] = "C1"
        with self.assertRaises(ValueError): validate_proposal(p, **kwargs)
        p = response_fixture("Chief")
        p["candidate_reviews"][0]["citations"] = ["hidden-batch-entry"]
        with self.assertRaises(ValueError): validate_proposal(p, **kwargs)
        p = response_fixture("Chief")
        p["comparison_nomination_ids"] = ["C1", "outside-core"]
        with self.assertRaises(ValueError): validate_proposal(p, **kwargs)
        p = response_fixture("Chief")
        p["candidate_reviews"][0]["verdict"] = "reject"
        with self.assertRaises(ValueError): validate_proposal(p, **kwargs)

    def test_specialists_only_receive_their_own_upstream(self):
        visible = build_inputs(pool_fixture(), {"common_brief": {}, "informed_history_payload": {}})["history_omitted"]
        prior = {"Upstream": response_fixture("Upstream"), "Chemistry": {"private_review": "not-yet-visible"}}
        request, citations = request_for(visible, "Safety", prior, ["hidden-history"], {})
        body = json.loads(request["user_prompt"])
        self.assertEqual(set(body["prior_within_arm_reviews"]), {"Upstream"})
        self.assertNotIn("hidden-history", citations)

    def test_all_fourteen_stages_archived_and_inputs_immutable(self):
        class MockGateway:
            calls = []
            def __init__(self, *args): pass
            def execute_request(self, request, *, role, arm):
                self.calls.append((arm, role))
                return {"parsed_json": response_fixture(role), "cost_USD": 0}
            def audit(self): return {"attempts": len(self.calls), "mock": True}
        evidence = {"common_brief": {}, "informed_history_payload": {"batch": "fixture"}, "coverage_proof": {"fixture": True}}
        with tempfile.TemporaryDirectory() as d:
            result = run_campaign(d, config={"source_hashes": {}}, pool=pool_fixture(), evidence=evidence,
                                  evidence_ids=[], gateway_class=MockGateway, progress=lambda *a, **k: None)
            self.assertEqual(len(MockGateway.calls), 14)
            self.assertEqual([r for a, r in MockGateway.calls[:7]], list(STAGES))
            self.assertFalse(result["laboratory_released"])
            self.assertTrue((Path(d)/"nominations_freeze.json").exists())
            evidence["common_brief"]["changed"] = True
            with self.assertRaises(ValueError):
                run_campaign(d, config={"source_hashes": {}}, pool=pool_fixture(), evidence=evidence,
                             evidence_ids=[], gateway_class=MockGateway)


if __name__ == "__main__": unittest.main()
