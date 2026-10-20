"""Offline runner tests. Fake gateway only; never inspect API credentials."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowpilotx.gateway import PLANNER_SYSTEM, REVIEW_SYSTEM, digest as context_digest, planning_context
from khu_sulfoxidation.flowpilotx_profile import build_profile
from khu_sulfoxidation.flowpilotx_astra import (ROLES, SPECIALISTS, ARM, execute_prepared,
    prepare_round, request_for, run_round)
from khu_sulfoxidation.model_gateway import ROUTING, campaign_template

ROOT = Path(__file__).resolve().parents[1]


def plan():
    cite = ["khu:homogeneous:raw-entry:0001"]
    return {"selected_candidate_ids": [f"KX{i:02d}" for i in range(1, 7)],
        "design_rationales": [{"candidate_id": f"KX{i:02d}", "question": "Synthetic response fixture",
            "evidence_ids": cite, "expected_observations": "Measure separate components; no prediction",
            "confounders": "Sampling and transport remain unknown", "next_decision": "Review actual assay"}
            for i in range(1, 7)],
        "hypotheses": [{"id": "synthetic-H1", "question": "Synthetic question, not a scientific result",
            "competing_explanations": ["ExplanationA", "ExplanationB"], "expected_observations": ["Compare measured components"],
            "measurements": ["Separate analytes"], "confounders": ["Transport"], "decision_rules": ["Review measurements"], "evidence_ids": cite}],
        "set_rationale": "Synthetic software fixture across three paired coordinates",
        "constraints_acknowledged": [], "unresolved": ["Laboratory details"], "rejected_candidates": []}


class FakeGateway:
    def __init__(self, root, config, fail_role=None, bad_citation=False):
        self.root = Path(root); self.root.mkdir()
        self.config = deepcopy(config)
        self.entries, self.calls = [], []
        self.fail_role, self.bad_citation = fail_role, bad_citation

    def audit(self):
        return {"entries": deepcopy(self.entries), "attempt_count": len(self.entries), "paid_api_cost_USD": 0,
                "test_fixture": True}

    def execute_request(self, request, *, role, arm):
        self.calls.append((role, deepcopy(request)))
        attempt = "attempt-" + str(len(self.entries) + 1)
        folder = self.root / "attempts" / attempt; folder.mkdir(parents=True)
        parsed = plan() if role in {"Upstream", "Chief"} else {"role": role, "verdict": "accept_conditional",
            "issues": [], "required_changes": [], "evidence_ids": ["khu:homogeneous:raw-entry:0001"], "unresolved": ["Synthetic unresolved review"]}
        if self.bad_citation and role == "Upstream":
            parsed["design_rationales"][0]["evidence_ids"] = ["khu:heterogeneous:raw-entry:0001"]
        ok = role != self.fail_role
        answer = {"ok": ok, "parsed_json": parsed, "text": json.dumps(parsed), "attempt_id": attempt,
                  "status": "completed" if ok else "failed", "model": ROUTING[role], "archive_path": str(folder)}
        (folder / "input.json").write_text(json.dumps(request))
        (folder / "result.json").write_text(json.dumps(answer))
        self.entries.append({"role": role, "arm": arm, "attempt_id": attempt, "status": answer["status"]})
        return answer


class AstraRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = build_profile(ROOT)
        request = {"requested_count": 6, "repeat_count": 1, "allow_conditional": True,
                   "researcher_instructions": "Synthetic runner test: three time coordinates, paired oxygen levels"}
        context = planning_context({"spec": spec, "results": []}, request)
        # Match the original Qwen boundary, before empty-feedback metadata was added.
        context.pop("feedback_eligibility")
        cls.frozen = {"campaign_id": "synthetic-parent", "campaign_version": 5,
            "campaign_revision_sha256": "a" * 64, "campaign_spec": spec,
            "context": context, "context_sha256": context_digest(context),
            "planner_system": PLANNER_SYSTEM, "review_system": REVIEW_SYSTEM, "request": request,
            "comparison_limits": ["Synthetic API responses; no scientific inference"]}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "frozen-source.txt"; self.source.write_text("Synthetic immutable source")
        frozen_path = self.root / "frozen_inputs.json"
        frozen_path.write_text(json.dumps(self.frozen))
        self.config = campaign_template(campaign_id="synthetic-astra-round", authorization_reference="authorization.json",
            authorized_at="Synthetic test only", source_hashes={self.source.name: hashlib.sha256(self.source.read_bytes()).hexdigest(),
                frozen_path.name: hashlib.sha256(frozen_path.read_bytes()).hexdigest()})
        self.config.update(max_attempts=7, max_attempts_per_arm=7, maximum_cost_USD=5)
        self.network = patch("khu_sulfoxidation.model_gateway._http_post", side_effect=AssertionError("No network permitted"))
        self.http = self.network.start(); self.addCleanup(self.network.stop)

    def authorize_fixture(self):
        self.config["authorized"] = True
        (self.root / "authorization.json").write_text(json.dumps({"campaign_id": self.config["campaign_id"],
            "user_instruction": "Synthetic software fixture only; never real call authorization",
            "maximum_cost_USD": 5, "maximum_attempts": 7,
            "authorized_route": {"Upstream": "gpt-6-astra", "Council": "gpt-6.1-sol"}}))

    def gateway(self, **kwargs):
        self.authorize_fixture()
        return FakeGateway(self.root / "api", self.config, **kwargs)

    def test_preflight_offline_disabled_and_exact_upstream_text(self):
        with patch("khu_sulfoxidation.flowpilotx_astra.CampaignGateway") as constructor:
            report = prepare_round(self.frozen, self.config, self.root)
            constructor.assert_not_called()
        self.assertFalse(report["authorized"])
        self.assertEqual(report["status"], "prepared_not_executed")
        self.assertEqual(report["full_seven_call_reservation_USD"], 3.52)
        self.assertEqual((report["homogeneous_records"], report["physical_candidates"]), (116, 12))
        upstream = request_for(self.frozen, "Upstream", {}, self.config["source_hashes"])
        self.assertEqual(upstream["user_prompt"], json.dumps(self.frozen["context"], ensure_ascii=False, separators=(",", ":")))
        self.assertEqual(upstream["system_prompt"], PLANNER_SYSTEM)
        self.http.assert_not_called()

    def test_execution_requires_new_explicit_campaign_authorization(self):
        with patch("khu_sulfoxidation.flowpilotx_astra.CampaignGateway") as constructor:
            with self.assertRaisesRegex(ValueError, "fresh explicit"):
                execute_prepared(self.frozen, self.config, self.root, self.root / "review")
            self.config["authorized"] = True
            with self.assertRaisesRegex(ValueError, "authorization file is missing"):
                execute_prepared(self.frozen, self.config, self.root, self.root / "review")
            self.authorize_fixture()
            record = json.loads((self.root / "authorization.json").read_text()); record["campaign_id"] = "old-exhausted-campaign"
            (self.root / "authorization.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "another campaign"):
                execute_prepared(self.frozen, self.config, self.root, self.root / "review")
            constructor.assert_not_called()
        self.http.assert_not_called()

    def test_seven_roles_own_chain_only_pending_review_no_production_mutation(self):
        gateway = self.gateway(); output = self.root / "review"
        report = run_round(self.frozen, gateway, output, project_root=self.root)
        self.assertEqual([r for r, _ in gateway.calls], list(ROLES))
        self.assertEqual(report["repeat_of"], "KX04")
        self.assertEqual(report["scientific_review_status"], "pending")
        self.assertFalse(report["production_mutation"])
        self.assertFalse(report["laboratory_released"])
        self.assertTrue((output / "nominations_freeze.json").exists())
        self.assertEqual(gateway.calls[0][1]["user_prompt"], json.dumps(self.frozen["context"], ensure_ascii=False, separators=(",", ":")))
        for role, request in gateway.calls:
            context = json.loads(request["user_prompt"])
            self.assertEqual(context["evidence"], self.frozen["context"]["evidence"])
            self.assertEqual(context["candidates"], self.frozen["context"]["candidates"])
            self.assertNotIn("engineering_corrections_from_human_audit", context)
            self.assertNotIn("model_review", context)
            if role == "Upstream":
                self.assertNotIn("upstream_proposal", context)
            if role in SPECIALISTS:
                self.assertEqual(context["upstream_proposal"], plan())
                self.assertNotIn("specialist_reviews", context)
            if role in {"Skeptic", "Chief"}:
                self.assertEqual(len(context["specialist_reviews"]), 4)
            if role == "Chief":
                self.assertEqual(context["skeptic"]["role"], "Skeptic")
        before = (output / "campaign_result.json").read_bytes()
        self.assertEqual(run_round(self.frozen, gateway, output, project_root=self.root), report)
        self.assertEqual(len(gateway.calls), 7)
        self.assertEqual((output / "campaign_result.json").read_bytes(), before)
        self.http.assert_not_called()

    def test_failed_attempt_retained_and_never_retried(self):
        gateway = self.gateway(fail_role="Chemistry"); output = self.root / "review"
        with self.assertRaisesRegex(ValueError, "API attempt failed"):
            run_round(self.frozen, gateway, output, project_root=self.root)
        self.assertTrue((output / "Chemistry_failure.json").exists())
        with self.assertRaisesRegex(ValueError, "not retried"):
            run_round(self.frozen, gateway, output, project_root=self.root)
        self.assertEqual(len(gateway.calls), 2)
        self.assertFalse((output / "nominations_freeze.json").exists())

    def test_citation_outside_visible_history_fails_despite_archived_heterogeneous_data(self):
        gateway = self.gateway(bad_citation=True); output = self.root / "review"
        with self.assertRaisesRegex(ValueError, "outside visible"):
            run_round(self.frozen, gateway, output, project_root=self.root)
        stage = json.loads((output / "Upstream.json").read_text())
        self.assertFalse(stage["validation"]["passed"])
        self.assertEqual(stage["response"]["design_rationales"][0]["evidence_ids"], ["khu:heterogeneous:raw-entry:0001"])
        self.assertEqual(len(gateway.calls), 1)

    def test_source_and_context_changes_refused_before_gateway(self):
        different = deepcopy(self.frozen); different["comparison_limits"].append("Unhashed amendment")
        with self.assertRaisesRegex(ValueError, "explicitly hashed"):
            prepare_round(different, self.config, self.root)
        self.source.write_text("Changed after freeze")
        with self.assertRaisesRegex(ValueError, "Source hash changed"):
            prepare_round(self.frozen, self.config, self.root)
        altered = deepcopy(self.frozen); altered["context"]["Qwen_answers"] = {"selected": ["KX01"]}
        altered["context_sha256"] = context_digest(altered["context"])
        with self.assertRaisesRegex(ValueError, "Context differs"):
            prepare_round(altered, self.config, self.root)
        self.http.assert_not_called()

    def test_successful_gateway_attempt_can_recover_missing_stage_without_recalling(self):
        gateway = self.gateway(); output = self.root / "review"
        report = run_round(self.frozen, gateway, output, project_root=self.root)
        (output / "Chemistry.json").unlink()  # Emulate success before runner publication.
        (output / "nominations_freeze.json").unlink()
        self.assertEqual(run_round(self.frozen, gateway, output, project_root=self.root), report)
        self.assertEqual(len(gateway.calls), 7)
        self.assertTrue((output / "Chemistry.json").exists())
        self.assertTrue((output / "nominations_freeze.json").exists())

    def test_stage_response_tampering_never_becomes_cached_success(self):
        gateway = self.gateway(); output = self.root / "review"
        run_round(self.frozen, gateway, output, project_root=self.root)
        path = output / "Upstream.json"; stage = json.loads(path.read_text())
        stage["response"]["set_rationale"] = "Altered"
        path.write_text(json.dumps(stage))
        with self.assertRaisesRegex(ValueError, "audited API response"):
            run_round(self.frozen, gateway, output, project_root=self.root)
        self.assertEqual(len(gateway.calls), 7)


if __name__ == "__main__":
    unittest.main()
